"""
ai/vlm/event_engine.py
──────────────────────
Event-Driven VLM Engine — replaces frame-driven periodic VLM polling.

Architecture:
  ActivityAnalyzer (per-frame, local rules)
       │
       ▼
   EventEngine ─── detects state changes ───→ VLM (only when needed)
       │                                            │
       │  • person state memory                     │
       │  • cooldown logic (60s default)            │
       │  • event trigger detection                 │
       │  • VLM cache (60s TTL)                     │
       │  • request throttling (1 req/s global)     │
       │  • degraded mode (429 → 5min backoff)      │
       │  • metrics                                 │
       ▼
  AI Insight (meaningful warehouse events only)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ai.analyzer.activity_analyzer import ActivityResult, ActivityLabel
from ai.vlm.vlm_client import VLMClient, VLMResult
from ai.tracker.person_tracker import TrackedPerson
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Events that trigger VLM ─────────────────────────────────────────────────

VLM_TRIGGER_EVENTS: set[str] = {
    "zone_entered",
    "restricted_area_entry",
    "unauthorized_entry",
    "fall_detected",
    "loitering",
    "running_detected",
    "object_pickup",
    "object_drop",
    "carrying_started",
    "worker_vehicle_interaction",
    "anomaly_state_changed",
}

NON_TRIGGER_ACTIVITIES: set[str] = {
    ActivityLabel.WALKING,
    ActivityLabel.STANDING,
    ActivityLabel.UNKNOWN,
}


# ── Person State ────────────────────────────────────────────────────────────

@dataclass
class PersonState:
    person_id:        str
    camera_id:        str
    current_zone:     str            = ""
    previous_zone:    str            = ""
    current_activity: str            = ""
    previous_activity: str           = ""
    anomaly_label:    str            = "normal"
    previous_anomaly: str            = "normal"
    last_vlm_time:    float          = 0.0
    last_vlm_description: str        = ""
    last_vlm_event:   str            = ""
    track_uuid:       str            = ""
    last_vlm_frame_number: int       = 0
    last_vlm_activity_type: str      = "unknown"


# ── VLM Cache Entry ─────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    result:      VLMResult
    cached_at:   float
    description: str = ""


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class VLMMetrics:
    gather_started:    int = 0
    requests_started:  int = 0
    requests_completed: int = 0
    requests_failed:   int = 0
    requests_429:      int = 0
    cache_hits:        int = 0
    cache_misses:      int = 0
    queue_depth:       int = 0
    cooldown_skips:    int = 0
    event_triggers:    int = 0
    first_detection:   int = 0


# ── Event Engine ────────────────────────────────────────────────────────────

class EventEngine:
    """
    Event-driven VLM trigger engine.

    Receives per-frame ActivityResults from the pipeline and decides
    whether to call the VLM based on state changes, cooldowns, and events.

    Usage:
        engine = EventEngine(vlm_client)
        should_call, reason = engine.evaluate(person, activity)
        if should_call:
            result = await vlm_client.analyze_person(...)
            engine.record_vlm_call(person, activity, result, reason)
    """

    def __init__(self, vlm_client: VLMClient) -> None:
        self._vlm = vlm_client

        # Person state memory: track_uuid → PersonState
        self._persons: dict[str, PersonState] = {}

        # VLM cache: (track_uuid, zone, activity) → _CacheEntry
        self._cache: dict[tuple[str, str, str], _CacheEntry] = {}

        # Request throttle: global rate limiter
        self._last_request_time: float = 0.0
        self._request_queue: list[asyncio.Event] = []
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Degraded mode
        self._degraded_until: float = 0.0
        self._consecutive_429s: int = 0

        # Metrics
        self.metrics = VLMMetrics()

        # Start queue processor
        self._start_queue_processor()

    # ── Person state management ─────────────────────────────────────────────

    def get_or_create_state(self, person: TrackedPerson) -> PersonState:
        uid = person.track_uuid
        if uid not in self._persons:
            self._persons[uid] = PersonState(
                person_id=    person.person_id,
                camera_id=    person.camera_id or "",
                current_zone= person.zone_id or "",
                track_uuid=   uid,
            )
        return self._persons[uid]

    def evaluate(
        self,
        person:   TrackedPerson,
        activity: ActivityResult,
    ) -> tuple[bool, str]:
        """
        Evaluate whether a VLM call should be made for this person.

        Returns:
            (should_call: bool, reason: str)
        """
        state = self.get_or_create_state(person)

        # ── Save previous state ──────────────────────────────────────────
        state.previous_zone     = state.current_zone
        state.previous_activity = state.current_activity
        state.previous_anomaly  = state.anomaly_label

        # ── Update current state ─────────────────────────────────────────
        state.current_zone     = activity.zone_id or person.zone_id or ""
        state.current_activity = activity.activity_type
        state.anomaly_label    = activity.anomaly_label

        # ── Detect events ────────────────────────────────────────────────

        # Zone change
        zone_changed = (state.current_zone != state.previous_zone
                        and state.previous_zone != "")

        # Restricted area entry
        restricted_entry = (zone_changed
                            and person.is_restricted
                            and state.current_activity == ActivityLabel.UNAUTHORIZED_ENTRY)

        # Activity change
        activity_changed = (state.current_activity != state.previous_activity
                            and state.previous_activity != "")

        # Anomaly status change
        anomaly_changed = (state.anomaly_label != state.previous_anomaly
                           and state.previous_anomaly != "")

        # Specific event detection
        event = self._classify_event(activity, person, state, zone_changed)

        # ── Determine if VLM should be triggered ─────────────────────────

        # Degraded mode check
        if self._degraded_until > time.time():
            logger.info(
                f"[VLM-DEGRADED] Skipping VLM for {person.person_id} "
                f"(degraded until t={self._degraded_until:.1f})"
            )
            return False, "degraded_mode"

        # Check cache first
        cache_key = self._cache_key(person, activity)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.cached_at < settings.EVENT_VLM_CACHE_TTL:
                self.metrics.cache_hits += 1
                return False, "cache_hit"

        self.metrics.cache_misses += 1

        # First detection trigger — every new track gets exactly one immediate VLM analysis
        if state.last_vlm_time == 0.0:
            self.metrics.first_detection += 1
            logger.info(
                f"[VLM-FIRST-DETECTION] person_id={person.person_id} "
                f"trigger=first_detection"
            )
            return True, "first_detection"

        # Event-based trigger
        if event in VLM_TRIGGER_EVENTS:
            self.metrics.event_triggers += 1
            return True, f"event:{event}"

        # Zone changed
        if zone_changed:
            return True, "zone_changed"

        # Restricted entry
        if restricted_entry:
            return True, "restricted_entry"

        # Activity changed to a trigger-worthy activity
        if activity_changed and state.current_activity in VLM_TRIGGER_EVENTS:
            return True, f"activity_changed:{state.current_activity}"

        # Anomaly status changed
        if anomaly_changed:
            return True, "anomaly_changed"

        # Cooldown check — only allow if state changed OR cooldown expired
        now = time.time()
        time_since_last_vlm = now - state.last_vlm_time

        if time_since_last_vlm < settings.EVENT_VLM_COOLDOWN:
            self.metrics.cooldown_skips += 1
            return False, "cooldown"

        # Still trigger on periodic refresh if cooldown expired AND has meaningful activity
        if (state.current_activity not in NON_TRIGGER_ACTIVITIES
                and time_since_last_vlm >= settings.EVENT_VLM_COOLDOWN):
            return True, "cooldown_expired_refresh"

        return False, "no_event"

    def _classify_event(
        self,
        activity: ActivityResult,
        person:   TrackedPerson,
        state:    PersonState,
        zone_changed: bool,
    ) -> str:
        """Classify the event type from activity + state changes."""
        act = activity.activity_type

        if act == ActivityLabel.FALLING:
            return "fall_detected"
        if act == ActivityLabel.LOITERING:
            return "loitering"
        if act == ActivityLabel.RUNNING:
            return "running_detected"
        if act == ActivityLabel.UNAUTHORIZED_ENTRY:
            return "unauthorized_entry"
        if act == ActivityLabel.CARRYING_OBJECT:
            if (state.previous_activity != ActivityLabel.CARRYING_OBJECT
                    and state.previous_activity != ""):
                return "carrying_started"
        if (zone_changed and person.is_restricted
                and act == ActivityLabel.UNAUTHORIZED_ENTRY):
            return "restricted_area_entry"
        if zone_changed:
            return "zone_entered"
        if (state.anomaly_label != state.previous_anomaly
                and state.previous_anomaly != ""):
            return "anomaly_state_changed"

        return ""

    # ── VLM call recording ─────────────────────────────────────────────────

    def record_vlm_call(
        self,
        person:   TrackedPerson,
        activity: ActivityResult,
        result:   Optional[VLMResult],
        reason:   str,
    ) -> None:
        """Record a completed VLM call into person state and cache."""
        uid = person.track_uuid
        state = self._persons.get(uid)
        if state is None:
            # fallback
            state = self._persons.get(person.person_id)
        if state is None:
            return

        # Check for stale result to prevent race conditions
        if result is not None and result.frame_number < state.last_vlm_frame_number:
            logger.warning(
                f"[VLM-STALE-DISCARD] Discarding stale VLM response for {person.person_id} "
                f"(result frame {result.frame_number} < latest frame {state.last_vlm_frame_number})"
            )
            return

        now = time.time()
        state.last_vlm_time = now
        state.last_vlm_event = reason

        if result is not None:
            state.last_vlm_description = result.description
            state.last_vlm_activity_type = result.activity_type
            state.last_vlm_frame_number = result.frame_number
            state.anomaly_label = result.anomaly_label
            state.current_activity = result.activity_type
            # Cache the result
            cache_key = self._cache_key(person, activity, state)
            self._cache[cache_key] = _CacheEntry(
                result=result,
                cached_at=now,
                description=result.description,
            )

    def _cache_key(
        self,
        person:   TrackedPerson,
        activity: Optional[ActivityResult] = None,
        state:    Optional[PersonState] = None,
    ) -> tuple:
        """Build a cache key: (track_uuid, zone, activity)."""
        if state is None:
            state = self._persons.get(person.track_uuid) or self._persons.get(person.person_id)
        zone = state.current_zone if state else (activity.zone_id if activity else person.zone_id)
        act = activity.activity_type if activity else (state.current_activity if state else "unknown")
        return (person.track_uuid, zone, act)

    # ── Throttling ─────────────────────────────────────────────────────────

    def _start_queue_processor(self) -> None:
        """Start background task that processes the request queue at 1 req/s."""
        if self._queue_processor_task is None:
            self._queue_processor_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Process queued VLM requests at a rate of 1 per second."""
        while True:
            if self._request_queue:
                event = self._request_queue.pop(0)
                event.set()
                self.metrics.queue_depth = len(self._request_queue)
            await asyncio.sleep(1.0)

    async def acquire_throttle(self) -> None:
        """
        Acquire a throttle slot.
        Waits if the global rate limit would be exceeded.
        Rate is controlled by EVENT_VLM_RATE_LIMIT (requests per second).
        Default: 1 req/s → 1.0s interval.
        """
        self.metrics.requests_started += 1
        now = time.time()
        interval = 1.0 / max(settings.EVENT_VLM_RATE_LIMIT, 0.1)
        wait = self._last_request_time + interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.time()

    # ── Degraded mode ──────────────────────────────────────────────────────

    def handle_429(self) -> None:
        """Enter degraded mode after receiving a 429."""
        self._consecutive_429s += 1
        backoff = min(300, 60 * self._consecutive_429s)  # 1min, 2min, ... up to 5min
        self._degraded_until = time.time() + backoff
        self.metrics.requests_429 += 1
        logger.warning(
            f"[VLM-DEGRADED] HTTP 429 received. "
            f"Disabling VLM for {backoff}s (until t={self._degraded_until:.1f}). "
            f"Consecutive 429s: {self._consecutive_429s}"
        )

    def handle_success(self) -> None:
        """Reset consecutive 429 counter on success."""
        self._consecutive_429s = 0
        self._degraded_until = 0.0

    def is_degraded(self) -> bool:
        """Check if the system is in degraded mode."""
        if self._degraded_until > time.time():
            return True
        return False

    # ── Cleanup ────────────────────────────────────────────────────────────

    def evict_stale_persons(self, active_track_uuids: set[str]) -> None:
        """Remove state for persons no longer being tracked."""
        stale = set(self._persons.keys()) - active_track_uuids
        for uid in stale:
            del self._persons[uid]

    def evict_stale_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        stale_keys = [
            k for k, v in self._cache.items()
            if now - v.cached_at > settings.EVENT_VLM_CACHE_TTL
        ]
        for k in stale_keys:
            del self._cache[k]

    # ── Metrics ────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return current VLM metrics as a dict."""
        return {
            "requests_started":   self.metrics.requests_started,
            "requests_completed": self.metrics.requests_completed,
            "requests_failed":    self.metrics.requests_failed,
            "requests_429":       self.metrics.requests_429,
            "cache_hits":         self.metrics.cache_hits,
            "cache_misses":       self.metrics.cache_misses,
            "queue_depth":        self.metrics.queue_depth,
            "cooldown_skips":     self.metrics.cooldown_skips,
            "event_triggers":     self.metrics.event_triggers,
            "first_detection":    self.metrics.first_detection,
            "is_degraded":        self.is_degraded(),
            "degraded_until":     self._degraded_until,
            "active_persons":     len(self._persons),
            "cache_entries":      len(self._cache),
            "consecutive_429s":   self._consecutive_429s,
        }

    def record_completed(self) -> None:
        self.metrics.requests_completed += 1

    def record_failed(self) -> None:
        self.metrics.requests_failed += 1

    # ── VLM data lookup (for downstream enrichment) ─────────────────────────

    def get_vlm_description(self, person_id: str, track_uuid: Optional[str] = None) -> str:
        """Return the last VLM description for a person, or empty string."""
        key = track_uuid
        if not key:
            for u, s in self._persons.items():
                if s.person_id == person_id:
                    key = u
                    break
        if not key:
            key = person_id
        state = self._persons.get(key)
        if state is None:
            return ""
        return state.last_vlm_description

    def get_vlm_data(self, person_id: str, track_uuid: Optional[str] = None) -> dict:
        """Return VLM data dict for a person (for frame update payloads)."""
        key = track_uuid
        if not key:
            for u, s in self._persons.items():
                if s.person_id == person_id:
                    key = u
                    break
        if not key:
            key = person_id
        state = self._persons.get(key)
        if state is None or not state.last_vlm_description:
            return {}
        return {
            "description":   state.last_vlm_description,
            "activity_type": getattr(state, "last_vlm_activity_type", "unknown"),
            "anomaly_label": state.anomaly_label,
            "event":         state.last_vlm_event,
        }
