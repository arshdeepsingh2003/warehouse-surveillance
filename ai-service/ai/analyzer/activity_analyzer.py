"""
ai/analyzer/activity_analyzer.py
──────────────────────────────────
Activity Analyzer — classifies what each tracked person is doing.

This module bridges raw tracker output (bounding box, dwell time, velocity)
and meaningful activity labels that the rules engine can act on.

Current mode: RULE-BASED (fast, no GPU, works immediately)
  Uses physics/geometry of the bounding box + motion to classify activities.

Future mode: VLM-BASED (much richer, Step 5 in roadmap)
  Crops the person region → sends to GPT-4V / LLaVA → returns description.
  The ActivityAnalyzer interface stays identical — only the backend changes.

Classification logic
────────────────────
  WALKING          → moderate velocity, upright aspect ratio
  RUNNING          → high velocity
  STANDING         → near-zero velocity for < LOITERING_SECONDS
  LOITERING        → near-zero velocity for > LOITERING_SECONDS
  FALLING          → bounding box suddenly wider than tall (person horizontal)
                     OR extreme downward velocity spike
  CARRYING_OBJECT  → bounding box wider than baseline person width
  CROUCHING        → aspect ratio below normal, low velocity
  UNAUTHORIZED_ENTRY → person in restricted zone (zone_config flag)

Output: ActivityResult
  {
    activity_type:  str     (walking / standing / loitering / ...)
    anomaly_label:  str     (normal / anomaly)
    description:    str     (human-readable sentence for the activity log)
    confidence:     float
    flags:          list[str]  (specific anomaly flags for rules engine)
  }
"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ai.tracker.person_tracker import TrackedPerson
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Activity labels (must match backend schema) ───────────────────────────────
class ActivityLabel:
    WALKING           = "walking"
    RUNNING           = "running"
    STANDING          = "standing"
    LOITERING         = "loitering"
    FALLING           = "falling"
    CARRYING_OBJECT   = "carrying_object"
    CROUCHING         = "crouching"
    HANDLING_ITEMS    = "handling_items"
    UNAUTHORIZED_ENTRY= "unauthorized_entry"
    UNKNOWN           = "unknown"


# ── Anomaly flags (used by the rules engine) ──────────────────────────────────
class AnomalyFlag:
    LOITERING          = "loitering"
    RESTRICTED_ZONE    = "restricted_zone"
    POSSIBLE_FALL      = "possible_fall"
    FAST_MOVEMENT      = "fast_movement"
    LONG_DWELL         = "long_dwell"
    PPE_ZONE_VIOLATION = "ppe_zone_violation"  # placeholder until PPE detector added


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ActivityResult:
    """One activity classification result for one person in one frame."""
    person_id:     str
    track_id:      int
    activity_type: str
    anomaly_label: str    = "normal"    # "normal" | "anomaly"
    description:   str    = ""
    confidence:    float  = 0.8
    flags:         list[str] = field(default_factory=list)
    zone_id:       str    = ""
    zone_name:     str    = ""
    dwell_time:    float  = 0.0

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly_label == "anomaly"


# ── Per-person history (needed for temporal analysis) ─────────────────────────

@dataclass
class _PersonHistory:
    """Stores recent observations for one person ID for temporal analysis."""
    track_id:          int
    # Recent aspect ratios (w/h) — used to detect fall (ratio spikes)
    aspect_ratios:     list[float] = field(default_factory=list)
    # Recent velocities — used to detect running / sudden stops
    velocities:        list[float] = field(default_factory=list)
    # Baseline aspect ratio (first few observations)
    baseline_ar:       Optional[float] = None
    # Time of last zone entry per zone_id
    zone_first_seen:   dict[str, float] = field(default_factory=dict)

    HISTORY_LEN = 20

    def add(self, ar: float, speed: float) -> None:
        self.aspect_ratios.append(ar)
        self.velocities.append(speed)
        if len(self.aspect_ratios) > self.HISTORY_LEN:
            self.aspect_ratios.pop(0)
            self.velocities.pop(0)
        if self.baseline_ar is None and len(self.aspect_ratios) >= 5:
            # Establish baseline aspect ratio from first observations
            self.baseline_ar = sum(self.aspect_ratios[:5]) / 5

    @property
    def avg_speed(self) -> float:
        return sum(self.velocities) / len(self.velocities) if self.velocities else 0.0

    @property
    def recent_speed(self) -> float:
        return self.velocities[-1] if self.velocities else 0.0

    @property
    def current_ar(self) -> float:
        return self.aspect_ratios[-1] if self.aspect_ratios else 1.0

    def ar_spike(self) -> bool:
        """True if aspect ratio suddenly got much wider (person falling)."""
        if self.baseline_ar is None or len(self.aspect_ratios) < 5:
            return False
        return self.current_ar > self.baseline_ar * 2.0


# ── Analyzer ──────────────────────────────────────────────────────────────────

class ActivityAnalyzer:
    """
    Classifies the activity of each tracked person.

    One shared instance per camera — persists _PersonHistory across frames.

    Usage:
        analyzer = ActivityAnalyzer(camera_id="cam-01")
        results  = analyzer.analyze(tracked_persons)
    """

    # Speed thresholds (pixels/frame — depends on resolution & FPS)
    SPEED_WALKING_MAX  = 8.0    # px/frame
    SPEED_RUNNING_MIN  = 8.0    # px/frame
    SPEED_STANDING_MAX = 1.5    # px/frame

    # Aspect ratio (width/height) thresholds
    AR_FALL_THRESHOLD  = 1.8    # w/h > 1.8 → person is horizontal

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._history: dict[int, _PersonHistory] = {}

    def analyze(self, persons: list[TrackedPerson]) -> list[ActivityResult]:
        """
        Classify activity for all tracked persons in one frame.

        Args:
            persons: Output of PersonTracker.update()

        Returns:
            List of ActivityResult, one per person.
        """
        results = []
        active_ids = {p.track_id for p in persons}

        # Prune history for persons no longer tracked
        self._history = {k: v for k, v in self._history.items() if k in active_ids}

        for person in persons:
            result = self._classify(person)
            results.append(result)

        return results

    def _classify(self, person: TrackedPerson) -> ActivityResult:
        """Classify one person's activity."""
        tid = person.track_id

        # Get or create history for this track
        if tid not in self._history:
            self._history[tid] = _PersonHistory(track_id=tid)

        hist = self._history[tid]

        # Compute motion features
        x1, y1, x2, y2 = person.bbox
        w = x2 - x1
        h = max(y2 - y1, 1)
        ar = w / h    # aspect ratio

        vx, vy = person.velocity
        speed  = math.sqrt(vx**2 + vy**2)

        hist.add(ar, speed)

        # ── Rule 1: Restricted zone entry ─────────────────────────────────────
        if person.is_restricted:
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                activity_type=ActivityLabel.UNAUTHORIZED_ENTRY,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} entered restricted zone "
                    f"'{person.zone_name}' without authorisation."
                ),
                confidence=   0.92,
                flags=        [AnomalyFlag.RESTRICTED_ZONE],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 2: Possible fall ─────────────────────────────────────────────
        is_fall = ar > self.AR_FALL_THRESHOLD or hist.ar_spike()
        if is_fall and person.age > 10:
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                activity_type=ActivityLabel.FALLING,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} appears to have fallen in "
                    f"'{person.zone_name}' — stationary horizontal posture detected."
                ),
                confidence=   0.85,
                flags=        [AnomalyFlag.POSSIBLE_FALL],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 3: Running ───────────────────────────────────────────────────
        if speed > self.SPEED_RUNNING_MIN:
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                activity_type=ActivityLabel.RUNNING,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} is running in '{person.zone_name}'. "
                    f"Speed: {speed:.1f} px/frame."
                ),
                confidence=   0.78,
                flags=        [AnomalyFlag.FAST_MOVEMENT],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 4: Loitering ─────────────────────────────────────────────────
        if (speed < self.SPEED_STANDING_MAX
                and person.dwell_time > settings.LOITERING_SECONDS):
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                activity_type=ActivityLabel.LOITERING,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} has been stationary in "
                    f"'{person.zone_name}' for {int(person.dwell_time)}s — loitering detected."
                ),
                confidence=   0.80,
                flags=        [AnomalyFlag.LOITERING, AnomalyFlag.LONG_DWELL],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 5: Normal activities (no anomaly) ────────────────────────────
        if speed < self.SPEED_STANDING_MAX:
            activity = ActivityLabel.STANDING
            desc = (f"{person.person_id} is standing in '{person.zone_name}'.")
        elif speed < self.SPEED_WALKING_MAX:
            activity = ActivityLabel.WALKING
            desc = (f"{person.person_id} is walking through '{person.zone_name}'.")
        else:
            activity = ActivityLabel.HANDLING_ITEMS
            desc = (f"{person.person_id} is active in '{person.zone_name}'.")

        return ActivityResult(
            person_id=    person.person_id,
            track_id=     tid,
            activity_type=activity,
            anomaly_label="normal",
            description=  desc,
            confidence=   0.85,
            flags=        [],
            zone_id=      person.zone_id,
            zone_name=    person.zone_name,
            dwell_time=   person.dwell_time,
        )

    # ── Future VLM integration ────────────────────────────────────────────────
    # When VLM is available, add this method:
    #
    # async def analyze_with_vlm(
    #     self,
    #     person: TrackedPerson,
    #     frame:  np.ndarray,
    # ) -> ActivityResult:
    #     """Crop person region and query VLM for rich description."""
    #     x1, y1, x2, y2 = person.bbox
    #     crop = frame[y1:y2, x1:x2]
    #
    #     # Query VLM (GPT-4V, LLaVA, or local Ollama model)
    #     description = await self._vlm_client.query(
    #         image=crop,
    #         prompt=(
    #             "Describe what this warehouse worker is doing in one sentence. "
    #             "Focus on: what they are holding, their posture, and any safety concerns."
    #         )
    #     )
    #
    #     # Parse VLM output into structured result
    #     anomaly_label = self._vlm_client.classify(description)
    #     return ActivityResult(
    #         person_id=    person.person_id,
    #         activity_type=ActivityLabel.UNKNOWN,
    #         anomaly_label=anomaly_label,
    #         description=  description,
    #         confidence=   0.90,
    #         zone_id=      person.zone_id,
    #     )
