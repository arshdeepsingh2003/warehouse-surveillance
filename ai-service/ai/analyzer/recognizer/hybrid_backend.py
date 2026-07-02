from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.analyzer.activity_analyzer import ActivityLabel, ActivityResult
from ai.analyzer.recognizer.base import BaseActivityBackend
from ai.tracker.person_tracker import TrackedPerson
from ai.vlm.vlm_client import VLMResult
from config.settings import settings

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.65
"""Activities below this confidence trigger a VLM fallback query."""

_UNCERTAIN_TYPES = {
    ActivityLabel.UNKNOWN,
    ActivityLabel.HANDLING_ITEMS,
    ActivityLabel.CARRYING_OBJECT,
}
"""Activity types that are inherently uncertain from rules alone."""


class HybridBackend(BaseActivityBackend):
    """
    Hybrid activity recognition: rules first, Groq VLM fallback.

    1. Run rule-based analysis (via RulesBackend wrapper of ActivityAnalyzer).
    2. For results with:
       - confidence < CONFIDENCE_THRESHOLD, OR
       - activity_type in uncertain set (unknown, handling_items, etc.)
       Query Groq VLM with the cropped person image.
    3. If VLM returns a confident result, merge it over the rule-based one.
    4. Cache VLM results per track ID (TTL-controlled).

    This gives the speed of rules with the accuracy of VLM where it matters.
    """

    def __init__(self, camera_id: str) -> None:
        from ai.analyzer.recognizer.rules_backend import RulesBackend
        from ai.vlm.vlm_client import VLMClient, GroqVLMBackend

        self._camera_id = camera_id
        self._rules = RulesBackend(camera_id)
        self._confidence_threshold = _CONFIDENCE_THRESHOLD
        self._uncertain_types = _UNCERTAIN_TYPES

        # VLM backed by Groq
        self._vlm = VLMClient()
        groq_model = getattr(
            settings, "GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self._vlm._backend = GroqVLMBackend(model=groq_model)

        # VLM cache: track_id → (timestamp, VLMResult)
        self._cache: dict[int, tuple[float, VLMResult]] = {}
        self._cache_ttl = settings.VLM_CACHE_TTL_SECONDS

        logger.info(
            f"[{camera_id}] HybridBackend ready | "
            f"confidence_threshold={self._confidence_threshold} | "
            f"cache_ttl={self._cache_ttl}s"
        )

    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        # Step 1: Run rules (with carryable objects for spatial analysis)
        results = self._rules._analyzer.analyze(persons, carryable_objects)

        # Step 2: VLM fallback for low-confidence / uncertain activities
        for i, (person, result) in enumerate(zip(persons, results)):
            if self._needs_vlm(result):
                vlm = await self._query_vlm(frame, person, camera_id)
                if vlm and vlm.confidence > result.confidence:
                    results[i] = self._merge_vlm(result, vlm, person)

        return results

    def _needs_vlm(self, result: ActivityResult) -> bool:
        """True if this activity would benefit from VLM analysis."""
        return (
            result.confidence < self._confidence_threshold
            or result.activity_type in self._uncertain_types
        )

    async def _query_vlm(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        camera_id: str,
    ) -> Optional[VLMResult]:
        """Check cache or query VLM. Returns None on error."""
        tid = person.track_id
        now = time.monotonic()

        if tid in self._cache:
            cached_at, cached = self._cache[tid]
            if now - cached_at < self._cache_ttl:
                logger.debug(f"HybridBackend VLM cache hit: track {tid}")
                return cached

        try:
            vlm = await self._vlm.analyze_person(
                frame=frame,
                bbox=person.bbox,
                person_id=person.person_id,
                camera_id=camera_id,
                zone_id=person.zone_id,
                zone_name=person.zone_name,
                is_restricted=person.is_restricted,
                extra_context=(
                    f"Rule-based detection: {person.zone_name}. "
                    f"Dwell time: {person.dwell_time:.0f}s."
                ),
            )
            # HARD RULE: never cache or use fallback results
            if vlm.backend_used == "fallback":
                logger.warning(
                    f"[FALLBACK-TRACE] person_id={person.person_id} camera_id={camera_id} "
                    f"reason=hybrid_backend_received_fallback "
                    f"desc=\"{vlm.description[:60]}\""
                )
                return None
            self._cache[tid] = (now, vlm)
            return vlm
        except Exception as e:
            logger.debug(f"HybridBackend VLM error for track {tid}: {e}")
            return None

    def _merge_vlm(
        self,
        rule_result: ActivityResult,
        vlm: VLMResult,
        person: TrackedPerson,
    ) -> ActivityResult:
        """Merge VLM result into rule-based result, preferring VLM data."""
        from ai.analyzer.activity_analyzer import AnomalyFlag

        flags = rule_result.flags.copy()
        if vlm.activity_type == "theft_attempt":
            flags.append(AnomalyFlag.THEFT_DETECTED)
        elif vlm.activity_type == "safety_violation":
            flags.append(AnomalyFlag.MISCONDUCT_DETECTED)
        elif vlm.activity_type == ActivityLabel.UNAUTHORIZED_ENTRY:
            flags.append(AnomalyFlag.RESTRICTED_ZONE)
        elif vlm.activity_type == ActivityLabel.FALLING:
            flags.append(AnomalyFlag.POSSIBLE_FALL)
        elif vlm.activity_type == ActivityLabel.LOITERING:
            flags.append(AnomalyFlag.LOITERING)
        elif vlm.activity_type == ActivityLabel.RUNNING:
            flags.append(AnomalyFlag.FAST_MOVEMENT)

        return ActivityResult(
            person_id=person.person_id,
            track_id=person.track_id,
            activity_type=vlm.activity_type,
            anomaly_label=(
                "anomaly"
                if vlm.is_anomaly or rule_result.is_anomaly or vlm.activity_type in ("theft_attempt", "safety_violation")
                else "normal"
            ),
            description=vlm.description or rule_result.description,
            confidence=max(vlm.confidence, rule_result.confidence),
            flags=list(set(flags)),
            zone_id=person.zone_id,
            zone_name=person.zone_name,
            dwell_time=person.dwell_time,
            backend_used=vlm.backend_used if vlm.backend_used else "groq",
        )
