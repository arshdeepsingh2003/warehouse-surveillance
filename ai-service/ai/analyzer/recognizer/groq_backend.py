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


class GroqBackend(BaseActivityBackend):
    """
    Groq VLM-only activity recognition backend.

    Every tracked person is sent to Groq's Llama 4 Scout (or configured
    Groq VLM model) for direct visual analysis. No rule-based inference.

    Caches VLM results per track ID to avoid redundant API calls.
    Cache TTL is controlled by VLM_CACHE_TTL_SECONDS (default 300s).
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._vlm = self._build_groq_vlm()
        # VLM cache: track_id → (timestamp, VLMResult)
        self._cache: dict[int, tuple[float, VLMResult]] = {}
        self._cache_ttl = settings.VLM_CACHE_TTL_SECONDS
        logger.info(
            f"[{camera_id}] GroqBackend ready | model={settings.GROQ_VLM_MODEL}"
        )

    def _build_groq_vlm(self):
        """Instantiate a VLMClient forced to use the Groq backend."""
        from ai.vlm.vlm_client import VLMClient, GroqVLMBackend

        client = VLMClient()
        groq_model = getattr(
            settings, "GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        client._backend = GroqVLMBackend(model=groq_model)
        return client

    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        results: list[ActivityResult] = []
        for person in persons:
            vlm = await self._query_vlm(frame, person, camera_id)
            if vlm is None:
                # HARD RULE: fallback — use a minimal safe result
                results.append(self._empty_result(person))
            else:
                results.append(self._vlm_to_activity(vlm, person))
        return results

    async def _query_vlm(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        camera_id: str,
    ) -> VLMResult:
        """Query VLM with cache check. Returns cached result if fresh."""
        tid = person.track_id
        now = time.monotonic()

        if tid in self._cache:
            cached_at, cached = self._cache[tid]
            if now - cached_at < self._cache_ttl:
                logger.debug(f"GroqBackend cache hit: track {tid}")
                return cached

        vlm = await self._vlm.analyze_person(
            frame=frame,
            bbox=person.bbox,
            person_id=person.person_id,
            camera_id=camera_id,
            zone_id=person.zone_id,
            zone_name=person.zone_name,
            is_restricted=person.is_restricted,
        )
        # HARD RULE: never cache or use fallback results
        if vlm.backend_used == "fallback":
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person.person_id} camera_id={camera_id} "
                f"reason=groq_backend_received_fallback "
                f"desc=\"{vlm.description[:60]}\""
            )
            return None
        self._cache[tid] = (now, vlm)
        return vlm

    def _vlm_to_activity(
        self, vlm: VLMResult, person: TrackedPerson
    ) -> ActivityResult:
        """Convert a VLMResult into an ActivityResult for the pipeline."""
        from ai.analyzer.activity_analyzer import AnomalyFlag

        flags = []
        if vlm.activity_type == "theft_attempt":
            flags.append(AnomalyFlag.THEFT_DETECTED)
        elif vlm.activity_type == "safety_violation":
            flags.append(AnomalyFlag.MISCONDUCT_DETECTED)
        elif vlm.activity_type == "unauthorized_entry":
            flags.append(AnomalyFlag.RESTRICTED_ZONE)
        elif vlm.activity_type == "falling":
            flags.append(AnomalyFlag.POSSIBLE_FALL)
        elif vlm.activity_type == "loitering":
            flags.append(AnomalyFlag.LOITERING)
        elif vlm.activity_type == "running":
            flags.append(AnomalyFlag.FAST_MOVEMENT)

        return ActivityResult(
            person_id=person.person_id,
            track_id=person.track_id,
            activity_type=vlm.activity_type,
            anomaly_label=vlm.anomaly_label,
            description=vlm.description,
            confidence=vlm.confidence,
            flags=flags,
            zone_id=person.zone_id,
            zone_name=person.zone_name,
            dwell_time=person.dwell_time,
            backend_used=vlm.backend_used if vlm.backend_used else "groq",
        )

    def _empty_result(self, person: TrackedPerson) -> ActivityResult:
        """Return a minimal safe result when VLM returns fallback."""
        return ActivityResult(
            person_id=person.person_id,
            track_id=person.track_id,
            activity_type="unknown",
            anomaly_label="normal",
            description="",
            confidence=0.0,
            flags=[],
            zone_id=person.zone_id,
            zone_name=person.zone_name,
            dwell_time=person.dwell_time,
            backend_used="groq_fallback",
        )
