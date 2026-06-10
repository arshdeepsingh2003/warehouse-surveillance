from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.analyzer.activity_analyzer import ActivityResult
from ai.analyzer.recognizer.base import BaseActivityBackend
from ai.tracker.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)


class MockBackend(BaseActivityBackend):
    """
    Legacy mock backend — no longer generates random activities.

    Logs a deprecation warning and returns a deterministic "unknown"
    activity label for every tracked person. Use RulesBackend (the
    default) for real activity detection from movement patterns.

    This class exists only to avoid breaking existing configurations
    that explicitly set ACTIVITY_BACKEND=mock.
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        logger.warning(
            f"[{camera_id}] MockBackend is deprecated — no activity "
            "classification performed. Set ACTIVITY_BACKEND=rules."
        )

    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        results: list[ActivityResult] = []
        for person in persons:
            results.append(ActivityResult(
                person_id=person.person_id,
                track_id=person.track_id,
                activity_type="unknown",
                anomaly_label="normal",
                description=(
                    f"{person.person_id} in '{person.zone_name}' "
                    f"(mock — no classification)."
                ),
                confidence=0.01,
                flags=[],
                zone_id=person.zone_id,
                zone_name=person.zone_name,
                dwell_time=person.dwell_time,
                backend_used="mock",
            ))
        return results
