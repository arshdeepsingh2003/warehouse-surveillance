from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.analyzer.activity_analyzer import ActivityResult
from ai.analyzer.recognizer.base import BaseActivityBackend
from ai.tracker.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)

_RANDOM_ACTIVITIES = [
    ("walking", "normal",
     "Person walking through the zone at normal pace."),
    ("handling_items", "normal",
     "Worker handling inventory boxes at shelf rack."),
    ("standing", "normal",
     "Person standing near workstation, reviewing clipboard."),
    ("carrying_object", "normal",
     "Worker carrying a cardboard box towards storage area."),
    ("walking", "normal",
     "Person moving between aisle sections carrying a scanner."),
    ("loitering", "anomaly",
     "Individual standing idle near loading dock for extended period."),
    ("running", "anomaly",
     "Person running in warehouse — prohibited activity."),
    ("falling", "anomaly",
     "Worker appears to have fallen near shelf rack."),
]


class MockBackend(BaseActivityBackend):
    """
    Mock activity recognition backend for demo / testing.

    Assigns random activities to each tracked person.
    Useful for UI development and demonstrations without real AI.

    This preserves the original demo behaviour while still using
    the recognizer abstraction.
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        logger.info(f"[{camera_id}] MockBackend initialised (demo mode)")

    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        results: list[ActivityResult] = []
        for person in persons:
            act_type, anomaly_label, description = random.choice(_RANDOM_ACTIVITIES)
            results.append(ActivityResult(
                person_id=person.person_id,
                track_id=person.track_id,
                activity_type=act_type,
                anomaly_label=anomaly_label,
                description=description,
                confidence=round(random.uniform(0.70, 0.95), 2),
                flags=[],
                zone_id=person.zone_id,
                zone_name=person.zone_name,
                dwell_time=person.dwell_time,
            ))
        return results
