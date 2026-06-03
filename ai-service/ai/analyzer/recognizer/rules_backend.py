from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.analyzer.activity_analyzer import ActivityAnalyzer, ActivityResult
from ai.analyzer.recognizer.base import BaseActivityBackend
from ai.tracker.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)


class RulesBackend(BaseActivityBackend):
    """
    Rule-based activity recognition backend.

    Delegates directly to ActivityAnalyzer which uses bounding-box
    geometry, velocity, aspect ratio, and dwell time to classify:
      walking, running, standing, loitering, falling,
      carrying_object, crouching, handling_items, unauthorized_entry

    This backend is synchronous (no external API calls).
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._analyzer = ActivityAnalyzer(camera_id)
        logger.info(f"[{camera_id}] RulesBackend initialised")

    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        return self._analyzer.analyze(persons, carryable_objects)
