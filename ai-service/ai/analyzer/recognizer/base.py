from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.tracker.person_tracker import TrackedPerson
from ai.analyzer.activity_analyzer import ActivityResult


class BaseActivityBackend(ABC):
    """
    Abstract base for all activity recognition backends.

    Each backend implements analyze() which takes the current frame +
    tracked persons and returns classified ActivityResult objects.
    """

    @abstractmethod
    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        ...

    async def warmup(self) -> None:
        """Optional pre-loading — called once at startup."""
