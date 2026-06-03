from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.analyzer.activity_analyzer import ActivityResult
from ai.analyzer.recognizer.base import BaseActivityBackend
from ai.tracker.person_tracker import TrackedPerson
from config.settings import settings

logger = logging.getLogger(__name__)

_BACKEND_MAP: dict[str, type[BaseActivityBackend]] = {}


def register_backend(name: str, backend_cls: type[BaseActivityBackend]) -> None:
    """Register a backend class so it can be selected via ACTIVITY_BACKEND."""
    _BACKEND_MAP[name] = backend_cls
    logger.debug(f"Activity backend registered: {name} -> {backend_cls.__name__}")


class ActivityRecognizer:
    """
    Pluggable activity recognition service.

    Selects the backend based on the ACTIVITY_BACKEND config:
      rules  → Rule-based only via ActivityAnalyzer (fast, no API calls)
      groq   → Groq VLM for every tracked person
      hybrid → Rules first, VLM fallback for low-confidence activities
      mock   → Random activity generation for demo/testing

    Each camera gets its own recognizer instance (per-camera state).

    Usage:
        recognizer = ActivityRecognizer(camera_id="cam-01")
        await recognizer.warmup()
        activities = await recognizer.analyze(frame, persons, "cam-01")
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._backend: Optional[BaseActivityBackend] = None

    def _ensure_backend(self) -> BaseActivityBackend:
        if self._backend is not None:
            return self._backend

        mode = settings.ACTIVITY_BACKEND
        backend_cls = _BACKEND_MAP.get(mode)

        if backend_cls is not None:
            self._backend = backend_cls(camera_id=self._camera_id)
        else:
            # Fallback chain
            if mode == "rules":
                from ai.analyzer.recognizer.rules_backend import RulesBackend
                self._backend = RulesBackend(self._camera_id)
            elif mode == "groq":
                from ai.analyzer.recognizer.groq_backend import GroqBackend
                self._backend = GroqBackend(self._camera_id)
            elif mode == "hybrid":
                from ai.analyzer.recognizer.hybrid_backend import HybridBackend
                self._backend = HybridBackend(self._camera_id)
            elif mode == "mock":
                from ai.analyzer.recognizer.mock_backend import MockBackend
                self._backend = MockBackend(self._camera_id)
            else:
                logger.warning(
                    f"Unknown ACTIVITY_BACKEND={mode!r}, falling back to 'rules'"
                )
                from ai.analyzer.recognizer.rules_backend import RulesBackend
                self._backend = RulesBackend(self._camera_id)
                mode = "rules"

        logger.info(
            f"[{self._camera_id}] ActivityRecognizer backend: {mode} "
            f"({type(self._backend).__name__})"
        )
        return self._backend

    async def warmup(self) -> None:
        backend = self._ensure_backend()
        await backend.warmup()

    async def analyze(
        self,
        frame: np.ndarray,
        persons: list[TrackedPerson],
        camera_id: str,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        backend = self._ensure_backend()
        return await backend.analyze(frame, persons, camera_id, carryable_objects)


# Register built-in backends

def _register_builtins() -> None:
    try:
        from ai.analyzer.recognizer.rules_backend import RulesBackend
        register_backend("rules", RulesBackend)
    except ImportError:
        pass
    try:
        from ai.analyzer.recognizer.groq_backend import GroqBackend
        register_backend("groq", GroqBackend)
    except ImportError:
        pass
    try:
        from ai.analyzer.recognizer.hybrid_backend import HybridBackend
        register_backend("hybrid", HybridBackend)
    except ImportError:
        pass
    try:
        from ai.analyzer.recognizer.mock_backend import MockBackend
        register_backend("mock", MockBackend)
    except ImportError:
        pass


_register_builtins()
