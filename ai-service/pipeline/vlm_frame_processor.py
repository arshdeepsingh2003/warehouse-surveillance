"""
pipeline/vlm_frame_processor.py
─────────────────────────────────
VLM-enhanced Frame Processor.

Extends AIFrameProcessor by adding VLM analysis on top of the
existing YOLO + tracker + rules pipeline.

Processing flow per frame:
  1. YOLO detection           (every PROCESS_EVERY_N_FRAMES)
  2. Person tracking          (every frame, cached on skip)
  3. VLM analysis             (every VLM_EVERY_N_FRAMES — much less frequent)
      └── crop person region
      └── query VLM → rich description
      └── merge into ActivityResult
  4. Rules engine             (uses merged result)
  5. LLM anomaly explanation  (async, only for HIGH alerts)
  6. Frame overlay            (annotated JPEG for MJPEG stream)
  7. Backend POST             (activities + alerts + VLM descriptions)

VLM analysis is intentionally less frequent than detection because:
  - VLM calls take 100ms–3s (vs ~20ms for YOLO)
  - Most frames don't change meaningfully between VLM calls
  - Caching handles stable scenes efficiently

The result: you get YOLO's speed + VLM's intelligence,
at a fraction of the cost of running VLM on every frame.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from ai.detector.person_detector import PersonDetector
from ai.tracker.person_tracker import PersonTracker, TrackedPerson
from ai.analyzer.activity_analyzer import ActivityAnalyzer, ActivityResult
from ai.rules.rules_engine import RulesEngine, AlertEvent
from ai.overlay.frame_overlay import FrameOverlay
from ai.vlm.vlm_client import VLMClient, VLMResult
from ai.llm.llm_client import LLMClient
from ai.llm.zone_summarizer import ZoneSummarizer
from pipeline.api_client import APIClient
from streams.frame_reader import FrameData
from config.settings import settings

logger = logging.getLogger(__name__)


class _CameraPipelineVLM:
    """All per-camera state including VLM components."""

    def __init__(self, camera_id: str, detector: PersonDetector) -> None:
        self.camera_id            = camera_id
        self.detector             = detector
        self.tracker              = PersonTracker(camera_id)
        self.analyzer             = ActivityAnalyzer(camera_id)
        self.rules                = RulesEngine(camera_id)
        self.overlay              = FrameOverlay(camera_id)
        self.frame_skip_counter   = 0
        self.vlm_skip_counter     = 0
        self._last_persons:       list[TrackedPerson] = []
        self._last_activities:    list[ActivityResult] = []
        # VLM results per person (person_id → VLMResult)
        self._vlm_results:        dict[str, VLMResult] = {}


class VLMFrameProcessor:
    """
    VLM-enhanced frame processor.

    Activate by setting USE_VLM=true in .env.
    Requires VLM_BACKEND to be configured (or uses mock by default).

    Usage in main.py:
        from pipeline.vlm_frame_processor import VLMFrameProcessor
        processor = VLMFrameProcessor(api_client)
    """

    def __init__(self, api_client: APIClient) -> None:
        self._api = api_client

        # Shared model objects
        self._detector = PersonDetector(
            model_path=        settings.YOLO_MODEL_PATH,
            confidence_thresh= settings.YOLO_CONFIDENCE,
            device=            settings.DEVICE,
            force_backend=     None if settings.DETECTOR_BACKEND == "auto"
                               else settings.DETECTOR_BACKEND,
        )
        self._vlm = VLMClient()
        self._llm = LLMClient()
        self._summarizer = ZoneSummarizer(api_client, self._llm)

        # Per-camera state
        self._pipelines: dict[str, _CameraPipelineVLM] = {}

        # MJPEG frames
        self._latest_frames: dict[str, bytes] = {}
        self._cam_stats:     dict[str, dict]  = {}
        self._alert_counts:  defaultdict[str, int] = defaultdict(int)

        logger.info(
            f"VLMFrameProcessor ready | "
            f"Detector: {self._detector.backend_name} | "
            f"VLM: {settings.VLM_BACKEND} | "
            f"LLM: {settings.LLM_BACKEND}"
        )

    async def start(self) -> None:
        """Warm up models and start background tasks."""
        await self._vlm.warmup()
        asyncio.create_task(self._summarizer.run())
        logger.info("VLMFrameProcessor started (summarizer running in background)")

    # ── Main entry point ──────────────────────────────────────────────────────

    async def process(self, frame_data: FrameData) -> None:
        """
        Process one frame through the full VLM pipeline.
        Called by StreamManager for every frame.
        """
        cam_id = frame_data.camera_id
        frame  = frame_data.frame.copy()

        # Lazy init
        if cam_id not in self._pipelines:
            self._pipelines[cam_id] = _CameraPipelineVLM(cam_id, self._detector)
            logger.info(f"[{cam_id}] VLM pipeline initialised")

        pipe = self._pipelines[cam_id]
        pipe.frame_skip_counter += 1
        pipe.vlm_skip_counter   += 1

        run_cv  = (pipe.frame_skip_counter % settings.PROCESS_EVERY_N_FRAMES == 0)
        run_vlm = (pipe.vlm_skip_counter   % settings.VLM_EVERY_N_FRAMES      == 0)

        persons    = pipe._last_persons
        activities = pipe._last_activities
        alerts:    list[AlertEvent] = []

        # ── Step 1: CV pipeline (detection + tracking) ────────────────────────
        if run_cv:
            detections = self._detector.detect(frame)
            persons    = pipe.tracker.update(detections, frame)
            pipe._last_persons = persons

        # ── Step 2: VLM analysis (much less frequent) ─────────────────────────
        if run_vlm and persons and settings.USE_VLM:
            vlm_tasks = [
                self._run_vlm_for_person(pipe, person, frame)
                for person in persons[:settings.VLM_MAX_PERSONS_PER_FRAME]
            ]
            # Run VLM calls concurrently for all persons in this frame
            vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True)

            for person, result in zip(persons, vlm_results):
                if isinstance(result, VLMResult):
                    pipe._vlm_results[person.person_id] = result

        # ── Step 3: Activity analysis (CV rules + optional VLM merge) ─────────
        if run_cv:
            activities = pipe.analyzer.analyze(persons)

            # Merge VLM descriptions into rule-based results
            for act in activities:
                vlm = pipe._vlm_results.get(act.person_id)
                if vlm:
                    act.description = vlm.description
                    # VLM overrides anomaly if it detects something the rules missed
                    if vlm.is_anomaly and act.anomaly_label == "normal":
                        act.anomaly_label = "anomaly"
                        logger.info(
                            f"[{cam_id}] VLM upgraded {act.person_id} to anomaly: {vlm.description[:60]}…"
                        )

            pipe._last_activities = activities

            # ── Step 4: Rules engine ──────────────────────────────────────────
            alerts = pipe.rules.evaluate(activities)

        # ── Step 5: Draw overlay ──────────────────────────────────────────────
        annotated = pipe.overlay.draw(frame, persons, activities, alerts)
        jpeg      = self._encode_jpeg(annotated)
        self._latest_frames[cam_id] = jpeg

        self._cam_stats[cam_id] = {
            "camera_id":    cam_id,
            "persons":      len(persons),
            "alerts":       self._alert_counts[cam_id],
            "vlm_enabled":  settings.USE_VLM,
            "vlm_backend":  settings.VLM_BACKEND,
        }

        # ── Step 6: Post to backend (fire-and-forget) ─────────────────────────
        if run_cv and (persons or alerts):
            asyncio.create_task(
                self._post_results(cam_id, persons, activities, alerts, annotated)
            )

    # ── VLM per-person analysis ───────────────────────────────────────────────

    async def _run_vlm_for_person(
        self,
        pipe:   _CameraPipelineVLM,
        person: TrackedPerson,
        frame:  np.ndarray,
    ) -> Optional[VLMResult]:
        """
        Query the VLM for one tracked person.
        Returns VLMResult or None on error.
        """
        try:
            return await self._vlm.analyze_person(
                frame=         frame,
                bbox=          person.bbox,
                person_id=     person.person_id,
                camera_id=     pipe.camera_id,
                zone_id=       person.zone_id,
                zone_name=     person.zone_name,
                is_restricted= person.is_restricted,
            )
        except Exception as e:
            logger.debug(f"VLM error for {person.person_id}: {e}")
            return None

    # ── Backend posting ───────────────────────────────────────────────────────

    async def _post_results(
        self,
        cam_id:     str,
        persons:    list[TrackedPerson],
        activities: list[ActivityResult],
        alerts:     list[AlertEvent],
        frame:      np.ndarray,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        for activity in activities:
            record = {
                "id":            str(uuid.uuid4()),
                "person_id":     activity.person_id,
                "camera_id":     cam_id,
                "zone":          activity.zone_id,
                "activity_type": activity.activity_type,
                "description":   activity.description,
                "anomaly_label": activity.anomaly_label,
                "dwell_seconds": int(activity.dwell_time),
                "confidence":    activity.confidence,
                "timestamp":     now,
            }
            await self._api.post_activity(**{
                k: v for k, v in record.items()
                if k in ["camera_id", "zone", "person_id", "activity_type",
                          "description", "anomaly_label", "dwell_seconds", "confidence"]
            })
            # Feed into summarizer buffer
            self._summarizer.log_activity(record)

        for alert in alerts:
            self._alert_counts[cam_id] += 1
            await self._api.post_alert(
                camera_id=   cam_id,
                zone=        alert.zone_id,
                alert_type=  alert.alert_type,
                severity=    alert.severity,
                description= alert.description,
                person_id=   alert.person_id,
                confidence=  alert.confidence,
                snapshot_b64=self._frame_to_b64(frame),
            )
            self._summarizer.log_alert(alert.to_dict())

            # Async LLM anomaly explanation for HIGH alerts
            if alert.severity == "high" and settings.USE_LLM:
                asyncio.create_task(
                    self._post_anomaly_explanation(alert, cam_id)
                )

        if persons:
            await self._api.broadcast_frame_update(
                camera_id=  cam_id,
                person_id=  persons[0].person_id,
                zone=       persons[0].zone_id,
                activity=   activities[0].activity_type if activities else "unknown",
                dwell_secs= int(persons[0].dwell_time),
            )

    async def _post_anomaly_explanation(self, alert: AlertEvent, cam_id: str) -> None:
        """Generate and post an LLM explanation for a high-severity alert."""
        try:
            explanation = await self._llm.explain_anomaly(
                alert_type=  alert.alert_type,
                description= alert.description,
                zone_name=   alert.zone_name,
                person_id=   alert.person_id,
                dwell_time=  alert.dwell_time,
            )
            await self._api._post("/api/v1/events/broadcast", {
                "type":        "anomaly_explanation",
                "alert_type":  alert.alert_type,
                "person_id":   alert.person_id,
                "camera_id":   cam_id,
                "explanation": explanation.explanation,
                "recommendation": explanation.recommendation,
                "false_positive_probability": explanation.false_positive_probability,
            })
            logger.info(f"Anomaly explanation posted for {alert.person_id}")
        except Exception as e:
            logger.debug(f"Anomaly explanation error: {e}")

    # ── MJPEG interface ───────────────────────────────────────────────────────

    def get_latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        return self._latest_frames.get(cam_id)

    def get_all_stats(self) -> dict:
        return self._cam_stats

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY])
        return buf.tobytes()

    def _frame_to_b64(self, frame: np.ndarray) -> str:
        small = cv2.resize(frame, (320, 180))
        jpeg  = self._encode_jpeg(small)
        return base64.b64encode(jpeg).decode("utf-8")
