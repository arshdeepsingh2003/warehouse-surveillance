"""
pipeline/ai_frame_processor.py
────────────────────────────────
Real AI Frame Processor — replaces the mock FrameProcessor.

This is the main pipeline orchestrator. For every frame it:

  1. Frame skipping   → only process every N frames (speed optimisation)
  2. Detection        → PersonDetector (YOLO or HOG fallback)
  3. Tracking         → PersonTracker (IoU Kalman tracker)
  4. Activity         → ActivityAnalyzer (rule-based, VLM-ready)
  5. Rules            → RulesEngine (configurable alert policies)
  6. Overlay          → FrameOverlay (draws boxes on frame)
  7. JPEG encode      → for MJPEG streaming to dashboard
  8. API calls        → post activities + alerts to backend

Per-camera state (detector, tracker, analyzer, rules engine) is created
lazily on first frame for that camera. This means you can add cameras at
runtime without restarting.

Optimization flags (all configurable in .env):
  PROCESS_EVERY_N_FRAMES=3   → run AI on 1 of 3 frames, draw on all
  DEVICE=cpu / cuda:0        → GPU acceleration when available
  YOLO_CONFIDENCE=0.40       → lower = more detections, higher = more precise
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
from ai.analyzer.activity_analyzer import ActivityResult
from ai.analyzer.recognizer import ActivityRecognizer
from ai.rules.rules_engine import RulesEngine, AlertEvent
from ai.overlay.frame_overlay import FrameOverlay
from pipeline.api_client import APIClient
from streams.frame_reader import FrameData
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Per-camera pipeline bundle ────────────────────────────────────────────────

class _CameraPipeline:
    """All per-camera AI components bundled together."""

    def __init__(self, camera_id: str, detector: PersonDetector) -> None:
        self.camera_id  = camera_id
        self.detector   = detector              # shared across cameras (thread-safe for inference)
        self.tracker    = PersonTracker(camera_id)
        self.recognizer = ActivityRecognizer(camera_id)
        self.rules      = RulesEngine(camera_id)
        self.overlay    = FrameOverlay(camera_id)
        self.frame_skip_counter = 0
        # Cache last tracking results for frames we skip detection on
        self._last_persons:    list[TrackedPerson] = []
        self._last_activities: list[ActivityResult] = []


# ── Main processor ────────────────────────────────────────────────────────────

class AIFrameProcessor:
    """
    Real AI frame processor.

    Replace FrameProcessor in main.py to activate:
        from pipeline.ai_frame_processor import AIFrameProcessor
        processor = AIFrameProcessor(api_client)

    The interface is identical to FrameProcessor.
    """

    def __init__(self, api_client: APIClient) -> None:
        self._api = api_client

        # Shared detector instance (expensive to create, safe to share)
        logger.info(f"Initialising detector backend: {settings.DETECTOR_BACKEND}")
        self._detector = PersonDetector(
            model_path=        settings.YOLO_MODEL_PATH,
            confidence_thresh= settings.YOLO_CONFIDENCE,
            device=            settings.DEVICE,
            force_backend=     None if settings.DETECTOR_BACKEND == "auto"
                               else settings.DETECTOR_BACKEND,
        )
        logger.info(f"Detector ready: {self._detector.backend_name}")

        # Per-camera pipelines (created lazily)
        self._pipelines: dict[str, _CameraPipeline] = {}

        # Latest JPEG per camera (for MJPEG streaming)
        self._latest_frames: dict[str, bytes] = {}

        # Stats per camera
        self._cam_stats: dict[str, dict] = {}

        # Recent alert count (for analytics)
        self._alert_counts: defaultdict[str, int] = defaultdict(int)

    # ── Public: process one frame ─────────────────────────────────────────────

    async def process(self, frame_data: FrameData) -> None:
        """
        Entry point — called by StreamManager for every frame.

        Steps:
          1. Get or create the per-camera pipeline
          2. Run detection + tracking (every N frames)
          3. Classify activities
          4. Evaluate rules → generate alerts
          5. Draw overlay
          6. Encode JPEG
          7. Post results to backend (async, non-blocking)
        """
        cam_id = frame_data.camera_id
        frame  = frame_data.frame.copy()

        # ── Lazy pipeline init ────────────────────────────────────────────────
        if cam_id not in self._pipelines:
            self._pipelines[cam_id] = _CameraPipeline(cam_id, self._detector)
            logger.info(f"[{cam_id}] AI pipeline initialised")

        pipe = self._pipelines[cam_id]
        pipe.frame_skip_counter += 1
        run_ai = (pipe.frame_skip_counter % settings.PROCESS_EVERY_N_FRAMES == 0)

        persons:    list[TrackedPerson] = pipe._last_persons
        activities: list[ActivityResult]= pipe._last_activities
        alerts:     list[AlertEvent]    = []

        # ── Run AI (every N frames) ───────────────────────────────────────────
        if run_ai:
            # Step 1: Detect persons
            detections = self._detector.detect(frame)

            # Step 2: Track persons across frames
            persons = pipe.tracker.update(detections, frame)

            # Step 3: Detect carryable objects (for spatial activity analysis)
            carryable = self._detector.detect_carryable_objects(frame)

            # Step 4: Classify activities (via pluggable recognizer)
            activities = await pipe.recognizer.analyze(frame, persons, cam_id, carryable)

            # Step 4: Evaluate rules → alerts
            alerts = pipe.rules.evaluate(activities)

            # Cache for skipped frames
            pipe._last_persons    = persons
            pipe._last_activities = activities

        # ── Draw overlay (every frame) ────────────────────────────────────────
        annotated = pipe.overlay.draw(frame, persons, activities, alerts)

        # ── Encode JPEG for MJPEG stream ──────────────────────────────────────
        jpeg = self._encode_jpeg(annotated)
        self._latest_frames[cam_id] = jpeg

        # ── Update stats ──────────────────────────────────────────────────────
        self._cam_stats[cam_id] = {
            "camera_id":    cam_id,
            "frame_number": frame_data.frame_number,
            "timestamp":    frame_data.timestamp,
            "fps":          frame_data.source_fps,
            "persons":      len(persons),
            "alerts":       self._alert_counts[cam_id],
        }

        # ── Post to backend (fire-and-forget) ─────────────────────────────────
        if run_ai and (persons or alerts):
            asyncio.create_task(
                self._post_results(cam_id, persons, activities, alerts, annotated)
            )

    # ── Backend posting ───────────────────────────────────────────────────────

    async def _post_results(
        self,
        cam_id:     str,
        persons:    list[TrackedPerson],
        activities: list[ActivityResult],
        alerts:     list[AlertEvent],
        frame:      np.ndarray,
    ) -> None:
        """Post activity logs and alerts to the backend API."""
        now = datetime.now(timezone.utc).isoformat()

        # Activity logs (one per tracked person)
        for activity in activities:
            await self._api.post_activity(
                camera_id=     cam_id,
                zone=          activity.zone_id,
                person_id=     activity.person_id,
                activity_type= activity.activity_type,
                description=   activity.description,
                anomaly_label= activity.anomaly_label,
                dwell_seconds= int(activity.dwell_time),
                confidence=    activity.confidence,
            )

        # Alerts (only anomalies)
        for alert in alerts:
            self._alert_counts[cam_id] += 1
            snapshot = self._frame_to_b64(frame)
            await self._api.post_alert(
                camera_id=   cam_id,
                zone=        alert.zone_id,
                alert_type=  alert.alert_type,
                severity=    alert.severity,
                description= alert.description,
                person_id=   alert.person_id,
                confidence=  alert.confidence,
                snapshot_b64=snapshot,
            )

        # Frame update broadcast (real-time person positions for dashboard)
        if persons:
            person_list: list[dict] = []
            for i, p in enumerate(persons):
                act = activities[i].activity_type if i < len(activities) else "unknown"
                person_list.append({
                    "person_id":     p.person_id,
                    "zone":          p.zone_id,
                    "activity":      act,
                    "dwell_seconds": int(p.dwell_time),
                    "bbox":          list(p.bbox),
                    "center":        list(p.center),
                })
            await self._api.broadcast_frame_update(camera_id=cam_id, persons=person_list)

    # ── MJPEG interface (same as mock processor) ──────────────────────────────

    def get_latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        return self._latest_frames.get(cam_id)

    def get_all_stats(self) -> dict:
        return self._cam_stats

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        _, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY],
        )
        return buf.tobytes()

    def _frame_to_b64(self, frame: np.ndarray) -> str:
        """Encode a frame as base64 JPEG string (for snapshot in alert)."""
        # Downscale snapshot to reduce payload size
        small = cv2.resize(frame, (320, 180))
        jpeg  = self._encode_jpeg(small)
        return base64.b64encode(jpeg).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# VLM-ENHANCED PROCESSOR
# Extends AIFrameProcessor with VLM + ZoneSummarizer integration
# ─────────────────────────────────────────────────────────────────────────────

class VLMAIFrameProcessor(AIFrameProcessor):
    """
    Full AI+VLM+LLM pipeline processor.

    Adds on top of AIFrameProcessor:
      • VLMClient: analyzes cropped persons with a vision model
      • ZoneSummarizer: generates LLM zone summaries on a schedule
      • LLMClient: generates anomaly explanations for high-severity alerts
      • Hybrid decision: merges rule-based + VLM anomaly labels

    Activation (in .env):
        USE_VLM=true
        VLM_BACKEND=mock          # or: openai | anthropic | ollama | gemini
        LLM_BACKEND=mock          # or: openai | anthropic | ollama
        VLM_EVERY_N_FRAMES=30     # query VLM every ~3 seconds at 10fps

    The VLM runs asynchronously and never blocks the main frame loop.
    """

    def __init__(self, api_client: APIClient) -> None:
        super().__init__(api_client)

        from ai.vlm.vlm_client import VLMClient
        from ai.llm.llm_client import LLMClient
        from ai.llm.zone_summarizer import ZoneSummarizer

        self._vlm       = VLMClient()
        self._llm       = LLMClient()
        self._summarizer = ZoneSummarizer(api_client, self._llm)

        # Counter for VLM throttling (per camera)
        self._vlm_counters: defaultdict[str, int] = defaultdict(int)

        # Latest VLM results per person_id (merged into activities)
        self._vlm_cache: dict[str, dict] = {}

        logger.info(
            f"VLMAIFrameProcessor ready | "
            f"VLM={settings.VLM_BACKEND} | LLM={settings.LLM_BACKEND}"
        )

    async def warmup(self) -> None:
        """Pre-load VLM model weights."""
        await self._vlm.warmup()

    def start_background_tasks(self) -> None:
        """Launch ZoneSummarizer loop. Call after the event loop starts."""
        asyncio.create_task(self._summarizer.run())
        logger.info("ZoneSummarizer background task started")

    async def process(self, frame_data: FrameData) -> None:
        """
        Override: identical to parent but adds VLM analysis and
        feeds the ZoneSummarizer buffer.
        """
        cam_id = frame_data.camera_id
        frame  = frame_data.frame.copy()

        # Lazy pipeline init
        if cam_id not in self._pipelines:
            self._pipelines[cam_id] = _CameraPipeline(cam_id, self._detector)
            logger.info(f"[{cam_id}] VLM pipeline initialised")

        pipe = self._pipelines[cam_id]
        pipe.frame_skip_counter += 1
        self._vlm_counters[cam_id] += 1
        run_ai  = (pipe.frame_skip_counter % settings.PROCESS_EVERY_N_FRAMES == 0)
        run_vlm = (self._vlm_counters[cam_id] % settings.VLM_EVERY_N_FRAMES == 0)

        persons:    list[TrackedPerson] = pipe._last_persons
        activities: list[ActivityResult]= pipe._last_activities
        alerts:     list[AlertEvent]    = []

        if run_ai:
            detections = self._detector.detect(frame)
            persons    = pipe.tracker.update(detections, frame)

            # Detect carryable objects (for spatial activity analysis)
            carryable = self._detector.detect_carryable_objects(frame)

            activities = await pipe.recognizer.analyze(frame, persons, cam_id, carryable)
            alerts     = pipe.rules.evaluate(activities)
            pipe._last_persons    = persons
            pipe._last_activities = activities

        # ── VLM enrichment (async, throttled, per confirmed person) ──────────
        if run_vlm and persons and settings.USE_VLM:
            asyncio.create_task(
                self._run_vlm_batch(cam_id, frame.copy(), persons, activities)
            )

        # Draw overlay + encode JPEG
        annotated = pipe.overlay.draw(frame, persons, activities, alerts)
        jpeg = self._encode_jpeg(annotated)
        self._latest_frames[cam_id] = jpeg

        self._cam_stats[cam_id] = {
            "camera_id":    cam_id,
            "frame_number": frame_data.frame_number,
            "timestamp":    frame_data.timestamp,
            "fps":          frame_data.source_fps,
            "persons":      len(persons),
            "alerts":       self._alert_counts[cam_id],
        }

        if run_ai and (persons or alerts):
            asyncio.create_task(
                self._post_results_with_vlm(cam_id, persons, activities, alerts, annotated)
            )

    # ── VLM batch analysis ────────────────────────────────────────────────────

    async def _run_vlm_batch(
        self,
        cam_id:     str,
        frame:      np.ndarray,
        persons:    list[TrackedPerson],
        activities: list[ActivityResult],
    ) -> None:
        """
        Query VLM for up to MAX_PERSONS_PER_FRAME persons concurrently.

        Merges VLM results back into the activity cache so the next
        _post_results call uses VLM-enriched descriptions.
        """
        # Prioritize anomalous persons and persons in restricted zones
        priority = sorted(
            persons,
            key=lambda p: (p.is_restricted, p.dwell_time),
            reverse=True,
        )
        batch = priority[:settings.VLM_MAX_PERSONS_PER_FRAME]

        # Build lookup: track_id → activity
        act_map = {a.track_id: a for a in activities}

        # Fire all VLM calls concurrently
        tasks = [
            self._vlm.analyze_person(
                frame=         frame,
                bbox=          p.bbox,
                person_id=     p.person_id,
                camera_id=     cam_id,
                zone_id=       p.zone_id,
                zone_name=     p.zone_name,
                is_restricted= p.is_restricted,
                extra_context= (
                    f"Rule-based detection: {act_map[p.track_id].activity_type if p.track_id in act_map else 'unknown'}. "
                    f"Dwell time: {p.dwell_time:.0f}s."
                ),
            )
            for p in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for person, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.debug(f"VLM task error for {person.person_id}: {result}")
                continue

            # Cache VLM result (used in next post_results call)
            self._vlm_cache[person.person_id] = {
                "description":   result.description,
                "activity_type": result.activity_type,
                "anomaly_label": result.anomaly_label,
                "severity":      result.severity,
                "confidence":    result.confidence,
                "backend_used":  result.backend_used,
                "latency_ms":    result.latency_ms,
            }

            logger.info(
                f"[VLM] [{cam_id}] {person.person_id} → "
                f"{'⚠ ANOMALY' if result.is_anomaly else 'normal'} | "
                f"{result.description[:60]}… | {result.latency_ms}ms"
            )

    # ── Enhanced post_results with VLM enrichment ─────────────────────────────

    async def _post_results_with_vlm(
        self,
        cam_id:     str,
        persons:    list[TrackedPerson],
        activities: list[ActivityResult],
        alerts:     list[AlertEvent],
        frame:      np.ndarray,
    ) -> None:
        """
        Post activities enriched with VLM descriptions.
        Falls back to rule-based description when VLM cache is empty.
        """
        for activity in activities:
            # Hybrid: prefer VLM description if available
            vlm_data   = self._vlm_cache.get(activity.person_id, {})
            description = vlm_data.get("description") or activity.description
            act_type    = vlm_data.get("activity_type") or activity.activity_type
            confidence  = vlm_data.get("confidence",  activity.confidence)

            # Hybrid anomaly decision: anomaly if EITHER rule-based OR VLM says so
            anomaly = activity.anomaly_label
            if vlm_data.get("anomaly_label") == "anomaly":
                anomaly = "anomaly"

            await self._api.post_activity(
                camera_id=     cam_id,
                zone=          activity.zone_id,
                person_id=     activity.person_id,
                activity_type= act_type,
                description=   description,
                anomaly_label= anomaly,
                dwell_seconds= int(activity.dwell_time),
                confidence=    confidence,
            )

            # Feed ZoneSummarizer buffer
            self._summarizer.log_activity({
                "person_id":     activity.person_id,
                "zone":          activity.zone_id,
                "activity_type": act_type,
                "description":   description,
                "anomaly_label": anomaly,
                "dwell_time":    activity.dwell_time,
            })

        # Alerts: generate LLM explanation for high-severity alerts
        for alert in alerts:
            self._alert_counts[cam_id] += 1
            snapshot = self._frame_to_b64(frame)

            # Get VLM-enriched description for alert
            vlm_data    = self._vlm_cache.get(alert.person_id, {})
            description = vlm_data.get("description") or alert.description

            await self._api.post_alert(
                camera_id=   cam_id,
                zone=        alert.zone_id,
                alert_type=  alert.alert_type,
                severity=    alert.severity,
                description= description,
                person_id=   alert.person_id,
                confidence=  alert.confidence,
                snapshot_b64=snapshot,
            )

            # Feed ZoneSummarizer alert buffer
            self._summarizer.log_alert({
                "person_id":  alert.person_id,
                "zone":       alert.zone_id,
                "alert_type": alert.alert_type,
                "severity":   alert.severity,
                "description":description,
            })

            # LLM explanation for high-severity alerts (async, non-blocking)
            if alert.severity == "high" and settings.USE_LLM:
                asyncio.create_task(
                    self._post_alert_explanation(alert, description)
                )

        if persons:
            person_list: list[dict] = []
            for i, p in enumerate(persons):
                act = activities[i].activity_type if i < len(activities) else "unknown"
                person_list.append({
                    "person_id":     p.person_id,
                    "zone":          p.zone_id,
                    "activity":      act,
                    "dwell_seconds": int(p.dwell_time),
                    "bbox":          list(p.bbox),
                    "center":        list(p.center),
                })
            await self._api.broadcast_frame_update(camera_id=cam_id, persons=person_list)

    async def _post_alert_explanation(
        self,
        alert: AlertEvent,
        description: str,
    ) -> None:
        """Generate and broadcast an LLM explanation for a high-severity alert."""
        try:
            explanation = await self._llm.explain_anomaly(
                alert_type=  alert.alert_type,
                description= description,
                zone_name=   alert.zone_name,
                person_id=   alert.person_id,
                dwell_time=  alert.dwell_time,
            )

            await self._api._post("/api/v1/events/broadcast", {
                "type":            "alert_explanation",
                "alert_type":      alert.alert_type,
                "person_id":       alert.person_id,
                "camera_id":       alert.camera_id,
                "explanation":     explanation.explanation,
                "recommendation":  explanation.recommendation,
                "false_positive":  explanation.false_positive_probability,
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            })
            logger.info(
                f"[LLM] Explanation posted for {alert.alert_type} | {alert.person_id}"
            )
        except Exception as e:
            logger.debug(f"LLM explanation error: {e}")
