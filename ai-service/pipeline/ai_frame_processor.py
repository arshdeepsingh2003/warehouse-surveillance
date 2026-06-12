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
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from ai.crop.crop_manager import CropManager
from ai.detector.person_detector import PersonDetector
from ai.tracker.person_tracker import PersonTracker, TrackedPerson
from ai.analyzer.activity_analyzer import ActivityResult
from ai.analyzer.recognizer import ActivityRecognizer
from ai.rules.rules_engine import RulesEngine, AlertEvent
from ai.overlay.frame_overlay import FrameOverlay
from ai.vlm.vlm_client import VLMClient, RateLimitError
from ai.vlm.event_engine import EventEngine
from pipeline.api_client import APIClient
from streams.frame_reader import FrameData
from config.settings import settings

logger = logging.getLogger(__name__)

_TRACE_FILE = os.path.join(os.path.dirname(__file__), "..", "vlm_overlay_trace.jsonl")
_TRACE_FILE = os.path.normpath(_TRACE_FILE)


def _write_trace(stage: str, **kwargs) -> None:
    """Append a JSON line to the overlay trace file."""
    try:
        record = {"stage": stage, "ts": time.time(), **kwargs}
        with open(_TRACE_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


# ── Per-camera pipeline bundle ────────────────────────────────────────────────

class _CameraPipeline:
    """All per-camera AI components bundled together."""

    def __init__(self, camera_id: str, detector: PersonDetector,
                 debug_dir: Optional[str] = None) -> None:
        self.camera_id  = camera_id
        self.detector   = detector              # shared across cameras (thread-safe for inference)
        self.tracker    = PersonTracker(camera_id, debug_dir=debug_dir)
        self.recognizer = ActivityRecognizer(camera_id)
        self.rules      = RulesEngine(camera_id)
        self.overlay    = FrameOverlay(camera_id)
        self.crop_mgr   = CropManager(camera_id)
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
            img_size=          settings.YOLO_IMG_SIZE,
            use_tta=           settings.YOLO_USE_TTA,
            use_clahe=         settings.YOLO_USE_CLAHE,
            debug_dir=         settings.DEBUG_SAVE_DIR if settings.DEBUG_SAVE_IMAGES else None,
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
        proc_ts = time.perf_counter()

        # ── Lazy pipeline init ────────────────────────────────────────────────
        if cam_id not in self._pipelines:
            self._pipelines[cam_id] = _CameraPipeline(
                cam_id, self._detector,
                debug_dir=settings.DEBUG_SAVE_DIR if settings.DEBUG_SAVE_IMAGES else None,
            )
            logger.info(f"[{cam_id}] AI pipeline initialised")

        pipe = self._pipelines[cam_id]
        pipe.frame_skip_counter += 1
        run_ai = (pipe.frame_skip_counter % settings.PROCESS_EVERY_N_FRAMES == 0)

        persons:    list[TrackedPerson] = pipe._last_persons
        activities: list[ActivityResult]= pipe._last_activities
        alerts:     list[AlertEvent]    = []
        n_detections = 0
        n_raw_detections = 0

        # ── Run AI (every N frames) ───────────────────────────────────────────
        if run_ai:
            t_ai_start = time.perf_counter()

            # Step 1: Detect persons
            detections = self._detector.detect(frame, cam_id)
            n_detections = len(detections)

            # ── Detection-to-tracker association logging ──────────────────────
            logger.debug(
                f"[{cam_id}] Detector → Tracker: {n_detections} detections passed to tracker"
            )
            for idx, d in enumerate(detections):
                logger.debug(
                    f"[{cam_id}]   Det #{idx}: "
                    f"box=({d.x1},{d.y1},{d.x2},{d.y2}) "
                    f"conf={d.confidence:.4f} area={d.area}"
                )

            # Step 2: Track persons across frames
            persons = pipe.tracker.update(detections, frame)

            # ── Association summary ───────────────────────────────────────────
            n_active = sum(1 for p in persons if not p.is_lost)
            n_lost   = sum(1 for p in persons if p.is_lost)
            logger.debug(
                f"[{cam_id}] Tracker → downstream: {len(persons)} tracked "
                f"(active={n_active}, lost={n_lost}) | "
                f"ids=[{', '.join(p.person_id for p in persons)}]"
            )

            # Step 3: Detect carryable objects (for spatial activity analysis)
            carryable = self._detector.detect_carryable_objects(frame, cam_id)

            # Step 4: Classify activities (via pluggable recognizer)
            activities = await pipe.recognizer.analyze(frame, persons, cam_id, carryable)

            # Step 5: Evaluate rules → alerts
            alerts = pipe.rules.evaluate(activities)

            # Cache for skipped frames
            pipe._last_persons    = persons
            pipe._last_activities = activities

            t_ai_elapsed = time.perf_counter() - t_ai_start

            logger.info(
                f"[{cam_id}] Frame {frame_data.frame_number:>6d} | AI | "
                f"detections={n_detections} | "
                f"active_tracks={n_active} | "
                f"total_tracks={len(persons)} | "
                f"alerts={len(alerts)} | "
                f"ai_ms={t_ai_elapsed*1000:.1f}"
            )

            # ── Save debug images (stage-by-stage) ────────────────────────────
            if settings.DEBUG_SAVE_IMAGES:
                self._save_debug_frame(cam_id, frame_data.frame_number, "input", frame)
                # Draw raw detections on a copy
                raw_viz = frame.copy()
                for d in detections:
                    cv2.rectangle(raw_viz, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    cv2.putText(raw_viz, f"{d.confidence:.2f}", (d.x1, d.y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self._save_debug_frame(cam_id, frame_data.frame_number, "detections", raw_viz)

        else:
            n_active = sum(1 for p in persons if not p.is_lost)
            logger.debug(
                f"[{cam_id}] Frame {frame_data.frame_number:>6d} | SKIP | "
                f"cached_tracks={len(persons)} (active={n_active})"
            )

        # ── Draw overlay ──────────────────────────────────────────────────────
        if run_ai:
            frame_shape_before = frame.shape
            n_persons_to_draw  = len([p for p in persons if not p.is_lost])

            annotated = pipe.overlay.draw(frame, persons, activities, alerts)
            has_annotations = n_persons_to_draw > 0 or len(alerts) > 0

            if settings.DEBUG_SAVE_IMAGES:
                self._save_debug_frame(cam_id, frame_data.frame_number, "annotated", annotated)

            jpeg_data = self._encode_jpeg(annotated)
            self._latest_frames[cam_id] = jpeg_data

            if settings.DEBUG_SAVE_IMAGES:
                tx_dir = os.path.join(settings.DEBUG_SAVE_DIR, cam_id, "transmitted_jpeg")
                os.makedirs(tx_dir, exist_ok=True)
                with open(os.path.join(tx_dir, f"frame_{frame_data.frame_number:06d}.jpg"), "wb") as f:
                    f.write(jpeg_data)

            logger.info(
                f"[{cam_id}] TRANSMIT | "
                f"frame={frame_data.frame_number} | "
                f"detections={n_detections} | "
                f"active_tracks={n_persons_to_draw} | "
                f"alerts={len(alerts)} | "
                f"frame_in={frame_shape_before[1]}x{frame_shape_before[0]} | "
                f"annotated={'yes' if has_annotations else 'no'} | "
                f"jpeg_encoded=yes | "
                f"jpeg_bytes={len(jpeg_data)} | "
                f"transmitted=yes"
            )
        else:
            cached_jpeg = self._latest_frames.get(cam_id)
            logger.info(
                f"[{cam_id}] TRANSMIT | "
                f"frame={frame_data.frame_number} | SKIP | "
                f"jpeg_cached={'yes' if cached_jpeg is not None else 'NO'}"
            )

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
                camera_id=       cam_id,
                zone=            activity.zone_id,
                person_id=       activity.person_id,
                activity_type=   activity.activity_type,
                description=     activity.description,
                anomaly_label=   activity.anomaly_label,
                dwell_seconds=   int(activity.dwell_time),
                confidence=      activity.confidence,
                objects_detected=[],
                backend_used=    activity.backend_used,
                latency_ms=      0,
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
                source=      "rules_engine",
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
            logger.info(
                f"🔍 TRACE[pre-ws] camera={cam_id} | "
                f"persons={len(person_list)} | "
                f"ids=[{', '.join(p['person_id'] for p in person_list)}] | "
                f"bboxes=[{'; '.join(str(p['bbox']) for p in person_list)}]"
            )
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

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def _save_debug_frame(
        self,
        cam_id:  str,
        frame_n: int,
        stage:   str,
        frame:   np.ndarray,
    ) -> None:
        """Save a frame to disk for per-camera stage-by-stage debug analysis."""
        save_dir = os.path.join(settings.DEBUG_SAVE_DIR, cam_id, stage)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"frame_{frame_n:06d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


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

        from ai.llm.llm_client import LLMClient
        from ai.llm.zone_summarizer import ZoneSummarizer

        self._vlm       = VLMClient()
        self._event_engine = EventEngine(self._vlm)
        self._llm       = LLMClient()
        self._summarizer = ZoneSummarizer(api_client, self._llm)

        # Track UUIDs currently being analyzed (prevents duplicate in-flight VLM calls)
        self._vlm_inflight_persons: set[str] = set()

        # ── Audit counters ────────────────────────────────────────────────────
        self._audit: dict[str, int] = {
            "persons_detected": 0,
            "vlm_tasks_created": 0,
            "groq_requests_started": 0,
            "groq_requests_completed": 0,
            "groq_requests_failed": 0,
            "vlm_results_parsed": 0,
            "vlm_insights_posted": 0,
            "vlm_results_orphaned": 0,
            "vlm_results_matched_by_position": 0,
            "vlm_results_attached_to_original_track": 0,
        }
        self._track_creation_times: dict[str, float] = {}  # track_uuid → creation time
        self._track_lifetimes: list[float] = []
        self._person_traces: dict[str, list[str]] = {}  # track_uuid → trace log

        # ── Verification checkpoints ──────────────────────────────────────────
        _groq_key_loaded = bool(settings.GROQ_API_KEY)
        _vlm_backend = settings.VLM_BACKEND
        _use_vlm = settings.USE_VLM
        _event_driven = settings.USE_EVENT_DRIVEN_VLM
        logger.info(
            f"VERIFY[1] GROQ_API_KEY loaded={'yes' if _groq_key_loaded else 'NO'} "
            f"(len={len(settings.GROQ_API_KEY) if _groq_key_loaded else 0})"
        )
        logger.info(
            f"VERIFY[2] VLM_BACKEND={_vlm_backend} "
            f"USE_VLM={_use_vlm} "
            f"EVENT_DRIVEN={_event_driven}"
        )
        logger.info(
            f"VLMAIFrameProcessor ready | "
            f"VLM={settings.VLM_BACKEND} | LLM={settings.LLM_BACKEND} | "
            f"EventDriven={_event_driven}"
        )

    async def warmup(self) -> None:
        """Pre-load VLM model weights."""
        await self._vlm.warmup()

    def clear_vlm_state(self) -> None:
        """Clear all VLM caches — call before restarting streams or after reconfig."""
        self._vlm.clear_cache()
        self._vlm_inflight_persons.clear()
        self._audit = {k: 0 for k in self._audit}
        logger.info("[VLM] Cleared all VLM state (cache, inflight, audit)")

    def start_background_tasks(self) -> None:
        """Launch ZoneSummarizer loop and crop cleanup tasks. Call after the event loop starts."""
        asyncio.create_task(self._summarizer.run())
        asyncio.create_task(self._audit_summary_loop())
        asyncio.create_task(self._event_engine_cleanup_loop())
        logger.info("ZoneSummarizer background task started")
        # Start crop cleanup for all existing pipelines
        for pipe in self._pipelines.values():
            pipe.crop_mgr.start_cleanup_task()

    async def _audit_summary_loop(self) -> None:
        """Log audit counters and tracker metrics every 60 seconds."""
        while True:
            await asyncio.sleep(60)
            counters = " | ".join(f"{k}={v}" for k, v in self._audit.items())
            logger.info(f"[AUDIT] {counters}")

            # Tracker-level metrics from all pipelines
            all_avg_lifetimes = []
            all_median_lifetimes = []
            all_id_switches = 0
            all_tracks_deleted_before_vlm = 0
            for pipe in self._pipelines.values():
                metrics = pipe.tracker.get_audit_metrics()
                all_avg_lifetimes.append(metrics.get("average_track_lifetime_s", 0))
                all_median_lifetimes.append(metrics.get("median_track_lifetime_s", 0))
                all_id_switches += metrics.get("id_switches_per_minute", 0)
                all_tracks_deleted_before_vlm += metrics.get("tracks_deleted_before_vlm", 0)

            overall_avg = (sum(all_avg_lifetimes) / len(all_avg_lifetimes)) if all_avg_lifetimes else 0
            overall_median = (sum(all_median_lifetimes) / len(all_median_lifetimes)) if all_median_lifetimes else 0

            logger.info(
                f"[AUDIT] TRACKER_METRICS "
                f"average_track_lifetime_s={overall_avg:.2f} "
                f"median_track_lifetime_s={overall_median:.2f} "
                f"id_switches_per_minute={all_id_switches} "
                f"tracks_deleted_before_vlm={all_tracks_deleted_before_vlm} "
                f"vlm_attached_to_original={self._audit.get('vlm_results_attached_to_original_track', 0)} "
                f"vlm_reattributed_by_position={self._audit.get('vlm_results_matched_by_position', 0)}"
            )

                # Log EventEngine metrics
            ee_metrics = self._event_engine.get_metrics()
            logger.info(
                f"[AUDIT] EVENT_ENGINE "
                f"requests_started={ee_metrics['requests_started']} "
                f"requests_completed={ee_metrics['requests_completed']} "
                f"requests_failed={ee_metrics['requests_failed']} "
                f"requests_429={ee_metrics['requests_429']} "
                f"cache_hits={ee_metrics['cache_hits']} "
                f"cache_misses={ee_metrics['cache_misses']} "
                f"cooldown_skips={ee_metrics['cooldown_skips']} "
                f"event_triggers={ee_metrics['event_triggers']} "
                f"queue_depth={ee_metrics['queue_depth']} "
                f"degraded={ee_metrics['is_degraded']}"
            )

            if self._track_lifetimes:
                avg = sum(self._track_lifetimes) / len(self._track_lifetimes)
                sorted_lts = sorted(self._track_lifetimes)
                n = len(sorted_lts)
                median = sorted_lts[n // 2] if n > 0 else 0.0
                logger.info(
                    f"[AUDIT] TRACK_LIFETIMES count={len(self._track_lifetimes)} "
                    f"avg={avg:.2f}s median={median:.2f}s "
                    f"max={max(self._track_lifetimes):.2f}s "
                    f"min={min(self._track_lifetimes):.2f}s"
                )

    async def process(self, frame_data: FrameData) -> None:
        """
        Override: identical to parent but adds VLM analysis and
        feeds the ZoneSummarizer buffer.
        """
        cam_id = frame_data.camera_id
        frame  = frame_data.frame.copy()

        try:
            await self._process_internal(frame_data, cam_id, frame)
        except Exception as e:
            logger.error(
                f"[{cam_id}] Pipeline error: {e}",
                exc_info=True,
            )
            if cam_id not in self._latest_frames:
                jpeg = self._encode_jpeg(frame)
                self._latest_frames[cam_id] = jpeg

    async def _process_internal(
        self,
        frame_data: FrameData,
        cam_id: str,
        frame: np.ndarray,
    ) -> None:
        """Core processing logic with error isolation per camera."""
        # Lazy pipeline init
        if cam_id not in self._pipelines:
            self._pipelines[cam_id] = _CameraPipeline(
                cam_id, self._detector,
                debug_dir=settings.DEBUG_SAVE_DIR if settings.DEBUG_SAVE_IMAGES else None,
            )
            logger.info(f"[{cam_id}] VLM pipeline initialised")
            self._pipelines[cam_id].crop_mgr.start_cleanup_task()

        pipe = self._pipelines[cam_id]
        pipe.frame_skip_counter += 1
        run_ai  = (pipe.frame_skip_counter % settings.PROCESS_EVERY_N_FRAMES == 0)

        persons:    list[TrackedPerson] = pipe._last_persons
        activities: list[ActivityResult]= pipe._last_activities
        alerts:     list[AlertEvent]    = []
        n_detections = 0
        n_active = 0

        if run_ai:
            try:
                detections = self._detector.detect(frame, cam_id)
                n_detections = len(detections)

                logger.debug(
                    f"[{cam_id}] Detector → Tracker: {n_detections} detections"
                )
                for idx, d in enumerate(detections):
                    logger.debug(
                        f"[{cam_id}]   Det #{idx}: "
                        f"box=({d.x1},{d.y1},{d.x2},{d.y2}) "
                        f"conf={d.confidence:.4f}"
                    )

                persons    = pipe.tracker.update(detections, frame)

                n_active = sum(1 for p in persons if not p.is_lost)
                n_lost   = sum(1 for p in persons if p.is_lost)
                logger.debug(
                    f"[{cam_id}] Tracker → downstream: {len(persons)} tracked "
                    f"(active={n_active}, lost={n_lost}) | "
                    f"ids=[{', '.join(p.person_id for p in persons)}]"
                )

                # ── Track lifecycle audit (keyed by track_uuid) ──────────
                current_uuids = {p.track_uuid for p in persons}
                for p in persons:
                    uid = p.track_uuid
                    if uid not in self._track_creation_times:
                        self._track_creation_times[uid] = time.monotonic()
                        self._audit["persons_detected"] += 1
                        self._person_traces[uid] = [f"DETECTED frame={frame_data.frame_number}"]
                        logger.info(f"[VLM-TRACE] {p.person_id} DETECTED uuid={uid}")
                        _write_trace("DETECTED", person_id=p.person_id, camera_id=cam_id, uuid=uid)
                # Detect deletions by checking previously known UUIDs
                if self._track_creation_times:
                    vanished = set(self._track_creation_times.keys()) - current_uuids
                    for uid in vanished:
                        created = self._track_creation_times.pop(uid, 0)
                        if created:
                            lifetime = time.monotonic() - created
                            self._track_lifetimes.append(lifetime)
                            logger.info(
                                f"[TRACK-LIFETIME] uuid={uid} lived {lifetime:.2f}s before deletion"
                            )

                # ── Crop Generation (for each tracked person) ─────────────
                ts = frame_data.timestamp or datetime.now(timezone.utc).isoformat()
                for p in persons:
                    pipe.crop_mgr.save_crop(
                        frame=        frame,
                        bbox=         p.bbox,
                        track_id=     p.track_id,
                        track_uuid=   p.track_uuid,
                        person_id=    p.person_id,
                        timestamp=    ts,
                        frame_number= frame_data.frame_number,
                    )

                carryable = self._detector.detect_carryable_objects(frame, cam_id)

                activities = await pipe.recognizer.analyze(frame, persons, cam_id, carryable)
                alerts     = pipe.rules.evaluate(activities)
                pipe._last_persons    = persons
                pipe._last_activities = activities

                logger.info(
                    f"[{cam_id}] Frame {frame_data.frame_number:>6d} | AI | "
                    f"detections={n_detections} | "
                    f"active_tracks={n_active} | "
                    f"total_tracks={len(persons)} | "
                    f"alerts={len(alerts)}"
                )

                if settings.DEBUG_SAVE_IMAGES:
                    self._save_debug_frame(cam_id, frame_data.frame_number, "input", frame)
                    raw_viz = frame.copy()
                    for d in detections:
                        cv2.rectangle(raw_viz, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                        cv2.putText(raw_viz, f"{d.confidence:.2f}", (d.x1, d.y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    self._save_debug_frame(cam_id, frame_data.frame_number, "detections", raw_viz)

            except Exception as e:
                logger.error(
                    f"[{cam_id}] AI inference error: {e}",
                    exc_info=True,
                )
                persons    = pipe._last_persons
                activities = pipe._last_activities
                alerts     = []
        else:
            n_active = sum(1 for p in persons if not p.is_lost)
            logger.debug(
                f"[{cam_id}] Frame {frame_data.frame_number:>6d} | SKIP | "
                f"cached_tracks={len(persons)} (active={n_active})"
            )

        # ── Event-driven VLM enrichment (replaces frame-based polling) ────────
        if settings.USE_EVENT_DRIVEN_VLM and persons and activities and settings.USE_VLM:
            vlm_candidates = []
            for person, activity in zip(persons, activities):
                if person.track_uuid in self._vlm_inflight_persons:
                    logger.debug(
                        f"[VLM-TRACE] {person.person_id} INFLIGHT_SKIP uuid={person.track_uuid}"
                    )
                    continue
                should_call, reason = self._event_engine.evaluate(person, activity)
                if should_call:
                    vlm_candidates.append((person, activity, reason))
                    self._vlm_inflight_persons.add(person.track_uuid)
                    self._audit["vlm_tasks_created"] += 1
                    logger.info(
                        f"[VLM-TRACE] {person.person_id} VLM_TRIGGERED uuid={person.track_uuid} "
                        f"reason={reason}"
                    )
                    _write_trace("VLM_TRIGGERED", person_id=person.person_id,
                                 camera_id=cam_id, uuid=person.track_uuid, reason=reason)

            if vlm_candidates:
                asyncio.create_task(
                    self._run_vlm_batch(cam_id, frame.copy(), vlm_candidates)
                )
        elif persons and settings.USE_VLM and not settings.USE_EVENT_DRIVEN_VLM:
            # Legacy fallback: frame-based VLM (original behavior)
            run_vlm = (pipe.frame_skip_counter % settings.VLM_EVERY_N_FRAMES == 0)
            vlm_candidates = []
            for p in persons:
                if p.track_uuid in self._vlm_inflight_persons:
                    continue
                vlm_candidates.append((p, None, "legacy_frame_trigger"))
                self._vlm_inflight_persons.add(p.track_uuid)
                self._audit["vlm_tasks_created"] += 1
            if vlm_candidates and run_vlm:
                asyncio.create_task(
                    self._run_vlm_batch(cam_id, frame.copy(), vlm_candidates)
                )

        # ── Overlay (AI frames only — skip frames reuse last annotated JPEG) ──
        if run_ai:
            frame_shape_before = frame.shape
            n_persons_to_draw = len([p for p in persons if not p.is_lost])
            n_alerts_drawn    = len(alerts)

            try:
                annotated = pipe.overlay.draw(frame, persons, activities, alerts)
                has_annotations = True
            except Exception as e:
                logger.error(
                    f"[{cam_id}] Overlay draw error: {e}",
                    exc_info=True,
                )
                annotated = frame
                has_annotations = False

            frame_shape_after = annotated.shape
            jpeg_size_before = len(self._latest_frames.get(cam_id, b""))
            jpeg_data = self._encode_jpeg(annotated)
            jpeg_size_after = len(jpeg_data)

            self._latest_frames[cam_id] = jpeg_data

            # ── Save transmitted JPEG for forensic comparison ──────────────────
            if settings.DEBUG_SAVE_IMAGES:
                tx_dir = os.path.join(settings.DEBUG_SAVE_DIR, cam_id, "transmitted_jpeg")
                os.makedirs(tx_dir, exist_ok=True)
                with open(os.path.join(tx_dir, f"frame_{frame_data.frame_number:06d}.jpg"), "wb") as f:
                    f.write(jpeg_data)

            # ── Transmission manifest log (single line, machine-parseable) ─────
            logger.info(
                f"[{cam_id}] TRANSMIT | "
                f"frame={frame_data.frame_number} | "
                f"detections={n_detections} | "
                f"active_tracks={n_persons_to_draw} | "
                f"alerts={n_alerts_drawn} | "
                f"frame_in={frame_shape_before[1]}x{frame_shape_before[0]} | "
                f"frame_out={frame_shape_after[1]}x{frame_shape_after[0]} | "
                f"annotated={'yes' if has_annotations else 'FAIL'} | "
                f"jpeg_encoded=yes | "
                f"jpeg_bytes={jpeg_size_after} | "
                f"prev_jpeg_bytes={jpeg_size_before} | "
                f"transmitted=yes"
            )
        else:
            # Skip frame — log that we're re-serving the last annotated JPEG
            cached_jpeg = self._latest_frames.get(cam_id)
            logger.info(
                f"[{cam_id}] TRANSMIT | "
                f"frame={frame_data.frame_number} | SKIP | "
                f"jpeg_cached={'yes' if cached_jpeg is not None else 'NO'}"
            )

        ts_now = frame_data.timestamp or datetime.now(timezone.utc).isoformat()
        self._cam_stats[cam_id] = {
            "camera_id":      cam_id,
            "frame_number":   frame_data.frame_number,
            "timestamp":      ts_now,
            "fps":            frame_data.source_fps,
            "persons":        len(persons),
            "alerts":         self._alert_counts[cam_id],
            "detections":     n_detections,
            "active_tracks":  n_active,
        }

        logger.info(
            f"🔍 TRACE[decision] camera={cam_id} run_ai={run_ai} "
            f"has_persons={len(persons) > 0} has_alerts={len(alerts) > 0} "
            f"will_post={(run_ai and (persons or alerts))}"
        )

        if run_ai and (persons or alerts):
            asyncio.create_task(
                self._post_results_with_vlm(cam_id, persons, activities, alerts, annotated)
            )

        # ── VLM batch analysis ────────────────────────────────────────────────────

    async def _run_vlm_batch(
        self,
        cam_id:  str,
        frame:   np.ndarray,
        candidates: list[tuple[TrackedPerson, Optional[ActivityResult], str]],
    ) -> None:
        """
        Event-driven VLM batch processing.

        Takes candidates from EventEngine (each with a trigger reason),
        acquires throttle slot, queries VLM, and records results.

        Args:
            cam_id:      Camera ID
            frame:       Full frame (for crop extraction)
            candidates:  List of (person, activity, reason) tuples to analyze
        """
        pipe = self._pipelines.get(cam_id)
        if pipe is None:
            return

        batch = candidates[:settings.VLM_MAX_PERSONS_PER_FRAME]
        enqueue_ts = time.monotonic()
        tasks_info: list[tuple[TrackedPerson, Optional[ActivityResult], str, object]] = []

        for person, activity, reason in batch:
            crop_array = pipe.crop_mgr.get_crop_array(person.person_id, track_uuid=person.track_uuid)
            if crop_array is None:
                logger.debug(
                    f"[{cam_id}] No crop available for {person.person_id} (uuid={person.track_uuid}), skipping VLM"
                )
                self._vlm_inflight_persons.discard(person.track_uuid)
                continue

            act_type = activity.activity_type if activity else "unknown"
            metadata = {
                "person_id":     person.person_id,
                "track_uuid":    person.track_uuid,
                "camera_id":     cam_id,
                "zone_id":       person.zone_id,
                "zone_name":     person.zone_name,
                "is_restricted": person.is_restricted,
                "extra_context": (
                    f"Event trigger: {reason}. "
                    f"Rule-based activity: {act_type}. "
                    f"Dwell time: {person.dwell_time:.0f}s."
                ),
            }
            tasks_info.append((
                person, activity, reason,
                self._vlm.analyze_crop(
                    crop_path="", metadata=metadata,
                    crop_array=crop_array, enqueue_ts=enqueue_ts,
                ),
            ))

        if not tasks_info:
            return

        self._audit["groq_requests_started"] += len(tasks_info)
        for p, _, _, _ in tasks_info:
            _write_trace("GROQ_REQUEST", person_id=p.person_id, camera_id=cam_id, uuid=p.track_uuid)

        # Acquire throttle before firing batch (rate limit: 1 req/s global)
        await self._event_engine.acquire_throttle()

        results = await asyncio.gather(
            *[task for _, _, _, task in tasks_info], return_exceptions=True,
        )

        for (person, activity, reason, _), result in zip(tasks_info, results):
            if isinstance(result, Exception):
                is_429 = isinstance(result, RateLimitError)
                if is_429:
                    self._event_engine.handle_429()
                    logger.warning(
                        f"[VLM-DEGRADED] {person.person_id} GROQ_429 uuid={person.track_uuid}"
                    )
                else:
                    self._event_engine.record_failed()
                    logger.info(
                        f"[VLM-TRACE] {person.person_id} GROQ_REQUEST_FAILED error={result}"
                    )
                _write_trace("GROQ_REQUEST_FAILED",
                    person_id=person.person_id, camera_id=cam_id, uuid=person.track_uuid,
                    error=str(result))
                self._audit["groq_requests_failed"] += 1
                self._vlm_inflight_persons.discard(person.track_uuid)
                continue

            self._event_engine.handle_success()
            self._event_engine.record_completed()

            # Record VLM call in EventEngine state
            self._event_engine.record_vlm_call(person, activity, result, reason)

            self._audit["groq_requests_completed"] += 1
            self._audit["vlm_results_parsed"] += 1

            # HARD RULE: skip fallback results
            if result.backend_used == "fallback":
                logger.warning(
                    f"[FALLBACK-TRACE] person_id={person.person_id} "
                    f"camera_id={cam_id} reason=VLM_returned_fallback"
                )
                self._audit["vlm_results_orphaned"] += 1
                self._vlm_inflight_persons.discard(person.track_uuid)
                continue

            overlay_summary = result.description[:120]

            # ── Persist VLM insight ─────────────────────────────────────
            await self._api.post_vlm_insight(
                camera_id=        cam_id,
                zone=             person.zone_id,
                person_id=        person.person_id,
                activity_type=    result.activity_type,
                description=      result.description,
                anomaly_label=    result.anomaly_label,
                confidence=       result.confidence,
                objects_detected= getattr(result, "objects_detected", []),
                backend_used=     result.backend_used,
                latency_ms=       result.latency_ms,
                source=           "vlm",
            )
            self._audit["vlm_insights_posted"] += 1
            logger.info(
                f"[VLM] [{cam_id}] {person.person_id} → "
                f"{'⚠ ANOMALY' if result.is_anomaly else 'normal'} | "
                f"{result.description[:60]}… | {result.latency_ms}ms | "
                f"trigger={reason}"
            )

            self._vlm_inflight_persons.discard(person.track_uuid)

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
        Post rule-based activities to the Activity Log endpoint.
        VLM insights are persisted separately in _run_vlm_batch.
        """
        for activity in activities:
            # Post rule-based (non-VLM) activity to Activity Log
            await self._api.post_activity(
                camera_id=       cam_id,
                zone=            activity.zone_id,
                person_id=       activity.person_id,
                activity_type=   activity.activity_type,
                description=     activity.description,
                anomaly_label=   activity.anomaly_label,
                dwell_seconds=   int(activity.dwell_time),
                confidence=      activity.confidence,
                objects_detected=[],
                backend_used=    activity.backend_used,
                latency_ms=      0,
            )

            # Feed ZoneSummarizer buffer
            self._summarizer.log_activity({
                "person_id":     activity.person_id,
                "zone":          activity.zone_id,
                "activity_type": activity.activity_type,
                "description":   activity.description,
                "anomaly_label": activity.anomaly_label,
                "dwell_time":    activity.dwell_time,
            })

        # Alerts: generate LLM explanation for high-severity alerts
        for alert in alerts:
            self._alert_counts[cam_id] += 1
            snapshot = self._frame_to_b64(frame)

            # Get VLM-enriched description for alert (from EventEngine)
            vlm_desc    = self._event_engine.get_vlm_description(alert.person_id)
            description = vlm_desc or alert.description

            await self._api.post_alert(
                camera_id=   cam_id,
                zone=        alert.zone_id,
                alert_type=  alert.alert_type,
                severity=    alert.severity,
                description= description,
                person_id=   alert.person_id,
                confidence=  alert.confidence,
                snapshot_b64=snapshot,
                source=      "rules_engine",
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
                entry: dict = {
                    "person_id":     p.person_id,
                    "track_uuid":    p.track_uuid,
                    "zone":          p.zone_id,
                    "activity":      act,
                    "dwell_seconds": int(p.dwell_time),
                    "bbox":          list(p.bbox),
                    "center":        list(p.center),
                }
                # Include VLM description in frame_update payload (from EventEngine)
                vlm_data = self._event_engine.get_vlm_data(p.person_id)
                if vlm_data:
                    entry["vlm_description"]   = vlm_data.get("description", "")
                    entry["vlm_anomaly_label"] = vlm_data.get("anomaly_label", "normal")
                    entry["vlm_event"]         = vlm_data.get("event", "")
                person_list.append(entry)
            logger.info(
                f"🔍 TRACE[pre-ws(vlm)] camera={cam_id} | "
                f"persons={len(person_list)} | "
                f"ids=[{', '.join(p['person_id'] for p in person_list)}] | "
                f"bboxes=[{'; '.join(str(p['bbox']) for p in person_list)}]"
            )
            for entry in person_list:
                _write_trace("5_frame_update_payload",
                    person_id=entry["person_id"], camera_id=cam_id,
                    overlay_summary=entry.get("vlm_description", ""),
                    backend_used="",
                    activity=entry.get("activity", "unknown"))
            await self._api.broadcast_frame_update(camera_id=cam_id, persons=person_list)
            for entry in person_list:
                pid = entry["person_id"]
                has_vlm = "vlm_description" in entry
                vlm_desc = entry.get("vlm_description", "")
                logger.info(
                    f"[VLM-TRACE] {pid} WS_SENT "
                    f"has_vlm={has_vlm} "
                    f"overlay_summary=\"{vlm_desc[:80]}\""
                )
                _write_trace("WS_SENT",
                    person_id=pid, camera_id=cam_id,
                    has_vlm=has_vlm,
                    overlay_summary=vlm_desc[:80])

    async def _event_engine_cleanup_loop(self) -> None:
        """Periodically evict stale EventEngine state and cache."""
        while True:
            await asyncio.sleep(30)
            try:
                # Collect active person IDs from all pipelines
                active_ids: set[str] = set()
                for pipe in self._pipelines.values():
                    for p in (pipe._last_persons or []):
                        active_ids.add(p.person_id)
                self._event_engine.evict_stale_persons(active_ids)
                self._event_engine.evict_stale_cache()
            except Exception:
                logger.debug("EventEngine cleanup error", exc_info=True)

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
