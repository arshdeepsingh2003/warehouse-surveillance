"""
pipeline/frame_processor.py
────────────────────────────
FrameProcessor — the heart of the AI pipeline.

Receives FrameData from the StreamManager and runs analysis on it.

Current mode: MOCK (no real AI)
  • Collects frames into batches
  • Every BATCH_SIZE frames → runs mock "AI analysis"
  • With ALERT_PROBABILITY chance → generates a mock alert
  • Always posts activity events to backend API

Future mode: REAL AI  (swap in steps 4-6 in the roadmap)
  • Step 1: YOLOv8 person detection on each frame
  • Step 2: DeepSORT tracking across frames
  • Step 3: Crop individual persons
  • Step 4: VLM query per person crop (GPT-4V or LLaVA)
  • Step 5: LLM + rules engine → normal vs anomaly
  • Step 6: Post real results to backend

The interface stays identical — only the internals change.

┌─────────────────────────────────────────────────────────────────┐
│                      FrameProcessor                             │
│                                                                 │
│  FrameData in                                                   │
│      │                                                          │
│      ▼                                                          │
│  ┌───────────────┐     ┌──────────────────┐                    │
│  │ Frame buffer  │────▶│  AI Analysis     │                    │
│  │ (batch)       │     │  (mock / real)   │                    │
│  └───────────────┘     └────────┬─────────┘                    │
│                                 │                               │
│                    ┌────────────┼───────────────┐              │
│                    ▼            ▼               ▼              │
│              Post activity  Raise alert    WS broadcast        │
│              to backend     to backend     (instant)           │
└─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import base64
import logging
import random
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from streams.frame_reader import FrameData
from pipeline.api_client import APIClient
from config.settings import settings
from ai.overlay.frame_overlay import FrameOverlay
from ai.tracker.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)


# ── Mock AI data ──────────────────────────────────────────────────────────────

_NORMAL_ACTIVITIES = [
    ("walking",         "Person walking through the zone at normal pace."),
    ("handling_items",  "Worker handling inventory boxes at shelf rack."),
    ("standing",        "Person standing near workstation, reviewing clipboard."),
    ("carrying_object", "Worker carrying a cardboard box towards storage area."),
    ("walking",         "Person moving between aisle sections carrying a scanner."),
]

_ANOMALY_ACTIVITIES = [
    ("unauthorized_entry",   "unauthorized_access",   "high",   "Person entered restricted area without badge scan."),
    ("loitering",            "loitering",             "medium", "Individual standing idle near loading dock for extended period."),
    ("falling",              "worker_fall",           "high",   "Worker appears to have fallen near shelf rack — stationary for 30+ seconds."),
    ("walking",              "ppe_violation",         "low",    "Worker detected without safety helmet in mandatory PPE zone."),
    ("unauthorized_entry",   "suspicious_activity",   "medium", "Person moving suspiciously near high-value storage rack."),
]

_PERSON_IDS = [f"P-{i}" for i in range(1001, 1060)]

_ZONE_MAP = {
    "cam-01": "entry_zone",
    "cam-02": "storage_area",
    "cam-03": "loading_zone",
    "cam-04": "storage_area",
    "cam-05": "restricted_area",
    "cam-06": "packing_area",
}


# ── Processor ─────────────────────────────────────────────────────────────────

class FrameProcessor:
    """
    Processes frames from all cameras.

    One shared instance handles all cameras.
    Per-camera frame buffers collect frames until BATCH_SIZE is reached,
    then "analysis" runs and results are posted to the backend.
    """

    def __init__(self, api_client: APIClient) -> None:
        self._api    = api_client
        self._buffers: dict[str, list[FrameData]] = defaultdict(list)

        # Per-camera stats for the stream server
        self._cam_stats: dict[str, dict] = {}

        # Latest JPEG frame per camera (for MJPEG streaming)
        self._latest_frames: dict[str, bytes] = {}

    # ── Public: receive a frame ───────────────────────────────────────────────

    async def process(self, frame_data: FrameData) -> None:
        """
        Entry point called by StreamManager for every frame.

        1. Encode frame to JPEG for MJPEG streaming.
        2. Add to batch buffer.
        3. When batch full → run mock AI analysis.
        """
        cam_id = frame_data.camera_id

        # ── Always: encode and store latest JPEG for streaming ────────────────
        jpeg = self._encode_frame(frame_data.frame)
        self._latest_frames[cam_id] = jpeg

        # ── Update per-camera stats ────────────────────────────────────────────
        self._cam_stats[cam_id] = {
            "camera_id":    cam_id,
            "frame_number": frame_data.frame_number,
            "timestamp":    frame_data.timestamp,
            "fps":          frame_data.source_fps,
        }

        # ── Batch frames ──────────────────────────────────────────────────────
        self._buffers[cam_id].append(frame_data)

        if len(self._buffers[cam_id]) >= settings.BATCH_SIZE:
            batch = self._buffers[cam_id].copy()
            self._buffers[cam_id].clear()
            # Run analysis in background (don't block the dispatcher)
            asyncio.create_task(self._analyse_batch(cam_id, batch))

    def get_latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        """Return the most recently encoded JPEG frame for a camera."""
        return self._latest_frames.get(cam_id)

    def get_all_stats(self) -> dict:
        return self._cam_stats

    # ── Analysis ──────────────────────────────────────────────────────────────

    async def _analyse_batch(self, cam_id: str, batch: list[FrameData]) -> None:
        """
        Mock AI analysis on a batch of frames.

        In production this is where you:
          1. Run YOLOv8 on key frames to get bounding boxes
          2. Run DeepSORT to assign person IDs
          3. Crop each person and query VLM
          4. Pass VLM output to LLM + rules engine
          5. Post results

        For now we generate plausible fake results.
        """
        logger.debug(f"[{cam_id}] Analysing batch of {len(batch)} frames.")

        zone      = _ZONE_MAP.get(cam_id, "general_zone")
        person_id = random.choice(_PERSON_IDS)

        # Decide normal or anomaly
        is_anomaly = random.random() < settings.ALERT_PROBABILITY

        if is_anomaly:
            act_type, alert_type, severity, description = random.choice(_ANOMALY_ACTIVITIES)
            anomaly_label = "anomaly"
        else:
            act_type, description = random.choice(_NORMAL_ACTIVITIES)
            alert_type  = None
            severity    = None
            anomaly_label = "normal"

        dwell_seconds = random.randint(5, 300)
        confidence    = round(random.uniform(0.72, 0.98), 2)

        # ── 1. Post activity log ───────────────────────────────────────────────
        await self._api.post_activity(
            camera_id=     cam_id,
            zone=          zone,
            person_id=     person_id,
            activity_type= act_type,
            description=   description,
            anomaly_label= anomaly_label,
            dwell_seconds= dwell_seconds,
            confidence=    confidence,
        )

        # ── 2. Post alert (anomalies only) ────────────────────────────────────
        if is_anomaly and alert_type:
            # Use the middle frame of the batch as the "snapshot"
            mid_frame = batch[len(batch) // 2].frame
            snapshot  = self._frame_to_base64(mid_frame)

            await self._api.post_alert(
                camera_id=   cam_id,
                zone=        zone,
                alert_type=  alert_type,
                severity=    severity,
                description= description,
                person_id=   person_id,
                confidence=  confidence,
                snapshot_b64=snapshot,
            )

        # ── 3/4. Annotate a representative frame for MJPEG streaming and
        #       broadcast an enriched frame_update (includes bbox/center)
        #       so the frontend overlay can render tracked positions.
        try:
            mid = len(batch) // 2
            frame = batch[mid].frame.copy()
            h, w = frame.shape[:2]

            # Generate a plausible mock bbox in the centre of the frame
            bw, bh = int(w * 0.18), int(h * 0.35)
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w - 1, x1 + bw)
            y2 = min(h - 1, y1 + bh)

            # Build a TrackedPerson for the overlay
            mock_person = TrackedPerson(
                track_id=1,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                zone_id=zone,
                zone_name=zone.replace("_", " ").title(),
                is_restricted=(zone == "restricted_area"),
                age=5,
                dwell_time=dwell_seconds,
                is_lost=False,
                velocity=(0.0, 0.0),
            )

            overlay = FrameOverlay(camera_id=cam_id)
            annotated = overlay.draw(frame, [mock_person], [], [])
            # Encode annotated frame and replace latest JPEG for this camera
            self._latest_frames[cam_id] = self._encode_frame(annotated)
            # Broadcast enriched frame update (includes bbox center)
            await self._api.broadcast_frame_update(
                camera_id=cam_id,
                persons=[{
                    "person_id":     person_id,
                    "zone":          zone,
                    "activity":      act_type,
                    "dwell_seconds": dwell_seconds,
                    "bbox":          [x1, y1, x2, y2],
                    "center":        [cx, cy],
                }],
            )
        except Exception:
            # Don't let overlay failures break analysis
            logger.exception("Failed to annotate frame for camera %s", cam_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode frame to JPEG bytes for MJPEG streaming."""
        # Add timestamp overlay to the frame
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, ts,
            (8, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (0, 229, 255), 1, cv2.LINE_AA
        )
        _, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
        )
        return buf.tobytes()

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert a frame to a base64-encoded JPEG string."""
        jpeg = self._encode_frame(frame.copy())
        return base64.b64encode(jpeg).decode("utf-8")


# ────────────────────────────────────────────────────────────────────────────
# NOTE: When integrating real AI (future step), replace _analyse_batch with:
#
#   async def _analyse_batch_real(self, cam_id, batch):
#       key_frames = self._sample_key_frames(batch)         # every Nth frame
#       for frame_data in key_frames:
#           boxes = yolo_model(frame_data.frame)             # YOLOv8
#           tracks = deepsort.update(boxes, frame_data.frame)# DeepSORT
#           for track in tracks:
#               crop = frame_data.frame[y1:y2, x1:x2]
#               description = await vlm_query(crop)          # VLM
#               label, severity = rules_engine(description, zone, dwell)
#               if label == "anomaly":
#                   await self._api.post_alert(...)
#               await self._api.post_activity(...)
# ────────────────────────────────────────────────────────────────────────────
