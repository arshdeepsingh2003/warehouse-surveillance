"""
streams/frame_reader.py
───────────────────────
OpenCV-based video frame reader.

Supports two source types:
  • FILE  — read from an .mp4 / .avi / .mkv file (loops if LOOP_VIDEO=true)
  • RTSP  — read from a live RTSP camera stream

This class is intentionally synchronous (OpenCV is not async).
The StreamManager runs each reader in a separate thread via asyncio.run_in_executor.

Lifecycle:
  reader = FrameReader("cam-01", "mock_sources/camera_01.mp4", source_type="file")
  reader.start()                  # opens capture
  frame_data = reader.read()      # returns FrameData or None
  reader.stop()                   # releases capture

The FrameData dict passed downstream contains everything the AI pipeline needs:
  {
    camera_id, frame (np.ndarray), timestamp, width, height,
    frame_number, source_fps
  }
"""

import cv2
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    """
    One extracted video frame plus all metadata needed downstream.
    Passed from FrameReader → FrameProcessor → AI pipeline.
    """
    camera_id:    str
    frame:        np.ndarray          # BGR image array (H × W × 3)
    timestamp:    float               # Unix timestamp of capture
    width:        int
    height:       int
    frame_number: int                 # Monotonically increasing frame counter
    source_fps:   float               # FPS reported by the source
    source_type:  str                 # "file" | "rtsp"

    def to_dict(self) -> dict:
        """Return metadata dict (without the heavy frame array) for API calls."""
        return {
            "camera_id":    self.camera_id,
            "timestamp":    self.timestamp,
            "width":        self.width,
            "height":       self.height,
            "frame_number": self.frame_number,
            "source_fps":   self.source_fps,
            "source_type":  self.source_type,
        }


# ── Reader ────────────────────────────────────────────────────────────────────

class FrameReader:
    """
    Wraps OpenCV VideoCapture for one camera source.

    Thread-safe: uses a lock so the StreamManager thread and frame
    processing thread can both safely call read().
    """

    def __init__(
        self,
        camera_id:   str,
        source:      str,                              # path or rtsp:// URL
        source_type: Literal["file", "rtsp"] = "file",
        loop:        bool = True,                      # only for file sources
        target_fps:  int  = 10,                        # max FPS to pull from source
        width:       int  = 640,
        height:      int  = 360,
    ):
        self.camera_id   = camera_id
        self.source      = source
        self.source_type = source_type
        self.loop        = loop
        self.target_fps  = target_fps
        self.width       = width
        self.height      = height

        self._cap:        Optional[cv2.VideoCapture] = None
        self._lock        = threading.Lock()
        self._running     = False
        self._frame_count = 0
        self._source_fps  = float(target_fps)

        # Throttle: only read a frame every (1 / target_fps) seconds
        self._frame_interval = 1.0 / max(target_fps, 1)
        self._last_read_time = 0.0
        self._first_read = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open the video source.
        Returns True on success, False if the source cannot be opened.
        """
        with self._lock:
            cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                logger.error(f"[{self.camera_id}] Cannot open source: {self.source}")
                return False

            # For RTSP: set buffer size small to reduce latency
            if self.source_type == "rtsp":
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            self._source_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.target_fps)
            self._cap     = cap
            self._running = True
            logger.info(
                f"[{self.camera_id}] Opened {self.source_type} source "
                f"(source FPS={self._source_fps:.1f}, target={self.target_fps})"
            )
            return True

    def stop(self) -> None:
        """Release the VideoCapture and free memory."""
        with self._lock:
            self._running = False
            if self._cap:
                self._cap.release()
                self._cap = None
        logger.info(f"[{self.camera_id}] Reader stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Reading ───────────────────────────────────────────────────────────────

    def read(self) -> Optional[FrameData]:
        """
        Read the next frame from the source.

        Handles:
          • FPS throttling  — skips reads faster than target_fps
          • Frame skipping  — for file sources, skips frames to match target FPS
          • Auto-loop       — rewinds file sources when they end
          • RTSP reconnect  — will return None on failure (caller should restart)

        Returns:
          FrameData if a frame was successfully read, None otherwise.
        """
        # FPS throttle: don't read faster than target_fps
        now = time.monotonic()
        elapsed = now - self._last_read_time
        if elapsed < self._frame_interval:
            return None

        with self._lock:
            if not self._cap or not self._running:
                return None

            # ── Frame skipping for file sources ───────────────────────────────
            # If source FPS >> target FPS, skip frames to avoid processing
            # every frame. Example: source=30fps, target=10fps → skip 2 of 3.
            if self.source_type == "file" and self._source_fps > self.target_fps:
                skip = int(self._source_fps / self.target_fps) - 1
                for _ in range(skip):
                    self._cap.grab()   # grab() decodes to buffer but doesn't decode pixel data

            ret, frame = self._cap.read()
            if self._first_read:
                logger.info(f"[{self.camera_id}] First cap.read() → ret={ret}, shape={frame.shape if ret else 'N/A'}")
                self._first_read = False
            logger.debug(f"[{self.camera_id}] cap.read() → ret={ret}, frame={'None' if not ret else f'shape={frame.shape}'}")

            # ── End-of-file handling ───────────────────────────────────────────
            if not ret:
                if self.source_type == "file" and self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
                    ret, frame = self._cap.read()
                    if not ret:
                        logger.warning(f"[{self.camera_id}] Cannot rewind — source empty?")
                        return None
                else:
                    logger.warning(f"[{self.camera_id}] Frame read failed — source ended or stream dropped.")
                    return None

            # ── Resize ────────────────────────────────────────────────────────
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            self._frame_count  += 1
            self._last_read_time = now

            return FrameData(
                camera_id=    self.camera_id,
                frame=        frame,
                timestamp=    time.time(),
                width=        self.width,
                height=       self.height,
                frame_number= self._frame_count,
                source_fps=   self._source_fps,
                source_type=  self.source_type,
            )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def encode_jpeg(self, frame: np.ndarray, quality: int = 75) -> bytes:
        """Encode an ndarray frame to JPEG bytes for MJPEG streaming."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    @property
    def frame_count(self) -> int:
        return self._frame_count
