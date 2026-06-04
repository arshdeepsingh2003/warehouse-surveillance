"""
streams/stream_manager.py
─────────────────────────
StreamManager — orchestrates all camera FrameReaders.

Responsibilities:
  1. Register cameras (file or RTSP sources)
  2. Start/stop individual camera streams
  3. Run each camera's read loop in a background asyncio task
  4. Maintain per-camera status (online / offline / fps)
  5. Deliver frames to the FrameProcessor via asyncio.Queue
  6. Auto-detect and recover from dropped streams

Architecture note:
  OpenCV's VideoCapture is synchronous (C++ under the hood).
  We run each camera's read() call in a thread pool via
  asyncio.get_event_loop().run_in_executor() so the async event loop
  stays unblocked.

                    ┌─────────────────────────────────────┐
                    │          StreamManager               │
                    │                                      │
  cam-01.mp4 ──→   │  FrameReader-01  ──→  asyncio.Queue │──→ FrameProcessor
  cam-02.mp4 ──→   │  FrameReader-02  ──→  asyncio.Queue │──→ FrameProcessor
  rtsp://... ──→   │  FrameReader-03  ──→  asyncio.Queue │──→ FrameProcessor
                    └─────────────────────────────────────┘
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Awaitable

from streams.frame_reader import FrameReader, FrameData
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Camera registration model ─────────────────────────────────────────────────

@dataclass
class CameraConfig:
    """Everything needed to open a camera stream."""
    camera_id:   str
    name:        str
    location:    str
    zone:        str
    source:      str                      # file path or rtsp:// URL
    source_type: str = "file"             # "file" | "rtsp"
    enabled:     bool = True


@dataclass
class CameraState:
    """Runtime state of one camera (updated by the read loop)."""
    camera_id:     str
    status:        str    = "starting"    # starting | online | offline | stopped
    fps_measured:  float  = 0.0
    frames_total:  int    = 0
    last_frame_at: float  = 0.0
    error_count:   int    = 0
    reader:        Optional[FrameReader] = None

    def to_dict(self) -> dict:
        return {
            "camera_id":    self.camera_id,
            "status":       self.status,
            "fps":          round(self.fps_measured, 1),
            "frames_total": self.frames_total,
            "last_frame_at":self.last_frame_at,
        }


# ── StreamManager ─────────────────────────────────────────────────────────────

class StreamManager:
    """
    Manages all camera streams.

    Usage:
        manager = StreamManager()
        manager.register(CameraConfig("cam-01", ...))
        manager.register(CameraConfig("cam-02", ...))
        await manager.start_all(frame_callback=my_async_callback)
        # ...
        await manager.stop_all()

    The frame_callback receives every FrameData as it arrives:
        async def my_callback(frame_data: FrameData): ...
    """

    def __init__(self) -> None:
        self._configs: Dict[str, CameraConfig] = {}
        self._states:  Dict[str, CameraState]  = {}
        self._tasks:   Dict[str, asyncio.Task]  = {}
        self._executor = ThreadPoolExecutor(
            max_workers=12,
            thread_name_prefix="cam-reader"
        )
        # Each camera gets its own queue. Larger buffer reduces frame drops
        # when AI processing is momentarily slower than the camera frame rate.
        self._queues:  Dict[str, asyncio.Queue] = {}
        self._frame_callback: Optional[Callable[[FrameData], Awaitable[None]]] = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, config: CameraConfig) -> None:
        """Register a camera source. Can be called before or after start_all()."""
        self._configs[config.camera_id] = config
        self._states[config.camera_id]  = CameraState(camera_id=config.camera_id)
        # Larger queue reduces frame drops when AI processing is slow.
        # At 10 FPS, 60 frames = 6 seconds of buffer.
        self._queues[config.camera_id]  = asyncio.Queue(maxsize=60)
        logger.info(f"Registered camera: {config.camera_id} ({config.source_type}) → {config.source}")

    def register_all_from_config(self) -> None:
        """
        Auto-register cameras based on .env settings.
        Scans MOCK_VIDEO_DIR for camera_0N.mp4 files.
        """
        import os, glob
        video_dir = settings.MOCK_VIDEO_DIR
        files = sorted(glob.glob(os.path.join(video_dir, "camera_0*.mp4")))

        ZONE_MAP = {
            "camera_01": ("Main Gate",      "entry_zone"),
            "camera_02": ("Warehouse Aisle","storage_area"),
            "camera_03": ("Loading Zone",   "loading_zone"),
            "camera_04": ("Storage Area",   "storage_area"),
            "camera_05": ("Restricted Area","restricted_area"),
            "camera_06": ("Packing Area",   "packing_area"),
        }

        for i, path in enumerate(files, start=1):
            stem = os.path.splitext(os.path.basename(path))[0]
            name, zone = ZONE_MAP.get(stem, (f"Camera {i:02d}", "general"))
            cam_id = f"cam-{i:02d}"
            size_kb = os.path.getsize(path) / 1024
            logger.info(f"  Mock source: {stem}.mp4 ({size_kb:.0f} KB) → {cam_id}")
            self.register(CameraConfig(
                camera_id=   cam_id,
                name=        name,
                location=    zone.replace("_", " ").title(),
                zone=        zone,
                source=      path,
                source_type= "file",
                enabled=     True,
            ))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start_all(
        self,
        frame_callback: Callable[[FrameData], Awaitable[None]],
    ) -> None:
        """
        Start read loops for all registered cameras, then
        start a dispatcher that calls frame_callback for every frame.
        """
        self._frame_callback = frame_callback
        loop = asyncio.get_event_loop()

        for cam_id, config in self._configs.items():
            if config.enabled:
                task = asyncio.create_task(
                    self._camera_loop(cam_id, loop),
                    name=f"stream-{cam_id}",
                )
                self._tasks[cam_id] = task

        # Single dispatcher task: pulls from all queues and calls callback
        asyncio.create_task(self._dispatcher(), name="frame-dispatcher")
        logger.info(f"StreamManager: started {len(self._tasks)} camera loops.")

    async def stop_all(self) -> None:
        """Stop all camera read loops."""
        for cam_id, task in self._tasks.items():
            task.cancel()
            state = self._states.get(cam_id)
            if state and state.reader:
                state.reader.stop()
            if state:
                state.status = "stopped"
        logger.info("StreamManager: all streams stopped.")

    async def start_camera(self, cam_id: str) -> bool:
        """Start a single camera (used by the cameras API start endpoint)."""
        if cam_id not in self._configs:
            return False
        if cam_id in self._tasks and not self._tasks[cam_id].done():
            return True   # already running
        loop = asyncio.get_event_loop()
        task = asyncio.create_task(self._camera_loop(cam_id, loop), name=f"stream-{cam_id}")
        self._tasks[cam_id] = task
        return True

    async def stop_camera(self, cam_id: str) -> bool:
        """Stop a single camera."""
        task = self._tasks.get(cam_id)
        if task:
            task.cancel()
        state = self._states.get(cam_id)
        if state:
            if state.reader:
                state.reader.stop()
            state.status = "stopped"
        return True

    # ── Status ────────────────────────────────────────────────────────────────

    def get_all_statuses(self) -> list[dict]:
        return [s.to_dict() for s in self._states.values()]

    def get_camera_status(self, cam_id: str) -> Optional[dict]:
        s = self._states.get(cam_id)
        return s.to_dict() if s else None

    # ── Internal: camera read loop ────────────────────────────────────────────

    async def _camera_loop(
        self,
        cam_id: str,
        loop:   asyncio.AbstractEventLoop,
    ) -> None:
        """
        Async wrapper around the synchronous FrameReader.read().

        Runs in the background for each camera. Calls reader.read() in a
        ThreadPoolExecutor so it doesn't block the event loop, then puts
        successful frames onto the camera's asyncio.Queue.
        """
        config = self._configs[cam_id]
        state  = self._states[cam_id]

        # Create and start the reader
        reader = FrameReader(
            camera_id=   cam_id,
            source=      config.source,
            source_type= config.source_type,
            loop=        settings.LOOP_VIDEO,
            target_fps=  settings.STREAM_FPS,
            width=       settings.FRAME_WIDTH,
            height=      settings.FRAME_HEIGHT,
        )

        if not reader.start():
            state.status = "offline"
            logger.error(f"[{cam_id}] Failed to start reader.")
            return

        state.reader = reader
        state.status = "online"
        queue        = self._queues[cam_id]

        # FPS measurement
        fps_window_start  = time.monotonic()
        fps_window_frames = 0

        logger.info(f"[{cam_id}] Read loop started.")

        try:
            while True:
                # Run blocking read() in thread pool
                frame_data = await loop.run_in_executor(
                    self._executor, reader.read
                )

                if frame_data is None:
                    # No frame yet (FPS throttle or brief hiccup) — yield and retry
                    await asyncio.sleep(0.02)
                    continue

                # Update state
                state.frames_total  += 1
                state.last_frame_at  = frame_data.timestamp
                state.error_count    = 0

                # Measure actual FPS over a 2-second window
                fps_window_frames += 1
                window_elapsed = time.monotonic() - fps_window_start
                if window_elapsed >= 2.0:
                    state.fps_measured   = fps_window_frames / window_elapsed
                    fps_window_frames    = 0
                    fps_window_start     = time.monotonic()

                # Blocking put creates natural backpressure: when the AI pipeline
                # can't keep up, the reader pauses instead of dropping frames.
                # This ensures all cameras get fair processing regardless of
                # round-robin position.
                if queue.full():
                    logger.warning(
                        f"[{cam_id}] Queue full ({queue.qsize()}/{queue.maxsize}) — "
                        f"reader will wait for space"
                    )
                await queue.put(frame_data)

        except asyncio.CancelledError:
            logger.info(f"[{cam_id}] Read loop cancelled.")
        except Exception as e:
            logger.error(f"[{cam_id}] Read loop error: {e}", exc_info=True)
            state.status = "offline"
        finally:
            reader.stop()

    # ── Internal: dispatcher ──────────────────────────────────────────────────

    async def _dispatcher(self) -> None:
        """
        Pulls frames from all camera queues in round-robin and calls
        the registered frame_callback with each FrameData.

        Running a single dispatcher (rather than per-camera callbacks)
        keeps the AI pipeline sequential and easy to reason about.
        """
        if not self._frame_callback:
            return

        cam_ids = list(self._queues.keys())
        idx = 0

        while True:
            if not cam_ids:
                await asyncio.sleep(0.1)
                continue

            cam_id = cam_ids[idx % len(cam_ids)]
            idx += 1

            try:
                # Non-blocking get — move on if this queue is empty
                frame_data = self._queues[cam_id].get_nowait()
                await self._frame_callback(frame_data)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)
            except Exception as e:
                logger.error(f"Dispatcher error for {cam_id}: {e}")
