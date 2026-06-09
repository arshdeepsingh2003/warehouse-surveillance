"""
ai/crop/crop_manager.py
────────────────────────
Crop Manager — extracts, persists, and manages person crop images
for downstream VLM analysis.

Directory layout:
    {CROP_DIR}/
        {camera_id}/
            {person_id}_{timestamp_safe}.jpg

Each crop is a tight person crop (with configurable padding) saved as
JPEG.  A CropRecord is kept in memory for every saved crop so the VLM
pipeline can look up the crop path by (person_id, camera_id) without
scanning the filesystem.

Retention:
    A background task removes crops older than CROP_RETENTION_DAYS.
    The cleanup runs every CROP_CLEANUP_INTERVAL_SECONDS.

Integration:
    Frame → Detection → Tracking → CropManager.save_crop() → Activity → VLM
                                                                          ↓
                                                             CropManager path
                                                             → VLMClient.analyze_crop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CropRecord:
    """Metadata for one saved person crop."""
    track_id:     int
    person_id:    str
    camera_id:    str
    timestamp:    str
    frame_number: int
    bbox:         tuple[int, int, int, int]
    crop_path:    str
    created_at:   float = field(default_factory=time.monotonic)


class CropManager:
    """
    Manages extraction, storage, and lifecycle of person crop images.

    One shared instance is created per camera pipeline.

    Usage:
        manager = CropManager(camera_id="cam-01")
        record = manager.save_crop(frame, bbox, track_id, person_id, ts, fn)
        path   = manager.get_crop_path(person_id)
    """

    def __init__(
        self,
        camera_id:     str,
        crop_dir:      Optional[str] = None,
        retention_days: Optional[int] = None,
        padding:       Optional[int] = None,
        quality:       Optional[int] = None,
    ) -> None:
        self._camera_id      = camera_id
        self._crop_dir       = crop_dir      or settings.CROP_DIR
        self._retention_days = retention_days or settings.CROP_RETENTION_DAYS
        self._padding        = padding       or settings.CROP_PADDING
        self._quality        = quality       or settings.CROP_QUALITY

        self._cam_dir = os.path.join(self._crop_dir, camera_id)
        os.makedirs(self._cam_dir, exist_ok=True)

        # In-memory metadata index: person_id → CropRecord (latest per person)
        self._latest: dict[str, CropRecord] = {}
        # Ordered history for cleanup & retrieval (FIFO, bounded at 5000)
        self._history: deque[CropRecord] = deque(maxlen=5000)

        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            f"[{camera_id}] CropManager ready → {self._cam_dir} "
            f"(retention={self._retention_days}d, padding={self._padding}px)"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_crop(
        self,
        frame:        np.ndarray,
        bbox:         tuple[int, int, int, int],
        track_id:     int,
        person_id:    str,
        timestamp:    str,
        frame_number: int,
    ) -> CropRecord:
        """
        Extract the person region from *frame* at *bbox*, save as JPEG, and
        return a CropRecord with the full path.

        The crop is saved to:
            {crop_dir}/{camera_id}/{person_id}_{safe_ts}.jpg
        """
        crop = self._extract_crop(frame, bbox)

        safe_ts = _safe_timestamp(timestamp)
        filename = f"{person_id}_{safe_ts}.jpg"
        filepath = os.path.join(self._cam_dir, filename)

        cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, self._quality])

        record = CropRecord(
            track_id=     track_id,
            person_id=    person_id,
            camera_id=    self._camera_id,
            timestamp=    timestamp,
            frame_number= frame_number,
            bbox=         bbox,
            crop_path=    filepath,
        )

        self._latest[person_id] = record
        self._history.append(record)

        logger.debug(
            f"[{self._camera_id}] Crop saved: {person_id} → {filepath} "
            f"({bbox})"
        )

        return record

    def get_crop_path(self, person_id: str) -> Optional[str]:
        """Return the crop path for the most recent crop of *person_id*."""
        record = self._latest.get(person_id)
        return record.crop_path if record else None

    def get_record(self, person_id: str) -> Optional[CropRecord]:
        """Return the most recent CropRecord for *person_id*."""
        return self._latest.get(person_id)

    def get_history(
        self,
        camera_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[CropRecord]:
        """Return the most recent *limit* CropRecords, optionally filtered."""
        if camera_id and camera_id != self._camera_id:
            return []
        result = list(self._history)
        return result[-limit:]

    # ── Crop extraction ────────────────────────────────────────────────────────

    def _extract_crop(
        self,
        frame: np.ndarray,
        bbox:  tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Extract a padded person crop from the frame.

        Returns the crop region or the full frame if the crop is empty.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - self._padding)
        y1 = max(0, y1 - self._padding)
        x2 = min(w, x2 + self._padding)
        y2 = min(h, y2 + self._padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning(
                f"[{self._camera_id}] Empty crop for bbox={bbox}, "
                f"returning full frame"
            )
            return frame
        return crop

    # ── Retention ──────────────────────────────────────────────────────────────

    def start_cleanup_task(self) -> None:
        """Launch the periodic cleanup background task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                f"[{self._camera_id}] Crop cleanup task started "
                f"(interval={settings.CROP_CLEANUP_INTERVAL_SECONDS}s)"
            )

    def stop_cleanup_task(self) -> None:
        """Cancel the cleanup background task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodically remove crops older than retention period."""
        while True:
            try:
                await asyncio.sleep(settings.CROP_CLEANUP_INTERVAL_SECONDS)
                removed = self._remove_old_crops()
                if removed:
                    logger.info(
                        f"[{self._camera_id}] Crop cleanup: removed {removed} file(s)"
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(f"[{self._camera_id}] Crop cleanup error")

    def _remove_old_crops(self) -> int:
        """
        Scan the camera's crop directory and delete files older than
        *retention_days* (using file mtime).

        Returns the number of files removed.
        """
        if not os.path.isdir(self._cam_dir):
            return 0

        cutoff = time.time() - (self._retention_days * 86400)
        removed = 0

        for fname in os.listdir(self._cam_dir):
            fpath = os.path.join(self._cam_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                mtime = os.path.getmtime(fpath)
                if mtime < cutoff:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                continue

        # Prune in-memory history for removed records
        self._history = deque(
            (r for r in self._history if os.path.isfile(r.crop_path)),
            maxlen=5000,
        )
        # Remove stale entries from latest index
        stale_ids = [
            pid for pid, rec in self._latest.items()
            if not os.path.isfile(rec.crop_path)
        ]
        for pid in stale_ids:
            self._latest.pop(pid, None)

        return removed


def _safe_timestamp(ts: str) -> str:
    """
    Convert an ISO-8601 timestamp string to a filesystem-safe form.

    "2025-06-08T14:30:25.123456+00:00" → "20250608_143025_123456"
    Falls back to current UTC if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        dt = datetime.now(timezone.utc)

    return dt.strftime("%Y%m%d_%H%M%S_%f")[:22]  # YYYYMMDD_HHMMSS_mmmmmm
