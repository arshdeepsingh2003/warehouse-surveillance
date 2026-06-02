"""
ai/tracker/person_tracker.py
─────────────────────────────
Person Tracker — assigns stable IDs to detected persons across frames.

Architecture: IoU-based Kalman tracker (ByteTrack-lite)
─────────────────────────────────────────────────────────
We implement a lightweight ByteTrack-inspired tracker that:
  1. Predicts each track's position using a Kalman filter
  2. Matches new detections to existing tracks using IoU
  3. Assigns a new ID to unmatched detections
  4. Removes tracks that haven't been matched for N frames

Why not use the full DeepSORT/ByteTrack library?
  • Full libraries add 200+ MB of dependencies
  • This implementation is 200 lines and covers 90% of warehouse use cases
  • Easy to read, debug, and modify
  • Drop-in replacement: same interface, just swap the class

When to upgrade to full ByteTrack:
  • You have crowds of 20+ people
  • People cross each other frequently
  • You need re-identification across camera cuts

Track lifecycle:
  NEW         → first seen, needs N confirmations
  CONFIRMED   → reliably tracked, published to downstream
  LOST        → not matched for a few frames (still in memory)
  DELETED     → missing too long → removed

Ghost-bounding-box prevention:
  • Tracks whose predicted bbox drifts mostly outside the frame are
    deleted immediately instead of lingering for MAX_MISSES frames.
  • Missed tracks pressed against the frame edge are pruned immediately
    (touches_edge check) — the bbox property clamps coordinates so
    a drifted-out box appears stuck at the border.
  • Velocity is damped (×0.7) on each miss to prevent runaway predictions.
  • IoU scores for missed tracks are penalized (×0.85) to prevent a ghost
    from latching onto a different person's detection.
  • Confidence decays (×0.85) on each missed frame.
  • MAX_MISSES = 3 — at 10 FPS this is 0.3 s of persistence.

Output per track:
  TrackedPerson = {
    track_id:   int             unique stable ID (P-1001, P-1002, ...)
    bbox:       (x1,y1,x2,y2)  current bounding box
    confidence: float
    zone_id:    str             which zone the person is in
    age:        int             frames this track has been active
    dwell_time: float           seconds in current zone
    is_lost:    bool
  }
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ai.detector.person_detector import Detection
from ai.zones.zone_config import get_zone_for_point, Zone

logger = logging.getLogger(__name__)


# ── Kalman filter constants ───────────────────────────────────────────────────
# State: [cx, cy, w, h, vcx, vcy, vw, vh]  (center_x, center_y, width, height + velocities)
_DT = 1.0  # one frame time step

# ── TrackedPerson dataclass ───────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """
    One tracked person — the output of the tracker per frame.
    Everything the rules engine and activity analyzer needs is here.
    """
    track_id:    int
    bbox:        tuple[int, int, int, int]   # x1,y1,x2,y2
    confidence:  float
    zone_id:     str
    zone_name:   str
    is_restricted: bool
    age:         int     = 0     # frames since first seen
    dwell_time:  float   = 0.0   # seconds in current zone
    is_lost:     bool    = False
    velocity:    tuple[float, float] = (0.0, 0.0)   # vx, vy pixels/frame

    @property
    def person_id(self) -> str:
        """Dashboard-friendly string ID."""
        return f"P-{self.track_id + 1000}"

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def feet(self) -> tuple[int, int]:
        """Bottom-center of bbox — used for zone testing."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)

    def to_dict(self) -> dict:
        return {
            "track_id":    self.track_id,
            "person_id":   self.person_id,
            "bbox":        list(self.bbox),
            "confidence":  round(self.confidence, 3),
            "zone_id":     self.zone_id,
            "zone_name":   self.zone_name,
            "is_restricted": self.is_restricted,
            "dwell_time":  round(self.dwell_time, 1),
            "age":         self.age,
            "center":      list(self.center),
        }


# ── Internal Track ────────────────────────────────────────────────────────────

class _Track:
    """Internal state for one tracked person (not exposed outside this module)."""

    _next_id = 1

    def __init__(self, detection: Detection, zone: Optional[Zone]) -> None:
        self.track_id      = _Track._next_id
        _Track._next_id   += 1

        x1, y1, x2, y2    = detection.bbox
        self.cx            = float((x1 + x2) / 2)
        self.cy            = float((y1 + y2) / 2)
        self.w             = float(x2 - x1)
        self.h             = float(y2 - y1)
        self.vx            = 0.0
        self.vy            = 0.0

        self.confidence    = detection.confidence
        self.age           = 1
        self.hits          = 1
        self.misses        = 0
        self.confirmed     = False    # needs MIN_HITS before publishing

        # Zone tracking
        self.zone:          Optional[Zone] = zone
        self.zone_entry_time: float = time.time()
        self.prev_zone_id:  Optional[str] = None

    # ── Kalman-style prediction ───────────────────────────────────────────────

    def predict(self) -> None:
        """Move the predicted position forward by one frame using velocity."""
        self.cx += self.vx
        self.cy += self.vy

    def update(self, detection: Detection, zone: Optional[Zone]) -> None:
        """Fuse new detection into this track."""
        x1, y1, x2, y2 = detection.bbox
        new_cx = float((x1 + x2) / 2)
        new_cy = float((y1 + y2) / 2)

        # Smooth velocity (exponential moving average)
        alpha = 0.6
        self.vx = alpha * (new_cx - self.cx) + (1 - alpha) * self.vx
        self.vy = alpha * (new_cy - self.cy) + (1 - alpha) * self.vy

        self.cx = new_cx
        self.cy = new_cy
        self.w  = float(x2 - x1)
        self.h  = float(y2 - y1)

        self.confidence = detection.confidence
        self.age       += 1
        self.hits      += 1
        self.misses     = 0

        if self.hits >= 3:
            self.confirmed = True

        # Update zone
        if zone is not None:
            if self.zone is None or zone.zone_id != self.zone.zone_id:
                self.prev_zone_id   = self.zone.zone_id if self.zone else None
                self.zone           = zone
                self.zone_entry_time= time.time()
        elif zone is None and self.zone is not None:
            pass  # keep last known zone

    def mark_missed(self) -> None:
        self.misses += 1
        # Confidence decay — each miss reduces reliability
        self.confidence *= 0.85
        # Dampen velocity to prevent runaway predictions into ghost boxes
        self.vx *= 0.7
        self.vy *= 0.7
        self.predict()  # move bbox forward by damped velocity

    def touches_edge(self, frame_w: int, frame_h: int, margin: int = 3) -> bool:
        """True if the clamped bbox is pressed against any frame edge."""
        x1, y1, x2, y2 = self.bbox
        return (x1 <= margin or x2 >= frame_w - margin or
                y1 <= margin or y2 >= frame_h - margin)

    def visible_ratio(self, frame_w: int, frame_h: int) -> float:
        """Fraction of the bbox area that falls inside the visible frame."""
        x1, y1, x2, y2 = self.bbox_unclamped
        total_area = max((x2 - x1) * (y2 - y1), 1)
        clipped_x1 = max(0, x1)
        clipped_y1 = max(0, y1)
        clipped_x2 = min(frame_w, x2)
        clipped_y2 = min(frame_h, y2)
        visible_w = max(0, clipped_x2 - clipped_x1)
        visible_h = max(0, clipped_y2 - clipped_y1)
        visible_area = visible_w * visible_h
        return visible_area / total_area

    @property
    def dwell_time(self) -> float:
        return time.time() - self.zone_entry_time

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        half_w = self.w / 2
        half_h = self.h / 2
        return (
            max(0, int(self.cx - half_w)),
            max(0, int(self.cy - half_h)),
            int(self.cx + half_w),
            int(self.cy + half_h),
        )

    @property
    def bbox_unclamped(self) -> tuple[int, int, int, int]:
        """Bbox without clamping — used for out-of-frame checks."""
        half_w = self.w / 2
        half_h = self.h / 2
        return (
            int(self.cx - half_w),
            int(self.cy - half_h),
            int(self.cx + half_w),
            int(self.cy + half_h),
        )


# ── IoU matching utilities ────────────────────────────────────────────────────

def _iou(bbox1: tuple, bbox2: tuple) -> float:
    x1a, y1a, x2a, y2a = bbox1
    x1b, y1b, x2b, y2b = bbox2
    ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (x2a - x1a) * (y2a - y1a)
    a2 = (x2b - x1b) * (y2b - y1b)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _hungarian_match(
    tracks: list[_Track],
    detections: list[Detection],
    iou_threshold: float = 0.25,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy IoU matching (approximation of Hungarian algorithm).

    Returns:
      matches:       [(track_idx, det_idx), ...]
      unmatched_tracks:  [track_idx, ...]
      unmatched_dets:    [det_idx, ...]
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    # Build IoU cost matrix
    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            iou = _iou(track.bbox, det.bbox)
            # Penalize missed tracks — harder to re-associate, prevents
            # a ghost at the frame edge from latching onto a valid detection.
            if track.misses > 0:
                iou *= 0.85
            iou_matrix[ti, di] = iou

    # Greedy: sort all pairs by IoU descending, assign each to at most one partner
    pairs = sorted(
        [(iou_matrix[ti, di], ti, di)
         for ti in range(len(tracks))
         for di in range(len(detections))],
        reverse=True,
    )

    matched_tracks = set()
    matched_dets   = set()
    matches        = []

    for score, ti, di in pairs:
        if score < iou_threshold:
            break
        if ti in matched_tracks or di in matched_dets:
            continue
        matches.append((ti, di))
        matched_tracks.add(ti)
        matched_dets.add(di)

    unmatched_tracks = [i for i in range(len(tracks))  if i not in matched_tracks]
    unmatched_dets   = [i for i in range(len(detections)) if i not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


# ── Public Tracker ────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Multi-object tracker for warehouse person tracking.

    Per-camera: create one PersonTracker instance per camera.

    Usage:
        tracker = PersonTracker(camera_id="cam-01")
        tracked = tracker.update(detections, frame)
        for person in tracked:
            print(person.person_id, person.zone_id, person.dwell_time)
    """

    # Tunable constants
    IOU_THRESHOLD            = 0.35   # minimum IoU to match detection to track
    MAX_MISSES               = 3      # frames without detection before deleting track
    MIN_HITS                 = 3      # detections before track is published
    FRAME_VISIBILITY_THRESHOLD = 0.35 # minimum fraction of bbox that must be
                                      # visible inside the frame — below this
                                      # the track is deleted immediately
    EDGE_MARGIN              = 3      # pixels from frame edge to consider "touching edge"

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._tracks:  list[_Track] = []
        self._frame_n: int = 0
        logger.info(f"[{camera_id}] PersonTracker initialised")

    def update(
        self,
        detections: list[Detection],
        frame:      np.ndarray,
    ) -> list[TrackedPerson]:
        """
        Process one frame worth of detections.

        Steps:
          1. Predict next positions for all existing tracks
          2. Prune tracks whose bbox has drifted mostly out of frame
          3. Match detections to tracks via IoU
          4. Update matched tracks
          5. Create new tracks for unmatched detections
          6. Delete tracks missing too long
          7. Return confirmed tracks as TrackedPerson objects

        Args:
            detections: Output of PersonDetector.detect()
            frame:      Current BGR frame (used for zone lookup + boundary check)

        Returns:
            List of confirmed TrackedPerson objects (published to rules engine)
        """
        self._frame_n += 1
        frame_h, frame_w = frame.shape[:2]

        # ── 1. Predict ────────────────────────────────────────────────────────
        for track in self._tracks:
            track.predict()

        # ── 2. Prune out-of-frame and edge-ghost tracks ────────────────────
        # If a predicted bbox has drifted mostly outside the visible frame,
        # delete it immediately instead of letting it ghost for MAX_MISSES frames.
        # Also prune missed tracks pressed against the frame edge — the bbox
        # property clamps coordinates so a drifted-out box appears stuck at edge.
        self._tracks = [
            t for t in self._tracks
            if t.visible_ratio(frame_w, frame_h) >= self.FRAME_VISIBILITY_THRESHOLD
            and not (t.misses > 0 and t.touches_edge(frame_w, frame_h, self.EDGE_MARGIN))
        ]

        # ── 3. Match ──────────────────────────────────────────────────────────
        matches, unmatched_t, unmatched_d = _hungarian_match(
            self._tracks, detections, self.IOU_THRESHOLD
        )

        # ── 4. Update matched tracks ──────────────────────────────────────────
        for ti, di in matches:
            det  = detections[di]
            zone = self._get_zone(det)
            self._tracks[ti].update(det, zone)

        # ── 5. Mark unmatched tracks as missed ────────────────────────────────
        for ti in unmatched_t:
            self._tracks[ti].mark_missed()

        # ── 6. Create new tracks for unmatched detections ─────────────────────
        for di in unmatched_d:
            det  = detections[di]
            zone = self._get_zone(det)
            self._tracks.append(_Track(det, zone))

        # ── 7. Delete tracks missing too long ─────────────────────────────────
        self._tracks = [t for t in self._tracks if t.misses < self.MAX_MISSES]

        # ── 8. Build output ───────────────────────────────────────────────────
        result = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            zone = t.zone
            result.append(TrackedPerson(
                track_id=     t.track_id,
                bbox=         t.bbox,
                confidence=   t.confidence,
                zone_id=      zone.zone_id      if zone else "unknown",
                zone_name=    zone.display_name if zone else "Unknown",
                is_restricted=zone.is_restricted if zone else False,
                age=          t.age,
                dwell_time=   t.dwell_time,
                is_lost=      t.misses > 0,
                velocity=     (t.vx, t.vy),
            ))

        return result

    def reset(self) -> None:
        """Clear all tracks (e.g. when stream reconnects)."""
        self._tracks.clear()
        logger.info(f"[{self.camera_id}] Tracker reset")

    def _get_zone(self, det: Detection) -> Optional[Zone]:
        """Find which zone this detection's feet fall in."""
        x1, y1, x2, y2 = det.bbox
        fx = (x1 + x2) // 2
        fy = y2   # feet = bottom center
        return get_zone_for_point(self.camera_id, fx, fy)

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._tracks if t.confirmed and t.misses == 0)
