"""
ai/tracker/person_tracker.py
─────────────────────────────
Person Tracker — assigns stable IDs to detected persons across frames.
Supports track_uuid persistence across tracker ID regeneration.

Architecture: IoU-based Kalman tracker (ByteTrack-lite)
─────────────────────────────────────────────────────────
We implement a lightweight ByteTrack-inspired tracker that:
  1. Predicts each track's position using a Kalman filter
  2. Matches new detections to existing tracks using IoU
  3. Assigns a new ID to unmatched detections (with re-identification)
  4. Removes tracks that haven't been matched for N frames

Track lifecycle:
  NEW         → first seen, needs N confirmations
  CONFIRMED   → reliably tracked, published to downstream
  LOST        → not matched for a few frames (still in memory)
  DELETED     → missing too long → removed (saved in recently_deleted buffer)

When a track is deleted and a new detection matches the last-known position
of a recently deleted track (IoU ≥ 0.3, within 5 seconds), the original
track_uuid is carried forward to maintain VLM cache continuity.

Output per track:
  TrackedPerson = {
    track_uuid: str             stable UUID surviving re-identification
    track_id:   int             transient counter ID (1001, 1002, ...)
    bbox:       (x1,y1,x2,y2)  current bounding box
    confidence: float
    zone_id:    str             which zone the person is in
    age:        int             frames this track has been active
    dwell_time: float           seconds in current zone
    is_lost:    bool
  }
"""

from __future__ import annotations

import hashlib
import time
import uuid
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ai.detector.person_detector import Detection
from ai.zones.zone_config import get_zone_for_point, Zone

logger = logging.getLogger(__name__)

# ── TrackedPerson dataclass ───────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """
    One tracked person — the output of the tracker per frame.
    Everything the rules engine and activity analyzer needs is here.
    """
    track_id:    int
    track_uuid:  str    # stable UUID that survives re-identification
    camera_id:   str    # camera this person was tracked on
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
        """Camera-aware ID — unique across the system (e.g. 01-P1001)."""
        cam = "".join(c for c in self.camera_id if c.isdigit())
        return f"{cam}-P{self.track_id + 1000}"

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
            "track_uuid":  self.track_uuid,
            "camera_id":   self.camera_id,
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


@dataclass
class _RecentTrack:
    """A recently deleted track kept for re-identification."""
    track_uuid: str
    last_bbox: tuple[int, int, int, int]
    last_center: tuple[float, float]
    deleted_at: float
    last_person_id: str
    feature_hash: Optional[str] = None  # HSV histogram hash for appearance matching


# ── Internal Track ────────────────────────────────────────────────────────────

class _Track:
    """Internal state for one tracked person (not exposed outside this module)."""

    def __init__(
        self,
        detection: Detection,
        zone: Optional[Zone],
        next_id_source: list[int],
        track_uuid: Optional[str] = None,
    ) -> None:
        self.track_id      = next_id_source[0]
        next_id_source[0] += 1

        # Use provided track_uuid (re-identification) or generate new one
        self.track_uuid    = track_uuid or uuid.uuid4().hex[:12]

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

        self.created_at: float = time.monotonic()  # for lifetime computation

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

        if self.hits >= PersonTracker.MIN_HITS:
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
        # Dampen velocity to prevent runaway predictions into ghost boxes.
        # NOTE: predict() is NOT called here because the main update() loop
        # already called it for ALL tracks at step 1. Calling it again would
        # double-predict and accelerate ghost drift.
        self.vx *= 0.7
        self.vy *= 0.7

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


def _compute_feature_hash(frame: np.ndarray, bbox: tuple) -> str:
    """
    Compute a simple appearance hash from the HSV histogram of the
    detection region. Used for re-identification when IoU alone
    would be ambiguous (e.g. people re-entering after occlusion).

    Returns a hex string hash of the quantised 3D HSV histogram.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return "0" * 32
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "0" * 32
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 8×4×4 bins in H,S,V
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = hist.astype(np.uint8).flatten()
        # Hash the flattened histogram
        return hashlib.md5(hist.tobytes()).hexdigest()
    except Exception:
        return "0" * 32


def _histogram_similarity(
    frame: np.ndarray,
    bbox1: tuple,
    bbox2: tuple,
) -> float:
    """Compare two detection regions by HSV histogram correlation (0–1)."""
    try:
        def _hist(b):
            x1, y1, x2, y2 = b
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            c = frame[y1:y2, x1:x2]
            if c.size == 0:
                return None
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                             [0, 180, 0, 256, 0, 256])
            cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
            return h
        h1 = _hist(bbox1)
        h2 = _hist(bbox2)
        if h1 is None or h2 is None:
            return 0.0
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    except Exception:
        return 0.0


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
    IOU_THRESHOLD            = 0.20   # minimum IoU to match detection to track (relaxed for occlusion recovery)
    MAX_MISSES               = 10     # frames without detection before deleting track (~2s at 10fps)
    MIN_HITS                 = 3      # confirmations required before publishing (reduces ghost tracks)
    FRAME_VISIBILITY_THRESHOLD = 0.15 # minimum fraction of bbox that must be
                                      # visible inside the frame — below this
                                      # the track is deleted immediately
    EDGE_MARGIN              = 3      # pixels from frame edge to consider "touching edge"
    REIDENTIFY_IOU_THRESHOLD = 0.30   # minimum IoU to re-identify a recently deleted track
    REIDENTIFY_TIME_LIMIT   = 5.0    # seconds to keep recently deleted tracks for re-identification

    def __init__(self, camera_id: str, debug_dir: Optional[str] = None) -> None:
        self.camera_id = camera_id
        self._tracks:  list[_Track] = []
        self._frame_n: int = 0
        # Each tracker gets its own ID counter (was a class-level global before)
        self._next_id: list[int] = [1]
        # Recently deleted tracks — used for track_uuid re-identification
        self._recently_deleted: deque[_RecentTrack] = deque(maxlen=50)
        # Audit counters
        self._track_lifetime_total: float = 0.0
        self._track_lifetime_count: int = 0
        self._track_lifetimes: list[float] = []
        self._id_switches_per_minute: list[float] = []
        self._tracks_deleted_before_vlm: int = 0
        self._debug_dir = debug_dir
        logger.info(
            f"[{camera_id}] PersonTracker initialised "
            f"MAX_MISSES={self.MAX_MISSES} MIN_HITS={self.MIN_HITS} "
            f"IOU_THRESHOLD={self.IOU_THRESHOLD} "
            f"VISIBILITY={self.FRAME_VISIBILITY_THRESHOLD}"
        )

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

        n_input_dets = len(detections)
        n_pre_tracks = len(self._tracks)

        # ── 1. Predict ────────────────────────────────────────────────────────
        for track in self._tracks:
            track.predict()

        # ── 2. Prune out-of-frame and edge-ghost tracks ────────────────────
        # If a predicted bbox has drifted mostly outside the visible frame,
        # delete it immediately instead of letting it ghost for MAX_MISSES frames.
        # Also prune missed tracks pressed against the frame edge — the bbox
        # property clamps coordinates so a drifted-out box appears stuck at edge.
        pre_prune = len(self._tracks)
        self._tracks = [
            t for t in self._tracks
            if t.visible_ratio(frame_w, frame_h) >= self.FRAME_VISIBILITY_THRESHOLD
            and not (t.misses >= 3 and t.touches_edge(frame_w, frame_h, self.EDGE_MARGIN))
        ]
        pruned = pre_prune - len(self._tracks)
        if pruned > 0:
            logger.debug(
                f"[{self.camera_id}] Pruned {pruned} out-of-frame/ghost tracks"
            )

        # ── 3. Match ──────────────────────────────────────────────────────────
        matches, unmatched_t, unmatched_d = _hungarian_match(
            self._tracks, detections, self.IOU_THRESHOLD
        )

        # ── Per-detection association log ─────────────────────────────────────
        logger.debug(
            f"[{self.camera_id}] Association: "
            f"{len(matches)} matches, {len(unmatched_t)} unmatched tracks, "
            f"{len(unmatched_d)} unmatched detections "
            f"(of {n_input_dets} input dets vs {n_pre_tracks} tracks)"
        )
        for ti, di in matches:
            t = self._tracks[ti]
            d = detections[di]
            logger.debug(
                f"[{self.camera_id}]   MATCH: track P-{t.track_id+1000} "
                f"(misses={t.misses}) ↔ det #{di} "
                f"box=({d.x1},{d.y1},{d.x2},{d.y2}) conf={d.confidence:.4f}"
            )
        for ti in unmatched_t:
            t = self._tracks[ti]
            logger.debug(
                f"[{self.camera_id}]   UNMATCHED TRACK: P-{t.track_id+1000} "
                f"(misses={t.misses}, conf={t.confidence:.4f})"
            )
        for di in unmatched_d:
            d = detections[di]
            logger.debug(
                f"[{self.camera_id}]   NEW DETECTION: #{di} "
                f"box=({d.x1},{d.y1},{d.x2},{d.y2}) conf={d.confidence:.4f}"
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
        n_reidentified = 0
        for di in unmatched_d:
            det  = detections[di]
            zone = self._get_zone(det)

            # Check recently_deleted for re-identification
            reid_uuid = self._match_recently_deleted(det, frame)
            if reid_uuid is not None:
                n_reidentified += 1
                self._tracks.append(_Track(det, zone, self._next_id, track_uuid=reid_uuid))
                logger.info(
                    f"[{self.camera_id}] REIDENTIFIED track_uuid={reid_uuid} "
                    f"(matched by position to recently deleted track)"
                )
            else:
                self._tracks.append(_Track(det, zone, self._next_id))

        # ── 7. Delete tracks missing too long ─────────────────────────────────
        n_before_delete = len(self._tracks)
        deleted_tracks = [t for t in self._tracks if t.misses >= self.MAX_MISSES]
        self._tracks = [t for t in self._tracks if t.misses < self.MAX_MISSES]
        n_deleted = n_before_delete - len(self._tracks)

        # Save deleted tracks to recently_deleted buffer
        now = time.monotonic()
        for t in deleted_tracks:
            bbox = t.bbox
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            feat = _compute_feature_hash(frame, bbox)
            self._recently_deleted.append(_RecentTrack(
                track_uuid=t.track_uuid,
                last_bbox=bbox,
                last_center=(cx, cy),
                deleted_at=now,
                last_person_id=f"{''.join(c for c in self.camera_id if c.isdigit())}-P{t.track_id + 1000}",
                feature_hash=feat,
            ))
            # Track lifetime for audit
            if t.hits >= PersonTracker.MIN_HITS:
                lifetime = now - t.created_at
                self._track_lifetime_total += lifetime
                self._track_lifetime_count += 1
                self._track_lifetimes.append(lifetime)
                self._tracks_deleted_before_vlm += 1

        # Prune expired recently_deleted entries
        self._prune_recently_deleted(now)

        # ── Track lifecycle summary ────────────────────────────────────────────
        n_new_tracks = len(unmatched_d) - n_reidentified
        n_matched = len(matches)
        n_missed = len(unmatched_t)
        n_confirmed = sum(1 for t in self._tracks if t.confirmed)
        n_unconfirmed = sum(1 for t in self._tracks if not t.confirmed)
        logger.info(
            f"🔍 TRACK[assoc] camera={self.camera_id} "
            f"dets_in={n_input_dets} "
            f"matched={n_matched} "
            f"new_tracks={n_new_tracks} "
            f"reidentified={n_reidentified} "
            f"missed={n_missed} "
            f"deleted={n_deleted} "
            f"total_tracks={len(self._tracks)} "
            f"confirmed={n_confirmed} "
            f"unconfirmed={n_unconfirmed}"
        )

        # ── 8. Build output ───────────────────────────────────────────────────
        result = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            zone = t.zone
            result.append(TrackedPerson(
                track_id=     t.track_id,
                track_uuid=   t.track_uuid,
                camera_id=    self.camera_id,
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

        # ── Debug logging per confirmed track ──────────────────────────────
        for p in result:
            feat = _compute_feature_hash(frame, p.bbox)
            logger.info(
                f"TRACK_ID={p.person_id} "
                f"DETECTION_BOX={list(p.bbox)} "
                f"PERSON_FEATURE_HASH={feat[:16]} "
                f"CAMERA_ID={self.camera_id}"
            )

        # ── Duplicate ID check ─────────────────────────────────────────────
        seen_ids: set[str] = set()
        for p in result:
            if p.person_id in seen_ids:
                logger.error(
                    f"[DUPLICATE-ID] camera={self.camera_id} "
                    f"DUPLICATE person_id={p.person_id} "
                    f"bbox={list(p.bbox)} — this should never happen!"
                )
            seen_ids.add(p.person_id)

        # ── Save debug image with tracked persons drawn ─────────────────────
        if self._debug_dir:
            import os as _os
            viz = frame.copy()
            for p in result:
                x1, y1, x2, y2 = p.bbox
                color = (0, 255, 255) if p.is_lost else (0, 255, 0)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz, f"{p.person_id} (active)" if not p.is_lost else f"{p.person_id} (lost)",
                            (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Also draw raw detections that came in
            for i, d in enumerate(detections):
                cv2.rectangle(viz, (d.x1, d.y1), (d.x2, d.y2), (0, 165, 255), 1)
                cv2.putText(viz, f"det#{i}", (d.x1, max(d.y1-15, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
            save_dir = _os.path.join(self._debug_dir, self.camera_id, "tracker_output")
            _os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                _os.path.join(save_dir, f"frame_{self._frame_n:06d}.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

        n_active = sum(1 for p in result if not p.is_lost)
        n_lost   = sum(1 for p in result if p.is_lost)
        confs    = [f"{p.confidence:.2f}" for p in result]

        logger.info(
            f"🔍 TRACE[tracker] camera={self.camera_id} | "
            f"active_tracks={n_active} | total_tracks={len(result)} | "
            f"ids=[{', '.join(p.person_id for p in result if not p.is_lost)}] | "
            f"bboxes=[{'; '.join(str(list(p.bbox)) for p in result if not p.is_lost)}]"
        )

        logger.info(
            f"[{self.camera_id}] Tracker frame={self._frame_n} | "
            f"detections={n_input_dets} | "
            f"tracks_in={n_pre_tracks} | "
            f"tracks_out={len(result)} (active={n_active}, lost={n_lost}) | "
            f"confidences=[{', '.join(confs)}]"
        )

        return result

    def reset(self) -> None:
        """Clear all tracks (e.g. when stream reconnects)."""
        self._tracks.clear()
        self._recently_deleted.clear()
        self._next_id = [1]
        logger.info(f"[{self.camera_id}] Tracker reset (ID counter cleared)")

    def _match_recently_deleted(
        self,
        detection: Detection,
        frame: np.ndarray,
    ) -> Optional[str]:
        """
        Check if *detection* matches a recently deleted track.
        Uses both IoU and appearance similarity for robust re-identification.
        Returns the track_uuid if a match is found, None otherwise.
        """
        det_feature = _compute_feature_hash(frame, detection.bbox)
        best_uuid: Optional[str] = None
        best_score = 0.0
        for recent in self._recently_deleted:
            iou = _iou(recent.last_bbox, detection.bbox)
            # Appearance similarity: compare current detection's feature hash
            # with the stored hash of the recently deleted track
            appearance = 0.0
            if recent.feature_hash and recent.feature_hash != "0" * 32:
                if det_feature == recent.feature_hash:
                    appearance = 1.0
            # Combined score: weight IoU more for position, appearance as tiebreaker
            score = iou * 0.7 + appearance * 0.3
            if score > best_score and iou >= self.REIDENTIFY_IOU_THRESHOLD:
                best_score = score
                best_uuid = recent.track_uuid
        if best_uuid:
            logger.info(
                f"[{self.camera_id}] Re-identifying det {detection.bbox} "
                f"→ track_uuid={best_uuid} (score={best_score:.3f})"
            )
        else:
            logger.debug(
                f"[{self.camera_id}] No re-id match for det {detection.bbox} "
                f"(feature={det_feature[:12]})"
            )
        return best_uuid

    def _prune_recently_deleted(self, now: float) -> None:
        """Remove expired entries from recently_deleted buffer."""
        self._recently_deleted = deque(
            (r for r in self._recently_deleted
             if now - r.deleted_at < self.REIDENTIFY_TIME_LIMIT),
            maxlen=50,
        )

    def _get_zone(self, det: Detection) -> Optional[Zone]:
        """Find which zone this detection's feet fall in."""
        x1, y1, x2, y2 = det.bbox
        fx = (x1 + x2) // 2
        fy = y2   # feet = bottom center
        return get_zone_for_point(self.camera_id, fx, fy)

    def get_audit_metrics(self) -> dict:
        """Return tracker audit metrics for reporting."""
        avg_lifetime = (self._track_lifetime_total / max(self._track_lifetime_count, 1))
        sorted_lts = sorted(self._track_lifetimes)
        n = len(sorted_lts)
        median = sorted_lts[n // 2] if n > 0 else 0.0
        now_ = time.monotonic()
        id_switches = sum(1 for r in self._recently_deleted if now_ - r.deleted_at < 60)
        return {
            "average_track_lifetime_s": round(avg_lifetime, 2),
            "median_track_lifetime_s": round(median, 2),
            "track_lifetime_count": self._track_lifetime_count,
            "id_switches_per_minute": id_switches,
            "tracks_deleted_before_vlm": self._tracks_deleted_before_vlm,
            "active_tracks": len(self._tracks),
            "recently_deleted_buffer_size": len(self._recently_deleted),
        }

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._tracks if t.confirmed and t.misses == 0)
