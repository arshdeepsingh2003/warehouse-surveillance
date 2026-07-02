"""
ai/analyzer/activity_analyzer.py
──────────────────────────────────
Activity Analyzer — classifies what each tracked person is doing
from real movement patterns rather than random assignment.

Activities are derived from tracked-person bounding-box geometry,
velocity, dwell time, and (where available) spatial overlap with
detected carryable objects.

Classification logic
────────────────────
  WALKING             → movement speed exceeds MOVEMENT_THRESHOLD
  RUNNING             → speed exceeds SPEED_RUNNING_MIN
  STANDING            → speed below MOVEMENT_THRESHOLD, short dwell
  LOITERING           → speed below MOVEMENT_THRESHOLD,
                         dwell > LOITERING_SECONDS,
                         centroid stays within LOITERING_RADIUS
  FALLING             → bounding box wider than tall (AR >
                         FALL_ASPECT_RATIO_THRESHOLD) OR
                         sudden AR spike OR
                         sudden centroid y-drop
  CARRYING_OBJECT     → spatial IoU/overlap with detected carryable
                         objects (boxes, bags, tools)
  HANDLING_ITEMS      → only emitted when carryable-object detection
                         data is present AND person interacts with
                         objects; otherwise unsupported
  UNAUTHORIZED_ENTRY  → person's feet land in a zone with
                         is_restricted=True (zone_config.py)

Output: ActivityResult  (same schema consumed by alerts, APIs,
                         WebSocket events, and the dashboard)
"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ai.detector.person_detector import Detection
from ai.tracker.person_tracker import TrackedPerson
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Activity labels (must match backend schema) ───────────────────────────────
class ActivityLabel:
    WALKING           = "walking"
    RUNNING           = "running"
    STANDING          = "standing"
    LOITERING         = "loitering"
    FALLING           = "falling"
    CARRYING_OBJECT   = "carrying_object"
    CROUCHING         = "crouching"
    HANDLING_ITEMS    = "handling_items"
    UNAUTHORIZED_ENTRY= "unauthorized_entry"
    UNKNOWN           = "unknown"


# ── Anomaly flags (used by the rules engine) ──────────────────────────────────
class AnomalyFlag:
    LOITERING          = "loitering"
    RESTRICTED_ZONE    = "restricted_zone"
    POSSIBLE_FALL      = "possible_fall"
    FAST_MOVEMENT      = "fast_movement"
    LONG_DWELL         = "long_dwell"
    PPE_ZONE_VIOLATION = "ppe_zone_violation"
    THEFT_DETECTED     = "theft_detected"
    MISCONDUCT_DETECTED = "misconduct_detected"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ActivityResult:
    """One activity classification result for one person in one frame."""
    person_id:     str
    track_id:      int
    activity_type: str
    track_uuid:    str    = ""    # stable UUID for VLM key continuity
    anomaly_label: str    = "normal"    # "normal" | "anomaly"
    description:   str    = ""
    confidence:    float  = 0.8
    flags:         list[str] = field(default_factory=list)
    zone_id:       str    = ""
    zone_name:     str    = ""
    dwell_time:    float  = 0.0
    backend_used:  str    = "rules"

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly_label == "anomaly"


# ── Per-person history (needed for temporal analysis) ─────────────────────────

@dataclass
class _PersonHistory:
    """Stores recent observations for one person ID for temporal analysis."""
    track_id:          int
    aspect_ratios:     list[float] = field(default_factory=list)
    velocities:        list[float] = field(default_factory=list)
    centroids:         list[tuple[float, float]] = field(default_factory=list)
    baseline_ar:       Optional[float] = None
    zone_first_seen:   dict[str, float] = field(default_factory=dict)

    HISTORY_LEN = 20

    def add(self, ar: float, speed: float,
            centroid: tuple[float, float]) -> None:
        self.aspect_ratios.append(ar)
        self.velocities.append(speed)
        self.centroids.append(centroid)
        if len(self.aspect_ratios) > self.HISTORY_LEN:
            self.aspect_ratios.pop(0)
            self.velocities.pop(0)
            self.centroids.pop(0)
        if self.baseline_ar is None and len(self.aspect_ratios) >= 5:
            self.baseline_ar = sum(self.aspect_ratios[:5]) / 5

    @property
    def avg_speed(self) -> float:
        return sum(self.velocities) / len(self.velocities) if self.velocities else 0.0

    @property
    def recent_speed(self) -> float:
        return self.velocities[-1] if self.velocities else 0.0

    @property
    def current_ar(self) -> float:
        return self.aspect_ratios[-1] if self.aspect_ratios else 1.0

    def ar_spike(self) -> bool:
        """True if aspect ratio suddenly got much wider (person falling)."""
        if self.baseline_ar is None or len(self.aspect_ratios) < 5:
            return False
        return self.current_ar > self.baseline_ar * 2.0

    @property
    def movement_radius(self) -> float:
        """
        Approximate radius of person's recent movement area.

        Used to distinguish loitering from standing: a person who
        walks within a small area will have a larger radius than
        someone standing still.
        """
        if len(self.centroids) < 2:
            return 0.0
        recent = self.centroids[-min(10, len(self.centroids)):]
        cx_vals = [c[0] for c in recent]
        cy_vals = [c[1] for c in recent]
        spread_x = max(cx_vals) - min(cx_vals)
        spread_y = max(cy_vals) - min(cy_vals)
        return max(spread_x, spread_y) / 2.0


# ── IoU utility ───────────────────────────────────────────────────────────────

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


# ── Analyzer ──────────────────────────────────────────────────────────────────

class ActivityAnalyzer:
    """
    Classifies the activity of each tracked person from real movement patterns.

    All key thresholds are configurable via settings.py / .env:
      • MOVEMENT_THRESHOLD        — min px/frame to be considered "walking"
      • LOITERING_SECONDS         — min dwell for loitering
      • LOITERING_RADIUS          — max centroid spread for loitering
      • FALL_ASPECT_RATIO_THRESHOLD — w/h ratio indicating horizontal posture

    One shared instance per camera — persists _PersonHistory across frames.

    Usage:
        analyzer = ActivityAnalyzer(camera_id="cam-01")
        results  = analyzer.analyze(tracked_persons, carryable_objects)
    """

    SPEED_WALKING_MAX  = 8.0       # px/frame — upper bound for walking
    SPEED_RUNNING_MIN  = 8.0       # px/frame — above this is running
    CENTROID_DROP_THRESHOLD = 30.0 # pixel y-delta suggesting a fall

    CARRY_IOU_THRESHOLD  = 0.05
    CARRY_DIST_THRESHOLD = 60.0

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._history: dict[int, _PersonHistory] = {}

    @property
    def _movement_threshold(self) -> float:
        return settings.MOVEMENT_THRESHOLD

    @property
    def _ar_fall_threshold(self) -> float:
        return settings.FALL_ASPECT_RATIO_THRESHOLD

    @property
    def _loitering_radius(self) -> float:
        return settings.LOITERING_RADIUS

    def analyze(
        self,
        persons: list[TrackedPerson],
        carryable_objects: Optional[list[Detection]] = None,
    ) -> list[ActivityResult]:
        results = []
        active_ids = {p.track_id for p in persons}
        self._history = {k: v for k, v in self._history.items() if k in active_ids}

        for person in persons:
            result = self._classify(person, carryable_objects)
            results.append(result)

        return results

    def _carrying_object(
        self,
        person: TrackedPerson,
        objects: list[Detection],
    ) -> Optional[float]:
        """Check spatial overlap between person and carryable objects."""
        px1, py1, px2, py2 = person.bbox
        pw = max(px2 - px1, 1)
        ph = max(py2 - py1, 1)

        best_conf = None
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj.bbox

            # Condition 1: Partial bounding box overlap
            ix1 = max(px1, ox1); iy1 = max(py1, oy1)
            ix2 = min(px2, ox2); iy2 = min(py2, oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            person_area = pw * ph
            obj_area = max(ox2 - ox1, 1) * max(oy2 - oy1, 1)
            union = person_area + obj_area - inter
            iou = inter / union if union > 0 else 0.0

            if iou > self.CARRY_IOU_THRESHOLD:
                score = min(1.0, 0.6 + iou * 2.0)
                if best_conf is None or score > best_conf:
                    best_conf = score
                continue

            # Condition 2: Object center lies within person bbox
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2
            if px1 <= ocx <= px2 and py1 <= ocy <= py2:
                score = 0.65
                if best_conf is None or score > best_conf:
                    best_conf = score
                continue

            # Condition 3: Object center close to person center
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            dist = math.sqrt((ocx - pcx) ** 2 + (ocy - pcy) ** 2)
            max_dim = max(pw, ph)
            if max_dim > 0 and dist < min(self.CARRY_DIST_THRESHOLD, max_dim * 1.2):
                proximity = 1.0 - (dist / max_dim)
                score = max(0.50, min(0.75, proximity))
                if best_conf is None or score > best_conf:
                    best_conf = score

        return best_conf

    def _classify(
        self,
        person: TrackedPerson,
        carryable_objects: Optional[list[Detection]] = None,
    ) -> ActivityResult:
        """Classify one person's activity from real movement and posture."""
        tid = person.track_id

        if tid not in self._history:
            self._history[tid] = _PersonHistory(track_id=tid)

        hist = self._history[tid]

        # Compute motion features from tracked-person data
        x1, y1, x2, y2 = person.bbox
        w = x2 - x1
        h = max(y2 - y1, 1)
        ar = w / h
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        vx, vy = person.velocity
        speed  = math.sqrt(vx**2 + vy**2)

        hist.add(ar, speed, (cx, cy))

        # ── Rule 1: Restricted zone entry → unauthorized entry ───────────────
        if person.is_restricted:
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                track_uuid=   person.track_uuid,
                activity_type=ActivityLabel.UNAUTHORIZED_ENTRY,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} entered restricted zone "
                    f"'{person.zone_name}' without authorisation."
                ),
                confidence=   0.92,
                flags=        [AnomalyFlag.RESTRICTED_ZONE],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 2: Possible fall (AR change + centroid drop) ─────────────────
        # A fall is detected when:
        #   a) bounding box becomes wider than tall (horizontal posture), OR
        #   b) aspect ratio spikes relative to baseline, OR
        #   c) centroid y-position drops suddenly (person falling down)
        is_ar_fall  = ar > self._ar_fall_threshold or hist.ar_spike()
        is_drop_fall = False
        if len(hist.centroids) >= 3:
            cy_vals = [c[1] for c in hist.centroids[-3:]]
            drop_rate = cy_vals[-1] - cy_vals[0]
            is_drop_fall = drop_rate > self.CENTROID_DROP_THRESHOLD

        if (is_ar_fall or is_drop_fall) and person.age > 10:
            desc_parts = []
            if is_ar_fall:
                desc_parts.append("horizontal posture detected")
            if is_drop_fall:
                desc_parts.append("sudden vertical drop")
            detail = " — ".join(desc_parts) if desc_parts else "fall detected"
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                track_uuid=   person.track_uuid,
                activity_type=ActivityLabel.FALLING,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} appears to have fallen in "
                    f"'{person.zone_name}' — {detail}."
                ),
                confidence=   0.85,
                flags=        [AnomalyFlag.POSSIBLE_FALL],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 3: Running (velocity exceeds running threshold) ──────────────
        if speed > self.SPEED_RUNNING_MIN:
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                track_uuid=   person.track_uuid,
                activity_type=ActivityLabel.RUNNING,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} is running in '{person.zone_name}'. "
                    f"Speed: {speed:.1f} px/frame."
                ),
                confidence=   0.78,
                flags=        [AnomalyFlag.FAST_MOVEMENT],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 4: Loitering (low speed + long dwell + small area) ───────────
        if (speed < self._movement_threshold
                and person.dwell_time > settings.LOITERING_SECONDS
                and hist.movement_radius < self._loitering_radius):
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                track_uuid=   person.track_uuid,
                activity_type=ActivityLabel.LOITERING,
                anomaly_label="anomaly",
                description=  (
                    f"{person.person_id} has been stationary in "
                    f"'{person.zone_name}' for {int(person.dwell_time)}s "
                    f"— loitering detected."
                ),
                confidence=   0.80,
                flags=        [AnomalyFlag.LOITERING, AnomalyFlag.LONG_DWELL],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 5: Carrying object (spatial overlap with detected objects) ──
        carry_conf = None
        if carryable_objects:
            carry_conf = self._carrying_object(person, carryable_objects)
        if carry_conf is not None:
            objects_nearby = sum(
                1 for o in carryable_objects
                if _iou(person.bbox, o.bbox) > 0
            )
            return ActivityResult(
                person_id=    person.person_id,
                track_id=     tid,
                track_uuid=   person.track_uuid,
                activity_type=ActivityLabel.CARRYING_OBJECT,
                anomaly_label="normal",
                description=  (
                    f"{person.person_id} is carrying {objects_nearby} object(s) "
                    f"in '{person.zone_name}'."
                ),
                confidence=   min(0.88, carry_conf),
                flags=        [],
                zone_id=      person.zone_id,
                zone_name=    person.zone_name,
                dwell_time=   person.dwell_time,
            )

        # ── Rule 5b: Handling items (only when object detection data exists) ──
        # handling_items is only emitted when carryable objects are present
        # AND the person is actively interacting with them. Without object
        # detection data this activity is unsupported — no fake events.
        if carryable_objects and len(carryable_objects) > 0:
            handling_conf = self._carrying_object(person, carryable_objects)
            if handling_conf is not None:
                return ActivityResult(
                    person_id=    person.person_id,
                    track_id=     tid,
                    track_uuid=   person.track_uuid,
                    activity_type=ActivityLabel.HANDLING_ITEMS,
                    anomaly_label="normal",
                    description=  (
                        f"{person.person_id} is handling items at "
                        f"'{person.zone_name}'."
                    ),
                    confidence=   min(0.80, handling_conf * 0.9),
                    flags=        [],
                    zone_id=      person.zone_id,
                    zone_name=    person.zone_name,
                    dwell_time=   person.dwell_time,
                )

        # ── Rule 6: Normal activities (no anomaly) ────────────────────────────
        if speed < self._movement_threshold:
            activity = ActivityLabel.STANDING
            desc = (f"{person.person_id} is standing in '{person.zone_name}'.")
        else:
            activity = ActivityLabel.WALKING
            desc = (f"{person.person_id} is walking through '{person.zone_name}'.")

        return ActivityResult(
            person_id=    person.person_id,
            track_id=     tid,
            track_uuid=   person.track_uuid,
            activity_type=activity,
            anomaly_label="normal",
            description=  desc,
            confidence=   0.85,
            flags=        [],
            zone_id=      person.zone_id,
            zone_name=    person.zone_name,
            dwell_time=   person.dwell_time,
        )
