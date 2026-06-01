"""
ai/zones/zone_config.py
────────────────────────
Zone Configuration — defines which areas of each camera frame
correspond to which logical warehouse zones.

A zone is a polygon drawn on the camera's coordinate space (0–640 × 0–360).
When a person's feet (bottom-center of bounding box) fall inside a polygon,
they are "in" that zone.

How to calibrate zones for a real camera:
  1. Capture a snapshot:  http://localhost:8001/snapshot/cam-01
  2. Open in an image editor (GIMP, Photoshop, or paint.net)
  3. Note the pixel coordinates of the zone corners
  4. Add them to ZONE_POLYGONS below

Zone IDs must match the zone names used in the backend
(cameras table, alerts, activities).

This file is the single source of truth for zone geometry.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Zone:
    """One monitoring zone within a camera's field of view."""
    zone_id:       str                      # e.g. "restricted_area"
    display_name:  str                      # e.g. "Restricted Area"
    polygon:       list[tuple[int, int]]    # [(x1,y1),(x2,y2),...]  pixel coords
    is_restricted: bool = False             # triggers entry alerts
    color_bgr:     tuple[int, int, int] = (0, 229, 255)  # overlay color

    def to_np(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.int32)

    def contains_point(self, x: int, y: int) -> bool:
        """
        Returns True if point (x, y) is inside this zone polygon.
        Uses OpenCV's pointPolygonTest.
        """
        import cv2
        pt = np.array(self.polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pt, (float(x), float(y)), False)
        return result >= 0


# ── Zone definitions per camera ───────────────────────────────────────────────
# Frame size: 640 × 360
# Coordinates: (x_from_left, y_from_top)
#
# For frames without calibrated zones we fall back to a single full-frame
# "general_zone". Add real polygons after capturing camera snapshots.

ZONE_CONFIG: dict[str, list[Zone]] = {

    # ── cam-01: Main Gate / Entry Zone ────────────────────────────────────────
    "cam-01": [
        Zone(
            zone_id="entry_zone",
            display_name="Entry Zone",
            polygon=[(0, 0), (640, 0), (640, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(0, 200, 100),
        ),
    ],

    # ── cam-02: Warehouse Aisle / Storage ─────────────────────────────────────
    "cam-02": [
        Zone(
            zone_id="storage_area",
            display_name="Storage Area",
            polygon=[(0, 0), (640, 0), (640, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(0, 180, 255),
        ),
    ],

    # ── cam-03: Loading Zone ──────────────────────────────────────────────────
    "cam-03": [
        Zone(
            zone_id="loading_zone",
            display_name="Loading Zone",
            polygon=[(0, 0), (640, 0), (640, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(0, 165, 255),
        ),
    ],

    # ── cam-04: Storage Rack Area ─────────────────────────────────────────────
    "cam-04": [
        Zone(
            zone_id="storage_area",
            display_name="Rack Storage",
            polygon=[(0, 120), (640, 120), (640, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(0, 180, 255),
        ),
    ],

    # ── cam-05: Restricted Area ───────────────────────────────────────────────
    # Split into two zones: a restricted corridor and an allowed walkway
    "cam-05": [
        Zone(
            zone_id="restricted_area",
            display_name="Restricted Area",
            # Right 2/3 of frame = restricted zone
            polygon=[(200, 0), (640, 0), (640, 360), (200, 360)],
            is_restricted=True,
            color_bgr=(0, 0, 220),
        ),
        Zone(
            zone_id="walkway",
            display_name="Allowed Walkway",
            polygon=[(0, 0), (200, 0), (200, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(0, 200, 100),
        ),
    ],

    # ── cam-06: Packing / Dispatch ────────────────────────────────────────────
    "cam-06": [
        Zone(
            zone_id="packing_area",
            display_name="Packing Area",
            polygon=[(0, 0), (640, 0), (640, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(180, 0, 200),
        ),
    ],
}


def get_zones_for_camera(camera_id: str) -> list[Zone]:
    """Return zone list for a camera, falling back to a full-frame default."""
    return ZONE_CONFIG.get(camera_id, [
        Zone(
            zone_id="general_zone",
            display_name="General Zone",
            polygon=[(0, 0), (640, 0), (640, 360), (0, 360)],
            is_restricted=False,
            color_bgr=(100, 100, 100),
        )
    ])


def get_zone_for_point(camera_id: str, x: int, y: int) -> Optional[Zone]:
    """
    Find which zone a point belongs to.
    Returns the first matching zone (order matters — put restricted first).
    Returns None if point is outside all zones.
    """
    for zone in get_zones_for_camera(camera_id):
        if zone.contains_point(x, y):
            return zone
    return None
