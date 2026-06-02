"""
ai/overlay/frame_overlay.py
─────────────────────────────
Frame Overlay — draws detection results onto video frames.

Renders on each frame:
  • Bounding boxes colored by activity / anomaly status
  • Person ID + zone label
  • Dwell time bar
  • Zone polygon outlines
  • Mini stats: person count, fps, timestamp

The annotated frame is then JPEG-encoded and served via the MJPEG endpoint,
so the dashboard shows bounding boxes directly in the camera feed.

Color coding:
  Cyan      → normal activity
  Orange    → warning (loitering, fast movement)
  Red       → high severity anomaly (fall, restricted entry)
  Green     → zone outlines (normal zones)
  Red tint  → zone outlines (restricted zones)
"""

from __future__ import annotations

import time
import cv2
import numpy as np

from ai.tracker.person_tracker import TrackedPerson
from ai.analyzer.activity_analyzer import ActivityResult
from ai.rules.rules_engine import AlertEvent
from ai.zones.zone_config import get_zones_for_camera


# ── Color palette (BGR) ───────────────────────────────────────────────────────
_CYAN    = (255, 229, 0)    # normal track
_ORANGE  = (0, 165, 255)    # warning
_RED     = (0, 50, 220)     # high severity anomaly
_GREEN   = (100, 220, 80)   # zone borders
_WHITE   = (220, 220, 220)
_DARK    = (20, 20, 30)
_GRID    = (0, 229, 255)    # scan-grid lines


def _severity_color(severity: str | None) -> tuple:
    return {
        "high":   _RED,
        "medium": _ORANGE,
        "low":    _CYAN,
        None:     _CYAN,
    }.get(severity, _CYAN)


class FrameOverlay:
    """
    Draws AI detection results onto a camera frame.

    Usage:
        overlay = FrameOverlay(camera_id="cam-01")
        annotated = overlay.draw(
            frame        = frame.copy(),
            persons      = tracked_persons,
            activities   = activity_results,
            active_alerts= alert_events,
        )
        # Pass annotated to the MJPEG encoder
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.36
    FONT_MED   = 0.45

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._frame_count = 0

    def draw(
        self,
        frame:         np.ndarray,
        persons:       list[TrackedPerson],
        activities:    list[ActivityResult],
        active_alerts: list[AlertEvent],
    ) -> np.ndarray:
        """
        Annotate a frame with tracking + activity results.

        Args:
            frame:         BGR frame to draw on (in-place).
            persons:       Tracked persons from PersonTracker.
            activities:    Activity results from ActivityAnalyzer.
            active_alerts: Any alerts generated this frame by RulesEngine.

        Returns:
            The annotated frame (same object as input, modified in-place).
        """
        self._frame_count += 1

        # Build lookup: track_id → activity
        act_by_track = {a.track_id: a for a in activities}
        alert_person_ids = {a.person_id for a in active_alerts}

        # ── 1. Draw zone polygons ─────────────────────────────────────────────
        self._draw_zones(frame)

        # ── 2. Draw each tracked person ───────────────────────────────────────
        for person in persons:
            if person.is_lost:
                continue  # skip ghost boxes — tracker cleans them up quickly
            activity = act_by_track.get(person.track_id)
            is_alert = person.person_id in alert_person_ids
            self._draw_person(frame, person, activity, is_alert)

        # ── 3. HUD overlay ────────────────────────────────────────────────────
        self._draw_hud(frame, persons, active_alerts)

        return frame

    # ── Zone polygons ─────────────────────────────────────────────────────────

    def _draw_zones(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        zones = get_zones_for_camera(self.camera_id)
        overlay = frame.copy()

        for zone in zones:
            pts = zone.to_np().reshape((-1, 1, 2))
            color = (0, 40, 180) if zone.is_restricted else (0, 60, 30)
            cv2.fillPoly(overlay, [pts], color)

        # Blend overlay with original (semi-transparent fill)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Draw zone borders
        for zone in zones:
            pts = zone.to_np().reshape((-1, 1, 2))
            border_color = (0, 30, 200) if zone.is_restricted else zone.color_bgr
            cv2.polylines(frame, [pts], True, border_color, 1, cv2.LINE_AA)

            # Zone label
            if len(zone.polygon) >= 3:
                cx = int(sum(p[0] for p in zone.polygon) / len(zone.polygon))
                cy = int(sum(p[1] for p in zone.polygon) / len(zone.polygon))
                cv2.putText(
                    frame, zone.display_name.upper(),
                    (cx - 30, cy), self.FONT, 0.28,
                    border_color, 1, cv2.LINE_AA,
                )

    # ── Person bounding box + label ───────────────────────────────────────────

    def _draw_person(
        self,
        frame:    np.ndarray,
        person:   TrackedPerson,
        activity: ActivityResult | None,
        is_alert: bool,
    ) -> None:
        x1, y1, x2, y2 = person.bbox

        # Choose color
        if is_alert:
            color = _RED
            thickness = 2
        elif activity and activity.is_anomaly:
            color = _ORANGE
            thickness = 2
        else:
            color = _CYAN
            thickness = 1

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        _draw_corners(frame, x1, y1, x2, y2, color, size=8)

        # ── Person ID badge ───────────────────────────────────────────────────
        label = person.person_id
        (lw, lh), _ = cv2.getTextSize(label, self.FONT, self.FONT_SMALL, 1)
        badge_y = max(y1 - 2, lh + 4)
        cv2.rectangle(frame, (x1, badge_y - lh - 4), (x1 + lw + 6, badge_y + 2), _DARK, -1)
        cv2.putText(frame, label, (x1 + 3, badge_y - 2),
                    self.FONT, self.FONT_SMALL, color, 1, cv2.LINE_AA)

        # ── Activity label ────────────────────────────────────────────────────
        if activity:
            act_label = activity.activity_type.replace("_", " ").upper()
            cv2.putText(frame, act_label,
                        (x1, y2 + 13), self.FONT, 0.30,
                        color, 1, cv2.LINE_AA)

        # ── Dwell time bar ────────────────────────────────────────────────────
        if person.dwell_time > 5:
            bar_w  = x2 - x1
            dwell_ratio = min(person.dwell_time / 120.0, 1.0)  # 2 min = full bar
            filled = int(bar_w * dwell_ratio)
            bar_y  = y2 + 3
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 3), _DARK, -1)
            bar_color = _RED if dwell_ratio > 0.7 else _ORANGE if dwell_ratio > 0.4 else _CYAN
            cv2.rectangle(frame, (x1, bar_y), (x1 + filled, bar_y + 3), bar_color, -1)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame:         np.ndarray,
        persons:       list[TrackedPerson],
        active_alerts: list[AlertEvent],
    ) -> None:
        h, w = frame.shape[:2]

        # Camera ID + timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"{self.camera_id.upper()}",
                    (8, 14), self.FONT, 0.38, _GRID, 1, cv2.LINE_AA)
        cv2.putText(frame, ts,
                    (8, h - 8), self.FONT, 0.32, (80, 150, 150), 1, cv2.LINE_AA)

        # Person count
        n_active = sum(1 for p in persons if not p.is_lost)
        cv2.putText(frame, f"PERSONS: {n_active}",
                    (w - 90, 14), self.FONT, 0.32, _WHITE, 1, cv2.LINE_AA)

        # Active alert indicator
        if active_alerts:
            sev = active_alerts[0].severity
            color = _RED if sev == "high" else _ORANGE
            cv2.rectangle(frame, (w - 16, 2), (w - 2, 22), color, -1)
            cv2.putText(frame, "!", (w - 12, 18),
                        self.FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Corner bracket decoration
        _draw_corners(frame, 0, 0, w - 1, h - 1, (0, 80, 80), size=12)


# ── Helper: draw corner brackets ─────────────────────────────────────────────

def _draw_corners(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    size: int = 10,
    thickness: int = 1,
) -> None:
    """Draw four L-shaped corner marks instead of a full rectangle."""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + size, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + size), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - size, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + size), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + size, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - size), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - size, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - size), color, thickness)
