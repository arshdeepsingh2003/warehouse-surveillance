"""
pipeline/api_client.py
───────────────────────
Async HTTP + WebSocket client used by FrameProcessor to:
  1. POST activity log events  →  /api/v1/activities  (backend REST)
  2. POST alert events         →  /api/v1/alerts       (backend REST)
  3. Broadcast frame updates   →  backend WebSocket hub (or direct WS)

Design choice — why POST to REST instead of direct WS write?
  The backend already owns alert deduplication, persistence, and
  notification dispatch. Posting via REST means alerts are stored in
  the DB and the backend's existing WS broadcaster fires them out to
  all dashboard clients automatically.

  The ai-service only needs a direct WS connection for its own
  camera-status heartbeat (a lightweight real-time signal the backend
  doesn't need to persist).
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)


class APIClient:
    """
    Shared aiohttp session for all backend calls.

    One instance is created at startup and shared across all coroutines.
    Uses a connection pool under the hood — safe for concurrent use.
    """

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        # Track recent alert IDs to skip duplicates within a short window
        self._recent_alerts: dict[str, float] = {}
        self._dedup_window_sec = 30.0

    async def start(self) -> None:
        """Create the shared aiohttp session. Call once at startup."""
        timeout = aiohttp.ClientTimeout(total=5)
        self._session = aiohttp.ClientSession(
            base_url=settings.BACKEND_URL,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        logger.info(f"APIClient ready → {settings.BACKEND_URL}")

    async def stop(self) -> None:
        """Close the session gracefully. Call at shutdown."""
        if self._session:
            await self._session.close()

    # ── Activity logging ──────────────────────────────────────────────────────

    async def post_activity(
        self,
        camera_id:       str,
        zone:            str,
        person_id:       str,
        activity_type:   str,
        description:     str,
        anomaly_label:   str,
        dwell_seconds:   int,
        confidence:      float,
        objects_detected: list | None = None,
        backend_used:    str = "",
        latency_ms:      int = 0,
    ) -> None:
        """
        POST a new activity record to /api/v1/activities/ingest.

        The backend stores this and it appears in the Activity Log page.
        In mock mode the backend stores it in memory (the mock list).
        """
        payload = {
            "id":              str(uuid.uuid4()),
            "person_id":       person_id,
            "camera_id":       camera_id,
            "zone":            zone,
            "activity_type":   activity_type,
            "description":     description,
            "anomaly_label":   anomaly_label,
            "dwell_seconds":   dwell_seconds,
            "confidence":      confidence,
            "objects_detected": objects_detected or [],
            "backend_used":    backend_used,
            "latency_ms":      latency_ms,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        await self._post("/api/v1/activities/ingest", payload)

    # ── Alert generation ──────────────────────────────────────────────────────

    async def post_alert(
        self,
        camera_id:    str,
        zone:         str,
        alert_type:   str,
        severity:     str,
        description:  str,
        person_id:    str,
        confidence:   float,
        snapshot_b64: Optional[str] = None,
        source:       str = "rules_engine",
    ) -> None:
        """
        POST a new alert to /api/v1/alerts/ingest.

        Includes deduplication: if the same (camera_id, alert_type)
        combination was seen within the last 30 seconds, skip.

        Args:
            source: Origin of the alert — "rules_engine" | "activity_analyzer" | "manual_test" | "other"
        """
        dedup_key = f"{camera_id}:{alert_type}"
        now = time.monotonic()

        # Prune expired dedup entries
        self._recent_alerts = {
            k: v for k, v in self._recent_alerts.items()
            if now - v < self._dedup_window_sec
        }

        if dedup_key in self._recent_alerts:
            logger.debug(f"Skipping duplicate alert: {dedup_key}")
            return

        self._recent_alerts[dedup_key] = now

        payload = {
            "id":           str(uuid.uuid4()),
            "camera_id":    camera_id,
            "zone":         zone,
            "alert_type":   alert_type,
            "severity":     severity,
            "description":  description,
            "person_id":    person_id,
            "confidence":   confidence,
            "status":       "active",
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_b64": snapshot_b64,
            "source":       source,
        }
        await self._post("/api/v1/alerts/ingest", payload)
        logger.info(f"🚨 Alert posted: [{severity.upper()}] {alert_type} @ {camera_id} ({zone}) source={source}")

    # ── Camera status heartbeat ───────────────────────────────────────────────

    async def post_camera_status(
        self,
        camera_id:  str,
        status:     str,
        fps:        float,
        latency_ms: int,
    ) -> None:
        """
        PATCH camera status to /api/v1/cameras/{id}/status.
        Called periodically by the stream manager heartbeat.
        """
        payload = {"status": status, "fps": round(fps, 1), "latency_ms": latency_ms}
        await self._patch(f"/api/v1/cameras/{camera_id}/status", payload)

    # ── WebSocket frame broadcast ─────────────────────────────────────────────

    async def broadcast_frame_update(
        self,
        camera_id:  str,
        persons:    list[dict],
    ) -> None:
        """
        POST a frame_update event so the backend WS hub broadcasts it.

        Accepts a list of person dicts so all detected persons in a
        single frame are sent in one broadcast. Each person dict should
        contain at minimum: person_id, zone, activity, dwell_seconds,
        and optionally bbox ([x1,y1,x2,y2]) and center ([cx,cy]).
        """
        logger.info(
            f"🔍 TRACE[api-client] SENDING frame_update camera={camera_id} | "
            f"persons={len(persons)} | "
            f"ids=[{', '.join(p['person_id'] for p in persons)}]"
        )
        payload = {
            "type":      "frame_update",
            "camera_id": camera_id,
            "persons":   persons,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._post("/api/v1/events/broadcast", payload)

    # ── VLM insight logging ──────────────────────────────────────────────────

    async def post_vlm_insight(
        self,
        camera_id:        str,
        zone:             str,
        person_id:        str,
        activity_type:    str,
        description:      str,
        anomaly_label:    str,
        confidence:       float,
        objects_detected: list | None = None,
        backend_used:     str = "",
        latency_ms:       int = 0,
        source:           str = "vlm",
    ) -> None:
        """
        POST a new VLM insight to /api/v1/vlm-insights/ingest.

        VLM insights are stored separately from Activity records to avoid
        duplication and overwriting issues. The AI Insights page reads from
        this endpoint; the Activity Log page reads from /activities/ingest.
        """
        payload = {
            "id":               str(uuid.uuid4()),
            "person_id":        person_id,
            "camera_id":        camera_id,
            "zone":             zone,
            "activity_type":    activity_type,
            "description":      description,
            "anomaly_label":    anomaly_label,
            "confidence":       confidence,
            "objects_detected": objects_detected or [],
            "backend_used":     backend_used,
            "latency_ms":       latency_ms,
            "source":           source,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }
        await self._post("/api/v1/vlm-insights/ingest", payload)

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    async def _post(self, path: str, payload: dict) -> None:
        if not self._session:
            logger.warning("APIClient not started — skipping POST")
            return
        try:
            async with self._session.post(path, json=payload) as resp:
                if resp.status not in (200, 201, 204):
                    body = await resp.text()
                    logger.warning(f"POST {path} → {resp.status}: {body[:200]}")
        except aiohttp.ClientConnectorError:
            logger.debug(f"Backend unreachable, skipping POST to {path}")
        except Exception as e:
            logger.debug(f"POST {path} error: {e!r}")

    async def _patch(self, path: str, payload: dict) -> None:
        if not self._session:
            return
        try:
            async with self._session.patch(path, json=payload) as resp:
                if resp.status not in (200, 204):
                    logger.debug(f"PATCH {path} → {resp.status}")
        except Exception:
            pass   # Silently skip — status updates are best-effort
