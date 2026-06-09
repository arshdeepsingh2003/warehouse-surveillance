"""
services/mock_data.py
─────────────────────
Mock persistence layer.

When USE_MOCK_DATA=true (set in .env), every API route returns data from
the in-memory lists below instead of hitting a real database.

What stays mock:
  • MOCK_CAMERAS          — static camera definitions (keep for demo)
  • MOCK_ACTIVITIES (empty) — populated by POST /api/v1/activities/ingest
  • MOCK_ALERTS     (empty) — populated by POST /api/v1/alerts/ingest

What was removed:
  • Hardcoded MOCK_ALERTS     (was 5 seeded alerts)
  • Hardcoded MOCK_ACTIVITIES (was 7 seeded activities)
  • Hardcoded MOCK_ANALYTICS  (was static KPIs derived from seeded data)

All activities and alerts now originate exclusively from the real AI pipeline:
  FrameProcessor → ActivityAnalyzer → RulesEngine → api_client
  → POST /api/v1/activities|alerts/ingest → in-memory lists → dashboard
"""

from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


def _now() -> datetime:
    return datetime.now(UTC)


def _ago(minutes: int = 0, hours: int = 0) -> datetime:
    return _now() - timedelta(minutes=minutes, hours=hours)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERAS  (static — keep)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_CAMERAS = [
    {
        "id": "cam-01",
        "name": "Main Gate",
        "location": "Building Entrance",
        "zone": "entry_zone",
        "stream_url": "rtsp://mock/cam-01",
        "camera_type": "mock",
        "status": "online",
        "fps": 15,
        "latency_ms": 42,
        "created_at": _ago(hours=72),
        "updated_at": _ago(minutes=1),
    },
    {
        "id": "cam-02",
        "name": "Warehouse Aisle",
        "location": "Aisle B, Row 3",
        "zone": "storage_area",
        "stream_url": "rtsp://mock/cam-02",
        "camera_type": "mock",
        "status": "online",
        "fps": 12,
        "latency_ms": 65,
        "created_at": _ago(hours=72),
        "updated_at": _ago(minutes=2),
    },
    {
        "id": "cam-03",
        "name": "Loading Zone",
        "location": "Dock 1",
        "zone": "loading_zone",
        "stream_url": "rtsp://mock/cam-03",
        "camera_type": "mock",
        "status": "online",
        "fps": 15,
        "latency_ms": 38,
        "created_at": _ago(hours=72),
        "updated_at": _ago(minutes=1),
    },
    {
        "id": "cam-04",
        "name": "Storage Area",
        "location": "Rack Section C",
        "zone": "storage_area",
        "stream_url": "rtsp://mock/cam-04",
        "camera_type": "mock",
        "status": "online",
        "fps": 10,
        "latency_ms": 80,
        "created_at": _ago(hours=72),
        "updated_at": _ago(minutes=3),
    },
    {
        "id": "cam-05",
        "name": "Restricted Area",
        "location": "Server Room Corridor",
        "zone": "restricted_area",
        "stream_url": "rtsp://mock/cam-05",
        "camera_type": "mock",
        "status": "online",
        "fps": 15,
        "latency_ms": 50,
        "created_at": _ago(hours=72),
        "updated_at": _ago(minutes=1),
    },
    {
        "id": "cam-06",
        "name": "Packing Area",
        "location": "Dispatch Floor",
        "zone": "packing_area",
        "stream_url": "rtsp://mock/cam-06",
        "camera_type": "mock",
        "status": "offline",
        "fps": 0,
        "latency_ms": 0,
        "created_at": _ago(hours=72),
        "updated_at": _ago(minutes=30),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITIES  (populated by POST /api/v1/activities/ingest — starts empty)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_ACTIVITIES: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# ALERTS  (populated by POST /api/v1/alerts/ingest — starts empty)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_ALERTS: list[dict] = []
