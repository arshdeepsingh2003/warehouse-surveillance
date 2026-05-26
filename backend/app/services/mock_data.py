"""
All mock / demo data lives here.

When USE_MOCK_DATA=true (set in .env), every API route returns data from
this file instead of hitting a real database.

This lets you:
  • Build and test the entire dashboard with zero real cameras or AI.
  • Demo the product to stakeholders before production is ready.
  • Swap to real data by changing one flag.

Structure: each section returns a list/dict of schema-compatible dicts.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")

def _now() -> datetime:
    return datetime.now(UTC)

def _ago(minutes: int = 0, hours: int = 0) -> datetime: # Creates past timestamps
    return _now() - timedelta(minutes=minutes, hours=hours)

# CAMERAS

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


# ALERTS
MOCK_ALERTS = [
    {
        "id": "alert-001",
        "camera_id": "cam-05",
        "zone": "restricted_area",
        "alert_type": "unauthorized_access",
        "severity": "high",
        "description": "Unauthorized person detected entering restricted server room corridor without badge scan.",
        "person_id": "P-1025",
        "snapshot_url": "https://placehold.co/640x360?text=Alert+001",
        "status": "active",
        "confidence": 0.94,
        "triggered_at": _ago(minutes=5),
        "resolved_at": None,
        "resolved_by": None,
    },
    {
        "id": "alert-002",
        "camera_id": "cam-02",
        "zone": "storage_area",
        "alert_type": "worker_fall",
        "severity": "high",
        "description": "Worker fall detected near aisle shelf. Person remained stationary for over 30 seconds.",
        "person_id": "P-1031",
        "snapshot_url": "https://placehold.co/640x360?text=Alert+002",
        "status": "resolved",
        "confidence": 0.88,
        "triggered_at": _ago(minutes=22),
        "resolved_at": _ago(minutes=15),
        "resolved_by": "security@warehouse.com",
    },
    {
        "id": "alert-003",
        "camera_id": "cam-03",
        "zone": "loading_zone",
        "alert_type": "loitering",
        "severity": "medium",
        "description": "Individual loitering near loading dock for 18 minutes without performing any work activity.",
        "person_id": "P-1019",
        "snapshot_url": "https://placehold.co/640x360?text=Alert+003",
        "status": "active",
        "confidence": 0.76,
        "triggered_at": _ago(minutes=35),
        "resolved_at": None,
        "resolved_by": None,
    },
    {
        "id": "alert-004",
        "camera_id": "cam-06",
        "zone": "packing_area",
        "alert_type": "ppe_violation",
        "severity": "low",
        "description": "Worker detected without safety helmet in mandatory PPE zone near dispatch floor.",
        "person_id": "P-1044",
        "snapshot_url": "https://placehold.co/640x360?text=Alert+004",
        "status": "active",
        "confidence": 0.91,
        "triggered_at": _ago(minutes=41),
        "resolved_at": None,
        "resolved_by": None,
    },
    {
        "id": "alert-005",
        "camera_id": "cam-04",
        "zone": "storage_area",
        "alert_type": "unauthorized_access",
        "severity": "high",
        "description": "Unauthorized access detected at storage rack section C. Person entered outside working hours.",
        "person_id": "P-1055",
        "snapshot_url": "https://placehold.co/640x360?text=Alert+005",
        "status": "active",
        "confidence": 0.97,
        "triggered_at": _ago(minutes=55),
        "resolved_at": None,
        "resolved_by": None,
    },
]


# ACTIVITIES

MOCK_ACTIVITIES = [
    {
        "id": "act-001",
        "person_id": "P-1025",
        "camera_id": "cam-01",
        "zone": "entry_zone",
        "activity_type": "walking",
        "description": "Person entered main gate, badge scanned, walking toward aisle B.",
        "anomaly_label": "normal",
        "dwell_seconds": 15,
        "confidence": 0.95,
        "timestamp": _ago(hours=1, minutes=5),
    },
    {
        "id": "act-002",
        "person_id": "P-1025",
        "camera_id": "cam-02",
        "zone": "storage_area",
        "activity_type": "handling_items",
        "description": "Person is picking boxes from shelf and stacking them onto a pallet.",
        "anomaly_label": "normal",
        "dwell_seconds": 480,
        "confidence": 0.93,
        "timestamp": _ago(hours=1),
    },
    {
        "id": "act-003",
        "person_id": "P-1025",
        "camera_id": "cam-05",
        "zone": "restricted_area",
        "activity_type": "unauthorized_entry",
        "description": "Person entered restricted server room corridor without visible badge. Tailgated after authorized personnel.",
        "anomaly_label": "anomaly",
        "dwell_seconds": 120,
        "confidence": 0.94,
        "timestamp": _ago(minutes=5),
    },
    {
        "id": "act-004",
        "person_id": "P-1031",
        "camera_id": "cam-02",
        "zone": "storage_area",
        "activity_type": "falling",
        "description": "Worker suddenly fell near metal shelf. Body remained horizontal and stationary for 30+ seconds.",
        "anomaly_label": "anomaly",
        "dwell_seconds": 35,
        "confidence": 0.88,
        "timestamp": _ago(minutes=22),
    },
    {
        "id": "act-005",
        "person_id": "P-1019",
        "camera_id": "cam-03",
        "zone": "loading_zone",
        "activity_type": "loitering",
        "description": "Individual standing near dock door, occasionally looking around, no cargo interaction.",
        "anomaly_label": "anomaly",
        "dwell_seconds": 1080,
        "confidence": 0.76,
        "timestamp": _ago(minutes=35),
    },
    {
        "id": "act-006",
        "person_id": "P-1044",
        "camera_id": "cam-06",
        "zone": "packing_area",
        "activity_type": "walking",
        "description": "Worker walking along dispatch aisle without safety helmet. PPE violation detected.",
        "anomaly_label": "anomaly",
        "dwell_seconds": 60,
        "confidence": 0.91,
        "timestamp": _ago(minutes=41),
    },
    {
        "id": "act-007",
        "person_id": "P-1010",
        "camera_id": "cam-01",
        "zone": "entry_zone",
        "activity_type": "walking",
        "description": "Worker passed through main gate, scanning badge correctly, proceeding to loading zone.",
        "anomaly_label": "normal",
        "dwell_seconds": 10,
        "confidence": 0.98,
        "timestamp": _ago(minutes=50),
    },
]

# ANALYTICS

MOCK_ANALYTICS = {
    "summary": {
        "total_cameras": 6,
        "cameras_online": 5,
        "total_alerts_today": 24,
        "active_alerts": 4,
        "high_severity_alerts": 3,
        "people_detected": 32,
        "most_risky_zone": "Restricted Area",
        "peak_activity_hour": "14:00 – 15:00",
        "system_status": "healthy",
    },
    "alert_trend": [
        {"hour": "00:00", "count": 0},
        {"hour": "02:00", "count": 1},
        {"hour": "04:00", "count": 0},
        {"hour": "06:00", "count": 2},
        {"hour": "08:00", "count": 5},
        {"hour": "10:00", "count": 8},
        {"hour": "12:00", "count": 6},
        {"hour": "14:00", "count": 12},  # peak
        {"hour": "16:00", "count": 9},
        {"hour": "18:00", "count": 4},
        {"hour": "20:00", "count": 2},
        {"hour": "22:00", "count": 1},
    ],
    "zone_risk": [
        {"zone": "Restricted Area", "incidents": 24, "percentage": 40.0},
        {"zone": "Warehouse Aisle", "incidents": 15, "percentage": 25.0},
        {"zone": "Loading Zone",    "incidents": 9,  "percentage": 15.0},
        {"zone": "Storage Area",    "incidents": 6,  "percentage": 10.0},
        {"zone": "Packing Area",    "incidents": 6,  "percentage": 10.0},
    ],
}

