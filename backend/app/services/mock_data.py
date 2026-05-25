from datetime import datetime, timedelta
from uuid import uuid4

now = datetime.utcnow()

cameras = [
    {"id": "cam-001", "name": "North Entrance", "rtsp_url": "rtsp://camera-north.local/stream", "zone_id": "zone-a", "status": "online", "created_at": now - timedelta(days=30)},
    {"id": "cam-002", "name": "South Loading Bay", "rtsp_url": "rtsp://camera-south.local/stream", "zone_id": "zone-b", "status": "online", "created_at": now - timedelta(days=30)},
    {"id": "cam-003", "name": "East Aisle", "rtsp_url": "rtsp://camera-east.local/stream", "zone_id": "zone-a", "status": "offline", "created_at": now - timedelta(days=30)},
    {"id": "cam-004", "name": "West Storage", "rtsp_url": "rtsp://camera-west.local/stream", "zone_id": "zone-c", "status": "online", "created_at": now - timedelta(days=30)},
    {"id": "cam-005", "name": "Main Floor Overview", "rtsp_url": "rtsp://camera-main.local/stream", "zone_id": "zone-a", "status": "online", "created_at": now - timedelta(days=30)},
]

alerts = [
    {"id": "alert-001", "camera_id": "cam-001", "camera_name": "North Entrance", "type": "unauthorized_entry", "severity": "high", "description": "Unauthorized person detected at north entrance after hours", "timestamp": now - timedelta(minutes=5), "acknowledged": False},
    {"id": "alert-002", "camera_id": "cam-002", "camera_name": "South Loading Bay", "type": "loitering", "severity": "medium", "description": "Person loitering near loading bay for over 5 minutes", "timestamp": now - timedelta(minutes=12), "acknowledged": False},
    {"id": "alert-003", "camera_id": "cam-004", "camera_name": "West Storage", "type": "restricted_access", "severity": "critical", "description": "Unauthorized access to restricted storage zone", "timestamp": now - timedelta(hours=1), "acknowledged": True},
    {"id": "alert-004", "camera_id": "cam-005", "camera_name": "Main Floor Overview", "type": "crowd_gathering", "severity": "low", "description": "Unusual crowd gathering detected on main floor", "timestamp": now - timedelta(minutes=30), "acknowledged": False},
    {"id": "alert-005", "camera_id": "cam-002", "camera_name": "South Loading Bay", "type": "equipment_misuse", "severity": "medium", "description": "Forklift operating outside designated hours", "timestamp": now - timedelta(hours=2), "acknowledged": True},
]

activities = [
    {"id": "act-001", "camera_id": "cam-001", "camera_name": "North Entrance", "person_id": "person-001", "activity_type": "walking", "timestamp": now - timedelta(minutes=5), "description": "Person entering through north entrance", "metadata": {"confidence": 0.95}},
    {"id": "act-002", "camera_id": "cam-002", "camera_name": "South Loading Bay", "person_id": "person-002", "activity_type": "loitering", "timestamp": now - timedelta(minutes=12), "description": "Person standing near loading dock", "metadata": {"confidence": 0.88, "duration_seconds": 320}},
    {"id": "act-003", "camera_id": "cam-004", "camera_name": "West Storage", "person_id": "person-003", "activity_type": "running", "timestamp": now - timedelta(hours=1), "description": "Person running through west storage aisle", "metadata": {"confidence": 0.92}},
    {"id": "act-004", "camera_id": "cam-005", "camera_name": "Main Floor Overview", "person_id": None, "activity_type": "crowd", "timestamp": now - timedelta(minutes=30), "description": "Group of 8+ people gathered near main floor center", "metadata": {"confidence": 0.78, "estimated_count": 10}},
    {"id": "act-005", "camera_id": "cam-002", "camera_name": "South Loading Bay", "person_id": "person-002", "activity_type": "operating_equipment", "timestamp": now - timedelta(hours=2), "description": "Person operating forklift in loading bay", "metadata": {"confidence": 0.91, "equipment": "forklift"}},
    {"id": "act-006", "camera_id": "cam-001", "camera_name": "North Entrance", "person_id": "person-001", "activity_type": "exiting", "timestamp": now - timedelta(minutes=2), "description": "Person exiting through north entrance", "metadata": {"confidence": 0.94}},
    {"id": "act-007", "camera_id": "cam-005", "camera_name": "Main Floor Overview", "person_id": "person-004", "activity_type": "carrying_item", "timestamp": now - timedelta(minutes=45), "description": "Person carrying large box across main floor", "metadata": {"confidence": 0.87, "item": "large_box"}},
]
