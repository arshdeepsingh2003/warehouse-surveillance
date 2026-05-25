# Warehouse AI Surveillance — Backend

FastAPI backend for the Warehouse Anomalous Activity Detection System.
Serves REST APIs for cameras, alerts, activities, and analytics,
plus a WebSocket endpoint for real-time dashboard updates.

---

## Quick start (3 steps)

### Step 1 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the server

```bash
uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**

| URL | Purpose |
|-----|---------|
| http://localhost:8000/docs | Interactive Swagger UI (test all APIs here) |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:8000/health | Health check |
| ws://localhost:8000/ws | WebSocket for real-time events |

---

## Test the APIs

Open http://localhost:8000/docs in your browser and click **Try it out** on any endpoint.

Or use curl:

```bash
# List all cameras
curl http://localhost:8000/api/v1/cameras

# Filter cameras by status
curl "http://localhost:8000/api/v1/cameras?status=online"

# Get active alerts
curl "http://localhost:8000/api/v1/alerts?status=active"

# Get high-severity alerts only
curl "http://localhost:8000/api/v1/alerts?severity=high"

# Activity log
curl http://localhost:8000/api/v1/activities

# Person timeline
curl http://localhost:8000/api/v1/activities/persons/P-1025/timeline

# Analytics summary
curl http://localhost:8000/api/v1/analytics/summary

# Resolve an alert
curl -X PATCH http://localhost:8000/api/v1/alerts/alert-001/resolve \
  -H "Content-Type: application/json" \
  -d '{"resolved_by": "admin@warehouse.com"}'
```

## Test WebSocket (real-time alerts)

Open your browser console and run:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onopen    = () => console.log("Connected!");
ws.onmessage = (e) => console.log("Event:", JSON.parse(e.data));
ws.onclose   = () => console.log("Disconnected.");
```

You will see events every few seconds:
- `frame_update`    — person tracking data per camera
- `alert_triggered` — new anomaly alert (every ~20 s)
- `camera_status`   — camera online/offline change
- `ping`            — keep-alive heartbeat

---

## Project structure

```
warehouse-backend/
├── app/
│   ├── main.py                  ← FastAPI app, CORS, routers, startup
│   ├── core/
│   │   └── config.py            ← Settings from .env (Pydantic Settings)
│   ├── api/
│   │   └── routes/
│   │       ├── cameras.py       ← GET/POST /cameras
│   │       ├── alerts.py        ← GET/PATCH /alerts
│   │       ├── activities.py    ← GET /activities + person timeline
│   │       ├── analytics.py     ← GET /analytics/summary
│   │       └── websocket.py     ← WS /ws + mock broadcaster
│   ├── schemas/
│   │   ├── camera.py            ← CameraOut, CameraCreate (Pydantic)
│   │   ├── alert.py             ← AlertOut, AlertResolve
│   │   ├── activity.py          ← ActivityOut, PersonTimeline
│   │   └── analytics.py        ← AnalyticsSummary
│   ├── services/
│   │   ├── mock_data.py         ← All demo data (cameras, alerts, activities)
│   │   ├── camera_service.py    ← Business logic for cameras
│   │   ├── alert_service.py     ← Business logic for alerts
│   │   ├── activity_service.py  ← Business logic for activities
│   │   └── analytics_service.py ← Business logic for analytics
│   └── ws/
│       └── connection_manager.py ← WebSocket room manager
├── .env                          ← Environment variables
└── requirements.txt
```

---

## What's next

| Step | Task |
|------|------|
| Frontend | Connect the React dashboard to these APIs |
| Mock RTSP | Add FFmpeg loop streams as camera sources |
| Real database | Uncomment asyncpg + SQLAlchemy in requirements.txt, create migrations |
| AI pipeline | Replace mock_ai_service with real YOLOv8 + VLM pipeline |
| Auth | Add JWT middleware to protect routes |
| Docker | Use the provided docker-compose.yml to run all services together |

Set `USE_MOCK_DATA=false` in `.env` when you're ready to connect a real database.