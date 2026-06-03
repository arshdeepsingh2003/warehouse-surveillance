# Warehouse AI Surveillance System

An end-to-end warehouse anomaly detection system with a FastAPI backend, React dashboard, and AI-powered video analysis pipeline.

---

## Architecture

```
frontend/  (React + Vite + TypeScript)   :5173
    │
    ├── REST API ──→ backend/  (FastAPI)  :8000
    │                     │
    │                     ├── PostgreSQL  (future)
    │                     └── Redis       (future)
    │
    └── MJPEG streams ──→ ai-service/     :8001
                             │
                             ├── mp4 files or RTSP cameras
                             └── YOLO + VLM pipeline  (future)
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | Required for backend & AI service |
| Node.js | 18+ | Required for frontend |
| npm | 9+ | Comes with Node.js |
| FFmpeg | 4.4+ | For mock video generation & RTSP |
| Git | — | Version control |
| Docker (optional) | 24+ | Run all services in containers |

Check your installations:

```bash
python --version
node --version
npm --version
ffmpeg -version
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd warehouse-surveillance
```

---

### 2. Backend (FastAPI)

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Environment variables are pre-configured in .env
# Review and adjust if needed:
#   USE_MOCK_DATA=true    — uses demo data (no database needed)
#   CORS_ORIGINS          — add your frontend URL if different

# Start the server
uvicorn app.main:app --reload
```

The backend starts at **http://localhost:8000**.

| URL | Purpose |
|-----|---------|
| http://localhost:8000/docs | Swagger UI (test APIs) |
| http://localhost:8000/health | Health check |
| ws://localhost:8000/ws | WebSocket for real-time events |

> **Note:** `.env` is included in `.gitignore`. Copy `backend/.env` as a template and create your own.

---

### 3. Frontend (React Dashboard)

Open a **new terminal**.

```bash
cd frontend

# Install dependencies
npm install

# Environment variables are pre-configured in .env
# VITE_STREAM_URL=http://localhost:8001  — points to AI service

# Start dev server
npm run dev
```

The frontend starts at **http://localhost:5173**.

Other commands:
```bash
npm run build    # Production build → frontend/dist/
npm run lint     # ESLint check
npm run preview  # Preview production build
```

---

### 4. AI Service (Video Streams & Analysis)

Open a **new terminal**.

```bash
cd ai-service

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 4a. Generate mock video sources (first time only)

The AI service reads from mp4 files. If the `mock_sources/` directory is empty, generate them:

```bash
# Windows — use the script (requires FFmpeg in PATH)
python scripts/generate_mock_videos.py

# Or manually with FFmpeg (cross-platform):
ffmpeg -f lavfi -i "color=c=gray:s=640x360:d=60" -c:v libx264 -pix_fmt yuv420p mock_sources/camera_01.mp4
# Repeat for camera_02.mp4 through camera_06.mp4

# macOS / Linux:
bash scripts/generate_mock_videos.sh
```

#### 4b. Start the AI service

```bash
python main.py
```

Expected output:
```
18:42:01 | INFO  | Registered 6 cameras
18:42:01 | INFO  | Stream loops started
18:42:01 | INFO  | MJPEG stream server starting on http://0.0.0.0:8001
```

| URL | Purpose |
|-----|---------|
| http://localhost:8001/stream/cam-01 | MJPEG stream for camera 1 |
| http://localhost:8001/snapshot/cam-01 | Single frame snapshot |
| http://localhost:8001/status | All camera statuses |

---

### 5. Verify Everything is Running

| Service | URL | Status |
|---------|-----|--------|
| Backend | http://localhost:8000/health | `{"status": "ok"}` |
| Frontend | http://localhost:5173 | Dashboard UI loads |
| AI Service | http://localhost:8001/status | Camera list with statuses |

Open the dashboard at **http://localhost:5173**, click **Live Feed** in the sidebar, and you should see video from all 6 cameras.

---

## Docker Setup (Alternative)

Run all services at once without manual setup:

```bash
cd docker
docker compose up -d
```

This starts PostgreSQL, Redis, backend, AI service, frontend, and Nginx reverse proxy.

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| AI Streams | http://localhost:8001 |

Stop with:
```bash
docker compose down
```

---

## Project Structure

```
warehouse-surveillance/
├── backend/               # FastAPI REST API
│   ├── app/
│   │   ├── main.py        # App entry, CORS, routers
│   │   ├── core/config.py # Settings from .env
│   │   ├── api/routes/    # cameras, alerts, activities, analytics, ws
│   │   ├── schemas/       # Pydantic models
│   │   ├── services/      # Business logic + mock data
│   │   └── ws/            # WebSocket connection manager
│   ├── .env               # Environment config
│   └── requirements.txt
│
├── frontend/              # React + TypeScript dashboard
│   ├── src/               # Components, pages, stores
│   ├── public/            # Static assets
│   ├── .env               # VITE_STREAM_URL
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
│
├── ai-service/            # Video pipeline & stream server
│   ├── main.py            # Entry point
│   ├── streams/           # FrameReader, StreamManager, MJPEG server
│   ├── pipeline/          # FrameProcessor, API client
│   ├── config/settings.py # Typed settings
│   ├── utils/             # Heartbeat, FFmpeg helper
│   ├── scripts/           # Mock video generation, start scripts
│   ├── mock_sources/      # camera_01.mp4 – camera_06.mp4
│   ├── .env               # AI service config
│   └── requirements.txt
│
├── docker/                # Docker Compose
│   ├── docker-compose.yml # All services + NGINX
│   └── nginx.conf
│
├── k8s/                   # Kubernetes manifests
│   ├── deployments/
│   ├── services/
│   ├── configmaps/
│   └── ingress/
│
└── Readme.MD              # This file
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_DATA` | `true` | Use demo data (no DB needed) |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:5173` | CORS origins |
| `DEBUG` | `true` | Debug mode |
| `DATABASE_URL` | — | PostgreSQL connection string (future) |

### AI Service (`ai-service/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | Backend REST API |
| `STREAM_SERVER_PORT` | `8001` | MJPEG stream port |
| `USE_MOCK_SOURCES` | `true` | Use mp4 files vs RTSP |
| `ACTIVITY_BACKEND` | `rules` | Activity detection: `rules` / `groq` / `hybrid` / `mock` |
| `GROQ_API_KEY` | — | Required for `groq` / `hybrid` backends |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_STREAM_URL` | `http://localhost:8001` | AI service URL for MJPEG streams |

---

## Running Different Parts Independently

### Backend + Frontend only (no video)

The backend works standalone with mock data. The frontend will load but live feeds won't stream.

### AI Service + Backend only (no dashboard)

Streams are served at `http://localhost:8001/stream/cam-*` and events are pushed to the backend.

### AI Service standalone (no backend, no dashboard)

Streams still work at `http://localhost:8001/stream/cam-*`; no alerts/activities are logged.

---

## Common Tasks

### Reset mock data

Restart the backend — it regenerates demo data on startup.

### Add a new camera

Update `backend/app/services/mock_data.py` and `ai-service/streams/stream_manager.py`.

### Switch to real RTSP cameras

In `ai-service/.env`:
```
USE_MOCK_SOURCES=false
RTSP_URL_CAM01=rtsp://admin:password@192.168.1.101:554/stream1
```

### Enable real AI pipeline

Uncomment the relevant lines in `ai-service/requirements.txt` and install:
```
ultralytics==8.2.0
torch==2.3.0
deep-sort-realtime==1.3.2
```

Then update `ai-service/pipeline/frame_processor.py` — search for the `# FUTURE (real AI)` comment.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip install` fails | Upgrade pip: `python -m pip install --upgrade pip` |
| Port 8000 in use | Change port: `uvicorn app.main:app --reload --port 8001` |
| Frontend can't reach backend | Check `CORS_ORIGINS` in `backend/.env` includes the frontend URL |
| No video in dashboard | Ensure AI service is running and `VITE_STREAM_URL` is correct |
| Mock videos missing | Run the generate script (step 4a) |
| Docker permission denied | Add your user to the `docker` group or use `sudo` |
| WebSocket not connecting | Ensure backend is running and accessible at `ws://localhost:8000/ws` |

---

## What's Next

| Step | Task |
|------|------|
| Frontend | Connect the React dashboard to these APIs |
| Mock RTSP | Add FFmpeg loop streams as camera sources |
| Real database | Uncomment asyncpg + SQLAlchemy in requirements.txt, create migrations |
| AI pipeline | Replace mock_ai_service with real YOLOv8 + VLM pipeline |
| Auth | Add JWT middleware to protect routes |
| Docker | Use the provided docker-compose.yml to run all services together |

Set `USE_MOCK_DATA=false` in `backend/.env` when you're ready to connect a real database.
