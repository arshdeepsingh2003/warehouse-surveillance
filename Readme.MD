# Warehouse AI Surveillance System

An end-to-end warehouse anomaly detection system with a FastAPI backend, React dashboard, and AI-powered video analysis pipeline.

---

## Table of Contents

- [Fresh Setup on a New Laptop](#-fresh-setup-on-a-new-laptop)
- [Architecture](#architecture)
- [Setup Instructions (Manual)](#setup-instructions-manual)
- [Docker Setup (Alternative)](#docker-setup-alternative)
- [Verify Everything is Running](#verify-everything-is-running)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Running Different Parts Independently](#running-different-parts-independently)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## 🖥️ Fresh Setup on a New Laptop

Use this section if you're setting up the project on a brand-new machine.

### Step 0 — Install Prerequisites

#### Git

| OS | Instructions |
|----|-------------|
| **Windows** | Download from https://git-scm.com/download/win — install with default options |
| **macOS** | `brew install git` or install Xcode Command Line Tools (`xcode-select --install`) |
| **Linux** | `sudo apt install git` (Debian/Ubuntu) or `sudo dnf install git` (Fedora) |

Verify:
```bash
git --version
```

#### Python 3.11+

| OS | Instructions |
|----|-------------|
| **Windows** | Download from https://www.python.org/downloads/ — **check "Add Python to PATH"** during install |
| **macOS** | `brew install python@3.11` |
| **Linux** | `sudo apt install python3.11 python3.11-venv python3-pip` |

Verify:
```bash
python --version
```
If `python` isn't found, try `python3`:
```bash
python3 --version
```

> **Windows PowerShell note:** If you get an execution policy error when activating the virtual environment, run this once:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

#### Node.js 18+ and npm

| OS | Instructions |
|----|-------------|
| **All** | Download from https://nodejs.org/ (LTS version recommended) |

Verify:
```bash
node --version
npm --version
```

#### FFmpeg 4.4+

| OS | Instructions |
|----|-------------|
| **Windows** | Download from https://www.gyan.dev/ffmpeg/builds/ (release essentials zip). Extract to `C:\ffmpeg` and **add `C:\ffmpeg\bin` to your PATH**. Or use `winget install FFmpeg` |
| **macOS** | `brew install ffmpeg` |
| **Linux** | `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora) |

Verify:
```bash
ffmpeg -version
```

> **Troubleshooting:** If any command shows "command not found", the program is either not installed or not in your PATH. Restart your terminal after installing.

---

### Step 1 — Clone the Repository

```bash
git clone <repo-url>
cd warehouse-surveillance
```

---

### Step 2 — Create Environment Files

> ⚠️ **Important:** The `.env` files are **not** committed to git (they contain local config). You must create them manually after cloning.

Create `backend/.env`:
```ini
APP_NAME="Warehouse AI Surveillance System"
APP_VERSION="1.0.0"
DEBUG=true
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
CORS_ORIGINS=*
```

Create `frontend/.env`:
```ini
VITE_STREAM_URL=http://localhost:8002
```

Create `ai-service/.env`:
```ini
BACKEND_URL=http://localhost:8000
BACKEND_WS_URL=ws://localhost:8000/ws
API_PREFIX=/api/v1

STREAM_SERVER_HOST=0.0.0.0
STREAM_SERVER_PORT=8002
STREAM_FPS=10
FRAME_WIDTH=640
FRAME_HEIGHT=360
JPEG_QUALITY=75

MOCK_VIDEO_DIR=./mock_sources
USE_MOCK_SOURCES=true
LOOP_VIDEO=true

DETECTOR_BACKEND=yolo
YOLO_MODEL_PATH=./models/yolov8s.pt
YOLO_CONFIDENCE=0.25
YOLO_IMG_SIZE=960
YOLO_USE_TTA=false
YOLO_USE_CLAHE=true
DEVICE=cpu
PROCESS_EVERY_N_FRAMES=2

ACTIVITY_BACKEND=hybrid
EVENT_VLM_RATE_LIMIT=0.5

# ── VLM Backend (choose one: mock | groq | openai | anthropic | ollama | gemini | qwen_vl | moondream)
VLM_BACKEND=groq
USE_VLM=true
USE_EVENT_DRIVEN_VLM=true
SAVE_DEBUG_CROPS=false

OLLAMA_HOST=http://localhost:11434
MOONDREAM_MODEL=moondream
QWEN_VL_MODEL=qwen2.5-vl
OLLAMA_MODEL=minicpm-v

LLM_BACKEND=mock
USE_LLM=false
SUMMARY_INTERVAL_SECONDS=30
LLM_MODEL=llama-3.3-70b-versatile

GROQ_VLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

---

### Step 3 — Backend (FastAPI)

```bash
cd backend

# Create Python virtual environment
python -m venv venv

# Activate it:
#   Windows (PowerShell):
venv\Scripts\Activate.ps1
#   Windows (cmd):
venv\Scripts\activate.bat
#   macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

The backend starts at **http://localhost:8000**.

| URL | Purpose |
|-----|---------|
| http://localhost:8000/docs | Swagger UI (test APIs) |
| http://localhost:8000/health | Health check |
| ws://localhost:8000/ws | WebSocket for real-time events |

> **Note:** The backend runs in mock mode by default (no database needed). Alerts and activities are populated in real-time by the AI service via the ingest API.

---

### Step 4 — Frontend (React Dashboard)

Open a **new terminal**.

```bash
cd frontend

# Install dependencies
npm install

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

### Step 5 — AI Service (Video Streams & Analysis)

Open a **new terminal**.

```bash
cd ai-service

# Create virtual environment
python -m venv venv

# Activate it:
#   Windows (PowerShell):
venv\Scripts\Activate.ps1
#   Windows (cmd):
venv\Scripts\activate.bat
#   macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 5a. Generate mock video sources (first time only)

The AI service reads from mp4 files. If the `mock_sources/` directory is empty, generate them:

```bash
# Windows — requires FFmpeg in PATH:
python scripts/generate_mock_videos.py

# macOS / Linux:
bash scripts/generate_mock_videos.sh
```

This creates 6 mp4 files (`camera_01.mp4` through `camera_06.mp4`) inside `ai-service/mock_sources/`.

> **Alternative — create videos manually:** If the script fails, run this FFmpeg command for each camera:
> ```bash
> ffmpeg -f lavfi -i "color=c=gray:s=640x360:d=60" -c:v libx264 -pix_fmt yuv420p mock_sources/camera_01.mp4
> ```
> Repeat for `camera_02.mp4` through `camera_06.mp4`.

#### 5b. Start the AI service

```bash
python main.py
```

Expected output:
```
18:42:01 | INFO  | Registered 6 cameras
18:42:01 | INFO  | Stream loops started
18:42:01 | INFO  | MJPEG stream server starting on http://0.0.0.0:8002
```

| URL | Purpose |
|-----|---------|
| http://localhost:8002/stream/cam-01 | MJPEG stream for camera 1 |
| http://localhost:8002/snapshot/cam-01 | Single frame snapshot |
| http://localhost:8002/status | All camera statuses |

---

### Step 6 — Verify Everything is Running

| Service | URL | Status |
|---------|-----|--------|
| Backend | http://localhost:8000/health | `{"status": "ok"}` |
| Frontend | http://localhost:5173 | Dashboard UI loads |
| AI Service | http://localhost:8002/status | Camera list with statuses |

Open the dashboard at **http://localhost:5173**, click **Live Feed** in the sidebar, and you should see video from all 6 cameras.

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
    └── MJPEG streams ──→ ai-service/     :8002
                             │
                             ├── mp4 files or RTSP cameras
                             └── YOLO + VLM pipeline
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

> See [Fresh Setup on a New Laptop](#-fresh-setup-on-a-new-laptop) above for detailed installation instructions for each prerequisite.

---

## Setup Instructions (Manual)

### 1. Clone the repository

```bash
git clone <repo-url>
cd warehouse-surveillance
```

---

### 2. Create Environment Files

`.env` files are not committed to git. Create them with the following content:

**`backend/.env`:**
```ini
APP_NAME="Warehouse AI Surveillance System"
APP_VERSION="1.0.0"
DEBUG=true
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
CORS_ORIGINS=*
```

**`frontend/.env`:**
```ini
VITE_STREAM_URL=http://localhost:8002
```

**`ai-service/.env`:**
```ini
BACKEND_URL=http://localhost:8000
BACKEND_WS_URL=ws://localhost:8000/ws
API_PREFIX=/api/v1

STREAM_SERVER_HOST=0.0.0.0
STREAM_SERVER_PORT=8002
STREAM_FPS=10
FRAME_WIDTH=640
FRAME_HEIGHT=360
JPEG_QUALITY=75

MOCK_VIDEO_DIR=./mock_sources
USE_MOCK_SOURCES=true
LOOP_VIDEO=true

DETECTOR_BACKEND=yolo
YOLO_MODEL_PATH=./models/yolov8s.pt
YOLO_CONFIDENCE=0.25
YOLO_IMG_SIZE=960
YOLO_USE_TTA=false
YOLO_USE_CLAHE=true
DEVICE=cpu
PROCESS_EVERY_N_FRAMES=2

ACTIVITY_BACKEND=hybrid
EVENT_VLM_RATE_LIMIT=0.5
VLM_BACKEND=groq
USE_VLM=true
USE_EVENT_DRIVEN_VLM=true
SAVE_DEBUG_CROPS=false

OLLAMA_HOST=http://localhost:11434
MOONDREAM_MODEL=moondream
QWEN_VL_MODEL=qwen2.5-vl
OLLAMA_MODEL=minicpm-v

LLM_BACKEND=mock
USE_LLM=false
SUMMARY_INTERVAL_SECONDS=30
LLM_MODEL=llama-3.3-70b-versatile
GROQ_VLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

---

### 3. Backend (FastAPI)

```bash
cd backend

# Create Python virtual environment
python -m venv venv

# Activate it:
#   Windows (PowerShell):
venv\Scripts\Activate.ps1
#   Windows (cmd):
venv\Scripts\activate.bat
#   macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

The backend starts at **http://localhost:8000**.

| URL | Purpose |
|-----|---------|
| http://localhost:8000/docs | Swagger UI (test APIs) |
| http://localhost:8000/health | Health check |
| ws://localhost:8000/ws | WebSocket for real-time events |

> **Note:** The backend runs in mock mode by default (`USE_MOCK_DATA=true`), so no database is needed.

---

### 4. Frontend (React Dashboard)

Open a **new terminal**.

```bash
cd frontend

# Install dependencies
npm install

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

### 5. AI Service (Video Streams & Analysis)

Open a **new terminal**.

```bash
cd ai-service

# Create virtual environment
python -m venv venv

# Activate it:
#   Windows (PowerShell):
venv\Scripts\Activate.ps1
#   Windows (cmd):
venv\Scripts\activate.bat
#   macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 5a. Generate mock video sources (first time only)

The AI service reads from mp4 files. If the `mock_sources/` directory is empty, generate them:

```bash
# Windows — requires FFmpeg in PATH:
python scripts/generate_mock_videos.py

# macOS / Linux:
bash scripts/generate_mock_videos.sh
```

> **Alternative:** If the script fails, create videos manually with FFmpeg:
> ```bash
> ffmpeg -f lavfi -i "color=c=gray:s=640x360:d=60" -c:v libx264 -pix_fmt yuv420p mock_sources/camera_01.mp4
> ```
> Repeat for `camera_02.mp4` through `camera_06.mp4`.

#### 5b. Start the AI service

```bash
python main.py
```

Expected output:
```
18:42:01 | INFO  | Registered 6 cameras
18:42:01 | INFO  | Stream loops started
18:42:01 | INFO  | MJPEG stream server starting on http://0.0.0.0:8002
```

| URL | Purpose |
|-----|---------|
| http://localhost:8002/stream/cam-01 | MJPEG stream for camera 1 |
| http://localhost:8002/snapshot/cam-01 | Single frame snapshot |
| http://localhost:8002/status | All camera statuses |

> Make sure `VITE_STREAM_URL` in `frontend/.env` matches the AI service's `STREAM_SERVER_PORT`.

---

### 6. Verify Everything is Running

| Service | URL | Status |
|---------|-----|--------|
| Backend | http://localhost:8000/health | `{"status": "ok"}` |
| Frontend | http://localhost:5173 | Dashboard UI loads |
| AI Service | http://localhost:8002/status | Camera list with statuses |

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
| AI Streams | http://localhost:8002 |

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
| `STREAM_SERVER_PORT` | `8002` | MJPEG stream port |
| `USE_MOCK_SOURCES` | `true` | Use mp4 files vs RTSP |
| `ACTIVITY_BACKEND` | `rules` | Activity detection: `rules` / `groq` / `hybrid` / `mock` |
| `GROQ_API_KEY` | — | Required for `groq` / `hybrid` backends |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_STREAM_URL` | `http://localhost:8002` | AI service URL for MJPEG streams |

> Make sure `VITE_STREAM_URL` matches the AI service's `STREAM_SERVER_PORT`.

---

## Running Different Parts Independently

### Backend + Frontend only (no video)

The backend works standalone with mock data. The frontend will load but live feeds won't stream.

### AI Service + Backend only (no dashboard)

Streams are served at `http://localhost:8002/stream/cam-*` and events are pushed to the backend.

### AI Service standalone (no backend, no dashboard)

Streams still work at `http://localhost:8002/stream/cam-*`; no alerts/activities are logged.

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
| `python` not found | Try `python3` instead. On Windows, reinstall Python and check **Add Python to PATH**. Restart your terminal. |
| `venv` module not found | Install it: `sudo apt install python3.11-venv` (Linux) or reinstall Python (Windows). |
| `pip install` fails | Upgrade pip: `python -m pip install --upgrade pip`. On Windows, you may need to run the terminal as Administrator. |
| `ffmpeg` not found | Install FFmpeg and add it to your PATH. Restart your terminal. |
| Port 8000 in use | Change port: `uvicorn app.main:app --reload --port 8001`, then update `frontend/.env` and `ai-service/.env` URLs. |
| Frontend can't reach backend | Check `ALLOWED_ORIGINS` in `backend/.env` includes the frontend URL. |
| No video in dashboard | Ensure AI service is running and `VITE_STREAM_URL` in `frontend/.env` matches the AI service's `STREAM_SERVER_PORT`. |
| Mock videos missing | Run the generate script (step 5a above). |
| Docker permission denied | Add your user to the `docker` group (`sudo usermod -aG docker $USER`) or use `sudo`. On Windows, ensure Docker Desktop is running. |
| WebSocket not connecting | Ensure backend is running and accessible at `ws://localhost:8000/ws`. |
| `npm install` fails | Delete `node_modules` and `package-lock.json`, then run `npm install` again. |
| Activate venv fails on Windows | If execution policy blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first. |
| "No cameras registered" error | Check `MOCK_VIDEO_DIR` in `ai-service/.env` points to the correct path, and mock videos exist. |
