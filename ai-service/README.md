# Warehouse AI Service — Camera Stream Manager

This service reads video from mp4 files (or real RTSP cameras), runs mock AI
analysis, and serves live MJPEG feeds to the React dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│                                                                  │
│  camera_01.mp4 ──┐                                              │
│  camera_02.mp4 ──┤                                              │
│  camera_03.mp4 ──┼──→  StreamManager  ──→  asyncio.Queue       │
│  camera_04.mp4 ──┤        (one FrameReader per camera)          │
│  camera_05.mp4 ──┤        runs in ThreadPoolExecutor            │
│  camera_06.mp4 ──┘                                              │
│                   OR                                             │
│  rtsp://cam-01 ──┐                                              │
│  rtsp://cam-02 ──┤──→  Same StreamManager                       │
│  (real cameras)  │      zero code change needed                 │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼  (FrameData: camera_id, frame, timestamp, ...)
┌──────────────────────────────────────────────────────────────────┐
│                        AI CORE LAYER                             │
│                                                                  │
│  FrameProcessor.process(frame_data)                             │
│    │                                                             │
│    ├── encode JPEG  ──→  latest_frames[cam_id]   (MJPEG store) │
│    │                                                             │
│    └── buffer frames ──→ when batch full:                       │
│           │                                                      │
│           ├── [MOCK]  random activity + anomaly decision        │
│           │                                                      │
│           └── [FUTURE REAL AI]                                  │
│                  YOLOv8 detection                               │
│                  DeepSORT tracking                              │
│                  Person crop → VLM query                        │
│                  LLM + rules engine → anomaly decision          │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
│                                                                  │
│  1. POST /api/v1/activities/ingest  ──→  FastAPI backend        │
│     (activity logged, WS broadcast fires to dashboard)          │
│                                                                  │
│  2. POST /api/v1/alerts/ingest      ──→  FastAPI backend        │
│     (alert stored, WS fires to Alert Panel)                     │
│                                                                  │
│  3. GET  /stream/{camera_id}        ──→  React dashboard        │
│     (MJPEG frames served directly)                              │
│                                                                  │
│  4. Heartbeat PATCH /cameras/{id}/status  every 10 s            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Folder structure

```
warehouse-ai-service/
├── main.py                      ← Entry point: wires all components
├── .env                         ← Configuration (port, FPS, paths)
├── requirements.txt
│
├── config/
│   └── settings.py              ← Typed settings (Pydantic Settings)
│
├── streams/
│   ├── frame_reader.py          ← OpenCV VideoCapture wrapper (file + RTSP)
│   ├── stream_manager.py        ← Multi-camera orchestrator
│   └── stream_server.py         ← MJPEG FastAPI server (:8001)
│
├── pipeline/
│   ├── frame_processor.py       ← Mock AI analysis + batch processing
│   └── api_client.py            ← Posts events to backend REST API
│
├── utils/
│   ├── heartbeat.py             ← Periodic camera status push
│   └── ffmpeg_helper.py         ← FFmpeg RTSP publisher + MediaMTX config
│
├── mock_sources/
│   ├── camera_01.mp4            ← Main Gate feed
│   ├── camera_02.mp4            ← Warehouse Aisle
│   ├── camera_03.mp4            ← Loading Zone
│   ├── camera_04.mp4            ← Storage Area
│   ├── camera_05.mp4            ← Restricted Area
│   └── camera_06.mp4            ← Packing Area
│
└── scripts/
    ├── generate_mock_videos.sh  ← Create mp4 files from scratch with FFmpeg
    └── start_all.sh             ← Start backend + AI service + frontend together
```

---

## Quick start

### Step 1 — Install dependencies

```bash
cd warehouse-ai-service
pip install -r requirements.txt
```

### Step 2 — Start the backend (separate terminal)

```bash
cd warehouse-backend
uvicorn app.main:app --reload
```

### Step 3 — Start the AI stream service

```bash
cd warehouse-ai-service
python main.py
```

You will see:
```
18:42:01 | INFO  | Registered 6 cameras
18:42:01 | INFO  | Stream loops started
18:42:01 | INFO  | MJPEG stream server starting on http://0.0.0.0:8001
18:42:01 | INFO  | http://localhost:8001/stream/cam-01
18:42:01 | INFO  | http://localhost:8001/stream/cam-02
...
```

### Step 4 — Open the dashboard

```bash
cd warehouse-dashboard
npm run dev
# Open http://localhost:5173
```

Click **Live Feed** in the sidebar — you will see real video from the mp4 files.

---

## Test streams without the dashboard

```bash
# Open in any browser:
http://localhost:8001/stream/cam-01

# Snapshot (single frame):
http://localhost:8001/snapshot/cam-01

# All camera statuses:
http://localhost:8001/status

# Test with ffplay:
ffplay http://localhost:8001/stream/cam-01

# Test with VLC:
vlc http://localhost:8001/stream/cam-01
```

---

## Simulate RTSP streams (optional — matches real cameras exactly)

This is the closest simulation to real CCTV cameras.

### Step 1 — Download MediaMTX (free RTSP server)

```bash
# Linux
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.8.2_linux_amd64.tar.gz
tar xf mediamtx_*.tar.gz
chmod +x mediamtx
```

### Step 2 — Generate config and start server

```bash
cd warehouse-ai-service
python -c "from utils.ffmpeg_helper import write_mediamtx_config; write_mediamtx_config()"
./mediamtx mediamtx.yml
```

### Step 3 — Push each camera as RTSP (6 terminals)

```bash
# Terminal 1
ffmpeg -re -stream_loop -1 -i mock_sources/camera_01.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/cam-01

# Terminal 2
ffmpeg -re -stream_loop -1 -i mock_sources/camera_02.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/cam-02

# ... repeat for cam-03 through cam-06
# Or run: python -c "from utils.ffmpeg_helper import print_rtsp_commands; print_rtsp_commands()"
```

### Step 4 — Switch AI service to RTSP mode

In `.env`:
```
USE_MOCK_SOURCES=false
RTSP_URL_CAM01=rtsp://localhost:8554/cam-01
RTSP_URL_CAM02=rtsp://localhost:8554/cam-02
RTSP_URL_CAM03=rtsp://localhost:8554/cam-03
RTSP_URL_CAM04=rtsp://localhost:8554/cam-04
RTSP_URL_CAM05=rtsp://localhost:8554/cam-05
RTSP_URL_CAM06=rtsp://localhost:8554/cam-06
```

The FrameReader code doesn't change — it just opens a different URL.

---

## Replacing mock feeds with real cameras

When your real CCTV cameras are available:

**Step 1** — Find the RTSP URL for each camera.  
Most cameras use: `rtsp://admin:password@192.168.1.100:554/stream1`  
Check your camera's manual or admin panel.

**Step 2** — Update `.env`:
```
USE_MOCK_SOURCES=false
RTSP_URL_CAM01=rtsp://admin:password@192.168.1.101:554/stream1
RTSP_URL_CAM02=rtsp://admin:password@192.168.1.102:554/stream1
...
```

**Step 3** — Restart the AI service:
```bash
python main.py
```

That's it. Zero code changes needed.

---

## Where the real AI pipeline connects

The FrameProcessor (`pipeline/frame_processor.py`) has a clear comment showing exactly where to plug in real AI:

```python
async def _analyse_batch(self, cam_id, batch):
    # TODAY (mock):
    is_anomaly = random.random() < 0.05
    description = random.choice(NORMAL_ACTIVITIES)

    # FUTURE (real AI) — replace the above with:
    # key_frames = self._sample_key_frames(batch)
    # for frame_data in key_frames:
    #   boxes  = yolo_model(frame_data.frame)        # Step 1: YOLOv8
    #   tracks = deepsort.update(boxes, frame)        # Step 2: DeepSORT
    #   for track in tracks:
    #     crop        = frame[y1:y2, x1:x2]
    #     description = await vlm_query(crop)         # Step 3: VLM
    #     label, sev  = rules_engine(description)     # Step 4: Rules engine
    #     if label == "anomaly":
    #         await self._api.post_alert(...)
    #     await self._api.post_activity(...)
```

The rest of the system (streaming, API, dashboard) requires zero changes.

---

## Key design decisions

| Decision | Reason |
|---|---|
| MJPEG over HLS | Works instantly in `<img>` tag, no JS library needed |
| ThreadPoolExecutor for OpenCV | OpenCV is synchronous C++; running in threads keeps asyncio unblocked |
| asyncio.Queue per camera | Backpressure: AI can't keep up → frames are dropped (not queued forever) |
| Batch processing | VLM calls are expensive; batching amortises the cost |
| Deduplication in APIClient | Prevents alert storms for sustained events (e.g. person loitering) |
| Mock mode flag | Full dashboard testable without any cameras or AI models |

---

## Scalability path

| Stage | Cameras | Architecture |
|---|---|---|
| Development | 1–6 | Single Python process, mp4 files |
| Pilot | 6–20 | Single process, RTSP sources, GPU for YOLO |
| Production | 20–100 | Multiple AI workers (one per camera group), Kafka, Redis |
| Enterprise | 100+ | Kubernetes, GPU node pool, distributed stream processing |
