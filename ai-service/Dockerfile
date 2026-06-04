FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download YOLO models at build time (cached in Docker layer)
# Primary: yolov8s (better accuracy); fallback: yolov8n
RUN python scripts/download_models.py yolov8s.pt || echo "yolov8s download failed"
RUN python scripts/download_models.py yolov8n.pt || echo "yolov8n download failed - will use HOG fallback"

EXPOSE 8001
CMD ["python", "main.py"]
