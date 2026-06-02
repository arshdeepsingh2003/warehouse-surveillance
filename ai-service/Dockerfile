FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download YOLOv8n model at build time (cached in Docker layer)
RUN python scripts/download_models.py || echo "Model download failed - will use HOG fallback"

EXPOSE 8001
CMD ["python", "main.py"]
