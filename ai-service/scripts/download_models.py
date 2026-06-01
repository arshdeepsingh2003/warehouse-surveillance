#!/usr/bin/env python3
"""
scripts/download_models.py
───────────────────────────
Downloads the YOLOv8n model for person detection.

Run once before starting the AI service:
  python scripts/download_models.py

What this downloads:
  models/yolov8n.pt   —  6 MB  YOLOv8 Nano (fast CPU inference)

Model selection guide:
  yolov8n.pt   6 MB   ~45fps on CPU  Good for: development, low-end servers
  yolov8s.pt  22 MB   ~25fps on CPU  Good for: production CPU deployment
  yolov8m.pt  52 MB   ~12fps on CPU  Good for: GPU servers, high accuracy
  yolov8l.pt  87 MB   ~8fps  on CPU  Good for: maximum accuracy with GPU

For warehouse surveillance: yolov8n.pt is the recommended starting point.
Switch to yolov8s.pt when you need better detection in crowded scenes.
"""

import os
import sys
import urllib.request
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DOWNLOAD_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
}


def progress(blocks: int, block_size: int, total: int) -> None:
    pct  = min(blocks * block_size / total * 100, 100)
    done = int(pct / 5)
    bar  = "█" * done + "░" * (20 - done)
    print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)


def download_model(name: str = "yolov8n.pt") -> Path:
    dest = MODELS_DIR / name
    url  = DOWNLOAD_URLS.get(name)

    if not url:
        print(f"✗ Unknown model: {name}")
        print(f"  Available: {list(DOWNLOAD_URLS.keys())}")
        sys.exit(1)

    if dest.exists():
        size_mb = dest.stat().st_size / 1_048_576
        print(f"✓ {name} already exists ({size_mb:.1f} MB) — skipping download")
        return dest

    print(f"Downloading {name} from GitHub releases...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    try:
        urllib.request.urlretrieve(url, dest, progress)
        print()   # newline after progress bar
        size_mb = dest.stat().st_size / 1_048_576
        print(f"✓ Downloaded: {dest} ({size_mb:.1f} MB)")
        return dest

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print()
        print("Manual download options:")
        print(f"  1. wget {url} -O models/{name}")
        print(f"  2. pip install ultralytics && python -c \"from ultralytics import YOLO; YOLO('{name}')\"")
        print(f"     Then copy from ~/.cache/ultralytics/ to models/")
        print()
        print("Fallback: the system will use OpenCV HOG detector (no model needed)")
        print("  Set DETECTOR_BACKEND=hog in .env")
        return None


def verify_model(path: Path) -> bool:
    """Quick sanity check that the .pt file is a valid PyTorch model."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        print(f"✓ Model verified: {path.name}")
        return True
    except Exception as e:
        print(f"⚠ Model verification failed: {e}")
        print("  The file may be incomplete. Try downloading again.")
        return False


def test_inference() -> None:
    """Quick test: run detection on a dummy frame."""
    model_path = MODELS_DIR / "yolov8n.pt"
    if not model_path.exists():
        print("⚠ Skipping inference test — model not downloaded.")
        return

    print("\nRunning inference test on dummy frame...")
    try:
        import numpy as np
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ai.detector.person_detector import YOLODetector

        det   = YOLODetector(str(model_path), device="cpu")
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        dets  = det.detect(dummy)
        print(f"✓ Inference OK — {len(dets)} detections on blank frame (expected: 0)")
    except Exception as e:
        print(f"⚠ Inference test error: {e}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"

    print("=" * 50)
    print("  Warehouse AI — Model Setup")
    print("=" * 50 + "\n")

    path = download_model(model_name)

    if path and path.exists():
        verify_model(path)
        test_inference()

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print()
    print("  Start the AI service:")
    print("    python main.py")
    print()
    print("  Or with HOG fallback (no model needed):")
    print("    DETECTOR_BACKEND=hog python main.py")
    print("=" * 50)