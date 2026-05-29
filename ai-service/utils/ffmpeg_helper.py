"""
utils/ffmpeg_helper.py
───────────────────────
FFmpeg helper utilities for generating fake RTSP streams from mp4 files.

This module provides:
  1. Python subprocess wrappers to launch FFmpeg RTSP publishers
  2. A MediaMTX config generator (open-source RTSP server)
  3. Human-readable command-line examples

──────────────────────────────────────────────────────────────────────
ARCHITECTURE: Mock RTSP pipeline
──────────────────────────────────────────────────────────────────────

  camera_01.mp4
       │
       │  FFmpeg (loops the file, re-encodes as RTSP)
       ▼
  MediaMTX (RTSP server)  ←── acts like a real IP camera
       │
       │  rtsp://localhost:8554/cam-01
       ▼
  FrameReader (OpenCV)    ←── reads RTSP as if it were a real camera
       │
       ▼
  FrameProcessor → backend API → dashboard

This is identical to how a real CCTV camera connects.
To switch to a real camera: just replace the RTSP URL in .env.

──────────────────────────────────────────────────────────────────────
QUICK START (two terminals):
──────────────────────────────────────────────────────────────────────

Terminal 1 — start MediaMTX (RTSP server):
  cd warehouse-ai-service
  ./bin/mediamtx mediamtx.yml

Terminal 2 — push one mp4 file as RTSP:
  ffmpeg -re -stream_loop -1 -i mock_sources/camera_01.mp4 \\
         -c copy -f rtsp rtsp://localhost:8554/cam-01

Or use the Python helper below to launch all 6 cameras at once.
"""

import asyncio
import logging
import os
import signal
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


# ── RTSP camera config ────────────────────────────────────────────────────────

RTSP_CAMERAS = [
    {"id": "cam-01", "file": "camera_01.mp4", "rtsp": "rtsp://localhost:8554/cam-01"},
    {"id": "cam-02", "file": "camera_02.mp4", "rtsp": "rtsp://localhost:8554/cam-02"},
    {"id": "cam-03", "file": "camera_03.mp4", "rtsp": "rtsp://localhost:8554/cam-03"},
    {"id": "cam-04", "file": "camera_04.mp4", "rtsp": "rtsp://localhost:8554/cam-04"},
    {"id": "cam-05", "file": "camera_05.mp4", "rtsp": "rtsp://localhost:8554/cam-05"},
    {"id": "cam-06", "file": "camera_06.mp4", "rtsp": "rtsp://localhost:8554/cam-06"},
]


# ── FFmpeg RTSP publisher ─────────────────────────────────────────────────────

def build_ffmpeg_command(
    input_file: str,
    rtsp_url:   str,
    loop:       bool = True,
    fps:        int  = 15,
) -> list[str]:
    """
    Build an FFmpeg command that:
      • Reads an mp4 file (optionally looping)
      • Re-publishes it as an RTSP stream at the given URL
      • Runs in real-time (-re flag: read at source fps, not as fast as possible)

    Args:
        input_file: Path to the .mp4 file
        rtsp_url:   Destination RTSP URL  e.g. rtsp://localhost:8554/cam-01
        loop:       If True, loops the file indefinitely
        fps:        Output frame rate

    Returns:
        List of command tokens for subprocess.Popen
    """
    loop_args = ["-stream_loop", "-1"] if loop else []

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",   "warning",
        "-re",                          # Real-time: read at native speed
        *loop_args,
        "-i",          input_file,
        "-c:v",        "libx264",       # H.264 video codec (RTSP standard)
        "-preset",     "ultrafast",     # Low encoding latency
        "-tune",       "zerolatency",   # Minimal buffering
        "-r",          str(fps),        # Output FPS
        "-g",          str(fps * 2),    # Keyframe interval = 2 seconds
        "-b:v",        "500k",          # 500 kbps bitrate
        "-an",                          # No audio
        "-f",          "rtsp",          # Output format: RTSP
        "-rtsp_transport", "tcp",       # TCP more reliable than UDP
        rtsp_url,
    ]


class RTSPPublisher:
    """
    Manages FFmpeg subprocesses for all mock RTSP streams.

    Usage:
        publisher = RTSPPublisher("./mock_sources")
        publisher.start_all()
        # ... cameras are now streaming at rtsp://localhost:8554/cam-0N
        publisher.stop_all()
    """

    def __init__(self, video_dir: str = "./mock_sources") -> None:
        self.video_dir  = video_dir
        self._processes: dict[str, subprocess.Popen] = {}

    def start_all(self) -> None:
        """Launch FFmpeg for all 6 mock cameras."""
        for cam in RTSP_CAMERAS:
            file_path = os.path.join(self.video_dir, cam["file"])
            if not os.path.exists(file_path):
                logger.warning(f"Video file not found: {file_path} — skipping {cam['id']}")
                continue
            self._start_one(cam["id"], file_path, cam["rtsp"])

    def _start_one(self, cam_id: str, file_path: str, rtsp_url: str) -> None:
        cmd = build_ffmpeg_command(file_path, rtsp_url)
        logger.info(f"[{cam_id}] Starting RTSP publisher → {rtsp_url}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._processes[cam_id] = proc
        logger.info(f"[{cam_id}] FFmpeg PID={proc.pid}")

    def stop_all(self) -> None:
        """Terminate all FFmpeg processes."""
        for cam_id, proc in self._processes.items():
            logger.info(f"[{cam_id}] Stopping FFmpeg PID={proc.pid}")
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self._processes.clear()

    def status(self) -> dict[str, str]:
        return {
            cam_id: ("running" if proc.poll() is None else "stopped")
            for cam_id, proc in self._processes.items()
        }


# ── MediaMTX config generator ─────────────────────────────────────────────────

MEDIAMTX_CONFIG = """\
# mediamtx.yml
# MediaMTX (formerly rtsp-simple-server) configuration
# Download: https://github.com/bluenviron/mediamtx/releases

# ── Network ──────────────────────────────────────────────────────────────────
rtspAddress: :8554          # RTSP port
rtmpAddress: :1935          # RTMP port (for OBS etc)
hlsAddress:  :8888          # HLS port (browser-compatible)
webrtcAddress: :8889        # WebRTC port (lowest latency)
apiAddress:  :9997          # REST API for management

# ── HLS (browser-compatible streaming) ───────────────────────────────────────
hls: yes
hlsVariant: lowLatency      # LL-HLS for ~1s latency
hlsSegmentDuration: 1s
hlsPartDuration: 200ms

# ── Paths ─────────────────────────────────────────────────────────────────────
paths:
  # Each path corresponds to one camera stream
  cam-01:
    source: publisher      # Accepts a publisher (FFmpeg)
  cam-02:
    source: publisher
  cam-03:
    source: publisher
  cam-04:
    source: publisher
  cam-05:
    source: publisher
  cam-06:
    source: publisher
"""


def write_mediamtx_config(output_path: str = "mediamtx.yml") -> None:
    """Write the MediaMTX config file to disk."""
    with open(output_path, "w") as f:
        f.write(MEDIAMTX_CONFIG)
    logger.info(f"MediaMTX config written to: {output_path}")
    print(f"✅ MediaMTX config → {output_path}")


# ── CLI helper: print all commands ───────────────────────────────────────────

def print_rtsp_commands(video_dir: str = "./mock_sources") -> None:
    """
    Print all the FFmpeg commands needed to simulate 6 live cameras.
    Run these in separate terminals (or use start_all() above).
    """
    print("\n" + "="*60)
    print(" Mock RTSP stream commands")
    print(" Run each in a separate terminal AFTER starting MediaMTX")
    print("="*60)

    for cam in RTSP_CAMERAS:
        cmd = build_ffmpeg_command(
            os.path.join(video_dir, cam["file"]),
            cam["rtsp"],
        )
        print(f"\n# {cam['id']} → {cam['rtsp']}")
        print(" ".join(cmd))

    print("\n" + "="*60)
    print(" After starting, test with VLC:")
    print("   vlc rtsp://localhost:8554/cam-01")
    print(" Or with ffplay:")
    print("   ffplay rtsp://localhost:8554/cam-01")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Running this file directly prints all commands and generates the config
    write_mediamtx_config()
    print_rtsp_commands()
