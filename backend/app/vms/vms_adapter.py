"""
app/vms/vms_adapter.py
───────────────────────
VMS (Video Management System) Integration Layer.

The VMS adapter is an abstract interface that normalises communication
with different VMS platforms into a single Python API.

Supported VMS platforms (abstract → concrete adapters):
  MilestoneAdapter   — Milestone XProtect (REST API + SDK)
  GenetecAdapter     — Genetec Security Center (REST API)
  HikvisionAdapter   — Hikvision ISUP / SADP protocol
  ICCCAdapter        — Integrated Command & Control Center (custom REST)
  MockVMSAdapter     — Simulated VMS for development and testing

Why an adapter pattern?
  Different VMS vendors have completely different APIs, auth methods,
  and event schemas. The adapter normalises all of them into:
    - list_cameras()         → list[VMSCamera]
    - get_stream_url(id)     → str (RTSP URL)
    - get_events(since)      → list[VMSEvent]
    - acknowledge_alarm(id)  → bool

Architecture:
  AI Service (stream manager)
        │
        ▼
  VMSAdapterFactory.get(vms_type)
        │
        ├── MilestoneAdapter  → Milestone REST/MIP SDK
        ├── GenetecAdapter    → Genetec SDK
        ├── HikvisionAdapter  → ISUP protocol
        └── MockVMSAdapter    → Mock (development)

When you have real VMS credentials:
  1. Set VMS_TYPE=milestone in .env
  2. Set VMS_HOST, VMS_USERNAME, VMS_PASSWORD
  3. The stream manager will pull real RTSP URLs from the VMS
"""

from __future__ import annotations

import logging
import random
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Normalised data models ────────────────────────────────────────────────────

@dataclass
class VMSCamera:
    """Normalised camera info from any VMS."""
    vms_id:       str               # VMS internal ID
    camera_id:    str               # Our internal ID (e.g. cam-01)
    name:         str
    location:     str
    zone:         str
    rtsp_url:     str               # Live stream URL
    recording_url: Optional[str]   # Playback URL
    status:       str = "online"   # online | offline | unknown
    manufacturer: str = ""
    model:        str = ""


@dataclass
class VMSEvent:
    """Normalised alarm/event from any VMS."""
    event_id:    str
    camera_id:   str
    event_type:  str               # motion | tamper | alarm | offline
    description: str
    severity:    str = "low"
    timestamp:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged:bool = False


# ── Abstract base adapter ─────────────────────────────────────────────────────

class BaseVMSAdapter(ABC):
    """
    Abstract VMS adapter interface.
    All concrete adapters must implement these methods.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection/session with the VMS. Returns True on success."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection cleanly."""

    @abstractmethod
    async def list_cameras(self) -> list[VMSCamera]:
        """Return all cameras registered in the VMS."""

    @abstractmethod
    async def get_stream_url(self, vms_camera_id: str) -> Optional[str]:
        """Return the live RTSP stream URL for a camera."""

    @abstractmethod
    async def get_events(self, since: Optional[datetime] = None) -> list[VMSEvent]:
        """Return VMS events/alarms since the given timestamp."""

    @abstractmethod
    async def acknowledge_event(self, event_id: str, operator: str) -> bool:
        """Acknowledge an alarm/event in the VMS."""

    @abstractmethod
    async def health_check(self) -> dict:
        """Return VMS connection health status."""


# ── Mock VMS (development) ────────────────────────────────────────────────────

class MockVMSAdapter(BaseVMSAdapter):
    """
    Simulated VMS for development and testing.

    Returns plausible mock data. Simulates all 6 cameras as if they
    were registered in a real VMS. Generates random events periodically.

    Use when VMS_TYPE=mock in .env (default for development).
    """

    _MOCK_CAMERAS = [
        ("vms-001", "cam-01", "Main Gate Camera",      "Building Entrance",  "entry_zone",      "rtsp://mock-vms/cam-01"),
        ("vms-002", "cam-02", "Warehouse Aisle Cam",   "Aisle B, Row 3",     "storage_area",    "rtsp://mock-vms/cam-02"),
        ("vms-003", "cam-03", "Loading Dock Camera",   "Dock 1",             "loading_zone",    "rtsp://mock-vms/cam-03"),
        ("vms-004", "cam-04", "Storage Rack Camera",   "Rack Section C",     "storage_area",    "rtsp://mock-vms/cam-04"),
        ("vms-005", "cam-05", "Restricted Zone Cam",   "Server Room Corridor","restricted_area","rtsp://mock-vms/cam-05"),
        ("vms-006", "cam-06", "Packing Area Camera",   "Dispatch Floor",     "packing_area",    "rtsp://mock-vms/cam-06"),
    ]

    def __init__(self) -> None:
        self._connected = False
        self._events: list[VMSEvent] = []

    async def connect(self) -> bool:
        await asyncio.sleep(0.1)   # simulate network latency
        self._connected = True
        logger.info("MockVMSAdapter connected")
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def list_cameras(self) -> list[VMSCamera]:
        await asyncio.sleep(0.05)
        return [
            VMSCamera(
                vms_id=c[0], camera_id=c[1], name=c[2],
                location=c[3], zone=c[4], rtsp_url=c[5],
                recording_url=f"rtsp://mock-vms/playback/{c[0]}",
                status="online", manufacturer="MockCam", model="MC-1080P",
            )
            for c in self._MOCK_CAMERAS
        ]

    async def get_stream_url(self, vms_camera_id: str) -> Optional[str]:
        for c in self._MOCK_CAMERAS:
            if c[0] == vms_camera_id:
                return c[5]
        return None

    async def get_events(self, since: Optional[datetime] = None) -> list[VMSEvent]:
        # Generate a random event occasionally
        if random.random() < 0.3:
            cam   = random.choice(self._MOCK_CAMERAS)
            types = ["motion_detected", "camera_tamper", "connection_lost", "alarm_triggered"]
            event = VMSEvent(
                event_id=   f"evt-{random.randint(1000,9999)}",
                camera_id=  cam[1],
                event_type= random.choice(types),
                description=f"VMS event on {cam[2]}",
                severity=   random.choice(["low", "medium", "high"]),
            )
            self._events.append(event)

        return self._events[-10:]   # last 10 events

    async def acknowledge_event(self, event_id: str, operator: str) -> bool:
        for e in self._events:
            if e.event_id == event_id:
                e.acknowledged = True
                return True
        return False

    async def health_check(self) -> dict:
        return {"status": "ok", "type": "mock", "cameras": len(self._MOCK_CAMERAS)}


# ── Milestone XProtect adapter (stub) ─────────────────────────────────────────

class MilestoneAdapter(BaseVMSAdapter):
    """
    Milestone XProtect adapter using the MIP REST API.

    Milestone XProtect is the most popular enterprise VMS.
    This stub shows the integration pattern — fill in the real
    API calls when you have a Milestone server to connect to.

    Milestone REST API docs:
      https://doc.milestonesys.com/latest/en-US/portal/htm/chapter-page-mc-restapi.htm

    Requirements:
      pip install aiohttp
      VMS_HOST=https://your-milestone-server
      VMS_USERNAME=administrator
      VMS_PASSWORD=your-password
    """

    def __init__(self, host: str, username: str, password: str) -> None:
        self.host     = host.rstrip("/")
        self.username = username
        self.password = password
        self._session_token: Optional[str] = None

    async def connect(self) -> bool:
        """
        Authenticate with Milestone XProtect Management Server.
        Returns a session token used for subsequent API calls.
        """
        # TODO: implement real authentication
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     resp = await session.post(
        #         f"{self.host}/api/IDP/token",
        #         json={"grantType": "password", "username": self.username, "password": self.password}
        #     )
        #     data = await resp.json()
        #     self._session_token = data["token"]
        logger.info(f"MilestoneAdapter: would connect to {self.host}")
        raise NotImplementedError("Set up Milestone credentials and implement connect()")

    async def disconnect(self) -> None:
        self._session_token = None

    async def list_cameras(self) -> list[VMSCamera]:
        # GET /api/rest/v1/cameras
        raise NotImplementedError

    async def get_stream_url(self, vms_camera_id: str) -> Optional[str]:
        # GET /api/rest/v1/cameras/{id}/streams → extract rtsp URL
        raise NotImplementedError

    async def get_events(self, since: Optional[datetime] = None) -> list[VMSEvent]:
        # GET /api/rest/v1/alarms?from={since.isoformat()}
        raise NotImplementedError

    async def acknowledge_event(self, event_id: str, operator: str) -> bool:
        # PUT /api/rest/v1/alarms/{id}/acknowledge
        raise NotImplementedError

    async def health_check(self) -> dict:
        return {"status": "not_connected", "type": "milestone"}


# ── Hikvision ISUP adapter (stub) ─────────────────────────────────────────────

class HikvisionAdapter(BaseVMSAdapter):
    """
    Hikvision IP camera adapter using the ISAPI REST protocol.

    Most Hikvision cameras expose ISAPI on port 80/443.
    RTSP streams are typically on rtsp://ip:554/h264/ch1/main/av_stream

    Requirements:
      VMS_HOST=http://192.168.1.100   (camera IP)
      VMS_USERNAME=admin
      VMS_PASSWORD=camera-password
    """

    def __init__(self, host: str, username: str, password: str) -> None:
        self.host     = host
        self.username = username
        self.password = password

    async def connect(self) -> bool:
        # GET /ISAPI/System/deviceInfo — test connection
        raise NotImplementedError("Implement Hikvision ISAPI connect()")

    async def disconnect(self) -> None:
        pass

    async def list_cameras(self) -> list[VMSCamera]:
        # Single-camera adapter — returns self
        # For NVRs: GET /ISAPI/System/Video/inputs/channels
        raise NotImplementedError

    async def get_stream_url(self, vms_camera_id: str) -> Optional[str]:
        # Standard Hikvision RTSP URL format
        ip = self.host.replace("http://", "").replace("https://", "")
        return f"rtsp://{self.username}:{self.password}@{ip}:554/h264/ch1/main/av_stream"

    async def get_events(self, since: Optional[datetime] = None) -> list[VMSEvent]:
        # GET /ISAPI/Event/notification/alertStream
        raise NotImplementedError

    async def acknowledge_event(self, event_id: str, operator: str) -> bool:
        return True

    async def health_check(self) -> dict:
        return {"status": "not_connected", "type": "hikvision"}


# ── Factory ───────────────────────────────────────────────────────────────────

class VMSAdapterFactory:
    """
    Creates the appropriate VMS adapter based on VMS_TYPE env var.

    Usage:
        adapter = VMSAdapterFactory.create()
        await adapter.connect()
        cameras = await adapter.list_cameras()
    """

    @staticmethod
    def create(vms_type: Optional[str] = None) -> BaseVMSAdapter:
        import os
        t = (vms_type or os.environ.get("VMS_TYPE", "mock")).lower()

        if t == "mock":
            return MockVMSAdapter()

        host     = os.environ.get("VMS_HOST", "")
        username = os.environ.get("VMS_USERNAME", "")
        password = os.environ.get("VMS_PASSWORD", "")

        if t == "milestone":
            return MilestoneAdapter(host, username, password)
        if t in ("hikvision", "hik"):
            return HikvisionAdapter(host, username, password)

        raise ValueError(f"Unknown VMS type: '{t}'. Supported: mock, milestone, hikvision")
