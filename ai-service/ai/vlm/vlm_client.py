"""
ai/vlm/vlm_client.py
─────────────────────
Unified VLM (Vision Language Model) client.

Supports multiple backends through a single interface:

  BACKEND          COST      LATENCY   ACCURACY  PRIVACY
  ─────────────────────────────────────────────────────────────
  mock             free      0ms       demo      ✅ local
  openai_gpt4v     $$        1-3s      ★★★★★    ☁ cloud
  anthropic_claude $$        1-3s      ★★★★★    ☁ cloud
  ollama_llava     free      2-8s*     ★★★☆☆    ✅ local
  ollama_qwen_vl   free      1-5s*     ★★★★☆    ✅ local
  moondream        free      5-15s**   ★★☆☆☆    ✅ local
  gemini           $         0.5-2s    ★★★★☆    ☁ cloud

  * depends on your hardware — GPU recommended for Ollama
  ** CPU-only estimate on a modern laptop; ~2-5s on Apple Silicon

Selection guide:
  → Development / demo:    BACKEND=mock
  → Best accuracy:         BACKEND=openai_gpt4v  (needs OPENAI_API_KEY)
  → Privacy / on-premise:  BACKEND=ollama_llava  (install Ollama locally)
  → CPU-only real-time:    BACKEND=moondream     (tiny 1.9B model)
  → Cost-efficient cloud:  BACKEND=gemini        (needs GEMINI_API_KEY)

Usage:
    client = VLMClient()                     # reads BACKEND from .env
    result = await client.analyze_person(
        frame=frame, bbox=(x1,y1,x2,y2),
        camera_id="cam-01", zone="restricted_area"
    )
    print(result.description)
    print(result.anomaly_label)   # "normal" | "anomaly"
    print(result.confidence)
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the VLM backend returns HTTP 429 (rate limited)."""
    pass


# Short backend names for frontend display
_BACKEND_SHORT_NAMES: dict[str, str] = {
    "MockVLMBackend":       "mock",
    "MoondreamVLMBackend":  "moondream",
    "QwenVLMBackend":       "qwen_vl",
    "OpenAIVLMBackend":     "openai",
    "AnthropicVLMBackend":  "anthropic",
    "OllamaVLMBackend":     "ollama",
    "GeminiVLMBackend":     "gemini",
    "GroqVLMBackend":       "groq",
}


def _short_backend_name(backend: object) -> str:
    """Map a backend class name to the short label used in WS events."""
    cls_name = type(backend).__name__
    return _BACKEND_SHORT_NAMES.get(cls_name, cls_name)


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class VLMResult:
    """Structured output from a VLM analysis call."""
    person_id:         str
    camera_id:         str
    zone_id:           str
    description:       str             # full natural-language description
    activity_type:     str             # inferred activity label
    anomaly_label:     str             # "normal" | "anomaly"
    severity:          str             # "none" | "low" | "medium" | "high"
    confidence:        float           # 0.0–1.0
    raw_response:      str = ""        # original model output (for debugging)
    latency_ms:        int = 0         # inference time
    backend_used:      str = ""        # which model was actually used
    objects_detected:  list = field(default_factory=list)  # objects identified by VLM

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly_label == "anomaly"


# ── Frame encoding helper ─────────────────────────────────────────────────────

def encode_frame_b64(frame: np.ndarray, quality: int = 75) -> str:
    """Encode a BGR numpy array as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def crop_person(frame: np.ndarray, bbox: tuple, padding: int = 20) -> np.ndarray:
    """
    Crop the person region from a frame with padding.

    Args:
        frame:   Full BGR frame
        bbox:    (x1, y1, x2, y2) bounding box
        padding: Pixels of context to include around the box

    Returns:
        Cropped BGR frame. Returns full frame if crop fails.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame


# ── Base backend ──────────────────────────────────────────────────────────────

class BaseVLMBackend:
    """All VLM backends implement this interface."""

    async def query(
        self,
        image_b64: str,
        prompt:    str,
    ) -> str:
        raise NotImplementedError

    async def warmup(self) -> None:
        """Optional: pre-load model weights."""
        pass


# ── Backend 1: Mock (no model needed) ────────────────────────────────────────

_MOCK_DESCRIPTIONS = [
    ("walking",           "normal",  "none",   "Worker is walking through the aisle carrying a clipboard. Movement appears purposeful and routine."),
    ("handling_items",    "normal",  "none",   "Employee is picking items from shelf level 3 and placing them into a transport cart. Standard inventory pick activity."),
    ("standing",          "normal",  "none",   "Worker is standing near the workstation, reviewing a document. No safety concerns observed."),
    ("loitering",         "anomaly", "medium", "Individual has been standing idle near the loading dock for an extended period without performing any work task. Behavior is unusual for this zone."),
    ("unauthorized_entry","anomaly", "high",   "Person detected entering the restricted server room corridor. No PPE or badge visible. Area is marked as restricted access only."),
    ("falling",           "anomaly", "high",   "Worker appears to have fallen near the storage rack. Person is horizontal and has remained stationary for over 30 seconds. Immediate assistance may be required."),
    ("carrying_object",   "normal",  "none",   "Employee is transporting a large box using proper lifting technique. Activity is consistent with normal warehouse operations."),
    ("running",           "anomaly", "medium", "Person is running in the warehouse floor area. Running is prohibited in this zone due to safety regulations."),
    ("handling_items",    "normal",  "none",   "Worker is scanning barcodes on packages at the dispatch station. Normal end-of-shift processing activity."),
    ("crouching",         "normal",  "none",   "Employee crouched to retrieve an item from the bottom shelf. Posture appears correct and safe."),
]

class MockVLMBackend(BaseVLMBackend):
    """
    Returns realistic pre-scripted descriptions.
    Used in development/demo mode when no API key is configured.
    """

    def __init__(self, latency_ms: int = 120) -> None:
        self._latency_ms = latency_ms

    async def query(self, image_b64: str, prompt: str) -> str:
        await asyncio.sleep(self._latency_ms / 1000)
        _, _, _, desc = random.choice(_MOCK_DESCRIPTIONS)
        return desc


# ── Backend 2: OpenAI GPT-4V ─────────────────────────────────────────────────

class OpenAIVLMBackend(BaseVLMBackend):
    """
    GPT-4o / GPT-4 Vision backend via the OpenAI API.

    Requirements:
        pip install openai
        Set OPENAI_API_KEY in .env

    Cost estimate: ~$0.003 per image at 640×360 with detail="low"
    Latency: 1–3 seconds per call
    """

    def __init__(self, model: str = "gpt-4o", detail: str = "low") -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model  = model
        self._detail = detail
        logger.info(f"OpenAI VLM backend ready: {model} (detail={detail})")

    async def query(self, image_b64: str, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{image_b64}",
                            "detail": self._detail,   # "low" = faster/cheaper
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content.strip()


# ── Backend 3: Anthropic Claude Vision ───────────────────────────────────────

class AnthropicVLMBackend(BaseVLMBackend):
    """
    Claude 3.5 Sonnet / Haiku vision backend.

    Requirements:
        pip install anthropic
        Set ANTHROPIC_API_KEY in .env

    Claude is excellent for nuanced safety analysis and structured output.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._model  = model
        logger.info(f"Anthropic VLM backend ready: {model}")

    async def query(self, image_b64: str, prompt: str) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return message.content[0].text.strip()


# ── Backend 4: Ollama (local, privacy-preserving) ─────────────────────────────

class OllamaVLMBackend(BaseVLMBackend):
    """
    Local VLM via Ollama — runs LLaVA, Qwen-VL, or llama-vision on your hardware.

    Requirements:
        1. Install Ollama: https://ollama.ai
        2. Pull a vision model:
             ollama pull llava:7b          # 4.7 GB, good general purpose
             ollama pull qwen2.5vl:7b      # 5.5 GB, excellent for detail
             ollama pull minicpm-v:latest  # 5.5 GB, fast and accurate
        3. Set OLLAMA_MODEL in .env

    Latency: 2–8s on CPU, 0.5–2s on GPU
    Privacy: 100% local — no data leaves your server
    Cost:    Free (electricity only)
    """

    def __init__(
        self,
        model:   str = "llava:7b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        import aiohttp
        self._model    = model
        self._base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Ollama VLM backend: {model} at {base_url}")

    async def warmup(self) -> None:
        """Ensure the model is loaded into memory."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": "hello", "stream": False},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Ollama model {self._model} warmed up")
                else:
                    logger.warning(f"Ollama warmup returned {resp.status}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e} — will use mock fallback")

    async def query(self, image_b64: str, prompt: str) -> str:
        if not self._session:
            import aiohttp
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate",
                json={
                    "model":  self._model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200},
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                return data.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            return ""


# ── Backend 5: Google Gemini Vision ──────────────────────────────────────────

class GeminiVLMBackend(BaseVLMBackend):
    """
    Google Gemini Pro Vision backend.

    Requirements:
        pip install google-generativeai
        Set GEMINI_API_KEY in .env

    Gemini is fast and cost-effective for high-volume surveillance.
    """

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(model)
        logger.info(f"Gemini VLM backend ready: {model}")

    async def query(self, image_b64: str, prompt: str) -> str:
        import google.generativeai as genai
        img_bytes = base64.b64decode(image_b64)
        img_part  = {"mime_type": "image/jpeg", "data": img_bytes}
        response  = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._model.generate_content([img_part, prompt])
        )
        return response.text.strip()


# ── Backend 6: Qwen2.5-VL via Ollama (dedicated warehouse surveillance) ─────

_WAREHOUSE_PROMPT = """\
You are a warehouse surveillance analysis AI. Analyze the worker in this image \
and return ONLY valid JSON (no markdown, no extra text):

{
  "activity_description": "Describe what the worker is doing in one sentence. \
Mention posture, movements, and interactions with objects.",
  "objects_detected": ["list of visible objects being handled, e.g. boxes, \
tools, forklift, shelves, bags, pallets, scanner, clipboard"],
  "activity_category": "one of: walking, standing, carrying, handling_items, \
loading, unloading, operating_equipment, crouching, climbing, inspecting, unknown",
  "confidence": 0.0-1.0
}

Rules:
- If no person is clearly visible, set activity_category to "unknown" and confidence to 0.0
- Note if activity appears normal or unusual for warehouse operations
- Flag safety concerns (missing PPE, running, unsafe ladder use, fall risk) in the description
- Be concise, factual, and specific"""


class QwenVLMBackend(BaseVLMBackend):
    """
    Qwen2.5-VL via local Ollama — dedicated warehouse surveillance backend.

    Requirements:
        1. Install Ollama: https://ollama.ai
        2. Pull the Qwen2.5-VL model:
             ollama pull qwen2.5-vl:7b
        3. Set in .env:
             VLM_BACKEND=qwen_vl
             OLLAMA_HOST=http://localhost:11434
             QWEN_VL_MODEL=qwen2.5-vl

    Privacy: 100% local — no data leaves your server
    Cost:    Free (electricity only)
    Latency: 1–5s on GPU, 5–15s on CPU
    """

    PROMPT = _WAREHOUSE_PROMPT

    def __init__(
        self,
        model:    str = "qwen2.5-vl",
        base_url: str = "http://localhost:11434",
    ) -> None:
        import aiohttp
        self._model    = model
        self._base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Qwen2.5-VL backend: {model} at {base_url}")

    async def warmup(self) -> None:
        """Ensure the Qwen model is loaded into Ollama memory."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": "hello", "stream": False},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Qwen2.5-VL model {self._model} warmed up")
                else:
                    logger.warning(f"Qwen2.5-VL warmup returned {resp.status}")
        except Exception as e:
            logger.warning(f"Qwen2.5-VL not available: {e}")

    async def query(self, image_b64: str, prompt: str) -> str:
        if not self._session:
            import aiohttp
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate",
                json={
                    "model":   self._model,
                    "prompt":  prompt,
                    "images":  [image_b64],
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 300},
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                return data.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Qwen2.5-VL query failed: {e}")
            return ""


# ── Backend 7: Moondream via Ollama (lightweight CPU vision) ─────────────────

_MOONDREAM_VLM_PROMPT = (
    "Describe what the person is doing in this warehouse image in one short sentence. "
    "Mention their activity, posture, and any objects visible. "
    "Is their behavior normal or anomalous?"
)


class MoondreamVLMBackend(BaseVLMBackend):
    """
    Moondream 2 (1.9B) via local Ollama — CPU-first lightweight vision model.

    Moondream is a tiny vision-language model (~1.9B params) designed to run
    on edge devices.  It is the only local VLM that can realistically achieve
    <10 s inference on CPU-only hardware.

    Requirements:
        1. Install Ollama: https://ollama.ai
        2. Pull the model:
             ollama pull moondream      # ~1.7 GB, Q4 quantized
        3. Set in .env:
             VLM_BACKEND=moondream
             OLLAMA_HOST=http://localhost:11434

    Design notes:
      • Prompt + output are kept short (moondream has a 2K context window).
      • num_predict=80 keeps text generation short → faster inference.
      • Uses the same _parse_response() format as the generic Ollama backend.

    Privacy: 100% local — no data leaves your server
    Cost:    Free (electricity only)
    Latency: 5–15 s on modern laptop CPU, 2–5 s on Apple Silicon
             (substantially faster than Qwen2.5-VL 7B at 180+ s)
    """

    PROMPT = _MOONDREAM_VLM_PROMPT

    def __init__(
        self,
        model:    str = "moondream",
        base_url: str = "http://localhost:11434",
    ) -> None:
        import aiohttp
        self._model    = model
        self._base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Moondream VLM backend: {model} at {base_url}")

    async def warmup(self) -> None:
        """Pre-load the tiny model into Ollama memory."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": "hello", "stream": False},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Moondream model {self._model} warmed up")
                else:
                    logger.warning(f"Moondream warmup returned {resp.status}")
        except Exception as e:
            logger.warning(f"Moondream not available: {e}")

    async def query(self, image_b64: str, prompt: str) -> str:
        if not self._session:
            import aiohttp
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate",
                json={
                    "model":   self._model,
                    "prompt":  prompt,
                    "images":  [image_b64],
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 80},
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                return data.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Moondream query failed: {e}")
            return ""


# ── Factory ───────────────────────────────────────────────────────────────────

def _build_backend(name: str) -> BaseVLMBackend:
    """Instantiate the configured VLM backend, falling back to mock if unavailable."""
    try:
        if name == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set")
            groq_model = getattr(settings, "GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
            return GroqVLMBackend(model=groq_model)

        if name == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            return OpenAIVLMBackend(model=settings.VLM_MODEL)

        if name == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return AnthropicVLMBackend(model=settings.VLM_MODEL)

        if name == "ollama":
            return OllamaVLMBackend(
                model=    settings.OLLAMA_MODEL,
                base_url= settings.OLLAMA_BASE_URL,
            )

        if name == "qwen_vl":
            return QwenVLMBackend(
                model=    settings.QWEN_VL_MODEL,
                base_url= settings.OLLAMA_HOST,
            )

        if name == "moondream":
            return MoondreamVLMBackend(
                model=    settings.MOONDREAM_MODEL,
                base_url= settings.OLLAMA_HOST,
            )

        if name == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            return GeminiVLMBackend(model=settings.VLM_MODEL)

    except Exception as e:
        logger.warning(f"VLM backend '{name}' unavailable: {e} — falling back to mock")

    logger.info("VLM backend: mock mode (set VLM_BACKEND in .env to activate real model)")
    return MockVLMBackend(latency_ms=settings.VLM_MOCK_LATENCY_MS)


# ── High-level VLM client ─────────────────────────────────────────────────────

class VLMClient:
    """
    High-level VLM client used by the AI pipeline.

    Handles:
      • Cropping person regions from full frames
      • Building context-aware prompts
      • Parsing model output into VLMResult
      • Fallback to rule-based result on error
      • Response caching (avoid redundant calls for static scenes)

    Usage:
        client = VLMClient()
        result = await client.analyze_person(
            frame=frame, bbox=(x1,y1,x2,y2),
            person_id="P-1025", camera_id="cam-05",
            zone_id="restricted_area", zone_name="Restricted Area"
        )
    """

    def __init__(self) -> None:
        self._backend = _build_backend(settings.VLM_BACKEND)
        # Simple in-memory response cache: key = (person_id, zone_id) → (timestamp, result)
        self._cache: dict[str, tuple[float, VLMResult]] = {}
        # Global concurrency throttle — max 3 concurrent Groq API calls
        self._semaphore = asyncio.Semaphore(3)

    async def warmup(self) -> None:
        await self._backend.warmup()

    async def analyze_person(
        self,
        frame:       np.ndarray,
        bbox:        tuple,
        person_id:   str,
        camera_id:   str,
        zone_id:     str,
        zone_name:   str,
        is_restricted: bool = False,
        extra_context: str = "",
        track_uuid:  Optional[str] = None,
    ) -> VLMResult:
        """
        Analyze one tracked person's behavior using the VLM.

        Args:
            frame:         Full BGR frame from the camera
            bbox:          (x1,y1,x2,y2) person bounding box
            person_id:     Tracked person ID (e.g. "P-1025")
            camera_id:     Camera ID (e.g. "cam-05")
            zone_id:       Zone identifier
            zone_name:     Human-readable zone name
            is_restricted: Whether the zone is a restricted area
            extra_context: Any additional context to include in the prompt
            track_uuid:    Stable UUID for cache key continuity (preferred over person_id)

        Returns:
            VLMResult with description, activity type, anomaly label
        """
        # Check cache (avoid querying VLM if person/zone unchanged recently)
        cache_key = f"{track_uuid or person_id}:{zone_id}"
        if cache_key in self._cache:
            cached_at, cached_result = self._cache[cache_key]
            if time.time() - cached_at < settings.VLM_CACHE_TTL_SECONDS:
                # HARD RULE: never return cached fallback
                if cached_result.backend_used == "fallback":
                    logger.warning(
                        f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                        f"reason=cached_fallback_evicted "
                        f"cache_key={cache_key}"
                    )
                    del self._cache[cache_key]
                else:
                    logger.debug(f"VLM cache hit: {cache_key}")
                    return cached_result

        # Crop person from frame
        crop = crop_person(frame, bbox, padding=30)
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=crop_too_small "
                f"crop_shape={crop.shape}"
            )
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        # Build context-aware prompt
        prompt = self._build_prompt(zone_name, is_restricted, extra_context)

        # Encode image
        image_b64 = encode_frame_b64(crop, quality=settings.VLM_JPEG_QUALITY)

        # Query VLM
        t0 = time.monotonic()
        try:
            raw = await self._backend.query(image_b64, prompt)
            latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as e:
            logger.error(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=VLM_query_exception error={e}"
            )
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        if not raw:
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=empty_vlm_response"
            )
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        # Parse response
        result = self._parse_response(
            raw=raw, person_id=person_id, camera_id=camera_id,
            zone_id=zone_id, zone_name=zone_name,
            latency_ms=latency_ms,
        )

        # HARD RULE: never cache fallback results
        if result.backend_used == "fallback":
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=parse_returned_fallback "
                f"raw_preview={raw[:100]}"
            )
            return result

        # Cache the result (only non-fallback)
        self._cache[cache_key] = (time.time(), result)
        logger.info(
            f"[VLM] {person_id} @ {zone_name}: "
            f"{'⚠ ANOMALY' if result.is_anomaly else 'normal'} | "
            f"{latency_ms}ms | {result.description[:60]}…"
        )
        logger.info(
            f"[GROQ-TRACE] person_id={person_id} camera_id={camera_id} "
            f"raw_response=\"{raw[:300]}\" "
            f"parsed_description=\"{result.description[:200]}\" "
            f"overlay_summary=\"{result.description[:120]}\""
        )
        return result

    def _build_prompt(
        self,
        zone_name:     str,
        is_restricted: bool,
        extra_context: str,
    ) -> str:
        """
        Build a context-aware prompt for the VLM.

        Good VLM prompts for surveillance:
          • State the context (warehouse, zone type)
          • Ask for specific observable facts (posture, objects, movement)
          • Ask for safety/anomaly assessment
          • Request concise, structured output
        """
        zone_context = (
            f"This is a RESTRICTED ACCESS area ({zone_name}). "
            "Any unauthorized person here is a security concern."
            if is_restricted
            else f"This is the {zone_name} of an industrial warehouse."
        )

        return (
            f"You are a warehouse safety monitoring AI.\n"
            f"Context: {zone_context}\n"
            f"{extra_context}\n\n"
            f"Analyze this image and respond in this exact format:\n"
            f"ACTIVITY: [one of: walking, running, standing, loitering, carrying_object, "
            f"handling_items, falling, crouching, unauthorized_entry, unknown]\n"
            f"ANOMALY: [normal or anomaly]\n"
            f"SEVERITY: [none, low, medium, or high]\n"
            f"DESCRIPTION: [one clear sentence describing what the person is doing "
            f"and any safety or security concern, noting any objects being carried]\n\n"
            f"Be concise and factual. Focus on observable behavior, posture, "
            f"and any objects being handled or carried."
        )

    def _parse_response(
        self,
        raw:        str,
        person_id:  str,
        camera_id:  str,
        zone_id:    str,
        zone_name:  str,
        latency_ms: int,
    ) -> VLMResult:
        """
        Parse the VLM's structured text response into a VLMResult.

        Handles both well-formed and messy outputs gracefully.
        """
        lines = {
            line.split(":")[0].strip().upper(): ":".join(line.split(":")[1:]).strip()
            for line in raw.splitlines()
            if ":" in line
        }

        activity    = lines.get("ACTIVITY", "unknown").lower().replace(" ", "_")
        anomaly     = lines.get("ANOMALY",  "normal").lower()
        severity    = lines.get("SEVERITY", "none").lower()
        description = lines.get("DESCRIPTION", raw[:500])

        # If no structured fields were found, try keyword classification
        # from free-form text (needed for Moondream which can't do colon format)
        if not lines and raw:
            text_lower = raw.lower()
            activity_keywords = {
                "walking":    ("walk", "walking", "walked"),
                "standing":   ("stand", "standing", "stationary"),
                "carrying":   ("carry", "carrying", "hold", "holding", "transport"),
                "handling_items": ("pick", "handling", "scan", "sort", "pack", "unpack",
                                   "picking", "placing", "arranging", "stacking"),
                "loitering":  ("loiter", "idle", "waiting", "unoccupied"),
                "crouching":  ("crouch", "bend", "bending", "kneel", "squat"),
                "running":    ("run", "running", "jog", "jogging"),
                "falling":    ("fall", "falling", "fallen", "collapse"),
            }
            for act, keywords in activity_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    activity = act
                    break
            if any(w in text_lower for w in ("suspicious", "unauthorized", "danger", "fall", "anomal", "concern")):
                anomaly = "anomaly"

        # Normalise anomaly label
        if anomaly not in ("normal", "anomaly"):
            anomaly = "anomaly" if any(
                w in raw.lower()
                for w in ("suspicious", "unauthorized", "danger", "fall", "anomal", "concern")
            ) else "normal"

        # Confidence: based on severity + whether structured output was parsed
        confidence = {
            "high": 0.92, "medium": 0.80, "low": 0.70, "none": 0.85,
        }.get(severity, 0.75)

        return VLMResult(
            person_id=     person_id,
            camera_id=     camera_id,
            zone_id=       zone_id,
            description=   description,
            activity_type= activity,
            anomaly_label= anomaly,
            severity=      severity if severity != "none" else "low",
            confidence=    confidence,
            raw_response=  raw,
            latency_ms=    latency_ms,
            backend_used=  _short_backend_name(self._backend),
        )

    def _parse_qwen_response(
        self,
        raw:        str,
        person_id:  str,
        camera_id:  str,
        zone_id:    str,
        zone_name:  str,
        latency_ms: int,
    ) -> VLMResult:
        """
        Parse Qwen2.5-VL JSON structured response into VLMResult.

        Expected JSON schema:
          {
            "activity_description": "...",
            "objects_detected": [...],
            "activity_category": "...",
            "confidence": 0.0-1.0
          }
        """
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)
            description = data.get("activity_description", raw[:200])
            objects     = data.get("objects_detected", [])
            category    = data.get("activity_category", "unknown").lower().replace(" ", "_")
            confidence  = float(data.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(f"Failed to parse Qwen JSON response, using raw text: {raw[:80]}…")
            description = raw[:200]
            objects     = []
            category    = "unknown"
            confidence  = 0.5

        # Derive anomaly + severity from activity category
        anomalous_cats = {"loitering", "unauthorized_entry", "running",
                          "falling", "climbing", "operating_equipment"}
        danger_cats    = {"falling", "unauthorized_entry"}

        if category in danger_cats:
            anomaly = "anomaly"
            severity = "high"
        elif category in anomalous_cats:
            anomaly = "anomaly"
            severity = "medium"
        else:
            anomaly = "normal"
            severity = "none"

        return VLMResult(
            person_id=        person_id,
            camera_id=        camera_id,
            zone_id=          zone_id,
            description=      description,
            activity_type=    category,
            anomaly_label=    anomaly,
            severity=         severity,
            confidence=       min(max(confidence, 0.0), 1.0),
            raw_response=     raw,
            latency_ms=       latency_ms,
            backend_used=     _short_backend_name(self._backend),
            objects_detected= objects,
        )

    def _fallback_result(
        self, person_id: str, camera_id: str, zone_id: str, zone_name: str
    ) -> VLMResult:
        """Return a safe default when VLM call fails."""
        # Build a compact call-stack snippet (last 3 frames after this one)
        stack = "|".join(
            f"{f.filename}:{f.lineno}"
            for f in inspect.stack()[1:4]
        )
        logger.warning(
            f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
            f"reason=VLM_call_failed call_stack=[{stack}]"
        )
        return VLMResult(
            person_id=     person_id,
            camera_id=     camera_id,
            zone_id=       zone_id,
            description=   f"Person detected in {zone_name}. VLM analysis unavailable.",
            activity_type= "unknown",
            anomaly_label= "normal",
            severity=      "none",
            confidence=    0.5,
            backend_used=  "fallback",
        )

    async def analyze_crop(
        self,
        crop_path:   str,
        metadata:    dict,
        crop_array:  Optional[np.ndarray] = None,
        enqueue_ts:  Optional[float] = None,
    ) -> VLMResult:
        """
        Analyze a person crop image — supports both in-memory and disk paths.

        Args:
            crop_path:   Absolute path to a JPEG crop file on disk (used only
                         when *crop_array* is not provided).
            metadata:    Dict with at minimum:
                            person_id     – str
                            camera_id     – str
                            zone_id       – str
                            zone_name     – str
                            is_restricted – bool
                            extra_context – str (optional rule context)
            crop_array:  In-memory BGR numpy array of the crop (fast path,
                         avoids disk read). Takes precedence over crop_path.
            enqueue_ts:  time.monotonic() timestamp when the VLM task was first
                         queued by the pipeline. Used to compute QUEUE_WAIT_MS.

        Returns:
            VLMResult  (same schema as analyze_person)
        """
        total_t0 = time.monotonic()
        person_id     = metadata.get("person_id",     "unknown")
        track_uuid    = metadata.get("track_uuid",    person_id)  # prefer stable UUID
        camera_id     = metadata.get("camera_id",     "unknown")
        zone_id       = metadata.get("zone_id",       "unknown")
        zone_name     = metadata.get("zone_name",     "Unknown")
        is_restricted = metadata.get("is_restricted", False)
        extra_context = metadata.get("extra_context", "")

        # Compute queue wait (time spent in asyncio queue before semaphore acquire)
        queue_wait_ms = 0
        if enqueue_ts is not None:
            queue_wait_ms = int((total_t0 - enqueue_ts) * 1000)

        # Check cache (keyed by track_uuid for stable lookup across ID regeneration)
        cache_key = f"{track_uuid}:{zone_id}"
        if cache_key in self._cache:
            cached_at, cached_result = self._cache[cache_key]
            if time.time() - cached_at < settings.VLM_CACHE_TTL_SECONDS:
                # HARD RULE: never return cached fallback
                if cached_result.backend_used == "fallback":
                    logger.warning(
                        f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                        f"reason=cached_fallback_evicted "
                        f"cache_key={cache_key}"
                    )
                    del self._cache[cache_key]
                else:
                    logger.debug(f"VLM cache hit: {cache_key}")
                    return cached_result

        # Get crop array: prefer in-memory, fall back to disk
        if crop_array is not None:
            crop = crop_array
        else:
            if not os.path.isfile(crop_path):
                logger.warning(
                    f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                    f"reason=crop_file_not_found path={crop_path}"
                )
                return self._fallback_result(person_id, camera_id, zone_id, zone_name)
            try:
                crop = cv2.imread(crop_path)
                if crop is None or crop.size == 0:
                    logger.warning(
                        f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                        f"reason=empty_crop path={crop_path}"
                    )
                    return self._fallback_result(person_id, camera_id, zone_id, zone_name)
            except Exception as e:
                logger.warning(
                    f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                    f"reason=crop_read_error path={crop_path} error={e}"
                )
                return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        if crop.shape[0] < 20 or crop.shape[1] < 20:
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=crop_too_small crop_shape={crop.shape}"
            )
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        # Detect backend type for prompt selection
        is_qwen      = isinstance(self._backend, QwenVLMBackend)
        is_moondream = isinstance(self._backend, MoondreamVLMBackend)

        if is_qwen:
            prompt = QwenVLMBackend.PROMPT
        elif is_moondream:
            prompt = MoondreamVLMBackend.PROMPT
        else:
            prompt = self._build_prompt(zone_name, is_restricted, extra_context)

        # Encode
        image_b64 = encode_frame_b64(crop, quality=settings.VLM_JPEG_QUALITY)

        # ── Throttled VLM query ────────────────────────────────────────────
        async with self._semaphore:
            groq_t0 = time.monotonic()
            try:
                logger.info(
                    f"[GROQ-REQUEST] person_id={person_id} camera_id={camera_id} "
                    f"prompt_preview={prompt[:120]}... "
                    f"image_b64_len={len(image_b64)}"
                )
                raw = await self._backend.query(image_b64, prompt)
                groq_latency_ms = int((time.monotonic() - groq_t0) * 1000)
                logger.info(
                    f"[GROQ-RESPONSE] person_id={person_id} camera_id={camera_id} "
                    f"latency_ms={groq_latency_ms} "
                    f"raw_response=\"{raw[:300]}\""
                )
            except Exception as e:
                logger.error(
                    f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                    f"reason=VLM_query_exception error={e}"
                )
                return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        if not raw:
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=empty_vlm_response"
            )
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        # Parse
        total_latency_ms = int((time.monotonic() - total_t0) * 1000)
        if is_qwen:
            result = self._parse_qwen_response(
                raw=raw, person_id=person_id, camera_id=camera_id,
                zone_id=zone_id, zone_name=zone_name,
                latency_ms=total_latency_ms,
            )
        else:
            result = self._parse_response(
                raw=raw, person_id=person_id, camera_id=camera_id,
                zone_id=zone_id, zone_name=zone_name,
                latency_ms=total_latency_ms,
            )

        # HARD RULE: never cache fallback results
        if result.backend_used == "fallback":
            logger.warning(
                f"[FALLBACK-TRACE] person_id={person_id} camera_id={camera_id} "
                f"reason=parse_returned_fallback "
                f"raw_preview={raw[:100]}"
            )
            return result

        self._cache[cache_key] = (time.time(), result)

        # ── Latency metrics + Groq trace logging ───────────────────────────
        logger.info(
            f"[VLM] camera={camera_id} person={person_id} "
            f"QUEUE_WAIT={queue_wait_ms}ms "
            f"GROQ_LATENCY={groq_latency_ms}ms "
            f"TOTAL_VLM_LATENCY={total_latency_ms}ms "
            f"backend={type(self._backend).__name__} "
            f"activity={result.activity_type} "
            f"anomaly={result.anomaly_label} "
            f"confidence={result.confidence:.2f} "
            f"desc=\"{result.description[:80]}\""
        )
        logger.info(
            f"[GROQ-TRACE] person_id={person_id} camera_id={camera_id} "
            f"raw_response=\"{raw[:300]}\" "
            f"parsed_description=\"{result.description[:200]}\" "
            f"overlay_summary=\"{result.description[:120]}\""
        )
        if is_qwen:
            obj_str = ",".join(result.objects_detected) if result.objects_detected else "none"
            logger.info(f"[VLM] camera={camera_id} person={person_id} objects=[{obj_str}]")
        return result

    def clear_cache(self) -> None:
        """Clear the in-memory VLM result cache."""
        n = len(self._cache)
        self._cache.clear()
        logger.info(f"[VLM] Cleared {n} cached entries")

    def get_cache_stats(self) -> dict:
        return {
            "cached_entries": len(self._cache),
            "backend": _short_backend_name(self._backend),
        }


# ── Backend: Groq Vision (Llama 4 Scout) ─────────────────────────────────────

class GroqVLMBackend(BaseVLMBackend):
    """
    Groq Vision backend using Llama-4 Scout (16E) multimodal model.

    Llama 4 Scout on Groq:
      • Model:   meta-llama/llama-4-scout-17b-16e-instruct
      • Speed:   ~500 tokens/sec (fastest available vision inference)
      • Cost:    Free tier available at console.groq.com
      • Context: 128K tokens

    Setup:
        GROQ_API_KEY=gsk_...
        VLM_BACKEND=groq
        GROQ_VLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
    """

    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> None:
        from groq import AsyncGroq
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model  = model
        logger.info(f"Groq VLM backend ready: {model}")

    async def query(self, image_b64: str, prompt: str) -> str:
        logger.info(
            f"[GROQ-REQUEST] model={self._model} "
            f"prompt_preview={prompt[:120]}... "
            f"image_b64_len={len(image_b64)}"
        )
        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=      self._model,
                max_tokens= 300,
                temperature=0.2,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            raw_text = response.choices[0].message.content.strip()
            logger.info(
                f"[GROQ-RESPONSE] latency_ms={latency_ms} "
                f"raw_response={raw_text[:200]}"
            )
            return raw_text
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                raise RateLimitError("Groq HTTP 429 rate limit exceeded") from e
            if hasattr(e, 'status') and e.status == 429:
                raise RateLimitError("Groq HTTP 429 rate limit exceeded") from e
            raise
