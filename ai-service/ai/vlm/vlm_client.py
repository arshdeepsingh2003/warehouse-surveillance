"""
ai/vlm/vlm_client.py
─────────────────────
Unified VLM (Vision Language Model) client.

Supports multiple backends through a single interface:

  BACKEND          COST      LATENCY   ACCURACY  PRIVACY
  ─────────────────────────────────────────────────────────
  mock             free      0ms       demo      ✅ local
  openai_gpt4v     $$        1-3s      ★★★★★    ☁ cloud
  anthropic_claude $$        1-3s      ★★★★★    ☁ cloud
  ollama_llava     free      2-8s*     ★★★☆☆    ✅ local
  ollama_qwen_vl   free      1-5s*     ★★★★☆    ✅ local
  gemini           $         0.5-2s    ★★★★☆    ☁ cloud

  * depends on your hardware — GPU recommended for Ollama

Selection guide:
  → Development / demo:    BACKEND=mock
  → Best accuracy:         BACKEND=openai_gpt4v  (needs OPENAI_API_KEY)
  → Privacy / on-premise:  BACKEND=ollama_llava  (install Ollama locally)
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
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class VLMResult:
    """Structured output from a VLM analysis call."""
    person_id:     str
    camera_id:     str
    zone_id:       str
    description:   str             # full natural-language description
    activity_type: str             # inferred activity label
    anomaly_label: str             # "normal" | "anomaly"
    severity:      str             # "none" | "low" | "medium" | "high"
    confidence:    float           # 0.0–1.0
    raw_response:  str = ""        # original model output (for debugging)
    latency_ms:    int = 0         # inference time
    backend_used:  str = ""        # which model was actually used

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
                timeout=aiohttp.ClientTimeout(total=30),
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
                timeout=aiohttp.ClientTimeout(total=30),
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

        Returns:
            VLMResult with description, activity type, anomaly label
        """
        # Check cache (avoid querying VLM if person/zone unchanged recently)
        cache_key = f"{person_id}:{zone_id}"
        if cache_key in self._cache:
            cached_at, cached_result = self._cache[cache_key]
            if time.time() - cached_at < settings.VLM_CACHE_TTL_SECONDS:
                logger.debug(f"VLM cache hit: {cache_key}")
                return cached_result

        # Crop person from frame
        crop = crop_person(frame, bbox, padding=30)
        if crop.shape[0] < 20 or crop.shape[1] < 20:
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
            logger.error(f"VLM query error for {person_id}: {e}")
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        if not raw:
            return self._fallback_result(person_id, camera_id, zone_id, zone_name)

        # Parse response
        result = self._parse_response(
            raw=raw, person_id=person_id, camera_id=camera_id,
            zone_id=zone_id, zone_name=zone_name,
            latency_ms=latency_ms,
        )

        # Cache the result
        self._cache[cache_key] = (time.time(), result)
        logger.info(
            f"[VLM] {person_id} @ {zone_name}: "
            f"{'⚠ ANOMALY' if result.is_anomaly else 'normal'} | "
            f"{latency_ms}ms | {result.description[:60]}…"
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
            f"and any safety or security concern]\n\n"
            f"Be concise and factual. Focus on observable behavior, posture, "
            f"and any objects being handled."
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
        description = lines.get("DESCRIPTION", raw[:200])

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
            backend_used=  type(self._backend).__name__,
        )

    def _fallback_result(
        self, person_id: str, camera_id: str, zone_id: str, zone_name: str
    ) -> VLMResult:
        """Return a safe default when VLM call fails."""
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

    def get_cache_stats(self) -> dict:
        return {
            "cached_entries": len(self._cache),
            "backend": type(self._backend).__name__,
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
        return response.choices[0].message.content.strip()
