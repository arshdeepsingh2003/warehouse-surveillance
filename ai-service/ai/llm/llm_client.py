"""
ai/llm/llm_client.py
─────────────────────
LLM (Large Language Model) client for report generation and reasoning.

While the VLM sees images and describes individual workers,
the LLM takes those descriptions + detection data and reasons
at a higher level:

  VLM output (per person)         LLM output (zone / shift level)
  ─────────────────────────────   ────────────────────────────────
  "05-P1025 is walking through     "Storage area has been mostly
   the storage aisle carrying       normal today with routine pick
   a large white box."              operations. One loitering event
                                    detected at 14:23 near the dock
  "05-P1031 has been standing       entrance — recommend follow-up.
   near the dock entrance for        Overall risk: LOW."
   18 minutes without activity."

LLM tasks:
  1. Zone-level activity summary (every N minutes)
  2. Shift-end consolidated report
  3. Anomaly explanation (why is this flagged as suspicious?)
  4. Natural-language alert description upgrade (more context-aware)

Supported backends: same as VLM (openai, anthropic, ollama, mock)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Output schemas ────────────────────────────────────────────────────────────

@dataclass
class ZoneSummary:
    """LLM-generated summary for one zone over a time window."""
    zone_id:         str
    zone_name:       str
    time_window_min: int
    summary:         str             # natural language paragraph
    risk_level:      str             # "low" | "medium" | "high"
    key_events:      list[str]       # bullet-point notable events
    person_count:    int
    alert_count:     int
    generated_at:    str             # ISO timestamp
    latency_ms:      int = 0


@dataclass
class ShiftReport:
    """Consolidated shift-end report across all zones."""
    report_date:     str
    shift:           str             # "morning" | "afternoon" | "night"
    summary:         str
    zone_summaries:  list[ZoneSummary]
    total_alerts:    int
    high_severity:   int
    recommendations: list[str]
    generated_at:    str
    latency_ms:      int = 0


@dataclass
class AnomalyExplanation:
    """Detailed LLM reasoning for why an alert was raised."""
    alert_id:      str
    alert_type:    str
    explanation:   str              # why this is suspicious / dangerous
    context:       str              # what normally happens here
    recommendation: str            # what security should do
    false_positive_probability: str  # "low" | "medium" | "high"
    latency_ms:    int = 0


# ── Mock report templates ─────────────────────────────────────────────────────

_MOCK_ZONE_SUMMARIES = {
    "restricted_area": (
        "The restricted area has seen {n_persons} personnel entries during this period. "
        "{n_alerts} unauthorized access {event} detected. "
        "Security protocols appear to need reinforcement — recommend badge-scan audit "
        "and additional camera coverage near the eastern corridor."
    ),
    "storage_area": (
        "Storage area activity has been {activity_level} with {n_persons} workers conducting "
        "inventory management, item retrieval, and restocking operations. "
        "No significant safety incidents observed. "
        "One worker was noted stationary for an extended period near rack row C — likely performing inventory count."
    ),
    "loading_zone": (
        "Loading dock operations ran {activity_level} during this window. "
        "{n_persons} personnel processed inbound and outbound shipments. "
        "{n_alerts} {event} flagged for review. "
        "Dock door 3 had elevated dwell time — possible congestion during peak hours."
    ),
    "entry_zone": (
        "Entry zone processed {n_persons} personnel entries and exits. "
        "Badge compliance appears normal. "
        "No tailgating events detected. "
        "Visitor flow was consistent with expected shift patterns."
    ),
    "packing_area": (
        "Packing and dispatch area had {n_persons} workers on shift. "
        "PPE compliance requires attention — {n_alerts} helmet violation {event} recorded. "
        "Output throughput appears normal based on movement patterns."
    ),
}

_ANOMALY_EXPLANATIONS = {
    "unauthorized_access": (
        "This alert was triggered because a person entered a zone designated as restricted access "
        "without triggering the badge reader or being accompanied by authorized personnel. "
        "Restricted zones typically contain high-value equipment or sensitive systems. "
        "Security should verify whether this was an authorized emergency access or a breach.",
        "Escort unauthorized individual out immediately and review access logs."
    ),
    "loitering": (
        "Loitering is flagged when a person remains stationary in an area for significantly "
        "longer than the operational norm for that zone. "
        "In warehouse environments, extended stationary periods without work activity "
        "can indicate distraction, illness, or potential theft preparation.",
        "Security patrol should visually verify the individual and confirm they are working."
    ),
    "worker_fall": (
        "A fall event is detected when a person's bounding box aspect ratio changes dramatically "
        "(person becomes horizontal) or when they remain stationary at ground level for an extended period. "
        "Worker falls in warehouse environments can cause serious injury. "
        "Immediate response is critical.",
        "Dispatch first aid immediately. Do not move the worker unless in immediate danger."
    ),
    "ppe_violation": (
        "PPE violation detected based on absence of expected safety equipment in a mandatory compliance zone. "
        "Warehouse regulations require helmets, vests, and steel-toed footwear in this area. "
        "Even brief non-compliance creates significant injury liability.",
        "Remind worker of PPE requirements. Log the violation for compliance reporting."
    ),
}


# ── Base backend ──────────────────────────────────────────────────────────────

class BaseLLMBackend:
    async def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


# ── Mock backend ──────────────────────────────────────────────────────────────

class MockLLMBackend(BaseLLMBackend):
    def __init__(self, latency_ms: int = 80) -> None:
        self._latency_ms = latency_ms

    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(self._latency_ms / 1000)
        # Return a generic plausible response
        return (
            "Based on the available surveillance data, activity in this zone appears consistent "
            "with normal warehouse operations. Personnel movement patterns match expected shift behavior. "
            "No critical safety concerns identified beyond the flagged alert events, "
            "which have been escalated through the standard notification channels."
        )


# ── OpenAI GPT-4 backend ──────────────────────────────────────────────────────

class OpenAILLMBackend(BaseLLMBackend):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model  = model
        logger.info(f"OpenAI LLM backend ready: {model}")

    async def complete(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=500,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


# ── Anthropic Claude backend ──────────────────────────────────────────────────

class AnthropicLLMBackend(BaseLLMBackend):
    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._model  = model
        logger.info(f"Anthropic LLM backend ready: {model}")

    async def complete(self, system: str, user: str) -> str:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()


# ── Ollama local LLM backend ──────────────────────────────────────────────────

class OllamaLLMBackend(BaseLLMBackend):
    """
    Local LLM via Ollama — runs Llama 3, Mistral, Phi-3, etc.

    Recommended models for report generation:
      ollama pull llama3.2:3b    # 2.0 GB — fast, good quality
      ollama pull mistral:7b     # 4.1 GB — excellent reasoning
      ollama pull phi3:mini      # 2.2 GB — very fast, decent quality
    """
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434") -> None:
        import aiohttp
        self._model    = model
        self._base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def complete(self, system: str, user: str) -> str:
        import aiohttp
        if not self._session:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                return data["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Ollama LLM error: {e}")
            return ""


# ── Factory ───────────────────────────────────────────────────────────────────

def _build_llm_backend(name: str) -> BaseLLMBackend:
    try:
        if name == "groq" and settings.GROQ_API_KEY:
            return GroqLLMBackend(model=settings.LLM_MODEL)
        if name == "openai" and settings.OPENAI_API_KEY:
            return OpenAILLMBackend(model=settings.LLM_MODEL)
        if name == "anthropic" and settings.ANTHROPIC_API_KEY:
            return AnthropicLLMBackend(model=settings.LLM_MODEL)
        if name == "ollama":
            return OllamaLLMBackend(model=settings.OLLAMA_LLM_MODEL, base_url=settings.OLLAMA_BASE_URL)
    except Exception as e:
        logger.warning(f"LLM backend '{name}' failed: {e}")

    logger.info("LLM backend: mock mode")
    return MockLLMBackend(latency_ms=settings.VLM_MOCK_LATENCY_MS)


# ── High-level LLM client ─────────────────────────────────────────────────────

class LLMClient:
    """
    High-level LLM client for generating zone summaries,
    shift reports, and anomaly explanations.
    """

    _SYSTEM_PROMPT = (
        "You are an AI safety analyst for an industrial warehouse surveillance system. "
        "You analyze worker activity data to identify safety risks, operational inefficiencies, "
        "and security concerns. Write in clear, professional English. "
        "Be concise and actionable. Do not speculate beyond the evidence provided."
    )

    def __init__(self) -> None:
        self._backend = _build_llm_backend(settings.LLM_BACKEND)

    async def generate_zone_summary(
        self,
        zone_id:      str,
        zone_name:    str,
        activities:   list[dict],
        alerts:       list[dict],
        window_min:   int = 15,
    ) -> ZoneSummary:
        """
        Generate a natural-language summary for one zone over a time window.

        Args:
            zone_id:    Zone identifier
            zone_name:  Human-readable zone name
            activities: List of activity dicts from the database
            alerts:     List of alert dicts from the database
            window_min: Time window in minutes

        Returns:
            ZoneSummary with narrative, risk level, and key events
        """
        n_persons = len({a.get("person_id") for a in activities})
        n_alerts  = len(alerts)
        alert_types = [a.get("alert_type", "unknown") for a in alerts]

        # Use mock template for mock backend
        if isinstance(self._backend, MockLLMBackend):
            return self._mock_zone_summary(zone_id, zone_name, n_persons, n_alerts, window_min)

        user_prompt = (
            f"Zone: {zone_name} (ID: {zone_id})\n"
            f"Time window: last {window_min} minutes\n"
            f"Unique persons detected: {n_persons}\n"
            f"Total activity events: {len(activities)}\n"
            f"Alerts raised: {n_alerts} ({', '.join(alert_types) if alert_types else 'none'})\n\n"
            f"Recent activities:\n"
            + "\n".join(f"  - {a.get('description', '')}" for a in activities[-10:])
            + "\n\nGenerate a concise zone summary (2-3 sentences), a risk level (low/medium/high), "
            + "and 2-3 key events as bullet points."
        )

        t0 = time.monotonic()
        raw = await self._backend.complete(self._SYSTEM_PROMPT, user_prompt)
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Parse risk level from response
        risk = "high" if n_alerts >= 3 else "medium" if n_alerts >= 1 else "low"
        for keyword, level in [("high risk", "high"), ("critical", "high"),
                                ("medium risk", "medium"), ("caution", "medium"),
                                ("low risk", "low"), ("normal", "low")]:
            if keyword in raw.lower():
                risk = level
                break

        key_events = [a.get("description", "")[:80] for a in alerts[:3]]

        return ZoneSummary(
            zone_id=zone_id, zone_name=zone_name,
            time_window_min=window_min,
            summary=raw, risk_level=risk,
            key_events=key_events,
            person_count=n_persons, alert_count=n_alerts,
            generated_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
        )

    async def generate_shift_report(
        self,
        zone_summaries: list[ZoneSummary],
        all_alerts:     list[dict],
        shift:          str = "current",
    ) -> ShiftReport:
        """Generate a consolidated shift report across all zones."""
        total_alerts = len(all_alerts)
        high_sev     = sum(1 for a in all_alerts if a.get("severity") == "high")

        if isinstance(self._backend, MockLLMBackend):
            summary = (
                f"Shift monitoring report: {total_alerts} total alerts across {len(zone_summaries)} zones. "
                f"{high_sev} high-severity incidents require follow-up. "
                f"Overall warehouse operations appear {'disrupted' if high_sev > 2 else 'within normal parameters'}. "
                f"Recommend reviewing access logs for restricted area entries "
                f"and ensuring all personnel completed PPE compliance checks."
            )
            recommendations = [
                "Review badge access logs for restricted area entries",
                "Schedule PPE compliance briefing for all shift workers",
                "Increase patrol frequency in loading zone during peak hours",
            ]
        else:
            zone_text = "\n".join(
                f"  {s.zone_name}: risk={s.risk_level}, alerts={s.alert_count}"
                for s in zone_summaries
            )
            user_prompt = (
                f"Shift: {shift}\n"
                f"Total alerts: {total_alerts} ({high_sev} high severity)\n"
                f"Zone breakdown:\n{zone_text}\n\n"
                f"Generate: 1) A 3-sentence consolidated shift summary, "
                f"2) 3 actionable recommendations for the next shift."
            )
            t0 = time.monotonic()
            raw = await self._backend.complete(self._SYSTEM_PROMPT, user_prompt)
            summary = raw
            recommendations = [
                line.strip().lstrip("•-123456789. ")
                for line in raw.splitlines()
                if line.strip() and any(line.strip().startswith(c) for c in ["•", "-", "1", "2", "3"])
            ][:3]

        return ShiftReport(
            report_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            shift=shift,
            summary=summary,
            zone_summaries=zone_summaries,
            total_alerts=total_alerts,
            high_severity=high_sev,
            recommendations=recommendations or ["Continue standard monitoring protocols."],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    async def explain_anomaly(
        self,
        alert_type:  str,
        description: str,
        zone_name:   str,
        person_id:   str,
        dwell_time:  float = 0.0,
    ) -> AnomalyExplanation:
        """
        Generate a detailed explanation for why an alert was raised
        and what action security should take.
        """
        if isinstance(self._backend, MockLLMBackend) or alert_type in _ANOMALY_EXPLANATIONS:
            base = _ANOMALY_EXPLANATIONS.get(
                alert_type,
                ("Anomalous behavior detected based on CV analysis.", "Verify the situation manually.")
            )
            return AnomalyExplanation(
                alert_id=      f"alert-{person_id}-{alert_type}",
                alert_type=    alert_type,
                explanation=   base[0],
                context=       f"Zone: {zone_name} | Person: {person_id} | Dwell: {dwell_time:.0f}s",
                recommendation=base[1],
                false_positive_probability="low" if dwell_time > 30 else "medium",
            )

        user_prompt = (
            f"Alert type: {alert_type}\n"
            f"Zone: {zone_name}\n"
            f"Person ID: {person_id}\n"
            f"Dwell time: {dwell_time:.0f} seconds\n"
            f"AI description: {description}\n\n"
            f"Explain: 1) why this was flagged, 2) what normally happens here, "
            f"3) recommended action, 4) false positive probability (low/medium/high)."
        )

        t0 = time.monotonic()
        raw = await self._backend.complete(self._SYSTEM_PROMPT, user_prompt)
        latency_ms = int((time.monotonic() - t0) * 1000)

        return AnomalyExplanation(
            alert_id=      f"alert-{person_id}-{alert_type}",
            alert_type=    alert_type,
            explanation=   raw,
            context=       f"Zone: {zone_name} | Person: {person_id}",
            recommendation="Follow standard security protocol for this alert type.",
            false_positive_probability="medium",
            latency_ms=    latency_ms,
        )

    # ── Mock helpers ──────────────────────────────────────────────────────────

    def _mock_zone_summary(
        self,
        zone_id:   str,
        zone_name: str,
        n_persons: int,
        n_alerts:  int,
        window_min: int,
    ) -> ZoneSummary:
        template = _MOCK_ZONE_SUMMARIES.get(zone_id, _MOCK_ZONE_SUMMARIES["storage_area"])
        activity_level = "high" if n_persons > 5 else "moderate" if n_persons > 2 else "low"
        event = "alert was" if n_alerts == 1 else "alerts were"
        summary = template.format(
            n_persons=n_persons, n_alerts=n_alerts,
            activity_level=activity_level, event=event,
        )
        risk = "high" if n_alerts >= 3 else "medium" if n_alerts >= 1 else "low"
        return ZoneSummary(
            zone_id=zone_id, zone_name=zone_name,
            time_window_min=window_min,
            summary=summary, risk_level=risk,
            key_events=[f"Zone monitored for {window_min} minutes", f"{n_persons} unique persons tracked"],
            person_count=n_persons, alert_count=n_alerts,
            generated_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=85,
        )


# ── Backend: Groq (fast cloud LLM, free tier available) ───────────────────────

class GroqLLMBackend(BaseLLMBackend):
    """
    Groq Cloud LLM backend — fastest available cloud inference.

    Why Groq for warehouse surveillance?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Speed:  Llama-3.3-70B runs at 1000+ tokens/sec (vs ~80 for GPT-4o-mini)
    • Cost:   Free tier: 14,400 requests/day, 500,000 tokens/minute
    • Models: Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B, Mixtral 8x7B
    • Latency: ~200ms for a 200-token report summary

    Setup:
        1. Sign up at https://console.groq.com (free)
        2. Create an API key
        3. Add to .env:  GROQ_API_KEY=gsk_...
        4. Set:          LLM_BACKEND=groq

    Recommended models:
        llama-3.3-70b-versatile  — best quality, free tier
        llama-3.1-8b-instant     — fastest, good for summaries
        gemma2-9b-it             — Google's model, balanced
        mixtral-8x7b-32768       — long context (good for shift reports)
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        from groq import AsyncGroq
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model  = model
        logger.info(f"Groq LLM backend ready: {model}")

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=      self._model,
            max_tokens= 600,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content.strip()
