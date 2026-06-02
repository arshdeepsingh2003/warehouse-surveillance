"""
ai/llm/zone_summarizer.py
──────────────────────────
Zone Summarizer — periodically generates LLM-powered zone summaries.

Runs as a background asyncio task. Every SUMMARY_INTERVAL_SECONDS:
  1. Reads recent activities + alerts from the local buffer
  2. Calls LLMClient.generate_zone_summary() for each zone
  3. Posts summaries to the backend API
  4. Broadcasts via WebSocket so the dashboard updates in real time

This decouples summary generation from the per-frame pipeline.
The frame pipeline runs at 10fps; summaries run every 30 seconds.

Architecture:
  AIFrameProcessor → activity_buffer → ZoneSummarizer → backend API
                                                      → WebSocket
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ai.llm.llm_client import LLMClient, ZoneSummary, ShiftReport
from ai.zones.zone_config import get_zones_for_camera, ZONE_CONFIG
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ActivityBuffer:
    """Thread-safe buffer of recent activities and alerts per zone."""
    activities: list[dict] = field(default_factory=list)
    alerts:     list[dict] = field(default_factory=list)
    MAX_SIZE: int = 200

    def add_activity(self, activity: dict) -> None:
        self.activities.append(activity)
        if len(self.activities) > self.MAX_SIZE:
            self.activities = self.activities[-self.MAX_SIZE:]

    def add_alert(self, alert: dict) -> None:
        self.alerts.append(alert)
        if len(self.alerts) > self.MAX_SIZE:
            self.alerts = self.alerts[-self.MAX_SIZE:]

    def flush(self) -> tuple[list[dict], list[dict]]:
        """Return and clear the buffer contents."""
        acts   = list(self.activities)
        alerts = list(self.alerts)
        self.activities.clear()
        self.alerts.clear()
        return acts, alerts


class ZoneSummarizer:
    """
    Background service that generates LLM zone summaries on a schedule.

    Usage:
        summarizer = ZoneSummarizer(api_client, llm_client)
        # In your pipeline, feed it data:
        summarizer.log_activity(activity_dict)
        summarizer.log_alert(alert_dict)
        # Start the background loop:
        asyncio.create_task(summarizer.run())
    """

    def __init__(self, api_client, llm_client: Optional[LLMClient] = None) -> None:
        self._api  = api_client
        self._llm  = llm_client or LLMClient()

        # Per-zone activity buffers
        self._buffers: dict[str, ActivityBuffer] = defaultdict(ActivityBuffer)

        # Most recent summary per zone (for dashboard API)
        self._latest_summaries: dict[str, ZoneSummary] = {}
        self._latest_report:    Optional[ShiftReport]  = None

    # ── Data ingestion (called from AIFrameProcessor) ─────────────────────────

    def log_activity(self, activity: dict) -> None:
        zone_id = activity.get("zone", "unknown")
        self._buffers[zone_id].add_activity(activity)

    def log_alert(self, alert: dict) -> None:
        zone_id = alert.get("zone", "unknown")
        self._buffers[zone_id].add_alert(alert)

    # ── Background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run zone summaries on a configurable schedule."""
        interval = settings.SUMMARY_INTERVAL_SECONDS
        logger.info(f"ZoneSummarizer started (interval={interval}s, backend={settings.LLM_BACKEND})")

        while True:
            await asyncio.sleep(interval)
            try:
                await self._generate_all_summaries()
            except Exception as e:
                logger.error(f"ZoneSummarizer error: {e}")

    async def _generate_all_summaries(self) -> None:
        """Generate summaries for all zones that have recent data."""
        # Collect all known zone IDs from the camera config
        all_zone_ids: set[str] = set()
        for zones in ZONE_CONFIG.values():
            for z in zones:
                all_zone_ids.add(z.zone_id)

        zone_summaries: list[ZoneSummary] = []
        all_alerts:     list[dict]         = []

        for zone_id in all_zone_ids:
            buffer = self._buffers.get(zone_id)
            if not buffer:
                continue

            activities, alerts = buffer.flush()
            all_alerts.extend(alerts)

            if not activities and not alerts:
                continue

            # Find zone display name
            zone_name = zone_id.replace("_", " ").title()
            for zones in ZONE_CONFIG.values():
                for z in zones:
                    if z.zone_id == zone_id:
                        zone_name = z.display_name

            logger.info(
                f"Generating summary: {zone_name} "
                f"({len(activities)} activities, {len(alerts)} alerts)"
            )

            summary = await self._llm.generate_zone_summary(
                zone_id=zone_id, zone_name=zone_name,
                activities=activities, alerts=alerts,
                window_min=max(1, settings.SUMMARY_INTERVAL_SECONDS // 60),
            )

            self._latest_summaries[zone_id] = summary
            zone_summaries.append(summary)

            # Post to backend
            await self._post_summary(summary)

        # Generate shift report if we have summaries
        if zone_summaries:
            shift = self._current_shift()
            report = await self._llm.generate_shift_report(
                zone_summaries=zone_summaries,
                all_alerts=all_alerts,
                shift=shift,
            )
            self._latest_report = report
            await self._post_shift_report(report)

    async def _post_summary(self, summary: ZoneSummary) -> None:
        """Post zone summary to backend and broadcast via WebSocket."""
        payload = {
            "zone_id":      summary.zone_id,
            "zone_name":    summary.zone_name,
            "summary":      summary.summary,
            "risk_level":   summary.risk_level,
            "key_events":   summary.key_events,
            "person_count": summary.person_count,
            "alert_count":  summary.alert_count,
            "generated_at": summary.generated_at,
        }
        # Store via ingest endpoint
        await self._api._post("/api/v1/summaries/ingest", payload)

        # Broadcast to dashboard
        await self._api._post("/api/v1/events/broadcast", {
            "type":    "zone_summary",
            "payload": payload,
        })

    async def _post_shift_report(self, report: ShiftReport) -> None:
        """Post the consolidated shift report."""
        payload = {
            "report_date":     report.report_date,
            "shift":           report.shift,
            "summary":         report.summary,
            "total_alerts":    report.total_alerts,
            "high_severity":   report.high_severity,
            "recommendations": report.recommendations,
            "generated_at":    report.generated_at,
        }
        await self._api._post("/api/v1/summaries/shift-report", payload)
        await self._api._post("/api/v1/events/broadcast", {
            "type":    "shift_report",
            "payload": payload,
        })
        logger.info(
            f"Shift report posted: {report.total_alerts} alerts, "
            f"{report.high_severity} high severity"
        )

    def get_latest_summary(self, zone_id: str) -> Optional[ZoneSummary]:
        return self._latest_summaries.get(zone_id)

    def get_latest_report(self) -> Optional[ShiftReport]:
        return self._latest_report

    @staticmethod
    def _current_shift() -> str:
        hour = datetime.now().hour
        if 6 <= hour < 14:  return "morning"
        if 14 <= hour < 22: return "afternoon"
        return "night"
