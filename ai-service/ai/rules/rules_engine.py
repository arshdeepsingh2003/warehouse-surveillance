"""
ai/rules/rules_engine.py
─────────────────────────
Rules Engine — converts activity classifications into structured alerts.

Architecture:
  ActivityResult  →  RulesEngine  →  AlertEvent (or None)

The rules engine is the policy layer. It answers the question:
  "Is this activity serious enough to raise an alert?"

Responsibilities:
  1. Apply configurable severity rules per anomaly type
  2. Enforce alert cooldown (no duplicate alerts within N seconds)
  3. Escalate severity based on context (e.g. restricted zone = always high)
  4. Add structured metadata to every alert (who, where, when, snapshot)

Design principle: rules are data, not code.
  The RULES dict defines everything. To change policy:
    • Edit the RULES dict (or load from a JSON config file)
    • No code change needed
  This also makes rules auditable and versionable.

Alert output:
  AlertEvent = {
    alert_type:  str       (matches backend schema)
    severity:    str       (low | medium | high)
    description: str
    person_id:   str
    camera_id:   str
    zone_id:     str
    confidence:  float
  }
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ai.analyzer.activity_analyzer import ActivityResult, AnomalyFlag
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Rule definition ───────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    One configured monitoring rule.

    trigger_flag:   The AnomalyFlag that activates this rule.
    alert_type:     Alert type string sent to the backend.
    severity:       "low" | "medium" | "high"
    min_dwell:      Minimum dwell time (seconds) before alert fires.
    description_fn: Callable that builds the alert description string.
    enabled:        Easy on/off switch.
    """
    trigger_flag:  str
    alert_type:    str
    severity:      str
    min_dwell:     float = 0.0
    description_fn: Optional[object] = None  # Callable[[ActivityResult], str]
    enabled:       bool  = True


# ── Rules registry ────────────────────────────────────────────────────────────
# Each rule maps an AnomalyFlag to an alert configuration.
# Rules are evaluated in order — first match wins.

RULES: list[Rule] = [
    Rule(
        trigger_flag  = AnomalyFlag.RESTRICTED_ZONE,
        alert_type    = "unauthorized_access",
        severity      = "high",
        min_dwell     = 0.0,   # fire immediately on entry
        description_fn= lambda r: (
            f"Unauthorized person {r.person_id} detected in restricted zone "
            f"'{r.zone_name}'. Immediate security response required."
        ),
    ),
    Rule(
        trigger_flag  = AnomalyFlag.POSSIBLE_FALL,
        alert_type    = "worker_fall",
        severity      = "high",
        min_dwell     = 5.0,   # wait 5s to avoid false positives from bending/crouching
        description_fn= lambda r: (
            f"Worker {r.person_id} may have fallen in '{r.zone_name}'. "
            f"Stationary for {int(r.dwell_time)}s. Medical assistance may be needed."
        ),
    ),
    Rule(
        trigger_flag  = AnomalyFlag.LOITERING,
        alert_type    = "loitering",
        severity      = "medium",
        min_dwell     = float(settings.LOITERING_SECONDS),
        description_fn= lambda r: (
            f"Person {r.person_id} has been loitering in '{r.zone_name}' "
            f"for {int(r.dwell_time)} seconds."
        ),
    ),
    Rule(
        trigger_flag  = AnomalyFlag.FAST_MOVEMENT,
        alert_type    = "suspicious_activity",
        severity      = "medium",
        min_dwell     = 0.0,
        description_fn= lambda r: (
            f"Person {r.person_id} is running in '{r.zone_name}'. "
            f"Running is prohibited in warehouse areas."
        ),
    ),
    Rule(
        trigger_flag  = AnomalyFlag.LONG_DWELL,
        alert_type    = "loitering",
        severity      = "low",
        min_dwell     = float(settings.LOITERING_SECONDS * 2),
        description_fn= lambda r: (
            f"Person {r.person_id} has been in '{r.zone_name}' "
            f"for {int(r.dwell_time)}s without activity."
        ),
    ),
    Rule(
        trigger_flag  = AnomalyFlag.PPE_ZONE_VIOLATION,
        alert_type    = "ppe_violation",
        severity      = "low",
        min_dwell     = 3.0,
        description_fn= lambda r: (
            f"Possible PPE violation detected for {r.person_id} in '{r.zone_name}'. "
            f"Please verify helmet and vest compliance."
        ),
    ),
]


# ── Alert event ───────────────────────────────────────────────────────────────

@dataclass
class AlertEvent:
    """Structured alert ready to be posted to the backend API."""
    alert_type:  str
    severity:    str
    description: str
    person_id:   str
    camera_id:   str
    zone_id:     str
    zone_name:   str
    confidence:  float
    dwell_time:  float
    track_uuid:  str    = ""    # stable UUID for VLM cache lookup
    flags:       list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "alert_type":  self.alert_type,
            "severity":    self.severity,
            "description": self.description,
            "person_id":   self.person_id,
            "camera_id":   self.camera_id,
            "zone":        self.zone_id,
            "confidence":  round(self.confidence, 3),
        }


# ── Rules engine ──────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Evaluates activity results against configured rules and generates alerts.

    One instance per camera — maintains cooldown state per (camera, person, alert_type).

    Usage:
        engine = RulesEngine(camera_id="cam-01")
        alerts = engine.evaluate(activity_results)
        for alert in alerts:
            await api_client.post_alert(**alert.to_dict())
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        # Cooldown tracker: (person_id, alert_type) → last_fired_timestamp
        self._last_fired: dict[tuple[str, str], float] = {}

    def evaluate(self, results: list[ActivityResult]) -> list[AlertEvent]:
        """
        Evaluate activity results against all rules.

        Args:
            results: List of ActivityResult from ActivityAnalyzer.analyze()

        Returns:
            List of AlertEvent to be posted to backend. May be empty.
        """
        alerts = []

        for result in results:
            if not result.is_anomaly:
                continue   # Normal activities don't trigger rules

            for rule in RULES:
                if not rule.enabled:
                    continue
                if rule.trigger_flag not in result.flags:
                    continue
                if result.dwell_time < rule.min_dwell:
                    continue   # Dwell threshold not met yet

                # Check cooldown
                cooldown_key = (result.person_id, rule.alert_type)
                now = time.monotonic()
                last = self._last_fired.get(cooldown_key, 0.0)
                if now - last < settings.ALERT_COOLDOWN_SECONDS:
                    logger.debug(
                        f"[{self.camera_id}] Alert {rule.alert_type} for "
                        f"{result.person_id} suppressed (cooldown)"
                    )
                    continue

                # Build description
                desc = (
                    rule.description_fn(result)
                    if rule.description_fn
                    else result.description
                )

                alert = AlertEvent(
                    alert_type=  rule.alert_type,
                    severity=    rule.severity,
                    description= desc,
                    person_id=   result.person_id,
                    track_uuid=  result.track_uuid,
                    camera_id=   self.camera_id,
                    zone_id=     result.zone_id,
                    zone_name=   result.zone_name,
                    confidence=  result.confidence,
                    dwell_time=  result.dwell_time,
                    flags=       result.flags,
                )

                alerts.append(alert)
                self._last_fired[cooldown_key] = now

                logger.info(
                    f"🚨 [{self.camera_id}] ALERT: [{alert.severity.upper()}] "
                    f"{alert.alert_type} | {alert.person_id} | {alert.zone_name}"
                )

                break   # First matching rule wins per person per frame

        # Prune stale cooldown entries (older than 10 minutes)
        cutoff = time.monotonic() - 600
        self._last_fired = {k: v for k, v in self._last_fired.items() if v > cutoff}

        return alerts

    def add_rule(self, rule: Rule) -> None:
        """Dynamically add a new rule at runtime."""
        RULES.append(rule)
        logger.info(f"Rule added: {rule.trigger_flag} → {rule.alert_type}")

    def disable_rule(self, alert_type: str) -> None:
        """Temporarily disable a rule by alert_type."""
        for rule in RULES:
            if rule.alert_type == alert_type:
                rule.enabled = False
                logger.info(f"Rule disabled: {alert_type}")
