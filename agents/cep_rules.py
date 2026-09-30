"""CEP Rule definitions for the PoMAS alerting system.

Defines alert thresholds, temporal rules, and escalation policies.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    name: str
    tier: str
    severity: Severity
    min_risk: float
    max_risk: float = 1.0
    consecutive_windows: int = 1
    cooldown_seconds: int = 300
    notify_clinician: bool = False


DEFAULT_RULES = [
    AlertRule(
        name="low_risk_log",
        tier="silent_log",
        severity=Severity.INFO,
        min_risk=0.3,
        max_risk=0.5,
        consecutive_windows=3,
    ),
    AlertRule(
        name="medium_risk_flag",
        tier="ehr_flag",
        severity=Severity.WARNING,
        min_risk=0.5,
        max_risk=0.7,
        consecutive_windows=2,
        cooldown_seconds=120,
    ),
    AlertRule(
        name="high_risk_alert",
        tier="clinician_notify",
        severity=Severity.CRITICAL,
        min_risk=0.7,
        consecutive_windows=1,
        cooldown_seconds=60,
        notify_clinician=True,
    ),
]


def evaluate_rules(risk_score: float, rules: list = None) -> Optional[AlertRule]:
    """Evaluate risk score against rules, return first matched rule."""
    rules = rules or DEFAULT_RULES
    for rule in sorted(rules, key=lambda r: r.min_risk, reverse=True):
        if rule.min_risk <= risk_score <= rule.max_risk:
            return rule
    return None
