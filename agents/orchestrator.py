"""Three-agent CEP Orchestration for PoMAS.

Phase 4 & 5: Monitoring Agent, Prediction Agent, Alert Agent.
Coordinates incoming EEG windows, consecutive-window suspicion filtering,
prediction execution, and CEP refractory-lockout alert tiering.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from agents.event_store import EventStore

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    EEG_WINDOW = "eeg_window"
    SUSPICIOUS_WINDOW = "suspicious_window"
    PREDICT_REQUEST = "predict_request"
    RISK_SCORE = "risk_score"
    ALERT = "alert"


class AlertTier(str, Enum):
    SILENT_LOG = "silent_log"
    EHR_FLAG = "ehr_flag"
    CLINICIAN_NOTIFY = "clinician_notify"


@dataclass
class Event:
    type: EventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    patient_id: str = ""
    recording_id: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class MonitoringAgent:
    """Tracks consecutive suspicious windows and fires PREDICT_REQUEST events."""

    def __init__(self, consecutive_threshold: int = 2, suspicious_threshold: float = 0.30):
        self.consecutive_threshold = consecutive_threshold
        self.suspicious_threshold = suspicious_threshold
        self._counters: Dict[str, int] = {}

    def process(self, event: Event) -> Optional[Event]:
        pid = event.patient_id
        if pid not in self._counters:
            self._counters[pid] = 0

        # Check if the incoming window has preliminary suspicion flag or score
        score = event.data.get("screening_score", event.data.get("risk_score", 0.0))
        is_suspicious = score >= self.suspicious_threshold

        if is_suspicious:
            self._counters[pid] += 1
        else:
            self._counters[pid] = max(0, self._counters[pid] - 1)

        if self._counters[pid] >= self.consecutive_threshold:
            # Threshold met: escalate to PredictionAgent
            return Event(
                type=EventType.PREDICT_REQUEST,
                data={**event.data, "consecutive_count": self._counters[pid]},
                timestamp=event.timestamp,
                patient_id=pid,
                recording_id=event.recording_id,
            )
        return None

    def reset(self, patient_id: Optional[str] = None) -> None:
        if patient_id:
            self._counters[patient_id] = 0
        else:
            self._counters.clear()


class PredictionAgent:
    """Consumes PREDICT_REQUEST, executes prediction model, returns RISK_SCORE."""

    def __init__(self, predictor_fn: Callable[[dict], dict]):
        self.predictor_fn = predictor_fn

    def process(self, event: Event) -> Event:
        risk_assessment = self.predictor_fn(event.data)
        return Event(
            type=EventType.RISK_SCORE,
            data={**event.data, **risk_assessment},
            timestamp=event.timestamp,
            patient_id=event.patient_id,
            recording_id=event.recording_id,
        )


class AlertAgent:
    """Applies CEP refractory cooldowns and produces tiered clinical alerts."""

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        refractory_seconds: float = 1800.0,  # 30-minute clinical refractory period
    ):
        self.thresholds = thresholds or {
            "silent_log": 0.30,
            "ehr_flag": 0.50,
            "clinician_notify": 0.70,
        }
        self.refractory_seconds = refractory_seconds
        self._last_alert_time: Dict[str, float] = {}
        self.total_alerts_fired: int = 0
        self.suppressed_refractory_count: int = 0

    def process(self, event: Event) -> Optional[Event]:
        risk = float(event.data.get("risk_score", 0.0))
        pid = event.patient_id

        # Determine alert tier
        if risk >= self.thresholds["clinician_notify"]:
            tier = AlertTier.CLINICIAN_NOTIFY
        elif risk >= self.thresholds["ehr_flag"]:
            tier = AlertTier.EHR_FLAG
        elif risk >= self.thresholds["silent_log"]:
            tier = AlertTier.SILENT_LOG
        else:
            return None

        # Check refractory cooldown to prevent spamming alarms on every 5s window
        last_time = self._last_alert_time.get(pid, -float("inf"))
        current_time = event.timestamp

        if (current_time - last_time) < self.refractory_seconds:
            self.suppressed_refractory_count += 1
            return None

        self._last_alert_time[pid] = current_time
        self.total_alerts_fired += 1

        return Event(
            type=EventType.ALERT,
            data={
                **event.data,
                "alert_tier": tier.value,
                "refractory_seconds": self.refractory_seconds,
            },
            timestamp=event.timestamp,
            patient_id=pid,
            recording_id=event.recording_id,
        )

    def reset(self, patient_id: Optional[str] = None) -> None:
        if patient_id:
            self._last_alert_time.pop(patient_id, None)
        else:
            self._last_alert_time.clear()
        self.total_alerts_fired = 0
        self.suppressed_refractory_count = 0


class Orchestrator:
    """Coordinates Monitoring, Prediction, and Alert Agents with the EventStore."""

    def __init__(
        self,
        predictor_fn: Callable[[dict], dict],
        event_store: Optional[EventStore] = None,
        consecutive_threshold: int = 2,
        refractory_seconds: float = 1800.0,
    ):
        self.monitor = MonitoringAgent(consecutive_threshold=consecutive_threshold)
        self.predictor = PredictionAgent(predictor_fn)
        self.alerter = AlertAgent(refractory_seconds=refractory_seconds)
        self.event_store = event_store

    def process_window(
        self,
        window_data: dict,
        patient_id: str,
        timestamp: float,
        recording_id: str = "",
        ground_truth: Optional[int] = None,
    ) -> Optional[dict]:
        """Process a single EEG window through the agentic pipeline."""
        eeg_event = Event(
            type=EventType.EEG_WINDOW,
            data=window_data,
            timestamp=timestamp,
            patient_id=patient_id,
            recording_id=recording_id,
        )

        # 1. Monitoring Agent checks for suspicious patterns
        pred_request = self.monitor.process(eeg_event)
        if pred_request is None:
            # Not yet escalated to PredictionAgent
            if self.event_store:
                self.event_store.record_event(
                    event_type=EventType.EEG_WINDOW.value,
                    patient_id=patient_id,
                    timestamp=timestamp,
                    recording_id=recording_id,
                    ground_truth=ground_truth,
                    risk_score=window_data.get("screening_score"),
                )
            return None

        # 2. Prediction Agent assesses risk
        risk_event = self.predictor.process(pred_request)
        risk_score = risk_event.data.get("risk_score")

        # 3. Alert Agent applies CEP rules & refractory lockout
        alert_event = self.alerter.process(risk_event)
        alert_tier = alert_event.data.get("alert_tier") if alert_event else None

        # 4. Record to Event Store
        if self.event_store:
            self.event_store.record_event(
                event_type=EventType.ALERT.value if alert_event else EventType.RISK_SCORE.value,
                patient_id=patient_id,
                timestamp=timestamp,
                recording_id=recording_id,
                ground_truth=ground_truth,
                risk_score=risk_score,
                alert_tier=alert_tier,
                details={
                    "method": risk_event.data.get("method_used", "EfficientNet-B0 + SVM"),
                    "consecutive_count": pred_request.data.get("consecutive_count"),
                },
            )

        return {
            "timestamp": timestamp,
            "patient_id": patient_id,
            "recording_id": recording_id,
            "risk_score": risk_score,
            "ground_truth": ground_truth,
            "alert": asdict(alert_event) if alert_event else None,
            "alert_tier": alert_tier,
        }
