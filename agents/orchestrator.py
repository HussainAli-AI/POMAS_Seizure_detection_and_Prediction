"""Three-agent CEP Orchestration.

Phase 5: Monitoring Agent, Prediction Agent, Alert Agent.
Implements a lightweight FSM-based event processing pipeline.
"""

import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    EEG_WINDOW = "eeg_window"
    SUSPICIOUS_WINDOW = "suspicious_window"
    PREDICT_REQUEST = "predict_request"
    RISK_SCORE = "risk_score"
    ALERT = "alert"


class AlertTier(Enum):
    SILENT_LOG = "silent_log"
    EHR_FLAG = "ehr_flag"
    CLINICIAN_NOTIFY = "clinician_notify"


@dataclass
class Event:
    type: EventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    patient_id: str = ""
    event_id: str = ""


class MonitoringAgent:
    """Tracks consecutive suspicious windows and fires PREDICT_REQUEST events."""

    def __init__(self, consecutive_threshold: int = 3):
        self.consecutive_threshold = consecutive_threshold
        self._counters = {}

    def process(self, event: Event) -> Optional[Event]:
        pid = event.patient_id
        if pid not in self._counters:
            self._counters[pid] = 0

        is_suspicious = event.data.get("risk_score", 0) > 0.3
        if is_suspicious:
            self._counters[pid] += 1
        else:
            self._counters[pid] = max(0, self._counters[pid] - 1)

        if self._counters[pid] >= self.consecutive_threshold:
            self._counters[pid] = 0
            return Event(
                type=EventType.PREDICT_REQUEST,
                data=event.data,
                patient_id=pid,
            )
        return None


class PredictionAgent:
    """Consumes PREDICT_REQUEST, runs predictor, returns RISK_SCORE."""

    def __init__(self, predictor_fn: Callable):
        self.predictor_fn = predictor_fn

    def process(self, event: Event) -> Event:
        risk_assessment = self.predictor_fn(event.data)
        return Event(
            type=EventType.RISK_SCORE,
            data={**event.data, **risk_assessment},
            patient_id=event.patient_id,
        )


class AlertAgent:
    """Applies tiering and generates alerts."""

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or {
            "silent_log": 0.3,
            "ehr_flag": 0.5,
            "clinician_notify": 0.7,
        }

    def process(self, event: Event) -> Optional[Event]:
        risk = event.data.get("risk_score", 0)
        if risk >= self.thresholds["clinician_notify"]:
            tier = AlertTier.CLINICIAN_NOTIFY
        elif risk >= self.thresholds["ehr_flag"]:
            tier = AlertTier.EHR_FLAG
        elif risk >= self.thresholds["silent_log"]:
            tier = AlertTier.SILENT_LOG
        else:
            return None

        return Event(
            type=EventType.ALERT,
            data={**event.data, "alert_tier": tier.value, "alert_time": time.time()},
            patient_id=event.patient_id,
        )


class Orchestrator:
    """Main orchestrator connecting the three agents."""

    def __init__(self, predictor_fn: Callable, event_store: Optional[Callable] = None):
        self.monitor = MonitoringAgent()
        self.predictor = PredictionAgent(predictor_fn)
        self.alerter = AlertAgent()
        self.event_store = event_store or (lambda e: None)

    def process_window(self, window_data: dict, patient_id: str) -> Optional[dict]:
        eeg_event = Event(
            type=EventType.EEG_WINDOW,
            data=window_data,
            patient_id=patient_id,
        )
        self.event_store(eeg_event)

        pred_request = self.monitor.process(eeg_event)
        if pred_request is None:
            return None

        risk_event = self.predictor.process(pred_request)
        self.event_store(risk_event)

        alert_event = self.alerter.process(risk_event)

        return {
            "risk_score": risk_event.data.get("risk_score"),
            "uncertainty": risk_event.data.get("uncertainty"),
            "method_used": risk_event.data.get("method_used"),
            "alert": asdict(alert_event) if alert_event else None,
        }
