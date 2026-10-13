"""Offline Multi-Agent Replay Pipeline for PoMAS.

Phase 4 & 5 Integration:
Replays continuous patient EEG test streams chronologically through the
three-agent CEP orchestrator (Monitoring -> Prediction -> Alert).
Enforces consecutive-window suspicion filtering and clinical refractory lockouts.
Logs all events to the persistent SQLite EventStore and reports consolidated
event-level metrics (FAR/h, Warning Latency, Episode Sensitivity).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from agents.event_store import EventStore
from agents.orchestrator import AlertTier, Orchestrator
from config import OUTPUT_DIR, STRIDE_SECONDS


def run_replay(
    subject: str = "chb01",
    refractory_minutes: float = 30.0,
    consecutive_windows: int = 2,
    screening_threshold: float = 0.20,
    clear_db: bool = True,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute chronological offline agent replay for a specified subject."""
    refractory_seconds = refractory_minutes * 60.0
    h5_path = OUTPUT_DIR / "preprocessed_global_v2" / f"{subject}_preprocessed_global_v2.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 archive not found: {h5_path}")

    # Load test results metadata & precomputed test scores
    pred_results_path = OUTPUT_DIR / f"efficientnet_{subject}_prediction_results.json"
    pred_npz_path = OUTPUT_DIR / f"efficientnet_{subject}_test_predictions.npz"

    if not pred_results_path.exists() or not pred_npz_path.exists():
        raise FileNotFoundError(
            f"Prediction artifacts missing for {subject}. Expected {pred_results_path} and {pred_npz_path}"
        )

    pred_res = json.loads(pred_results_path.read_text(encoding="utf-8"))
    test_recs = pred_res.get("test_recordings", [])
    optimal_threshold = float(pred_res["test"]["threshold"])

    npz_data = np.load(pred_npz_path)
    test_scores = npz_data["scores"]

    with h5py.File(h5_path, "r") as file:
        all_labels = file["labels"][:].astype(np.int64)
        all_recs = np.asarray(file["recording_ids"][:], dtype=str)
        all_timestamps = file["timestamps"][:]

    eligible = all_labels != 2
    test_indices = np.flatnonzero(eligible & np.isin(all_recs, test_recs))

    if len(test_indices) != len(test_scores):
        raise ValueError(
            f"Index mismatch: {len(test_indices)} eligible test windows vs {len(test_scores)} scores"
        )

    # Initialize EventStore and Orchestrator
    event_store = EventStore()
    if clear_db:
        event_store.clear(patient_id=subject)

    # Prediction function mapping window index to test score
    current_window_idx = [0]

    def predictor_fn(data: dict) -> dict:
        idx = data.get("test_index", current_window_idx[0])
        score = float(test_scores[idx])
        return {
            "risk_score": score,
            "uncertainty": float(np.abs(score - optimal_threshold)),
            "method_used": "EfficientNet-B0 + SVM",
        }

    orchestrator = Orchestrator(
        predictor_fn=predictor_fn,
        event_store=event_store,
        consecutive_threshold=consecutive_windows,
        refractory_seconds=refractory_seconds,
    )
    # Set alert agent thresholds aligning with optimal operating point
    orchestrator.alerter.thresholds = {
        "silent_log": max(0.10, optimal_threshold * 0.5),
        "ehr_flag": optimal_threshold,
        "clinician_notify": min(0.85, max(0.50, optimal_threshold * 1.5)),
    }

    # Tracking variables for metrics
    total_windows = len(test_indices)
    interictal_windows = 0
    preictal_windows = 0

    alerts_fired: List[Dict[str, Any]] = []
    episode_detections: Dict[int, List[Dict[str, Any]]] = {}

    # Identify preictal episode intervals
    # A preictal episode is a contiguous block of label 1 in test recordings
    episode_id = 0
    in_episode = False
    episode_starts: Dict[int, float] = {}
    episode_ends: Dict[int, float] = {}

    for i, global_idx in enumerate(test_indices):
        lbl = int(all_labels[global_idx])
        ts = float(all_timestamps[global_idx])
        if lbl == 1:
            if not in_episode:
                in_episode = True
                episode_id += 1
                episode_starts[episode_id] = ts
            episode_ends[episode_id] = ts
        else:
            in_episode = False

    total_episodes = episode_id

    # Replay simulation
    print(f"\n=======================================================================")
    print(f" PoMAS Chronological Offline Replay Simulation - Subject: {subject}")
    print(f" Test Recordings: {', '.join(test_recs)}")
    print(f" Windows: {total_windows:,} (10s window, 5s stride)")
    print(f" CEP Consecutive Threshold: {consecutive_windows} windows")
    print(f" CEP Refractory Lockout: {refractory_minutes:.1f} minutes")
    print(f" Operating Thresholds: {orchestrator.alerter.thresholds}")
    print(f"=======================================================================\n")

    for i, global_idx in enumerate(test_indices):
        current_window_idx[0] = i
        ts = float(all_timestamps[global_idx])
        rec = str(all_recs[global_idx])
        lbl = int(all_labels[global_idx])
        score = float(test_scores[i])

        if lbl == 0:
            interictal_windows += 1
        elif lbl == 1:
            preictal_windows += 1

        window_data = {
            "test_index": i,
            "screening_score": score,  # Monitoring agent uses this to detect suspicion
            "ground_truth": lbl,
        }

        # Process window through the 3-agent orchestrator
        result = orchestrator.process_window(
            window_data=window_data,
            patient_id=subject,
            timestamp=ts,
            recording_id=rec,
            ground_truth=lbl,
        )

        if result and result.get("alert"):
            alert_info = {
                "window_idx": i,
                "timestamp": ts,
                "recording_id": rec,
                "ground_truth": lbl,
                "risk_score": result["risk_score"],
                "alert_tier": result["alert_tier"],
            }
            alerts_fired.append(alert_info)

            # Check if this alert belongs to a preictal episode
            matched_ep = None
            for ep_num in range(1, total_episodes + 1):
                if episode_starts[ep_num] <= ts <= (episode_ends[ep_num] + STRIDE_SECONDS):
                    matched_ep = ep_num
                    break

            if matched_ep is not None:
                if matched_ep not in episode_detections:
                    episode_detections[matched_ep] = []
                episode_detections[matched_ep].append(alert_info)

    # Compute Event-Level Clinical Metrics
    interictal_hours = max((interictal_windows * STRIDE_SECONDS) / 3600.0, 1e-6)
    total_hours = (total_windows * STRIDE_SECONDS) / 3600.0

    # False alarms are alerts fired during interictal windows (ground_truth == 0)
    interictal_alerts = [a for a in alerts_fired if a["ground_truth"] == 0]
    consolidated_false_alarms = len(interictal_alerts)
    false_alarm_rate_per_hour = consolidated_false_alarms / interictal_hours

    # Window-level raw false positives without CEP consolidation
    raw_false_positives = int(np.sum((test_scores >= optimal_threshold) & (np.array([all_labels[idx] for idx in test_indices]) == 0)))
    raw_fp_per_hour = raw_false_positives / interictal_hours

    # Preictal Episode Sensitivity
    detected_episodes = len(episode_detections)
    episode_sensitivity = detected_episodes / max(total_episodes, 1)

    # Lead Times (Warning Latency) before seizure onset (end of preictal episode)
    lead_times_minutes: Dict[int, float] = {}
    for ep_num, ep_alerts in episode_detections.items():
        first_alert_ts = ep_alerts[0]["timestamp"]
        seizure_onset_ts = episode_ends[ep_num] + 10.0  # approximate seizure onset
        lead_time_min = max(0.0, (seizure_onset_ts - first_alert_ts) / 60.0)
        lead_times_minutes[ep_num] = lead_time_min

    # Tier breakdown
    tier_counts = {
        AlertTier.SILENT_LOG.value: sum(1 for a in alerts_fired if a["alert_tier"] == AlertTier.SILENT_LOG.value),
        AlertTier.EHR_FLAG.value: sum(1 for a in alerts_fired if a["alert_tier"] == AlertTier.EHR_FLAG.value),
        AlertTier.CLINICIAN_NOTIFY.value: sum(1 for a in alerts_fired if a["alert_tier"] == AlertTier.CLINICIAN_NOTIFY.value),
    }

    summary = {
        "subject": subject,
        "test_recordings": test_recs,
        "total_windows": total_windows,
        "total_hours": round(total_hours, 3),
        "interictal_hours": round(interictal_hours, 3),
        "preictal_windows": preictal_windows,
        "interictal_windows": interictal_windows,
        "total_preictal_episodes": total_episodes,
        "detected_preictal_episodes": detected_episodes,
        "preictal_episode_sensitivity": round(episode_sensitivity, 4),
        "lead_times_minutes": {f"episode_{k}": round(v, 2) for k, v in lead_times_minutes.items()},
        "average_lead_time_minutes": round(float(np.mean(list(lead_times_minutes.values()))), 2) if lead_times_minutes else 0.0,
        "raw_false_positive_windows": raw_false_positives,
        "raw_false_positive_windows_per_hour": round(raw_fp_per_hour, 2),
        "cep_consolidated_false_alarms": consolidated_false_alarms,
        "cep_false_alarms_per_hour": round(false_alarm_rate_per_hour, 3),
        "alarm_reduction_factor": round(raw_false_positives / max(consolidated_false_alarms, 1), 1),
        "suppressed_refractory_windows": orchestrator.alerter.suppressed_refractory_count,
        "total_alerts_fired": len(alerts_fired),
        "tier_counts": tier_counts,
        "consecutive_threshold_windows": consecutive_windows,
        "refractory_minutes": refractory_minutes,
        "event_store_db": str(event_store.db_path),
    }

    # Console Report
    print("-----------------------------------------------------------------------")
    print(f" REPLAY RESULTS & CLINICAL EVENT-LEVEL METRICS:")
    print("-----------------------------------------------------------------------")
    print(f" Monitored Duration:           {summary['total_hours']} hours ({total_windows:,} windows)")
    print(f" Interictal Monitoring Time:   {summary['interictal_hours']} hours")
    print(f" Preictal Episodes Evaluated:  {total_episodes}")
    print(f" Preictal Episodes Detected:   {detected_episodes}/{total_episodes} ({summary['preictal_episode_sensitivity']*100:.1f}%)")
    for ep, lt in lead_times_minutes.items():
        print(f"   -> Episode {ep} Lead Time:   {lt:.1f} minutes before seizure onset")
    print(f" Raw False Positive Windows:   {raw_false_positives} ({raw_fp_per_hour:.1f} FP windows/hour)")
    print(f" CEP Consolidated False Alarms: {consolidated_false_alarms} ({false_alarm_rate_per_hour:.2f} alarms/hour)")
    print(f" Alarm Reduction Factor:       {summary['alarm_reduction_factor']}x fewer alarms via CEP refractory rules")
    print(f" Alert Tier Distribution:      {tier_counts}")
    print(f" Events Logged in SQLite DB:   {summary['event_store_db']}")
    print("=======================================================================\n")

    output_path = output_json or (OUTPUT_DIR / f"replay_summary_{subject}.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved replay summary to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoMAS Offline Agent Replay Pipeline")
    parser.add_argument("--subject", type=str, default="chb01", help="Subject identifier")
    parser.add_argument("--refractory-minutes", type=float, default=30.0, help="Refractory lockout period in minutes")
    parser.add_argument("--consecutive-windows", type=int, default=2, help="Consecutive windows for suspicion")
    parser.add_argument("--screening-threshold", type=float, default=0.20, help="Screening threshold")
    parser.add_argument("--no-clear-db", action="store_true", help="Do not clear existing database before replay")
    args = parser.parse_args()

    run_replay(
        subject=args.subject,
        refractory_minutes=args.refractory_minutes,
        consecutive_windows=args.consecutive_windows,
        screening_threshold=args.screening_threshold,
        clear_db=not args.no_clear_db,
    )
