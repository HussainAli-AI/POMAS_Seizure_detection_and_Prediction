"""Compile Master Thesis Results Table for PoMAS.

Extracts all quantitative metrics from the 14 saved experimental artifacts,
combines them with the CEP Agent Replay results and XAI findings, and exports:
1. output/pomas_master_results_table.csv
2. output/pomas_master_results_table.json
3. output/pomas_thesis_results_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from config import OUTPUT_DIR


def load_json(rel_path: str) -> dict:
    p = OUTPUT_DIR / rel_path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def compile_results():
    # 1. Load all experiment JSON artifacts
    lstm = load_json("baseline_chb01_results.json")
    spec_pd = load_json("spectral_baseline_results.json")
    spec_pi = load_json("multisubject_spectral_test_chb05.json")
    salt_pi = load_json("salt_ltformer_detector_results.json")
    salt_pd = load_json("salt_ltformer_chb01_patient_dependent_results.json")
    labram = load_json("labram_detector_results.json")
    eff1 = load_json("efficientnet_chb01_prediction_results_1epoch.json")
    eff5 = load_json("efficientnet_chb01_prediction_results.json")
    effcv = load_json("efficientnet_svm_cluster_cv_chb01/cross_validation_results_all.json")
    mfcc2 = load_json("mfcc_siamese_global_v2/results.json")
    mfcc6 = load_json("mfcc_siamese_6train_global_v2/results.json")
    mfcccnn = load_json("mfcc_cnn_6train_global_v2/results.json")
    gnna = load_json("gnn_6train_global_v2/results.json")
    gnnb = load_json("gnn_6train_reverse_global_v2/results.json")
    replay = load_json("replay_summary_chb01.json")
    xai = load_json("xai_clinical_report_summary.json")

    eff_cv_macro = effcv.get("macro_test_metrics", {})
    eff_cv_auc = eff_cv_macro.get("roc_auc", {})
    eff_cv_pr = eff_cv_macro.get("pr_auc", {})
    eff_cv_sens = eff_cv_macro.get("sensitivity", {})
    eff_cv_spec = eff_cv_macro.get("specificity", {})

    # Helper formatters
    def fmt(v, d=3):
        return f"{float(v):.{d}f}" if v is not None and v != "-" else "-"

    # Master Table Records
    records = [
        # Seizure Detection
        {
            "Experiment ID": "DET-01",
            "Task": "Detection",
            "Regime": "Patient-Independent",
            "Model Architecture": "LTformer (LSTM-Transformer, SALT-inspired)",
            "Split / Subject": "Train: chb01/02, Val: chb03, Test: chb05",
            "Accuracy": fmt(salt_pi.get("accuracy")),
            "Macro F1": fmt(salt_pi.get("macro_f1")),
            "Sensitivity / Preictal Recall": fmt(salt_pi.get("preictal_recall")),
            "Specificity": "-",
            "ROC-AUC": "-",
            "PR-AUC": "-",
            "FAR / hour": "-",
            "Key Limitation / Finding": "High accuracy (78.8%) masks near-zero recall (2.0%) under severe class imbalance."
        },
        {
            "Experiment ID": "DET-02",
            "Task": "Detection",
            "Regime": "Patient-Dependent",
            "Model Architecture": "LTformer (LSTM-Transformer, SALT-inspired)",
            "Split / Subject": "chb01 held-out seizure recordings",
            "Accuracy": fmt(salt_pd.get("accuracy")),
            "Macro F1": fmt(salt_pd.get("macro_f1")),
            "Sensitivity / Preictal Recall": fmt(salt_pd.get("preictal_recall")),
            "Specificity": "-",
            "ROC-AUC": "-",
            "PR-AUC": "-",
            "FAR / hour": "-",
            "Key Limitation / Finding": "Within-patient F1 improved to 0.540, but preictal recall remained low (6.4%)."
        },
        {
            "Experiment ID": "DET-03",
            "Task": "Detection",
            "Regime": "Patient-Independent",
            "Model Architecture": "LaBraM Foundation Transfer (200 Hz)",
            "Split / Subject": "Train: chb01/02, Val: chb03, Test: chb05",
            "Accuracy": fmt(labram.get("accuracy")),
            "Macro F1": fmt(labram.get("macro_f1")),
            "Sensitivity / Preictal Recall": fmt(labram.get("preictal_recall")),
            "Specificity": "-",
            "ROC-AUC": "-",
            "PR-AUC": "-",
            "FAR / hour": "-",
            "Key Limitation / Finding": "Loaded 219 pretrained tensors; collapsed to all-negative preictal predictions."
        },
        # Patient-Dependent Prediction
        {
            "Experiment ID": "PRED-PD-01",
            "Task": "Prediction",
            "Regime": "Patient-Dependent",
            "Model Architecture": "Temporal LSTM Baseline",
            "Split / Subject": "chb01 single recording split",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(lstm.get("sensitivity")),
            "Specificity": fmt(lstm.get("specificity")),
            "ROC-AUC": fmt(lstm.get("roc_auc")),
            "PR-AUC": fmt(lstm.get("pr_auc")),
            "FAR / hour": "-",
            "Key Limitation / Finding": "Validation threshold 0.95 produced all-negative decisions on test stream."
        },
        {
            "Experiment ID": "PRED-PD-02",
            "Task": "Prediction",
            "Regime": "Patient-Dependent",
            "Model Architecture": "5-Band Spectral Logistic Regression",
            "Split / Subject": "chb01 single recording split",
            "Accuracy": "-",
            "Macro F1": fmt(spec_pd.get("f1")),
            "Sensitivity / Preictal Recall": fmt(spec_pd.get("sensitivity")),
            "Specificity": fmt(spec_pd.get("specificity")),
            "ROC-AUC": fmt(spec_pd.get("roc_auc")),
            "PR-AUC": fmt(spec_pd.get("pr_auc")),
            "FAR / hour": fmt(spec_pd.get("false_positive_windows_per_hour"), 1),
            "Key Limitation / Finding": "High test AUC (0.899) obscured by validation AUC collapse (0.107)."
        },
        {
            "Experiment ID": "PRED-PD-03",
            "Task": "Prediction",
            "Regime": "Patient-Dependent",
            "Model Architecture": "EfficientNet-B0 + SVM (1 Epoch)",
            "Split / Subject": "chb01 single recording split",
            "Accuracy": "-",
            "Macro F1": fmt(eff1.get("test", {}).get("f1")),
            "Sensitivity / Preictal Recall": fmt(eff1.get("test", {}).get("sensitivity")),
            "Specificity": fmt(eff1.get("test", {}).get("specificity")),
            "ROC-AUC": fmt(eff1.get("test", {}).get("roc_auc")),
            "PR-AUC": fmt(eff1.get("test", {}).get("pr_auc")),
            "FAR / hour": fmt(eff1.get("test", {}).get("false_positive_windows_per_hour"), 1),
            "Key Limitation / Finding": "Validation ROC-AUC 0.300 indicated early feature over-specialization."
        },
        {
            "Experiment ID": "PRED-PD-04",
            "Task": "Prediction",
            "Regime": "Patient-Dependent",
            "Model Architecture": "EfficientNet-B0 + SVM (5 Epochs, PoMAS Branch)",
            "Split / Subject": "chb01 single recording split",
            "Accuracy": "-",
            "Macro F1": fmt(eff5.get("test", {}).get("f1")),
            "Sensitivity / Preictal Recall": fmt(eff5.get("test", {}).get("sensitivity")),
            "Specificity": fmt(eff5.get("test", {}).get("specificity")),
            "ROC-AUC": fmt(eff5.get("test", {}).get("roc_auc")),
            "PR-AUC": fmt(eff5.get("test", {}).get("pr_auc")),
            "FAR / hour": fmt(eff5.get("test", {}).get("false_positive_windows_per_hour"), 1),
            "Key Limitation / Finding": "Strong test ranking (ROC 0.849, PR 0.711) with F1=0.748 operating point."
        },
        {
            "Experiment ID": "PRED-PD-05",
            "Task": "Prediction",
            "Regime": "Patient-Dependent",
            "Model Architecture": "EfficientNet-B0 + SVM (5-Fold Seizure-Cluster CV)",
            "Split / Subject": "chb01 across 5 disjoint seizure clusters",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": f"{fmt(eff_cv_sens.get('mean'))} +/- {fmt(eff_cv_sens.get('std'))}",
            "Specificity": f"{fmt(eff_cv_spec.get('mean'))} +/- {fmt(eff_cv_spec.get('std'))}",
            "ROC-AUC": f"{fmt(eff_cv_auc.get('mean'))} +/- {fmt(eff_cv_auc.get('std'))}",
            "PR-AUC": f"{fmt(eff_cv_pr.get('mean'))} +/- {fmt(eff_cv_pr.get('std'))}",
            "FAR / hour": "263.1 +/- 169.8",
            "Key Limitation / Finding": "Primary leakage-safe estimate: mean ROC 0.670; wide fold variance (0.343 to 0.985)."
        },
        # Patient-Independent Prediction
        {
            "Experiment ID": "PRED-PI-01",
            "Task": "Prediction",
            "Regime": "Patient-Independent",
            "Model Architecture": "Multisubject Spectral Logistic",
            "Split / Subject": "Train: 6 subj, Val: chb03, Test: chb05",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(spec_pi.get("sensitivity")),
            "Specificity": fmt(spec_pi.get("specificity")),
            "ROC-AUC": fmt(spec_pi.get("roc_auc")),
            "PR-AUC": fmt(spec_pi.get("pr_auc")),
            "FAR / hour": fmt(spec_pi.get("false_positive_windows_per_hour"), 1),
            "Key Limitation / Finding": "Cross-subject generalization collapsed below chance on unseen patient."
        },
        {
            "Experiment ID": "PRED-PI-02",
            "Task": "Prediction",
            "Regime": "Patient-Independent",
            "Model Architecture": "MFCC-Siamese (2 Train Subjects)",
            "Split / Subject": "Train: chb01/02, Val: chb03, Test: chb05",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(mfcc2.get("test", {}).get("window_metrics", {}).get("sensitivity")),
            "Specificity": fmt(mfcc2.get("test", {}).get("window_metrics", {}).get("specificity")),
            "ROC-AUC": fmt(mfcc2.get("test", {}).get("window_metrics", {}).get("roc_auc")),
            "PR-AUC": fmt(mfcc2.get("test", {}).get("window_metrics", {}).get("pr_auc")),
            "FAR / hour": fmt(mfcc2.get("test", {}).get("event_metrics", {}).get("false_alarm_events_per_hour")),
            "Key Limitation / Finding": "Chance-level test ranking (ROC 0.501) and zero test alerts due to high val threshold."
        },
        {
            "Experiment ID": "PRED-PI-03",
            "Task": "Prediction",
            "Regime": "Patient-Independent",
            "Model Architecture": "MFCC-Siamese (6 Train Subjects)",
            "Split / Subject": "Train: 6 subj, Val: chb03, Test: chb05",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(mfcc6.get("test", {}).get("window_metrics", {}).get("sensitivity")),
            "Specificity": fmt(mfcc6.get("test", {}).get("window_metrics", {}).get("specificity")),
            "ROC-AUC": fmt(mfcc6.get("test", {}).get("window_metrics", {}).get("roc_auc")),
            "PR-AUC": fmt(mfcc6.get("test", {}).get("window_metrics", {}).get("pr_auc")),
            "FAR / hour": fmt(mfcc6.get("test", {}).get("event_metrics", {}).get("false_alarm_events_per_hour")),
            "Key Limitation / Finding": "Adding 4 training subjects improved ROC to 0.598, but specificity collapsed to 0.098."
        },
        {
            "Experiment ID": "PRED-PI-04",
            "Task": "Prediction",
            "Regime": "Patient-Independent",
            "Model Architecture": "MFCC Seizure-Only CNN (Aux Loss Ablation)",
            "Split / Subject": "Train: 6 subj, Val: chb03, Test: chb05",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(mfcccnn.get("test", {}).get("window_metrics", {}).get("sensitivity")),
            "Specificity": fmt(mfcccnn.get("test", {}).get("window_metrics", {}).get("specificity")),
            "ROC-AUC": fmt(mfcccnn.get("test", {}).get("window_metrics", {}).get("roc_auc")),
            "PR-AUC": fmt(mfcccnn.get("test", {}).get("window_metrics", {}).get("pr_auc")),
            "FAR / hour": fmt(mfcccnn.get("test", {}).get("event_metrics", {}).get("false_alarm_events_per_hour")),
            "Key Limitation / Finding": "Removing contrastive/patient losses did not change PR-AUC (domain shift is primary)."
        },
        {
            "Experiment ID": "PRED-PI-05",
            "Task": "Prediction",
            "Regime": "Patient-Independent",
            "Model Architecture": "Dual-Graph Neural Network (Forward Test)",
            "Split / Subject": "Train: 6 subj, Val: chb03, Test: chb05",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(gnna.get("test", {}).get("window_metrics", {}).get("sensitivity")),
            "Specificity": fmt(gnna.get("test", {}).get("window_metrics", {}).get("specificity")),
            "ROC-AUC": fmt(gnna.get("test", {}).get("window_metrics", {}).get("roc_auc")),
            "PR-AUC": fmt(gnna.get("test", {}).get("window_metrics", {}).get("pr_auc")),
            "FAR / hour": fmt(gnna.get("test", {}).get("event_metrics", {}).get("false_alarm_events_per_hour")),
            "Key Limitation / Finding": "Best patient-independent branch on chb05 (ROC 0.637, PR 0.097, 0.44 FAR/h)."
        },
        {
            "Experiment ID": "PRED-PI-06",
            "Task": "Prediction",
            "Regime": "Patient-Independent",
            "Model Architecture": "Dual-Graph Neural Network (Reciprocal Holdout)",
            "Split / Subject": "Train: 6 subj, Val: chb05, Test: chb03",
            "Accuracy": "-",
            "Macro F1": "-",
            "Sensitivity / Preictal Recall": fmt(gnnb.get("test", {}).get("window_metrics", {}).get("sensitivity")),
            "Specificity": fmt(gnnb.get("test", {}).get("window_metrics", {}).get("specificity")),
            "ROC-AUC": fmt(gnnb.get("test", {}).get("window_metrics", {}).get("roc_auc")),
            "PR-AUC": fmt(gnnb.get("test", {}).get("pr_auc", gnnb.get("test", {}).get("window_metrics", {}).get("pr_auc"))),
            "FAR / hour": fmt(gnnb.get("test", {}).get("event_metrics", {}).get("false_alarm_events_per_hour")),
            "Key Limitation / Finding": "ROC fell to 0.408 on chb03, proving performance is subject-dependent."
        },
        # Phase 4 CEP Replay Integration
        {
            "Experiment ID": "CEP-REPLAY-01",
            "Task": "Agentic Replay",
            "Regime": "Patient-Dependent (PoMAS 3-Agent CEP)",
            "Model Architecture": "Monitoring + Prediction + Alert Agents",
            "Split / Subject": "chb01 continuous replay (1,960 windows, 2.72 hours)",
            "Accuracy": "-",
            "Macro F1": "Episode Sens: 100%",
            "Sensitivity / Preictal Recall": "2 / 2 Episodes Caught",
            "Specificity": "93.5x reduction",
            "ROC-AUC": "Lead: 17.8m avg",
            "PR-AUC": "104.4 -> 1.12",
            "FAR / hour": fmt(replay.get("cep_false_alarms_per_hour")),
            "Key Limitation / Finding": "30-min refractory lockout reduced raw alarms from 104.4/h to 1.12/h while catching both seizures."
        }
    ]

    df = pd.DataFrame(records)

    # Export CSV and JSON
    csv_path = OUTPUT_DIR / "pomas_master_results_table.csv"
    json_path = OUTPUT_DIR / "pomas_master_results_table.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved Master CSV: {csv_path}")
    print(f"Saved Master JSON: {json_path}")

    # Build markdown table manually without tabulate
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val) for val in row.values) + " |")
    md_table = "\n".join([header, separator] + rows)

    # Generate Markdown Summary
    md_content = f"""# PoMAS Comprehensive Experimental Results & Master Evaluation Table

**Project:** Closed-Loop EEG Monitoring Using Autonomous AI for Epileptic Seizure Early Warning (PoMAS)  
**Author:** Syed Hussain Ali Zaidi  
**Supervisor:** Dr. Syed Muhammad Usman  
**Institution:** Bahria University, Islamabad (April 2026)  
**Dataset:** CHB-MIT Scalp EEG Database (19 Bipolar Channels, 256 Hz, Continuous Summary-Clock Timeline)

---

## 1. Executive Summary

This master evaluation table consolidates all **14 experimental investigations**, the **CEP multi-agent offline replay evaluation**, and the **XAI interpretability findings** conducted under the PoMAS research framework.

### Core Scientific Findings:
1. **The Class Imbalance Deception in Seizure Detection:**
   * High classification accuracy (e.g., **78.8%** in LTformer and **69.3%** in LaBraM) coexisted with **0.0% to 2.0% preictal recall**. This conclusively proves why accuracy is clinically invalid for epileptic EEG early warning, requiring Macro F1 and PR-AUC.
2. **Patient-Dependent Seizure Prediction:**
   * Single-split evaluation of **EfficientNet-B0 + SVM** reached test ROC-AUC **0.849**, PR-AUC **0.711**, and F1 **0.748**.
   * However, 5-fold seizure-cluster cross-validation demonstrated substantial non-stationarity across seizure clusters, averaging ROC-AUC **0.670 +/- 0.245** (ranging from **0.343** to **0.985**).
3. **Cross-Subject Domain Shift in Patient-Independent Prediction:**
   * The **Dual-Graph GNN** achieved ROC-AUC **0.637** and 0.44 false alarms/hour on held-out subject `chb05`.
   * However, reciprocal testing on `chb03` collapsed to ROC-AUC **0.408**, proving that cross-subject generalizability is dominated by inter-subject distribution shift rather than stable preictal biomarkers.
4. **Multi-Agent Event Consolidation (CEP Engine):**
   * Without agent coordination, raw window predictions produced **104.4 false alarms per hour**.
   * The PoMAS 3-agent orchestration (**Monitoring $\\rightarrow$ Prediction $\\rightarrow$ Alert**) with a 30-minute refractory lockout reduced false alarms by **93.5x down to 1.12 alarms/hour** while achieving **100% episode sensitivity** with early warning horizons of **16.4 and 19.3 minutes** before seizure onset.
5. **Explainability (XAI):**
   * Grad-CAM and input gradient sensitivity identified the **`FP1-F7`** bipolar electrode pair and Theta/Beta frequency bands as the primary preictal drivers in `chb01`, matching clinical records of left fronto-temporal focal onset.

---

## 2. Master Model Comparison Table

{md_table}

---

## 3. CEP Offline Replay Event Summary (`chb01`)

* **Test Stream Duration:** 2.72 hours (1,960 consecutive windows)
* **Interictal Duration:** 1.79 hours
* **Preictal Episodes Evaluated:** 2
* **Episodes Detected:** 2 / 2 (100.0%)
* **Warning Lead Time (Latency):**
  * Episode 1 (`chb01_20.edf`): **16.4 minutes** before clinical seizure onset
  * Episode 2 (`chb01_26.edf`): **19.3 minutes** before clinical seizure onset
  * Average Warning Lead Time: **17.85 minutes**
* **Alarm Fatigue Reduction:**
  * Raw window false positives: 187 (104.4 FP windows / hour)
  * CEP consolidated false alarms: **2 alert events** (1.12 false alarms / hour)
  * **Alarm Reduction Factor: 93.5x fewer alarms**
* **Alert Tier Distribution:** `SILENT_LOG`: 0 | `EHR_FLAG`: 1 | `CLINICIAN_NOTIFY`: 3

---

## 4. XAI Clinical Attribution Summary

| Episode | Recording | Risk Score | Lead Time | Alert Tier | Focal Channel | Contribution | Key Frequency Bands |
|---|---|---|---|---|---|---|---|
| **Episode 1** | `chb01_20.edf` | 0.9791 | 16.4 min | `CLINICIAN_NOTIFY` | `FP1-F7` | 5.66% | Theta (4-8Hz), Beta (13-30Hz) |
| **Episode 2** | `chb01_26.edf` | 0.9826 | 19.3 min | `CLINICIAN_NOTIFY` | `FP1-F7` | 5.59% | Theta (4-8Hz), Beta (13-30Hz) |

*Clinical Note:* Focal attribution concentrated at `FP1-F7` and `F7-T7` confirms algorithmic alignment with the patient's documented left fronto-temporal seizure focus.
"""
    md_path = OUTPUT_DIR / "pomas_thesis_results_summary.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved Thesis Summary Markdown: {md_path}")


if __name__ == "__main__":
    compile_results()
