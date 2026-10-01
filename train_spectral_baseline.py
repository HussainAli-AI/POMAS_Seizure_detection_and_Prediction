"""Recording-level spectral baseline for preictal versus interictal EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, SAMPLE_RATE
from data.hdf5_dataset import HDF5WindowDataset
from train_baseline import binary_recording_split, score_metrics

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0))


def bandpower_features(windows: np.ndarray) -> np.ndarray:
    """Return 5 log-power features per EEG channel for a batch of windows."""
    frequencies, power = welch(windows, fs=SAMPLE_RATE, nperseg=256, axis=-1)
    features = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        features.append(np.log10(power[..., mask].mean(axis=-1) + 1e-12))
    return np.stack(features, axis=-1).reshape(len(windows), -1).astype(np.float32)


def extract(dataset: HDF5WindowDataset, batch_size: int = 32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features, labels = [], []
    for windows, targets, _ in loader:
        features.append(bandpower_features(windows.numpy()))
        labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def select_balanced_threshold(labels: np.ndarray, scores: np.ndarray, criterion: str = "f1") -> float:
    candidates = np.linspace(0.01, 0.99, 100)
    if criterion == "f1":
        return float(max(candidates, key=lambda t: score_metrics(labels, scores, t)["f1"]))
    return float(max(
        candidates,
        key=lambda t: score_metrics(labels, scores, t)["sensitivity"] + score_metrics(labels, scores, t)["specificity"],
    ))


def train(data: Path, validation_recordings: list[str], test_recordings: list[str]):
    split = binary_recording_split(data, validation_recordings, test_recordings)
    train_i, val_i, test_i = split[:3]
    train_set, val_set, test_set = (HDF5WindowDataset(data, indices) for indices in (train_i, val_i, test_i))
    print("Extracting train spectral features...")
    x_train, y_train = extract(train_set)
    print("Extracting validation spectral features...")
    x_val, y_val = extract(val_set)
    print("Extracting test spectral features...")
    x_test, y_test = extract(test_set)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    model.fit(x_train, y_train)
    validation_scores = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    threshold = select_balanced_threshold(y_val, validation_scores)
    metrics = score_metrics(y_test, test_scores, threshold)
    validation_metrics = score_metrics(y_val, validation_scores, threshold)
    metrics.update({
        "validation_threshold": threshold,
        "validation_metrics_at_threshold": validation_metrics,
        "bands_hz": BANDS,
        "validation_recordings": validation_recordings,
        "test_recordings": test_recordings,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "spectral_baseline_results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, OUTPUT_DIR / "spectral_baseline_model.joblib")
    print(json.dumps(metrics, indent=2))
    for dataset in (train_set, val_set, test_set): dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT_DIR / "preprocessed" / "chb01_preprocessed.h5")
    parser.add_argument("--validation-recordings", nargs="+", default=["chb01_16.edf", "chb01_18.edf"])
    parser.add_argument("--test-recordings", nargs="+", default=["chb01_21.edf", "chb01_26.edf"])
    args = parser.parse_args()
    train(args.data, args.validation_recordings, args.test_recordings)
