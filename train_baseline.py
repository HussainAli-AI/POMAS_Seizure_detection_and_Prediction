"""Patient-dependent, recording-level baseline for preictal EEG classification.

This baseline separates complete EDF recordings between train and test sets.
It classifies preictal (label 1) versus interictal (label 0); ictal windows are
excluded because the task is early warning rather than seizure detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import OUTPUT_DIR, STRIDE_SECONDS
from data.hdf5_dataset import HDF5WindowDataset


class TemporalLSTM(nn.Module):
    """Compact temporal baseline suitable for a 6 GB GPU."""

    def __init__(self, n_channels: int = 19, hidden_size: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(64)  # 10 s -> 64 temporal tokens
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(0.2), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).transpose(1, 2)  # (batch, time, channels)
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1]).squeeze(1)


def binary_recording_split(h5_path: Path, validation_recordings: list[str], test_recordings: list[str]):
    with h5py.File(h5_path, "r") as file:
        labels = file["labels"][:]
        recording_ids = np.asarray(file["recording_ids"][:], dtype=str)

    if set(validation_recordings) & set(test_recordings):
        raise ValueError("validation and test recordings must be disjoint")
    eligible = labels != 2
    test_mask = np.isin(recording_ids, test_recordings) & eligible
    validation_mask = np.isin(recording_ids, validation_recordings) & eligible
    train_mask = ~np.isin(recording_ids, test_recordings + validation_recordings) & eligible
    train_indices, validation_indices, test_indices = np.flatnonzero(train_mask), np.flatnonzero(validation_mask), np.flatnonzero(test_mask)
    train_labels, validation_labels, test_labels = labels[train_indices], labels[validation_indices], labels[test_indices]
    for split_name, split_labels in (("train", train_labels), ("validation", validation_labels), ("test", test_labels)):
        if set(np.unique(split_labels)) != {0, 1}:
            raise ValueError(f"{split_name} split must contain both interictal and preictal windows; got {np.unique(split_labels)}")
    return (train_indices, validation_indices, test_indices,
            train_labels.astype(np.int64), validation_labels.astype(np.int64), test_labels.astype(np.int64))


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for windows, targets, _ in loader:
            logits = model(windows.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.numpy())
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    return y_true, y_score


def score_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    interictal_hours = max((tn + fp) * STRIDE_SECONDS / 3600, 1e-12)
    return {
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "false_positive_windows_per_hour": float(fp / interictal_hours),
        "threshold": float(threshold),
        "n_test_windows": int(len(y_true)),
        "n_test_preictal": int(y_true.sum()),
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, max_fpr_per_hour: float = 12.0) -> float:
    """Choose on validation only: highest sensitivity under a window-level FPR cap."""
    candidates = np.linspace(0.05, 0.95, 91)
    feasible = [(score_metrics(y_true, y_score, t), float(t)) for t in candidates]
    feasible = [(m, t) for m, t in feasible if m["false_positive_windows_per_hour"] <= max_fpr_per_hour]
    if not feasible:
        return min(candidates, key=lambda t: score_metrics(y_true, y_score, t)["false_positive_windows_per_hour"])
    return max(feasible, key=lambda item: (item[0]["sensitivity"], item[0]["specificity"]))[1]


def train(h5_path: Path, validation_recordings: list[str], test_recordings: list[str], epochs: int, batch_size: int, learning_rate: float):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_indices, validation_indices, test_indices, train_labels, validation_labels, test_labels = binary_recording_split(h5_path, validation_recordings, test_recordings)
    train_set = HDF5WindowDataset(h5_path, train_indices)
    validation_set = HDF5WindowDataset(h5_path, validation_indices)
    test_set = HDF5WindowDataset(h5_path, test_indices)
    counts = np.bincount(train_labels, minlength=2)
    weights = (1.0 / counts[train_labels]).astype(np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = TemporalLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"device={device}; train={len(train_set)} {dict(zip(*np.unique(train_labels, return_counts=True)))}; validation={len(validation_set)}; test={len(test_set)}")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for windows, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(windows.to(device, non_blocking=True))
                loss = criterion(logits, targets.float().to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        val_labels, val_scores = predict(model, validation_loader, device)
        val_metrics = score_metrics(val_labels, val_scores, 0.5)
        print(f"epoch={epoch}/{epochs} loss={loss_sum / len(train_loader):.4f} val_roc_auc={val_metrics['roc_auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}")

    val_labels, val_scores = predict(model, validation_loader, device)
    threshold = choose_threshold(val_labels, val_scores)
    test_labels_out, test_scores = predict(model, test_loader, device)
    metrics = score_metrics(test_labels_out, test_scores, threshold)
    metrics["validation_threshold"] = threshold
    metrics["validation_recordings"] = validation_recordings
    metrics["test_recordings"] = test_recordings

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / "baseline_chb01_temporal_lstm.pt"
    results_path = OUTPUT_DIR / "baseline_chb01_results.json"
    torch.save({"model_state": model.state_dict(), "validation_recordings": validation_recordings, "test_recordings": test_recordings, "threshold": threshold}, checkpoint)
    results_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint}\nmetrics={results_path}")
    train_set.close()
    validation_set.close()
    test_set.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT_DIR / "preprocessed" / "chb01_preprocessed.h5")
    parser.add_argument("--test-recordings", nargs="+", default=["chb01_21.edf", "chb01_26.edf"])
    parser.add_argument("--validation-recordings", nargs="+", default=["chb01_16.edf", "chb01_18.edf"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, args.validation_recordings, args.test_recordings, args.epochs, args.batch_size, args.lr)
