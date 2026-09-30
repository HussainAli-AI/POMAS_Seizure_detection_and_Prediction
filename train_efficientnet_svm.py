"""Patient-dependent STFT + correlation + EfficientNet-B0 + SVM experiment.

The prediction task is binary: interictal (0) versus preictal (1). Ictal
windows are excluded. Complete EDF recordings are kept in disjoint splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import OUTPUT_DIR, STRIDE_SECONDS
from data.hdf5_dataset import HDF5WindowDataset
from models.prediction_branch import EfficientNetAdapter


def recording_split(path: Path, validation_recordings: list[str], test_recordings: list[str], interictal_ratio: float):
    with h5py.File(path, "r") as file:
        labels = file["labels"][:].astype(np.int64)
        recording_ids = np.asarray(file["recording_ids"][:], dtype=str)
    if set(validation_recordings) & set(test_recordings):
        raise ValueError("Validation and test recordings must be disjoint")
    eligible = labels != 2
    val = np.flatnonzero(eligible & np.isin(recording_ids, validation_recordings))
    test = np.flatnonzero(eligible & np.isin(recording_ids, test_recordings))
    train_all = np.flatnonzero(eligible & ~np.isin(recording_ids, validation_recordings + test_recordings))
    positive = train_all[labels[train_all] == 1]
    negative = train_all[labels[train_all] == 0]
    rng = np.random.default_rng(42)
    limit = min(len(negative), int(interictal_ratio * len(positive)))
    train = np.sort(np.concatenate([positive, rng.choice(negative, limit, replace=False)]))
    for name, indices in (("train", train), ("validation", val), ("test", test)):
        if set(np.unique(labels[indices])) != {0, 1}:
            raise ValueError(f"{name} split must contain interictal and preictal windows")
    return train, val, test, labels


def stft_spectrogram(windows: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Log-magnitude STFT: (B,19,2560) -> (B,19,129,37)."""
    windows = windows.to(device, non_blocking=True)
    batch, channels, samples = windows.shape
    flat = windows.reshape(batch * channels, samples)
    transform = torch.stft(flat, n_fft=256, hop_length=64, win_length=256,
                           window=torch.hann_window(256, device=device),
                           center=False, return_complex=True)
    spectrogram = torch.log1p(transform.abs()).reshape(batch, channels, 129, -1)
    mean = spectrogram.mean(dim=(-2, -1), keepdim=True)
    std = spectrogram.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (spectrogram - mean) / std


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    interictal_hours = max((tn + fp) * STRIDE_SECONDS / 3600, 1e-12)
    return {
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "false_positive_windows_per_hour": float(fp / interictal_hours),
        "threshold": float(threshold),
        "n_windows": int(len(labels)),
        "n_preictal": int(labels.sum()),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "interictal_hours": float(interictal_hours),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite validation threshold is available")
    finite_indices = np.flatnonzero(finite)
    best = finite_indices[np.argmax((true_positive_rate - false_positive_rate)[finite])]
    return float(thresholds[best])


def correlation_features(windows: torch.Tensor) -> np.ndarray:
    arrays = windows.numpy()
    upper = np.triu_indices(arrays.shape[1], k=1)
    return np.stack([np.corrcoef(window)[upper] for window in arrays]).astype(np.float32)


def extract_features(model, dataset, batch_size, device):
    deep, correlations, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for windows, targets, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            deep.append(model(stft_spectrogram(windows, device)).cpu().numpy())
            correlations.append(correlation_features(windows))
            labels.append(targets.numpy())
    return np.concatenate([np.concatenate(deep), np.concatenate(correlations)], axis=1), np.concatenate(labels)


def validation_loss(backbone, classifier, dataset, batch_size, device, criterion):
    backbone.eval(); classifier.eval(); total = 0.0; batches = 0
    with torch.no_grad():
        for windows, targets, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            logits = classifier(backbone(stft_spectrogram(windows, device))).squeeze(1)
            total += criterion(logits, targets.float().to(device)).item(); batches += 1
    return total / max(batches, 1)


def run(data: Path, epochs: int, batch_size: int, interictal_ratio: float):
    validation_recordings = ["chb01_17.edf", "chb01_18.edf", "chb01_19.edf"]
    test_recordings = ["chb01_20.edf", "chb01_21.edf", "chb01_25.edf", "chb01_26.edf", "chb01_27.edf"]
    train_i, val_i, test_i, all_labels = recording_split(data, validation_recordings, test_recordings, interictal_ratio)
    train = HDF5WindowDataset(data, train_i); validation = HDF5WindowDataset(data, val_i); test = HDF5WindowDataset(data, test_i)
    train_labels = all_labels[train_i]
    class_counts = np.bincount(train_labels, minlength=2)
    sampler_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_i), replacement=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = EfficientNetAdapter(n_channels=19, pretrained=True).to(device)
    classifier = nn.Linear(backbone.feature_dim, 1).to(device)
    # Fine-tune the input stem and last two EfficientNet stages; preserve the
    # remaining pretrained representation.
    for parameter in backbone.parameters(): parameter.requires_grad = False
    for parameter in backbone.backbone.features[0].parameters(): parameter.requires_grad = True
    for stage in backbone.backbone.features[-2:]:
        for parameter in stage.parameters(): parameter.requires_grad = True
    optimizer = torch.optim.AdamW([p for p in list(backbone.parameters()) + list(classifier.parameters()) if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    best_loss, best_state = float("inf"), None
    print(f"device={device}; train={len(train)} validation={len(validation)} test={len(test)} train_counts={class_counts.tolist()}")
    for epoch in range(1, epochs + 1):
        backbone.train(); classifier.train(); total = 0.0
        # Small EEG batches make BatchNorm running statistics unstable.
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        for windows, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = classifier(backbone(stft_spectrogram(windows, device))).squeeze(1)
                loss = criterion(logits, targets.float().to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += loss.item()
        epoch_loss = total / len(loader)
        val_loss = validation_loss(backbone, classifier, validation, batch_size * 2, device, criterion)
        print(f"epoch={epoch}/{epochs} train_loss={epoch_loss:.4f} validation_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {"backbone": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
                          "classifier": {k: v.detach().cpu() for k, v in classifier.state_dict().items()}}
    backbone.load_state_dict(best_state["backbone"])
    print("Extracting EfficientNet and correlation features...")
    x_train, y_train = extract_features(backbone, train, batch_size * 2, device)
    x_val, y_val = extract_features(backbone, validation, batch_size * 2, device)
    x_test, y_test = extract_features(backbone, test, batch_size * 2, device)
    # SVC probability=True is deprecated in scikit-learn 1.9. Decision margins
    # preserve ROC/PR ranking and avoid a second, randomly split calibration CV.
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=42))
    svm.fit(x_train, y_train)
    validation_scores = svm.decision_function(x_val)
    threshold = select_threshold(y_val, validation_scores)
    test_scores = svm.decision_function(x_test)
    results = {
        "validation": metrics(y_val, validation_scores, threshold),
        "test": metrics(y_test, test_scores, threshold),
        "validation_recordings": validation_recordings,
        "test_recordings": test_recordings,
        "train_interictal_ratio": interictal_ratio,
        "score_type": "svm_decision_function",
        "deep_feature_dim": backbone.feature_dim,
        "correlation_feature_dim": 171,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUTPUT_DIR / "efficientnet_chb01_prediction.pt")
    joblib.dump(svm, OUTPUT_DIR / "efficientnet_chb01_prediction_svm.joblib")
    np.savez_compressed(OUTPUT_DIR / "efficientnet_chb01_test_predictions.npz", labels=y_test, scores=test_scores)
    (OUTPUT_DIR / "efficientnet_chb01_prediction_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    for dataset in (train, validation, test): dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT_DIR / "preprocessed_global_v2" / "chb01_preprocessed_global_v2.h5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--interictal-ratio", type=float, default=3.0)
    args = parser.parse_args()
    run(args.data, args.epochs, args.batch_size, args.interictal_ratio)
