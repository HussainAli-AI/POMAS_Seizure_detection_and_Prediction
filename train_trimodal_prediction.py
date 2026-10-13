"""Tri-Modal Feature Fusion Prediction Pipeline for PoMAS.

Extends the primary patient-dependent prediction model by fusing:
1. Modality 1: Deep Spatial Spectrogram Features (1,280 dims, EfficientNet-B0)
2. Modality 2: Functional Connectivity Features (171 dims, Pearson correlation matrix)
3. Modality 3: Electrophysiological Spectral Energy (95 dims, Welch PSD across 5 canonical bands)
Total Feature Dimension: 1,280 + 171 + 95 = 1,546 dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import OUTPUT_DIR, SAMPLE_RATE, STRIDE_SECONDS
from data.hdf5_dataset import HDF5WindowDataset
from models.prediction_branch import EfficientNetAdapter
from train_efficientnet_svm import (
    correlation_features,
    metrics,
    recording_split,
    select_threshold,
    stft_spectrogram,
    validation_loss,
)

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0))


def bandpower_features(windows: np.ndarray) -> np.ndarray:
    """Extract 5 log-power features per EEG channel (19 channels * 5 bands = 95 features)."""
    frequencies, power = welch(windows, fs=SAMPLE_RATE, nperseg=256, axis=-1)
    features = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        features.append(np.log10(power[..., mask].mean(axis=-1) + 1e-12))
    return np.stack(features, axis=-1).reshape(len(windows), -1).astype(np.float32)


def extract_trimodal_features(model: nn.Module, dataset: HDF5WindowDataset, batch_size: int, device: torch.device):
    """Extract and concatenate Deep (1,280) + Correlation (171) + Spectral (95) features."""
    deep, correlations, spectral, labels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for windows, targets, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            # 1. Deep Spatial Features (1,280)
            deep.append(model(stft_spectrogram(windows, device)).cpu().numpy())
            # 2. Functional Connectivity (171)
            correlations.append(correlation_features(windows))
            # 3. Electrophysiological Bandpowers (95)
            spectral.append(bandpower_features(windows.numpy()))
            labels.append(targets.numpy())

    x_deep = np.concatenate(deep, axis=0)
    x_corr = np.concatenate(correlations, axis=0)
    x_spec = np.concatenate(spectral, axis=0)
    x_combined = np.concatenate([x_deep, x_corr, x_spec], axis=1)
    y_combined = np.concatenate(labels, axis=0)

    print(f"  Extracted Tri-Modal shapes: Deep={x_deep.shape}, Corr={x_corr.shape}, Spec={x_spec.shape} -> Combined={x_combined.shape}")
    return x_combined, y_combined


SUBJECT_SPLITS: dict[str, dict[str, list[str]]] = {
    "chb01": {
        "val": ["chb01_17.edf", "chb01_18.edf", "chb01_19.edf"],
        "test": ["chb01_20.edf", "chb01_21.edf", "chb01_25.edf", "chb01_26.edf", "chb01_27.edf"],
    },
    "chb02": {
        "val": ["chb02_14.edf", "chb02_16.edf"],
        "test": ["chb02_16+.edf", "chb02_19.edf", "chb02_20.edf"],
    },
    "chb03": {
        "val": ["chb03_03.edf", "chb03_04.edf"],
        "test": ["chb03_34.edf", "chb03_35.edf", "chb03_36.edf"],
    },
}


def run_trimodal(
    subject: str = "chb01",
    data: Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    interictal_ratio: float = 3.0,
    retrain_backbone: bool = False,
):
    if data is None:
        data = OUTPUT_DIR / "preprocessed_global_v2" / f"{subject}_preprocessed_global_v2.h5"

    if subject not in SUBJECT_SPLITS:
        raise ValueError(f"Unknown subject '{subject}'. Supported: {list(SUBJECT_SPLITS.keys())}")

    validation_recordings = SUBJECT_SPLITS[subject]["val"]
    test_recordings = SUBJECT_SPLITS[subject]["test"]

    print("=" * 70)
    print(f"STARTING TRI-MODAL FEATURE FUSION SEIZURE PREDICTION (Subject: {subject.upper()})")
    print("=" * 70)

    train_i, val_i, test_i, all_labels = recording_split(data, validation_recordings, test_recordings, interictal_ratio)
    train = HDF5WindowDataset(data, train_i)
    validation = HDF5WindowDataset(data, val_i)
    test = HDF5WindowDataset(data, test_i)

    train_labels = all_labels[train_i]
    class_counts = np.bincount(train_labels, minlength=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Train: {len(train):,} | Validation: {len(validation):,} | Test: {len(test):,}")

    backbone = EfficientNetAdapter(n_channels=19, pretrained=True).to(device)
    classifier = nn.Linear(backbone.feature_dim, 1).to(device)

    ckpt_path = OUTPUT_DIR / f"efficientnet_{subject}_prediction.pt"
    if ckpt_path.exists() and not retrain_backbone:
        print(f"Loading existing fine-tuned EfficientNet-B0 weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(state["backbone"])
        classifier.load_state_dict(state["classifier"])
    else:
        print(f"Training EfficientNet-B0 backbone for {subject} with cosine schedule...")
        sampler_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_i), replacement=True)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        for parameter in backbone.backbone.features[0].parameters():
            parameter.requires_grad = True
        for stage in backbone.backbone.features[-2:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for p in list(backbone.parameters()) + list(classifier.parameters()) if p.requires_grad],
            lr=1e-4,
            weight_decay=1e-4,
        )
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")

        best_loss, best_state = float("inf"), None
        for epoch in range(1, epochs + 1):
            backbone.train()
            classifier.train()
            total = 0.0
            for module in backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            for windows, targets, _ in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = classifier(backbone(stft_spectrogram(windows, device))).squeeze(1)
                    loss = criterion(logits, targets.float().to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()

            epoch_loss = total / len(loader)
            val_loss = validation_loss(backbone, classifier, validation, batch_size * 2, device, criterion)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    "backbone": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
                    "classifier": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
                }
        backbone.load_state_dict(best_state["backbone"])
        torch.save(best_state, ckpt_path)
        print(f"Saved fine-tuned backbone to: {ckpt_path}")

    # ---------------------------------------------------------
    # Tri-Modal Feature Extraction
    # ---------------------------------------------------------
    print(f"\nExtracting Tri-Modal features for {subject} Train set...")
    x_train, y_train = extract_trimodal_features(backbone, train, batch_size * 2, device)

    print(f"\nExtracting Tri-Modal features for {subject} Validation set...")
    x_val, y_val = extract_trimodal_features(backbone, validation, batch_size * 2, device)

    print(f"\nExtracting Tri-Modal features for {subject} Test set...")
    x_test, y_test = extract_trimodal_features(backbone, test, batch_size * 2, device)

    # ---------------------------------------------------------
    # Train Non-Linear RBF Support Vector Machine on Tri-Modal Representation
    # ---------------------------------------------------------
    print(f"\nFitting Support Vector Machine (RBF Kernel) on 1,546 Tri-Modal features for {subject}...")
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, probability=False, class_weight="balanced", random_state=42),
    )
    svm.fit(x_train, y_train)

    # Threshold Optimization on Validation Set
    val_scores = svm.decision_function(x_val)
    val_threshold = select_threshold(y_val, val_scores, criterion="f1")
    val_metrics = metrics(y_val, val_scores, val_threshold)

    # Test Evaluation at Operating (Val) Threshold
    test_scores = svm.decision_function(x_test)
    test_metrics = metrics(y_test, test_scores, val_threshold)

    # Test Evaluation at Optimal Test F1 Threshold
    test_threshold_f1 = select_threshold(y_test, test_scores, criterion="f1")
    test_f1_optimal = metrics(y_test, test_scores, test_threshold_f1)
    test_f1_optimal["threshold"] = float(test_threshold_f1)

    results = {
        "subject": subject,
        "model_architecture": "Tri-Modal Feature Fusion (EfficientNet-B0 + Correlation + Welch Spectral Bandpowers)",
        "feature_dimensions": {
            "deep_spatial_dim": 1280,
            "correlation_connectivity_dim": 171,
            "welch_spectral_bandpower_dim": 95,
            "total_dim": 1546,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_f1_optimal": test_f1_optimal,
        "validation_recordings": validation_recordings,
        "test_recordings": test_recordings,
        "train_interictal_ratio": interictal_ratio,
        "operating_threshold": float(val_threshold),
    }

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = OUTPUT_DIR / f"trimodal_{subject}_prediction_results.json"
    svm_path = OUTPUT_DIR / f"trimodal_{subject}_prediction_svm.joblib"
    npz_path = OUTPUT_DIR / f"trimodal_{subject}_test_predictions.npz"

    joblib.dump(svm, svm_path)
    np.savez_compressed(npz_path, labels=y_test, scores=test_scores)
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"TRI-MODAL SEIZURE PREDICTION EVALUATION RESULTS ({subject.upper()})")
    print("=" * 70)
    print("Total Feature Dimensions: 1,546 (Deep: 1,280 | Correlation: 171 | Spectral: 95)")
    print(f"Optimal Validation Threshold: {val_threshold:.4f}")
    print(f"Test Sensitivity (Recall)  : {test_metrics['sensitivity']*100:.2f}% ({test_metrics['true_positives']} / {test_metrics['n_preictal']})")
    print(f"Test Specificity           : {test_metrics['specificity']*100:.2f}% ({test_metrics['true_negatives']} / {test_metrics['n_windows'] - test_metrics['n_preictal']})")
    print(f"Test Precision             : {test_metrics['precision']*100:.2f}%")
    print(f"Test F1-Score              : {test_metrics['f1']*100:.2f}%")
    print(f"Test ROC-AUC               : {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC                : {test_metrics['pr_auc']:.4f}")
    print(f"False Positives / Hour     : {test_metrics['false_positive_windows_per_hour']:.1f}")
    print(f"Test Optimal F1 Threshold  : {test_threshold_f1:.4f} -> F1={test_f1_optimal['f1']*100:.2f}%, Sens={test_f1_optimal['sensitivity']*100:.2f}%, Spec={test_f1_optimal['specificity']*100:.2f}%")
    print("=" * 70)

    for d in (train, validation, test):
        d.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-Modal Feature Fusion Seizure Prediction Pipeline")
    parser.add_argument("--subject", type=str, default="chb01", help="Subject identifier (e.g. chb01, chb02, chb03)")
    parser.add_argument("--data", type=Path, default=None, help="Explicit HDF5 data path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--interictal-ratio", type=float, default=3.0)
    parser.add_argument("--retrain", action="store_true", default=False)
    args = parser.parse_args()
    run_trimodal(args.subject, args.data, args.epochs, args.batch_size, args.interictal_ratio, args.retrain)
