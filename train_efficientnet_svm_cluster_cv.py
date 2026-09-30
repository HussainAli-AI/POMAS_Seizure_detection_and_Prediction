"""Leakage-safe seizure-cluster cross-validation for the prediction branch.

Each fold holds out one independent seizure cluster for testing and the next
cluster for validation. A cluster contains seizures whose 30-minute preictal
and postictal context intervals overlap. Complete EDF recordings intersecting
either context are assigned together, so overlapping windows and neighboring
recordings cannot cross split boundaries.

Outputs use a separate directory and completed folds are never replaced unless
--overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import DATASET_DIR, OUTPUT_DIR, POSTICTAL_SECONDS, PREICTAL_SECONDS
from data.hdf5_dataset import HDF5WindowDataset
from data.summary_parser import parse_summary
from models.prediction_branch import EfficientNetAdapter
from prepare_global_labels import clock_seconds, global_file_offsets
from train_efficientnet_svm import (
    extract_features,
    metrics,
    select_threshold,
    stft_spectrogram,
    validation_loss,
)


def decode_recording_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def seizure_cluster_manifest(data: Path) -> dict:
    with h5py.File(data, "r") as file:
        subject = str(file.attrs["subject_id"])
        if subject.startswith("b'"):
            subject = file.attrs["subject_id"].decode("utf-8")
        label_timeline = str(file.attrs.get("label_timeline", ""))
        if label_timeline != "subject_global_summary_clock_v2":
            raise ValueError(
                "Cluster CV requires the corrected global-v2 archive; "
                f"found label_timeline={label_timeline!r}"
            )
        if file["windows"].shape[1] != 19:
            raise ValueError("EfficientNet prediction branch requires 19 EEG channels")
        available = set(decode_recording_ids(file["recording_ids"][:]).tolist())

    summary_path = DATASET_DIR / subject / f"{subject}-summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing subject summary: {summary_path}")
    summary = parse_summary(summary_path)
    offsets = global_file_offsets(summary["files"])
    base_time = min(offsets.values())

    events = []
    file_intervals = {}
    for file_info in summary["files"]:
        start = offsets[file_info["name"]]
        duration = clock_seconds(file_info["end_time"]) - clock_seconds(file_info["start_time"])
        while duration <= 0:
            duration += 24 * 3600
        file_intervals[file_info["name"]] = (start - base_time, start - base_time + duration)
        for onset, end in file_info.get("seizures", []):
            if onset is None or end is None:
                continue
            events.append({
                "recording": file_info["name"],
                "onset_seconds": float(start + onset - base_time),
                "end_seconds": float(start + end - base_time),
                "context_start_seconds": float(start + onset - base_time - PREICTAL_SECONDS),
                "context_end_seconds": float(start + end - base_time + POSTICTAL_SECONDS),
            })
    events.sort(key=lambda event: event["onset_seconds"])
    if not events:
        raise ValueError(f"No seizure annotations found for {subject}")

    clusters = []
    for event in events:
        if not clusters or event["context_start_seconds"] > clusters[-1]["context_end_seconds"]:
            clusters.append({
                "context_start_seconds": event["context_start_seconds"],
                "context_end_seconds": event["context_end_seconds"],
                "seizures": [event],
            })
        else:
            clusters[-1]["context_end_seconds"] = max(
                clusters[-1]["context_end_seconds"], event["context_end_seconds"]
            )
            clusters[-1]["seizures"].append(event)

    owner = {}
    for number, cluster in enumerate(clusters, start=1):
        all_recordings = [
            name for name, (start, end) in file_intervals.items()
            if start < cluster["context_end_seconds"] and end > cluster["context_start_seconds"]
        ]
        recordings = [name for name in all_recordings if name in available]
        if not recordings:
            raise ValueError(f"Cluster {number} has no recordings in {data}")
        for recording in recordings:
            if recording in owner:
                raise ValueError(
                    f"{recording} intersects clusters {owner[recording]} and {number}; "
                    "the cluster definition must be widened before cross-validation"
                )
            owner[recording] = number
        cluster["cluster_id"] = number
        cluster["recordings"] = recordings
        cluster["summary_recordings"] = all_recordings

    return {
        "subject": subject,
        "data": str(data.resolve()),
        "label_timeline": label_timeline,
        "preictal_seconds": PREICTAL_SECONDS,
        "postictal_seconds": POSTICTAL_SECONDS,
        "n_seizures": len(events),
        "n_clusters": len(clusters),
        "clusters": clusters,
    }


def split_indices(
    data: Path,
    validation_recordings: list[str],
    test_recordings: list[str],
    interictal_ratio: float,
    seed: int,
):
    with h5py.File(data, "r") as file:
        labels = file["labels"][:].astype(np.int64)
        recording_ids = decode_recording_ids(file["recording_ids"][:])
    if set(validation_recordings) & set(test_recordings):
        raise ValueError("Validation and test recordings overlap")

    eligible = labels != 2
    validation = np.flatnonzero(eligible & np.isin(recording_ids, validation_recordings))
    test = np.flatnonzero(eligible & np.isin(recording_ids, test_recordings))
    train_all = np.flatnonzero(
        eligible & ~np.isin(recording_ids, validation_recordings + test_recordings)
    )
    positive = train_all[labels[train_all] == 1]
    negative = train_all[labels[train_all] == 0]
    if len(positive) == 0:
        raise ValueError("Training split has no preictal windows")
    limit = min(len(negative), int(interictal_ratio * len(positive)))
    rng = np.random.default_rng(seed)
    train = np.sort(np.concatenate([positive, rng.choice(negative, limit, replace=False)]))

    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        if set(np.unique(labels[indices]).tolist()) != {0, 1}:
            raise ValueError(f"{name} split must contain both interictal and preictal windows")
    return train, validation, test, labels, recording_ids


def fold_plan(data: Path, manifest: dict, interictal_ratio: float, seed: int) -> list[dict]:
    clusters = manifest["clusters"]
    if len(clusters) < 3:
        raise ValueError("At least three independent seizure clusters are required")
    plans = []
    for index, test_cluster in enumerate(clusters):
        validation_cluster = clusters[(index + 1) % len(clusters)]
        train_i, val_i, test_i, labels, _ = split_indices(
            data,
            validation_cluster["recordings"],
            test_cluster["recordings"],
            interictal_ratio,
            seed + index,
        )
        plans.append({
            "fold": index + 1,
            "test_cluster": test_cluster["cluster_id"],
            "validation_cluster": validation_cluster["cluster_id"],
            "validation_recordings": validation_cluster["recordings"],
            "test_recordings": test_cluster["recordings"],
            "n_train": int(len(train_i)),
            "n_validation": int(len(val_i)),
            "n_test": int(len(test_i)),
            "train_counts": np.bincount(labels[train_i], minlength=2).tolist(),
            "validation_counts": np.bincount(labels[val_i], minlength=2).tolist(),
            "test_counts": np.bincount(labels[test_i], minlength=2).tolist(),
        })
    return plans


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_model(device: torch.device):
    backbone = EfficientNetAdapter(n_channels=19, pretrained=True).to(device)
    classifier = nn.Linear(backbone.feature_dim, 1).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in backbone.backbone.features[0].parameters():
        parameter.requires_grad = True
    for stage in backbone.backbone.features[-2:]:
        for parameter in stage.parameters():
            parameter.requires_grad = True
    return backbone, classifier


def run_fold(
    data: Path,
    plan: dict,
    fold_dir: Path,
    epochs: int,
    batch_size: int,
    interictal_ratio: float,
    seed: int,
) -> dict:
    fold_seed = seed + plan["fold"] - 1
    seed_everything(fold_seed)
    train_i, val_i, test_i, all_labels, recording_ids = split_indices(
        data,
        plan["validation_recordings"],
        plan["test_recordings"],
        interictal_ratio,
        fold_seed,
    )
    train = HDF5WindowDataset(data, train_i)
    validation = HDF5WindowDataset(data, val_i)
    test = HDF5WindowDataset(data, test_i)
    try:
        train_labels = all_labels[train_i]
        class_counts = np.bincount(train_labels, minlength=2)
        sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_i),
            replacement=True,
            generator=torch.Generator().manual_seed(fold_seed),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone, classifier = new_model(device)
        trainable = [
            parameter for parameter in list(backbone.parameters()) + list(classifier.parameters())
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        loader = DataLoader(
            train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        best_loss = float("inf")
        best_state = None
        print(
            f"fold={plan['fold']} device={device} train={len(train)} "
            f"validation={len(validation)} test={len(test)} counts={class_counts.tolist()}"
        )
        for epoch in range(1, epochs + 1):
            backbone.train()
            classifier.train()
            for module in backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            total = 0.0
            for windows, targets, _ in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = classifier(backbone(stft_spectrogram(windows, device))).squeeze(1)
                    loss = criterion(logits, targets.float().to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
            train_loss = total / max(len(loader), 1)
            val_loss = validation_loss(
                backbone, classifier, validation, batch_size * 2, device, criterion
            )
            print(
                f"fold={plan['fold']} epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.4f} validation_loss={val_loss:.4f}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    "backbone": {
                        key: value.detach().cpu().clone()
                        for key, value in backbone.state_dict().items()
                    },
                    "classifier": {
                        key: value.detach().cpu().clone()
                        for key, value in classifier.state_dict().items()
                    },
                }
        if best_state is None:
            raise RuntimeError("No checkpoint was selected")
        backbone.load_state_dict(best_state["backbone"])

        print(f"fold={plan['fold']} extracting EfficientNet and correlation features...")
        x_train, y_train = extract_features(backbone, train, batch_size * 2, device)
        x_validation, y_validation = extract_features(
            backbone, validation, batch_size * 2, device
        )
        x_test, y_test = extract_features(backbone, test, batch_size * 2, device)
        svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                probability=False,
                class_weight="balanced",
                random_state=fold_seed,
            ),
        )
        svm.fit(x_train, y_train)
        validation_scores = svm.decision_function(x_validation)
        threshold = select_threshold(y_validation, validation_scores)
        test_scores = svm.decision_function(x_test)

        result = {
            **plan,
            "seed": fold_seed,
            "epochs": epochs,
            "best_validation_loss": float(best_loss),
            "score_type": "svm_decision_function",
            "validation": metrics(y_validation, validation_scores, threshold),
            "test": metrics(y_test, test_scores, threshold),
            "deep_feature_dim": backbone.feature_dim,
            "correlation_feature_dim": 171,
        }
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, fold_dir / "efficientnet_checkpoint.pt")
        joblib.dump(svm, fold_dir / "svm.joblib")
        np.savez_compressed(
            fold_dir / "test_predictions.npz",
            labels=y_test,
            scores=test_scores,
            predictions=(test_scores >= threshold).astype(np.int8),
            recording_ids=recording_ids[test_i],
            source_indices=test_i,
        )
        (fold_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return result
    finally:
        train.close()
        validation.close()
        test.close()


def aggregate_results(results: list[dict], expected_folds: int) -> dict:
    keys = [
        "sensitivity",
        "specificity",
        "roc_auc",
        "pr_auc",
        "false_positive_windows_per_hour",
    ]
    macro = {}
    for key in keys:
        values = np.asarray([result["test"][key] for result in results], dtype=float)
        macro[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    totals = {
        name: sum(result["test"][name] for result in results)
        for name in ("true_positives", "true_negatives", "false_positives", "false_negatives")
    }
    interictal_hours = sum(result["test"]["interictal_hours"] for result in results)
    pooled = {
        **{name: int(value) for name, value in totals.items()},
        "sensitivity": float(
            totals["true_positives"]
            / max(totals["true_positives"] + totals["false_negatives"], 1)
        ),
        "specificity": float(
            totals["true_negatives"]
            / max(totals["true_negatives"] + totals["false_positives"], 1)
        ),
        "false_positive_windows_per_hour": float(
            totals["false_positives"] / max(interictal_hours, 1e-12)
        ),
    }
    return {
        "complete_cross_validation": len(results) == expected_folds,
        "n_completed_folds": len(results),
        "n_expected_folds": expected_folds,
        "macro_test_metrics": macro,
        "pooled_thresholded_test_metrics": pooled,
        "pooling_note": (
            "ROC-AUC and PR-AUC are macro-averaged across folds. Raw SVM margins "
            "from separately fitted folds are not pooled as if they were calibrated probabilities."
        ),
        "folds": results,
    }


def run(args) -> None:
    data = args.data.resolve()
    manifest = seizure_cluster_manifest(data)
    plans = fold_plan(data, manifest, args.interictal_ratio, args.seed)
    preview = {"cluster_manifest": manifest, "fold_plan": plans}
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fold_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != preview:
            raise FileExistsError(
                f"{manifest_path} describes a different experiment; choose a new --output-dir"
            )
    else:
        manifest_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")

    if args.fold is not None and args.fold > len(plans):
        raise ValueError(f"--fold must be between 1 and {len(plans)} for this subject")
    selected = plans if args.fold is None else [plans[args.fold - 1]]
    results = []
    for plan in selected:
        fold_dir = output_dir / f"fold_{plan['fold']:02d}"
        result_path = fold_dir / "results.json"
        if result_path.exists() and not args.overwrite:
            print(f"fold={plan['fold']} already complete; loading preserved result")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        if fold_dir.exists() and any(fold_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Incomplete output exists in {fold_dir}; inspect it, then use a new "
                "--output-dir or explicitly pass --overwrite"
            )
        results.append(
            run_fold(
                data,
                plan,
                fold_dir,
                args.epochs,
                args.batch_size,
                args.interictal_ratio,
                args.seed,
            )
        )

    aggregate = aggregate_results(results, len(plans))
    suffix = "all" if args.fold is None else f"fold_{args.fold:02d}"
    aggregate_path = output_dir / f"cross_validation_results_{suffix}.json"
    if aggregate_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {aggregate_path}")
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=OUTPUT_DIR / "preprocessed_global_v2" / "chb01_preprocessed_global_v2.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "efficientnet_svm_cluster_cv_chb01",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--interictal-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(1, 100))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    run(parsed)
