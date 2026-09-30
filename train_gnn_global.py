"""Patient-independent dual-graph GNN training on global-v2 archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import NUM_CHANNELS, OUTPUT_DIR, SAMPLE_RATE, SELECTED_CHANNELS, WINDOW_SECONDS
from features.gnn_patient_independent import BANDS_HZ, DualGraphSeizurePredictor
from train_mfcc_siamese_global import (
    SubjectArchive,
    SubjectEvaluationWindows,
    alarm_event_metrics,
    archive_path,
    seed_everything,
    select_validation_threshold,
    validate_subject_split,
    window_metrics,
)


class BalancedSubjectWindows(Dataset):
    """Uniform sampling over training subjects and binary seizure classes."""

    def __init__(self, archives: list[SubjectArchive], samples_per_epoch: int, seed: int) -> None:
        if len(archives) < 2:
            raise ValueError("Patient-independent training requires at least two subjects")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.archives = archives
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        subject_index = index % len(self.archives)
        label = (index // len(self.archives)) % 2
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
        archive = self.archives[subject_index]
        source_index = int(rng.choice(archive.by_label[label]))
        return archive.window(source_index), torch.tensor(label, dtype=torch.float32)


def evaluate(
    model: DualGraphSeizurePredictor,
    loader: DataLoader,
    archive: SubjectArchive,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    with torch.no_grad():
        for windows, target, timestamp, source_index in loader:
            windows = windows.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(windows)
            labels.append(target.numpy())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            timestamps.append(timestamp.numpy())
            source_indices.append(source_index.numpy())
    result = (
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(timestamps),
        np.concatenate(source_indices),
    )
    if not np.array_equal(result[0], archive.labels[result[3]]):
        raise RuntimeError("Evaluation order no longer matches archive labels")
    return result


def train_epoch(
    model: DualGraphSeizurePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    for windows, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        windows = windows.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(windows)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
    return total / max(len(loader), 1)


def count_physical_edges(channels: list[str]) -> int:
    endpoints = [set(channel.upper().split("-")) for channel in channels]
    return sum(
        bool(endpoints[first] & endpoints[second])
        for first in range(len(endpoints))
        for second in range(first + 1, len(endpoints))
    )


def run(args: argparse.Namespace) -> None:
    validate_subject_split(args.train_subjects, args.validation_subject, args.test_subject)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_archives = [
        SubjectArchive(archive_path(args.data_dir, subject), subject, patient_index=index)
        for index, subject in enumerate(args.train_subjects)
    ]
    validation_archive = SubjectArchive(
        archive_path(args.data_dir, args.validation_subject), args.validation_subject
    )
    test_archive = SubjectArchive(
        archive_path(args.data_dir, args.test_subject), args.test_subject
    )
    all_archives = train_archives + [validation_archive, test_archive]

    try:
        manifests = [archive.manifest.__dict__ for archive in all_archives]
        print(json.dumps({"device": str(device), "archives": manifests}, indent=2))
        training_set = BalancedSubjectWindows(
            train_archives, args.samples_per_epoch, args.seed
        )
        model = DualGraphSeizurePredictor(
            channels=SELECTED_CHANNELS,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            functional_top_k=args.functional_top_k,
            sample_rate=SAMPLE_RATE,
        ).to(device)

        if args.dry_run:
            windows, labels = next(
                iter(DataLoader(training_set, batch_size=4, shuffle=False, num_workers=0))
            )
            with torch.no_grad():
                logits = model(windows.to(device))
            print(
                json.dumps(
                    {
                        "dry_run": "ok",
                        "windows": list(windows.shape),
                        "labels": labels.tolist(),
                        "logits": list(logits.shape),
                        "physical_graph": list(model.physical_graph.shape),
                        "physical_edges_without_self_loops": count_physical_edges(
                            SELECTED_CHANNELS
                        ),
                        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                    },
                    indent=2,
                )
            )
            return

        output_files = {
            "checkpoint": args.output_dir / "gnn_checkpoint.pt",
            "results": args.output_dir / "results.json",
            "predictions": args.output_dir / "test_predictions.npz",
        }
        existing = [path for path in output_files.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing GNN outputs: "
                + ", ".join(str(path) for path in existing)
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)

        validation_set = SubjectEvaluationWindows(validation_archive)
        test_set = SubjectEvaluationWindows(test_archive)
        train_loader = DataLoader(
            training_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        best_key = (-np.inf, -np.inf)
        best_epoch = 0
        best_state: Optional[dict[str, torch.Tensor]] = None
        history: list[dict] = []

        for epoch in range(1, args.epochs + 1):
            training_set.set_epoch(epoch)
            loss = train_epoch(model, train_loader, optimizer, scaler, device)
            val_labels, val_scores, _, _ = evaluate(
                model, validation_loader, validation_archive, device
            )
            val_threshold = select_validation_threshold(val_labels, val_scores)
            val_metrics = window_metrics(val_labels, val_scores, val_threshold)
            history.append({"epoch": epoch, "loss": loss, "validation": val_metrics})
            print(
                f"epoch={epoch}/{args.epochs} loss={loss:.4f} "
                f"val_roc_auc={val_metrics['roc_auc']:.4f} "
                f"val_pr_auc={val_metrics['pr_auc']:.4f}"
            )
            key = (val_metrics["pr_auc"], val_metrics["roc_auc"])
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")
        model.load_state_dict(best_state)
        val_labels, val_scores, val_timestamps, val_indices = evaluate(
            model, validation_loader, validation_archive, device
        )
        threshold = select_validation_threshold(val_labels, val_scores)
        test_labels, test_scores, test_timestamps, test_indices = evaluate(
            model, test_loader, test_archive, device
        )

        validation_window = window_metrics(val_labels, val_scores, threshold)
        test_window = window_metrics(test_labels, test_scores, threshold)
        validation_events = alarm_event_metrics(
            val_labels, val_scores, val_timestamps, threshold, args.refractory_seconds
        )
        test_events = alarm_event_metrics(
            test_labels, test_scores, test_timestamps, threshold, args.refractory_seconds
        )
        configuration = {
            "train_subjects": args.train_subjects,
            "validation_subject": args.validation_subject,
            "test_subject": args.test_subject,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "functional_top_k": args.functional_top_k,
            "refractory_seconds": args.refractory_seconds,
            "seed": args.seed,
            "input_channels": NUM_CHANNELS,
            "channel_names": SELECTED_CHANNELS,
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "bands_hz": BANDS_HZ,
            "physical_edges_without_self_loops": count_physical_edges(SELECTED_CHANNELS),
            "method_note": (
                "PoMAS adaptation using bipolar-channel nodes, shared-electrode physical edges, "
                "and per-window sparse absolute-correlation functional graphs; not an exact "
                "reproduction of the cited electrode-grid implementation."
            ),
        }
        results = {
            "complete": True,
            "configuration": configuration,
            "archives": manifests,
            "validation": {
                "window_metrics": validation_window,
                "event_metrics": validation_events,
            },
            "test": {"window_metrics": test_window, "event_metrics": test_events},
            "training_history": history,
        }

        torch.save(
            {
                "model_state": best_state,
                "configuration": configuration,
                "validation_threshold": threshold,
                "physical_graph": model.physical_graph.detach().cpu(),
            },
            output_files["checkpoint"],
        )
        np.savez_compressed(
            output_files["predictions"],
            labels=test_labels,
            scores=test_scores,
            predictions=(test_scores >= threshold).astype(np.int8),
            timestamps=test_timestamps,
            source_indices=test_indices,
            recording_ids=test_archive.recording_ids[test_indices],
        )
        output_files["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({"validation": results["validation"], "test": results["test"]}, indent=2))
        print(f"checkpoint={output_files['checkpoint']}")
        print(f"metrics={output_files['results']}")
    finally:
        for archive in all_archives:
            archive.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=OUTPUT_DIR / "preprocessed_global_v2"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR / "gnn_6train_global_v2"
    )
    parser.add_argument(
        "--train-subjects",
        nargs="+",
        default=["chb01", "chb02", "chb06", "chb07", "chb08", "chb10"],
    )
    parser.add_argument("--validation-subject", default="chb03")
    parser.add_argument("--test-subject", default="chb05")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--functional-top-k", type=int, default=4)
    parser.add_argument("--refractory-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

