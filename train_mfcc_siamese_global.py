"""Patient-independent MFCC-Siamese prediction on global-v2 archives.

The split is by complete subject: chb01/chb02 train, chb03 validation, and
chb05 test by default.  Ictal windows are excluded because this branch is an
early-warning task (interictal=0 versus preictal=1).

All outputs use a new directory.  Existing results are preserved unless the
user explicitly supplies ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import NUM_CHANNELS, OUTPUT_DIR, SAMPLE_RATE, STRIDE_SECONDS, WINDOW_SECONDS, WINDOW_SIZE
from features.mfcc_siamese import MFCCSiamesePredictor, patient_contrastive_loss


EXPECTED_TIMELINE = "subject_global_summary_clock_v2"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ArchiveManifest:
    subject: str
    path: str
    n_windows: int
    n_interictal: int
    n_preictal: int
    n_ictal: int
    window_shape: tuple[int, int]
    label_timeline: str


class SubjectArchive:
    """Validated, lazily opened global-v2 archive for one subject."""

    def __init__(self, path: Path, subject: str, patient_index: int = -1) -> None:
        self.path = Path(path)
        self.subject = subject
        self.patient_index = patient_index
        self._file: Optional[h5py.File] = None
        if not self.path.exists():
            raise FileNotFoundError(f"Missing global-v2 archive: {self.path}")

        with h5py.File(self.path, "r") as file:
            required = {"windows", "labels", "timestamps", "recording_ids"}
            missing = required - set(file.keys())
            if missing:
                raise ValueError(f"{self.path} is missing datasets: {sorted(missing)}")
            shape = tuple(file["windows"].shape)
            if len(shape) != 3 or shape[1:] != (NUM_CHANNELS, WINDOW_SIZE):
                raise ValueError(
                    f"{self.path} has window shape {shape[1:]}; "
                    f"expected {(NUM_CHANNELS, WINDOW_SIZE)}"
                )
            timeline = str(file.attrs.get("label_timeline", ""))
            if timeline != EXPECTED_TIMELINE:
                raise ValueError(
                    f"{self.path} is not a corrected global-v2 archive "
                    f"(label_timeline={timeline!r})"
                )
            archive_subject = str(file.attrs.get("subject_id", ""))
            if archive_subject and archive_subject != subject:
                raise ValueError(
                    f"Subject mismatch: requested {subject}, archive contains {archive_subject}"
                )
            self.labels = file["labels"][:].astype(np.int64)
            self.timestamps = file["timestamps"][:].astype(np.float64)
            self.recording_ids = file["recording_ids"].asstr()[:]

        self.eligible = np.flatnonzero(np.isin(self.labels, [0, 1]))
        self.by_label = {
            label: np.flatnonzero(self.labels == label).astype(np.int64)
            for label in (0, 1)
        }
        for label, indices in self.by_label.items():
            if not len(indices):
                raise ValueError(f"{subject} has no windows for class {label}")

    @property
    def manifest(self) -> ArchiveManifest:
        counts = np.bincount(self.labels.clip(min=0), minlength=3)
        return ArchiveManifest(
            subject=self.subject,
            path=str(self.path.resolve()),
            n_windows=int(len(self.labels)),
            n_interictal=int(counts[0]),
            n_preictal=int(counts[1]),
            n_ictal=int(counts[2]),
            window_shape=(NUM_CHANNELS, WINDOW_SIZE),
            label_timeline=EXPECTED_TIMELINE,
        )

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def window(self, source_index: int) -> torch.Tensor:
        array = self._handle()["windows"][int(source_index)].astype(np.float32, copy=False)
        return torch.from_numpy(array)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state


class BalancedSiamesePairs(Dataset):
    """Deterministic balanced seizure-class and patient-relation pairs."""

    def __init__(self, archives: list[SubjectArchive], samples_per_epoch: int, seed: int) -> None:
        if len(archives) < 2:
            raise ValueError("Siamese training requires at least two subjects")
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
        mixed_seed = self.seed + self.epoch * 1_000_003 + index * 97
        rng = np.random.default_rng(mixed_seed)
        n_subjects = len(self.archives)

        anchor_subject = index % n_subjects
        anchor_label = (index // n_subjects) % 2
        same_patient = (index // (2 * n_subjects)) % 2 == 0
        if same_patient:
            partner_subject = anchor_subject
        else:
            choices = [subject for subject in range(n_subjects) if subject != anchor_subject]
            partner_subject = int(rng.choice(choices))
        partner_label = int(rng.integers(0, 2))

        first_archive = self.archives[anchor_subject]
        second_archive = self.archives[partner_subject]
        first_index = int(rng.choice(first_archive.by_label[anchor_label]))
        second_index = int(rng.choice(second_archive.by_label[partner_label]))

        return (
            first_archive.window(first_index),
            torch.tensor(anchor_label, dtype=torch.float32),
            torch.tensor(anchor_subject, dtype=torch.long),
            second_archive.window(second_index),
            torch.tensor(partner_label, dtype=torch.float32),
            torch.tensor(partner_subject, dtype=torch.long),
            torch.tensor(float(same_patient), dtype=torch.float32),
        )


class SubjectEvaluationWindows(Dataset):
    """All non-ictal windows from one unseen subject, in archive order."""

    def __init__(self, archive: SubjectArchive) -> None:
        self.archive = archive
        self.indices = archive.eligible

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        return (
            self.archive.window(source_index),
            torch.tensor(int(self.archive.labels[source_index]), dtype=torch.long),
            torch.tensor(float(self.archive.timestamps[source_index]), dtype=torch.float64),
            torch.tensor(source_index, dtype=torch.long),
        )


def window_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    prediction = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    interictal_hours = max((tn + fp) * STRIDE_SECONDS / 3600, 1e-12)
    return {
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": float(
            0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        ),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "false_positive_windows_per_hour": float(fp / interictal_hours),
        "threshold": float(threshold),
        "n_windows": int(len(y_true)),
        "n_preictal": int(y_true.sum()),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "interictal_hours": float(interictal_hours),
    }


def select_validation_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Select a balanced-accuracy threshold using validation scores only."""
    candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 199)))
    if not len(candidates):
        return 0.5
    best_key = (-np.inf, -np.inf, -np.inf)
    best_threshold = float(candidates[0])
    for candidate in candidates:
        prediction = y_score >= candidate
        positive = y_true == 1
        negative = ~positive
        sensitivity = float(np.sum(prediction & positive) / max(np.sum(positive), 1))
        specificity = float(np.sum(~prediction & negative) / max(np.sum(negative), 1))
        key = (0.5 * (sensitivity + specificity), sensitivity, specificity)
        if key > best_key:
            best_key = key
            best_threshold = float(candidate)
    return best_threshold


def alarm_event_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    refractory_seconds: float = 1800.0,
) -> dict:
    """Evaluate refractory alarm events and contiguous preictal episodes.

    These are offline research metrics.  ``preictal_episode_sensitivity`` is
    deliberately not called seizure-level sensitivity because adjacent
    seizures can share one 30-minute preictal episode.
    """
    order = np.argsort(timestamps, kind="stable")
    labels = y_true[order]
    scores = y_score[order]
    times = timestamps[order]

    episode_ranges: list[tuple[float, float]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    maximum_gap = STRIDE_SECONDS * 1.5
    for label, timestamp in zip(labels, times):
        if label == 1:
            if current_end is None or timestamp - current_end > maximum_gap:
                if current_start is not None and current_end is not None:
                    episode_ranges.append((current_start, current_end + WINDOW_SECONDS))
                current_start = float(timestamp)
            current_end = float(timestamp)
        elif current_start is not None and current_end is not None:
            episode_ranges.append((current_start, current_end + WINDOW_SECONDS))
            current_start = current_end = None
    if current_start is not None and current_end is not None:
        episode_ranges.append((current_start, current_end + WINDOW_SECONDS))

    alarm_times: list[float] = []
    last_alarm = -np.inf
    for timestamp, score in zip(times, scores):
        if score >= threshold and timestamp - last_alarm >= refractory_seconds:
            alarm_times.append(float(timestamp))
            last_alarm = float(timestamp)

    detected = sum(
        any(start <= alarm < end for alarm in alarm_times)
        for start, end in episode_ranges
    )
    true_alarms = sum(
        any(start <= alarm < end for start, end in episode_ranges)
        for alarm in alarm_times
    )
    false_alarms = len(alarm_times) - true_alarms
    interictal_hours = max(float(np.sum(labels == 0) * STRIDE_SECONDS / 3600), 1e-12)
    return {
        "preictal_episode_sensitivity": float(detected / max(len(episode_ranges), 1)),
        "n_preictal_episodes": int(len(episode_ranges)),
        "n_detected_preictal_episodes": int(detected),
        "n_alarm_events": int(len(alarm_times)),
        "n_true_alarm_events": int(true_alarms),
        "n_false_alarm_events": int(false_alarms),
        "false_alarm_events_per_hour": float(false_alarms / interictal_hours),
        "refractory_seconds": float(refractory_seconds),
    }


def evaluate(
    model: MFCCSiamesePredictor,
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
                logits, _, _ = model(windows)
            labels.append(target.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
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
    model: MFCCSiamesePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    patient_loss_weight: float,
    contrastive_loss_weight: float,
) -> dict:
    model.train()
    seizure_criterion = nn.BCEWithLogitsLoss()
    patient_criterion = nn.CrossEntropyLoss()
    totals = {"loss": 0.0, "seizure": 0.0, "patient": 0.0, "contrastive": 0.0}

    for first, first_label, first_patient, second, second_label, second_patient, same in loader:
        optimizer.zero_grad(set_to_none=True)
        combined = torch.cat((first, second), dim=0).to(device, non_blocking=True)
        seizure_targets = torch.cat((first_label, second_label)).to(device, non_blocking=True)
        patient_targets = torch.cat((first_patient, second_patient)).to(device, non_blocking=True)
        same = same.to(device, non_blocking=True)
        pair_count = len(first)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            seizure_logits, patient_logits, embeddings = model(combined)
            seizure_loss = seizure_criterion(seizure_logits, seizure_targets)
            patient_loss = patient_criterion(patient_logits, patient_targets)
            contrastive = patient_contrastive_loss(
                embeddings[:pair_count], embeddings[pair_count:], same
            )
            loss = (
                seizure_loss
                + patient_loss_weight * patient_loss
                + contrastive_loss_weight * contrastive
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.item())
        totals["seizure"] += float(seizure_loss.item())
        totals["patient"] += float(patient_loss.item())
        totals["contrastive"] += float(contrastive.item())

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def archive_path(data_dir: Path, subject: str) -> Path:
    return data_dir / f"{subject}_preprocessed_global_v2.h5"


def validate_subject_split(
    train_subjects: list[str], validation_subject: str, test_subject: str
) -> None:
    if len(set(train_subjects)) != len(train_subjects):
        raise ValueError("Training subjects must be unique")
    if len(train_subjects) < 2:
        raise ValueError("At least two training subjects are required")
    groups = set(train_subjects), {validation_subject}, {test_subject}
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Training, validation, and test subjects must be disjoint")


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
    test_archive = SubjectArchive(archive_path(args.data_dir, args.test_subject), args.test_subject)
    all_archives = train_archives + [validation_archive, test_archive]

    try:
        manifests = [archive.manifest.__dict__ for archive in all_archives]
        print(json.dumps({"device": str(device), "archives": manifests}, indent=2))
        pairs = BalancedSiamesePairs(train_archives, args.samples_per_epoch, args.seed)
        model = MFCCSiamesePredictor(
            n_channels=NUM_CHANNELS,
            n_train_subjects=len(train_archives),
            sample_rate=SAMPLE_RATE,
        ).to(device)

        if args.dry_run:
            sample = next(iter(DataLoader(pairs, batch_size=2, num_workers=0)))
            combined = torch.cat((sample[0], sample[3]), dim=0).to(device)
            with torch.no_grad():
                seizure_logits, patient_logits, embeddings = model(combined)
            print(
                json.dumps(
                    {
                        "dry_run": "ok",
                        "combined_windows": list(combined.shape),
                        "seizure_logits": list(seizure_logits.shape),
                        "patient_logits": list(patient_logits.shape),
                        "embeddings": list(embeddings.shape),
                        "parameters": int(sum(p.numel() for p in model.parameters())),
                    },
                    indent=2,
                )
            )
            return

        output_files = {
            "checkpoint": args.output_dir / "mfcc_siamese_checkpoint.pt",
            "results": args.output_dir / "results.json",
            "predictions": args.output_dir / "test_predictions.npz",
        }
        existing = [path for path in output_files.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing MFCC-Siamese outputs: "
                + ", ".join(str(path) for path in existing)
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)

        validation_set = SubjectEvaluationWindows(validation_archive)
        test_set = SubjectEvaluationWindows(test_archive)
        train_loader = DataLoader(
            pairs,
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
            pairs.set_epoch(epoch)
            losses = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                args.patient_loss_weight,
                args.contrastive_loss_weight,
            )
            val_labels, val_scores, _, _ = evaluate(
                model, validation_loader, validation_archive, device
            )
            val_threshold = select_validation_threshold(val_labels, val_scores)
            val_metrics = window_metrics(val_labels, val_scores, val_threshold)
            key = (val_metrics["pr_auc"], val_metrics["roc_auc"])
            history.append({"epoch": epoch, "losses": losses, "validation": val_metrics})
            print(
                f"epoch={epoch}/{args.epochs} loss={losses['loss']:.4f} "
                f"seizure_loss={losses['seizure']:.4f} "
                f"val_roc_auc={val_metrics['roc_auc']:.4f} "
                f"val_pr_auc={val_metrics['pr_auc']:.4f}"
            )
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
            "batch_size_pairs": args.batch_size,
            "samples_per_epoch_pairs": args.samples_per_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patient_loss_weight": args.patient_loss_weight,
            "contrastive_loss_weight": args.contrastive_loss_weight,
            "refractory_seconds": args.refractory_seconds,
            "seed": args.seed,
            "input_channels": NUM_CHANNELS,
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "mfcc_coefficients": 13,
            "mfcc_hop_length": 64,
            "embedding_dim": 100,
            "method_note": (
                "PoMAS 19-channel/256-Hz adaptation retaining MFCC time maps; "
                "not an exact 23-channel reproduction of the cited paper."
            ),
        }
        results = {
            "complete": True,
            "configuration": configuration,
            "archives": manifests,
            "validation": {"window_metrics": validation_window, "event_metrics": validation_events},
            "test": {"window_metrics": test_window, "event_metrics": test_events},
            "training_history": history,
        }

        torch.save(
            {
                "model_state": best_state,
                "configuration": configuration,
                "validation_threshold": threshold,
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
        "--output-dir", type=Path, default=OUTPUT_DIR / "mfcc_siamese_global_v2"
    )
    parser.add_argument("--train-subjects", nargs="+", default=["chb01", "chb02"])
    parser.add_argument("--validation-subject", default="chb03")
    parser.add_argument("--test-subject", default="chb05")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16, help="Siamese pairs per batch")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patient-loss-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--refractory-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
