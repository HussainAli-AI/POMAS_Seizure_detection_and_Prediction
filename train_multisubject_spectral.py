"""One patient-independent spectral baseline fold for the PoMAS HDF5 archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR
from data.hdf5_dataset import HDF5WindowDataset
from train_baseline import score_metrics
from train_spectral_baseline import extract, select_balanced_threshold


def load_subject(path: Path):
    dataset = HDF5WindowDataset(path)
    keep = dataset.labels != 2  # early-warning task: preictal vs interictal
    filtered = HDF5WindowDataset(path, np.flatnonzero(keep))
    dataset.close()
    x, y = extract(filtered)
    filtered.close()
    return x, y


def load_group(data_dir: Path, subjects: list[str]):
    pieces = [load_subject(data_dir / f"{subject}_preprocessed.h5") for subject in subjects]
    return np.concatenate([x for x, _ in pieces]), np.concatenate([y for _, y in pieces])


def run(data_dir: Path, train_subjects: list[str], validation_subject: str, test_subject: str):
    print(f"Extracting training features: {', '.join(train_subjects)}")
    x_train, y_train = load_group(data_dir, train_subjects)
    print(f"Extracting validation features: {validation_subject}")
    x_val, y_val = load_subject(data_dir / f"{validation_subject}_preprocessed.h5")
    print(f"Extracting test features: {test_subject}")
    x_test, y_test = load_subject(data_dir / f"{test_subject}_preprocessed.h5")

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    threshold = select_balanced_threshold(y_val, val_scores)
    results = score_metrics(y_test, test_scores, threshold)
    results.update({
        "train_subjects": train_subjects,
        "validation_subject": validation_subject,
        "test_subject": test_subject,
        "validation_threshold": threshold,
        "validation_metrics_at_threshold": score_metrics(y_val, val_scores, threshold),
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"multisubject_spectral_test_{test_subject}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(model, OUTPUT_DIR / f"multisubject_spectral_test_{test_subject}.joblib")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "preprocessed")
    parser.add_argument("--train-subjects", nargs="+", default=["chb01", "chb02"])
    parser.add_argument("--validation-subject", default="chb03")
    parser.add_argument("--test-subject", default="chb05")
    args = parser.parse_args()
    run(args.data_dir, args.train_subjects, args.validation_subject, args.test_subject)
