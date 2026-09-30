from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split

from data.dataset import CHBMITDataset


def loso_split(dataset: CHBMITDataset) -> Dict[str, Dict[str, List]]:
    """Leave-One-Subject-Out split.

    Returns dict mapping test_subject -> {"train": [...], "test": [...]}
    where values are lists of indices into the dataset.
    """
    patient_to_indices = {}
    for idx, item in enumerate(dataset.windows):
        pid = item["patient_id"]
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    splits = {}
    for test_patient, test_indices in patient_to_indices.items():
        train_indices = []
        for train_patient, indices in patient_to_indices.items():
            if train_patient != test_patient:
                train_indices.extend(indices)
        splits[test_patient] = {"train": train_indices, "test": test_indices}

    return splits


def patient_dependent_split(
    dataset: CHBMITDataset, test_ratio: float = 0.2
) -> Dict[str, Dict[str, List]]:
    """Per-subject held-out last 20% for patient-dependent evaluation."""
    patient_to_indices = {}
    for idx, item in enumerate(dataset.windows):
        pid = item["patient_id"]
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    splits = {}
    for patient, indices in patient_to_indices.items():
        indices.sort(key=lambda i: dataset.windows[i]["timestamp"])
        n_test = max(1, int(len(indices) * test_ratio))
        train_idx = indices[:-n_test]
        test_idx = indices[-n_test:]
        splits[patient] = {"train": train_idx, "test": test_idx}

    return splits
