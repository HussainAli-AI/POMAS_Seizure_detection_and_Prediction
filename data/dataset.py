from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATASET_DIR, SAMPLE_RATE, WINDOW_SIZE, STRIDE,
    SELECTED_CHANNELS, NUM_CHANNELS, PREICTAL_SECONDS, POSTICTAL_SECONDS,
    OUTPUT_DIR,
)
from data.summary_parser import parse_all_summaries
from data.chb_mit_parser import load_edf


def label_window(seizures: List[Tuple[int, int]], window_start: float, window_end: float) -> int:
    """Assign label for a window.

    0 = interictal
    1 = preictal (within 30 min before seizure onset)
    2 = ictal (during seizure)
    -1 = excluded postictal/recovery window
    """
    for sz_start, sz_end in seizures:
        if window_start < sz_end and window_end > sz_start:
            return 2
    for sz_start, sz_end in seizures:
        if sz_start - PREICTAL_SECONDS <= window_end < sz_start:
            return 1
    for _, sz_end in seizures:
        if sz_end is not None and sz_end <= window_start < sz_end + POSTICTAL_SECONDS:
            return -1
    return 0


class CHBMITDataset(Dataset):
    """Dataset for CHB-MIT EEG recordings.

    Yields (eeg_window, label, patient_id, timestamp) tuples.
    """

    def __init__(
        self,
        dataset_dir: Path = DATASET_DIR,
        subjects: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        transform: Optional[Callable] = None,
        preload: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.window_size = window_size
        self.stride = stride
        self.channels = channels or SELECTED_CHANNELS
        self.transform = transform

        summaries = parse_all_summaries(self.dataset_dir)
        if subjects:
            summaries = {k: v for k, v in summaries.items() if k in subjects}

        self.windows = []
        if preload:
            self._preload(summaries)

    def _get_available_channels(self, raw) -> Optional[List[str]]:
        """Return the requested montage, or None when it is unavailable.

        CHB-MIT records contain two ``T8-P8`` labels.  MNE disambiguates them
        as ``T8-P8-0`` and ``T8-P8-1``; the first is the montage channel used
        here.  Older MNE versions may use ``T8-P8-2`` instead.
        """
        available = []
        for ch in self.channels:
            if ch in raw.ch_names:
                available.append(ch)
            elif ch == "T8-P8":
                alias = next((name for name in ("T8-P8-0", "T8-P8-2") if name in raw.ch_names), None)
                if alias is None:
                    return None
                available.append(alias)
            else:
                return None
        return available

    def _preload(self, summaries: Dict):
        for subj_name, info in summaries.items():
            subj_dir = self.dataset_dir / subj_name
            for file_info in info["files"]:
                edf_path = subj_dir / file_info["name"]
                if not edf_path.exists():
                    continue
                try:
                    raw = load_edf(edf_path)
                except Exception:
                    continue
                pick = self._get_available_channels(raw)
                if pick is None:
                    continue
                try:
                    raw.pick(pick)
                except Exception:
                    continue
                data, _ = raw.get_data(return_times=True)
                n_samples = data.shape[1]
                seizures = file_info.get("seizures", [])
                file_start_sec = 0

                for start in range(0, n_samples - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    window_data = data[:, start:end]
                    win_start_sec = file_start_sec + start / SAMPLE_RATE
                    win_end_sec = win_start_sec + self.window_size / SAMPLE_RATE
                    label = label_window(seizures, win_start_sec, win_end_sec)
                    if label == -1:
                        continue
                    self.windows.append({
                        "data": window_data.astype(np.float32),
                        "label": label,
                        "patient_id": subj_name,
                        "timestamp": win_start_sec,
                    "file": file_info["name"],
                    })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        x = torch.from_numpy(item["data"])
        y = torch.tensor(item["label"], dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y, item["patient_id"], item["timestamp"]


def create_dataloader(
    dataset: CHBMITDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
