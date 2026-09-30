"""Memory-efficient storage and loading for preprocessed EEG windows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5WindowWriter:
    """Append EEG windows to an HDF5 file without retaining prior files in RAM."""

    def __init__(self, path: Path, n_channels: int, window_size: int, overwrite: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        self.file = h5py.File(self.path, mode)
        self.windows = self.file.create_dataset(
            "windows", shape=(0, n_channels, window_size),
            maxshape=(None, n_channels, window_size), dtype="float32",
            chunks=(1, n_channels, window_size), compression="lzf",
        )
        self.labels = self.file.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="int8", chunks=True)
        self.timestamps = self.file.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="float64", chunks=True)
        text_type = h5py.string_dtype(encoding="utf-8")
        self.recording_ids = self.file.create_dataset("recording_ids", shape=(0,), maxshape=(None,), dtype=text_type, chunks=True)

    def append(self, windows: np.ndarray, labels: np.ndarray, recording_id: str, timestamps: np.ndarray) -> None:
        if len(windows) != len(labels) or len(windows) != len(timestamps):
            raise ValueError("windows, labels, and timestamps must have equal lengths")
        start = len(self.labels)
        end = start + len(labels)
        for dataset in (self.windows, self.labels, self.timestamps, self.recording_ids):
            dataset.resize((end,) + dataset.shape[1:])
        self.windows[start:end] = windows.astype(np.float32, copy=False)
        self.labels[start:end] = labels.astype(np.int8, copy=False)
        self.timestamps[start:end] = timestamps
        self.recording_ids[start:end] = [recording_id] * len(labels)

    def set_metadata(self, *, subject_id: str, channel_names: list[str], sample_rate: int) -> None:
        self.file.attrs["subject_id"] = subject_id
        self.file.attrs["channel_names"] = np.asarray(channel_names, dtype=h5py.string_dtype("utf-8"))
        self.file.attrs["sample_rate"] = sample_rate

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HDF5WindowDataset(Dataset):
    """PyTorch dataset that opens an HDF5 archive lazily per worker."""

    def __init__(self, path: Path, indices: Optional[np.ndarray] = None):
        self.path = str(Path(path))
        self._file: Optional[h5py.File] = None
        with h5py.File(self.path, "r") as file:
            self.labels = file["labels"][:].astype(np.int64)
            self.timestamps = file["timestamps"][:]
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices, dtype=np.int64)

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        file = self._handle()
        window = torch.from_numpy(file["windows"][source_index].astype(np.float32, copy=False))
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        return window, label, float(self.timestamps[source_index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
