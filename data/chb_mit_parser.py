import struct
from pathlib import Path
from typing import List, Tuple, Optional
import mne
import numpy as np


def parse_seizures_file(filepath: Path) -> List[Tuple[int, int]]:
    """Parse a .edf.seizures binary file.

    Format (per Shoeb 2009): each record is 3 integers (4 bytes each, little-endian):
      - start_sample_offset (from beginning of recording)
      - end_sample_offset
      - channel_number (or 0 for all channels)
    """
    seizures = []
    data = filepath.read_bytes()
    record_size = 12
    for i in range(0, len(data), record_size):
        if i + record_size > len(data):
            break
        start, end, ch = struct.unpack_from("<III", data, i)
        seizures.append((int(start), int(end)))
    return seizures


def load_edf(filepath: Path, montage: Optional[List[str]] = None, pick_channels: Optional[List[str]] = None):
    """Load an EDF file using MNE, optionally pick channels."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    if pick_channels:
        available = [ch for ch in pick_channels if ch in raw.ch_names]
        if available:
            raw.pick(available)
    if montage is not None:
        try:
            raw.set_montage(montage, verbose=False)
        except Exception:
            pass
    return raw


def get_data_and_times(raw) -> Tuple[np.ndarray, np.ndarray]:
    """Extract data array and time vector from MNE Raw object.

    Returns (data, times) where data shape is (n_channels, n_times).
    """
    data, times = raw.get_data(return_times=True)
    return data, times
