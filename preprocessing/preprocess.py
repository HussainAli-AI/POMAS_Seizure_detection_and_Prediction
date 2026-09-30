from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from config import SAMPLE_RATE, BAND_FILTER_LOW, BAND_FILTER_HIGH, WINDOW_SIZE, STRIDE
from data.chb_mit_parser import load_edf


def bandpass_filter(data: np.ndarray, low: float = BAND_FILTER_LOW, high: float = BAND_FILTER_HIGH, fs: int = SAMPLE_RATE) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


def resample(data: np.ndarray, orig_fs: int, target_fs: int = SAMPLE_RATE) -> np.ndarray:
    if orig_fs == target_fs:
        return data
    n_samples = int(data.shape[1] * target_fs / orig_fs)
    return signal.resample(data, n_samples, axis=1)


def notch_filter(data: np.ndarray, freq: float = 60.0, fs: int = SAMPLE_RATE, quality: float = 30.0) -> np.ndarray:
    """Apply notch filter at specified frequency (60 Hz for US power line)."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, data, axis=1)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mean) / std


def extract_windows(data: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    n_channels, n_samples = data.shape
    starts = np.arange(0, n_samples - window_size + 1, stride)
    windows = np.zeros((len(starts), n_channels, window_size), dtype=np.float32)
    for i, s in enumerate(starts):
        windows[i] = data[:, s : s + window_size]
    return windows


def ica_artifact_removal(data: np.ndarray, fs: int = SAMPLE_RATE, n_components: Optional[int] = None) -> np.ndarray:
    """FastICA-based artifact removal (ECG/EOG)."""
    from sklearn.decomposition import FastICA
    n_channels = data.shape[0]
    n_comp = n_components or n_channels
    n_comp = min(n_comp, n_channels)
    ica = FastICA(n_components=n_comp, max_iter=500, random_state=42)
    components = ica.fit_transform(data.T)
    reconstructed = ica.inverse_transform(components)
    return reconstructed.T.astype(data.dtype)


def emd_filter(data: np.ndarray, max_imfs: int = 7, discard_first: bool = True) -> np.ndarray:
    """EMD-based filtering: decompose into IMFs, optionally discard first IMF (highest frequency noise)."""
    try:
        from PyEMD import EMD
    except ImportError:
        return data
    n_channels, n_samples = data.shape
    result = np.zeros_like(data)
    emd = EMD()
    for ch in range(n_channels):
        imfs = emd(data[ch], max_imf=max_imfs)
        if imfs.ndim == 1:
            imfs = imfs[np.newaxis, :]
        start = 1 if discard_first and imfs.shape[0] > 1 else 0
        result[ch] = np.sum(imfs[start:], axis=0)
    return result


def preprocess_file(
    edf_path: Path,
    pick_channels: Optional[List[str]] = None,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
    apply_ica: bool = False,
    apply_emd: bool = False,
    apply_zscore: bool = True,
    target_fs: int = SAMPLE_RATE,
    band_low: float = BAND_FILTER_LOW,
    band_high: float = BAND_FILTER_HIGH,
    notch_freq: float = 60.0,
) -> np.ndarray:
    raw = load_edf(edf_path)
    if pick_channels:
        available = [ch for ch in pick_channels if ch in raw.ch_names]
        missing_dup = [ch for ch in pick_channels if ch not in raw.ch_names]
        for ch in missing_dup:
            if ch == "T8-P8":
                alias = next((name for name in ("T8-P8-0", "T8-P8-2") if name in raw.ch_names), None)
                if alias is not None:
                    available.append(alias)
        if len(available) != len(pick_channels):
            missing = [
                ch for ch in pick_channels
                if ch not in raw.ch_names
                and not (ch == "T8-P8" and any(alias in raw.ch_names for alias in ("T8-P8-0", "T8-P8-2")))
            ]
            raise ValueError(f"Recording does not support the required montage; missing: {missing}")
        raw.pick(available)
    data, _ = raw.get_data(return_times=True)
    orig_fs = int(raw.info["sfreq"])

    if orig_fs != target_fs:
        data = resample(data, orig_fs, target_fs)

    if apply_bandpass:
        data = bandpass_filter(data, low=band_low, high=band_high, fs=target_fs)

    if apply_notch:
        data = notch_filter(data, freq=notch_freq, fs=target_fs)

    if apply_ica:
        data = ica_artifact_removal(data, fs=target_fs)

    if apply_emd:
        data = emd_filter(data)

    if apply_zscore:
        data = zscore_normalize(data)

    return data.astype(np.float32)
