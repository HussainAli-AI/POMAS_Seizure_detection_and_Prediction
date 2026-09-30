"""STFT Spectrogram and Correlation Map feature extraction.

Phase 3A: Per-channel STFT -> multichannel spectrogram image + Pearson correlation map.
"""

import numpy as np
from scipy import signal as scipy_signal
from config import SAMPLE_RATE


def compute_stft(
    data: np.ndarray,
    fs: int = SAMPLE_RATE,
    nperseg: int = 256,
    noverlap: int = 192,
    nfft: int = 256,
) -> np.ndarray:
    """Compute STFT spectrogram for each channel.

    Args:
        data: (n_channels, n_samples) EEG window
        fs: sampling rate
        nperseg: samples per segment (1s at 256 Hz = 256)
        noverlap: overlap samples (256-64=192 for hop of 64 = 0.25s)
        nfft: FFT size

    Returns:
        spectrograms: (n_channels, n_freqs, n_times) where
            n_freqs = nfft // 2 + 1 = 129
            n_times ~ 39 for a 10s window with hop=64
    """
    n_channels = data.shape[0]
    freqs, times, stft = scipy_signal.stft(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=1,
    )
    magnitude = np.abs(stft)
    return magnitude.astype(np.float32), freqs, times


def compute_correlation_map(data: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation map between all channel pairs.

    Args:
        data: (n_channels, n_samples) EEG window

    Returns:
        corr_map: (n_channels, n_channels) correlation matrix
    """
    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr.astype(np.float32)


def extract_stft_features(windows: np.ndarray) -> dict:
    """Extract STFT and correlation features from EEG windows.

    Args:
        windows: (n_windows, n_channels, n_samples) preprocessed EEG

    Returns:
        dict with keys:
            - spectrograms: (n_windows, n_channels, 129, n_times)
            - corr_maps: (n_windows, n_channels, n_channels)
    """
    n_windows, n_channels, n_samples = windows.shape
    first_spectrogram, _, _ = compute_stft(windows[0])
    n_freqs, n_times = first_spectrogram.shape[1:]

    spectrograms = np.zeros((n_windows, n_channels, n_freqs, n_times), dtype=np.float32)
    corr_maps = np.zeros((n_windows, n_channels, n_channels), dtype=np.float32)

    for i in range(n_windows):
        spec, _, _ = compute_stft(windows[i])
        spectrograms[i] = spec
        corr_maps[i] = compute_correlation_map(windows[i])

    return {"spectrograms": spectrograms, "corr_maps": corr_maps}
