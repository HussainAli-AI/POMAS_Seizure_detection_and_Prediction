"""MFCC + 1D-CNN feature extractor and Siamese network.

Phase 3D: 13 MFCC coefficients per channel -> 13 x 19 matrix per window.
3-layer 1D-CNN + global pooling + FC head for classification.
Siamese variant for patient-independent matching (contrastive loss).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SAMPLE_RATE


def compute_mfcc(data: np.ndarray, fs: int = SAMPLE_RATE, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCC features for each channel.

    Args:
        data: (n_channels, n_samples) EEG window
        fs: sampling rate
        n_mfcc: number of MFCC coefficients (default 13)

    Returns:
        mfcc_features: (n_channels, n_mfcc) MFCC matrix
    """
    try:
        import librosa
    except ImportError:
        return np.zeros((data.shape[0], n_mfcc), dtype=np.float32)

    n_channels = data.shape[0]
    features = np.zeros((n_channels, n_mfcc), dtype=np.float32)

    for ch in range(n_channels):
        mfcc = librosa.feature.mfcc(y=data[ch], sr=fs, n_mfcc=n_mfcc, n_fft=256, hop_length=64)
        features[ch] = np.mean(mfcc, axis=1)

    return features


class MFCCCNN(nn.Module):
    """3-layer 1D-CNN for MFCC-based EEG classification.

    Input: (batch, n_channels, n_mfcc) MFCC features
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        n_channels: int = 18,
        n_mfcc: int = 13,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class SiameseMFCC(nn.Module):
    """Siamese network for patient-independent MFCC matching.

    Uses contrastive loss: pulls same-class pairs together,
    pushes different-class pairs apart.

    Input: two (batch, n_channels, n_mfcc) MFCC feature tensors
    Output: similarity score (cosine similarity of embeddings)
    """

    def __init__(
        self,
        n_channels: int = 18,
        n_mfcc: int = 13,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        return torch.sum(emb1 * emb2, dim=1)


def extract_mfcc_features(windows: np.ndarray) -> np.ndarray:
    """Extract MFCC features from EEG windows.

    Args:
        windows: (n_windows, n_channels, n_samples) EEG windows

    Returns:
        mfcc_features: (n_windows, n_channels, n_mfcc)
    """
    n_windows = windows.shape[0]
    n_channels = windows.shape[1]
    n_mfcc = 13
    features = np.zeros((n_windows, n_channels, n_mfcc), dtype=np.float32)
    for i in range(n_windows):
        features[i] = compute_mfcc(windows[i])
    return features
