"""MFCC-Siamese components for patient-independent seizure prediction.

This is a PoMAS-compatible adaptation of the patient-independent method cited
in the proposal.  It keeps the temporal MFCC map instead of averaging it, uses
same-patient/different-patient contrastive pairs, and adds both seizure and
training-patient classification heads.  The input remains the common 19-channel
CHB-MIT montage used by the rest of PoMAS.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _dct_basis(n_mfcc: int, n_mels: int) -> torch.Tensor:
    """Return an orthonormal DCT-II basis."""
    mel_positions = np.arange(n_mels, dtype=np.float32) + 0.5
    coefficient_ids = np.arange(n_mfcc, dtype=np.float32)[:, None]
    basis = np.cos(math.pi / n_mels * coefficient_ids * mel_positions)
    basis[0] *= math.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= math.sqrt(2.0 / n_mels)
    return torch.from_numpy(basis.astype(np.float32))


class TorchMFCC(nn.Module):
    """Batch MFCC extraction that retains the coefficient-by-time map.

    Input shape is ``(batch, channels, samples)``.  Output shape is
    ``(batch, channels, n_mfcc, frames)``.
    """

    def __init__(
        self,
        sample_rate: int = 256,
        n_fft: int = 256,
        hop_length: int = 64,
        n_mels: int = 13,
        n_mfcc: int = 13,
        fmin: float = 0.5,
        fmax: float = 50.0,
    ) -> None:
        super().__init__()
        if n_mfcc > n_mels:
            raise ValueError("n_mfcc cannot exceed n_mels")
        if fmax > sample_rate / 2:
            raise ValueError("fmax cannot exceed the Nyquist frequency")

        try:
            import librosa
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("librosa is required to construct the MFCC filter bank") from exc

        mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm="slaney",
        ).astype(np.float32)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.register_buffer("window", torch.hann_window(n_fft), persistent=True)
        self.register_buffer("mel_filter", torch.from_numpy(mel_filter), persistent=True)
        self.register_buffer("dct", _dct_basis(n_mfcc, n_mels), persistent=True)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("TorchMFCC expects (batch, channels, samples)")
        batch, channels, samples = windows.shape
        if samples < self.n_fft:
            raise ValueError("EEG windows must contain at least n_fft samples")

        # CUDA autocast can downcast the real input to a type unsupported by
        # complex STFT.  MFCC extraction is deterministic and intentionally
        # performed in float32; the CNN that follows can still use autocast.
        with torch.amp.autocast(device_type=windows.device.type, enabled=False):
            flattened = windows.float().reshape(batch * channels, samples)
            spectrum = torch.stft(
                flattened,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                return_complex=True,
            )
            power = spectrum.abs().square()
            mel_power = torch.einsum("mf,bft->bmt", self.mel_filter, power)
            log_mel = torch.log(mel_power.clamp_min(1e-10))
            mfcc = torch.einsum("cm,bmt->bct", self.dct, log_mel)
            mfcc = mfcc.reshape(batch, channels, self.n_mfcc, -1)
            mean = mfcc.mean(dim=(-2, -1), keepdim=True)
            std = mfcc.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            return (mfcc - mean) / std


class MFCCSiamesePredictor(nn.Module):
    """Dual-kernel MFCC encoder with seizure and patient-identity heads."""

    def __init__(
        self,
        n_channels: int = 19,
        n_train_subjects: int = 2,
        embedding_dim: int = 100,
        dropout: float = 0.3,
        sample_rate: int = 256,
    ) -> None:
        super().__init__()
        if n_train_subjects < 2:
            raise ValueError("At least two training subjects are required for Siamese pairing")
        self.mfcc = TorchMFCC(sample_rate=sample_rate)
        self.branch_9 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(5, 9), padding=(2, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.branch_11 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(5, 11), padding=(2, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.shared = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.seizure_head = nn.Linear(embedding_dim, 1)
        self.patient_head = nn.Linear(embedding_dim, n_train_subjects)

    def encode(self, windows: torch.Tensor) -> torch.Tensor:
        features = self.mfcc(windows)
        parallel = torch.cat((self.branch_9(features), self.branch_11(features)), dim=1)
        embedding = self.embedding(self.shared(parallel))
        # Keep the shared representation and distance calculation in float32.
        # Mixed-precision convolutions remain enabled, but threshold-sensitive
        # probabilities should not be quantized to float16 steps.
        return F.normalize(embedding.float(), dim=1)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(windows)
        with torch.amp.autocast(device_type=embedding.device.type, enabled=False):
            seizure_logits = self.seizure_head(embedding).squeeze(1)
            patient_logits = self.patient_head(embedding)
        return seizure_logits, patient_logits, embedding


def patient_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    same_patient: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pull same-patient embeddings together and separate different patients."""
    distances = F.pairwise_distance(first, second)
    same_patient = same_patient.float()
    positive = same_patient * distances.square()
    negative = (1.0 - same_patient) * F.relu(margin - distances).square()
    return (positive + negative).mean()
