"""Dual-graph GNN components for patient-independent seizure prediction.

Each node is one bipolar EEG derivation from the common PoMAS montage.  The
physical graph connects derivations that share an electrode.  A second graph
is constructed for every window from absolute inter-channel correlation.
Node features are log power in delta, theta, alpha, beta, and gamma bands.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import SAMPLE_RATE, SELECTED_CHANNELS


BANDS_HZ = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 50.0),
)


def channel_electrodes(channel: str) -> frozenset[str]:
    """Return the canonical electrode endpoints of a bipolar derivation."""
    pieces = [piece.strip().upper() for piece in channel.split("-")]
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"Invalid bipolar channel name: {channel!r}")
    return frozenset(pieces)


def physical_adjacency(channels: list[str] | tuple[str, ...]) -> torch.Tensor:
    """Build a normalized graph connecting channels with a shared electrode."""
    endpoints = [channel_electrodes(channel) for channel in channels]
    count = len(endpoints)
    adjacency = torch.zeros(count, count, dtype=torch.float32)
    for first in range(count):
        for second in range(first + 1, count):
            if endpoints[first] & endpoints[second]:
                adjacency[first, second] = 1.0
                adjacency[second, first] = 1.0
    return normalize_adjacency(adjacency)


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric D^-1/2 (A + I) D^-1/2 normalization."""
    if adjacency.ndim not in (2, 3) or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("Adjacency must be square with shape (nodes,nodes) or (batch,nodes,nodes)")
    nodes = adjacency.shape[-1]
    identity = torch.eye(nodes, device=adjacency.device, dtype=adjacency.dtype)
    if adjacency.ndim == 3:
        identity = identity.unsqueeze(0)
    with_self = adjacency + identity
    degree = with_self.sum(dim=-1).clamp_min(1e-8)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt.unsqueeze(-1) * with_self * inverse_sqrt.unsqueeze(-2)


def functional_adjacency(windows: torch.Tensor, top_k: int = 4) -> torch.Tensor:
    """Construct normalized sparse absolute-correlation graphs per window."""
    if windows.ndim != 3:
        raise ValueError("functional_adjacency expects (batch, channels, samples)")
    channels = windows.shape[1]
    if not 1 <= top_k < channels:
        raise ValueError("top_k must be between 1 and channels-1")

    with torch.amp.autocast(device_type=windows.device.type, enabled=False):
        centered = windows.float() - windows.float().mean(dim=-1, keepdim=True)
        normalized = centered / centered.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        correlation = torch.bmm(normalized, normalized.transpose(1, 2)).abs()
        diagonal = torch.eye(channels, device=windows.device, dtype=torch.bool).unsqueeze(0)
        correlation = correlation.masked_fill(diagonal, 0.0)
        values, indices = torch.topk(correlation, k=top_k, dim=-1)
        sparse = torch.zeros_like(correlation).scatter(-1, indices, values)
        sparse = torch.maximum(sparse, sparse.transpose(1, 2))
        return normalize_adjacency(sparse)


class TorchBandPower(nn.Module):
    """Batched differentiable-free spectral node feature extraction."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 256,
        hop_length: int = 128,
        bands: tuple[tuple[float, float], ...] = BANDS_HZ,
    ) -> None:
        super().__init__()
        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        filters = []
        for index, (low, high) in enumerate(bands):
            if index == len(bands) - 1:
                mask = (frequencies >= low) & (frequencies <= high)
            else:
                mask = (frequencies >= low) & (frequencies < high)
            if not mask.any():
                raise ValueError(f"No FFT bins available for band {(low, high)}")
            weight = mask.float()
            filters.append(weight / weight.sum())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bands = bands
        self.register_buffer("window", torch.hann_window(n_fft), persistent=True)
        self.register_buffer("band_filters", torch.stack(filters), persistent=True)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("TorchBandPower expects (batch, channels, samples)")
        batch, channels, samples = windows.shape
        if samples < self.n_fft:
            raise ValueError("EEG windows must contain at least n_fft samples")
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
            band_power = torch.einsum("kf,bft->bkt", self.band_filters, power).mean(dim=-1)
            return torch.log(band_power.clamp_min(1e-10)).reshape(batch, channels, -1)


class GraphConvolution(nn.Module):
    """Normalized graph aggregation followed by a learned projection."""

    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)
        self.normalization = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features, bias=False)
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(features.shape[0], -1, -1)
        aggregated = torch.bmm(adjacency, features)
        update = self.normalization(self.projection(aggregated))
        return self.dropout(F.gelu(update) + self.residual(features))


class GraphStream(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (
                GraphConvolution(in_features, hidden_dim, dropout),
                GraphConvolution(hidden_dim, hidden_dim, dropout),
                GraphConvolution(hidden_dim, hidden_dim, dropout),
            )
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            features = layer(features, adjacency)
        return features


class DualGraphSeizurePredictor(nn.Module):
    """Physical/functional dual-stream GNN with a binary prediction head."""

    def __init__(
        self,
        channels: list[str] | tuple[str, ...] = tuple(SELECTED_CHANNELS),
        hidden_dim: int = 48,
        dropout: float = 0.3,
        functional_top_k: int = 4,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.functional_top_k = functional_top_k
        self.band_power = TorchBandPower(sample_rate=sample_rate)
        self.node_normalization = nn.LayerNorm(len(BANDS_HZ))
        self.physical_stream = GraphStream(len(BANDS_HZ), hidden_dim, dropout)
        self.functional_stream = GraphStream(len(BANDS_HZ), hidden_dim, dropout)
        self.register_buffer(
            "physical_graph", physical_adjacency(self.channels), persistent=True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.shape[1] != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} channels, received {windows.shape[1]}"
            )
        node_features = self.node_normalization(self.band_power(windows))
        functional_graph = functional_adjacency(windows, self.functional_top_k)
        physical = self.physical_stream(node_features, self.physical_graph)
        functional = self.functional_stream(node_features, functional_graph)
        combined = torch.cat((self._pool(physical), self._pool(functional)), dim=1)
        with torch.amp.autocast(device_type=combined.device.type, enabled=False):
            return self.classifier(combined.float()).squeeze(1)

