"""GNN Electrode Topology feature extractor.

Phase 3C: Graph with 19 nodes (electrodes), spatial-distance adjacency.
Node features: spectral power in 5 bands (delta/theta/alpha/beta/gamma).
2-layer GraphConv + global pooling -> seizure classification.
Lightweight (< 1M params), runs on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal
from config import SAMPLE_RATE

BAND_FREQS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}
N_BANDS = len(BAND_FREQS)
N_CHANNELS = 19


def compute_spectral_power(data: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Compute band power features for each channel.

    Args:
        data: (n_channels, n_samples) EEG window

    Returns:
        features: (n_channels, N_BANDS) spectral power in each band
    """
    n_channels, n_samples = data.shape
    features = np.zeros((n_channels, N_BANDS), dtype=np.float32)

    for i, (band, (low, high)) in enumerate(BAND_FREQS.items()):
        sos = scipy_signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, data, axis=1)
        features[:, i] = np.mean(filtered ** 2, axis=1)

    features = np.log1p(features)
    return features


def compute_adjacency_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute spatial-distance adjacency matrix from electrode positions.

    Args:
        positions: (n_nodes, 2) xy coordinates of electrodes

    Returns:
        adj: (n_nodes, n_nodes) adjacency matrix with Gaussian kernel weights
    """
    from sklearn.metrics.pairwise import rbf_kernel
    gamma = 1.0 / (2 * (0.15 ** 2))
    adj = rbf_kernel(positions, gamma=gamma)
    np.fill_diagonal(adj, 0)
    return adj.astype(np.float32)


ELECTRODE_POSITIONS = np.array([
    [0.0, 0.0],  # placeholder - expand with actual 10-20 positions
])


class GraphConvLayer(nn.Module):
    """Simple graph convolution: H' = sigma(A_hat @ H @ W)"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return F.relu(x)


class GNNTopology(nn.Module):
    """2-layer GraphConv network for electrode topology.

    Input: (batch, n_nodes, n_features) node features
    Output: (batch, num_classes) classification logits
    """

    def __init__(
        self,
        n_nodes: int = 19,
        n_features: int = N_BANDS,
        hidden_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.conv1 = GraphConvLayer(n_features, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.conv1(x, adj)
        x = self.dropout(x)
        x = self.conv2(x, adj)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def extract_gnn_features(windows: np.ndarray, adj: np.ndarray) -> dict:
    """Extract GNN node features from EEG windows.

    Args:
        windows: (n_windows, n_channels, n_samples) EEG windows
        adj: (n_nodes, n_nodes) precomputed adjacency matrix

    Returns:
        dict with keys:
            - node_features: (n_windows, n_nodes, N_BANDS)
            - adj: (n_nodes, n_nodes)
    """
    n_windows = windows.shape[0]
    features = np.zeros((n_windows, windows.shape[1], N_BANDS), dtype=np.float32)
    for i in range(n_windows):
        features[i] = compute_spectral_power(windows[i])
    return {"node_features": features, "adj": adj}
