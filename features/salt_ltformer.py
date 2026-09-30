"""SALT LTformer Detection Backbone.

Phase 3B: 2-layer LSTM (256 hidden) + Transformer encoder (4 heads, 2 layers, d_model=128)
-> classification head for seizure detection.

Supports fallback to LSTM-only variant if GPU memory is constrained (6GB VRAM constraint).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SALT_LTformer(nn.Module):
    """SALT LTformer: LSTM + Transformer encoder for EEG seizure detection.

    Args:
        n_channels: number of EEG channels (default 19)
        window_size: number of time samples per window (default 2560 for 10s at 256Hz)
        lstm_hidden: LSTM hidden size (default 256)
        lstm_layers: number of LSTM layers (default 2)
        d_model: Transformer model dimension (default 128)
        nhead: number of Transformer heads (default 4)
        num_layers: number of Transformer encoder layers (default 2)
        num_classes: number of output classes (default 3: interictal/preictal/ictal)
        dropout: dropout rate (default 0.3)
    """

    def __init__(
        self,
        n_channels: int = 19,
        window_size: int = 2560,
        patch_size: int = 32,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.patch_size = patch_size
        self.lstm_hidden = lstm_hidden
        self.d_model = d_model

        if window_size % patch_size:
            raise ValueError("window_size must be divisible by patch_size")
        # Each token represents 125 ms of all 19 EEG channels.  This gives the
        # recurrent and Transformer layers a temporal, rather than channel,
        # sequence to model.
        self.input_proj = nn.Linear(n_channels * patch_size, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        self.pos_encoder = PositionalEncoding(lstm_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=nhead,
            dim_feedforward=lstm_hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_samples = x.shape
        if n_channels != self.n_channels or n_samples != self.window_size:
            raise ValueError(f"Expected ({self.n_channels}, {self.window_size}) EEG window, got ({n_channels}, {n_samples})")
        x = x.reshape(batch_size, n_channels, n_samples // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, n_channels * self.patch_size)
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class LSTMOnly(nn.Module):
    """LSTM-only fallback variant for constrained GPU memory.

    Replaces Transformer with a second LSTM layer.
    """

    def __init__(
        self,
        n_channels: int = 19,
        window_size: int = 2560,
        lstm_hidden: int = 256,
        lstm_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(window_size, lstm_hidden)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_samples = x.shape
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def create_model(use_transformer: bool = True, **kwargs) -> nn.Module:
    """Factory function to create LTformer or LSTM-only model.

    Args:
        use_transformer: if True, create SALT_LTformer; else create LSTMOnly
        **kwargs: model kwargs

    Returns:
        nn.Module
    """
    if use_transformer:
        return SALT_LTformer(**kwargs)
    return LSTMOnly(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
