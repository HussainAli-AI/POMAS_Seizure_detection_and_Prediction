"""Test all feature extractors on a single EEG window."""

import numpy as np
import torch

from config import SELECTED_CHANNELS, SAMPLE_RATE, WINDOW_SIZE
from data.dataset import CHBMITDataset
from features.stft import compute_stft, compute_correlation_map, extract_stft_features
from features.gnn_topology import compute_spectral_power, GNNTopology
from features.mfcc_cnn import compute_mfcc, MFCCCNN, SiameseMFCC
from features.salt_ltformer import SALT_LTformer, LSTMOnly, count_parameters


def main():
    print("Loading dataset (chb01)...")
    ds = CHBMITDataset(subjects=["chb01"], preload=True)
    print(f"Dataset: {len(ds)} windows")

    sample = ds.windows[0]
    data = sample["data"]
    n_channels = data.shape[0]
    print(f"Sample shape: {data.shape}")

    # === STFT ===
    print("\n--- STFT Spectrograms ---")
    spec, freqs, times = compute_stft(data)
    print(f"Spectrogram shape: {spec.shape}")
    print(f"  Frequencies: {len(freqs)} ({freqs[0]:.1f} - {freqs[-1]:.1f} Hz)")
    print(f"  Time frames: {len(times)}")

    corr = compute_correlation_map(data)
    print(f"Correlation map shape: {corr.shape}")

    # === Spectral Power (GNN features) ===
    print("\n--- Spectral Power (GNN Node Features) ---")
    sp = compute_spectral_power(data)
    print(f"Spectral power shape: {sp.shape}")
    print(f"  Bands: delta/theta/alpha/beta/gamma")

    # === MFCC ===
    print("\n--- MFCC Features ---")
    mfcc = compute_mfcc(data)
    print(f"MFCC shape: {mfcc.shape}")

    # === Models ===
    print("\n--- Model Parameter Counts ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ltformer = SALT_LTformer(n_channels=n_channels)
    print(f"SALT_LTformer params: {count_parameters(ltformer):,}")

    lstm_only = LSTMOnly(n_channels=n_channels)
    print(f"LSTMOnly params: {count_parameters(lstm_only):,}")

    gnn = GNNTopology(n_nodes=n_channels)
    print(f"GNNTopology params: {count_parameters(gnn):,}")

    mfcc_cnn = MFCCCNN(n_channels=n_channels)
    print(f"MFCCCNN params: {count_parameters(mfcc_cnn):,}")

    siamese = SiameseMFCC(n_channels=n_channels)
    print(f"SiameseMFCC params: {count_parameters(siamese):,}")

    # === Forward Pass Tests ===
    print("\n--- Forward Pass Tests ---")
    x = torch.from_numpy(data).unsqueeze(0).float()

    # SALT LTformer
    with torch.no_grad():
        out = ltformer(x)
        print(f"SALT_LTformer output shape: {out.shape}")

        out2 = lstm_only(x)
        print(f"LSTMOnly output shape: {out2.shape}")

        sp_tensor = torch.from_numpy(sp).unsqueeze(0).float()
        adj = torch.eye(n_channels).unsqueeze(0).float()
        gnn_out = gnn(sp_tensor, adj)
        print(f"GNN output shape: {gnn_out.shape}")

        mfcc_tensor = torch.from_numpy(mfcc).unsqueeze(0).float()
        mfcc_out = mfcc_cnn(mfcc_tensor)
        print(f"MFCCCNN output shape: {mfcc_out.shape}")

        siamese_out = siamese(mfcc_tensor, mfcc_tensor)
        print(f"Siamese output shape: {siamese_out.shape}")

    print("\nAll feature extractors working!")


if __name__ == "__main__":
    main()
