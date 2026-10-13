"""Generate Publication-Quality XAI Clinical Report Visualizations for Thesis.

Phase 5: Explainable AI & Clinical Decision Support.
Produces multi-panel clinical interpretability reports for detected preictal episodes:
- Panel A: 19-Channel Scalp EEG Waveform (Bipolar Montage)
- Panel B: Time-Frequency STFT Spectrogram (0.5 - 50 Hz)
- Panel C: Grad-CAM Frequency-Band Activation Map (Delta, Theta, Alpha, Beta, Gamma)
- Panel D: 19-Channel Feature Attribution & Hemispheric Localization
- Panel E: Structured Clinical Alert Decision Card (Tier, Lead Time, Action)
"""

from __future__ import annotations

import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from config import OUTPUT_DIR, SELECTED_CHANNELS, SAMPLE_RATE, STRIDE_SECONDS
from models.prediction_branch import EfficientNetAdapter
from train_efficientnet_svm import stft_spectrogram

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class GradCAMExplainer:
    """Grad-CAM for EfficientNet-B0 on EEG Spectrograms."""

    def __init__(self, model: nn.Module, target_layer_name: str = "features.8"):
        self.model = model
        self.gradients = None
        self.activations = None

        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name.endswith(target_layer_name):
                target_layer = module
                break

        if target_layer is None:
            # Fallback to last conv
            target_layer = self.model.backbone.features[8]

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, spectrogram: torch.Tensor, classifier: nn.Module) -> tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM activation heatmap and channel importance.
        
        spectrogram: (1, 19, 129, 37)
        """
        spectrogram = spectrogram.clone().detach().requires_grad_(True)
        self.model.eval()
        classifier.eval()

        # Forward pass
        deep_features = self.model(spectrogram)
        logit = classifier(deep_features).squeeze()

        # Backward pass for preictal score
        self.model.zero_grad()
        classifier.zero_grad()
        logit.backward(retain_graph=True)

        # Grad-CAM on target convolutional feature map
        # gradients shape: (1, C, H, W), activations shape: (1, C, H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate CAM to spectrogram dimensions (129 freq bins, 37 time steps)
        cam_resized = F.interpolate(cam, size=(129, 37), mode="bilinear", align_corners=False)
        cam_map = cam_resized.squeeze().cpu().numpy()
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

        # Channel attribution via input gradient sensitivity
        channel_grads = spectrogram.grad.abs().mean(dim=(2, 3)).squeeze().cpu().numpy()
        channel_importance = channel_grads / (channel_grads.sum() + 1e-8)

        return cam_map, channel_importance


def generate_clinical_report(
    episode_id: int,
    recording_id: str,
    window_idx: int,
    raw_eeg: np.ndarray,          # (19, 2560)
    risk_score: float,
    optimal_threshold: float,
    lead_time_minutes: float,
    alert_tier: str,
    cam_map: np.ndarray,          # (129, 37)
    channel_importance: np.ndarray,# (19,)
    output_path: Path,
):
    """Render a comprehensive 5-panel publication-grade clinical interpretability report."""
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 10.5), dpi=300)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1.2, 0.9], width_ratios=[1.3, 1.0, 1.0],
                           hspace=0.38, wspace=0.32)

    # -------------------------------------------------------------
    # Panel A: 19-Channel Scalp EEG Montage (Top Left)
    # -------------------------------------------------------------
    ax_eeg = fig.add_subplot(gs[0, :2])
    time_axis = np.linspace(0, 10, raw_eeg.shape[1])
    n_display = min(8, len(SELECTED_CHANNELS))
    # Pick top contributing channels for clear display
    top_channel_indices = np.argsort(channel_importance)[::-1][:n_display]
    
    offsets = np.arange(n_display) * 120.0
    for idx, ch_i in enumerate(top_channel_indices):
        signal = raw_eeg[ch_i] - np.mean(raw_eeg[ch_i])
        ax_eeg.plot(time_axis, signal + offsets[idx], lw=0.9, color="#12355B" if idx > 0 else "#C44536",
                    label=SELECTED_CHANNELS[ch_i] if idx == 0 else None)
    
    ax_eeg.set_yticks(offsets)
    ax_eeg.set_yticklabels([SELECTED_CHANNELS[i] for i in top_channel_indices], fontsize=8, fontweight="bold")
    ax_eeg.set_xlabel("Time within Window (Seconds)", fontsize=9, fontweight="bold")
    ax_eeg.set_ylabel("Electrode Channel", fontsize=9, fontweight="bold")
    ax_eeg.set_title("Panel A: Multi-Channel Scalp EEG Waveform (Top Attributed Bipolar Sites)", fontsize=10.5, fontweight="bold", pad=8, loc="left")
    ax_eeg.grid(axis="x", linestyle=":", alpha=0.5)
    ax_eeg.set_xlim([0, 10])

    # -------------------------------------------------------------
    # Panel B: Decision & Alert Summary Card (Top Right)
    # -------------------------------------------------------------
    ax_card = fig.add_subplot(gs[0, 2])
    ax_card.axis("off")

    tier_color = "#C44536" if alert_tier == "clinician_notify" else "#E9B949"
    card_bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.03,rounding_size=0.06",
                             facecolor="#F8FAFC", edgecolor=tier_color, linewidth=2)
    ax_card.add_patch(card_bg)

    ax_card.text(0.08, 0.88, "PoMAS Decision Summary Card", fontsize=11, fontweight="bold", color="#12355B")
    ax_card.text(0.08, 0.77, f"Patient: chb01  |  Recording: {recording_id}", fontsize=8.5, color="#52606D")
    ax_card.text(0.08, 0.67, f"Episode: Preictal Episode {episode_id}", fontsize=8.5, fontweight="bold", color="#1F2933")
    
    ax_card.text(0.08, 0.54, f"Assessed Risk Score:", fontsize=8.5, color="#52606D")
    ax_card.text(0.60, 0.54, f"{risk_score:.3f}", fontsize=11, fontweight="bold", color=tier_color)
    
    ax_card.text(0.08, 0.42, f"Optimal Decision Thresh:", fontsize=8.5, color="#52606D")
    ax_card.text(0.60, 0.42, f"{optimal_threshold:.4f}", fontsize=8.5, color="#1F2933")

    ax_card.text(0.08, 0.30, f"Early Warning Lead Time:", fontsize=8.5, color="#52606D")
    ax_card.text(0.60, 0.30, f"{lead_time_minutes:.1f} minutes", fontsize=10, fontweight="bold", color="#2A9D8F")

    ax_card.text(0.08, 0.18, f"Alert Action Tier:", fontsize=8.5, color="#52606D")
    ax_card.text(0.60, 0.18, f"{alert_tier.upper()}", fontsize=9, fontweight="bold", color=tier_color)

    ax_card.text(0.08, 0.07, f"Clinical Directive: Immediate bedside evaluation; check patient safety.",
                 fontsize=7.5, fontstyle="italic", color="#52606D")

    # -------------------------------------------------------------
    # Panel C: STFT Spectrogram of Dominant Channel (Middle Left)
    # -------------------------------------------------------------
    ax_spec = fig.add_subplot(gs[1, 0])
    top_ch = top_channel_indices[0]
    top_signal = raw_eeg[top_ch]
    # Compute spectrogram
    spec_data = np.log1p(np.abs(np.fft.rfft(top_signal.reshape(-1, 64), axis=1))).T
    freqs = np.linspace(0.5, 50, 129)
    times = np.linspace(0, 10, 37)
    
    # Compute log-magnitude STFT
    w_tensor = torch.from_numpy(raw_eeg[None, ...]).float()
    stft_cube = stft_spectrogram(w_tensor, torch.device("cpu")).squeeze(0)[top_ch].numpy()
    
    im_spec = ax_spec.imshow(stft_cube, origin="lower", aspect="auto", cmap="viridis",
                             extent=[0, 10, 0.5, 50])
    ax_spec.set_title(f"Panel B: STFT Spectrogram ({SELECTED_CHANNELS[top_ch]})", fontsize=10, fontweight="bold", pad=8, loc="left")
    ax_spec.set_xlabel("Time (Seconds)", fontsize=9, fontweight="bold")
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold")
    cbar_spec = fig.colorbar(im_spec, ax=ax_spec, fraction=0.046, pad=0.04)
    cbar_spec.set_label("Log Magnitude (Z-norm)", fontsize=7.5)

    # -------------------------------------------------------------
    # Panel D: Grad-CAM Frequency Activation Heatmap (Middle Center/Right)
    # -------------------------------------------------------------
    ax_cam = fig.add_subplot(gs[1, 1:])
    im_cam = ax_cam.imshow(cam_map, origin="lower", aspect="auto", cmap="inferno",
                           extent=[0, 10, 0.5, 50])
    ax_cam.set_title("Panel C: Grad-CAM Class Activation Map (Frequency-Temporal Neural Attention)", fontsize=10, fontweight="bold", pad=8, loc="left")
    ax_cam.set_xlabel("Time within Window (Seconds)", fontsize=9, fontweight="bold")
    ax_cam.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold")
    cbar_cam = fig.colorbar(im_cam, ax=ax_cam, fraction=0.03, pad=0.02)
    cbar_cam.set_label("Grad-CAM Activation Intensity", fontsize=7.5)

    # Frequency band annotations
    ax_cam.axhline(4, color="#ffffff", linestyle="--", lw=0.8, alpha=0.7)
    ax_cam.axhline(8, color="#ffffff", linestyle="--", lw=0.8, alpha=0.7)
    ax_cam.axhline(13, color="#ffffff", linestyle="--", lw=0.8, alpha=0.7)
    ax_cam.axhline(30, color="#ffffff", linestyle="--", lw=0.8, alpha=0.7)
    ax_cam.text(10.15, 2, "Delta (0.5-4Hz)", fontsize=7.5, color="#52606D", va="center")
    ax_cam.text(10.15, 6, "Theta (4-8Hz)", fontsize=7.5, color="#52606D", va="center")
    ax_cam.text(10.15, 10.5, "Alpha (8-13Hz)", fontsize=7.5, color="#52606D", va="center")
    ax_cam.text(10.15, 21.5, "Beta (13-30Hz)", fontsize=7.5, color="#52606D", va="center")
    ax_cam.text(10.15, 40, "Gamma (30-50Hz)", fontsize=7.5, color="#52606D", va="center")

    # -------------------------------------------------------------
    # Panel E: 19-Channel Electrode Feature Attribution (Bottom)
    # -------------------------------------------------------------
    ax_attr = fig.add_subplot(gs[2, :])
    sorted_order = np.argsort(channel_importance)[::-1]
    sorted_names = [SELECTED_CHANNELS[i] for i in sorted_order]
    sorted_scores = channel_importance[sorted_order] * 100.0

    bar_colors = ["#C44536" if i < 3 else "#168AAD" if "FP1" in sorted_names[i] or "F7" in sorted_names[i] or "T7" in sorted_names[i] else "#CBD5E1" for i in range(len(sorted_names))]
    bars = ax_attr.bar(range(len(sorted_names)), sorted_scores, color=bar_colors, edgecolor="#12355B", lw=0.6)
    
    for bar in bars[:5]:
        h = bar.get_height()
        ax_attr.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax_attr.set_xticks(range(len(sorted_names)))
    ax_attr.set_xticklabels(sorted_names, rotation=35, ha="right", fontsize=8, fontweight="bold")
    ax_attr.set_ylabel("Contribution Share (%)", fontsize=9, fontweight="bold")
    ax_attr.set_ylim([0, max(sorted_scores) * 1.25])
    ax_attr.set_title("Panel D: 19-Channel Feature Attribution & Scalp Topography Contribution", fontsize=10, fontweight="bold", pad=8, loc="left")
    ax_attr.grid(axis="y", linestyle=":", alpha=0.5)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Clinical Report: {output_path}")


def run_xai_generation():
    """Load model, find detected preictal episodes, and generate reports."""
    h5_path = OUTPUT_DIR / "preprocessed_global_v2" / "chb01_preprocessed_global_v2.h5"
    pred_res_path = OUTPUT_DIR / "efficientnet_chb01_prediction_results.json"
    pred_npz_path = OUTPUT_DIR / "efficientnet_chb01_test_predictions.npz"
    weights_path = OUTPUT_DIR / "efficientnet_chb01_prediction.pt"

    pred_res = json.loads(pred_res_path.read_text(encoding="utf-8"))
    test_recs = pred_res.get("test_recordings", [])
    optimal_threshold = float(pred_res["test"]["threshold"])
    npz_data = np.load(pred_npz_path)
    test_scores = npz_data["scores"]

    with h5py.File(h5_path, "r") as file:
        all_labels = file["labels"][:].astype(np.int64)
        all_recs = np.asarray(file["recording_ids"][:], dtype=str)
        all_timestamps = file["timestamps"][:]
        all_windows = file["windows"]

        eligible = all_labels != 2
        test_indices = np.flatnonzero(eligible & np.isin(all_recs, test_recs))

        # Reconstruct the model
        device = torch.device("cpu")
        backbone = EfficientNetAdapter(n_channels=19, pretrained=False).to(device)
        classifier = nn.Linear(backbone.feature_dim, 1).to(device)

        checkpoint = torch.load(weights_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone"])
        classifier.load_state_dict(checkpoint["classifier"])
        backbone.eval()
        classifier.eval()

        explainer = GradCAMExplainer(backbone, target_layer_name="features.8")

        # Find target preictal alert windows
        # Episode 1 in chb01_20.edf / chb01_21.edf; Episode 2 in chb01_26.edf
        episodes_targets = [
            {"episode": 1, "rec": "chb01_20.edf", "target_label": 1, "lead_time": 16.4, "tier": "clinician_notify"},
            {"episode": 2, "rec": "chb01_26.edf", "target_label": 1, "lead_time": 19.3, "tier": "clinician_notify"},
        ]

        reports_summary = []

        for target in episodes_targets:
            ep_num = target["episode"]
            target_rec = target["rec"]

            # Find a representative preictal window with high risk score
            cand_indices = []
            for i, global_i in enumerate(test_indices):
                if all_recs[global_i] == target_rec and all_labels[global_i] == 1:
                    cand_indices.append((i, global_i, test_scores[i]))

            if not cand_indices:
                continue

            # Pick the top-scoring preictal window in this episode
            cand_indices.sort(key=lambda x: x[2], reverse=True)
            chosen_local_i, chosen_global_i, chosen_score = cand_indices[0]

            raw_window = all_windows[chosen_global_i]  # (19, 2560)
            window_tensor = torch.from_numpy(raw_window[None, ...]).float()
            spectrogram = stft_spectrogram(window_tensor, device)

            # Generate Grad-CAM and channel attribution
            cam_map, channel_importance = explainer.generate(spectrogram, classifier)

            out_png = FIGURES_DIR / f"fig{4 + ep_num}_xai_clinical_report_episode{ep_num}.png"
            generate_clinical_report(
                episode_id=ep_num,
                recording_id=target_rec,
                window_idx=chosen_local_i,
                raw_eeg=raw_window,
                risk_score=chosen_score,
                optimal_threshold=optimal_threshold,
                lead_time_minutes=target["lead_time"],
                alert_tier=target["tier"],
                cam_map=cam_map,
                channel_importance=channel_importance,
                output_path=out_png,
            )

            top_ch_idx = int(np.argmax(channel_importance))
            reports_summary.append({
                "episode": ep_num,
                "recording": target_rec,
                "risk_score": round(float(chosen_score), 4),
                "lead_time_minutes": target["lead_time"],
                "alert_tier": target["tier"],
                "top_channel": SELECTED_CHANNELS[top_ch_idx],
                "top_channel_contribution": round(float(channel_importance[top_ch_idx] * 100), 2),
                "figure": str(out_png),
            })

        summary_json = OUTPUT_DIR / "xai_clinical_report_summary.json"
        summary_json.write_text(json.dumps(reports_summary, indent=2), encoding="utf-8")
        print(f"\nAll XAI Clinical Reports successfully saved! Summary: {summary_json}")


if __name__ == "__main__":
    run_xai_generation()
