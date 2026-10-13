"""Generate publication-ready figures for thesis from experimental records."""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.0

OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load predictions
npz_path = OUTPUT_DIR / "trimodal_chb01_test_predictions.npz"
trimodal_results_path = OUTPUT_DIR / "trimodal_chb01_prediction_results.json"
bimodal_results_path = OUTPUT_DIR / "efficientnet_chb01_prediction_results.json"
spectral_results_path = OUTPUT_DIR / "spectral_baseline_results.json"
lstm_results_path = OUTPUT_DIR / "baseline_chb01_results.json"

data = np.load(npz_path)
y_true = data["labels"]
y_scores = data["scores"]

tri_res = json.loads(trimodal_results_path.read_text(encoding="utf-8"))
bi_res = json.loads(bimodal_results_path.read_text(encoding="utf-8"))
spectral_res = json.loads(spectral_results_path.read_text(encoding="utf-8"))
lstm_res = json.loads(lstm_results_path.read_text(encoding="utf-8"))

optimal_threshold = float(tri_res.get("test_f1_optimal", {}).get("threshold", tri_res.get("operating_threshold", -0.9477)))

# -------------------------------------------------------------
# Figure 1: ROC Curve (Tri-Modal Feature Fusion SOTA)
# -------------------------------------------------------------
fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Find operating point on ROC
op_idx = np.argmin(np.abs(roc_thresholds - optimal_threshold))
op_fpr = fpr[op_idx]
op_tpr = tpr[op_idx]

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Tri-Modal PoMAS (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Chance Level (AUC = 0.500)")
ax.scatter([op_fpr], [op_tpr], color="#d62728", s=90, zorder=5, 
           label=f"F1-Optimal Operating Point\n(Sens={op_tpr*100:.1f}%, Spec={(1-op_fpr)*100:.1f}%)")
ax.fill_between(fpr, tpr, alpha=0.12, color="#1f77b4")

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
ax.set_title("Receiver Operating Characteristic (ROC)\nTri-Modal PoMAS Prediction (Patient chb01)", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#bdc3c7", fontsize=9.5)
ax.grid(True, linestyle=":", alpha=0.6, color="#bdc3c7")
plt.tight_layout()
fig_roc_path = FIGURES_DIR / "fig1_roc_curve_efficientnet_svm.png"
fig.savefig(fig_roc_path)
plt.close(fig)
print(f"Saved: {fig_roc_path}")

# -------------------------------------------------------------
# Figure 2: Precision-Recall Curve (Tri-Modal SOTA)
# -------------------------------------------------------------
precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recalls, precisions)
prevalence = np.mean(y_true)

# Find operating point on PR
pr_idx = np.argmin(np.abs(pr_thresholds - optimal_threshold))
op_rec = recalls[pr_idx]
op_prec = precisions[pr_idx]

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
ax.plot(recalls, precisions, color="#2ca02c", lw=2.5, label=f"Tri-Modal PoMAS (PR-AUC = {pr_auc:.3f})")
ax.axhline(prevalence, color="#7f8c8d", lw=1.5, linestyle="--", label=f"Baseline Prevalence ({prevalence*100:.1f}%)")
ax.scatter([op_rec], [op_prec], color="#d62728", s=90, zorder=5, 
           label=f"Selected Threshold ({optimal_threshold:.4f})\n(Precision={op_prec*100:.1f}%, Recall={op_rec*100:.1f}%)")
ax.fill_between(recalls, precisions, alpha=0.12, color="#2ca02c")

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
ax.set_title("Precision-Recall (PR) Curve\nTri-Modal PoMAS Prediction (Patient chb01)", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#bdc3c7", fontsize=9.5)
ax.grid(True, linestyle=":", alpha=0.6, color="#bdc3c7")
plt.tight_layout()
fig_pr_path = FIGURES_DIR / "fig2_pr_curve_efficientnet_svm.png"
fig.savefig(fig_pr_path)
plt.close(fig)
print(f"Saved: {fig_pr_path}")

# -------------------------------------------------------------
# Figure 3: Confusion Matrix Heatmap (Tri-Modal SOTA)
# -------------------------------------------------------------
y_pred = (y_scores >= optimal_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=300)
cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)

labels_text = [
    [f"TN\n{tn:,}\n({cm_norm[0,0]*100:.1f}%)", f"FP\n{fp:,}\n({cm_norm[0,1]*100:.1f}%)"],
    [f"FN\n{fn:,}\n({cm_norm[1,0]*100:.1f}%)", f"TP\n{tp:,}\n({cm_norm[1,1]*100:.1f}%)"],
]

for i in range(2):
    for j in range(2):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "#2c3e50"
        ax.text(j, i, labels_text[i][j], ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted Interictal (0)", "Predicted Preictal (1)"], fontsize=10, fontweight="bold")
ax.set_yticklabels(["Actual Interictal (0)", "Actual Preictal (1)"], fontsize=10, fontweight="bold")
ax.set_title("Test Confusion Matrix (chb01)\nTri-Modal PoMAS: 1,546 Dims (N = 1,960 Windows)", fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
fig_cm_path = FIGURES_DIR / "fig3_confusion_matrix_efficientnet_svm.png"
fig.savefig(fig_cm_path)
plt.close(fig)
print(f"Saved: {fig_cm_path}")

# -------------------------------------------------------------
# Figure 4: Model Performance Comparison Bar Chart (4-Model Progression)
# -------------------------------------------------------------
models = ["Temporal LSTM", "Spectral Baseline", "Bi-Modal PoMAS", "Tri-Modal PoMAS (SOTA)"]
tri_opt = tri_res.get("test_f1_optimal", tri_res.get("test_metrics", {}))

sensitivity = [
    lstm_res.get("sensitivity", 0.0) * 100,
    spectral_res.get("sensitivity", 0.0) * 100,
    bi_res["test"]["sensitivity"] * 100,
    tri_opt["sensitivity"] * 100,
]
precisions = [
    0.0,
    spectral_res.get("precision", 0.0) * 100,
    bi_res["test"]["precision"] * 100,
    tri_opt["precision"] * 100,
]
f1_scores = [
    0.0,
    spectral_res.get("f1", 0.0) * 100,
    bi_res["test"]["f1"] * 100,
    tri_opt["f1"] * 100,
]
roc_aucs = [
    lstm_res.get("roc_auc", 0.0) * 100,
    spectral_res.get("roc_auc", 0.0) * 100,
    bi_res["test"]["roc_auc"] * 100,
    tri_opt["roc_auc"] * 100,
]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=300)
rects1 = ax.bar(x - 1.5*width, sensitivity, width, label="Sensitivity (Recall)", color="#3498db")
rects2 = ax.bar(x - 0.5*width, precisions, width, label="Precision", color="#2ecc71")
rects3 = ax.bar(x + 0.5*width, f1_scores, width, label="F1-Score", color="#e74c3c")
rects4 = ax.bar(x + 1.5*width, roc_aucs, width, label="ROC-AUC", color="#9b59b6")

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax.set_title("Benchmarked Model Progression on chb01 Test Recordings", fontsize=12, fontweight="bold", pad=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5, fontweight="bold")
ax.set_ylim([0, 115])
ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#bdc3c7", fontsize=9.5)
ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bdc3c7")
plt.tight_layout()
fig_comp_path = FIGURES_DIR / "fig4_model_comparison_barchart.png"
fig.savefig(fig_comp_path)
plt.close(fig)
print(f"Saved: {fig_comp_path}")

print("\nAll thesis figures successfully regenerated with Tri-Modal SOTA in output/figures/!")
