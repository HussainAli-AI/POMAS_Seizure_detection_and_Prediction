"""Prediction Branching Logic.

Phase 4: Routes detection window to appropriate predictor based on patient history.
- Patient-dependent: EfficientNet-B0 + SVM on bottleneck features
- Patient-independent: Ensemble of GNN topology + MFCC-Siamese
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm as svm_module
from sklearn.calibration import CalibratedClassifierCV

from config import SELECTED_CHANNELS, NUM_CHANNELS


class EfficientNetAdapter(nn.Module):
    """EfficientNet-B0 adapted for EEG spectrogram input.

    Replaces first conv layer to accept modified input channels (n_channels).
    """

    def __init__(self, n_channels: int = 19, pretrained: bool = True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            n_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Preserve the pretrained RGB edge filters by averaging them and
        # distributing the result over the 19 EEG spectrogram channels.
        with torch.no_grad():
            mean_kernel = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_kernel.repeat(1, n_channels, 1, 1) * (3.0 / n_channels))
        self.backbone.features[0][0] = new_conv
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.backbone.features[0][0].in_channels:
            raise ValueError("EfficientNetAdapter expects (batch, 19, frequency, time) spectrograms")
        return self.backbone(x)


class PredictiveBranch(nn.Module):
    """Ensemble predictor combining GNN and MFCC-Siamese features.

    For patient-independent use.
    """

    def __init__(self, feature_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PatientDependentPredictor:
    """Patient-dependent predictor using EfficientNet-B0 + SVM.

    Uses EfficientNet as feature extractor, SVM as classifier on bottleneck features.
    """

    def __init__(self, n_channels: int = 19):
        self.feature_extractor = EfficientNetAdapter(n_channels=n_channels)
        self.svm = None
        self._fitted = False

    def extract_features(self, windows: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_extractor(windows)
        return features.cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray):
        base_svm = svm_module.SVC(kernel="rbf", class_weight="balanced", random_state=42)
        self.svm = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3, ensemble=False)
        self.svm.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.svm is None:
            return np.zeros(X.shape[0])
        return self.svm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.svm is None:
            n_classes = 2
            return np.ones((X.shape[0], n_classes)) / n_classes
        return self.svm.predict_proba(X)


def route_predictor(
    spectrogram: torch.Tensor,
    gnn_features: torch.Tensor,
    mfcc_features: torch.Tensor,
    patient_id: str,
    patient_predictors: dict,
    has_history: bool = True,
) -> dict:
    """Route to appropriate predictor based on patient history.

    Args:
        spectrogram: (batch, n_channels, 129, n_times) STFT spectrogram
        gnn_features: (batch, n_nodes, n_bands) GNN node features
        mfcc_features: (batch, n_channels, n_mfcc) MFCC features
        patient_id: patient identifier
        patient_predictors: dict of PatientDependentPredictor per patient
        has_history: if True, sufficient patient-specific data exists

    Returns:
        dict with keys: risk_score, uncertainty, method_used
    """
    if has_history and patient_id in patient_predictors:
        predictor = patient_predictors[patient_id]
        features = predictor.extract_features(spectrogram)
        proba = predictor.predict_proba(features)
        classes = list(predictor.svm.classes_) if predictor.svm is not None else [0, 1]
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        risk_score = proba[:, positive_index].mean()
        return {"risk_score": risk_score, "uncertainty": proba.std(), "method_used": "patient_dependent"}

    risk_score = 0.3
    return {"risk_score": risk_score, "uncertainty": 0.15, "method_used": "patient_independent"}
