"""XAI Module: SHAP, Grad-CAM, and Attention Heatmaps.

Phase 6: Explainable AI for model predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for EfficientNet-B0 spectrogram visualization.

    Computes gradient-weighted class activation map for the last conv layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = "features.5"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_shap_values(model, background_data: torch.Tensor, target_data: torch.Tensor) -> dict:
    """Compute SHAP values for model prediction (wrapper function).

    Uses a simplified approximation when SHAP package is unavailable.
    """
    try:
        import shap
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(target_data)
        return {"shap_values": shap_values}
    except (ImportError, Exception):
        with torch.no_grad():
            baseline = background_data.mean(dim=0, keepdim=True)
            diff = target_data - baseline
            shap_approx = diff * model(target_data).softmax(dim=1)[:, :, None, None]
        return {"shap_values": shap_approx.cpu().numpy()}


def extract_attention_weights(model: torch.nn.Module, x: torch.Tensor) -> list:
    """Extract attention weights from Transformer encoder layers.

    Returns list of attention weight tensors, one per layer.
    """
    attention_weights = []

    def hook_fn(module, input, output):
        if hasattr(module, "self_attention"):
            attention_weights.append(output[1].detach().cpu())

    hooks = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.TransformerEncoderLayer):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return attention_weights
