"""Model loading and inference utilities."""
import logging
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_NORM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
_PREPROCESS = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), _NORM])


def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> Tuple[List[int], List[int], List[float]]:
    """Return (all_preds, all_targets, all_confidences)."""
    logger.info("Running inference on %d batches", len(loader))
    model.eval()
    model.to(DEVICE)
    all_preds, all_targets, all_confs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            all_confs.extend(confs.cpu().tolist())
    logger.info("Inference complete: %d samples", len(all_preds))
    return all_preds, all_targets, all_confs


def predict_single(model: nn.Module, tensor: torch.Tensor) -> Tuple[int, np.ndarray]:
    """Return (predicted_class_idx, probability_array)."""
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs


def preprocess_uploaded_image(image: Image.Image) -> torch.Tensor:
    """Prepare a PIL image for model inference."""
    return _PREPROCESS(image.convert("RGB"))


# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────

class GradCAM:
    """Minimal Grad-CAM for any nn.Module, targeting a named layer."""

    def __init__(self, model: nn.Module, target_layer_name: str) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # resolve the target layer by dotted name
        layer = model
        for part in target_layer_name.split("."):
            layer = getattr(layer, part)

        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m: Any, _i: Any, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _m: Any, _i: Any, grad_output: Any) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return heatmap (H×W float32, values 0–1)."""
        self.model.eval()
        self.model.to(DEVICE)
        input_t = tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

        logits = self.model(input_t)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # type: ignore[union-attr]
        cam = (weights * self._activations).sum(dim=1).squeeze()   # type: ignore[union-attr]
        cam = torch.clamp(cam, min=0)
        cam = cam.cpu().numpy()

        # resize to 32×32
        cam = np.array(
            Image.fromarray(cam).resize((32, 32), Image.BILINEAR)  # type: ignore[arg-type]
        )
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)
