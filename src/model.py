import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class FlowerResNet(nn.Module):
    """ResNet-18 fine-tuned for 102-class flower classification.

    Improvements over baseline SimpleNN:
      - Deep residual backbone
      - ImageNet pre-trained weights
      - Batch normalisation throughout
      - Dropout before the final head
    """

    def __init__(self, n_classes: int = 102, pretrained: bool = True, dropout: float = 0.4) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, n_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the model from config."""
    n_classes = cfg.get("n_classes", 102)
    pretrained = cfg.get("pretrained", True)
    model = FlowerResNet(n_classes=n_classes, pretrained=pretrained)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: FlowerResNet-18 | Classes: %d | Pretrained: %s", n_classes, pretrained)
    logger.info("Parameters — total: %d | trainable: %d", total_params, trainable_params)
    return model
