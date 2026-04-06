import logging
from typing import Any, Dict

import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class CIFAR10ResNet(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32×32 input, 10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()  # type: ignore[assignment]
        backbone.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
        self.net = backbone

    def forward(self, x: Any) -> Any:
        return self.net(x)


def build_model(cfg: Dict[str, Any]) -> CIFAR10ResNet:
    """Instantiate model from config."""
    num_classes: int = cfg["model"]["num_classes"]
    model = CIFAR10ResNet(num_classes=num_classes)
    logger.info("Model: CIFAR10ResNet with %d output classes", num_classes)
    return model
