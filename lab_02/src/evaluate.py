import logging
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set and return metrics dict."""
    model.eval()
    test_loss = 0.0
    all_preds: list = []
    all_targets: list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_fn(outputs, labels).item()
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    metrics: Dict[str, float] = {
        "test_loss": test_loss / len(test_loader),
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
        "f1_score": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }

    logger.info(
        "Test results: loss=%.4f | acc=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        metrics["test_loss"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    return metrics
