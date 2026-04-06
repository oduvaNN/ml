import logging
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set. Returns dict with loss, accuracy, precision, recall, F1."""
    model.eval()
    test_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += loss_function(outputs, targets).item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    metrics: Dict[str, float] = {
        "test_loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    logger.info("--- Test Results ---")
    for k, v in metrics.items():
        logger.info("  %-12s: %.4f", k, v)

    return metrics
