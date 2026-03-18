import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    num_epochs: int,
    device: torch.device,
    save_path: Path = Path("best_model.pth"),
) -> Tuple[Path, Dict[str, List[float]]]:
    """Training loop with per-epoch validation and best-model checkpointing."""
    model.to(device)
    best_val_loss: float = float("inf")
    best_model_path: Path = save_path
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    logger.info("Starting training for %d epochs ...", num_epochs)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_targets).item()
                preds = val_outputs.argmax(dim=1)
                correct += (preds == val_targets).sum().item()
                total += val_targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch [%2d/%d] | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f | lr: %.6f",
            epoch,
            num_epochs,
            avg_train_loss,
            avg_val_loss,
            val_acc,
            scheduler.get_last_lr()[0],
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info("  New best model saved (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)
    return best_model_path, history
