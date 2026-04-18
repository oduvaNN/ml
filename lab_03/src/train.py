"""Stage 2: Train model and save checkpoint + metrics."""
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import StepLR

from src.data import create_loaders, load_datasets
from src.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training stage started | device: %s", device)

    artifact_dir = Path(params["artifacts"]["dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_datasets(params)
    train_loader, val_loader, _ = create_loaders(
        train_ds, val_ds, val_ds, params["training"]["batch_size"]
    )

    model = build_model(params).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=params["training"]["label_smoothing"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["training"]["lr"],
        weight_decay=params["training"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=params["training"]["lr_step_size"],
        gamma=params["training"]["lr_gamma"],
    )

    save_path = artifact_dir / params["artifacts"]["model_name"]
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, params["training"]["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch, params["training"]["num_epochs"], train_loss, val_loss, val_acc,
        )

    train_metrics = {
        "best_val_loss": best_val_loss,
        "final_val_acc": history["val_acc"][-1],
    }
    metrics_path = artifact_dir / params["artifacts"]["train_metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(epochs, history["val_acc"], color="green", label="Val Accuracy")
        ax2.set_title("Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(artifact_dir / params["artifacts"]["history_plot"], dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not save history plot: %s", e)

    logger.info("Training stage complete | best_val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
