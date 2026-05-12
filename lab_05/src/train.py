"""Train model with Weights & Biases tracking."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
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
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train_and_log(
    params: Dict[str, Any],
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one training session under a W&B run, return final metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(params["artifacts"]["dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=params["wandb"]["project"],
        entity=params["wandb"]["entity"],
        name=run_name,
        config={
            "dataset":         params["data"]["dataset"],
            "num_classes":     params["data"]["num_classes"],
            "dropout":         params["model"]["dropout"],
            "num_epochs":      params["training"]["num_epochs"],
            "batch_size":      params["training"]["batch_size"],
            "lr":              params["training"]["lr"],
            "weight_decay":    params["training"]["weight_decay"],
            "lr_step_size":    params["training"]["lr_step_size"],
            "lr_gamma":        params["training"]["lr_gamma"],
            "seed":            params["training"]["seed"],
            "label_smoothing": params["training"]["label_smoothing"],
        },
        reinit=True,
    )

    # log config file as artifact
    config_artifact = wandb.Artifact("config", type="config")
    config_artifact.add_file("config.yaml")
    run.log_artifact(config_artifact)

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

        # log per-epoch metrics to W&B
        wandb.log(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch, params["training"]["num_epochs"], train_loss, val_loss, val_acc,
        )

    # log summary metrics
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["final_val_acc"] = history["val_acc"][-1]

    # log best checkpoint as model artifact
    model_artifact = wandb.Artifact("best-model", type="model")
    model_artifact.add_file(str(save_path))
    run.log_artifact(model_artifact)

    logger.info("Run '%s' complete | best_val_loss=%.4f", run_name, best_val_loss)
    run_id = run.id
    run.finish()

    return {
        "best_val_loss": best_val_loss,
        "final_val_acc": history["val_acc"][-1],
        "run_id": run_id,
    }
