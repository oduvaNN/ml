"""Train model with MLflow tracking."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
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


def train_and_log(
    params: Dict[str, Any],
    run_name: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run one training session under an MLflow run, return final metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(params["artifacts"]["dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs: Dict[str, Any] = {"run_name": run_name}
    if parent_run_id:
        run_kwargs["parent_run_id"] = parent_run_id
        run_kwargs["nested"] = True
    if tags:
        run_kwargs["tags"] = tags

    with mlflow.start_run(**run_kwargs) as run:
        # log config file as artifact
        mlflow.log_artifact("config.yaml")

        # log all parameters
        mlflow.log_params(
            {
                "dataset": params["data"]["dataset"],
                "num_classes": params["data"]["num_classes"],
                "dropout": params["model"]["dropout"],
                "num_epochs": params["training"]["num_epochs"],
                "batch_size": params["training"]["batch_size"],
                "lr": params["training"]["lr"],
                "weight_decay": params["training"]["weight_decay"],
                "lr_step_size": params["training"]["lr_step_size"],
                "lr_gamma": params["training"]["lr_gamma"],
                "seed": params["training"]["seed"],
                "label_smoothing": params["training"]["label_smoothing"],
            }
        )

        train_ds, val_ds, _ = load_datasets(params)
        train_loader, val_loader, _ = create_loaders(
            train_ds, val_ds, val_ds, params["training"]["batch_size"]
        )

        model = build_model(params).to(device)
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=params["training"]["label_smoothing"]
        )
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
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_val_loss = float("inf")

        for epoch in range(1, params["training"]["num_epochs"] + 1):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = val_epoch(model, val_loader, loss_fn, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # log metrics every epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

            logger.info(
                "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch,
                params["training"]["num_epochs"],
                train_loss,
                val_loss,
                val_acc,
            )

        # summary metrics
        final_metrics = {
            "best_val_loss": best_val_loss,
            "final_val_acc": history["val_acc"][-1],
        }
        mlflow.log_metrics(final_metrics)

        # log model artifact
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="CIFAR10ResNet",
        )
        # also log the best checkpoint file
        mlflow.log_artifact(str(save_path), artifact_path="checkpoints")

        logger.info(
            "Run '%s' complete | best_val_loss=%.4f | run_id=%s",
            run_name,
            best_val_loss,
            run.info.run_id,
        )
        return {**final_metrics, "run_id": run.info.run_id}
