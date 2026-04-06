import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.config import load_config, setup_logging
from src.data import create_loaders, download_cifar10, split_into_batches
from src.evaluate import test_model
from src.model import build_model
from src.train import train_model
from src.visualize import plot_experiment_comparison, plot_training_history

logger = logging.getLogger(__name__)


def run_experiment(
    experiment_name: str,
    cfg: dict,
    train_dataset,
    test_dataset,
    train_batches,
    test_batches,
    device: torch.device,
    artifact_dir: Path,
) -> Dict[str, float]:
    logger.info("=" * 55)
    logger.info("Starting experiment: %s", experiment_name)
    logger.info("=" * 55)

    train_loader, val_loader, test_loader = create_loaders(
        cfg, experiment_name, train_dataset, test_dataset, train_batches, test_batches
    )

    model = build_model(cfg).to(device)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg["training"]["lr_step_size"],
        gamma=cfg["training"]["lr_gamma"],
    )

    ckpt_path = artifact_dir / f"best_model_{experiment_name}.pth"
    best_path, history = train_model(
        model,
        train_loader,
        val_loader,
        train_loss_fn,
        optimizer,
        scheduler,
        num_epochs=cfg["training"]["num_epochs"],
        device=device,
        save_path=ckpt_path,
    )

    plot_training_history(
        history,
        experiment=experiment_name,
        save_path=str(artifact_dir / f"history_{experiment_name}.png"),
    )

    model.load_state_dict(torch.load(best_path, weights_only=True))
    metrics = test_model(model, test_loader, test_loss_fn, device)
    return metrics


def main() -> None:
    """Run all experiments and compare results."""
    setup_logging()
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = download_cifar10(cfg["data_dir"])

    train_batches = split_into_batches(len(train_dataset), cfg["num_batches"], cfg["seed"])
    test_batches = split_into_batches(len(test_dataset), cfg["num_batches"], cfg["seed"])

    all_metrics: Dict[str, Dict[str, float]] = {}
    for experiment_name in cfg["experiments"]:
        metrics = run_experiment(
            experiment_name,
            cfg,
            train_dataset,
            test_dataset,
            train_batches,
            test_batches,
            device,
            artifact_dir,
        )
        all_metrics[experiment_name] = metrics

    plot_experiment_comparison(
        all_metrics, save_path=str(artifact_dir / "experiment_comparison.png")
    )

    metrics_path = artifact_dir / "all_metrics.txt"
    with open(metrics_path, "w") as f:
        for exp, metrics in all_metrics.items():
            f.write(f"\n[{exp}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
    logger.info("All metrics saved to %s", metrics_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
