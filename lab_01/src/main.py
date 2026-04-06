import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.config import load_config, setup_logging
from src.data import create_data_loader, download_and_extract, process_data
from src.evaluate import test_model
from src.model import build_model
from src.train import train_model
from src.visualize import plot_metrics_bar, plot_training_history

logger = logging.getLogger(__name__)


def main() -> None:
    """End-to-end training pipeline."""
    setup_logging()

    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("=" * 55)
    logger.info("ML Engineering Lab 1 — Training Pipeline")
    logger.info("=" * 55)

    download_and_extract(cfg["images_url"], cfg["data_dir"] + "/images")
    download_and_extract(cfg["labels_url"], cfg["data_dir"] + "/labels")
    labels_file = str(Path(cfg["data_dir"]) / "labels" / "imagelabels.mat")
    images_dir = cfg["data_dir"] + "/images"

    train_df, val_df, test_df = process_data(images_dir, labels_file, cfg)

    train_loader = create_data_loader(train_df, cfg, is_train=True)
    val_loader = create_data_loader(val_df, cfg, is_train=False)
    test_loader = create_data_loader(test_df, cfg, is_train=False)

    model = build_model(cfg).to(device)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 5e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg.get("lr_step_size", 5),
        gamma=cfg.get("lr_gamma", 0.5),
    )

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = artifact_dir / cfg["best_model_name"]

    best_model_path, history = train_model(
        model,
        train_loader,
        val_loader,
        train_loss_fn,
        optimizer,
        scheduler,
        num_epochs=cfg.get("num_epochs", 15),
        device=device,
        save_path=best_ckpt_path,
    )

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    metrics = test_model(model, test_loader, test_loss_fn, device)

    plot_training_history(history, save_path=str(artifact_dir / "training_history.png"))
    plot_metrics_bar(metrics, save_path=str(artifact_dir / "test_metrics.png"))

    metrics_path = artifact_dir / "metrics.txt"
    with open(metrics_path, "w") as fh:
        for k, v in metrics.items():
            fh.write(f"{k}: {v:.4f}\n")
    logger.info("Metrics saved to: %s", metrics_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
