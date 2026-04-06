import logging
from typing import Dict, List

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict[str, List[float]], experiment: str, save_path: str) -> None:
    """Plot train/val loss and val accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title(f"{experiment} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_title(f"{experiment} — Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Training history saved to %s", save_path)


def plot_experiment_comparison(
    all_metrics: Dict[str, Dict[str, float]], save_path: str
) -> None:
    """Bar chart comparing test metrics across experiments."""
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    experiments = list(all_metrics.keys())
    x = range(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, exp in enumerate(experiments):
        values = [all_metrics[exp][m] for m in metric_names]
        offset = (i - len(experiments) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=exp)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics by Experiment")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Experiment comparison saved to %s", save_path)
