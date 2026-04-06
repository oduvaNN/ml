import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot train/val loss and validation accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — FlowerResNet-18 on Oxford 102 Flowers", fontsize=13)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val loss", markersize=4)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(epochs, [v * 100 for v in history["val_acc"]], "g-o", markersize=4)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to: %s", save_path)
    plt.close()


def plot_metrics_bar(metrics: Dict[str, float], save_path: Optional[str] = None) -> None:
    """Bar chart of test metrics."""
    keys = [k for k in metrics if k != "test_loss"]
    values = [metrics[k] for k in keys]
    colors = ["steelblue", "darkorange", "seagreen", "tomato"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(keys, values, color=colors[: len(keys)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Metrics")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to: %s", save_path)
    plt.close()
