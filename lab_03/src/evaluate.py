"""Stage 3: Evaluate trained model on test set."""
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import create_loaders, load_datasets
from src.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation stage started | device: %s", device)

    artifact_dir = Path(params["artifacts"]["dir"])
    _, _, test_ds = load_datasets(params)
    _, _, test_loader = create_loaders(test_ds, test_ds, test_ds, params["training"]["batch_size"])

    model = build_model(params).to(device)
    ckpt = artifact_dir / params["artifacts"]["model_name"]
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    all_preds: list = []
    all_targets: list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_fn(outputs, labels).item()
            all_preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    metrics = {
        "test_loss": round(test_loss / len(test_loader), 4),
        "accuracy": round(accuracy_score(all_targets, all_preds), 4),
        "precision": round(precision_score(all_targets, all_preds, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(all_targets, all_preds, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(all_targets, all_preds, average="weighted", zero_division=0), 4),
    }

    metrics_path = artifact_dir / params["artifacts"]["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Test results: loss=%.4f | acc=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        metrics["test_loss"], metrics["accuracy"],
        metrics["precision"], metrics["recall"], metrics["f1_score"],
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        values = [metrics[m] for m in metric_names]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(metric_names, values, color=["steelblue", "seagreen", "tomato", "orchid"])
        ax.set_ylim(0, 1.05)
        ax.set_title("Test Metrics")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(artifact_dir / params["artifacts"]["metrics_plot"], dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not save metrics plot: %s", e)

    logger.info("Evaluation stage complete")


if __name__ == "__main__":
    main()
