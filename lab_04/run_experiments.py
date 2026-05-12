"""Run multiple MLflow experiments with different hyperparameter configurations.

Hierarchical structure:
  Experiment: CIFAR10-Classification
    Parent run: "Experiment 1 - Hyperparameter Search"
      Child run: "Experiment 1 - Stage 1: LR=0.001 BS=128"
      Child run: "Experiment 1 - Stage 2: LR=0.005 BS=128"
      Child run: "Experiment 1 - Stage 3: LR=0.001 BS=256"
"""
import copy
import logging

import mlflow
import yaml

from src.train import train_and_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "CIFAR10-Classification"

# Each entry overrides base config params for one child run
CONFIGS = [
    {
        "run_name": "Experiment 1 - Stage 1: LR=0.001 BS=128",
        "tags": {"stage": "baseline", "lr": "0.001", "batch_size": "128"},
        "overrides": {"training": {"lr": 1e-3, "batch_size": 128}},
    },
    {
        "run_name": "Experiment 1 - Stage 2: LR=0.005 BS=128",
        "tags": {"stage": "high_lr", "lr": "0.005", "batch_size": "128"},
        "overrides": {"training": {"lr": 5e-3, "batch_size": 128}},
    },
    {
        "run_name": "Experiment 1 - Stage 3: LR=0.001 BS=256",
        "tags": {"stage": "large_batch", "lr": "0.001", "batch_size": "256"},
        "overrides": {"training": {"lr": 1e-3, "batch_size": 256}},
    },
    {
        "run_name": "Experiment 1 - Stage 4: Dropout=0.5 BS=128",
        "tags": {"stage": "high_dropout", "dropout": "0.5", "batch_size": "128"},
        "overrides": {"model": {"dropout": 0.5}, "training": {"batch_size": 128}},
    },
]


def apply_overrides(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for section, values in overrides.items():
        result[section].update(values)
    return result


def main() -> None:
    with open("config.yaml") as f:
        base_params = yaml.safe_load(f)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Starting experiment: %s", EXPERIMENT_NAME)

    # parent run groups all child runs under one umbrella
    with mlflow.start_run(
        run_name="Experiment 1 - Hyperparameter Search",
        tags={"mlflow.note.content": "Grid search over LR and batch size"},
    ) as parent_run:
        parent_id = parent_run.info.run_id
        logger.info("Parent run id: %s", parent_id)

        results = []
        for cfg in CONFIGS:
            params = apply_overrides(base_params, cfg["overrides"])
            metrics = train_and_log(
                params=params,
                run_name=cfg["run_name"],
                parent_run_id=parent_id,
                tags=cfg["tags"],
            )
            results.append({"run_name": cfg["run_name"], **metrics})

        # log summary of child runs on the parent
        best = min(results, key=lambda r: r["best_val_loss"])
        mlflow.log_params({"best_child_run": best["run_name"]})
        mlflow.log_metrics(
            {
                "best_val_loss_overall": best["best_val_loss"],
                "best_final_val_acc_overall": best["final_val_acc"],
            }
        )

    logger.info("All runs complete.")
    logger.info("Best run: %s | val_loss=%.4f | val_acc=%.4f",
                best["run_name"], best["best_val_loss"], best["final_val_acc"])


if __name__ == "__main__":
    main()
