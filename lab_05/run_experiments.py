"""Run multiple W&B runs with structured naming and different hyperparameters."""
import copy
import logging

import yaml

from src.train import train_and_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Structured naming: "Run N - <description>"
CONFIGS = [
    {
        "run_name": "Run 1 - Default Config",
        "overrides": {},
    },
    {
        "run_name": "Run 2 - High LR (0.005)",
        "overrides": {"training": {"lr": 5e-3}},
    },
    {
        "run_name": "Run 3 - Large Batch (256)",
        "overrides": {"training": {"batch_size": 256}},
    },
    {
        "run_name": "Run 4 - High Dropout (0.5)",
        "overrides": {"model": {"dropout": 0.5}},
    },
    {
        "run_name": "Run 5 - Low Weight Decay (1e-5)",
        "overrides": {"training": {"weight_decay": 1e-5}},
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

    results = []
    for cfg in CONFIGS:
        params = apply_overrides(base_params, cfg["overrides"])
        logger.info("Starting: %s", cfg["run_name"])
        metrics = train_and_log(params=params, run_name=cfg["run_name"])
        results.append({"run_name": cfg["run_name"], **metrics})

    best = min(results, key=lambda r: r["best_val_loss"])
    logger.info("All runs complete.")
    logger.info(
        "Best run: %s | val_loss=%.4f | val_acc=%.4f",
        best["run_name"], best["best_val_loss"], best["final_val_acc"],
    )

    print("\nResults summary:")
    print(f"{'Run':<40} {'best_val_loss':>15} {'final_val_acc':>15}")
    print("-" * 72)
    for r in results:
        print(f"{r['run_name']:<40} {r['best_val_loss']:>15.4f} {r['final_val_acc']:>15.4f}")


if __name__ == "__main__":
    main()
