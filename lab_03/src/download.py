"""Stage 1: Download CIFAR-10 dataset."""
import logging
import sys
from pathlib import Path

import yaml
from torchvision import datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    data_dir: str = params["data"]["dir"]
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CIFAR-10 to '%s'", data_dir)
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    logger.info("Download stage complete")


if __name__ == "__main__":
    main()
