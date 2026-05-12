"""Data loading and dataset utilities."""
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def load_datasets(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Return (train_dataset, val_dataset, test_dataset)."""
    data_dir: str = cfg["data"]["dir"]
    seed: int = cfg["data"]["seed"]
    logger.info("Loading CIFAR-10 from %s", data_dir)

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=get_transforms(is_train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=get_transforms(is_train=False)
    )

    val_size = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    logger.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )
    return train_dataset, val_dataset, test_dataset


def get_class_distribution(dataset: Any) -> Dict[str, int]:
    """Count samples per class in a dataset."""
    counts: Dict[str, int] = {cls: 0 for cls in CIFAR10_CLASSES}
    for _, label in dataset:
        counts[CIFAR10_CLASSES[label]] += 1
    return counts


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor → HWC uint8 numpy array for display."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def get_test_loader(cfg: Dict[str, Any], batch_size: int = 256) -> DataLoader:
    data_dir: str = cfg["data"]["dir"]
    test_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=get_transforms(is_train=False)
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)


def get_raw_test_dataset(cfg: Dict[str, Any]) -> Any:
    """Test dataset with NO normalization — for image display."""
    data_dir: str = cfg["data"]["dir"]
    return datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
