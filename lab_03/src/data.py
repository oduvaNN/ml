import logging
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


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


def load_datasets(params: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Load CIFAR-10 and split into train/val/test."""
    data_dir: str = params["data"]["dir"]
    seed: int = params["training"]["seed"]

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=get_transforms(is_train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=get_transforms(is_train=False)
    )

    val_size = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    logger.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    return train_dataset, val_dataset, test_dataset


def create_loaders(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader
