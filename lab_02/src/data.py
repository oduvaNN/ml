import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_transforms(is_train: bool) -> transforms.Compose:
    """Return appropriate transforms for train or eval split."""
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


def download_cifar10(data_dir: str) -> tuple:
    """Download CIFAR-10 and return full train and test datasets."""
    logger.info("Loading CIFAR-10 dataset from '%s'", data_dir)
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_transforms(is_train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(is_train=False)
    )
    logger.info(
        "CIFAR-10 loaded: %d train samples, %d test samples",
        len(train_dataset),
        len(test_dataset),
    )
    return train_dataset, test_dataset


def split_into_batches(dataset_size: int, num_batches: int, seed: int) -> List[List[int]]:
    """Split dataset indices into equal-sized batches."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    batch_size = dataset_size // num_batches
    batches = [indices[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    logger.info(
        "Split %d samples into %d batches of ~%d each", dataset_size, num_batches, batch_size
    )
    return batches


def build_loader(
    dataset: Any,
    batches: List[List[int]],
    selected_batch_ids: List[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader from selected batch indices."""
    indices: List[int] = []
    for bid in selected_batch_ids:
        indices.extend(batches[bid])
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def create_loaders(
    cfg: Dict[str, Any],
    experiment_name: str,
    train_dataset: Any,
    test_dataset: Any,
    train_batches: List[List[int]],
    test_batches: List[List[int]],
) -> tuple:
    """Return (train_loader, val_loader, test_loader) for a given experiment config."""
    exp_cfg = cfg["experiments"][experiment_name]
    train_batch_ids: List[int] = exp_cfg["train_batches"]
    val_batch_ids: List[int] = exp_cfg["val_batches"]
    test_batch_ids: List[int] = cfg["test_batches"]
    batch_size: int = cfg["training"]["batch_size"]

    train_loader = build_loader(train_dataset, train_batches, train_batch_ids, batch_size, shuffle=True)
    val_loader = build_loader(train_dataset, train_batches, val_batch_ids, batch_size, shuffle=False)
    test_loader = build_loader(test_dataset, test_batches, test_batch_ids, batch_size, shuffle=False)

    logger.info(
        "Experiment '%s': train batches=%s (%d samples), val batches=%s (%d samples), test batches=%s (%d samples)",
        experiment_name,
        train_batch_ids,
        len(train_loader.dataset),  # type: ignore[arg-type]
        val_batch_ids,
        len(val_loader.dataset),  # type: ignore[arg-type]
        test_batch_ids,
        len(test_loader.dataset),  # type: ignore[arg-type]
    )
    return train_loader, val_loader, test_loader
