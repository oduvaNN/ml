import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

logger = logging.getLogger(__name__)


def download_and_extract(
    url: str,
    save_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Download a file and extract it if it is an archive (.zip/.tar.gz/.tgz)."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    file_path = save_path / filename

    if file_path.exists():
        logger.info("File '%s' already exists — skipping download.", filename)
        return _resolve_extracted_path(save_path, filename)

    logger.info("Downloading '%s' from '%s' ...", filename, url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(file_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    logger.info("Saved to '%s'", file_path)

    return _extract_if_archive(file_path, save_path)


def _extract_if_archive(file_path: Path, save_path: Path) -> str:
    """Extract archive and return the path to the extracted content."""
    name = file_path.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(save_path)
        file_path.unlink()
        return str(save_path)
    elif any(name.endswith(ext) for ext in (".tar.gz", ".tgz", ".gz", ".tar")):
        if name.endswith((".gz", ".tgz")):
            tf = tarfile.open(str(file_path), "r:gz")
        else:
            tf = tarfile.open(str(file_path), "r")
        with tf:
            tf.extractall(save_path)
        file_path.unlink()
        return str(save_path)
    return str(file_path)


def _resolve_extracted_path(save_path: Path, filename: str) -> str:
    """If archive was already extracted, return directory; otherwise return file path."""
    file_path = save_path / filename
    if file_path.exists():
        return str(file_path)
    return str(save_path)


def train_test_split(
    data: pd.DataFrame,
    test_size: Union[float, int] = 0.25,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into (train, test) using a random permutation."""
    if random_state is not None:
        np.random.seed(random_state)
    n = len(data)
    n_test = int(n * test_size) if isinstance(test_size, float) else test_size
    idx = np.random.permutation(n)
    return data.iloc[idx[n_test:]], data.iloc[idx[:n_test]]


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load Oxford 102 Flowers labels from a .mat file."""
    mat = sio.loadmat(labels_path)
    labels = mat["labels"].flatten().astype(int)
    df = pd.DataFrame({"label": labels})
    logger.info("Loaded %d labels, %d unique classes.", len(df), df["label"].nunique())
    return df


def find_add_images_to_labels(
    images_dir: str,
    labels: pd.DataFrame,
    image_ext: str = "jpg",
) -> pd.DataFrame:
    """Discover images and attach paths to the labels DataFrame."""
    paths = sorted(str(p.absolute()) for p in Path(images_dir).rglob(f"*.{image_ext}"))
    if len(paths) != len(labels):
        raise ValueError(
            f"Image/label count mismatch: {len(paths)} images vs {len(labels)} labels."
        )
    df = labels.copy(deep=True)
    df["image_path"] = paths
    return df[["image_path", "label"]]


def process_data(
    images_dir: str,
    labels_path: str,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full data-ingestion pipeline: load labels, attach images, split into train/val/test."""
    logger.info("Starting data ingestion ...")
    labels_df = load_labels(labels_path)
    full_df = find_add_images_to_labels(images_dir, labels_df)

    train_df, test_df = train_test_split(
        full_df,
        test_size=cfg.get("test_size", 0.15),
        random_state=cfg.get("random_state", 42),
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.get("val_size", 0.15),
        random_state=cfg.get("random_state", 42),
    )

    logger.info(
        "Data split — train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def build_transforms(cfg: Dict[str, Any], is_train: bool) -> transforms.Compose:
    """Build torchvision transforms with augmentation for training."""
    size = cfg.get("image_size", 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class ImageDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset for Oxford 102 Flowers images."""

    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[idx]
        image = read_image(row["image_path"]).float() / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = self.transform(image)
        label = int(row["label"]) - 1
        return image, label


def create_data_loader(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    is_train: bool = False,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader from a DataFrame."""
    tfm = build_transforms(cfg, is_train=is_train)
    dataset = ImageDataset(df, transform=tfm)
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=is_train,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )
