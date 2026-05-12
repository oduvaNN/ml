"""Tab 1: Dataset Exploration."""
import logging
from typing import Any, Dict

import streamlit as st

from utils.data_utils import (
    CIFAR10_CLASSES,
    denormalize_image,
    get_class_distribution,
    get_raw_test_dataset,
    load_datasets,
)
from utils.viz_utils import plot_class_distribution, plot_split_sizes

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner="Loading datasets…")
def _load(cfg_hash: str, cfg: Dict[str, Any]) -> tuple:
    train_ds, val_ds, test_ds = load_datasets(cfg)
    return train_ds, val_ds, test_ds


@st.cache_data(show_spinner="Computing class distribution…")
def _class_dist(cfg_hash: str, cfg: Dict[str, Any]) -> Dict[str, int]:
    _, _, test_ds = load_datasets(cfg)
    return get_class_distribution(test_ds)


def render(cfg: Dict[str, Any]) -> None:
    st.header("Dataset Exploration")

    try:
        train_ds, val_ds, test_ds = _load(str(cfg["data"]["dir"]), cfg)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        logger.error("Dataset load error: %s", exc)
        return

    # ── Overview ──────────────────────────────────────────────
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training samples", len(train_ds))
    col2.metric("Validation samples", len(val_ds))
    col3.metric("Test samples", len(test_ds))
    st.metric("Number of classes", cfg["data"]["num_classes"])

    sizes = {"Train": len(train_ds), "Validation": len(val_ds), "Test": len(test_ds)}
    st.plotly_chart(plot_split_sizes(sizes), width="stretch")

    # ── Class distribution ────────────────────────────────────
    st.subheader("Class Distribution (Test Set)")
    try:
        dist = _class_dist(str(cfg["data"]["dir"]), cfg)
        st.plotly_chart(plot_class_distribution(dist), width="stretch")
    except Exception as exc:
        st.warning(f"Could not compute class distribution: {exc}")

    # ── Sample Inspection ─────────────────────────────────────
    st.subheader("Sample Inspection")

    split_name = st.selectbox("Select split", ["Train", "Validation", "Test"])
    split_map = {"Train": train_ds, "Validation": val_ds, "Test": test_ds}
    selected_ds = split_map[split_name]

    filter_class = st.selectbox("Filter by class (optional)", ["All"] + CIFAR10_CLASSES)

    if filter_class != "All":
        class_idx = CIFAR10_CLASSES.index(filter_class)
        indices = [i for i in range(len(selected_ds)) if selected_ds[i][1] == class_idx]
    else:
        indices = list(range(len(selected_ds)))

    if not indices:
        st.warning("No samples found for this filter.")
        return

    sample_idx = st.slider("Sample index", 0, len(indices) - 1, 0)
    real_idx = indices[sample_idx]
    tensor, label = selected_ds[real_idx]

    img = denormalize_image(tensor)
    st.image(img, caption=f"Label: {CIFAR10_CLASSES[label]}", width=200)
    st.write(f"**Class index:** {label} | **Class name:** `{CIFAR10_CLASSES[label]}`")
    st.write(f"**Tensor shape:** {tuple(tensor.shape)} | **Dataset index:** {real_idx}")
