"""Visualization utilities (matplotlib / plotly)."""
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_class_distribution(counts: Dict[str, int]) -> go.Figure:
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        labels={"x": "Class", "y": "Count"},
        title="Class Distribution",
        color=list(counts.keys()),
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_split_sizes(sizes: Dict[str, int]) -> go.Figure:
    fig = px.pie(
        names=list(sizes.keys()),
        values=list(sizes.values()),
        title="Train / Val / Test Split",
    )
    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    if class_names is None:
        class_names = CIFAR10_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "True"},
        title="Confusion Matrix",
    )
    fig.update_layout(width=700, height=600)
    return fig


def plot_per_class_errors(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    if class_names is None:
        class_names = CIFAR10_CLASSES
    errors = [0] * len(class_names)
    for t, p in zip(y_true, y_pred):
        if t != p:
            errors[t] += 1
    fig = px.bar(
        x=class_names,
        y=errors,
        labels={"x": "Class", "y": "Error count"},
        title="Per-class Error Counts",
        color=class_names,
    )
    fig.update_layout(showlegend=False)
    return fig


def overlay_gradcam(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap (H×W float) on RGB image (H×W×3 uint8)."""
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap)[..., :3]  # drop alpha
    heatmap_uint8 = (heatmap_color * 255).astype(np.uint8)
    blended = (0.55 * image_rgb + 0.45 * heatmap_uint8).astype(np.uint8)
    return blended


def plot_probability_bar(probs: np.ndarray, class_names: Optional[List[str]] = None) -> go.Figure:
    if class_names is None:
        class_names = CIFAR10_CLASSES
    fig = px.bar(
        x=class_names,
        y=probs.tolist(),
        labels={"x": "Class", "y": "Probability"},
        title="Prediction Probability Distribution",
        color=class_names,
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    return fig
