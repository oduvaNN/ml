"""Tab 3: Prediction & Explainability (Grad-CAM)."""
import logging
from io import BytesIO
from typing import Any, Dict

import numpy as np
import streamlit as st
import torch
from PIL import Image

from utils.data_utils import CIFAR10_CLASSES, denormalize_image, load_datasets
from utils.mlflow_utils import list_experiments, list_runs, load_model_from_run
from utils.model_utils import GradCAM, predict_single, preprocess_uploaded_image
from utils.viz_utils import overlay_gradcam, plot_probability_bar

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading model from MLflow…")
def _load_model(run_id: str) -> Any:
    return load_model_from_run(run_id)


@st.cache_data(show_spinner="Loading test dataset…")
def _load_test(data_dir: str, seed: int) -> Any:
    cfg = {"data": {"dir": data_dir, "seed": seed, "num_classes": 10}}
    from utils.data_utils import get_transforms
    from torchvision import datasets
    return datasets.CIFAR10(
        root=data_dir, train=False, download=False,
        transform=get_transforms(is_train=False),
    )


@st.cache_data(show_spinner="Loading raw test images…")
def _load_raw_test(data_dir: str) -> Any:
    from torchvision import datasets, transforms
    return datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transforms.ToTensor()
    )


def _run_gradcam(
    model: Any,
    tensor: torch.Tensor,
    target_layer: str,
    class_idx: int | None,
    raw_image: np.ndarray,
) -> np.ndarray:
    cam_model = GradCAM(model, target_layer)
    heatmap = cam_model.generate(tensor, class_idx=class_idx)
    return overlay_gradcam(raw_image, heatmap)


def render(cfg: Dict[str, Any]) -> None:
    st.header("Prediction & Explainability")

    tracking_uri = cfg["mlflow"]["tracking_uri"]

    # ── Run Selection ─────────────────────────────────────────
    st.subheader("Select MLflow Run")
    try:
        experiments = list_experiments(tracking_uri)
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        return

    if not experiments:
        st.warning("No experiments found.")
        return

    exp_names = [e.name for e in experiments]
    selected_exp_name = st.selectbox("Experiment", exp_names, key="exp_tab3")
    selected_exp = next(e for e in experiments if e.name == selected_exp_name)

    try:
        runs = list_runs(selected_exp.experiment_id)
    except Exception as exc:
        st.error(f"Failed to list runs: {exc}")
        return

    if not runs:
        st.warning("No runs found.")
        return

    run_options = {r.info.run_name or r.info.run_id: r.info.run_id for r in runs}
    selected_run_name = st.selectbox("Run", list(run_options.keys()), key="run_tab3")
    selected_run_id = run_options[selected_run_name]

    try:
        model = _load_model(selected_run_id)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        logger.error("Model load failed: %s", exc)
        return

    target_layer = cfg["gradcam"]["target_layer"]

    # ── Input Source ──────────────────────────────────────────
    st.subheader("Input")
    source = st.radio("Image source", ["Dataset sample", "Upload image"])

    if source == "Dataset sample":
        test_ds = _load_test(cfg["data"]["dir"], cfg["data"]["seed"])
        raw_test_ds = _load_raw_test(cfg["data"]["dir"])

        idx = st.slider("Test sample index", 0, len(test_ds) - 1, 0)
        tensor, true_label = test_ds[idx]
        raw_tensor, _ = raw_test_ds[idx]
        raw_img = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        st.image(raw_img, caption=f"True label: {CIFAR10_CLASSES[true_label]}", width=180)

    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.info("Upload an image to run inference.")
            return
        try:
            pil_img = Image.open(BytesIO(uploaded.read()))
        except Exception as exc:
            st.error(f"Invalid image file: {exc}")
            return
        tensor = preprocess_uploaded_image(pil_img)
        raw_img = np.array(pil_img.convert("RGB").resize((32, 32)))
        true_label = None
        st.image(raw_img, caption="Uploaded image (resized to 32×32)", width=180)

    # ── Inference ─────────────────────────────────────────────
    st.subheader("Inference Result")
    try:
        pred_idx, probs = predict_single(model, tensor)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        logger.error("Inference error: %s", exc)
        return

    col1, col2 = st.columns(2)
    col1.metric("Predicted class", CIFAR10_CLASSES[pred_idx])
    col1.metric("Confidence", f"{probs[pred_idx]:.4f}")
    if true_label is not None:
        col2.metric("True class", CIFAR10_CLASSES[true_label])
        col2.metric("Correct", "✓" if pred_idx == true_label else "✗")

    st.plotly_chart(plot_probability_bar(probs, CIFAR10_CLASSES), width="stretch")

    # ── Grad-CAM ──────────────────────────────────────────────
    st.subheader("Grad-CAM Explainability")

    # optional: let user choose which class to explain
    explain_class_name = st.selectbox(
        "Class to explain (default = top prediction)",
        ["Top prediction"] + CIFAR10_CLASSES,
    )
    explain_class_idx = (
        None if explain_class_name == "Top prediction"
        else CIFAR10_CLASSES.index(explain_class_name)
    )

    try:
        overlaid = _run_gradcam(model, tensor, target_layer, explain_class_idx, raw_img)
    except Exception as exc:
        st.error(f"Grad-CAM failed: {exc}")
        logger.error("Grad-CAM error: %s", exc)
        return

    col_a, col_b = st.columns(2)
    col_a.image(raw_img, caption="Original image", width=200)
    col_b.image(overlaid, caption="Grad-CAM overlay", width=200)

    explained_cls = CIFAR10_CLASSES[explain_class_idx] if explain_class_idx is not None else CIFAR10_CLASSES[pred_idx]
    st.caption(
        f"Grad-CAM highlights the regions the model attends to when predicting **{explained_cls}**. "
        "Red/warm areas have the highest activation; blue/cool areas contribute least."
    )
