"""Tab 2: Model Error Analysis (MLflow Integration)."""
import logging
from typing import Any, Dict, List

import numpy as np
import streamlit as st
import torch

from utils.data_utils import CIFAR10_CLASSES, denormalize_image, get_raw_test_dataset, get_test_loader
from utils.mlflow_utils import (
    get_metric_history,
    get_run_metrics,
    list_experiments,
    list_runs,
    load_model_from_run,
)
from utils.model_utils import run_inference
from utils.viz_utils import plot_confusion_matrix, plot_per_class_errors

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading model from MLflow…")
def _load_model(run_id: str) -> Any:
    return load_model_from_run(run_id)


@st.cache_data(show_spinner="Running inference on test set…")
def _run_inference(run_id: str, data_dir: str, _model: Any) -> tuple:
    from utils.data_utils import get_test_loader
    cfg = {"data": {"dir": data_dir, "seed": 42}}
    loader = get_test_loader(cfg)
    preds, targets, confs = run_inference(_model, loader)
    return preds, targets, confs


def render(cfg: Dict[str, Any]) -> None:
    st.header("Model Error Analysis")

    tracking_uri = cfg["mlflow"]["tracking_uri"]

    # ── Experiment & Run Selection ────────────────────────────
    st.subheader("MLflow Run Selection")
    try:
        experiments = list_experiments(tracking_uri)
    except Exception as exc:
        st.error(f"Cannot connect to MLflow at {tracking_uri}. Start the tracking server first.\n\n{exc}")
        logger.error("MLflow connection error: %s", exc)
        return

    if not experiments:
        st.warning("No MLflow experiments found.")
        return

    exp_names = [e.name for e in experiments]
    selected_exp_name = st.selectbox("Experiment", exp_names)
    selected_exp = next(e for e in experiments if e.name == selected_exp_name)

    try:
        runs = list_runs(selected_exp.experiment_id)
    except Exception as exc:
        st.error(f"Failed to list runs: {exc}")
        return

    if not runs:
        st.warning("No runs found in this experiment.")
        return

    run_options = {r.info.run_name or r.info.run_id: r.info.run_id for r in runs}
    selected_run_name = st.selectbox("Run", list(run_options.keys()))
    selected_run_id = run_options[selected_run_name]

    metrics = get_run_metrics(selected_run_id)
    if metrics:
        st.write("**Run summary metrics:**")
        st.json({k: round(v, 4) for k, v in metrics.items()})

    # ── Load Model & Run Inference ────────────────────────────
    if st.button("Load model and run inference"):
        st.session_state["error_run_id"] = selected_run_id
        st.session_state["error_data_dir"] = cfg["data"]["dir"]

    if "error_run_id" not in st.session_state:
        st.info("Select a run and click **Load model and run inference**.")
        return

    try:
        model = _load_model(st.session_state["error_run_id"])
    except Exception as exc:
        st.error(f"Failed to load model artifact: {exc}")
        logger.error("Model load failed: %s", exc)
        return

    try:
        preds, targets, confs = _run_inference(
            st.session_state["error_run_id"],
            st.session_state["error_data_dir"],
            model,
        )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    accuracy = sum(p == t for p, t in zip(preds, targets)) / len(targets)
    st.metric("Test accuracy", f"{accuracy:.4f}")

    # ── Visualizations ────────────────────────────────────────
    st.subheader("Confusion Matrix")
    st.plotly_chart(
        plot_confusion_matrix(targets, preds, CIFAR10_CLASSES), width="stretch"
    )

    st.subheader("Per-class Error Counts")
    st.plotly_chart(
        plot_per_class_errors(targets, preds, CIFAR10_CLASSES),
        width="stretch",
    )

    # ── Misclassified Examples ────────────────────────────────
    st.subheader("Misclassified Examples")

    wrong_idx = [i for i, (p, t) in enumerate(zip(preds, targets)) if p != t]
    st.write(f"**Total misclassified:** {len(wrong_idx)} / {len(targets)}")

    # Sort by confidence (highest confidence wrong predictions first)
    wrong_idx_sorted = sorted(wrong_idx, key=lambda i: confs[i], reverse=True)

    filter_class = st.selectbox(
        "Filter errors by true class", ["All"] + CIFAR10_CLASSES, key="err_filter"
    )
    if filter_class != "All":
        cls_idx = CIFAR10_CLASSES.index(filter_class)
        wrong_idx_sorted = [i for i in wrong_idx_sorted if targets[i] == cls_idx]

    n_show = st.slider("Number of examples to show", 1, min(20, len(wrong_idx_sorted)), 6)

    raw_test_ds = get_raw_test_dataset(cfg)

    cols = st.columns(3)
    for k, idx in enumerate(wrong_idx_sorted[:n_show]):
        raw_tensor, _ = raw_test_ds[idx]
        img = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        col = cols[k % 3]
        col.image(img, width=120)
        col.caption(
            f"True: **{CIFAR10_CLASSES[targets[idx]]}**\n"
            f"Pred: **{CIFAR10_CLASSES[preds[idx]]}**\n"
            f"Conf: {confs[idx]:.3f}"
        )
