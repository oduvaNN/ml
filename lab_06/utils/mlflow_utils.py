"""MLflow interaction utilities."""
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)
LAB04_ROOT = Path(__file__).resolve().parents[2] / "lab_04"


@contextmanager
def _temporary_import_path(path: Path):
    path_str = str(path)
    added = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        added = True
    try:
        yield
    finally:
        if added and path_str in sys.path:
            sys.path.remove(path_str)


def connect(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)


def list_experiments(tracking_uri: str) -> List[mlflow.entities.Experiment]:
    connect(tracking_uri)
    experiments = mlflow.search_experiments()
    logger.info("Found %d experiments", len(experiments))
    return experiments


def list_runs(experiment_id: str) -> List[mlflow.entities.Run]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="list",
    )
    logger.info("Found %d runs in experiment %s", len(runs), experiment_id)
    return runs  # type: ignore[return-value]


def get_run_metrics(run_id: str) -> Dict[str, Any]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def load_model_from_run(run_id: str) -> Any:
    """Load pytorch model artifact from an MLflow run."""
    logger.info("Loading model from run %s", run_id)
    try:
        model_uri = f"runs:/{run_id}/model"
        # lab_04 logged a PyTorch model whose class lives in lab_04/src/model.py,
        # so we temporarily expose that project root during deserialization.
        with _temporary_import_path(LAB04_ROOT):
            model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        logger.info("Model loaded successfully from %s", model_uri)
        return model
    except Exception as exc:
        logger.error("Failed to load model from run %s: %s", run_id, exc)
        raise


def get_metric_history(run_id: str, metric_name: str) -> List[Tuple[int, float]]:
    """Return [(step, value), ...] for a metric across epochs."""
    client = mlflow.tracking.MlflowClient()
    try:
        history = client.get_metric_history(run_id, metric_name)
        return [(m.step, m.value) for m in history]
    except Exception as exc:
        logger.warning("Could not fetch metric '%s' for run %s: %s", metric_name, run_id, exc)
        return []
