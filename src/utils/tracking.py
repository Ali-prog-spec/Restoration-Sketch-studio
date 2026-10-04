"""Thin MLflow wrapper.

Why MLflow (not W&B): it runs fully locally (SQLite backend, no account or API key),
so an evaluator can reproduce and browse every run offline with
``mlflow ui --backend-store-uri sqlite:///mlflow.db``.

If tracking is disabled (``tracking.enabled: false`` or ``--no-tracking``) every
method is a no-op, so training code never needs ``if tracking:`` branches.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .config import ROOT, flatten

DEFAULT_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACTS = (ROOT / "mlartifacts").as_uri()


class Tracker:
    def __init__(self, cfg: dict):
        tcfg = cfg.get("tracking", {})
        self.enabled = bool(tcfg.get("enabled", True))
        self.experiment = tcfg.get("experiment", "genai-assignment1")
        self._mlflow = None
        if self.enabled:
            import mlflow

            uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
            mlflow.set_tracking_uri(uri)
            if mlflow.get_experiment_by_name(self.experiment) is None:
                mlflow.create_experiment(self.experiment, artifact_location=DEFAULT_ARTIFACTS + "/" + self.experiment)
            mlflow.set_experiment(self.experiment)
            self._mlflow = mlflow

    @contextmanager
    def run(self, name: str, nested: bool = False, tags: dict | None = None):
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=name, nested=nested, tags=tags) as r:
            yield r

    def log_params(self, params: dict) -> None:
        if not self.enabled:
            return
        flat = {k: str(v)[:500] for k, v in flatten(params).items()}
        # MLflow limits batch size; log in chunks.
        items = list(flat.items())
        for i in range(0, len(items), 90):
            self._mlflow.log_params(dict(items[i : i + 90]))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, directory: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(directory).exists():
            self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def set_tags(self, tags: dict) -> None:
        if self.enabled:
            self._mlflow.set_tags(tags)
