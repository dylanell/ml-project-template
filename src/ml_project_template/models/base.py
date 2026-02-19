"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
import inspect
import json
import os
import tempfile
import numpy as np

import mlflow

from ml_project_template.data import BaseDataset


class BaseModel(ABC):
    """Abstract base class for all models.

    Uses the template method pattern: train() handles MLflow orchestration
    and delegates to _fit() for model-specific training logic.

    Model params are captured automatically: any argument passed to __init__
    at any level of the inheritance chain is recorded in self._model_params.
    These are logged to MLflow automatically when train() is called.
    Subclasses do NOT need to manually build param dicts or override get_params().
    """

    # --- Automatic __init__ arg capture ---
    # When a subclass defines __init__, this hook wraps it so that all arguments
    # are recorded into self._model_params after the original __init__ runs.
    # This works across the full inheritance chain: BasePytorchModel.__init__
    # captures fabric args, then MLPClassifier.__init__ merges in architecture args.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only wrap classes that define their own __init__
        if '__init__' not in cls.__dict__:
            return

        original_init = cls.__dict__['__init__']

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kw):
            # Run the original __init__ (which calls super().__init__(), so
            # parent-level params get captured first)
            original_init(self, *args, **kw)

            # Bind the actual call args to the __init__ signature to get
            # a complete dict of param names -> values (including defaults)
            sig = inspect.signature(original_init)
            bound = sig.bind(self, *args, **kw)
            bound.apply_defaults()

            # Build a flat dict of params, unpacking **kwargs if present
            # (e.g. __init__(self, **kwargs) -> unpack kwargs into individual keys
            # rather than storing {"kwargs": {...}})
            params = {}
            for k, v in bound.arguments.items():
                if k == 'self':
                    continue
                if sig.parameters[k].kind == inspect.Parameter.VAR_KEYWORD:
                    params.update(v)
                else:
                    params[k] = v

            # Merge this level's params into _model_params (parent params
            # were already set by the parent's wrapped __init__)
            if not hasattr(self, '_model_params'):
                self._model_params = {}
            self._model_params.update(params)

        cls.__init__ = wrapped_init

    def get_params(self) -> dict:
        """Return model parameters for logging. Automatically populated from __init__ args."""
        return self._model_params

    def log_param(self, key: str, value) -> None:
        """Log a parameter to MLflow if tracking is enabled."""
        if self._tracking:
            mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric to MLflow if tracking is enabled."""
        if self._tracking:
            mlflow.log_metric(key, value, step=step)

    def train(
        self,
        *,
        experiment_name: str = "",
        train_data: BaseDataset,
        val_data: BaseDataset | None = None,
        run_name: str | None = None,
        model_path: str | None = None,
        extra_params: dict | None = None,
        tracking: bool = True,
        seed: int | None = None,
        **train_kwargs,
    ) -> None:
        """Full training pipeline with optional MLflow tracking.

        Args:
            experiment_name: MLflow experiment name
            train_data: Training dataset
            val_data: Optional validation dataset
            run_name: Optional MLflow run name
            model_path: Optional path to save model artifact
            extra_params: Optional extra params to log (e.g. data/preprocessing config)
            tracking: Whether to enable MLflow tracking (default True)
            seed: Optional random seed for reproducibility (seeds all libraries before training)
            **train_kwargs: Model-specific training arguments passed to _fit()
        """
        self._tracking = tracking

        # Seed all random number generators before training
        if seed is not None:
            from ml_project_template.utils import seed_everything
            seed_everything(seed)

        if not tracking:
            self._fit(train_data, val_data=val_data, **train_kwargs)
            if model_path:
                self.save(model_path)
            return

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # Log seed
            if seed is not None:
                mlflow.log_param("seed", seed)

            # Log model params
            mlflow.log_params(self.get_params())

            # Log extra params (data config, preprocessing config, etc.)
            if extra_params:
                mlflow.log_params(extra_params)

            # Model-specific training
            # Training params manually logged
            self._fit(train_data, val_data=val_data, **train_kwargs)

            # Save model artifact
            if model_path:
                if model_path.startswith("s3://"):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        saved_path = self._save_to_s3(model_path, tmp_dir)
                        mlflow.log_artifact(saved_path)
                else:
                    saved_path = self.save(model_path)
                    mlflow.log_artifact(saved_path)

    def _save_to_s3(self, s3_path: str, tmp_dir: str) -> str:
        """Save model to a temp dir, upload to S3, return local path for MLflow logging."""
        from ml_project_template.utils import get_s3_filesystem
        fs = get_s3_filesystem()
        local_path = os.path.join(tmp_dir, os.path.basename(s3_path))
        saved_path = self.save(local_path)
        fs.put(saved_path, s3_path, recursive=True)
        return saved_path

    @abstractmethod
    def _fit(self, train_data: Dataset, val_data: Dataset | None = None, **kwargs) -> None:
        """Model-specific training logic. Subclasses implement this."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input features."""
        raise NotImplementedError

    def save(self, path: str) -> str:
        """Save model to a directory with config.json and weights. Returns the directory path."""
        from ml_project_template.models.registry import ModelRegistry

        os.makedirs(path, exist_ok=True)

        config = {
            "model_name": ModelRegistry.get_name(type(self)),
            "model_params": self.get_params(),
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

        self._save_weights(path)
        return path

    @abstractmethod
    def _save_weights(self, dir_path: str) -> None:
        """Save model weights to directory. Subclasses implement this."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load model from disk. Supports both directory-based and legacy single-file paths."""
        if os.path.isdir(path):
            self._load_weights(path)
        else:
            self._load_weights_legacy(path)

    @abstractmethod
    def _load_weights(self, dir_path: str) -> None:
        """Load model weights from directory. Subclasses implement this."""
        raise NotImplementedError

    @abstractmethod
    def _load_weights_legacy(self, path: str) -> None:
        """Load model weights from legacy single-file path. Subclasses implement this."""
        raise NotImplementedError
