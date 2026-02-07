"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
import inspect
import numpy as np
from typing import Union, Any, Optional

from pathlib import Path

import mlflow
import torch

import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ml_project_template.data import Dataset


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

    def train(
        self,
        *,
        experiment_name: str,
        train_data: Dataset,
        val_data: Dataset | None = None,
        run_name: str | None = None,
        model_path: str | None = None,
        **train_kwargs,
    ) -> None:
        """Full training pipeline with MLflow tracking.

        Args:
            experiment_name: MLflow experiment name
            train_data: Training dataset
            val_data: Optional validation dataset
            run_name: Optional MLflow run name
            model_path: Optional path to save model artifact
            **train_kwargs: Model-specific training arguments passed to _fit()
        """
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # Log model params
            mlflow.log_params(self.get_params())

            # Model-specific training
            # Training params manually logged 
            self._fit(train_data, val_data=val_data, **train_kwargs)

            # Save model artifact
            if model_path:
                saved_path = self.save(model_path)
                mlflow.log_artifact(saved_path)

    @abstractmethod
    def _fit(self, train_data: Dataset, val_data: Dataset | None = None, **kwargs) -> None:
        """Model-specific training logic. Subclasses implement this."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input features."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> str:
        """Save model to disk. Returns the actual path saved."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        raise NotImplementedError


class BasePytorchModel(BaseModel, ABC):
    """Abstract base class for Pytorch models.

    Ref: https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None
    ):
        self.model: torch.nn.Module  # Subclasses must set this

        # Initialize fabric
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw model output as numpy array."""
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.fabric.device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return output.cpu().numpy()

    def save(self, path: str) -> str:
        """Save model state dict to disk. Returns the actual path saved."""
        path = f"{path}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fabric.save(path, {"model": self.model})
        return path

    def load(self, path: str) -> None:
        """Load model state dict from disk."""
        state = {"model": self.model}
        self.fabric.load(f"{path}.pt", state)
