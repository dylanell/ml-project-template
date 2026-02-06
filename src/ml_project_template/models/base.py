"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Any, Optional

import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ml_project_template.data import Dataset


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def train(self, dataset: Dataset) -> None:
        """Train the model on a dataset."""
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

    @abstractmethod
    def get_params(self) -> dict:
        """Return model parameters for logging."""
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

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ):
        raise NotImplementedError