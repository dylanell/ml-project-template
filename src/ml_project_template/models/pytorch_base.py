"""Base class for PyTorch models using Lightning Fabric."""

from __future__ import annotations

from abc import ABC
import os
from typing import Union, Any, Optional

import numpy as np
import torch
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ml_project_template.models.base import BaseModel


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

    def _save_weights(self, dir_path: str) -> None:
        """Save model state dict to directory."""
        self.fabric.save(os.path.join(dir_path, "model.pt"), {"model": self.model})

    def _load_weights(self, dir_path: str) -> None:
        """Load model state dict from directory."""
        state = {"model": self.model}
        self.fabric.load(os.path.join(dir_path, "model.pt"), state)

