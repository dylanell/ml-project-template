"""Base model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

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
