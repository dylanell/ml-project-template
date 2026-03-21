"""Model registry for discovering and instantiating models."""

from __future__ import annotations

import json
import os

from .base import BaseModel
from .gb_classifier import GBClassifier
from .mlp_classifier import MLPClassifier


class ModelRegistry:
    """Registry for model classes."""

    _models: dict[str, type[BaseModel]] = {
        "gb_classifier": GBClassifier,
        "mlp_classifier": MLPClassifier
    }

    @classmethod
    def list(cls) -> list[str]:
        """List available model names."""
        return list(cls._models.keys())

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {cls.list()}")
        return cls._models[name]

    @classmethod
    def load(cls, path: str, device=None) -> BaseModel:
        """Load a model from a directory containing config.json and weights.

        Args:
            path: Directory containing config.json and weights.
            device: Optional device to move the model to after loading
                    (e.g. "cuda", "mps", torch.device("cuda:0")).
                    No-op for non-PyTorch models.
        """
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model_class = cls.get(config["model_name"])
        model = model_class.load(path)
        if device is not None:
            model.to(device)
        return model

    @classmethod
    def register(cls, name: str, model_class: type[BaseModel]) -> None:
        """Register a new model class."""
        cls._models[name] = model_class
