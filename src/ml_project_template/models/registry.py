"""Model registry for discovering and instantiating models."""

from __future__ import annotations

from ml_project_template.models.base import BaseModel
from ml_project_template.models.gb_classifier import GBClassifier
from ml_project_template.models.mlp_classifier import MLPClassifier


class ModelRegistry:
    """Registry for model classes."""

    _models: dict[str, type[BaseModel]] = {
        "gb_classifier": GBClassifier,
        "mlp_classifier": MLPClassifier,
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
    def register(cls, name: str, model_class: type[BaseModel]) -> None:
        """Register a new model class."""
        cls._models[name] = model_class
