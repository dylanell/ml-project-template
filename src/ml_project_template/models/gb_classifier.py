"""Gradient Boosting model implementation using scikit-learn."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier

from ml_project_template.data import Dataset
from ml_project_template.models.base import BaseModel


class GBClassifier(BaseModel):
    """Scikit-learn Gradient Boosting classifier wrapper."""

    def __init__(self, **kwargs):
        self.model = _GradientBoostingClassifier(**kwargs)

    def train(self, dataset: Dataset) -> None:
        self.model.fit(dataset.X, dataset.y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> str:
        """Save model to disk. Returns the actual path saved."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        full_path = f"{path}.joblib"
        joblib.dump(self.model, full_path)
        return full_path

    def load(self, path: str) -> None:
        self.model = joblib.load(f"{path}.joblib")

    def get_params(self) -> dict:
        return self.model.get_params()
