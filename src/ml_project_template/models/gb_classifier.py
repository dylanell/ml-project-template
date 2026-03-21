"""Gradient Boosting model implementation using scikit-learn."""

from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier

from ..data import TabularDataset
from .base import BaseModel


class GBClassifier(BaseModel):
    """Scikit-learn Gradient Boosting classifier wrapper."""

    name = "gb_classifier"

    def __init__(self, **kwargs):
        self.model = _GradientBoostingClassifier(**kwargs)

    def _fit(
        self,
        train_data: TabularDataset,
        val_data: TabularDataset,
        **kwargs,
    ) -> None:
        self.model.fit(train_data.X, train_data.y)

        # Final validation metrics
        metrics = self.evaluate(val_data)
        for k, v in metrics.items():
            self.log_metric(f"val_{k}", v)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _save_weights(self, dir_path: str) -> None:
        """Save model to directory."""
        joblib.dump(self.model, os.path.join(dir_path, "model.joblib"))

    def _load_weights(self, dir_path: str) -> None:
        """Load model from directory."""
        self.model = joblib.load(os.path.join(dir_path, "model.joblib"))

    # Override: BaseModel auto-captures __init__ args, but since this class
    # takes **kwargs, it would record {"kwargs": {...}} (a nested dict).
    # Sklearn's get_params() gives us a flat dict of all params with defaults.
    def get_params(self) -> dict:
        return self.model.get_params()
