"""Training orchestration."""

from __future__ import annotations

import mlflow
from sklearn.metrics import accuracy_score

from ml_project_template.data import Dataset
from ml_project_template.models.base import BaseModel


class Trainer:
    """Orchestrates model training, evaluation, and artifact management."""

    def __init__(
        self,
        model: BaseModel,
        experiment_name: str = "default",
        tracking_uri: str | None = None,
    ):
        self.model = model
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def run(
        self,
        train_data: Dataset,
        val_data: Dataset | None = None,
        model_path: str | None = None,
        run_name: str | None = None,
    ) -> dict:
        """Full training pipeline with MLflow tracking.

        Args:
            train_data: Training dataset
            val_data: Optional test dataset for validation
            model_path: Optional path to save model artifact
            run_name: Optional MLflow run name (auto-generated if not provided)

        Returns:
            Dict of metrics (empty if no val_data provided)
        """
        with mlflow.start_run(run_name=run_name):
            # Log model parameters
            mlflow.log_params(self.model.get_params())

            # Train model
            self.model.train(train_data=train_data, val_data=val_data)

            # If test data is provided, predict and compute metrics
            metrics = {}
            if val_data is not None:
                predictions = self.model.predict(val_data.X)
                metrics = {"accuracy": accuracy_score(val_data.y, predictions)}
                mlflow.log_metrics(metrics)

            # If model path is provided, write model artifact
            if model_path:
                saved_path = self.model.save(model_path)
                mlflow.log_artifact(saved_path)

            return metrics
