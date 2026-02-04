"""PyTorch MLP classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mlflow

from ml_project_template.data import Dataset
from ml_project_template.models.base import BaseModel


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class MLPClassifier(BaseModel):
    """Simple 2-layer MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        self._params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        }

        self.epochs = epochs
        self.batch_size = batch_size

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataset: Dataset) -> None:
        dataloader = dataset.to_pytorch(batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            cum_epoch_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                cum_epoch_loss += loss.item()

            epoch_loss = cum_epoch_loss / len(dataloader)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

            # Run an eval loop

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict model outputs."""

        # Compute outputs from input array
        self.model.eval()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            output = self.model(X_tensor)
            predictions = output.argmax(dim=1)

        return predictions.numpy()

    def save(self, path: str) -> str:
        """Save model to disk. Returns the actual path saved."""

        # Append extension to path
        path = f"{path}.pt"

        # Make sure directory to the artifact file exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

        return path

    def load(self, path: str) -> None:
        """Load model from checkpoint file."""

        self.model.load_state_dict(torch.load(f"{path}.pt"))

    def get_params(self) -> dict:
        return self._params
