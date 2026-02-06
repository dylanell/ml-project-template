"""PyTorch MLP classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Union, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import mlflow

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ml_project_template.data import Dataset
from ml_project_template.models.base import BasePytorchModel


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


class MLPClassifier(BasePytorchModel):
    """Simple 2-layer MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None
    ):
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers
        )

        self._params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "lr": lr,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "val_frequency": val_frequency,
            "patience": patience,
            "accelerator": accelerator,
            "strategy": strategy,
            "devices": devices,
            "precision": precision,
            "plugins": plugins,
            "callbacks": callbacks,
            "loggers": loggers
        }

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.patience = patience

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Wrap model and optimizers to fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Setup losses and metrics
        self.loss_fcn = nn.CrossEntropyLoss()

    def train(
        self, 
        train_data: Dataset,
        val_data: Optional[Dataset] = None
    ) -> None:
        if self.patience > 0 and val_data is None:
            raise ValueError("Patience requires a validation dataset.")
        
        train_dataloader = train_data.to_pytorch(batch_size=self.batch_size, shuffle=True)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_data is not None:
            val_dataloader = val_data.to_pytorch(batch_size=self.batch_size, shuffle=False)
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        epochs_without_improvement = 0
        best_val_loss = float('inf')

        # Train over all epochs
        for epoch in range(self.max_epochs):
            # Update over full training set
            cum_train_loss = 0
            self.model.train()
            for X_batch, y_batch in train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fcn(output, y_batch)
                self.fabric.backward(loss)
                self.optimizer.step()
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(train_dataloader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Compute metrics over full validation set
            # Only if validation set provided and we are at a validation epoch
            if (val_data is not None) and ((epoch + 1) % self.val_frequency == 0):
                cum_val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_dataloader:
                        output = self.model(X_batch)
                        loss = self.loss_fcn(output, y_batch)
                        cum_val_loss += loss.item()
                val_loss = cum_val_loss / len(val_dataloader)

                mlflow.log_metric("val_loss", val_loss, step=epoch)

                # Update patience count
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Check for patience count or max epochs
                if epochs_without_improvement == self.patience:
                    print(f"{self.patience} epochs without improvement, exiting.")
                    break

        # Log final losses/metrics

        # Save model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict model outputs."""

        # Compute outputs from input array
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.fabric.device)
        with torch.no_grad():
            output = self.model(X_tensor)
            predictions = output.argmax(dim=1)

        return predictions.cpu().numpy()

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

        self.model.load_state_dict(torch.load(f"{path}.pt", map_location=self.fabric.device))

    def get_params(self) -> dict:
        return self._params
