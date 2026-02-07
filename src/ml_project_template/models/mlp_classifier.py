"""PyTorch MLP classifier."""

from __future__ import annotations

from typing import Union, Any, Optional

import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm

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

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        self.loss_fcn = nn.CrossEntropyLoss()

    def _fit(
        self,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        *,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1
    ) -> None:
        if patience > 0 and val_data is None:
            raise ValueError("Patience requires a validation dataset.")

        # Initialize optimizer and fabric
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        train_dataloader = train_data.to_pytorch(batch_size=batch_size, shuffle=True)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_data is not None:
            val_dataloader = val_data.to_pytorch(batch_size=batch_size, shuffle=False)
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        epochs_without_improvement = 0
        val_loss = float("inf")
        best_val_loss = float("inf")

        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            # Train
            cum_train_loss = 0
            model.train()
            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = self.loss_fcn(output, y_batch)
                self.fabric.backward(loss)
                optimizer.step()
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(train_dataloader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Validate
            if (val_data is not None) and ((epoch + 1) % val_frequency == 0):
                cum_val_loss = 0
                model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_dataloader:
                        output = model(X_batch)
                        loss = self.loss_fcn(output, y_batch)
                        cum_val_loss += loss.item()
                val_loss = cum_val_loss / len(val_dataloader)

                mlflow.log_metric("val_loss", val_loss, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    print(f"{patience} epochs reached without improvement. Early stopping.")
                    break

            status = f"Epoch: {epoch+1}/{max_epochs} | train_loss: {train_loss:.4f} | "\
                f"val_loss: {val_loss:.4f} | best_val_loss: {best_val_loss:.4f}"
            pbar.set_description(status)

        # Log training parameters
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("val_frequency", val_frequency)
        mlflow.log_param("patience", patience)

