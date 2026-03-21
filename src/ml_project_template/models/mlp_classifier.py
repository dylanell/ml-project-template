"""PyTorch MLP classifier."""

from __future__ import annotations

from typing import List, Literal

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import lightning as L

from ..data import TabularDataset
from .base import BaseModel
from ..modules.fully_connected import FullyConnected


class MLPClassifier(BaseModel):
    """Simple 2-layer MLP classifier."""

    name = "mlp_classifier"

    def __init__(
        self,
        layer_dims: List[int],
        hidden_activation: str = "ReLU",
        output_activation: str = "Identity",
        use_bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
    ):
        super().__init__()

        self.model = FullyConnected(
            layer_dims=layer_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            use_bias=use_bias,
            norm=norm,
        )

        self.loss_fcn = nn.CrossEntropyLoss()

        # Fabric handles device placement automatically (accelerator="auto" picks
        # GPU if available, falls back to CPU). Lives on the instance so it isn't
        # garbage collected between _fit() and later use — premature GC hangs on
        # macOS waiting on a semaphore.
        self._fabric = L.Fabric(accelerator="auto")

    def _fit(
        self,
        train_data: TabularDataset,
        val_data: TabularDataset,
        *,
        model_path: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1,
    ) -> None:
        # Log training parameters before training starts
        self.log_param("lr", lr)
        self.log_param("weight_decay", weight_decay)
        self.log_param("batch_size", batch_size)
        self.log_param("max_epochs", max_epochs)
        self.log_param("val_frequency", val_frequency)
        self.log_param("patience", patience)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        model, optimizer = self._fabric.setup(self.model, optimizer)

        # Initialize dataloaders
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_dataloader = self._fabric.setup_dataloaders(train_dataloader)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        val_dataloader = self._fabric.setup_dataloaders(val_dataloader)

        epochs_without_improvement = 0
        val_loss = float("inf")
        best_val_loss = float("inf")

        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            # Train
            cum_train_loss = 0
            model.train()
            for batch in train_dataloader:
                X_batch = batch["features"]
                y_batch = batch["targets"]
                optimizer.zero_grad()
                output = model(X_batch)
                loss = self.loss_fcn(output, y_batch)
                self._fabric.backward(loss)
                optimizer.step()
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(train_dataloader)

            self.log_metric("train_loss", train_loss, step=epoch)

            # Validate
            if (epoch + 1) % val_frequency == 0:
                cum_val_loss = 0
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        X_batch = batch["features"]
                        y_batch = batch["targets"]
                        output = model(X_batch)
                        loss = self.loss_fcn(output, y_batch)
                        cum_val_loss += loss.item()
                val_loss = cum_val_loss / len(val_dataloader)

                self.log_metric("val_loss", val_loss, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save(f"{model_path}_best")
                else:
                    epochs_without_improvement += 1

                if patience >= 0 and epochs_without_improvement >= patience:
                    print(
                        f"{patience} epochs reached without improvement. "
                        f"Early stopping."
                    )
                    break

            status = (
                f"Epoch: {epoch + 1}/{max_epochs} | train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | best_val_loss: {best_val_loss:.4f}"
            )
            pbar.set_description(status)

        # Final validation metrics
        metrics = self.evaluate(val_data)
        for k, v in metrics.items():
            self.log_metric(f"val_{k}", v)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw model output as numpy array."""
        self.model.eval()
        # Auto-detect current device — works whether the model was just trained
        # (weights on Fabric's device) or freshly loaded (weights on CPU).
        device = next(self.model.parameters()).device
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return output.cpu().numpy()

    def to(self, device) -> MLPClassifier:
        """Move model to device. Returns self for chaining."""
        self.model.to(device)
        return self

    def _save_weights(self, dir_path: str) -> None:
        """Save model state dict to directory."""
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model.pt"))

    def _load_weights(self, dir_path: str) -> None:
        """Load model state dict from directory."""
        raw = torch.load(
            os.path.join(dir_path, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        # Handle checkpoints saved by the old fabric.save() format: {"model": state_dict}
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        self.model.load_state_dict(state_dict)
