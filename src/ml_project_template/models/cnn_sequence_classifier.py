"""PyTorch CNN sequence classifier."""


from __future__ import annotations

from typing import Union, Any, Optional, List, Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ml_project_template.data import SequenceDataset
from ml_project_template.models.pytorch_base import BasePytorchModel
from ml_project_template.modules.sequence_cnn import SequenceCNN

class CNNSequenceClassifier(BasePytorchModel):
    """Sequence classifier model using 1D CNNs."""

    def __init__(
        self,
        embed_dims: List[int],
        kernel_spec: List[List[int]],
        seq_length: int,
        output_dim: int,
        padding_idx: int = 0,
        hidden_activation: str = "ReLU",
        output_activation: str = "Identity",
        use_bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
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

        self.seq_length = seq_length

        self.model = SequenceCNN(
            embed_dims=embed_dims,
            kernel_spec=kernel_spec,
            seq_length=seq_length,
            output_dim=output_dim,
            padding_idx=padding_idx,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            use_bias=use_bias,
            norm=norm
        )

        self.loss_fcn = nn.CrossEntropyLoss()

    def predict(self, sequences: list[list[int]]) -> np.ndarray:
        """Run inference on a list of encoded sequences.

        Sequences are truncated/padded to self.seq_length before inference,
        matching the fixed-length batches used during training.
        """
        import numpy as np
        pad = 0
        X = np.zeros((len(sequences), self.seq_length), dtype=np.int64)
        for i, seq in enumerate(sequences):
            trunc = seq[:self.seq_length]
            X[i, :len(trunc)] = trunc
        self.model.eval()
        X_tensor = torch.from_numpy(X).long().to(self.fabric.device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return output.cpu().numpy()

    def _fit(
        self,
        train_data: SequenceDataset,
        val_data: Optional[SequenceDataset] = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1
    ):
        if patience > 0 and val_data is None:
            raise ValueError("Patience requires a validation dataset.")

        # Initialize optimizer and fabric
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # Initialize dataloaders
        train_dataloader = train_data.to_pytorch(batch_size=batch_size, shuffle=True, seq_length=self.seq_length)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_data is not None:
            val_dataloader = val_data.to_pytorch(batch_size=batch_size, shuffle=False, seq_length=self.seq_length)
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

            self.log_metric("train_loss", train_loss, step=epoch)

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

                self.log_metric("val_loss", val_loss, step=epoch)

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
        self.log_param("lr", lr)
        self.log_param("batch_size", batch_size)
        self.log_param("seq_length", self.seq_length)
        self.log_param("max_epochs", max_epochs)
        self.log_param("val_frequency", val_frequency)
        self.log_param("patience", patience)