"""1D CNN for processing sequernces."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(
        self,
        dim0: int,
        dim1: int
    ):
        super().__init__()

        self._dim0 = dim0
        self._dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self._dim0, self._dim1)


class SequenceCNN(nn.Module):
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
        norm: Literal["batch", "layer"] | None = None
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            embed_dims[0],
            embed_dims[1],
            padding_idx=padding_idx
        )

        cnn_layers = []
        for i in range(len(kernel_spec)):
            # Make a "1D" convolution using a 2D conv layer by setting the kernel
            # width to the full width of the previous layer.
            in_channels = 1
            kernel_height, out_channels, h_stride = kernel_spec[i]
            kernel_width = kernel_spec[i-1][1] if i != 0 else embed_dims[1]
            cnn_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_height, kernel_width),
                stride=(h_stride, 1),
                bias=use_bias
            ))

            if i == len(kernel_spec) - 1:
                # Output layer: Transpose → output activation, no norm
                cnn_layers.append(Transpose(dim0=3, dim1=1))
                cnn_layers.append(getattr(nn, output_activation)())
            else:
                # Hidden layer: apply norm at the appropriate point, then activation
                # Batch norm operates on [B, out_channels, seq_len, 1] — before Transpose
                if norm == "batch":
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                # The result of the conv above will be [B, out_channels, new_seq_length, 1].
                # We swap dims 1 and 3 to make the output have a 1-dim channel dimension.
                cnn_layers.append(Transpose(dim0=3, dim1=1))
                # Layer norm operates on [B, 1, seq_len, out_channels] — after Transpose
                if norm == "layer":
                    cnn_layers.append(nn.LayerNorm(out_channels))
                cnn_layers.append(getattr(nn, hidden_activation)())

            # Calculate sequence length after this conv layer 
            seq_length = (seq_length - kernel_height) // h_stride + 1

        # Record final sequence length
        final_seq_length = seq_length

        self.cnn = nn.Sequential(*cnn_layers)

        # Compute flattened CNN output size and attach classifier head
        cnn_output_dim = final_seq_length * kernel_spec[-1][1]
        self.linear = nn.Linear(cnn_output_dim, output_dim, bias=use_bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)          # [B, 1, seq_len, embed_dim]
        x = self.cnn(x)             # [B, 1, final_seq_len, last_out_channels]
        x = x.flatten(start_dim=1)  # [B, cnn_output_dim]
        x = self.linear(x)          # [B, output_dim]
        return x
