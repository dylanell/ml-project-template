"""Sequence dataset implementation for text/token data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from ml_project_template.data.base import BaseDataset


PAD_TOKEN = "<PAD>"  # index 0
UNK_TOKEN = "<UNK>"  # index 1


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, keep alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


def _build_vocab(texts: list[str], max_vocab_size: int | None = None) -> dict[str, int]:
    """Build token → index vocabulary from a list of raw text strings."""
    counts = Counter(token for text in texts for token in _tokenize(text))
    tokens = [token for token, _ in counts.most_common(max_vocab_size)]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in tokens:
        vocab[token] = len(vocab)
    return vocab


def _encode(text: str, vocab: dict[str, int]) -> list[int]:
    """Encode a text string to a list of token indices."""
    unk = vocab[UNK_TOKEN]
    return [vocab.get(token, unk) for token in _tokenize(text)]


class _SequenceTorchDataset(torch.utils.data.Dataset):
    """Internal PyTorch Dataset wrapping encoded sequences and labels."""

    def __init__(self, sequences: list[list[int]], labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def _make_collate_fn(seq_length: int | None, pad_value: int = 0):
    """Return a collate_fn that pads sequences in each batch.

    Args:
        seq_length: If given, truncate/pad all sequences to this fixed length.
                    If None, pad to the longest sequence in each batch.
        pad_value: Token index used for padding (should be PAD_TOKEN index = 0).
    """
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        sequences, labels = zip(*batch)
        if seq_length is not None:
            padded = torch.full((len(sequences), seq_length), pad_value, dtype=torch.long)
            for i, seq in enumerate(sequences):
                trunc = seq[:seq_length]
                padded[i, : len(trunc)] = trunc
        else:
            padded = rnn_utils.pad_sequence(
                sequences, batch_first=True, padding_value=pad_value
            )
        return padded, torch.stack(labels)

    return collate_fn


@dataclass
class SequenceDataset(BaseDataset):
    """Dataset for variable-length text/token sequences.

    Attributes:
        sequences: Encoded sequences as lists of token indices.
        labels: Integer class labels.
        vocab: Token → index mapping. PAD=0, UNK=1, then vocab tokens.
        class_names: Human-readable label names.
        label_encoder: LabelEncoder instance (None if labels were already integers).
    """

    sequences: list[list[int]]
    labels: np.ndarray
    vocab: dict[str, int]
    class_names: list[str]
    label_encoder: LabelEncoder | None = None

    @classmethod
    def from_csv(
        cls,
        path: str,
        text_column: str,
        label_column: str,
        sep: str = "\t",
        max_vocab_size: int | None = None,
        vocab: dict[str, int] | None = None,
        storage_options: dict | None = None,
    ) -> SequenceDataset:
        """Load a sequence dataset from a delimited text file (local or S3).

        Args:
            path: Path to the file (local path or s3:// URI).
            text_column: Name of the column containing raw text.
            label_column: Name of the column containing class labels.
            sep: Delimiter character (default '\\t' for TSV).
            max_vocab_size: Maximum number of tokens to keep in the vocabulary
                (most frequent first). None keeps all tokens.
            vocab: Pre-built vocabulary to use instead of building from data.
                Pass the training set vocabulary when loading val/test splits
                to ensure consistent token indices.
            storage_options: S3 credentials dict for remote paths (from get_storage_options()).
        """
        df = pd.read_csv(path, sep=sep, storage_options=storage_options or {})

        texts = df[text_column].astype(str).tolist()

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df[label_column])
        class_names = [str(c) for c in label_encoder.classes_]

        if vocab is None:
            vocab = _build_vocab(texts, max_vocab_size=max_vocab_size)

        sequences = [_encode(text, vocab) for text in texts]

        return cls(
            sequences=sequences,
            labels=encoded_labels,
            vocab=vocab,
            class_names=class_names,
            label_encoder=label_encoder,
        )

    def split(
        self, test_size: float = 0.2, random_state: int | None = None
    ) -> tuple[SequenceDataset, SequenceDataset]:
        """Split into train and test datasets by position (no shuffle).

        Splitting is always positional to preserve order (time-aware).
        The random_state argument is accepted for API consistency but ignored.
        Both splits share the same vocabulary.
        """
        n = len(self)
        split_idx = int(n * (1 - test_size))

        train = SequenceDataset(
            sequences=self.sequences[:split_idx],
            labels=self.labels[:split_idx],
            vocab=self.vocab,
            class_names=self.class_names,
            label_encoder=self.label_encoder,
        )
        test = SequenceDataset(
            sequences=self.sequences[split_idx:],
            labels=self.labels[split_idx:],
            vocab=self.vocab,
            class_names=self.class_names,
            label_encoder=self.label_encoder,
        )
        return train, test

    def to_pytorch(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        seq_length: int | None = None,
    ) -> DataLoader:
        """Convert to a PyTorch DataLoader.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle samples each epoch.
            seq_length: If given, all sequences are truncated or padded to this
                fixed length. If None, sequences are padded to the longest
                sequence in each batch (variable-length mode).
        """
        dataset = _SequenceTorchDataset(self.sequences, self.labels)
        collate_fn = _make_collate_fn(seq_length=seq_length, pad_value=0)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def __len__(self) -> int:
        return len(self.sequences)
