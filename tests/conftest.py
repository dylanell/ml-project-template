"""Shared test fixtures."""

import numpy as np
import pytest

from ml_project_template.data import TabularDataset, SequenceDataset


@pytest.fixture
def iris_tiny():
    """Tiny 20-sample, 4-feature, 3-class dataset built from numpy arrays."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)
    return TabularDataset(
        X=X,
        y=y,
        feature_names=["f0", "f1", "f2", "f3"],
        class_names=["a", "b", "c"],
    )


@pytest.fixture
def sst2_tiny():
    """Tiny 20-sample binary sentiment dataset built from numpy arrays.

    Vocab is minimal: 10 real tokens + PAD + UNK = 12 entries.
    Sequences are short (4-6 tokens) to keep tests fast.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1, "good": 2, "great": 3, "bad": 4,
             "awful": 5, "the": 6, "film": 7, "was": 8, "not": 9, "very": 10, "ok": 11}
    sequences = [
        [2, 8, 3],        # "good was great"
        [4, 8, 5],        # "bad was awful"
        [6, 7, 8, 2],     # "the film was good"
        [6, 7, 8, 4],     # "the film was bad"
        [10, 3, 7],       # "very great film"
        [9, 3, 7],        # "not great film"
        [2, 3, 3, 3],     # "good great great great"
        [4, 5, 5, 5],     # "bad awful awful awful"
        [6, 8, 10, 2],    # "the was very good"
        [6, 8, 9, 3],     # "the was not great"
        [3, 7, 8, 2],     # "great film was good"
        [5, 7, 8, 4],     # "awful film was bad"
        [2, 2, 2],        # "good good good"
        [4, 4, 4],        # "bad bad bad"
        [10, 2, 7],       # "very good film"
        [9, 4, 7],        # "not bad film"
        [3, 3, 6, 7],     # "great great the film"
        [5, 5, 6, 7],     # "awful awful the film"
        [8, 10, 3],       # "was very great"
        [8, 9, 4],        # "was not bad"
    ]
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    return SequenceDataset(
        sequences=sequences,
        labels=y,
        vocab=vocab,
        class_names=["negative", "positive"],
    )
