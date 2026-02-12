"""Tests for TabularDataset."""

import torch


def test_split(iris_tiny):
    train, test = iris_tiny.split(test_size=0.3, random_state=0)
    assert len(train) == 14
    assert len(test) == 6
    assert train.feature_names == iris_tiny.feature_names
    assert test.class_names == iris_tiny.class_names


def test_to_pytorch(iris_tiny):
    loader = iris_tiny.to_pytorch(batch_size=10, shuffle=False)
    X_batch, y_batch = next(iter(loader))
    assert isinstance(X_batch, torch.Tensor)
    assert X_batch.shape == (10, 4)
    assert y_batch.shape == (10,)
