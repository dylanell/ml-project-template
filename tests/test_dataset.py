"""Tests for TabularDataset.

Tests cover dataset operations that don't require external services:
- attributes: X, y, feature_names, class_names are correctly populated on load.
- getitem: __getitem__ returns the expected feature/target dict for DataLoader use.
- from_csv_local_with_s3_env_set: Loading a local CSV works even with S3 env vars
  set (regression test for get_storage_options bug).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ml_project_template.data import TabularDataset


def test_attributes(iris_tiny):
    """Dataset should expose X, y, feature_names, and class_names after loading."""
    assert iris_tiny.X.shape == (20, 4)
    assert iris_tiny.y.shape == (20,)
    assert iris_tiny.feature_names == ["f0", "f1", "f2", "f3"]
    assert iris_tiny.class_names == ["a", "b", "c"]
    assert len(iris_tiny) == 20


def test_getitem(iris_tiny):
    """__getitem__ should return float32 features and an integer target for DataLoader use."""
    loader = DataLoader(iris_tiny, batch_size=10, shuffle=False)
    batch = next(iter(loader))
    assert "features" in batch and "targets" in batch
    assert batch["features"].shape == (10, 4)
    assert batch["features"].dtype == torch.float32
    assert batch["targets"].shape == (10,)


def test_from_csv_local_with_s3_env_set(iris_tiny, monkeypatch, tmp_path):
    """Loading a local CSV should work even when S3 env vars are set.

    Regression test: get_storage_options() previously checked whether
    S3_ENDPOINT_URL was set (not whether the path was S3), so local file
    loading would fail when S3 env vars were present.
    """
    # Simulate a local dev environment with MinIO credentials set
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:7000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")

    # Write a small CSV from the fixture data
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(iris_tiny.X, columns=iris_tiny.feature_names)
    df["target"] = iris_tiny.y
    df.to_csv(csv_path, index=False)

    # This should succeed despite S3 env vars being set
    dataset = TabularDataset(csv_path=str(csv_path), target_col="target")
    assert len(dataset) == 20
    assert dataset.feature_names == iris_tiny.feature_names
