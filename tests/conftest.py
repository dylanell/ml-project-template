"""Shared test fixtures."""

# torch must be imported before sklearn to ensure PyTorch's Accelerate BLAS is
# initialized first. On macOS x86_64, importing sklearn first causes its OpenBLAS
# to conflict with PyTorch's Accelerate framework, producing a segfault during
# any forward/backward pass. The notebook avoids this by importing MLPClassifier
# (which pulls in torch) before anything sklearn-related.
import torch  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from ml_project_template.data import TabularDataset


@pytest.fixture
def iris_tiny(tmp_path):
    """Tiny 20-sample, 4-feature, 3-class dataset written to a temp CSV and loaded."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    # String labels so LabelEncoder produces 3 class_names = ["a", "b", "c"]
    labels = np.array(["a", "b", "c"] * 6 + ["a", "b"])

    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["target"] = labels

    csv_path = tmp_path / "iris_tiny.csv"
    df.to_csv(csv_path, index=False)

    return TabularDataset(csv_path=str(csv_path), target_col="target")
