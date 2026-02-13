"""Tests for model registry."""

import tempfile

import numpy as np
import pytest

from ml_project_template.models import ModelRegistry
from ml_project_template.models.gb_classifier import GBClassifier
from ml_project_template.models.mlp_classifier import MLPClassifier


def test_list():
    names = ModelRegistry.list()
    assert "gb_classifier" in names
    assert "mlp_classifier" in names


def test_get():
    assert ModelRegistry.get("gb_classifier") is GBClassifier
    assert ModelRegistry.get("mlp_classifier") is MLPClassifier


def test_get_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelRegistry.get("nonexistent")


def test_get_name():
    assert ModelRegistry.get_name(MLPClassifier) == "mlp_classifier"
    assert ModelRegistry.get_name(GBClassifier) == "gb_classifier"


def test_get_name_unknown():
    class FakeModel:
        pass
    with pytest.raises(ValueError, match="not registered"):
        ModelRegistry.get_name(FakeModel)


def test_load_mlp(iris_tiny):
    model = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3)
    model.train(train_data=iris_tiny, tracking=False, max_epochs=5)
    preds = model.predict(iris_tiny.X)

    with tempfile.TemporaryDirectory() as tmp:
        model.save(f"{tmp}/model")
        loaded = ModelRegistry.load(f"{tmp}/model")

        assert isinstance(loaded, MLPClassifier)
        preds2 = loaded.predict(iris_tiny.X)
        np.testing.assert_array_equal(preds, preds2)


def test_load_gb(iris_tiny):
    model = GBClassifier(n_estimators=10, max_depth=2)
    model.train(train_data=iris_tiny, tracking=False)
    preds = model.predict(iris_tiny.X)

    with tempfile.TemporaryDirectory() as tmp:
        model.save(f"{tmp}/model")
        loaded = ModelRegistry.load(f"{tmp}/model")

        assert isinstance(loaded, GBClassifier)
        preds2 = loaded.predict(iris_tiny.X)
        np.testing.assert_array_equal(preds, preds2)
