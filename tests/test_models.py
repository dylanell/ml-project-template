"""Tests for model implementations."""

import json
import os
import tempfile

import numpy as np

from ml_project_template.models.mlp_classifier import MLPClassifier
from ml_project_template.models.gb_classifier import GBClassifier


class TestMLPClassifier:
    def test_lifecycle(self, iris_tiny):
        model = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3)
        model.train(
            train_data=iris_tiny,
            tracking=False,
            max_epochs=5,
        )

        preds = model.predict(iris_tiny.X)
        assert preds.shape == (20, 3)

        with tempfile.TemporaryDirectory() as tmp:
            saved_path = model.save(f"{tmp}/model")

            # Verify directory structure
            assert os.path.isdir(saved_path)
            assert os.path.exists(f"{tmp}/model/config.json")
            assert os.path.exists(f"{tmp}/model/model.pt")

            with open(f"{tmp}/model/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "mlp_classifier"
            assert config["model_params"]["input_dim"] == 4
            assert config["model_params"]["hidden_dim"] == 8
            assert config["model_params"]["num_classes"] == 3

            model2 = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3)
            model2.load(f"{tmp}/model")

            preds2 = model2.predict(iris_tiny.X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        model = MLPClassifier(input_dim=4, hidden_dim=16, num_classes=3)
        params = model.get_params()
        assert params["input_dim"] == 4
        assert params["hidden_dim"] == 16
        assert params["num_classes"] == 3
        assert params["accelerator"] == "auto"


class TestGBClassifier:
    def test_lifecycle(self, iris_tiny):
        model = GBClassifier(n_estimators=10, max_depth=2)
        model.train(
            train_data=iris_tiny,
            tracking=False,
        )

        preds = model.predict(iris_tiny.X)
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1, 2})

        with tempfile.TemporaryDirectory() as tmp:
            saved_path = model.save(f"{tmp}/model")

            # Verify directory structure
            assert os.path.isdir(saved_path)
            assert os.path.exists(f"{tmp}/model/config.json")
            assert os.path.exists(f"{tmp}/model/model.joblib")

            with open(f"{tmp}/model/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "gb_classifier"
            assert config["model_params"]["n_estimators"] == 10
            assert config["model_params"]["max_depth"] == 2

            model2 = GBClassifier(n_estimators=10, max_depth=2)
            model2.load(f"{tmp}/model")

            preds2 = model2.predict(iris_tiny.X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        model = GBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05)
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3
        assert params["learning_rate"] == 0.05
