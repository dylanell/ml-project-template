"""Tests for model implementations.

Each model has two tests:
- test_lifecycle: Full roundtrip — create model, train (with tracking=False to
  skip MLflow), predict (check output shape), verify the directory-based artifact
  structure (config.json + weights file), load into a fresh instance, and confirm
  predictions match exactly. train() saves to model_path + "_final" internally,
  so all assertions reference that path — no explicit save() call needed.
- test_get_params: Verify that get_params() returns the expected keys and values,
  including both model-specific args and inherited defaults (e.g. Fabric args).
"""

import json

import numpy as np

from ml_project_template.models.mlp_classifier import MLPClassifier
from ml_project_template.models.gb_classifier import GBClassifier


class TestMLPClassifier:
    def test_lifecycle(self, iris_tiny, tmp_path):
        # Create model: 4 input features → 8 hidden units → 3 output classes
        model = MLPClassifier(
            layer_dims=[4, 8, 3],
            hidden_activation="ReLU",
            output_activation="Identity",
            use_bias=True,
        )

        # train() saves to model_path + "_final" (and "_best" on val improvement)
        model.train(
            train_data=iris_tiny,
            val_data=iris_tiny,
            tracking=False,
            max_epochs=5,
            model_path=str(tmp_path / "model"),
        )

        # MLP returns raw logits — shape is (num_samples, num_classes)
        preds = model.predict(iris_tiny.X)
        assert preds.shape == (20, 3)

        # Verify artifact structure written by train()
        assert (tmp_path / "model_final").is_dir()
        assert (tmp_path / "model_final" / "config.json").exists()
        assert (tmp_path / "model_final" / "model.pt").exists()

        # config.json should contain the model name and init params
        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "mlp_classifier"
        assert config["model_params"]["layer_dims"] == [4, 8, 3]
        assert config["model_params"]["hidden_activation"] == "ReLU"
        assert config["model_params"]["output_activation"] == "Identity"
        assert config["model_params"]["use_bias"] is True
        assert config["model_params"]["norm"] is None

        model2 = MLPClassifier.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(iris_tiny.X)
        np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        """Verify auto-captured __init__ params contain only architecture args."""
        model = MLPClassifier(
            layer_dims=[4, 16, 3],
            hidden_activation="ReLU",
            output_activation="Identity",
            use_bias=True,
            norm="layer",
        )
        params = model.get_params()
        assert params["layer_dims"] == [4, 16, 3]
        assert params["hidden_activation"] == "ReLU"
        assert params["output_activation"] == "Identity"
        assert params["use_bias"] is True
        assert params["norm"] == "layer"
        # Fabric kwargs are training-time args, not architecture — should not appear
        assert "accelerator" not in params


class TestGBClassifier:
    def test_lifecycle(self, iris_tiny, tmp_path):
        model = GBClassifier(n_estimators=10, max_depth=2)

        # train() saves to model_path + "_final" internally
        model.train(
            train_data=iris_tiny,
            val_data=iris_tiny,
            tracking=False,
            model_path=str(tmp_path / "model"),
        )

        # GB returns class labels — shape is (num_samples,) with values in {0, 1, 2}
        preds = model.predict(iris_tiny.X)
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1, 2})

        # Verify artifact structure written by train()
        assert (tmp_path / "model_final").is_dir()
        assert (tmp_path / "model_final" / "config.json").exists()
        assert (tmp_path / "model_final" / "model.joblib").exists()

        # config.json should contain the model name and init params
        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "gb_classifier"
        assert config["model_params"]["n_estimators"] == 10
        assert config["model_params"]["max_depth"] == 2

        model2 = GBClassifier.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(iris_tiny.X)
        np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        """Verify sklearn's get_params() returns the expected hyperparameters."""
        model = GBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05)
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3
        assert params["learning_rate"] == 0.05
