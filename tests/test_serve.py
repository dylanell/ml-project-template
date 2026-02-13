"""Tests for the serving endpoint."""

import tempfile

import numpy as np
from fastapi.testclient import TestClient

from ml_project_template.models.mlp_classifier import MLPClassifier
from ml_project_template.models.gb_classifier import GBClassifier
from ml_project_template.serving.iris_classifier import create_app


def _make_config(model_name, model_params, model_path):
    return {
        "data": {"path": "unused", "target_column": "species"},
        "model": {"name": model_name, "params": model_params},
        "training": {"model_path": model_path},
    }


def _train_and_save_mlp(iris_tiny, tmp_dir):
    model = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3)
    model.train(train_data=iris_tiny, tracking=False, max_epochs=5)
    model.save(f"{tmp_dir}/model")
    return _make_config("mlp_classifier", {"input_dim": 4, "hidden_dim": 8, "num_classes": 3}, f"{tmp_dir}/model")


def _train_and_save_gb(iris_tiny, tmp_dir):
    model = GBClassifier(n_estimators=10, max_depth=2)
    model.train(train_data=iris_tiny, tracking=False)
    model.save(f"{tmp_dir}/model")
    return _make_config("gb_classifier", {"n_estimators": 10, "max_depth": 2}, f"{tmp_dir}/model")


class TestHealth:
    def test_health(self, iris_tiny):
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestInfo:
    def test_info(self, iris_tiny):
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.get("/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_name"] == "mlp_classifier"
            assert data["feature_names"] == ["f0", "f1", "f2", "f3"]
            assert data["class_names"] == ["a", "b", "c"]
            assert data["model_params"]["input_dim"] == 4


class TestPredict:
    def test_predict_mlp(self, iris_tiny):
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.post("/predict", json={"features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]})
            assert resp.status_code == 200
            preds = resp.json()["predictions"]
            assert len(preds) == 2
            assert len(preds[0]) == 3  # num_classes=3, MLP returns (n, 3)

    def test_predict_gb(self, iris_tiny):
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_gb(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.post("/predict", json={"features": [[1.0, 2.0, 3.0, 4.0]]})
            assert resp.status_code == 200
            preds = resp.json()["predictions"]
            assert len(preds) == 1
            assert len(preds[0]) == 1  # GB returns (n,) reshaped to (n, 1)

    def test_predict_invalid_features(self, iris_tiny):
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.post("/predict", json={"features": [[1.0, 2.0]]})
            assert resp.status_code == 422
            assert "expected 4" in resp.json()["detail"]
