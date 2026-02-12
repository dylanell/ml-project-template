"""Tests for model registry."""

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
