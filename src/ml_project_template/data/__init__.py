"""Data loading and preprocessing."""

from ml_project_template.data.base import BaseDataset
from ml_project_template.data.tabular import TabularDataset

# Backwards compatibility alias
Dataset = TabularDataset

__all__ = ["BaseDataset", "TabularDataset", "Dataset"]
