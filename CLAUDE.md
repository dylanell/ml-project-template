# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project template for model R&D and API serving. Uses uv for package management with Python 3.11+.

## Commands

```bash
# Setup
uv venv && uv pip install -e "."

# Run scripts/notebooks
uv run python scripts/onboard_iris_dataset.py
uv run jupyter notebook

# Start MLflow server (for experiment tracking UI)
mlflow server --host 127.0.0.1 --port 5000
```

## Architecture

```
src/ml_project_template/
├── data/                    # Dataset abstractions
│   ├── base.py              # BaseDataset ABC
│   └── tabular.py           # TabularDataset for numerical data
├── models/                  # Model implementations
│   ├── base.py              # BaseModel ABC
│   ├── registry.py          # ModelRegistry for model discovery
│   ├── gb_classifier.py     # Sklearn GradientBoosting wrapper
│   └── mlp_classifier.py    # PyTorch MLP (MLPModule + MLPClassifier)
└── training.py              # Trainer class with MLflow integration

.data/                       # Raw datasets (gitignored)
.models/                     # Saved model artifacts (gitignored)
scripts/                     # Data onboarding scripts
notebooks/                   # R&D notebooks
```

## Key Patterns

### Data Loading
```python
from ml_project_template.data import Dataset  # Alias for TabularDataset
dataset = Dataset.from_csv(".data/iris/iris.csv", target_column="species")
train_data, test_data = dataset.split(test_size=0.2, random_state=42)
```

### Model Registry
```python
from ml_project_template.models import ModelRegistry
ModelRegistry.list()  # ['gb_classifier', 'mlp_classifier']
model = ModelRegistry.get("gb_classifier")(n_estimators=100, max_depth=3)
```

### Training with MLflow
```python
from ml_project_template.training import Trainer
trainer = Trainer(model=model, experiment_name="my-experiment")
metrics = trainer.run(
    train_data=train_data,
    test_data=test_data,
    model_path=".models/my_model",
    run_name="run-1",  # optional
)
```
- `Trainer.run()` manages MLflow run lifecycle internally
- Logs params via `model.get_params()`, metrics, and artifacts automatically

### Adding New Models
1. Create `src/ml_project_template/models/my_model.py` extending `BaseModel`
2. Implement: `train()`, `predict()`, `save()`, `load()`, `get_params()`
3. Register in `registry.py`

### BaseModel Interface
```python
class BaseModel(ABC):
    def train(self, dataset: Dataset) -> None
    def predict(self, X: np.ndarray) -> np.ndarray
    def save(self, path: str) -> str  # Returns actual path with extension
    def load(self, path: str) -> None
    def get_params(self) -> dict  # For MLflow param logging
```

## Conventions

- **Notebooks run from project root** - VS Code setting `jupyter.notebookFileRoot` is set
- **Data in `.data/`** - Raw datasets, gitignored
- **Models in `.models/`** - Saved artifacts, gitignored
- **PyTorch models**: Separate `nn.Module` class from `BaseModel` wrapper in same file
- **Sklearn models**: Delegate to underlying model's `get_params()` for param logging
- **NumPy I/O**: All models use numpy arrays at interface, convert internally (e.g., PyTorch tensors)

## Environment Variables

- `MLFLOW_TRACKING_URI` - MLflow server URL (default: local `./mlruns`)
