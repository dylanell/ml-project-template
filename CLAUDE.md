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

# Train a model from config
uv run python scripts/train.py --config configs/iris_mlp.json
uv run python scripts/train.py --config configs/iris_gb.json

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
│   ├── base.py              # BaseModel ABC + BasePytorchModel (MLflow, Fabric, save/load/predict)
│   ├── registry.py          # ModelRegistry for model discovery
│   ├── gb_classifier.py     # Sklearn GradientBoosting wrapper
│   └── mlp_classifier.py    # PyTorch MLP (MLP nn.Module + MLPClassifier)

configs/                     # Training configs (JSON)
.data/                       # Raw datasets (gitignored)
.models/                     # Saved model artifacts (gitignored)
scripts/                     # Data onboarding + training scripts
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
model = ModelRegistry.get("mlp_classifier")(input_dim=4, hidden_dim=16, num_classes=3)
```

### Training with MLflow
```python
# BaseModel.train() handles MLflow orchestration (params, artifacts)
# Model-specific training kwargs are forwarded to _fit()
# Training params are logged manually inside _fit()
model.train(
    experiment_name="my-experiment",
    train_data=train_data,
    val_data=val_data,
    model_path=".models/my_model",
    run_name="run-1",  # optional
    # Model-specific training kwargs (e.g. for MLP):
    lr=1e-3,
    max_epochs=100,
    batch_size=32,
)
```

### Adding New Models
1. Create `src/ml_project_template/models/my_model.py` extending `BaseModel` (or `BasePytorchModel` for PyTorch)
2. Implement `_fit()` (and `get_params()` only if automatic capture doesn't work — see below)
3. Register in `registry.py`

### BaseModel Interface
```python
class BaseModel(ABC):
    # Public API — MLflow orchestration (set experiment, start run, log params, save artifact)
    def train(self, *, experiment_name, train_data, **kwargs) -> None

    # Subclasses implement — model-specific training logic
    def _fit(self, train_data, val_data=None, **kwargs) -> None

    # Abstract — subclasses must implement
    def predict(self, X: np.ndarray) -> np.ndarray
    def save(self, path: str) -> str
    def load(self, path: str) -> None

    # Auto-populated from __init__ args via __init_subclass__ — no override needed
    # Override only if automatic capture is insufficient (e.g. sklearn **kwargs)
    def get_params(self) -> dict
```

### BasePytorchModel
Extends `BaseModel` with Lightning Fabric and shared PyTorch boilerplate:
- `predict()` — numpy→tensor→device→inference→cpu→numpy
- `save()`/`load()` — Fabric-based state dict persistence
- `self.fabric` — initialized from constructor args (accelerator, devices, precision, etc.)

Subclasses only need to implement `_fit()` and set `self.model`.

### Automatic `__init__` Param Capture
`BaseModel` uses `__init_subclass__` to automatically record all `__init__` arguments
into `self._model_params`. Works across the inheritance chain (e.g. Fabric args from
`BasePytorchModel` + architecture args from `MLPClassifier`). Logged to MLflow
automatically in `train()`. Override `get_params()` only when needed (e.g. sklearn
models where `**kwargs` doesn't capture individual params with defaults).

## Conventions

- **Notebooks run from project root** - VS Code setting `jupyter.notebookFileRoot` is set
- **Data in `.data/`** - Raw datasets, gitignored
- **Models in `.models/`** - Saved artifacts, gitignored
- **PyTorch models**: Separate `nn.Module` class from `BasePytorchModel` wrapper in same file
- **Sklearn models**: Override `get_params()` to delegate to sklearn's introspection
- **NumPy I/O**: All models return raw numpy output from `predict()` — caller handles post-processing (argmax, etc.)
- **Training params**: Logged manually inside `_fit()`, not auto-captured (only `__init__` args are auto-captured)

## Environment Variables

- `MLFLOW_TRACKING_URI` - MLflow server URL (default: local `./mlruns`)
