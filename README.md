# ML Project Template

Template for machine learning (ML) projects and API serving. 

## Using This Template

To start a new project from this template, rename the package directory and update all references:

1. Rename `src/ml_project_template/` to `src/<your_project_name>/`
2. Update `pyproject.toml`:
   - `name` (line 2) — the installable package name (use hyphens, e.g. `my-project`)
   - `[project.scripts]` entry — the CLI name and module path
3. Find and replace `ml_project_template` with `your_project_name` in all Python files (imports)
4. Update `CLAUDE.md` and this `README.md`

Then reinstall:
```bash
uv pip install -e "."
```

## Setup

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e "." --group dev
```

Configure VSCode notebooks (add to .vscode/settings.json)
```
{                                                                                                                                                           
  "jupyter.notebookFileRoot": "${workspaceFolder}"                                                                                                          
}
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
├── utils/
│   └── io.py                # S3-compatible I/O utilities (get_storage_options, get_s3_filesystem)

configs/                     # Training configs (JSON)
docker/                                  # Dockerfiles per pipeline stage
├── preprocess-iris-dataset/Dockerfile   # Preprocessing image
└── train-iris-classifier/Dockerfile     # Training image
argo/                        # Argo Workflow pipelines
scripts/                     # Data onboarding, preprocessing + training scripts
notebooks/                   # R&D notebooks
tests/                       # Test suite (no external services needed)
```

## Key Patterns

### Data Loading
```python
from ml_project_template.data import Dataset  # Alias for TabularDataset
from ml_project_template.utils import get_storage_options

dataset = Dataset.from_csv("s3://data/iris/iris.csv", target_column="species", storage_options=get_storage_options())
train_data, test_data = dataset.split(test_size=0.2, random_state=42)
```

### Model Registry
```python
from ml_project_template.models import ModelRegistry
ModelRegistry.list()  # ['gb_classifier', 'mlp_classifier']
model = ModelRegistry.get("mlp_classifier")(input_dim=4, hidden_dim=16, num_classes=3)
```

### Training
```python
# BaseModel.train() handles MLflow orchestration (params, artifacts)
# Model-specific training kwargs are forwarded to _fit()
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

# Or train without MLflow tracking for quick iteration
model.train(train_data=train_data, tracking=False, max_epochs=10)
```

### Adding New Models
1. Create `src/ml_project_template/models/my_model.py` extending `BaseModel` (or `BasePytorchModel` for PyTorch)
2. Implement `_fit()` (and `get_params()` only if automatic capture doesn't work — see below)
3. Register in `registry.py`
4. Add lifecycle and get_params tests in `tests/test_models.py`

### BaseModel Interface
```python
class BaseModel(ABC):
    # Public API — MLflow orchestration (set experiment, start run, log params, save artifact)
    def train(self, *, experiment_name, train_data, tracking=True, **kwargs) -> None

    # Subclasses implement — model-specific training logic
    def _fit(self, train_data, val_data=None, **kwargs) -> None

    # Abstract — subclasses must implement
    def predict(self, X: np.ndarray) -> np.ndarray
    def save(self, path: str) -> str
    def load(self, path: str) -> None

    # Log to MLflow (no-op when tracking=False) — use in _fit() instead of mlflow directly
    def log_param(self, key, value) -> None
    def log_metric(self, key, value, step=None) -> None

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
`BaseModel` uses `__init_subclass__` to automatically record all `__init__` arguments into `self._model_params`. Works across the inheritance chain (e.g. Fabric args from `BasePytorchModel` + architecture args from `MLPClassifier`). Logged to MLflow automatically in `train()`. Override `get_params()` only when needed (e.g. sklearn models where `**kwargs` doesn't capture individual params with defaults).

## Quick Start

### Local Services (MinIO + MLflow)

Start MinIO (S3-compatible storage) and MLflow (experiment tracking) via Docker Compose:

```bash
docker compose up -d
```

- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **MinIO Console:** [http://localhost:7001](http://localhost:7001) (login: `minioadmin`/`minioadmin`)

Create `data` and `models` buckets in the MinIO console, then onboard data:

```bash
uv run python scripts/onboard_iris_dataset.py --dest s3://data/iris/
```

Stop services with `docker compose down`. Data persists in Docker volumes across restarts.

### Docker

```bash
# Build images
docker build -t preprocessing-job -f docker/preprocess-iris-dataset/Dockerfile .
docker build -t training-job -f docker/train-iris-classifier/Dockerfile .

# Run preprocessing (reads/writes data via S3)
docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -v $(pwd)/configs:/app/configs \
  preprocessing-job --config configs/iris_mlp_classifier.json

# Run training (reads data via S3, saves model to S3, logs to MLflow)
docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -v $(pwd)/configs:/app/configs \
  training-job --config configs/iris_mlp_classifier.json
```

> `--env-file .env` loads S3 and MLflow credentials. The `-e` flags override the endpoint URLs to use `host.docker.internal`, which resolves to the host machine from inside Docker containers (Mac/Windows). On Linux, add `--add-host=host.docker.internal:host-gateway` to the `docker run` command.

### Argo Workflows (Local)

Run the full pipeline (preprocess → train) as an Argo Workflow DAG on Docker Desktop's K8s cluster. See [argo/README.md](argo/README.md) for details.

```bash
# First-time: install Argo Workflows
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml

# First-time: create secret and config map in argo namespace
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/

# Submit a pipeline (uses MLP config by default)
argo submit -n argo argo/iris-classifier-pipeline.yaml --watch

# Or specify a different config
argo submit -n argo argo/iris-classifier-pipeline.yaml -p config=configs/iris_gb_classifier.json --watch

# Argo UI (optional)
kubectl port-forward -n argo svc/argo-server 2746:2746
# Then open https://localhost:2746
```

To update configs after changing them locally:

```bash
kubectl create configmap training-configs --namespace argo --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Testing

Run the test suite with:

```bash
uv run pytest tests/ -v
```

Tests use `tracking=False` to skip MLflow, so no external services (MLflow, MinIO) are needed. They run entirely from in-memory numpy data.

When adding a new model, add matching tests in `tests/test_models.py` following the existing pattern:
- **`test_lifecycle`** — create, train (with `tracking=False`), predict (check output shape), save to a temp dir, load into a fresh instance, predict again (outputs match)
- **`test_get_params`** — verify `get_params()` returns the expected keys and values

## ToDo

- [x] **Add tests.** Test suite in `tests/` covers model lifecycle, registry, and dataset operations.
- [ ] **Save model metadata alongside artifacts.** Loading a model currently requires knowing the exact init args. Save the params dict next to the weights so `load()` can reconstruct the model without the caller needing to know the architecture.
- [x] **Decouple MLflow from the training path.** `train(tracking=False)` skips all MLflow calls. Models use `self.log_param()`/`self.log_metric()` helpers that respect the flag.
- [ ] **Move BasePytorchModel to its own file.** `base.py` imports torch and lightning unconditionally, so even sklearn-only usage pays for the full PyTorch import chain. Splitting BasePytorchModel out would fix this.
- [ ] **Document the `__init_subclass__` param capture for onboarding.** The auto-capture of `_model_params` is non-obvious to newcomers. Add an explanation to the contributing guide or model-authoring docs.
- [ ] **Add central seed management for reproducibility.** No way to set random seeds (torch, numpy, python) from a central place. "Can't reproduce your result" will be a recurring team issue without this.
- [ ] **Consider decorator-based model registration.** Currently adding a model requires editing two files (model file + registry.py), which causes merge conflicts when multiple people add models simultaneously.
- [ ] **Add API serving.** FastAPI server for model inference. Load a saved model by name, expose a `/predict` endpoint, containerize for deployment.
- [ ] **Support regression tasks in Dataset.** `y` is always cast to `long()` and `LabelEncoder` is built in, so the data layer is classification-only. Regression would need a new dataset class or modifications.
