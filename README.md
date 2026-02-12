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
uv pip install -e ".[all]"
```

Install only what you need:
```bash
uv pip install -e ".[dev]"        # Development tools
uv pip install -e ".[training]"   # MLflow, tensorboard
uv pip install -e ".[api]"        # FastAPI serving
```

Configure VSCode notebooks (add to .vscode/settings.json)
```
{                                                                                                                                                           
  "jupyter.notebookFileRoot": "${workspaceFolder}"                                                                                                          
}
```

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

## ToDo

- [ ] **Add tests.** No tests exist to enforce the BaseModel contract. A small suite that exercises create, train, predict, save, load, predict-again for each model would catch most regressions.
- [ ] **Save model metadata alongside artifacts.** Loading a model currently requires knowing the exact init args. Save the params dict next to the weights so `load()` can reconstruct the model without the caller needing to know the architecture.
- [ ] **Decouple MLflow from the training path.** `train()` always starts an MLflow run. Add a way to train without tracking (e.g. a `tracking=False` flag or a separate `fit()` public method) for quick iteration and debugging.
- [ ] **Move BasePytorchModel to its own file.** `base.py` imports torch and lightning unconditionally, so even sklearn-only usage pays for the full PyTorch import chain. Splitting BasePytorchModel out would fix this.
- [ ] **Document the `__init_subclass__` param capture for onboarding.** The auto-capture of `_model_params` is non-obvious to newcomers. Add an explanation to the contributing guide or model-authoring docs.
- [ ] **Add central seed management for reproducibility.** No way to set random seeds (torch, numpy, python) from a central place. "Can't reproduce your result" will be a recurring team issue without this.
- [ ] **Consider decorator-based model registration.** Currently adding a model requires editing two files (model file + registry.py), which causes merge conflicts when multiple people add models simultaneously.
- [ ] **Add API serving.** FastAPI server for model inference. Load a saved model by name, expose a `/predict` endpoint, containerize for deployment.
- [ ] **Support regression tasks in Dataset.** `y` is always cast to `long()` and `LabelEncoder` is built in, so the data layer is classification-only. Regression would need a new dataset class or modifications.
