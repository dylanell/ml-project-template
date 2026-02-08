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

## Project Structure

```

```

## Quick Start

### Run MLFlow Locally

Run the following in a terminal then navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

```
uv run mlflow server --host 127.0.0.1 --port 5000
```

### Training

```python

```

### Inference

```python

```

### API Server

```

```

### Testing

## ToDo

- [ ] **Add tests.** No tests exist to enforce the BaseModel contract. A small suite that exercises create, train, predict, save, load, predict-again for each model would catch most regressions.
- [ ] **Save model metadata alongside artifacts.** Loading a model currently requires knowing the exact init args. Save the params dict next to the weights so `load()` can reconstruct the model without the caller needing to know the architecture.
- [ ] **Decouple MLflow from the training path.** `train()` always starts an MLflow run. Add a way to train without tracking (e.g. a `tracking=False` flag or a separate `fit()` public method) for quick iteration and debugging.
- [ ] **Move BasePytorchModel to its own file.** `base.py` imports torch and lightning unconditionally, so even sklearn-only usage pays for the full PyTorch import chain. Splitting BasePytorchModel out would fix this.
- [ ] **Document the `__init_subclass__` param capture for onboarding.** The auto-capture of `_model_params` is non-obvious to newcomers. Add an explanation to the contributing guide or model-authoring docs.
- [ ] **Add central seed management for reproducibility.** No way to set random seeds (torch, numpy, python) from a central place. "Can't reproduce your result" will be a recurring team issue without this.
- [ ] **Consider decorator-based model registration.** Currently adding a model requires editing two files (model file + registry.py), which causes merge conflicts when multiple people add models simultaneously.
- [ ] **Support regression tasks in Dataset.** `y` is always cast to `long()` and `LabelEncoder` is built in, so the data layer is classification-only. Regression would need a new dataset class or modifications.
