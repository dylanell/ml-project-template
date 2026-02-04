# ML Project Template

Template for machine learning (ML) projects and API serving. 

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
