"""Config-driven training script.

Usage:
    uv run python scripts/train.py --config configs/iris_mlp.json
"""

import argparse
import json
import sys

from ml_project_template.data import Dataset
from ml_project_template.models import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="Train a model from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[train.py] Training with config:")
    print(f"{json.dumps(config, indent=2)}")

    # Validate required top-level keys
    for key in ("data", "model", "training"):
        if key not in config:
            print(f"[train.py] Error: config missing required key '{key}'")
            sys.exit(1)

    # Load and split data
    data_cfg = config["data"]
    dataset = Dataset.from_csv(data_cfg["path"], target_column=data_cfg["target_column"])
    train_data, val_data = dataset.split(
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )

    # Create model
    model_cfg = config["model"]
    model = ModelRegistry.get(model_cfg["name"])(**model_cfg.get("params", {}))

    # Build training args — shallow copy so we don't mutate the loaded config
    train_cfg = dict(config["training"])
    experiment_name = train_cfg.pop("experiment_name")
    run_name = train_cfg.pop("run_name", None)
    model_path = train_cfg.pop("model_path", None)

    # Everything remaining is model-specific training kwargs
    model.train(
        experiment_name=experiment_name,
        train_data=train_data,
        val_data=val_data,
        run_name=run_name,
        model_path=model_path,
        **train_cfg,
    )

    print(f"[train.py] Training complete.")


if __name__ == "__main__":
    main()
