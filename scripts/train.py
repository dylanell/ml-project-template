"""Config-driven training script for Iris classifiers.

Usage:
    uv run python scripts/train.py --config configs/local/iris_mlp_classifier.json
    uv run python scripts/train.py --config configs/local/iris_gb_classifier.json
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from ml_project_template.data import TabularDataset
from ml_project_template.models import ModelRegistry
from ml_project_template.utils import seed_everything

load_dotenv()

_SCRIPT = "train.py"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train an Iris classifier from a JSON config."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logger.info(f"[{_SCRIPT}] Running with config:")
    logger.info(json.dumps(config, indent=2))

    # Validate required top-level keys
    for key in ("data", "model", "training"):
        if key not in config:
            logger.error(f"[{_SCRIPT}] Error: config missing required key '{key}'")
            sys.exit(1)

    seed = config.get("seed")

    # Load preprocessed train/valid CSVs from preprocessing.output_dir
    data_cfg = config["data"]
    preprocess_cfg = config.get("preprocessing", {})
    output_dir = preprocess_cfg.get("output_dir", "")
    target_col = data_cfg["target_column"]

    if not output_dir:
        logger.error(f"[{_SCRIPT}] Error: config missing preprocessing.output_dir")
        sys.exit(1)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "valid.csv")

    logger.info(f"[{_SCRIPT}] Loading train data from {train_path}")
    logger.info(f"[{_SCRIPT}] Loading val data from {val_path}")

    train_data = TabularDataset(csv_path=train_path, target_col=target_col)
    val_data = TabularDataset(csv_path=val_path, target_col=target_col)

    # Seed immediately before model construction so weight init is reproducible
    if seed is not None:
        seed_everything(seed)

    # Create model
    model_cfg = config["model"]
    model = ModelRegistry.get(model_cfg["name"])(**model_cfg.get("params", {}))

    # Build training args — shallow copy so we don't mutate the loaded config
    train_cfg = dict(config["training"])
    experiment_name = train_cfg.pop("experiment_name")
    run_name = train_cfg.pop("run_name", None)
    model_path = train_cfg.pop("model_path")

    # Flatten data + preprocessing config for MLflow logging
    extra_params = {f"data.{k}": v for k, v in data_cfg.items()}
    extra_params.update({f"preprocessing.{k}": v for k, v in preprocess_cfg.items()})

    model.train(
        experiment_name=experiment_name,
        train_data=train_data,
        val_data=val_data,
        run_name=run_name,
        model_path=model_path,
        extra_params=extra_params,
        **train_cfg,
    )

    logger.info(f"[{_SCRIPT}] Training complete.")


if __name__ == "__main__":
    main()
