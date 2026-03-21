"""Config-driven preprocessing script for the Iris dataset.

Scales feature columns and writes a processed CSV for downstream training.
Supports "standardize" (zero mean, unit variance) and "min-max" (0-1 range).

Usage:
    uv run python scripts/preprocess.py --config configs/local/iris_mlp_classifier.json
"""

import argparse
import logging
import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from ml_project_template.utils import get_storage_options, seed_everything

load_dotenv()

_SCRIPT = "preprocess.py"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Iris dataset from a JSON config."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logger.info(f"[{_SCRIPT}] Running with config:")
    logger.info(json.dumps(config, indent=2))

    # Validate required keys
    for key in ("data", "preprocessing"):
        if key not in config:
            logger.error(f"[{_SCRIPT}] Error: config missing required key '{key}'")
            sys.exit(1)

    # Seed for reproducible preprocessing
    seed = config.get("seed")
    if seed is not None:
        seed_everything(seed)

    data_cfg = config["data"]
    preprocess_cfg = config["preprocessing"]

    # Load raw data
    raw_path = data_cfg["path"]
    storage_options = get_storage_options(raw_path)
    target_column = data_cfg["target_column"]
    logger.info(f"[{_SCRIPT}] Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path, storage_options=storage_options)
    logger.info(f"[{_SCRIPT}] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Identify feature columns (everything except target)
    feature_cols = [c for c in df.columns if c != target_column]
    logger.info(f"[{_SCRIPT}] Feature columns: {feature_cols}")
    logger.info(f"[{_SCRIPT}] Target column: {target_column}")

    # Split data
    val_size = data_cfg["valid_size"]
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)

    # Scale features
    scaling = preprocess_cfg.get("scaling", "standardize")
    valid_methods = ("standardize", "min-max")
    if scaling not in valid_methods:
        logger.error(
            f"[{_SCRIPT}] Error: unknown scaling method '{scaling}', "
            f"must be one of {valid_methods}"
        )
        sys.exit(1)

    logger.info(f"\n[{_SCRIPT}] Scaling method: {scaling}")

    features = train_df[feature_cols].values.astype(np.float64)
    if scaling == "standardize":
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        train_df[feature_cols] = (train_df[feature_cols] - means) / stds
        val_df[feature_cols] = (val_df[feature_cols] - means) / stds
    elif scaling == "min-max":
        mins = features.min(axis=0)
        maxs = features.max(axis=0)
        train_df[feature_cols] = (train_df[feature_cols] - mins) / (maxs - mins)
        val_df[feature_cols] = (val_df[feature_cols] - mins) / (maxs - mins)

    # Write processed CSVs
    output_dir = preprocess_cfg["output_dir"]
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(
        os.path.join(output_dir, "train.csv"),
        index=False,
        storage_options=get_storage_options(output_dir),
    )
    val_df.to_csv(
        os.path.join(output_dir, "valid.csv"),
        index=False,
        storage_options=get_storage_options(output_dir),
    )
    logger.info(f"[{_SCRIPT}] Wrote processed data to {output_dir}")
    logger.info(f"[{_SCRIPT}] Done.")


if __name__ == "__main__":
    main()
