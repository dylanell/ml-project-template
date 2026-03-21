"""Download the Iris dataset to a local directory or S3 prefix.

Usage:
    uv run python scripts/onboard.py --dest .data/iris/raw
    uv run python scripts/onboard.py --dest s3://data/iris/raw
"""

import argparse
import logging
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from ml_project_template.utils import get_s3_filesystem

load_dotenv()

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
FILENAME = "iris.csv"
_SCRIPT = "onboard.py"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download the Iris dataset.")
    parser.add_argument(
        "--dest",
        default=".data/iris/raw",
        help="Destination directory (local or s3://)",
    )
    args = parser.parse_args()

    # Download raw data
    logger.info(f"[{_SCRIPT}] Downloading from '{DATA_URL}'")
    response = urllib.request.urlopen(DATA_URL)
    raw_data = response.read().decode("utf-8").strip()
    csv_content = ",".join(COLUMN_NAMES) + "\n" + raw_data + "\n"

    # Write to destination
    dest = args.dest.rstrip("/")
    file_path = f"{dest}/{FILENAME}"

    if dest.startswith("s3://"):
        fs = get_s3_filesystem()
        with fs.open(file_path, "w") as fp:
            fp.write(csv_content)
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fp:
            fp.write(csv_content)

    logger.info(f"[{_SCRIPT}] Saved {FILENAME} to {dest}/")


if __name__ == "__main__":
    main()
