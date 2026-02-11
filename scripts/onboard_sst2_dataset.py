"""Download the SST-2 sentiment dataset to a local path or S3 bucket.

Usage:
    uv run python scripts/onboard_sst2_dataset.py
    uv run python scripts/onboard_sst2_dataset.py --dest s3://data/sst2/
"""

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from ml_project_template.utils import get_s3_filesystem

load_dotenv()

DATA_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"


def main():
    parser = argparse.ArgumentParser(description="Download the SST-2 dataset.")
    parser.add_argument("--dest", default=".data/sst2", help="Destination directory (local or s3://)")
    args = parser.parse_args()

    # Download and extract to a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        zip_path = tmp_dir / "SST-2.zip"

        print(f"Downloading from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Collected extracted files (from SST-2/ subdirectory)
        extracted_dir = tmp_dir / "SST-2"
        files = list(extracted_dir.iterdir())

        # Write to destination
        if args.dest.startswith("s3://"):
            fs = get_s3_filesystem()
            dest = args.dest.rstrip("/")
            for file in files:
                s3_path = f"{dest}/{file.name}"
                fs.put(str(file), s3_path)
                print(f"  Uploaded {file.name} → {s3_path}")
        else:
            dest = Path(args.dest)
            dest.mkdir(parents=True, exist_ok=True)
            for file in files:
                file.rename(dest / file.name)
                print(f"  {file.name}")

    print(f"Saved to {args.dest}")


if __name__ == "__main__":
    main()
