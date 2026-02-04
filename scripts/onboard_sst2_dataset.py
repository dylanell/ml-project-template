"""Download the SST-2 sentiment dataset to data/sst2/."""

import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"


def main():
    data_dir = Path(".data/sst2")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "SST-2.zip"

    # Download
    print(f"Downloading from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # Clean up zip
    zip_path.unlink()

    # Move files up from SST-2/ subdirectory
    extracted_dir = data_dir / "SST-2"
    for file in extracted_dir.iterdir():
        file.rename(data_dir / file.name)
    extracted_dir.rmdir()

    print(f"Saved to {data_dir}/")
    print("Files:")
    for file in sorted(data_dir.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()
