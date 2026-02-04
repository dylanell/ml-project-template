"""Download the Iris dataset to data/iris/."""

import urllib.request
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

def main():
    # Setup paths
    data_dir = Path(".data/iris")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "iris.csv"

    # Download raw data
    print(f"Downloading from '{DATA_URL}'")
    response = urllib.request.urlopen(DATA_URL)
    raw_data = response.read().decode("utf-8").strip()

    # Write CSV with header
    with open(output_path, "w") as fp:
        fp.write(",".join(COLUMN_NAMES) + "\n")
        fp.write(raw_data + "\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
