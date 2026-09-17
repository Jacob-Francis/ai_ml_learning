"""
Download the TinyShakespeare dataset (~1MB, plain text) if it isn't already present.

Source: Karpathy's char-rnn repo - this is the same file the HuggingFace
`karpathy/tiny_shakespeare` dataset page links back to. Fetched directly via
urllib (stdlib only) rather than the `datasets` library, per the no-HuggingFace
constraint in main.py's docstring.
"""

import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "input.txt"


def download_data(data_file: Path = DATA_FILE, url: str = DATA_URL) -> Path:
    """Return the path to the dataset, downloading it first if it's not already there."""
    if data_file.exists():
        print(f"Data already present at {data_file}, skipping download.")
        return data_file

    print(f"Downloading TinyShakespeare to {data_file} ...")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, data_file)
    print(f"Downloaded {data_file.stat().st_size:,} bytes.")
    return data_file


if __name__ == "__main__":
    download_data()
