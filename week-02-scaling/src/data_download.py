"""
Download the TinyStories validation split (~19.4MB, plain text) if it isn't already present.

Why TinyStories (not week-1's TinyShakespeare): week 2 is about measuring a *scaling law*, which
needs enough data that even the biggest (~1M param) model here isn't just memorizing the whole
dataset in a few epochs. TinyStories is a synthetic dataset of very simple short stories (GPT-
generated, restricted vocabulary) built specifically for training tiny language models, which
matches this week's tiny-model regime much better than Shakespeare's denser, more idiosyncratic
prose.

Why the "-valid" split specifically (not the ~1.9GB full train split): the validation split is
already a stand-alone ~19MB file, which is plenty of characters for 0.1M-1M parameter models
(see notes.md for the actual character budget used per run) and downloads/trains fast on a CPU-
only machine - the full train split would be ~100x more data than these tiny models can usefully
soak up in a few hours of CPU time. We further split *this* file into our own train/val partition
below (same 90/10 contiguous-split pattern as week-1's train.py), so "validation" in the scaling-
law results refers to our own held-out slice of this file, not HuggingFace's split name.

Fetched directly via urllib (stdlib only), matching week-1's dependency-light approach (no
`datasets` / `huggingface_hub` library needed for a single plain-text file).
"""

import urllib.request
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "TinyStories-valid.txt"


def download_data(data_file: Path = DATA_FILE, url: str = DATA_URL) -> Path:
    """Return the path to the dataset, downloading it first if it's not already there."""
    if data_file.exists():
        print(f"Data already present at {data_file}, skipping download.")
        return data_file

    print(f"Downloading TinyStories-valid to {data_file} ...")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, data_file)
    print(f"Downloaded {data_file.stat().st_size:,} bytes.")
    return data_file


if __name__ == "__main__":
    download_data()
