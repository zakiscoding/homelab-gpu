"""Download FER2013 train/test CSVs (torchvision 0.26 resources)."""

from __future__ import annotations

import os
from pathlib import Path

from torchvision.datasets.utils import download_url

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
BASE = DATA / "fer2013"

# torchvision.datasets.fer2013.FER2013._RESOURCES
FILES = {
    "train.csv": ("https://www.dropbox.com/s/opuvvdv5uligypx/train.csv?dl=1", "3f0dfb3d3fd99c811a1299cb947e3131"),
    "test.csv": ("https://www.dropbox.com/s/opuvvdv5uligypx/train.csv?dl=1", "b02c2298636a634e8c2faabbf3ea9a23"),
}


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    # Official URLs from torchvision — use their helper when possible
    from torchvision.datasets import fer2013 as fer_mod

    resources = fer_mod.FER2013._RESOURCES
    for split, (fname, md5) in resources.items():
        if split in ("fer", "icml"):
            continue
        url = fer_mod.FER2013._URL if hasattr(fer_mod.FER2013, "_URL") else None
        dest = BASE / fname
        if dest.is_file():
            print("exists", dest)
            continue
        # torchvision 0.26 uses download_and_extract_archive in __init__
        print("need", fname, "— place manually or use Kaggle fer2013.csv")


if __name__ == "__main__":
    main()
