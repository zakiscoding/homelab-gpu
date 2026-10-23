"""Download FER2013 via torchvision (train.csv + test.csv)."""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))


def main() -> None:
    from torchvision.datasets import FER2013

    root = DATA
    root.mkdir(parents=True, exist_ok=True)
    FER2013(root=str(root), split="train")
    FER2013(root=str(root), split="test")
    base = root / "fer2013"
    print("FER2013 ready under", base)
    for name in ("fer2013.csv", "train.csv", "test.csv"):
        p = base / name
        if p.is_file():
            print(" ", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
