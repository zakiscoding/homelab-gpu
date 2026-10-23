"""Download FER2013 via torchvision into datasets/fer2013/fer2013.csv if missing."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
OUT = DATA / "fer2013" / "fer2013.csv"


def main() -> None:
    if OUT.is_file() and OUT.stat().st_size > 1_000_000:
        print("exists", OUT)
        return
    from torchvision.datasets import FER2013

    root = DATA
    root.mkdir(parents=True, exist_ok=True)
    print("downloading FER2013 via torchvision...")
    FER2013(str(root), split="train")
    FER2013(str(root), split="test")
    base = DATA / "fer2013"
    for name in ("fer2013.csv", "train.csv", "test.csv"):
        found = list(base.rglob(name)) if base.is_dir() else []
        if not found:
            found = list(root.rglob(name))
        for p in found:
            if p.name == "fer2013.csv" or (name == "fer2013.csv" and p.suffix == ".csv"):
                shutil.copy2(p, OUT)
                print("copied", p, "->", OUT)
                return
    # torchvision may place icml_face_data.csv — search
    for p in root.rglob("*.csv"):
        if p.stat().st_size > 5_000_000:
            shutil.copy2(p, OUT)
            print("copied large csv", p, "->", OUT)
            return
    raise FileNotFoundError(f"No fer2013.csv found under {root}")


if __name__ == "__main__":
    main()
