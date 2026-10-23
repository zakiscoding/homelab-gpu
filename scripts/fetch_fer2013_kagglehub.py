"""Download FER2013 via kagglehub and copy fer2013.csv into datasets/fer2013/."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
OUT = DATA / "fer2013" / "fer2013.csv"


def main() -> None:
    if OUT.is_file():
        print("exists", OUT)
        return
    import kagglehub

    path = Path(kagglehub.dataset_download("msambare/fer2013"))
    print("downloaded to", path)
    found = list(path.rglob("fer2013.csv")) or list(path.rglob("fer2013-new.csv"))
    dest_root = DATA / "fer2013"
    dest_root.mkdir(parents=True, exist_ok=True)
    if found:
        shutil.copy2(found[0], OUT)
        print("copied csv to", OUT)
        return
    train_src = path / "train"
    test_src = path / "test"
    if train_src.is_dir() and test_src.is_dir():
        for split, src in (("train", train_src), ("test", test_src)):
            dst = dest_root / split
            if dst.is_dir():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        print("copied ImageFolder layout to", dest_root)
        return
    raise FileNotFoundError(f"Unrecognized FER2013 layout under {path}")


if __name__ == "__main__":
    main()
