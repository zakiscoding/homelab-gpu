"""Export fer2013.csv from Hugging Face for FER+ aligned training."""

from __future__ import annotations

import csv
import os
from pathlib import Path

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
OUT = DATA / "fer2013" / "fer2013.csv"

USAGE_MAP = {"train": "Training", "test": "PublicTest", "validation": "PrivateTest"}


def main() -> None:
    if OUT.is_file() and OUT.stat().st_size > 1_000_000:
        print("exists", OUT)
        return
    from datasets import load_dataset

    print("loading HF fer2013...")
    ds = load_dataset("Abirate/fer2013")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "pixels", "Usage"])
        for split, rows in ds.items():
            usage = USAGE_MAP.get(split, split)
            for row in rows:
                emotion = row.get("label", row.get("emotion"))
                pixels = row.get("pixels", "")
                if isinstance(pixels, list):
                    pixels = " ".join(str(p) for p in pixels)
                w.writerow([emotion, pixels, usage])
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
