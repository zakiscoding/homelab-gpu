"""Download FER+ label file (fer2013new.csv) from Microsoft FERPlus repo."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
OUT = DATA / "fer2013" / "fer2013new.csv"
URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size > 1000:
        print("exists", OUT)
        return
    print("downloading", URL)
    urllib.request.urlretrieve(URL, OUT)
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
