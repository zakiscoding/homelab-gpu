"""Ensure fer2013.csv exists (required for FER+ aligned labels)."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
OUT = DATA / "fer2013" / "fer2013.csv"


def _from_kagglehub() -> bool:
    try:
        import kagglehub
    except ImportError:
        return False
    path = Path(kagglehub.dataset_download("msambare/fer2013"))
    found = list(path.rglob("fer2013.csv")) or list(path.rglob("fer2013-new.csv"))
    if not found:
        return False
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found[0], OUT)
    print("copied", found[0], "->", OUT)
    return True


def _from_kaggle_cli() -> bool:
    import subprocess

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.parent / "_kaggle_dl"
    if tmp.is_dir():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "msambare/fer2013", "-p", str(tmp), "--unzip"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return False
    found = list(tmp.rglob("fer2013.csv"))
    if not found:
        return False
    shutil.copy2(found[0], OUT)
    shutil.rmtree(tmp, ignore_errors=True)
    print("kaggle cli ->", OUT)
    return True


def main() -> None:
    if OUT.is_file() and OUT.stat().st_size > 1_000_000:
        print("exists", OUT)
        return
    if _from_kagglehub():
        return
    if _from_kaggle_cli():
        return
    raise FileNotFoundError(
        f"Could not obtain fer2013.csv at {OUT}. "
        "Install kagglehub (`pip install kagglehub`) or place the CSV manually."
    )


if __name__ == "__main__":
    main()
