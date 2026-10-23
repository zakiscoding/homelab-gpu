"""FER2013 / FER+ datasets — shared by local and distributed reaction training."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))
FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

FERPLUS_MAP = (
    ("neutral", 6),
    ("happiness", 3),
    ("surprise", 5),
    ("sadness", 4),
    ("anger", 0),
    ("disgust", 1),
    ("fear", 2),
    ("contempt", 1),
)


class Fer2013CsvDataset(Dataset):
    def __init__(self, rows: list[tuple[int, str]], transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        label, pixels = self.rows[idx]
        px = [int(p) for p in pixels.split()]
        arr = np.array(px, dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def ferplus_label_from_row(row: dict) -> int | None:
    votes: dict[int, int] = {idx: 0 for _, idx in FERPLUS_MAP}
    for col, idx in FERPLUS_MAP:
        try:
            votes[idx] += int(float(row.get(col, 0) or 0))
        except ValueError:
            pass
    unknown = int(float(row.get("unknown", 0) or 0)) + int(float(row.get("NF", 0) or 0))
    best_idx = max(votes, key=lambda k: votes[k])
    if votes[best_idx] <= unknown:
        return None
    return best_idx


def load_fer2013_rows(label_mode: str = "fer") -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    csv_path = DATA / "fer2013" / "fer2013.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"fer2013.csv missing at {csv_path}. Run scripts/fetch_fer2013_csv.py on the GPU PC."
        )
    ferplus_path = DATA / "fer2013" / "fer2013new.csv"
    use_ferplus = label_mode == "ferplus" and ferplus_path.is_file()
    ferplus_rows: list[dict] = []
    if use_ferplus:
        with ferplus_path.open(newline="", encoding="utf-8") as f:
            ferplus_rows = list(csv.DictReader(f))

    train_rows: list[tuple[int, str]] = []
    val_rows: list[tuple[int, str]] = []
    skipped = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            usage = (row.get("Usage") or "").strip()
            pixels = row.get("pixels", "")
            if not pixels:
                continue
            if use_ferplus and i < len(ferplus_rows):
                label = ferplus_label_from_row(ferplus_rows[i])
                if label is None:
                    skipped += 1
                    continue
            else:
                try:
                    label = int(row["emotion"])
                except (KeyError, ValueError):
                    continue
            if usage == "Training":
                train_rows.append((label, pixels))
            elif usage == "PublicTest":
                val_rows.append((label, pixels))
    if use_ferplus:
        print(f"[fer+] labels (skipped {skipped} unknown/NF)")
    if not train_rows:
        raise ValueError(f"No Training rows in {csv_path}")
    return train_rows, val_rows


def build_fer_transforms(*, accurate_aug: bool = False):
    if accurate_aug:
        tf_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(12, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
            ]
        )
    else:
        tf_train = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    tf_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf_train, tf_val


def class_weights_tensor(ds, num_classes: int) -> torch.Tensor:
    from collections import Counter

    counts = Counter()
    n = len(ds)
    for i in range(n):
        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            label = ds.dataset[ds.indices[i]][1]
        else:
            label = ds[i][1]
        counts[int(label)] += 1
    weights = [n / (num_classes * max(counts.get(c, 1), 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
