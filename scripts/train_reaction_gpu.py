"""
GPU-only reaction / expression training — CIFAR-10, FER2013/FER+, or ImageFolder.

Research basis: FER2013 (Goodfellow 2013), FER+ (Barsoum 2016), ResNet (He 2016).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CKPT = Path(os.environ.get("CHECKPOINT_DIR", r"C:\Users\zakar\homelab-data\checkpoints"))
DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))

FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# FER+ vote columns -> FER2013 class index
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


def _ferplus_label_from_row(row: dict) -> int | None:
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


def _load_fer2013_csv(csv_path: Path, label_mode: str) -> tuple[list, list]:
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
                label = _ferplus_label_from_row(ferplus_rows[i])
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
        print(f"[fer+] labels from fer2013new.csv (skipped {skipped} unknown/NF rows)")
    if not train_rows:
        raise ValueError(f"No Training rows in {csv_path}")
    return train_rows, val_rows


def _class_weights_from_dataset(ds, num_classes: int) -> torch.Tensor:
    counts = Counter()
    n = len(ds)
    for i in range(n):
        if isinstance(ds, torch.utils.data.Subset):
            label = ds.dataset[ds.indices[i]][1]
        else:
            label = ds[i][1]
        counts[int(label)] += 1
    weights = []
    for c in range(num_classes):
        weights.append(n / (num_classes * max(counts.get(c, 1), 1)))
    return torch.tensor(weights, dtype=torch.float32)


def _build_transforms(accurate_aug: bool):
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


def _build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _build_datasets(dataset: str, version: str, label_mode: str, tf_train, tf_val):
    name = dataset.lower()
    if name == "cifar10":
        root = DATA / "cifar10"
        root.mkdir(parents=True, exist_ok=True)
        train_ds = datasets.CIFAR10(str(root), train=True, download=True, transform=tf_train)
        val_ds = datasets.CIFAR10(str(root), train=False, download=True, transform=tf_val)
        return train_ds, val_ds, 10, list(train_ds.classes)

    if name == "fer2013":
        csv_path = DATA / "fer2013" / "fer2013.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                "fer2013.csv missing. Run: python scripts/fetch_fer2013_kagglehub.py "
                "(FER+ mode needs aligned pixel CSV + fer2013new.csv)"
            )
        train_rows, val_rows = _load_fer2013_csv(csv_path, label_mode)
        train_ds = Fer2013CsvDataset(train_rows, tf_train)
        val_ds = Fer2013CsvDataset(val_rows, tf_val) if val_rows else None
        return train_ds, val_ds, 7, FER_LABELS

    if name in ("folder", "reactions", "me-reactions"):
        ver = version or "me-reactions"
        root = DATA / ver
        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Expected {train_dir}")
        train_ds = datasets.ImageFolder(str(train_dir), transform=tf_train)
        val_ds = (
            datasets.ImageFolder(str(val_dir), transform=tf_val) if val_dir.is_dir() else None
        )
        return train_ds, val_ds, len(train_ds.classes), train_ds.classes

    raise ValueError(f"Unknown dataset: {dataset}")


def _train_loop(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    grad_accum: int,
    out: Path,
    class_names: list,
    meta: dict,
    lr: float,
    label_smoothing: float,
    class_weights: torch.Tensor | None,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    use_cuda = device.type == "cuda"
    scaler = GradScaler(enabled=use_cuda)
    best = 0.0
    history = []
    patience = 12
    stale = 0

    for ep in range(1, epochs + 1):
        model.train()
        micro = 0
        opt.zero_grad(set_to_none=True)
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=use_cuda):
                loss = crit(model(images), labels) / grad_accum
            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro += 1
            if micro % grad_accum == 0:
                if use_cuda:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
        if micro % grad_accum:
            if use_cuda:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        sched.step()

        model.eval()
        vc = vt = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=use_cuda):
                    logits = model(images)
                vc += (logits.argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_acc = vc / max(vt, 1)
        history.append({"epoch": ep, "val_acc": round(val_acc, 4), "lr": round(sched.get_last_lr()[0], 6)})
        print(f"epoch {ep}/{epochs} val_acc={val_acc:.3f} lr={sched.get_last_lr()[0]:.2e}")
        if val_acc >= best:
            best = val_acc
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": list(class_names),
                    "val_acc": val_acc,
                    "meta": meta,
                },
                out / "best_model.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep} (no improvement for {patience} epochs)")
                break

    manifest = {**meta, "best_val_acc": round(best, 4), "epochs": epochs, "stopped_epoch": ep}
    (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        from training.mlflow_util import log_training_run

        log_training_run(
            experiment=os.environ.get("MLFLOW_EXPERIMENT", "reaction-vision"),
            run_name=str(meta.get("job_id", "train"))[:12],
            params={
                "dataset": meta.get("dataset"),
                "labels": meta.get("labels"),
                "arch": meta.get("arch"),
                "epochs": epochs,
                "accurate_aug": meta.get("accurate_aug"),
            },
            metrics={"best_val_acc": float(manifest["best_val_acc"])},
            artifacts_dir=out,
            tags={"task": "facial_expression", "host": os.environ.get("COMPUTERNAME", "gpu")},
        )
    except Exception:
        pass
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="GPU reaction/expression trainer")
    p.add_argument("--dataset", default="fer2013", choices=["cifar10", "fer2013", "folder", "reactions"])
    p.add_argument("--dataset-version", default="")
    p.add_argument("--labels", default="fer", choices=["fer", "ferplus"], help="FER2013 or FER+ crowd labels")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=-1, help="-1 = auto from CPU count")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--accurate-aug", action="store_true")
    args = p.parse_args()

    stop = REPO / "scripts" / "stop-ollama.ps1"
    if stop.exists():
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(stop)],
            check=False,
        )

    from training.cpu_io import configure_cpu_for_training, dataloader_kwargs

    cpu_info = configure_cpu_for_training()
    nw = None if args.num_workers < 0 else args.num_workers
    print("[cpu-io]", cpu_info)

    tf_train, tf_val = _build_transforms(args.accurate_aug)
    ver = args.dataset_version or ("me-reactions" if args.dataset in ("folder", "reactions") else "")
    train_ds, val_ds, num_classes, class_names = _build_datasets(
        args.dataset, ver, args.labels, tf_train, tf_val
    )

    if val_ds is None:
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

    job_id = str(uuid.uuid4())
    out = CKPT / job_id
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args.arch, num_classes).to(device)

    cw = None
    if args.class_weights:
        cw = _class_weights_from_dataset(train_ds, num_classes).to(device)
        print("[train] class weights:", [round(x, 3) for x in cw.tolist()])

    dl_kw = dataloader_kwargs(batch_size=args.batch_size, shuffle=True, cuda=device.type == "cuda", num_workers=nw)
    train_loader = DataLoader(train_ds, **dl_kw)
    val_loader = DataLoader(
        val_ds,
        **dataloader_kwargs(
            batch_size=args.batch_size, shuffle=False, cuda=device.type == "cuda", num_workers=nw
        ),
    )

    meta = {
        "job_id": job_id,
        "dataset": args.dataset,
        "labels": args.labels,
        "arch": args.arch,
        "dataset_version": ver,
        "model_version": "reaction-resnet34-ferplus-v1" if args.arch == "resnet34" else "reaction-resnet18-v1",
        "classes": list(class_names),
        "task": "facial_expression",
        "accurate_aug": args.accurate_aug,
        "cpu_io": cpu_info,
    }
    print(f"[reaction-train] job={job_id} labels={args.labels} arch={args.arch}")
    manifest = _train_loop(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.grad_accum,
        out,
        class_names,
        meta,
        args.lr,
        args.label_smoothing,
        cw,
    )
    print("done", out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
