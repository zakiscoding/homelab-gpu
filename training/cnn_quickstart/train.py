"""
CNN image classifier fine-tune — ResNet18 on CIFAR-10 or custom ImageFolder.

RTX 5070: stop Ollama before train (uses ~2–4 GB VRAM for CIFAR-10).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DATASETS = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))


def _stop_ollama() -> None:
    script = _REPO / "scripts" / "stop-ollama.ps1"
    if sys.platform == "win32" and script.exists():
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            check=False,
        )


def _build_loaders(
    dataset_name: str,
    dataset_version: str | None,
    batch_size: int,
):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf_train = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    name = (dataset_name or "cifar10").lower()
    if name == "cifar10":
        root = _DATASETS / "cifar10"
        root.mkdir(parents=True, exist_ok=True)
        train_ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tf_train)
        val_ds = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tf_val)
        num_classes = 10
        class_names = train_ds.classes
        return train_ds, val_ds, num_classes, class_names

    if name in ("folder", "imagefolder", "custom", "reactions"):
        ver = dataset_version or ("me-reactions" if name == "reactions" else "default")
        root = _DATASETS / ver
        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                f"Custom dataset expected at {train_dir} (subfolders = class names). "
                f"See homelab docs/CNN-FINETUNE.md"
            )
        train_ds = datasets.ImageFolder(str(train_dir), transform=tf_train)
        val_ds = (
            datasets.ImageFolder(str(val_dir), transform=tf_val)
            if val_dir.is_dir()
            else None
        )
        num_classes = len(train_ds.classes)
        return train_ds, val_ds, num_classes, train_ds.classes

    if name == "fer2013":
        from training.fer2013_data import FER_LABELS, Fer2013CsvDataset, build_fer_transforms, load_fer2013_rows

        label_mode = os.environ.get("FER_LABELS", "fer")
        tf_train, tf_val = build_fer_transforms(accurate_aug=os.environ.get("FER_ACCURATE_AUG") == "1")
        train_rows, val_rows = load_fer2013_rows(label_mode)
        train_ds = Fer2013CsvDataset(train_rows, tf_train)
        val_ds = Fer2013CsvDataset(val_rows, tf_val) if val_rows else None
        return train_ds, val_ds, 7, list(FER_LABELS)

    raise ValueError(f"Unknown dataset '{dataset_name}'. Use cifar10, fer2013, or folder.")


def run_cnn(job_id: str, out_dir: Path, payload: dict) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import models

    from training.cpu_io import configure_cpu_for_training, dataloader_kwargs, resolve_num_workers

    cpu_info = configure_cpu_for_training()
    print("[cpu-io]", cpu_info)

    _stop_ollama()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_model.pt"
    if weights_path.exists() and (out_dir / "manifest.json").exists():
        return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    dataset_name = payload.get("dataset") or payload.get("cnn_dataset") or "cifar10"
    dataset_version = payload.get("dataset_version")
    arch = (payload.get("architecture") or os.environ.get("CNN_ARCH", "resnet18")).lower()
    epochs = int(payload.get("epochs", os.environ.get("CNN_EPOCHS", "5")))
    batch_size = int(payload.get("batch_size", os.environ.get("CNN_BATCH_SIZE", "32")))
    lr = float(payload.get("learning_rate", os.environ.get("CNN_LR", "1e-3")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, num_classes, class_names = _build_loaders(
        dataset_name, dataset_version, batch_size
    )

    if val_ds is None:
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

    distributed = bool(payload.get("distributed"))
    if distributed:
        from training.cnn_quickstart.distributed_train import run_cnn_distributed

        return run_cnn_distributed(job_id, out_dir, payload)

    nw = int(payload.get("num_workers", resolve_num_workers()))
    cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, **dataloader_kwargs(batch_size=batch_size, shuffle=True, cuda=cuda, num_workers=nw))
    val_loader = DataLoader(
        val_ds, **dataloader_kwargs(batch_size=batch_size, shuffle=False, cuda=cuda, num_workers=nw)
    )

    from torch.cuda.amp import GradScaler, autocast

    use_amp = payload.get("use_amp", True)
    if isinstance(use_amp, str):
        use_amp = use_amp.lower() in ("1", "true", "yes")
    use_amp = use_amp and device.type == "cuda"
    grad_accum = max(1, int(payload.get("grad_accum_steps", 1)))

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro += 1
            if micro % grad_accum == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * grad_accum * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        if micro % grad_accum != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        v_correct = 0
        v_total = 0
        v_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)
        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )
        print(f"[cnn] epoch {epoch}/{epochs} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": arch,
                    "num_classes": num_classes,
                    "class_names": list(class_names),
                    "val_acc": val_acc,
                },
                weights_path,
            )

    manifest = {
        "job_id": job_id,
        "train_mode": "cnn",
        "dataset": dataset_name,
        "dataset_version": dataset_version,
        "model_version": payload.get("model_version") or f"cnn-{job_id[:8]}",
        "architecture": arch,
        "epochs": epochs,
        "best_val_acc": round(best_acc, 4),
        "weights_path": str(weights_path),
        "class_names": list(class_names),
        "host": os.environ.get("COMPUTERNAME", "gpu"),
        "cpu_io": cpu_info,
        "num_workers": nw,
    }
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
