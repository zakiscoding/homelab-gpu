"""Minimal GPU-only CIFAR-10 ResNet18 train (single file, no package layout)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

REPO = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("CHECKPOINT_DIR", r"C:\Users\zakar\homelab-data\checkpoints"))
DATA = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))


def main() -> None:
    epochs = int(os.environ.get("CNN_EPOCHS", "10"))
    batch_size = int(os.environ.get("CNN_BATCH_SIZE", "32"))
    grad_accum = int(os.environ.get("CNN_GRAD_ACCUM", "4"))
    job_id = str(uuid.uuid4())
    out = CKPT / job_id
    out.mkdir(parents=True, exist_ok=True)

    stop = REPO / "scripts" / "stop-ollama.ps1"
    if stop.exists():
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(stop)],
            check=False,
        )

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
    root = DATA / "cifar10"
    train_ds = datasets.CIFAR10(str(root), train=True, download=True, transform=tf_train)
    val_ds = datasets.CIFAR10(str(root), train=False, download=True, transform=tf_val)

    sys.path.insert(0, str(REPO))
    from training.cpu_io import configure_cpu_for_training, dataloader_kwargs

    cpu_info = configure_cpu_for_training()
    print("[cpu-io]", cpu_info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=device.type == "cuda")
    cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, **dataloader_kwargs(batch_size=batch_size, shuffle=True, cuda=cuda))
    val_loader = DataLoader(val_ds, **dataloader_kwargs(batch_size=batch_size, shuffle=False, cuda=cuda))

    best = 0.0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = corr = total = micro = 0
        opt.zero_grad(set_to_none=True)
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                loss = crit(model(images), labels) / grad_accum
            scaler.scale(loss).backward() if device.type == "cuda" else loss.backward()
            micro += 1
            if micro % grad_accum == 0:
                if device.type == "cuda":
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * grad_accum * images.size(0)
            corr += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
        if micro % grad_accum:
            scaler.step(opt) if device.type == "cuda" else opt.step()
            scaler.update() if device.type == "cuda" else None
        model.eval()
        vc = vt = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=device.type == "cuda"):
                    out_logits = model(images)
                vc += (out_logits.argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_acc = vc / max(vt, 1)
        history.append({"epoch": ep, "val_acc": round(val_acc, 4)})
        print(f"epoch {ep}/{epochs} val_acc={val_acc:.3f}")
        if val_acc >= best:
            best = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": train_ds.classes, "val_acc": val_acc},
                out / "best_model.pt",
            )

    manifest = {"job_id": job_id, "dataset": "cifar10", "best_val_acc": best, "epochs": epochs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main()
