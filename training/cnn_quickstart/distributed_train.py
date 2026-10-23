"""Distributed CNN train on GPU — consumes iMac shard plan + AMP + grad accumulation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from training.cnn_quickstart.plan_client import fetch_train_plan
from training.cnn_quickstart.train import _build_loaders, _stop_ollama


def run_cnn_distributed(job_id: str, out_dir: Path, payload: dict[str, Any]) -> dict:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Subset
    from torchvision import models

    plan = fetch_train_plan(job_id)
    _stop_ollama()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_model.pt"

    dataset_name = plan.get("dataset") or payload.get("dataset") or "cifar10"
    dataset_version = plan.get("dataset_version") or payload.get("dataset_version")
    label_mode = (payload.get("labels") or "fer").lower()
    if label_mode == "ferplus":
        os.environ["FER_LABELS"] = "ferplus"
    if payload.get("accurate_aug"):
        os.environ["FER_ACCURATE_AUG"] = "1"
    arch = (payload.get("architecture") or payload.get("arch") or "resnet18").lower()
    epochs = int(payload.get("epochs", 5))
    batch_size = int(payload.get("batch_size", 32))
    lr = float(payload.get("learning_rate", 1e-3))
    grad_accum = int(payload.get("grad_accum_steps", 2))
    use_amp = payload.get("use_amp", True)
    if isinstance(use_amp, str):
        use_amp = use_amp.lower() in ("1", "true", "yes")

    from training.cpu_io import configure_cpu_for_training, dataloader_kwargs, resolve_num_workers

    cpu_info = configure_cpu_for_training()
    num_workers = int(payload.get("num_workers", resolve_num_workers()))
    print("[cpu-io]", cpu_info, "num_workers=", num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"

    full_train, full_val, num_classes, class_names = _build_loaders(
        dataset_name, dataset_version, batch_size
    )
    val_indices = plan.get("val_indices") or []
    if val_indices:
        val_ds = Subset(full_train, val_indices)
    elif full_val is not None:
        val_ds = full_val
    else:
        val_ds = full_train

    shards = plan.get("shards") or []
    if not shards:
        raise RuntimeError("train plan has no shards")

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = model.to(device)
    class_weights = None
    if payload.get("class_weights") and dataset_name.lower() == "fer2013":
        from training.fer2013_data import class_weights_tensor

        class_weights = class_weights_tensor(full_train, num_classes).to(device)
        print("[cnn-dist] class weights:", [round(x, 3) for x in class_weights.tolist()])
    label_smoothing = float(payload.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = GradScaler(enabled=use_amp)

    val_loader = DataLoader(
        val_ds,
        **dataloader_kwargs(
            batch_size=batch_size, shuffle=False, cuda=device.type == "cuda", num_workers=num_workers
        ),
    )

    best_acc = 0.0
    history: list[dict] = []
    seed = int(plan.get("seed", 42))

    for epoch in range(1, epochs + 1):
        model.train()
        shard_order = list(shards)
        random.Random(seed + epoch).shuffle(shard_order)
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        micro = 0

        for shard in shard_order:
            indices = shard["indices"]
            shard_ds = Subset(full_train, indices)
            train_loader = DataLoader(
                shard_ds,
                **dataloader_kwargs(
                    batch_size=batch_size, shuffle=True, cuda=device.type == "cuda", num_workers=num_workers
                ),
            )
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        scheduler.step()

        model.eval()
        v_correct = v_total = 0
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
                "shards": len(shard_order),
            }
        )
        print(
            f"[cnn-dist] epoch {epoch}/{epochs} shards={len(shard_order)} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": arch,
                    "arch": arch,
                    "num_classes": num_classes,
                    "classes": list(class_names),
                    "class_names": list(class_names),
                    "val_acc": val_acc,
                    "distributed": True,
                    "shard_count": len(shards),
                    "meta": {
                        "arch": arch,
                        "labels": label_mode,
                        "classes": list(class_names),
                        "dataset": dataset_name,
                    },
                },
                weights_path,
            )

    manifest = {
        "job_id": job_id,
        "train_mode": "cnn",
        "distributed": True,
        "dataset": dataset_name,
        "dataset_version": dataset_version,
        "model_version": payload.get("model_version") or f"cnn-{job_id[:8]}",
        "architecture": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "use_amp": use_amp,
        "shard_count": len(shards),
        "best_val_acc": round(best_acc, 4),
        "weights_path": str(weights_path),
        "class_names": list(class_names),
        "host": __import__("os").environ.get("COMPUTERNAME", "gpu"),
        "cpu_io": cpu_info,
        "num_workers": num_workers,
    }
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
