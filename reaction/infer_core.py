"""Shared reaction CNN inference (ResNet18/34 checkpoints)."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

_DEFAULT_CKPT = Path(
    os.environ.get(
        "REACTION_CHECKPOINT",
        r"C:\Users\zakar\homelab-data\checkpoints\3536fd8e-b6ae-4b58-9ea3-eb1868883d0f\best_model.pt",
    )
)
_THRESHOLD = float(os.environ.get("REACTION_THRESHOLD", "0.45"))

_model: nn.Module | None = None
_classes: list[str] = []
_device: torch.device | None = None
_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def resolve_checkpoint(path: str | None = None) -> Path:
    if path:
        p = Path(path)
        if p.is_file():
            return p
        if (p / "best_model.pt").is_file():
            return p / "best_model.pt"
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if _DEFAULT_CKPT.is_file():
        return _DEFAULT_CKPT
    ckpt_root = Path(os.environ.get("CHECKPOINT_DIR", r"C:\Users\zakar\homelab-data\checkpoints"))
    candidates = sorted(ckpt_root.glob("*/best_model.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No best_model.pt under {ckpt_root}")


def load_model(checkpoint: str | Path | None = None) -> tuple[nn.Module, list[str], torch.device]:
    global _model, _classes, _device
    ckpt_path = resolve_checkpoint(str(checkpoint) if checkpoint else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta") or {}
    classes = list(ckpt.get("classes") or meta.get("classes") or [])
    arch = str(meta.get("arch") or "resnet18").lower()
    if arch == "resnet34":
        model = models.resnet34(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    _model, _classes, _device = model, classes, device
    return model, classes, device


def model_info() -> dict[str, Any]:
    if _model is None:
        load_model()
    ckpt_path = resolve_checkpoint()
    return {
        "classes": _classes,
        "checkpoint": str(ckpt_path),
        "device": str(_device),
        "threshold_default": _THRESHOLD,
        "task": "facial_expression",
    }


def _decode_image(image_b64: str) -> np.ndarray:
    raw = image_b64.strip()
    if "," in raw:
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _largest_face_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def predict_image_b64(image_b64: str, threshold: float | None = None) -> dict[str, Any]:
    if _model is None:
        load_model()
    thresh = threshold if threshold is not None else _THRESHOLD
    rgb = _decode_image(image_b64)
    bbox = _largest_face_bbox(rgb)
    if bbox is None:
        return {
            "label": None,
            "confidence": 0.0,
            "reason": "no_face",
            "face_detected": False,
            "probabilities": {},
        }
    x, y, w, h = bbox
    roi = rgb[y : y + h, x : x + w]
    pil = Image.fromarray(roi)
    tensor = _transform(pil).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    conf = float(probs.max())
    idx = int(probs.argmax())
    prob_map = {cls: round(float(probs[i]), 4) for i, cls in enumerate(_classes)}
    if conf >= thresh:
        return {
            "label": _classes[idx],
            "confidence": round(conf, 4),
            "reason": "fer_classifier",
            "face_detected": True,
            "probabilities": prob_map,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
        }
    return {
        "label": "ambiguous",
        "confidence": round(conf, 4),
        "reason": "ambiguous_expression",
        "face_detected": True,
        "probabilities": prob_map,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
    }
