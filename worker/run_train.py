"""
Train handler — Phase 0A stub or Phase 3 QLoRA (RTX 5070).

Idempotent: artifacts under checkpoints/{job_id}/ only.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def checkpoint_root() -> Path:
    base = os.environ.get("CHECKPOINT_DIR", r"C:\Users\zakar\homelab-data\checkpoints")
    return Path(base)


def _write_published(root: Path, job_id: str, manifest: dict[str, Any]) -> None:
    (root / "published.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "published",
                "host": socket.gethostname(),
                "manifest": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_stub(job_id: str, *, steps: int = 5, step_sleep_sec: float = 2.0) -> tuple[Path, dict]:
    root = checkpoint_root() / job_id
    root.mkdir(parents=True, exist_ok=True)
    published = root / "published.json"
    if published.exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return root, manifest

    progress = root / "progress.json"
    for i in range(steps):
        progress.write_text(
            json.dumps({"step": i + 1, "total": steps, "host": socket.gethostname()}),
            encoding="utf-8",
        )
        time.sleep(step_sleep_sec)

    manifest = {"job_id": job_id, "train_mode": "stub", "note": "phase0a-stub"}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_published(root, job_id, manifest)
    return root, manifest


def _run_qlora(job_id: str, payload: dict[str, Any]) -> tuple[Path, dict]:
    root = checkpoint_root() / job_id
    root.mkdir(parents=True, exist_ok=True)
    if (root / "published.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return root, manifest

    from training.lora_quickstart.train import run_qlora  # noqa: WPS433

    manifest = run_qlora(job_id, root, payload)
    _write_published(root, job_id, manifest)
    return root, manifest


def run_train(job_id: str, payload: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    """
    Run train job. Returns (artifact_dir, manifest).
    TRAIN_MODE=stub for gate tests; default qlora.
    """
    mode = os.environ.get("TRAIN_MODE", "qlora").lower()
    pl = payload or {}
    if mode == "stub":
        return _run_stub(job_id)
    return _run_qlora(job_id, pl)
