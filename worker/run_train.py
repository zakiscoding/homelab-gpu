"""
Phase 0A train handler — stub for gate test (RTX 5070: stop Ollama before real CUDA).

Idempotent: all artifacts under checkpoints/{job_id}/ only.
Re-run same job_id must not create a second published marker.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path


def checkpoint_root() -> Path:
    base = os.environ.get("CHECKPOINT_DIR", r"C:\Users\zakar\homelab-data\checkpoints")
    return Path(base)


def run_train(job_id: str, *, steps: int = 5, step_sleep_sec: float = 2.0) -> Path:
    """
    Simulated train loop for Phase 0A gate (kill -9 mid-run).
    Writes progress files; final publish marker once.
    """
    root = checkpoint_root() / job_id
    root.mkdir(parents=True, exist_ok=True)

    published = root / "published.json"
    if published.exists():
        # Idempotent: already finished artifact publish for this job_id
        return root

    progress = root / "progress.json"
    for i in range(steps):
        progress.write_text(
            json.dumps({"step": i + 1, "total": steps, "host": socket.gethostname()}),
            encoding="utf-8",
        )
        time.sleep(step_sleep_sec)

    published.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "published",
                "host": socket.gethostname(),
                "note": "phase0a-stub",
            }
        ),
        encoding="utf-8",
    )
    return root
