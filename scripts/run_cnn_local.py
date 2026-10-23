"""Entrypoint for GPU-only CNN train (no iMac job queue)."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from worker.run_train import run_train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="GPU-only CNN train (optimized: AMP + grad accum)")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "folder"])
    p.add_argument("--dataset-version", default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--model-version", default="gpu-cnn-v1")
    args = p.parse_args()

    job_id = str(uuid.uuid4())
    payload: dict = {
        "train_mode": "cnn",
        "dataset": args.dataset,
        "model_version": args.model_version,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "use_amp": not args.no_amp,
        "distributed": False,
    }
    if args.dataset_version:
        payload["dataset_version"] = args.dataset_version

    print(f"[gpu-train] job_id={job_id} dataset={args.dataset} epochs={args.epochs}")
    root, manifest = run_train(job_id, payload)
    print(f"[gpu-train] done artifacts={root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
