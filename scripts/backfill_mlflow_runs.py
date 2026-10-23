#!/usr/bin/env python3
"""Upload existing checkpoint dirs to MLflow (one-time / after MLflow deploy)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.mlflow_util import log_training_run, tracking_uri  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill MLflow from homelab-data checkpoints")
    p.add_argument(
        "--checkpoints-root",
        default=os.environ.get("HOMELAB_DATA", r"C:\Users\zakar\homelab-data") + r"\checkpoints",
    )
    p.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT", "reaction-vision"))
    p.add_argument("--limit", type=int, default=0, help="Max dirs to upload (0 = all with manifest)")
    p.add_argument("--job-id", default="", help="Only this job UUID folder")
    args = p.parse_args()

    uri = tracking_uri()
    if not uri:
        print("Set MLFLOW_TRACKING_URI (e.g. http://100.91.113.98:5001)")
        sys.exit(1)

    root = Path(args.checkpoints_root)
    if not root.is_dir():
        print(f"Missing checkpoints root: {root}")
        sys.exit(1)

    dirs = sorted(root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
    if args.job_id:
        dirs = [root / args.job_id]
    logged = 0
    for d in dirs:
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = {}
        if "best_val_acc" in manifest:
            metrics["best_val_acc"] = float(manifest["best_val_acc"])
        params = {
            k: manifest[k]
            for k in (
                "job_id",
                "dataset",
                "labels",
                "arch",
                "dataset_version",
                "model_version",
                "epochs",
                "accurate_aug",
                "task",
            )
            if k in manifest
        }
        run_name = str(manifest.get("job_id", d.name))[:36]
        rid = log_training_run(
            experiment=args.experiment,
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts_dir=d,
            tags={"backfill": "true", "host": os.environ.get("COMPUTERNAME", "gpu")},
        )
        if rid:
            logged += 1
            print(f"OK {d.name} best_val_acc={metrics.get('best_val_acc')}")
        if args.limit and logged >= args.limit:
            break
    print(f"Done: {logged} run(s) -> {uri} experiment={args.experiment}")


if __name__ == "__main__":
    main()
