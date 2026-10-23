#!/usr/bin/env python3
"""Register PyTorch checkpoints in MLflow Model Registry (for runs already logged)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.mlflow_util import (  # noqa: E402
    register_existing_run,
    registered_model_name,
    tracking_uri,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Register homelab checkpoints in MLflow Models tab")
    p.add_argument(
        "--checkpoints-root",
        default=os.environ.get("HOMELAB_DATA", r"C:\Users\zakar\homelab-data") + r"\checkpoints",
    )
    p.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT", "reaction-vision"))
    p.add_argument("--job-id", default="", help="Checkpoint folder / homelab job UUID")
    p.add_argument("--best-only", action="store_true", help="Only highest best_val_acc with manifest")
    p.add_argument(
        "--artifact-only",
        action="store_true",
        help="Register runs:/.../best_model.pt without loading PyTorch (SSH-safe)",
    )
    p.add_argument("--registered-name", default="", help="Override MLFLOW_REGISTERED_MODEL")
    args = p.parse_args()

    uri = tracking_uri()
    if not uri:
        print("Set MLFLOW_TRACKING_URI (e.g. http://100.91.113.98:5001)")
        sys.exit(1)

    if args.registered_name:
        os.environ["MLFLOW_REGISTERED_MODEL"] = args.registered_name
    if args.artifact_only:
        os.environ["MLFLOW_REGISTER_ARTIFACT_ONLY"] = "1"

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"Experiment not found: {args.experiment}")
        sys.exit(1)

    root = Path(args.checkpoints_root)
    runs = client.search_runs([exp.experiment_id], max_results=200)
    by_name = {r.info.run_name: r.info.run_id for r in runs if r.info.run_name}

    if args.job_id:
        dirs = [root / args.job_id]
    elif args.best_only:
        best_dir = None
        best_acc = -1.0
        for d in root.iterdir():
            m = d / "manifest.json"
            if not m.is_file():
                continue
            import json

            man = json.loads(m.read_text(encoding="utf-8"))
            task = str(man.get("task") or "")
            if task and task != "facial_expression":
                continue
            acc = man.get("best_val_acc")
            if acc is None:
                continue
            acc = float(acc)
            if acc > 0.85:
                continue
            if acc > best_acc:
                best_acc = acc
                best_dir = d
        dirs = [best_dir] if best_dir else []
    else:
        dirs = [d for d in root.iterdir() if d.is_dir()]

    reg_name = registered_model_name()
    count = 0
    for d in dirs:
        if d is None or not d.is_dir():
            continue
        run_name = d.name
        run_id = by_name.get(run_name)
        if not run_id:
            print(f"skip {run_name}: no MLflow run (run backfill_mlflow_runs.py first)")
            continue
        name = register_existing_run(run_id=run_id, checkpoint_dir=d, registered_name=reg_name if args.registered_name else None)
        if name:
            count += 1
            print(f"OK {run_name} -> Models/{name}")

    print(f"Done: {count} model version(s); open Models tab at {uri}")


if __name__ == "__main__":
    main()
