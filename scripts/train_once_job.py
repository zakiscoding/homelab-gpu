"""Run QLoRA for a single job id (local GPU debug)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from worker.run_train import run_train  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: train_once_job.py <job_id> [payload.json]", file=sys.stderr)
        sys.exit(1)
    job_id = sys.argv[1]
    payload: dict = {}
    if len(sys.argv) > 2:
        payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    else:
        payload = {
            "dataset_version": "smoke",
            "model_version": "phase3-smoke-v1",
        }
    root, manifest = run_train(job_id, payload)
    print(f"done root={root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
