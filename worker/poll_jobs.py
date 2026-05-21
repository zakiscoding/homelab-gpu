"""
MVW GPU worker — Phase 0A.

loop:
  claim job (via iMac API)
  run train stub
  mark succeeded / failed
  sleep

RTX 5070: run scripts/stop-ollama.ps1 before real training (stub does not need GPU).
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import httpx

# Allow importing run_train when launched from repo root
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from worker.run_train import run_train  # noqa: E402

API_BASE = os.environ.get("API_BASE_URL", "http://100.91.113.98").rstrip("/")
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
POLL_INTERVAL = float(os.environ.get("WORKER_POLL_SEC", "10"))


def _headers() -> dict[str, str]:
    return {
        "X-Worker-Key": WORKER_API_KEY,
        "X-Worker-Id": WORKER_ID,
    }


def reset_stale(client: httpx.Client) -> None:
    r = client.post(f"{API_BASE}/internal/worker/jobs/reset-stale", headers=_headers())
    if r.status_code == 200 and r.json().get("reset_count", 0) > 0:
        print(f"[worker] reset {r.json()['reset_count']} stale running job(s)")


def claim_job(client: httpx.Client) -> dict | None:
    r = client.post(f"{API_BASE}/internal/worker/jobs/claim", headers=_headers())
    r.raise_for_status()
    if r.status_code == 204:
        return None
    return r.json()


def mark_succeeded(client: httpx.Client, job_id: str) -> None:
    r = client.post(
        f"{API_BASE}/internal/worker/jobs/{job_id}/succeeded",
        headers=_headers(),
    )
    r.raise_for_status()


def mark_failed(client: httpx.Client, job_id: str, msg: str) -> None:
    r = client.post(
        f"{API_BASE}/internal/worker/jobs/{job_id}/failed",
        headers=_headers(),
        json={"error_message": msg[:2000]},
    )
    r.raise_for_status()


def run_once(client: httpx.Client) -> bool:
    """Returns True if a job was processed."""
    job = claim_job(client)
    if not job:
        return False

    job_id = job["id"]
    print(f"[worker] claimed job {job_id}")
    try:
        artifact_dir = run_train(job_id)
        print(f"[worker] artifacts at {artifact_dir}")
        mark_succeeded(client, job_id)
        print(f"[worker] job {job_id} succeeded")
    except Exception as e:
        print(f"[worker] job {job_id} failed: {e}")
        try:
            mark_failed(client, job_id, str(e))
        except Exception as mark_err:
            print(f"[worker] could not mark failed: {mark_err}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GPU train job worker (Phase 0A)")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle then exit")
    args = parser.parse_args()

    if not WORKER_API_KEY:
        print("WORKER_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    print(f"[worker] id={WORKER_ID} api={API_BASE} poll={POLL_INTERVAL}s")
    with httpx.Client(timeout=60.0) as client:
        reset_stale(client)
        while True:
            try:
                run_once(client)
            except httpx.HTTPError as e:
                print(f"[worker] HTTP error: {e}")
            if args.once:
                break
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
