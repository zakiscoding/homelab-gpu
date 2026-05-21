"""
MVW GPU worker — Phase 0A/0B.

loop:
  heartbeat + expire stale leases (via API)
  claim train job (lease on iMac Postgres)
  run train stub
  mark succeeded / failed (transient → auto-retry on API)
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

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from worker.run_train import run_train  # noqa: E402

API_BASE = os.environ.get("API_BASE_URL", "http://100.91.113.98").rstrip("/")
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
WORKER_HOST = os.environ.get("WORKER_HOST", socket.gethostname())
POLL_INTERVAL = float(os.environ.get("WORKER_POLL_SEC", "10"))


def _headers() -> dict[str, str]:
    return {
        "X-Worker-Key": WORKER_API_KEY,
        "X-Worker-Id": WORKER_ID,
        "X-Worker-Host": WORKER_HOST,
    }


def send_heartbeat(client: httpx.Client) -> None:
    r = client.post(f"{API_BASE}/internal/worker/heartbeat", headers=_headers())
    r.raise_for_status()


def reset_stale(client: httpx.Client) -> None:
    r = client.post(f"{API_BASE}/internal/worker/jobs/reset-stale", headers=_headers())
    r.raise_for_status()
    if r.status_code != 200:
        return
    data = r.json()
    expired = data.get("lease_expired_requeued", 0)
    forced = data.get("forced_reset", 0)
    # Phase 0A compat
    if not expired and not forced:
        forced = data.get("reset_count", 0)
    if expired:
        print(f"[worker] requeued {expired} lease-expired job(s)")
    if forced:
        print(f"[worker] force-reset {forced} stale running job(s)")


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


def mark_failed(client: httpx.Client, job_id: str, msg: str, *, transient: bool = True) -> None:
    r = client.post(
        f"{API_BASE}/internal/worker/jobs/{job_id}/failed",
        headers=_headers(),
        json={
            "error_message": msg[:2000],
            "error_class": "transient" if transient else "permanent",
        },
    )
    r.raise_for_status()


def run_once(client: httpx.Client) -> bool:
    """Returns True if a job was processed."""
    send_heartbeat(client)
    job = claim_job(client)
    if not job:
        return False

    job_id = job["id"]
    attempt = job.get("attempt", 0)
    print(f"[worker] claimed job {job_id} (attempt={attempt})")
    try:
        artifact_dir = run_train(job_id)
        print(f"[worker] artifacts at {artifact_dir}")
        mark_succeeded(client, job_id)
        print(f"[worker] job {job_id} succeeded")
    except Exception as e:
        print(f"[worker] job {job_id} failed: {e}")
        try:
            mark_failed(client, job_id, str(e), transient=True)
        except Exception as mark_err:
            print(f"[worker] could not mark failed: {mark_err}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GPU train job worker (Phase 0A/0B)")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle then exit")
    args = parser.parse_args()

    if not WORKER_API_KEY:
        print("WORKER_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    print(f"[worker] id={WORKER_ID} host={WORKER_HOST} api={API_BASE} poll={POLL_INTERVAL}s")
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
