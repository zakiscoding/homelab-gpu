"""
MVW GPU worker — Phase 0A/0B.

loop:
  heartbeat + expire stale leases (via API)
  claim train job (lease on iMac Postgres)
  run train stub
  mark succeeded / failed (transient → auto-retry on API)
  sleep

RTX 5070: QLoRA default (TRAIN_MODE=qlora); stub for gate tests. stop-ollama runs in train.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from worker.job_wake import WAKE_TRAIN, wait_for_wake  # noqa: E402
from worker.run_train import run_train  # noqa: E402

API_BASE = os.environ.get("API_BASE_URL", "http://100.91.113.98").strip().rstrip("/")
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
WORKER_HOST = os.environ.get("WORKER_HOST", socket.gethostname())
POLL_INTERVAL = float(os.environ.get("WORKER_POLL_SEC", "10"))
# train leases are 15 min on the iMac API; renew every 5 so long QLoRA runs aren't requeued
LEASE_RENEW_SEC = float(os.environ.get("WORKER_LEASE_RENEW_SEC", "300"))


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


class LeaseKeepAlive:
    """Renew a job's lease in a background thread while training runs.

    POST /internal/worker/jobs/{id}/renew every LEASE_RENEW_SEC with its own HTTP client. Without
    it, a run longer than the 15-min lease is requeued by the API's lease sweep and can run twice.
    Errors (e.g. 409: job taken over) are logged, never raised. Use as a context manager.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        with httpx.Client(timeout=30.0) as c:
            while not self.stop.wait(LEASE_RENEW_SEC):
                try:
                    c.post(f"{API_BASE}/internal/worker/jobs/{self.job_id}/renew", headers=_headers()).raise_for_status()
                except httpx.HTTPError as e:
                    print(f"[worker] lease renew failed for {self.job_id}: {e}")

    def __enter__(self) -> "LeaseKeepAlive":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def claim_job(client: httpx.Client) -> dict | None:
    r = client.post(f"{API_BASE}/internal/worker/jobs/claim", headers=_headers())
    r.raise_for_status()
    if r.status_code == 204:
        return None
    return r.json()


def register_experiment(client: httpx.Client, job_id: str, manifest: dict) -> None:
    r = client.post(
        f"{API_BASE}/internal/worker/jobs/{job_id}/experiment",
        headers=_headers(),
        json={
            "dataset_version": manifest.get("dataset_version"),
            "model_version": manifest.get("model_version"),
            "output_path": manifest.get("adapter_path"),
            "hyperparams": {
                "base_model": manifest.get("base_model"),
                "train_mode": manifest.get("train_mode"),
                "use_4bit": manifest.get("use_4bit"),
            },
        },
    )
    r.raise_for_status()


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
    payload = job.get("payload") or {}
    try:
        with LeaseKeepAlive(job_id):
            artifact_dir, manifest = run_train(job_id, payload)
        print(f"[worker] artifacts at {artifact_dir}")
        try:
            register_experiment(client, job_id, manifest)
        except Exception as exp_err:
            print(f"[worker] experiment register warning: {exp_err}")
        mark_succeeded(client, job_id)
        print(f"[worker] job {job_id} succeeded")
    except Exception as e:
        print(f"[worker] job {job_id} failed: {e}")
        try:
            mark_failed(client, job_id, str(e), transient=True)
        except Exception as mark_err:
            print(f"[worker] could not mark failed: {mark_err}")
    return True


def wait_for_api(client: httpx.Client, timeout_s: float = 600.0) -> None:
    """Block until the iMac API answers /health (backoff 1 s -> 30 s).

    At Windows boot the scheduled task can start before Tailscale reaches the iMac; without this
    the startup reset-stale call raised and the worker exited. Gives up after ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    delay = 1.0
    while True:
        try:
            client.get(f"{API_BASE}/health", timeout=10.0).raise_for_status()
            return
        except httpx.HTTPError as e:
            if time.monotonic() > deadline:
                raise
            print(f"[worker] waiting for API ({e.__class__.__name__}); retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GPU train job worker (Phase 0A/0B)")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle then exit")
    args = parser.parse_args()

    if not WORKER_API_KEY:
        print("WORKER_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    notify = os.environ.get("JOB_NOTIFY_ENABLED", "true").lower() == "true"
    print(
        f"[worker] id={WORKER_ID} host={WORKER_HOST} api={API_BASE} "
        f"poll={POLL_INTERVAL}s notify={notify}"
    )
    train_timeout = float(os.environ.get("WORKER_TRAIN_TIMEOUT_SEC", "1800"))
    with httpx.Client(timeout=train_timeout) as client:
        wait_for_api(client)
        reset_stale(client)
        while True:
            try:
                processed = run_once(client)
            except httpx.HTTPError as e:
                print(f"[worker] HTTP error: {e}")
                processed = False
            if args.once:
                break
            if not processed:
                wait_for_wake([WAKE_TRAIN], POLL_INTERVAL)
            else:
                time.sleep(0.2)


if __name__ == "__main__":
    main()
