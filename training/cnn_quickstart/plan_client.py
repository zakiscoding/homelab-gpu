"""Fetch distributed train plan from iMac API (Postgres shards)."""

from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE = os.environ.get("API_BASE_URL", "http://100.91.113.98").strip().rstrip("/")
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")


def fetch_train_plan(train_job_id: str, *, timeout: float = 60.0) -> dict[str, Any]:
    headers = {"X-Worker-Key": WORKER_API_KEY}
    url = f"{API_BASE}/internal/worker/train/{train_job_id}/plan"
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()
