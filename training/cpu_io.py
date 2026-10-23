"""
Use the host CPU (e.g. i7-14700F) to feed the GPU — parallel decode/augment in DataLoader workers.

Set CNN_NUM_WORKERS to override auto-detected worker count (default targets ~10 on 20-core CPUs).
"""

from __future__ import annotations

import os
from typing import Any


def resolve_num_workers(override: int | None = None) -> int:
    """DataLoader worker processes — leave headroom for GPU driver + main thread."""
    if override is not None and override >= 0:
        return override
    env = os.environ.get("CNN_NUM_WORKERS")
    if env is not None and env.strip() != "":
        return max(0, int(env))
    logical = os.cpu_count() or 8
    # Heavy aug (FER): ~half of logical cores, capped for RAM (~2–4 GB per worker at peak)
    return max(4, min(12, logical // 2))


def _worker_init_fn(_worker_id: int) -> None:
    """Avoid 10 workers × 28 OpenMP threads oversubscribing the CPU."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def configure_cpu_for_training() -> dict[str, Any]:
    """
    Call once at train start (main process).
    Returns a small dict logged in manifests.
    """
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "num_workers": resolve_num_workers(),
    }
    main_threads = int(os.environ.get("TRAIN_MAIN_THREADS", "4"))
    try:
        import torch

        torch.set_num_threads(main_threads)
        info["main_threads"] = main_threads
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            info["cuda_tf32"] = True
            info["cudnn_benchmark"] = True
    except Exception:
        pass
    return info


def dataloader_kwargs(
    *,
    batch_size: int,
    shuffle: bool,
    cuda: bool,
    num_workers: int | None = None,
) -> dict[str, Any]:
    nw = resolve_num_workers(num_workers)
    kw: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": nw,
        "pin_memory": cuda,
    }
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = int(os.environ.get("CNN_PREFETCH", "4"))
        kw["worker_init_fn"] = _worker_init_fn
    return kw
