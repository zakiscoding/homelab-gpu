"""Log homelab training runs to MLflow on the iMac (runs + model registry)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_DEFAULT_REGISTERED = "homelab-reaction-fer"


def tracking_uri() -> str | None:
    return os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_URI")


def register_model_enabled() -> bool:
    return os.environ.get("MLFLOW_REGISTER_MODEL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def registered_model_name(manifest: dict[str, Any] | None = None) -> str:
    if os.environ.get("MLFLOW_REGISTERED_MODEL"):
        return _sanitize_name(os.environ["MLFLOW_REGISTERED_MODEL"])
    if manifest:
        if manifest.get("model_version"):
            return _sanitize_name(str(manifest["model_version"]))
        arch = manifest.get("arch")
        if arch:
            return _sanitize_name(f"homelab-reaction-{arch}")
    return _DEFAULT_REGISTERED


def _sanitize_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-.]", "-", name.strip())
    return s[:128] or _DEFAULT_REGISTERED


def load_model_from_checkpoint(ckpt_path: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = dict(ckpt.get("meta") or {})
    classes = list(ckpt.get("classes") or meta.get("classes") or [])
    arch = str(meta.get("arch") or "resnet18").lower()
    if arch == "resnet34":
        model = models.resnet34(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta["classes"] = classes
    meta["arch"] = arch
    return model, meta


def register_pytorch_model(
    *,
    model: Any,
    manifest: dict[str, Any],
    registered_name: str | None = None,
) -> str | None:
    import mlflow

    name = registered_name or registered_model_name(manifest)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=name,
        metadata={
            "job_id": str(manifest.get("job_id", "")),
            "arch": str(manifest.get("arch", "")),
            "classes": json.dumps(manifest.get("classes") or []),
            "best_val_acc": str(manifest.get("best_val_acc", "")),
        },
    )
    print(f"[mlflow] registered model '{name}' (new version)")
    return name


def log_training_run(
    *,
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: Path,
    tags: dict[str, str] | None = None,
    register_model: bool | None = None,
) -> str | None:
    """Returns run_id or None if MLflow disabled/unreachable."""
    uri = tracking_uri()
    if not uri:
        return None
    try:
        import mlflow
    except ImportError:
        print("[mlflow] pip install mlflow to enable experiment tracking")
        return None

    ckpt_path = artifacts_dir / "best_model.pt"
    manifest_path = artifacts_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    do_register = register_model if register_model is not None else register_model_enabled()

    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            if manifest.get("job_id"):
                mlflow.set_tag("job_id", str(manifest["job_id"]))
            safe_params = {k: str(v)[:250] for k, v in params.items()}
            mlflow.log_params(safe_params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            for name in ("best_model.pt", "manifest.json", "metrics.json"):
                path = artifacts_dir / name
                if path.is_file():
                    mlflow.log_artifact(str(path))
            if do_register and ckpt_path.is_file():
                try:
                    model, meta = load_model_from_checkpoint(ckpt_path)
                    register_pytorch_model(model=model, manifest={**manifest, **meta})
                except Exception as reg_exc:
                    print(f"[mlflow] model registry skip ({reg_exc})")
            print(f"[mlflow] logged run {run.info.run_id} -> {uri}")
            return run.info.run_id
    except Exception as exc:
        print(f"[mlflow] skip ({exc})")
        return None


def register_existing_run(
    *,
    run_id: str,
    checkpoint_dir: Path,
    registered_name: str | None = None,
) -> str | None:
    """Attach a registered model to an existing MLflow run (PyTorch or artifact URI)."""
    uri = tracking_uri()
    if not uri:
        return None
    name = registered_name or registered_model_name(
        _read_manifest(checkpoint_dir)
    )
    if os.environ.get("MLFLOW_REGISTER_ARTIFACT_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return register_run_artifact(run_id=run_id, registered_name=name)

    ckpt_path = checkpoint_dir / "best_model.pt"
    if not ckpt_path.is_file():
        print(f"[mlflow] no checkpoint: {ckpt_path}")
        return None
    manifest = _read_manifest(checkpoint_dir)

    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        model, meta = load_model_from_checkpoint(ckpt_path)
        with mlflow.start_run(run_id=run_id):
            return register_pytorch_model(
                model=model,
                manifest={**manifest, **meta},
                registered_name=name,
            )
    except Exception as exc:
        print(f"[mlflow] pytorch register failed ({exc}); trying artifact URI")
        return register_run_artifact(run_id=run_id, registered_name=name)


def _read_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def register_run_artifact(*, run_id: str, registered_name: str) -> str | None:
    """Register best_model.pt already logged on the run (no local torch load)."""
    uri = tracking_uri()
    if not uri:
        return None
    try:
        from mlflow import MlflowClient
        from mlflow.exceptions import RestException

        client = MlflowClient(tracking_uri=uri)
        try:
            client.create_registered_model(registered_name)
        except RestException:
            pass
        source = f"runs:/{run_id}/best_model.pt"
        mv = client.create_model_version(
            name=registered_name,
            source=source,
            run_id=run_id,
        )
        print(f"[mlflow] registered artifact {source} as {registered_name} v{mv.version}")
        return registered_name
    except Exception as exc:
        print(f"[mlflow] artifact register skip ({exc})")
        return None
