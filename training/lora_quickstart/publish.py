"""
Publish a completed train job to Ollama on zak_pc.

QLoRA adapters: merge to fp16 safetensors, then `ollama create` via Modelfile.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "ollama_bases.yaml"
_CHECKPOINTS = Path(
    os.environ.get("CHECKPOINT_DIR", r"C:\Users\zakar\homelab-data\checkpoints"),
)


def _load_bases() -> dict:
    if not _CONFIG.exists():
        return {"bases": {}}
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8")) or {"bases": {}}


def _resolve_job_dir(job_id: str) -> Path:
    root = _CHECKPOINTS / job_id
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"No manifest for job {job_id} under {root}")
    return root


def _ollama_exe() -> str:
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if local.exists():
        return str(local)
    cmd = subprocess.run(["where", "ollama"], capture_output=True, text=True, check=False)
    if cmd.returncode == 0 and cmd.stdout.strip():
        return cmd.stdout.strip().splitlines()[0]
    raise FileNotFoundError("ollama.exe not found")


def stop_ollama() -> None:
    script = _REPO / "scripts" / "stop-ollama.ps1"
    if sys.platform == "win32" and script.exists():
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            check=False,
        )


def start_ollama() -> None:
    script = _REPO / "scripts" / "start-ollama.ps1"
    if sys.platform == "win32" and script.exists():
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            check=False,
        )


def merge_adapter(adapter_dir: Path, base_model: str, merged_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (merged_dir / "config.json").exists():
        return merged_dir

    stop_ollama()
    merged_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def _write_modelfile(
    path: Path,
    *,
    from_line: str,
    adapter_dir: Path | None = None,
    system: str = "",
) -> None:
    lines = [f"FROM {from_line}"]
    if adapter_dir is not None:
        lines.append(f"ADAPTER {adapter_dir}")
    if system:
        lines.append(f'SYSTEM """{system}"""')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_job(
    job_id: str,
    *,
    ollama_name: str | None = None,
    skip_merge: bool = False,
    restart_ollama: bool = True,
) -> dict:
    job_dir = _resolve_job_dir(job_id)
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    base_model = manifest.get("base_model") or os.environ.get(
        "TRAIN_BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    model_version = manifest.get("model_version") or f"adapter-{job_id[:8]}"
    ollama_tag = ollama_name or f"homelab-{model_version}".replace(" ", "-").lower()

    adapter_dir = job_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter missing under {adapter_dir}")

    bases = _load_bases().get("bases", {})
    cfg = bases.get(base_model, {})
    ollama_base = cfg.get("ollama_base")
    use_adapter = bool(cfg.get("use_adapter")) and not skip_merge

    modelfile_dir = job_dir / "ollama_publish"
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile = modelfile_dir / "Modelfile"

    publish_mode = "merged"
    if use_adapter and ollama_base:
        publish_mode = "adapter"
        _write_modelfile(modelfile, from_line=ollama_base, adapter_dir=adapter_dir.resolve())
    else:
        merged_dir = job_dir / "merged"
        if not skip_merge or not (merged_dir / "config.json").exists():
            merge_adapter(adapter_dir, base_model, merged_dir)
        _write_modelfile(modelfile, from_line=str(merged_dir.resolve()))

    ollama = _ollama_exe()
    if publish_mode == "adapter" and ollama_base:
        pull = subprocess.run([ollama, "pull", ollama_base], check=False)
        if pull.returncode != 0:
            raise RuntimeError(f"ollama pull {ollama_base} failed")

    create = subprocess.run(
        [ollama, "create", ollama_tag, "-f", str(modelfile.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode != 0:
        raise RuntimeError(
            f"ollama create failed ({create.returncode}): {create.stderr or create.stdout}"
        )

    if restart_ollama:
        start_ollama()

    result = {
        "job_id": job_id,
        "ollama_model": ollama_tag,
        "publish_mode": publish_mode,
        "base_model": base_model,
        "modelfile": str(modelfile),
    }
    (job_dir / "published_ollama.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish train job adapter to Ollama")
    parser.add_argument("job_id", help="Train job UUID")
    parser.add_argument("--name", help="Ollama model tag (default homelab-{model_version})")
    parser.add_argument("--skip-merge", action="store_true", help="Use ADAPTER path only")
    parser.add_argument("--no-restart-ollama", action="store_true")
    args = parser.parse_args()
    out = publish_job(
        args.job_id,
        ollama_name=args.name,
        skip_merge=args.skip_merge,
        restart_ollama=not args.no_restart_ollama,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
