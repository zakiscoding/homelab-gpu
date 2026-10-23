"""
Minimal QLoRA train for Phase 3 — RTX 5070 12GB.

Uses SmolLM2-360M by default; optional 4-bit via bitsandbytes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # homelab-gpu root
_TRAINING = Path(__file__).resolve().parents[1]
DEFAULT_BASE = os.environ.get("TRAIN_BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
MAX_STEPS = int(os.environ.get("TRAIN_MAX_STEPS", "30"))
DATASET_DIR = Path(os.environ.get("DATASET_DIR", r"C:\Users\zakar\homelab-data\datasets"))


def stop_ollama() -> None:
    script = _REPO / "scripts" / "stop-ollama.ps1"
    if sys.platform == "win32" and script.exists():
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            check=False,
            creationflags=flags,
        )


def _resolve_dataset(payload: dict) -> Path:
    version = payload.get("dataset_version")
    if version:
        path = DATASET_DIR / version / "train.jsonl"
        if path.exists():
            return path
    return Path(__file__).resolve().parent / "smoke_train.jsonl"


def run_qlora(
    job_id: str,
    out_dir: Path,
    payload: dict,
) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

    from training.cpu_io import configure_cpu_for_training, resolve_num_workers

    cpu_info = configure_cpu_for_training()
    print("[cpu-io]", cpu_info)

    stop_ollama()
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        return _load_manifest(out_dir, job_id, payload, skipped=True)

    data_path = _resolve_dataset(payload)
    ds = load_dataset("json", data_files=str(data_path), split="train")

    model_name = payload.get("base_model") or DEFAULT_BASE
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = os.environ.get("TRAIN_USE_4BIT", "true").lower() == "true"
    model_kwargs: dict = {"device_map": "auto"}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            use_4bit = False

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=int(payload.get("lora_r", 8)),
        lora_alpha=int(payload.get("lora_alpha", 16)),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(payload.get("max_seq_len", 512)),
            padding="max_length",
        )
        # Causal LM: labels = input_ids for Trainer loss
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    map_workers = max(1, min(4, resolve_num_workers() // 2))
    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=map_workers,
    )

    dl_workers = resolve_num_workers()
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer_out"),
        max_steps=min(MAX_STEPS, len(tokenized)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=MAX_STEPS,
        save_total_limit=1,
        report_to=[],
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=dl_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "job_id": job_id,
        "base_model": model_name,
        "dataset_version": payload.get("dataset_version") or "smoke",
        "model_version": payload.get("model_version") or f"adapter-{job_id[:8]}",
        "adapter_path": str(adapter_dir),
        "train_mode": "qlora",
        "use_4bit": use_4bit,
        "max_steps": args.max_steps,
        "host": os.environ.get("COMPUTERNAME", "gpu"),
        "cpu_io": cpu_info,
        "dataloader_num_workers": dl_workers,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_manifest(out_dir: Path, job_id: str, payload: dict, *, skipped: bool) -> dict:
    path = out_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "job_id": job_id,
        "dataset_version": payload.get("dataset_version"),
        "model_version": payload.get("model_version"),
        "skipped_retrain": skipped,
    }
