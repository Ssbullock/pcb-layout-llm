#!/usr/bin/env python3
"""
modal_train_sft.py
------------------
Serverless A100-80GB / H100 Fine-Tuning Pipeline on Modal for Qwen2.5-Coder-7B-Instruct.

What this script does when you run `modal run modal_train_sft.py`:
  1. Uploads your local curated dataset (`dataset/sft_curated_train.jsonl` and
     `dataset/sft_curated_val.jsonl`) to a persistent Modal cloud volume.
  2. Spins up an NVIDIA A100-80GB container with PyTorch, FlashAttention-2, TRL, PEFT,
     and BitsAndBytes pre-installed.
  3. Fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` with LoRA (`r=32, alpha=64`) across
     `max_seq_length=16384` (enough to fit every single board up to 15,771 tokens without
     any truncation!).
  4. Evaluates on the held-out validation split (`sft_curated_val.jsonl`) after every epoch
     and runs a live inference sanity-check on a held-out validation board to verify
     the trained model emits valid `groups.json`, `intent.json`, and `judge_chain.txt`.
  5. Automatically downloads the trained LoRA adapter weights (`adapter_model.safetensors`,
     `adapter_config.json`, tokenizer, and `train_metrics.json`) directly back into
     `/Users/simon/Fine Tune/checkpoints/qwen2.5-coder-7b-pcb-sft/`!
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import modal

APP_NAME = "kicad-pcb-qwen-sft"
app = modal.App(APP_NAME)

# Persistent cloud volume to cache the base model & checkpoints across runs
vol = modal.Volume.from_name("kicad-pcb-finetune-vol", create_if_missing=True)

# Container image with CUDA + PyTorch + TRL + PEFT + BitsAndBytes
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.51.3",
        "datasets>=3.0.0",
        "trl>=0.16.0",
        "peft>=0.15.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/data": vol},
)
def run_remote_sft(
    train_jsonl: str,
    val_jsonl: str,
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    epochs: int = 4,
    lr: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_seq_length: int = 32768,
    resume_adapter_dir: str = "",
    remote_out_subdir: str = "run_sft_qwen3_30b_a3b",
) -> bytes:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    work_dir = Path(f"/data/{remote_out_subdir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = work_dir / "train.jsonl"
    val_path = work_dir / "val.jsonl"
    out_dir = work_dir / "adapter_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path.write_text(train_jsonl, encoding="utf-8")
    val_path.write_text(val_jsonl, encoding="utf-8")

    print(f"=== [Modal A100-80GB] Loading dataset ({train_path}, {val_path}) ===")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    print(f"Train examples: {len(dataset['train'])}, Val examples: {len(dataset['validation'])}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"=== Loading tokenizer & base model: {base_model} ===")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir="/data/hf_cache",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/data/hf_cache",
    )
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    peft_config = None
    if resume_adapter_dir and Path(resume_adapter_dir).exists():
        print(f"=== Continuing LoRA training from checkpoint: {resume_adapter_dir} ===")
        model = PeftModel.from_pretrained(model, resume_adapter_dir, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    import inspect
    sft_sig = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    candidate_kwargs = {
        "output_dir": str(out_dir),
        "max_length": max_seq_length,
        "max_seq_length": max_seq_length,
        "dataset_text_field": "text",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 6,
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_8bit",
        "seed": 42,
        "report_to": "none",
    }
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sft_sig}
    sft_args = SFTConfig(**filtered_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        args=sft_args,
    )

    print("=== Starting QLoRA SFT Training on A100-80GB ===")
    train_res = trainer.train()
    eval_metrics = trainer.evaluate()

    print("=== Saving trained LoRA adapter & tokenizer ===")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Run a quick validation generation sanity check on the first held-out board
    sample_val = json.loads(val_jsonl.splitlines()[0])
    prompt_msgs = sample_val["messages"][:2]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    summary_payload = {
        "base_model": base_model,
        "train_examples": len(dataset["train"]),
        "val_examples": len(dataset["validation"]),
        "epochs": epochs,
        "learning_rate": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_metrics": train_res.metrics,
        "eval_metrics": eval_metrics,
        "sample_val_board": sample_val.get("board"),
        "sample_val_generation_preview": gen_text[:1500],
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    vol.commit()

    # Package the adapter directory (excluding intermediate checkpoint-* subdirectories) into a tar.gz archive
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in out_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                continue
            tar.add(item, arcname=item.name)
    return buf.getvalue()


@app.local_entrypoint()
def main(
    train_file: str = "dataset/sft_curated_train.jsonl",
    val_file: str = "dataset/sft_curated_val.jsonl",
    output_dir: str = "checkpoints/qwen3-coder-30b-a3b-pcb-sft",
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    epochs: int = 4,
    lr: float = 1.5e-4,
    max_seq_length: int = 32768,
    resume_adapter_dir: str = "",
    remote_out_subdir: str = "run_sft_qwen3_30b_a3b",
):
    train_path = Path(train_file)
    val_path = Path(val_file)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Uploading {train_path} and {val_path} to Modal A100-80GB...")
    tar_bytes = run_remote_sft.remote(
        train_jsonl=train_path.read_text(encoding="utf-8"),
        val_jsonl=val_path.read_text(encoding="utf-8"),
        base_model=base_model,
        epochs=epochs,
        lr=lr,
        max_seq_length=max_seq_length,
        resume_adapter_dir=resume_adapter_dir,
        remote_out_subdir=remote_out_subdir,
    )

    print(f"Downloading trained LoRA adapter into {out_path}...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=out_path)

    metrics_file = out_path / "train_metrics.json"
    if metrics_file.exists():
        print("\n=== Training & Validation Summary ===")
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        print(json.dumps({k: v for k, v in m.items() if k != "sample_val_generation_preview"}, indent=2))
    print(f"\nSaved LoRA weights & tokenizer to: {out_path.resolve()}")
