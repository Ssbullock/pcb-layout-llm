#!/usr/bin/env python3
"""
train_sft_cloud.py
------------------
Step 4B: Cloud GPU Stage-1 Supervised Fine-Tuning (SFT) for Qwen2.5-Coder on
KiCad PCB Placement & Routing (`dataset/sft_curated_train.jsonl`).

Designed for NVIDIA Cloud GPUs (RunPod, Modal, Lambda, Colab A100/H100/L40S/RTX 4090).
Automatically uses Unsloth 2x-faster kernels if installed, or falls back cleanly to
standard Hugging Face `trl.SFTTrainer` + `peft.LoraConfig`.

Usage on Cloud GPU:
  pip install -U torch transformers datasets trl peft accelerate bitsandbytes
  # Optional (recommended on single GPU): pip install unsloth

  python3 train_sft_cloud.py \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --train-file dataset/sft_curated_train.jsonl \
      --val-file dataset/sft_curated_val.jsonl \
      --output-dir checkpoints/qwen2.5-coder-7b-pcb-sft \
      --epochs 4 \
      --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Stage-1 LoRA SFT Trainer for KiCad PCB Placement LLM")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="Base HuggingFace model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct or 14B/32B)")
    ap.add_argument("--train-file", type=Path, default=Path("dataset/sft_curated_train.jsonl"))
    ap.add_argument("--val-file", type=Path, default=Path("dataset/sft_curated_val.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen2.5-coder-7b-pcb-sft"))
    ap.add_argument("--max-seq-length", type=int, default=12288,
                    help="Context window (median board is ~5K tokens, max ~10K tokens)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--load-in-4bit", action="store_true", default=True,
                    help="Use 4-bit QLoRA quantization (fits 7B/14B easily on a single 24GB GPU)")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    print(f"Loading SFT dataset from {args.train_file} and {args.val_file}...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.val_file),
        },
    )

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        pass

    if use_unsloth:
        print(f"[Unsloth] Loading {args.model} with max_seq_length={args.max_seq_length}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        print(f"[HF + PEFT] Loading {args.model} with max_seq_length={args.max_seq_length}...")
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = None
        if args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
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

    dataset = dataset.map(format_chat, num_proc=2)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=None if use_unsloth else peft_config,
        args=sft_config,
    )

    print("Starting Stage-1 SFT training...")
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metrics = train_result.metrics
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Stage-1 SFT complete! Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
