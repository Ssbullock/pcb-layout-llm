#!/usr/bin/env python3
"""
train_grpo_cloud.py
-------------------
Step 5: Stage-2 Group Relative Policy Optimization (GRPO) for KiCad PCB Placement & Routing.

Connects Hugging Face `trl.GRPOTrainer` directly to our deterministic physics verifier
(`gym_rollout.py` -> `check_groups.py` -> `place_simple.py` -> `judge_seeds.py` -> `quick_score.py`).

How it works:
  1. For each board prompt in `dataset/grpo_prompts.jsonl`, the model samples G=4 candidate
     placement plans (`groups.json` + `intent.json` + `judge_chain.txt`).
  2. Our multi-tier reward function evaluates each completion:
     - `format_and_schema_reward_func`: Fast syntactic + schema gate (+10 for valid JSON with
       required `groups`, `intent`, and `judge_chain` keys; -50 for invalid JSON).
     - `physics_rollout_reward_func`: Writes the model's `groups.json` and `intent.json` into a
       temporary rollout directory, runs `gym_rollout.py` (`place_simple.py` + `judge_seeds.py`),
       and returns the exact `rl_reward` (`+100 - 0.15*vias - 0.005*copper_mm` when `blocking == 0`,
       or `-35*blocking - 8*joins` when open nets remain).
  3. GRPO computes the group-relative advantage across the G=4 completions and updates the LoRA
     weights toward the placements that achieved fewer unrouted nets, zero overlaps, and lower via count.

Usage on Cloud GPU (or hybrid Cloud GPU + local/remote KiCadRoutingTools worker):
  python3 train_grpo_cloud.py \
      --model checkpoints/qwen2.5-coder-7b-pcb-sft \
      --prompts-file dataset/grpo_prompts.jsonl \
      --output-dir checkpoints/qwen2.5-coder-7b-pcb-grpo \
      --num-generations 4
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from gym_rollout import run_rollout


def extract_json_block(text: str) -> dict | None:
    """Parse JSON object from model completion (handles raw JSON or ```json fenced blocks)."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(s[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def format_and_schema_reward_func(completions, **kwargs) -> list[float]:
    """Fast structural reward: verifies valid JSON and required top-level placement keys."""
    rewards = []
    for comp in completions:
        txt = completion_to_text(comp)
        payload = extract_json_block(txt)
        if payload is None:
            rewards.append(-50.0)
            continue
        groups = payload.get("groups")
        intent = payload.get("intent")
        chain = payload.get("judge_chain")
        if not (isinstance(groups, dict) and isinstance(intent, dict) and isinstance(chain, list)):
            rewards.append(-25.0)
            continue
        grp_list = groups.get("groups")
        if not isinstance(grp_list, list) or len(grp_list) == 0:
            rewards.append(-15.0)
            continue
        # Valid schema bonus
        rewards.append(10.0)
    return rewards


def make_physics_rollout_reward_func(corpus_dir: Path, scattered: int = 1, clustered: int = 1, judge: int = 1):
    """Returns a TRL-compatible reward function that executes `gym_rollout.py` on each valid completion."""

    def physics_rollout_reward_func(prompts, completions, board, set, **kwargs) -> list[float]:
        rewards = []
        for idx, comp in enumerate(completions):
            bname = board[idx]
            sname = set[idx]
            txt = completion_to_text(comp)
            payload = extract_json_block(txt)
            if payload is None or not isinstance(payload.get("groups"), dict) or not isinstance(payload.get("intent"), dict):
                rewards.append(-200.0)
                continue

            with tempfile.TemporaryDirectory(prefix=f"grpo_{bname}_") as tmp_dir:
                tpath = Path(tmp_dir)
                custom_groups = tpath / "pred_groups.json"
                custom_intent = tpath / "pred_intent.json"
                custom_groups.write_text(json.dumps(payload["groups"], indent=2), encoding="utf-8")
                custom_intent.write_text(json.dumps(payload["intent"], indent=2), encoding="utf-8")

                try:
                    res = run_rollout(
                        board_name=bname,
                        corpus_dir=corpus_dir,
                        work_dir=tpath / "work",
                        scattered_seeds=scattered,
                        clustered_seeds=clustered,
                        judge_count=judge,
                        jobs=2,
                        custom_groups=custom_groups,
                        custom_intent=custom_intent,
                        set_name=sname,
                    )
                    r_val = float(res.get("rl_reward", -250.0))
                except Exception as exc:
                    print(f"  [GRPO Rollout Error on {bname}]: {exc}")
                    r_val = -250.0

                rewards.append(r_val)
        return rewards

    return physics_rollout_reward_func


def main():
    ap = argparse.ArgumentParser(description="Stage-2 GRPO RL Trainer for KiCad PCB Placement LLM")
    ap.add_argument("--model", type=str, default="checkpoints/qwen2.5-coder-7b-pcb-sft",
                    help="Path to Stage-1 SFT checkpoint or HuggingFace model ID")
    ap.add_argument("--prompts-file", type=Path, default=Path("dataset/grpo_prompts.jsonl"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen2.5-coder-7b-pcb-grpo"))
    ap.add_argument("--num-generations", type=int, default=4, help="Group size G for GRPO")
    ap.add_argument("--max-prompt-length", type=int, default=8192)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("json", data_files={"train": str(args.prompts_file)})["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_steps=25,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    physics_reward_fn = make_physics_rollout_reward_func(args.corpus)

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[format_and_schema_reward_func, physics_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting Stage-2 GRPO Reinforcement Learning...")
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Stage-2 GRPO complete! Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
