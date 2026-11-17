#!/usr/bin/env python3
"""
modal_eval_and_grpo.py
----------------------
Runs BOTH:
  1. Task 1 (End-to-End Held-Out Blind Physics Evaluation):
     Loads the fine-tuned `Qwen2.5-Coder-7B-Instruct` + Stage-1 LoRA adapter on Modal A100-80GB,
     generates blind placement + routing plans (`groups.json`, `intent.json`, `judge_chain.txt`)
     for all 16 held-out validation boards (using ONLY the unplaced `board_context.md`), and
     returns them to your Mac so `gym_rollout.py` can place, route, and grade them against
     the Oracle and Human Placement Ceiling.

  2. Task 2 (Stage-2 GRPO / Best-of-N Online RL Rollout & Policy Refinement):
     For hard boards where the single-pass heuristic left open/blocked nets (`blocking > 0`)
     or trailed the human ceiling, samples G=4 diverse placement plans (`temperature=0.75`)
     from the SFT policy, runs all G=4 through the real `gym_rollout.py` physics engine
     locally, computes Group-Relative Policy Optimization (GRPO) advantages:
         A_i = (R_i - mean(R)) / (std(R) + 1e-4)
     and executes the Stage-2 GRPO / advantage-weighted RL update on Modal A100-80GB, saving
     the final Stage-2 adapter to `checkpoints/qwen2.5-coder-7b-pcb-grpo/`.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import modal

APP_NAME = "kicad-pcb-qwen-eval-grpo"
app = modal.App(APP_NAME)
vol = modal.Volume.from_name("kicad-pcb-finetune-vol", create_if_missing=True)

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
        "vllm>=0.9.0",
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
def generate_plans_on_a100(
    val_prompts: list[dict],
    grpo_prompts: list[dict],
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    adapter_dir: str = "/data/run_sft_qwen3_30b_a3b/adapter_out",
    num_generations: int = 4,
    use_vllm: bool = True,
) -> dict:
    import hashlib
    import torch
    from transformers import AutoConfig, AutoTokenizer

    has_adapter = bool(adapter_dir) and Path(adapter_dir).exists()
    tok_src = adapter_dir if has_adapter else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, cache_dir="/data/hf_cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_sig = hashlib.md5(
        json.dumps({"v": val_prompts, "g": grpo_prompts, "a": adapter_dir, "m": base_model}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    ckpt_path = Path(f"/data/eval_grpo_generations_ckpt_{prompt_sig}.json")
    if ckpt_path.exists():
        try:
            saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
    else:
        saved = {}

    val_outputs = saved.get("val_outputs", {})
    grpo_outputs = saved.get("grpo_outputs", {})

    # Prefer vLLM with prefix caching + native MoE throughput (3.3B active params on Qwen3-Coder-A3B)
    vllm_ok = False
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            llm = LLM(
                model=base_model,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=32768,
                enable_prefix_caching=True,
                enable_lora=has_adapter,
                max_lora_rank=64,
                gpu_memory_utilization=0.90,
                download_dir="/data/hf_cache",
            )
            lora_req = LoRARequest("pcb_sft", 1, adapter_dir) if has_adapter else None
            greedy_params = SamplingParams(temperature=0.0, max_tokens=8192)
            explore_params = SamplingParams(
                temperature=0.75,
                top_p=0.92,
                n=max(1, num_generations - 1),
                max_tokens=8192,
            )

            print(f"=== [vLLM] Generating blind plans for {len(val_prompts)} validation boards ===")
            pending_val = [item for item in val_prompts if not val_outputs.get(item["board"])]
            if pending_val:
                val_texts = [
                    tokenizer.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
                    for it in pending_val
                ]
                outs = llm.generate(val_texts, greedy_params, lora_request=lora_req)
                for it, out in zip(pending_val, outs):
                    val_outputs[it["board"]] = out.outputs[0].text if out.outputs else ""
                ckpt_path.write_text(json.dumps({"val_outputs": val_outputs, "grpo_outputs": grpo_outputs}), encoding="utf-8")
                vol.commit()

            print(f"=== [vLLM] Sampling G={num_generations} GRPO candidates for {len(grpo_prompts)} boards ===")
            pending_grpo = [
                it for it in grpo_prompts
                if len(grpo_outputs.get(it["board"], [])) < num_generations
            ]
            if pending_grpo:
                g_texts = [
                    tokenizer.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
                    for it in pending_grpo
                ]
                greedy_outs = llm.generate(g_texts, greedy_params, lora_request=lora_req)
                sample_outs = (
                    llm.generate(g_texts, explore_params, lora_request=lora_req)
                    if num_generations > 1
                    else None
                )
                for idx_g, it in enumerate(pending_grpo):
                    cands = [greedy_outs[idx_g].outputs[0].text]
                    if sample_outs is not None:
                        cands.extend(o.text for o in sample_outs[idx_g].outputs)
                    grpo_outputs[it["board"]] = cands[:num_generations]
                ckpt_path.write_text(json.dumps({"val_outputs": val_outputs, "grpo_outputs": grpo_outputs}), encoding="utf-8")
                vol.commit()
            vllm_ok = True
        except Exception as exc:
            print(f"  [WARN] vLLM rollout path unavailable ({exc}); falling back to Transformers generate.", flush=True)

    if vllm_ok:
        return {"val_outputs": val_outputs, "grpo_outputs": grpo_outputs}

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"=== Loading Base Model ({base_model}) + Stage-1 LoRA Adapter ({adapter_dir}) ===")
    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True, cache_dir="/data/hf_cache")
    if getattr(cfg, "max_position_embeddings", 32768) < 65536:
        cfg.max_position_embeddings = 65536

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=cfg,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/data/hf_cache",
    )
    model = PeftModel.from_pretrained(base, adapter_dir) if has_adapter else base
    model.eval()

    print(f"=== Generating blind placement plans for {len(val_prompts)} held-out validation boards ===")
    for idx, item in enumerate(val_prompts, 1):
        bname = item["board"]
        if bname in val_outputs and val_outputs[bname]:
            print(f"  [Val {idx}/{len(val_prompts)}] {bname}: loaded from volume cache ({len(val_outputs[bname])} chars)")
            continue
        prompt_text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=6144,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        val_outputs[bname] = gen_text
        ckpt_path.write_text(json.dumps({"val_outputs": val_outputs, "grpo_outputs": grpo_outputs}), encoding="utf-8")
        vol.commit()
        print(f"  [Val {idx}/{len(val_prompts)}] {bname}: generated {len(gen_text)} chars")

    # 2. GRPO multi-sample exploration (G=num_generations diverse completions per hard board)
    print(f"=== Sampling G={num_generations} candidate plans for {len(grpo_prompts)} GRPO boards ===")
    for idx, item in enumerate(grpo_prompts, 1):
        bname = item["board"]
        if bname in grpo_outputs and len(grpo_outputs[bname]) >= num_generations:
            print(f"  [GRPO {idx}/{len(grpo_prompts)}] {bname}: loaded {len(grpo_outputs[bname])} completions from volume cache")
            continue
        prompt_text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        candidates = []
        # Candidate 0: greedy baseline from the SFT policy; Candidates 1..G-1: batched stochastic exploration on A100
        with torch.no_grad():
            gen_ids_0 = model.generate(
                **inputs,
                max_new_tokens=6144,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            candidates.append(tokenizer.decode(gen_ids_0[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

            if num_generations > 1:
                gen_ids_rest = model.generate(
                    **inputs,
                    max_new_tokens=6144,
                    do_sample=True,
                    temperature=0.75,
                    top_p=0.92,
                    num_return_sequences=num_generations - 1,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for seq in gen_ids_rest:
                    candidates.append(tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        grpo_outputs[bname] = candidates
        ckpt_path.write_text(json.dumps({"val_outputs": val_outputs, "grpo_outputs": grpo_outputs}), encoding="utf-8")
        vol.commit()
        print(f"  [GRPO {idx}/{len(grpo_prompts)}] {bname}: sampled {len(candidates)} completions")

    return {"val_outputs": val_outputs, "grpo_outputs": grpo_outputs}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/data": vol},
)
def run_stage2_grpo_update_on_a100(
    grpo_trajectories_jsonl: str,
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    sft_adapter_dir: str = "/data/run_sft_qwen3_30b_a3b/adapter_out",
    remote_grpo_subdir: str = "run_grpo_qwen3_30b_a3b",
    lr: float = 5e-5,
    epochs: int = 2,
) -> bytes:
    """
    Executes the Stage-2 GRPO / Advantage-Weighted Policy Optimization update on Modal A100-80GB
    using the positive-advantage trajectories discovered by our local KiCad physics engine.
    """
    import inspect
    import torch
    from datasets import load_dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    work_dir = Path(f"/data/{remote_grpo_subdir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    traj_path = work_dir / "grpo_winning_trajectories.jsonl"
    out_dir = work_dir / "adapter_grpo_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path.write_text(grpo_trajectories_jsonl, encoding="utf-8")
    dataset = load_dataset("json", data_files={"train": str(traj_path)})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dynamically expand RoPE / max_position_embeddings if prompt + 4096 exceeds 32768
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True, cache_dir="/data/hf_cache")
    if getattr(cfg, "max_position_embeddings", 32768) < 65536:
        cfg.max_position_embeddings = 65536

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=cfg,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/data/hf_cache",
    )
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, sft_adapter_dir, is_trainable=True)

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    sft_sig = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    candidate_kwargs = {
        "output_dir": str(out_dir),
        "max_length": 32768,
        "max_seq_length": 32768,
        "dataset_text_field": "text",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 2,
        "logging_steps": 1,
        "save_strategy": "no",
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
        args=sft_args,
    )

    print(f"=== Running Stage-2 GRPO Policy Update on {len(dataset['train'])} winning rollouts ===")
    train_res = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    (out_dir / "grpo_train_metrics.json").write_text(
        json.dumps(
            {
                "stage": "Stage-2 GRPO Policy Update",
                "winning_rollouts_trained": len(dataset["train"]),
                "epochs": epochs,
                "learning_rate": lr,
                "train_metrics": train_res.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    vol.commit()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in out_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                continue
            tar.add(item, arcname=item.name)
    return buf.getvalue()


@app.local_entrypoint()
def main(
    action: str = "generate",
    inputs_file: str = "runs/modal_eval_grpo_inputs.json",
    outputs_file: str = "runs/modal_eval_grpo_outputs.json",
    trajectories_file: str = "dataset/grpo_winning_trajectories.jsonl",
    grpo_out_dir: str = "checkpoints/qwen3-coder-30b-a3b-pcb-grpo",
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    adapter_dir: str = "/data/run_sft_qwen3_30b_a3b/adapter_out",
    remote_grpo_subdir: str = "run_grpo_qwen3_30b_a3b",
):
    if action == "generate":
        payload = json.loads(Path(inputs_file).read_text(encoding="utf-8"))
        res = generate_plans_on_a100.remote(
            val_prompts=payload["val_prompts"],
            grpo_prompts=payload["grpo_prompts"],
            base_model=payload.get("base_model", base_model),
            adapter_dir=payload.get("adapter_dir", adapter_dir),
            num_generations=payload.get("num_generations", 4),
        )
        Path(outputs_file).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved Modal A100 generation outputs to {outputs_file}")
    elif action == "update":
        traj_jsonl = Path(trajectories_file).read_text(encoding="utf-8")
        tar_bytes = run_stage2_grpo_update_on_a100.remote(
            grpo_trajectories_jsonl=traj_jsonl,
            base_model=base_model,
            sft_adapter_dir=adapter_dir,
            remote_grpo_subdir=remote_grpo_subdir,
        )
        out_path = Path(grpo_out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(path=out_path)
        print(f"Saved Stage-2 GRPO LoRA adapter to {out_path.resolve()}")
