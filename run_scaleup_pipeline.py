#!/usr/bin/env python3
"""
run_scaleup_pipeline.py
-----------------------
End-to-End v5 32B Retraining Pipeline (`Qwen/Qwen2.5-Coder-32B-Instruct`):
  1. Rebuilds the v5 SFT & GRPO dataset (`dataset/sft_v5.jsonl`) across all 191 evaluated boards:
     - Compresses `board_context.md` from ~16K-35K tokens down to ~3.5K-4.5K tokens (zero >32K context overflow)
     - Enforces 1-IC-per-group splitting so major ICs are never merged into a single catch-all group
     - Embeds `templates`, `proximity`, `keepouts`, `--same-net-pad-clearance 0.15` (anti-via-in-pad),
       `--via-cost 110` (anti-layer-weaving), and `--power-nets` / `--power-nets-widths` in the targets.
  2. Curates the rejection-sampled train/val/GRPO splits (`sft_curated_train.jsonl`,
     `sft_curated_val.jsonl`, `grpo_prompts.jsonl`).
  3. Trains `Qwen/Qwen2.5-Coder-32B-Instruct` with Stage-1 QLoRA SFT on Modal `A100-80GB`
     (`max_seq_length=16384`, `epochs=3`, `lr=1.5e-4`) -> `checkpoints/qwen2.5-coder-32b-pcb-sft`.
  4. Runs Blind Held-Out Validation + Stage-2 Online GRPO Exploration (`G=4` candidate plans per
     hard board graded by our new multi-metric physical quality evaluator: layer weaving, via-in-pad,
     trace width compliance, and channel symmetry) + 32B RL Policy Update on Modal `A100-80GB`
     -> `checkpoints/qwen2.5-coder-32b-pcb-grpo`.
  5. Evaluates Simon's Hydroponics boards (`AI_Hydro_Controller` & `HydroPCB`) end-to-end with the
     newly trained 32B GRPO model (`--production-finish`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
MODAL_BIN = str(WORKSPACE / ".venv/bin/modal")
BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def main() -> None:
    curation_report = json.loads((WORKSPACE / "dataset/curation_report.json").read_text(encoding="utf-8"))
    print("\n=== Qwen3-Coder-30B-A3B-Instruct Dataset Curation Summary ===", flush=True)
    print(json.dumps({k: v for k, v in curation_report.items() if k != "rejected_boards"}, indent=2), flush=True)

    sft_metrics_path = WORKSPACE / "checkpoints/qwen3-coder-30b-a3b-pcb-sft/train_metrics.json"
    sft_done = False
    if sft_metrics_path.exists():
        try:
            m = json.loads(sft_metrics_path.read_text(encoding="utf-8"))
            sft_done = m.get("version") == "v8_qwen3_coder_a3b"
        except Exception:
            sft_done = False

    if not sft_done:
        print("\n=====================================================================", flush=True)
        print(f" STEP 2: Training Stage-1 QLoRA SFT Model ({BASE_MODEL}) on Modal A100-80GB", flush=True)
        print("=====================================================================", flush=True)
        import time
        sft_cmd = [
            MODAL_BIN,
            "run",
            str(WORKSPACE / "modal_train_sft.py"),
            "--base-model",
            BASE_MODEL,
            "--train-file",
            str(WORKSPACE / "dataset/sft_curated_train.jsonl"),
            "--val-file",
            str(WORKSPACE / "dataset/sft_curated_val.jsonl"),
            "--output-dir",
            str(WORKSPACE / "checkpoints/qwen3-coder-30b-a3b-pcb-sft"),
            "--epochs",
            "3",
            "--lr",
            "1.5e-4",
            "--max-seq-length",
            "32768",
            "--remote-out-subdir",
            "run_sft_qwen3_30b_a3b",
        ]
        for attempt in range(1, 6):
            res = subprocess.run(sft_cmd, check=False)
            if res.returncode == 0:
                if sft_metrics_path.exists():
                    m = json.loads(sft_metrics_path.read_text(encoding="utf-8"))
                    m["version"] = "v8_qwen3_coder_a3b"
                    sft_metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
                break
            print(f"  [WARN] Modal SFT exited with {res.returncode} (attempt {attempt}/5); retrying in 15s...", flush=True)
            time.sleep(15)
    else:
        print("\n=== Reusing completed Stage-1 Qwen3-Coder-A3B SFT checkpoint (/data/run_sft_qwen3_30b_a3b/adapter_out) ===", flush=True)

    print("\n=====================================================================", flush=True)
    print(" STEP 3: Running Blind Eval + Stage-2 Model-First GRPO Rollouts & RL Update (vLLM)", flush=True)
    print("=====================================================================", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(WORKSPACE / "run_eval_and_grpo_pipeline.py"),
            "--base-model",
            BASE_MODEL,
            "--adapter-dir",
            "/data/run_sft_qwen3_30b_a3b/adapter_out",
            "--remote-grpo-subdir",
            "run_grpo_qwen3_30b_a3b",
            "--grpo-out-dir",
            str(WORKSPACE / "checkpoints/qwen3-coder-30b-a3b-pcb-grpo"),
            "--model-first",
            "--force-generate",
        ],
        check=True,
    )

    print("\n=====================================================================", flush=True)
    print(" STEP 4: Blind Evaluation on Hydroponics Boards (Qwen3-Coder-A3B GRPO Model)", flush=True)
    print("=====================================================================", flush=True)
    hydro_out = WORKSPACE / "runs/hydro_modal_grpo_outputs.json"
    if hydro_out.exists():
        hydro_out.unlink()

    subprocess.run(
        [
            sys.executable,
            str(WORKSPACE / "run_hydro_benchmark.py"),
            "--base-model",
            BASE_MODEL,
            "--adapter-dir",
            "/data/run_grpo_qwen3_30b_a3b/adapter_grpo_out",
        ],
        check=True,
    )
    print("\n=== END-TO-END v7_FULL 254-BOARD 32B SFT + GRPO RETRAINING & HYDROPONICS BENCHMARK COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
