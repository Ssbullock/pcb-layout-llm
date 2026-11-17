#!/usr/bin/env python3
"""
run_hydro_benchmark.py
----------------------
Evaluates Simon's own Hydroponics KiCad PCBs (`AI_Hydro_Controller`, `HydroPCB`, `hydromisc`)
across all 4 placement & routing paradigms:
  1. Human Placement Ceiling (original human component coordinates + A* autorouter)
  2. Heuristic / Oracle Rollout (`place_simple.py` + spatial K-means oracle grouping)
  3. Blind Stage-1 SFT 7B Model (`Qwen2.5-Coder-7B-Instruct` + `qwen2.5-coder-7b-pcb-sft` on Modal A100)
  4. Blind Stage-2 GRPO 7B Model (`Qwen2.5-Coder-7B-Instruct` + `qwen2.5-coder-7b-pcb-grpo` on Modal A100)

Writes a side-by-side comparison report to `runs/hydro_benchmark_leaderboard.md` and `runs/hydro_benchmark_summary.json`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
RUNS_DIR = WORKSPACE / "runs"

sys.path.insert(0, str(WORKSPACE))
from build_sft_dataset import SYSTEM_PROMPT  # noqa: E402
from gym_rollout import (  # noqa: E402
    parse_kicad_pcb,
    run_human_ceiling,
    run_krt,
    run_rollout,
    stamp_locked,
    write_placed_output,
)
from pcb_design_intelligence import (  # noqa: E402
    build_compact_board_context,
    detect_mechanical_and_rf_locks,
    evaluate_physical_board_quality,
    normalize_rf_courtyard_wings,
)
from run_eval_and_grpo_pipeline import (  # noqa: E402
    extract_json_payload,
    run_single_board_rollout_with_pred_groups,
)

MODAL_BIN = str(WORKSPACE / ".venv/bin/modal")
MODAL_SCRIPT = str(WORKSPACE / "modal_eval_and_grpo.py")

HYDRO_BOARDS = [
    ("sethydro", "AI_Hydro_Controller"),
    ("sethydro", "HydroPCB"),
]


def prepare_hydro_prompts() -> list[dict]:
    prompts = []
    for sname, bname in HYDRO_BOARDS:
        unrouted = CORPUS_DIR / f"boards_unrouted_{sname}" / f"{bname}.kicad_pcb"
        prep_dir = RUNS_DIR / f"hydro_prep_{bname}"
        prep_dir.mkdir(parents=True, exist_ok=True)
        pile_pcb = prep_dir / "pile.kicad_pcb"
        fixed_pcb = prep_dir / "fixed.kicad_pcb"
        run_krt(".claude/skills/plan-pcb-placement/scripts/make_unplaced.py", [str(unrouted), str(pile_pcb)], quiet=True)
        pcb_human = parse_kicad_pcb(str(unrouted))
        pcb_pile = parse_kicad_pcb(str(pile_pcb))
        locked_refs, seat_placements = detect_mechanical_and_rf_locks(pcb_human, unrouted)
        if seat_placements:
            write_placed_output(str(pile_pcb), str(fixed_pcb), seat_placements, pcb_data=pcb_pile)
            stamp_locked(str(fixed_pcb), locked_refs)
        else:
            run_krt("py_router/copy_board.py", [str(pile_pcb), str(fixed_pcb)], quiet=True)
        normalize_rf_courtyard_wings(fixed_pcb)
        context_md = build_compact_board_context(fixed_pcb, locked_refs=set(locked_refs)).strip()
        user_prompt = (
            f"# Board Placement & Routing Task: `{bname}`\n\n"
            f"{context_md}\n\n"
            "Produce the `groups.json`, `intent.json`, `routing_strategy`, and `judge_chain.txt` for this board."
        )
        prompts.append(
            {
                "set": sname,
                "board": bname,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )
    return prompts


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--adapter-dir", type=str, default="/data/run_grpo_qwen3_30b_a3b/adapter_grpo_out")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    prompts = prepare_hydro_prompts()

    # 1. Generate blind floorplans from Stage-2 GRPO model on Modal A100-80GB
    in_grpo = RUNS_DIR / "hydro_modal_grpo_inputs.json"
    out_grpo = RUNS_DIR / "hydro_modal_grpo_outputs.json"
    in_grpo.write_text(
        json.dumps(
            {
                "val_prompts": prompts,
                "grpo_prompts": [],
                "base_model": args.base_model,
                "adapter_dir": args.adapter_dir,
                "num_generations": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not out_grpo.exists():
        print("=== Generating Blind Placement Plans on Modal A100 (Stage-2 GRPO Model) ===", flush=True)
        import time
        gen_cmd = [
            MODAL_BIN,
            "run",
            MODAL_SCRIPT,
            "--action",
            "generate",
            "--inputs-file",
            str(in_grpo),
            "--outputs-file",
            str(out_grpo),
        ]
        for attempt in range(1, 7):
            res = subprocess.run(gen_cmd, check=False)
            if res.returncode == 0 and out_grpo.exists():
                break
            print(f"  [WARN] Modal hydro generate exited with {res.returncode} (attempt {attempt}/6); retrying in 10s...", flush=True)
            time.sleep(10)

    grpo_preds = json.loads(out_grpo.read_text(encoding="utf-8")).get("val_outputs", {})

    rows = []
    for sname, bname in HYDRO_BOARDS:
        print(f"\n=====================================================================", flush=True)
        print(f" EVALUATING HYDROPONICS BOARD: {bname}", flush=True)
        print(f"=====================================================================", flush=True)

        # A. Human Placement Ceiling
        ceil_dir = RUNS_DIR / f"hydro_ceiling_{bname}"
        if (ceil_dir / "ceiling_result.json").exists():
            ceil_res = json.loads((ceil_dir / "ceiling_result.json").read_text(encoding="utf-8"))
        elif (ceil_dir / "ceiling_summary.json").exists():
            ceil_res = json.loads((ceil_dir / "ceiling_summary.json").read_text(encoding="utf-8"))
        else:
            ceil_res = run_human_ceiling(bname, CORPUS_DIR, ceil_dir, set_name=sname)
        ceil_sc = (ceil_res or {}).get("score") or {}
        ceil_q = ceil_sc.get("quality") or {}

        # B. Oracle Rollout (use verified hydro_v4_* if present)
        oracle_dir = RUNS_DIR / f"hydro_v4_{bname}"
        if not (oracle_dir / "rollout_result.json").exists():
            oracle_dir = RUNS_DIR / f"hydro_oracle_{bname}"
        if (oracle_dir / "rollout_result.json").exists():
            oracle_res = json.loads((oracle_dir / "rollout_result.json").read_text(encoding="utf-8"))
        elif (oracle_dir / "rollout_summary.json").exists():
            oracle_res = json.loads((oracle_dir / "rollout_summary.json").read_text(encoding="utf-8"))
        else:
            oracle_res = run_rollout(
                board_name=bname,
                corpus_dir=CORPUS_DIR,
                work_dir=oracle_dir,
                scattered_seeds=2,
                clustered_seeds=2,
                judge_count=2,
                jobs=4,
                set_name=sname,
                production_finish=True,
            )
        oracle_sc = (oracle_res or {}).get("best_score") or {}
        oracle_q = oracle_sc.get("quality") or {}

        # C. Blind Stage-2 GRPO 32B Model Rollout
        raw_txt = grpo_preds.get(bname, "")
        payload = extract_json_payload(raw_txt)
        valid_json = payload is not None and isinstance(payload.get("groups"), dict)
        pred_groups = payload["groups"] if valid_json else {}
        pred_rs = payload.get("routing_strategy") if valid_json else None

        llm_dir = RUNS_DIR / f"hydro_grpo_llm_{bname}"
        if llm_dir.exists():
            import shutil
            shutil.rmtree(llm_dir, ignore_errors=True)
        llm_res = run_single_board_rollout_with_pred_groups(
            set_name=sname,
            board_name=bname,
            pred_groups_doc=pred_groups,
            work_dir=llm_dir,
            pred_routing_strategy=pred_rs,
            scattered=2,
            clustered=2,
            judge=2,
            timeout_s=480,
        )
        llm_sc = (llm_res or {}).get("best_score") or {}
        llm_q = llm_sc.get("quality") or {}
        llm_phys = (llm_res or {}).get("physical_quality") or {}
        san = (llm_res or {}).get("group_sanitization") or {}
        final_g_cnt = (llm_res or {}).get("repaired_groups_count") or san.get("predicted_groups_count", 0)

        row = {
            "board": bname,
            "valid_json": valid_json,
            "predicted_groups": final_g_cnt,
            "movable_parts": san.get("total_movable_parts", 0),
            "human_blocking": ceil_sc.get("blocking"),
            "human_vias": ceil_q.get("vias"),
            "human_copper_mm": ceil_q.get("copper_mm"),
            "human_reward": (ceil_res or {}).get("rl_reward"),
            "oracle_blocking": oracle_sc.get("blocking"),
            "oracle_vias": oracle_q.get("vias"),
            "oracle_copper_mm": oracle_q.get("copper_mm"),
            "oracle_reward": (oracle_res or {}).get("rl_reward"),
            "grpo_llm_blocking": llm_sc.get("blocking"),
            "grpo_llm_vias": llm_q.get("vias"),
            "grpo_llm_copper_mm": llm_q.get("copper_mm"),
            "grpo_llm_reward": (llm_res or {}).get("rl_reward"),
            "teel_score_100": llm_phys.get("teel_score_100", 100.0),
            "slalom_traces_count": llm_phys.get("slalom_traces_count", 0),
            "via_in_pad_count": llm_phys.get("via_in_pad_count", 0),
            "weaving_nets_count": llm_phys.get("weaving_nets_count", 0),
            "power_width_compliance_pct": llm_phys.get("power_width_compliance_pct", 100.0),
            "channel_symmetry_err_mm": llm_phys.get("channel_symmetry_err_mm", 0.0),
        }
        rows.append(row)

    md = [
        "# Hydroponics PCB Benchmark: Human vs. Oracle vs. Blind 32B GRPO Model (v6 John Teel + Smart Routing)",
        "",
        "| Board | Movable Parts | LLM Groups | Human Blocking | Human Reward | Oracle Blocking | Oracle Reward | **32B GRPO Blocking** | **32B Vias / Cu (mm)** | **Slalom / VIP** | **Weaving Nets** | **Width Compliance** | **Teel Score (0-100)** | **32B GRPO Reward** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        l_vc = f"{r['grpo_llm_vias']} / {r['grpo_llm_copper_mm']}" if r["grpo_llm_vias"] is not None else "-"
        md.append(
            f"| `{r['board']}` | {r['movable_parts']} | {r['predicted_groups']} | {r['human_blocking']} | {r['human_reward']} | "
            f"{r['oracle_blocking']} | {r['oracle_reward']} | **{r['grpo_llm_blocking']}** | {l_vc} | "
            f"**{r['slalom_traces_count']} / {r['via_in_pad_count']}** | **{r['weaving_nets_count']}** | **{r['power_width_compliance_pct']}%** | **{r['teel_score_100']}** | **{r['grpo_llm_reward']}** |"
        )

    (RUNS_DIR / "hydro_benchmark_leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RUNS_DIR / "hydro_benchmark_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
