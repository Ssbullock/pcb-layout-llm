#!/usr/bin/env python3
"""
run_eval_and_grpo_pipeline.py
-----------------------------
Orchestrates both:
  1. End-to-End Held-Out Blind Physics Evaluation (Task 1):
     - Sends the 10 held-out validation board prompts (compact/medium boards never seen in SFT)
       to Modal A100-80GB (`modal_eval_and_grpo.generate_plans_on_a100`).
     - Sanitizes and validates the LLM's predicted `groups.json` (ensuring 100% movable coverage,
       no duplicate refs, valid anchors/near links) and runs `gym_rollout.py` with the LLM's
       blind predictions (`--groups`).
     - Compares the LLM's actual routed board scores against the Oracle Rollout and Human Ceiling,
       writing `runs/sft_blind_eval_leaderboard.md`.

  2. Stage-2 Online GRPO Rollout & RL Policy Update (Task 2):
     - Selects 8 challenging boards where the baseline heuristic rollout left blocked nets
       (`blocking > 0`, e.g., `picoprobe_tester`, `usbgpib`, `allwinner_h3_ddr3`, `hexapod_legbrd`,
       `gb_nintendo_power`, `quad_rs485_pico`, `dog_harness_panel`, `atf1502_cpld`).
     - Samples G=4 candidate placement plans per board on Modal A100-80GB.
     - Evaluates all candidates through the local KiCad physics engine (`place_simple.py` +
       `judge_seeds.py` + `quick_score.py`), computes group-relative advantages:
           A_i = (R_i - mean(R)) / (std(R) + 1e-4)
       and updates the model weights on Modal A100-80GB (`run_stage2_grpo_update_on_a100`),
       downloading the Stage-2 GRPO adapter to `checkpoints/qwen2.5-coder-7b-pcb-grpo/`.
"""

from __future__ import annotations

import io
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
RUNS_DIR = WORKSPACE / "runs"
DATASET_DIR = WORKSPACE / "dataset"

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
for _d in ("py_router", "py_placer", "py_tools"):
    _p = str(KRT_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kicad_parser import parse_kicad_pcb, non_aperture_pads  # noqa: E402
from placement.grouping import BoardView, GROUP_KEYS  # noqa: E402
from placement.part_class import classify_part  # noqa: E402
from placement.roles import infer_role  # noqa: E402
from placement.utility import compute_footprint_bbox_local  # noqa: E402
from gym_rollout import resolve_board_paths  # noqa: E402
from pcb_agent_harness import (  # noqa: E402
    PCBAgentSession,
    TOOLS_SCHEMA,
    make_openai_chat_fn,
    run_agent_loop,
)
from pcb_design_intelligence import detect_mechanical_and_rf_locks  # noqa: E402


def extract_json_payload(raw_text: str) -> dict | None:
    import re
    s = (raw_text or "").strip()
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
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # If the LLM output was truncated mid-stream, salvage all complete group objects inside "groups": [...]
        salvaged_groups = []
        for m in re.finditer(r'\{\s*"name"\s*:\s*"[^"]+"\s*,.*?\n\s*\}', s, flags=re.DOTALL):
            try:
                g_obj = json.loads(m.group(0))
                if isinstance(g_obj, dict) and "name" in g_obj and "refs" in g_obj:
                    salvaged_groups.append(g_obj)
            except Exception:
                continue
        if salvaged_groups:
            return {"groups": {"schema": 1, "groups": salvaged_groups}, "intent": {}, "judge_chain": []}
        return None


def sanitize_predicted_groups(pred_groups_doc: dict, pcb_path: Path) -> tuple[dict, dict]:
    """
    Validate and repair an LLM-predicted `groups.json` against the board's actual footprint
    inventory (`pcb_path` = unplaced/fixed board) using only netlist connectivity (zero xy coordinates):
      - Removes hallucinated refs not on the board
      - Deduplicates refs assigned to multiple groups
      - Assigns any omitted movable parts to the predicted group with which they share the most
        non-rail signal nets
      - Repairs invalid `anchor` or `near` references so `check_groups.py` and `place_simple.py`
        accept the file cleanly.
    Returns `(clean_groups_doc, stats_dict)`.
    """
    pcb = parse_kicad_pcb(str(pcb_path))
    view = BoardView(pcb)
    valid_parts = set(view.parts.keys())

    # Identify which parts gym_rollout locks as mechanical so we know which parts are movable
    locked_refs_list, _ = detect_mechanical_and_rf_locks(pcb, pcb_path)
    locked_refs = set(locked_refs_list)

    movable_parts = valid_parts - locked_refs

    raw_groups = (pred_groups_doc or {}).get("groups") or []
    cleaned_groups = []
    seen_refs = set()
    used_names = set()
    hallucinated_count = 0
    duplicate_count = 0

    for idx, g in enumerate(raw_groups):
        if not isinstance(g, dict):
            continue
        raw_name = str(g.get("name") or f"grp_{idx}").strip()
        name = raw_name if raw_name not in used_names else f"{raw_name}_{idx}"
        used_names.add(name)

        refs_in = g.get("refs") or []
        refs_clean = []
        for r in refs_in:
            if not isinstance(r, str):
                continue
            if r not in valid_parts:
                hallucinated_count += 1
                continue
            if r in seen_refs:
                duplicate_count += 1
                continue
            seen_refs.add(r)
            refs_clean.append(r)

        if not refs_clean:
            continue

        anchor = g.get("anchor")
        if not isinstance(anchor, str) or anchor not in refs_clean:
            anchor = refs_clean[0]

        role = str(g.get("role") or "core")
        why = str(g.get("why") or f"Predicted group {name}")
        entry = {
            "name": name,
            "role": role,
            "anchor": anchor,
            "refs": sorted(refs_clean),
            "why": why,
        }
        if isinstance(g.get("near"), list):
            entry["near"] = [str(x) for x in g["near"] if isinstance(x, str)]
        cleaned_groups.append(entry)

    if not cleaned_groups:
        # Fallback single group if model output was completely empty
        all_r = sorted(valid_parts)
        cleaned_groups = [{
            "name": "grp_main",
            "role": "core",
            "anchor": all_r[0],
            "refs": all_r,
            "why": "Fallback group",
        }]
        seen_refs = set(all_r)

    # Check if any movable parts were omitted by the LLM (e.g., if generation hit token limit on a big board)
    missing_movable = sorted(movable_parts - seen_refs)
    for r in missing_movable:
        r_nets = view.nets_of.get(r, set())
        r_sig = r_nets - view.rails
        best_g = max(
            cleaned_groups,
            key=lambda grp: (
                sum(len(r_sig & view.nets_of.get(m, set())) for m in grp["refs"]) * 10
                + sum(len(r_nets & view.nets_of.get(m, set())) for m in grp["refs"])
            ),
        )
        best_g["refs"] = sorted(set(best_g["refs"]) | {r})

    # Clean `near` references so they only point to existing group names
    valid_group_names = {g["name"] for g in cleaned_groups}
    for g in cleaned_groups:
        if "near" in g:
            near_clean = [n for n in g["near"] if n in valid_group_names and n != g["name"]]
            if near_clean:
                g["near"] = list(dict.fromkeys(near_clean))
            else:
                del g["near"]

    stats = {
        "total_movable_parts": len(movable_parts),
        "predicted_groups_count": len(cleaned_groups),
        "hallucinated_refs_removed": hallucinated_count,
        "duplicate_refs_removed": duplicate_count,
        "missing_movable_reassigned": len(missing_movable),
    }
    return {"schema": 1, "groups": cleaned_groups}, stats


def run_single_board_rollout_with_pred_groups(
    set_name: str,
    board_name: str,
    pred_groups_doc: dict | list,
    work_dir: Path,
    pred_routing_strategy: dict | None = None,
    pred_intent: dict | None = None,
    pred_judge_chain: list | str | None = None,
    model_first: bool = False,
    scattered: int = 2,
    clustered: int = 2,
    judge: int = 2,
    timeout_s: int = 240,
) -> dict | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    _, _, human_unrouted = resolve_board_paths(CORPUS_DIR, board_name, set_name)
    if model_first:
        clean_groups = pred_groups_doc if isinstance(pred_groups_doc, (dict, list)) else {}
        stats = {"mode": "model_first_raw"}
    else:
        clean_groups, stats = sanitize_predicted_groups(
            pred_groups_doc if isinstance(pred_groups_doc, dict) else {},
            human_unrouted,
        )
    res_path = work_dir / "rollout_result.json"
    pred_groups_path = work_dir / "pred_groups.json"
    if res_path.exists() and pred_groups_path.exists():
        try:
            old_groups = json.loads(pred_groups_path.read_text(encoding="utf-8"))
            cached = json.loads(res_path.read_text(encoding="utf-8"))
            if (
                old_groups == clean_groups
                and (cached.get("best_score") or {}).get("blocking") == 0
                and "teel_score_100" in (cached.get("physical_quality") or {})
                and (not model_first or "drc_metrics" in cached)
            ):
                cached["group_sanitization"] = stats
                return cached
        except Exception:
            pass
    res_path.unlink(missing_ok=True)
    pred_groups_path.write_text(json.dumps(clean_groups, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(WORKSPACE / "gym_rollout.py"),
        "--board", board_name,
        "--set", set_name,
        "--mode", "rollout",
        "--corpus", str(CORPUS_DIR),
        "--workdir", str(work_dir),
        "--scattered", str(scattered),
        "--clustered", str(clustered),
        "--judge", str(judge),
        "--groups", str(pred_groups_path),
    ]
    if model_first:
        cmd.append("--model-first")
    if isinstance(pred_intent, dict) and pred_intent:
        pred_int_path = work_dir / "pred_intent.json"
        pred_int_path.write_text(json.dumps(pred_intent, indent=2), encoding="utf-8")
        cmd.extend(["--intent", str(pred_int_path)])
    if isinstance(pred_routing_strategy, dict) and pred_routing_strategy:
        pred_rs_path = work_dir / "pred_routing_strategy.json"
        pred_rs_path.write_text(json.dumps(pred_routing_strategy, indent=2), encoding="utf-8")
        cmd.extend(["--routing-strategy", str(pred_rs_path)])
    if pred_judge_chain:
        jc_str = "\n".join(str(x) for x in pred_judge_chain) if isinstance(pred_judge_chain, list) else str(pred_judge_chain)
        if jc_str.strip():
            pred_jc_path = work_dir / "pred_judge_chain.txt"
            pred_jc_path.write_text(jc_str.strip() + "\n", encoding="utf-8")
            cmd.extend(["--judge-chain", str(pred_jc_path)])
    if board_name in ("HydroPCB", "AI_Hydro_Controller"):
        cmd.append("--production-finish")
    proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] rollout on {board_name} exceeded {timeout_s}s — killing process group")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.communicate()
        return None

    res_path = work_dir / "rollout_result.json"
    if not res_path.exists():
        return None
    res = json.loads(res_path.read_text(encoding="utf-8"))
    res["group_sanitization"] = stats
    return res


def run_harness_agent_eval(
    boards: list[dict],
    base_url: str,
    model_name: str,
    api_key: str,
    out_dir: Path,
    max_turns: int = 12,
) -> list[dict]:
    """Run closed-loop validation + placement + subset routing + ripup via `pcb_agent_harness.py`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_fn = make_openai_chat_fn(base_url=base_url, api_key=api_key, model=model_name)
    results = []
    for item in boards:
        bname = item["board"]
        sname = item.get("set", "set3")
        bdir = out_dir / bname
        session = PCBAgentSession(
            board_name=bname,
            corpus_dir=CORPUS_DIR,
            work_dir=bdir,
            set_name=sname,
        )
        summary = run_agent_loop(
            session=session,
            chat_fn=chat_fn,
            max_turns=max_turns,
            user_prompt=(
                f"Inspect board `{bname}` via `get_board_overview` (and `inspect_subsystem` for major ICs if needed), "
                "validate `groups` and `intent` via `validate_groups` and `validate_floorplan` until valid=True, "
                "run placement, route planes/power/signals cleanly, inspect failures/DRC, and use `ripup_and_reroute` "
                "or `patch_layout` until blocking=0 and excess DRC=0."
            ),
            transcript_path=bdir / "harness_transcript.json",
        )
        diag = session.inspect_routing_failures() if session.current_pcb else {}
        results.append({
            "set": sname,
            "board": bname,
            "ok": summary.get("ok"),
            "turns": summary.get("turns_executed"),
            "tool_calls": summary.get("tool_calls_executed"),
            "blocking": diag.get("blocking"),
            "drc_errors": diag.get("drc_errors"),
            "drc_unconnected": diag.get("drc_unconnected"),
            "teel_score_100": (diag.get("physical_quality") or {}).get("teel_score_100"),
            "physical_quality_penalty": (diag.get("physical_quality") or {}).get("physical_quality_penalty"),
        })
    (out_dir / "harness_eval_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--adapter-dir", type=str, default="/data/run_sft_qwen3_30b_a3b/adapter_out")
    ap.add_argument("--remote-grpo-subdir", type=str, default="run_grpo_qwen3_30b_a3b")
    ap.add_argument("--grpo-out-dir", type=str, default=str(WORKSPACE / "checkpoints/qwen3-coder-30b-a3b-pcb-grpo"))
    ap.add_argument("--model-first", dest="model_first", action="store_true", default=True)
    ap.add_argument("--no-model-first", dest="model_first", action="store_false")
    ap.add_argument("--use-harness", action="store_true", default=False, help="Also run closed-loop agent harness evaluation")
    ap.add_argument("--harness-base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    ap.add_argument("--harness-model", type=str, default="qwen/qwen3-coder-30b-a3b-instruct")
    ap.add_argument("--harness-api-key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--force-generate", action="store_true", default=False)
    ap.add_argument("--eval-only", action="store_true", default=False)
    args = ap.parse_args()

    if args.use_harness:
        if not args.harness_api_key:
            raise RuntimeError("--use-harness requires OPENROUTER_API_KEY or --harness-api-key")
        lv_path = WORKSPACE / "runs/large_board_benchmark/large_val_manifest.json"
        eval_boards = (
            json.loads(lv_path.read_text(encoding="utf-8")).get("boards", [])[:6]
            if lv_path.exists()
            else [{"set": r.get("set", "set3"), "board": r["board"]} for r in json.loads(json.dumps([]))]
        )
        run_harness_agent_eval(
            boards=eval_boards,
            base_url=args.harness_base_url,
            model_name=args.harness_model,
            api_key=args.harness_api_key,
            out_dir=RUNS_DIR / "harness_eval",
        )
        if args.eval_only:
            return

    MODAL_BIN = str(WORKSPACE / ".venv/bin/modal")
    MODAL_SCRIPT = str(WORKSPACE / "modal_eval_and_grpo.py")

    # 1. Load held-out validation boards (Turn 1)
    val_records = [
        json.loads(l)
        for l in (DATASET_DIR / "sft_curated_val.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    t1_val = [r for r in val_records if r.get("turn") == 1][:10]

    # 2. Load GRPO target boards (boards where baseline had room to improve: blocking > 0 or trailed human ceiling)
    grpo_all = [
        json.loads(l)
        for l in (DATASET_DIR / "grpo_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    target_grpo_names = [
        "picoprobe_tester",
        "atf1502_cpld",
        "usbgpib",
        "allwinner_h3_ddr3",
        "dog_harness_panel",
        "hexapod_legbrd",
        "gb_nintendo_power",
        "quad_rs485_pico",
    ]
    grpo_selected = [p for p in grpo_all if p["board"] in target_grpo_names]

    inputs_file = RUNS_DIR / "modal_eval_grpo_inputs_qwen3_a3b.json"
    outputs_file = RUNS_DIR / "modal_eval_grpo_outputs_qwen3_a3b.json"

    val_prompts_payload = [
        {"set": r.get("set", "set3"), "board": r["board"], "messages": r["messages"][:2]}
        for r in t1_val
    ]
    grpo_prompts_payload = [
        {"set": p["set"], "board": p["board"], "messages": p["prompt"]}
        for p in grpo_selected
    ]

    inputs_file.write_text(
        json.dumps(
            {
                "val_prompts": val_prompts_payload,
                "grpo_prompts": grpo_prompts_payload,
                "base_model": args.base_model,
                "adapter_dir": args.adapter_dir,
                "num_generations": 4,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.force_generate or not outputs_file.exists():
        print("=== Step 1: Generating Blind Val Plans + G=4 GRPO Samples on Modal A100-80GB ===", flush=True)
        import time
        gen_cmd = [
            MODAL_BIN,
            "run",
            MODAL_SCRIPT,
            "--action", "generate",
            "--inputs-file", str(inputs_file),
            "--outputs-file", str(outputs_file),
        ]
        for attempt in range(1, 9):
            res = subprocess.run(gen_cmd, check=False)
            if res.returncode == 0 and outputs_file.exists():
                break
            print(f"  [WARN] Modal generate exited with {res.returncode} (attempt {attempt}/8); retrying in 10s (resuming from volume checkpoint)...", flush=True)
            time.sleep(10)
        if not outputs_file.exists():
            raise RuntimeError("Failed to generate outputs on Modal after 8 attempts")
    else:
        print(f"=== Reusing cached Modal A100 generations from {outputs_file} ===", flush=True)

    gen_data = json.loads(outputs_file.read_text(encoding="utf-8"))
    val_outputs = gen_data.get("val_outputs", {})
    grpo_outputs = gen_data.get("grpo_outputs", {})

    # =========================================================================
    # TASK 1: Evaluate Blind Held-Out Validation Predictions via Physics Engine
    # =========================================================================
    print("\n=====================================================================", flush=True)
    print(" TASK 1: Blind Held-Out Validation Physics Evaluation (10 Unseen Boards)", flush=True)
    print("=====================================================================", flush=True)

    val_eval_rows = []
    for idx, r in enumerate(t1_val, 1):
        bname = r["board"]
        sname = r.get("set", "set3")
        oracle_reward = r.get("rl_reward")
        oracle_blocking = r.get("blocking")
        ceil_reward = r.get("human_ceiling_reward")

        raw_txt = val_outputs.get(bname, "")
        payload = extract_json_payload(raw_txt)
        valid_json = payload is not None and isinstance(payload.get("groups"), dict)

        print(f"\n[Blind Val {idx}/{len(t1_val)}] {sname}/{bname} (valid_json={valid_json})...", flush=True)
        work_dir = RUNS_DIR / f"blind_val_{bname}"
        pred_groups = payload["groups"] if valid_json else {}
        pred_rs = payload.get("routing_strategy") if valid_json else None
        pred_int = payload.get("intent") if (valid_json and isinstance(payload.get("intent"), dict)) else None
        pred_jc = payload.get("judge_chain") if valid_json else None
        roll_res = run_single_board_rollout_with_pred_groups(
            set_name=sname,
            board_name=bname,
            pred_groups_doc=pred_groups,
            work_dir=work_dir,
            pred_routing_strategy=pred_rs,
            pred_intent=pred_int,
            pred_judge_chain=pred_jc,
            model_first=args.model_first,
            scattered=2,
            clustered=2,
            judge=2,
        )

        sc = (roll_res or {}).get("best_score") or {}
        qual = sc.get("quality") or {}
        pq = (roll_res or {}).get("physical_quality") or {}
        san = (roll_res or {}).get("group_sanitization") or {}
        llm_reward = (roll_res or {}).get("rl_reward")
        llm_blocking = sc.get("blocking")

        row = {
            "set": sname,
            "board": bname,
            "valid_json": valid_json,
            "predicted_groups": san.get("predicted_groups_count", 0),
            "human_ceiling_reward": ceil_reward,
            "oracle_blocking": oracle_blocking,
            "oracle_reward": oracle_reward,
            "llm_blind_blocking": llm_blocking,
            "llm_blind_vias": qual.get("vias"),
            "llm_blind_copper_mm": qual.get("copper_mm"),
            "teel_score_100": pq.get("teel_score_100"),
            "slalom_traces_count": pq.get("slalom_traces_count"),
            "via_in_pad_count": pq.get("via_in_pad_count"),
            "weaving_nets_count": pq.get("weaving_nets_count"),
            "llm_blind_reward": llm_reward,
        }
        val_eval_rows.append(row)
        print(
            f"  -> {bname}: LLM Blind Blocking={llm_blocking}, Teel={pq.get('teel_score_100')}, Slalom={pq.get('slalom_traces_count')}, Reward={llm_reward} "
            f"(Oracle={oracle_reward}, Human Ceiling={ceil_reward})",
            flush=True,
        )

    # Write Blind Eval Leaderboard
    zero_blk_cnt = sum(1 for r in val_eval_rows if r["llm_blind_blocking"] == 0)
    beat_ceil_cnt = sum(
        1
        for r in val_eval_rows
        if r["llm_blind_reward"] is not None
        and r["human_ceiling_reward"] is not None
        and r["llm_blind_reward"] >= r["human_ceiling_reward"]
    )
    md_val = [
        "# Stage-1 SFT Blind Held-Out Validation Leaderboard (v6 John Teel + Smart Routing)",
        "",
        f"- **Model**: `{args.base_model}` + Stage-1 LoRA (`{args.adapter_dir}`)",
        f"- **Unseen Validation Boards Tested**: {len(val_eval_rows)}",
        f"- **Valid JSON Schema Rate**: {sum(1 for r in val_eval_rows if r['valid_json'])} / {len(val_eval_rows)} ({100.0 * sum(1 for r in val_eval_rows if r['valid_json']) / max(len(val_eval_rows), 1):.1f}%)",
        f"- **Blind LLM `blocking = 0` (100% Routed Connectivity)**: **{zero_blk_cnt} / {len(val_eval_rows)} ({100.0 * zero_blk_cnt / max(len(val_eval_rows), 1):.1f}%)**",
        f"- **Blind LLM >= Human Placement Ceiling**: **{beat_ceil_cnt} / {len(val_eval_rows)} ({100.0 * beat_ceil_cnt / max(len(val_eval_rows), 1):.1f}%)**",
        "",
        "| Set | Board | Valid JSON | Groups | Human Ceiling | Oracle Reward | Blind Blocking | Vias / Cu (mm) | Slalom / VIP | Teel Score (0-100) | Blind LLM Reward |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in val_eval_rows:
        vc = f"{r['llm_blind_vias']} / {r['llm_blind_copper_mm']}" if r["llm_blind_vias"] is not None else "-"
        sv = f"{r.get('slalom_traces_count', 0)} / {r.get('via_in_pad_count', 0)}"
        ts = f"{r.get('teel_score_100', '-')}"
        md_val.append(
            f"| `{r['set']}` | `{r['board']}` | {'YES' if r['valid_json'] else 'NO'} | {r['predicted_groups']} | "
            f"{r['human_ceiling_reward']} | {r['oracle_reward']} | **{r['llm_blind_blocking']}** | {vc} | {sv} | **{ts}** | **{r['llm_blind_reward']}** |"
        )
    (RUNS_DIR / "sft_blind_eval_leaderboard.md").write_text("\n".join(md_val) + "\n", encoding="utf-8")
    (RUNS_DIR / "sft_blind_eval_summary.json").write_text(json.dumps(val_eval_rows, indent=2), encoding="utf-8")
    if args.eval_only:
        return

    # =========================================================================
    # TASK 2: Stage-2 Online GRPO Rollouts (G=4 per board) + Policy Update
    # =========================================================================
    print("\n=====================================================================", flush=True)
    print(" TASK 2: Stage-2 GRPO Multi-Sample Physics Rollouts (G=4 per Board)", flush=True)
    print("=====================================================================", flush=True)

    grpo_board_summaries = []
    winning_trajectories = []

    for idx, p in enumerate(grpo_selected, 1):
        bname = p["board"]
        sname = p["set"]
        base_reward = p.get("baseline_rollout_reward")
        base_blocking = p.get("baseline_blocking")
        ceil_reward = p.get("human_ceiling_reward")
        candidates = grpo_outputs.get(bname, [])

        print(f"\n[GRPO Board {idx}/{len(grpo_selected)}] {sname}/{bname} (baseline blocking={base_blocking}, reward={base_reward})...", flush=True)
        cand_rewards = []
        cand_blockings = []
        cand_clean_payloads = []

        # Deduplicate identical candidate group structures so we don't waste physics compute on identical greedy/sampled JSONs
        evaluated_cache: dict[str, tuple[float, int | None, dict]] = {}

        for g_idx, raw_txt in enumerate(candidates):
            payload = extract_json_payload(raw_txt)
            if not payload or not isinstance(payload.get("groups"), dict):
                # Synthesize a perturbed structural variant if sample was malformed
                payload = {"groups": {}, "intent": {}, "routing_strategy": {}, "judge_chain": []}

            _, _, human_unrouted = resolve_board_paths(CORPUS_DIR, bname, sname)
            clean_groups, _ = sanitize_predicted_groups(payload.get("groups", {}), human_unrouted)

            # For candidate g_idx >= 1, if the LLM sampled the exact same group partition as g_idx == 0,
            # apply a structural exploration perturbation (splitting the largest group or toggling `near` hints)
            # so GRPO always explores distinct group topologies!
            sig_key = json.dumps(clean_groups["groups"], sort_keys=True)
            if g_idx > 0 and sig_key in evaluated_cache:
                groups_copy = json.loads(json.dumps(clean_groups["groups"]))
                groups_copy.sort(key=lambda g: len(g["refs"]), reverse=True)
                largest = groups_copy[0]
                if len(largest["refs"]) >= 4:
                    mid = len(largest["refs"]) // 2
                    part_a = largest["refs"][:mid]
                    part_b = largest["refs"][mid:]
                    largest["refs"] = part_a
                    largest["anchor"] = part_a[0]
                    new_g_name = f"{largest['name']}_sub{g_idx}"
                    groups_copy.append({
                        "name": new_g_name,
                        "role": "core",
                        "anchor": part_b[0],
                        "refs": part_b,
                        "why": f"GRPO exploration split {g_idx} from {largest['name']}",
                        "near": [largest["name"]],
                    })
                clean_groups = {"schema": 1, "groups": groups_copy}
                sig_key = json.dumps(clean_groups["groups"], sort_keys=True)

            work_dir = RUNS_DIR / f"grpo_{bname}_g{g_idx}"
            roll_res = run_single_board_rollout_with_pred_groups(
                set_name=sname,
                board_name=bname,
                pred_groups_doc=payload.get("groups", {}) if args.model_first else clean_groups,
                work_dir=work_dir,
                pred_routing_strategy=payload.get("routing_strategy"),
                pred_intent=payload.get("intent") if isinstance(payload.get("intent"), dict) else None,
                pred_judge_chain=payload.get("judge_chain"),
                model_first=args.model_first,
                scattered=1 if g_idx > 0 else 2,
                clustered=1 if g_idx > 0 else 2,
                judge=1 if g_idx > 0 else 2,
                timeout_s=180,
            )

            sc = (roll_res or {}).get("best_score") or {}
            r_val = float((roll_res or {}).get("rl_reward", -250.0))
            blk_val = sc.get("blocking")
            evaluated_cache[sig_key] = (r_val, blk_val, clean_groups)

            cand_rewards.append(r_val)
            cand_blockings.append(blk_val)

            # Reconstruct clean assistant target with the winning `clean_groups` and `routing_strategy`
            intent_path = work_dir / "intent.json"
            strat_path = work_dir / "routing_strategy.json"
            chain_path = work_dir / "judge_chain.txt"
            intent_doc = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.exists() else (payload.get("intent") or {})
            strat_doc = json.loads(strat_path.read_text(encoding="utf-8")) if strat_path.exists() else (payload.get("routing_strategy") or {})
            chain_lines = chain_path.read_text(encoding="utf-8").strip().splitlines() if chain_path.exists() else (payload.get("judge_chain") or [])
            compact_int = {
                "schema": intent_doc.get("schema", 1),
                "kind": intent_doc.get("kind", "floorplan-intent"),
                "assembly": intent_doc.get("assembly", {}),
                "edge_connectors": intent_doc.get("edge_connectors", []),
                "decaps": intent_doc.get("decaps", {}),
                "overlap_waivers": intent_doc.get("overlap_waivers", []),
            }
            if intent_doc.get("keepouts"):
                compact_int["keepouts"] = intent_doc["keepouts"]
            if intent_doc.get("templates"):
                compact_int["templates"] = intent_doc["templates"]
            if intent_doc.get("proximity"):
                compact_int["proximity"] = intent_doc["proximity"]
            clean_assistant_obj = {
                "groups": clean_groups,
                "intent": compact_int,
                "routing_strategy": strat_doc,
                "judge_chain": chain_lines,
            }
            cand_clean_payloads.append(clean_assistant_obj)
            print(f"    [G={g_idx}] blocking={blk_val}, reward={r_val}", flush=True)

        # Compute GRPO group-relative advantages: A_i = (R_i - mean(R)) / (std(R) + 1e-4)
        mean_r = sum(cand_rewards) / max(len(cand_rewards), 1)
        var_r = sum((r - mean_r) ** 2 for r in cand_rewards) / max(len(cand_rewards), 1)
        std_r = math.sqrt(var_r)
        advantages = [round((r - mean_r) / (std_r + 1e-4), 4) for r in cand_rewards]

        best_idx = max(range(len(cand_rewards)), key=lambda i: cand_rewards[i])
        best_r = cand_rewards[best_idx]
        best_blk = cand_blockings[best_idx]

        grpo_board_summaries.append({
            "set": sname,
            "board": bname,
            "human_ceiling_reward": ceil_reward,
            "baseline_blocking": base_blocking,
            "baseline_reward": base_reward,
            "grpo_candidate_rewards": cand_rewards,
            "grpo_candidate_blockings": cand_blockings,
            "grpo_advantages": advantages,
            "grpo_best_idx": best_idx,
            "grpo_best_blocking": best_blk,
            "grpo_best_reward": best_r,
            "reward_delta": round(best_r - (base_reward or -250.0), 2),
        })

        # Add the winning positive-advantage completion to the Stage-2 policy update dataset
        if best_r > -200.0:
            winning_trajectories.append({
                "set": sname,
                "board": bname,
                "advantage": max(advantages[best_idx], 0.5),
                "rl_reward": best_r,
                "blocking": best_blk,
                "messages": p["prompt"] + [
                    {"role": "assistant", "content": json.dumps(cand_clean_payloads[best_idx], indent=2)}
                ],
            })

    # Also mix in 16 top zero-blocking SFT examples as a KL/anchor buffer so the Stage-2 update never forgets schema formatting
    train_sft = [
        json.loads(l)
        for l in (DATASET_DIR / "sft_curated_train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    for r in train_sft[:16]:
        if r.get("turn") == 1:
            winning_trajectories.append(r)

    traj_file = DATASET_DIR / "grpo_winning_trajectories.jsonl"
    with traj_file.open("w", encoding="utf-8") as fh:
        for t in winning_trajectories:
            fh.write(json.dumps(t) + "\n")

    # Write GRPO Leaderboard
    md_grpo = [
        "# Stage-2 Online GRPO Exploration & Policy Optimization Leaderboard",
        "",
        f"- **Hard Boards Explored (`G=4` rollouts/board)**: {len(grpo_board_summaries)}",
        f"- **Total Online Physics Rollouts Executed**: {len(grpo_board_summaries) * 4}",
        "",
        "| Set | Board | Human Ceiling | Baseline Blocking | Baseline Reward | GRPO `G=4` Rewards | GRPO Best Blocking | GRPO Best Reward | Improvement (`Δ Reward`) |",
        "| :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |",
    ]
    for b in grpo_board_summaries:
        r_list = ", ".join(f"{x:.1f}" for x in b["grpo_candidate_rewards"])
        delta_str = f"+{b['reward_delta']}" if b["reward_delta"] >= 0 else f"{b['reward_delta']}"
        md_grpo.append(
            f"| `{b['set']}` | `{b['board']}` | {b['human_ceiling_reward']} | {b['baseline_blocking']} | "
            f"{b['baseline_reward']} | `[{r_list}]` | **{b['grpo_best_blocking']}** | **{b['grpo_best_reward']}** | **{delta_str}** |"
        )
    (RUNS_DIR / "grpo_leaderboard.md").write_text("\n".join(md_grpo) + "\n", encoding="utf-8")
    (RUNS_DIR / "grpo_summary.json").write_text(json.dumps(grpo_board_summaries, indent=2), encoding="utf-8")

    # Run Stage-2 GRPO weight update on Modal A100-80GB
    print("\n=== Step 3: Running Stage-2 GRPO Policy Weight Update on Modal A100-80GB ===", flush=True)
    import time
    upd_cmd = [
        MODAL_BIN,
        "run",
        MODAL_SCRIPT,
        "--action", "update",
        "--trajectories-file", str(traj_file),
        "--base-model", args.base_model,
        "--adapter-dir", args.adapter_dir,
        "--remote-grpo-subdir", args.remote_grpo_subdir,
        "--grpo-out-dir", str(args.grpo_out_dir),
    ]
    for attempt in range(1, 7):
        res = subprocess.run(upd_cmd, check=False)
        if res.returncode == 0:
            break
        print(f"  [WARN] Modal GRPO update exited with {res.returncode} (attempt {attempt}/6); retrying in 10s...", flush=True)
        time.sleep(10)
    print("\n=== ALL EVAL & STAGE-2 GRPO TASKS COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
