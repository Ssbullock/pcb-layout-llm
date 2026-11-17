#!/usr/bin/env python3
"""
run_model_first_benchmark.py
----------------------------
Runs all 23 benchmark boards (21 blind_val + 2 hydro) through `gym_rollout.py --model-first`
using the raw `finetuned-32b` model outputs (`groups`, `intent`, `routing_strategy`,
`judge_chain`) without Python `repair_and_enrich_groups`, strategy synthesis/clamping,
smart judge chain generation, or Turn-2 scripted rotation repair.

Compares Model-First vs. Script-Assisted Baseline vs. Human Ground Truth across all 23 boards.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
RUNS_DIR = WORKSPACE / "runs"
BENCH_DIR = RUNS_DIR / "frontier_benchmark"
MF_DIR = BENCH_DIR / "model_first_rollouts"
SA_ROLLOUTS_DIR = BENCH_DIR / "rollouts" / "finetuned-32b"

sys.path.insert(0, str(WORKSPACE))
from benchmark_frontier_vs_finetuned import run_kicad_drc  # noqa: E402
from gym_rollout import parse_kicad_pcb  # noqa: E402
from pcb_design_intelligence import evaluate_physical_board_quality  # noqa: E402
from run_eval_and_grpo_pipeline import extract_json_payload  # noqa: E402


def load_raw_finetuned_outputs() -> dict[str, str]:
    """Load raw text responses for all 23 boards from saved finetuned-32b outputs."""
    raw_map: dict[str, str] = {}

    # 1. 11 remaining boards from responses/finetuned-32b/*.json
    resp_dir = BENCH_DIR / "responses" / "finetuned-32b"
    if resp_dir.exists():
        for p in sorted(resp_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            raw_text = data.get("content") or data.get("raw_text") or ""
            if raw_text:
                raw_map[p.stem] = raw_text

    # 2. 10 blind_val boards from runs/modal_eval_grpo_outputs_32b.json
    modal_32b_path = RUNS_DIR / "modal_eval_grpo_outputs_32b.json"
    if modal_32b_path.exists():
        d = json.loads(modal_32b_path.read_text(encoding="utf-8"))
        for bname, raw_text in (d.get("val_outputs") or {}).items():
            if raw_text and bname not in raw_map:
                raw_map[bname] = raw_text

    # 3. 2 hydro boards from runs/hydro_modal_grpo_outputs.json
    hydro_path = RUNS_DIR / "hydro_modal_grpo_outputs.json"
    if hydro_path.exists():
        d = json.loads(hydro_path.read_text(encoding="utf-8"))
        for bname, raw_text in (d.get("val_outputs") or {}).items():
            if raw_text and bname not in raw_map:
                raw_map[bname] = raw_text

    return raw_map


def prepare_board_inputs(board: str, raw_text: str) -> dict[str, Path | None]:
    """Extract raw JSON components and write per-board input files."""
    in_dir = MF_DIR / board / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    payload = extract_json_payload(raw_text)
    if not payload or not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {board}")

    paths: dict[str, Path | None] = {
        "groups": None,
        "intent": None,
        "routing_strategy": None,
        "judge_chain": None,
    }

    if "groups" in payload and payload["groups"] is not None:
        gpath = in_dir / "groups.json"
        gpath.write_text(json.dumps(payload["groups"], indent=2), encoding="utf-8")
        paths["groups"] = gpath

    if "intent" in payload and payload["intent"] is not None:
        ipath = in_dir / "intent.json"
        ipath.write_text(json.dumps(payload["intent"], indent=2), encoding="utf-8")
        paths["intent"] = ipath

    if "routing_strategy" in payload and payload["routing_strategy"] is not None:
        rpath = in_dir / "routing_strategy.json"
        rpath.write_text(json.dumps(payload["routing_strategy"], indent=2), encoding="utf-8")
        paths["routing_strategy"] = rpath

    if "judge_chain" in payload and payload["judge_chain"]:
        jc = payload["judge_chain"]
        jc_text = "\n".join(str(x) for x in jc) if isinstance(jc, list) else str(jc)
        jpath = in_dir / "judge_chain.txt"
        jpath.write_text(jc_text.strip() + "\n", encoding="utf-8")
        paths["judge_chain"] = jpath

    return paths


def run_one_board(prompt_entry: dict, raw_text: str) -> dict:
    """Run gym_rollout.py --model-first for one board."""
    board = prompt_entry["board"]
    set_name = prompt_entry["set"]
    work_dir = MF_DIR / board
    result_json = work_dir / "rollout_result.json"

    if result_json.exists():
        return {"board": board, "status": "cached", "error": None}

    try:
        paths = prepare_board_inputs(board, raw_text)
    except Exception as exc:
        return {"board": board, "status": "input_error", "error": str(exc)}

    cmd = [
        "python3",
        "-X",
        "utf8",
        str(WORKSPACE / "gym_rollout.py"),
        "--board",
        board,
        "--set",
        set_name,
        "--corpus",
        str(CORPUS_DIR),
        "--workdir",
        str(work_dir),
        "--mode",
        "rollout",
        "--model-first",
        "--scattered",
        "2",
        "--clustered",
        "2",
        "--anchored",
        "2",
        "--judge",
        "2",
        "--jobs",
        "2",
    ]
    if paths["groups"]:
        cmd.extend(["--groups", str(paths["groups"])])
    if paths["intent"]:
        cmd.extend(["--intent", str(paths["intent"])])
    if paths["routing_strategy"]:
        cmd.extend(["--routing-strategy", str(paths["routing_strategy"])])
    if paths["judge_chain"]:
        cmd.extend(["--judge-chain", str(paths["judge_chain"])])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            check=False,
        )
        (work_dir / "rollout.log").write_text(proc.stdout or "", encoding="utf-8")
        elapsed = round(time.time() - t0, 2)
        if proc.returncode != 0 or not result_json.exists():
            tail = (proc.stdout or "")[-800:]
            return {
                "board": board,
                "status": "rollout_failed",
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
                "error": tail,
            }
        return {"board": board, "status": "ok", "elapsed_sec": elapsed, "error": None}
    except subprocess.TimeoutExpired:
        return {"board": board, "status": "timeout", "elapsed_sec": 600.0, "error": "600s timeout"}
    except Exception as exc:
        return {"board": board, "status": "exception", "error": str(exc)}


def evaluate_rollout_dir(board: str, work_dir: Path, run_status: str | None = None, error: str | None = None) -> dict:
    """Read rollout_result.json, run KiCad DRC, and compute Teel physical board quality."""
    res_path = work_dir / "rollout_result.json"
    if not res_path.exists():
        return {
            "board": board,
            "status": run_status or "missing",
            "error": error,
            "blocking": None,
            "drc_errors": None,
            "drc_unconnected": None,
            "teel_score_100": None,
            "copper_mm": None,
            "vias": None,
            "timing_sec": None,
        }

    rr = json.loads(res_path.read_text(encoding="utf-8"))
    score = rr.get("best_score") or rr.get("score") or {}
    best_seed = rr.get("best_seed")
    pcb_str = rr.get("winner_routed_pcb")
    if pcb_str:
        pcb_path = Path(pcb_str)
    elif best_seed is not None:
        pcb_path = work_dir / f"s{best_seed}_fast.kicad_pcb"
    else:
        pcb_path = None

    drc = {"drc_errors": None, "drc_unconnected": None}
    pq: dict = {}
    if pcb_path and pcb_path.exists():
        drc = run_kicad_drc(pcb_path, work_dir / "drc.json")
        try:
            pq = evaluate_physical_board_quality(pcb_path)
        except Exception as exc:
            pq = {"error": str(exc)}

    timing_obj = rr.get("timing_sec")
    total_sec = timing_obj.get("total_sec") if isinstance(timing_obj, dict) else timing_obj
    qual = score.get("quality") or {}

    return {
        "board": board,
        "status": run_status or "ok",
        "error": error,
        "blocking": score.get("blocking"),
        "blocking_by": score.get("blocking_by"),
        "joins_needed": score.get("joins_needed"),
        "rl_reward": rr.get("rl_reward"),
        "drc_errors": drc.get("drc_errors"),
        "drc_unconnected": drc.get("drc_unconnected"),
        "teel_score_100": pq.get("teel_score_100"),
        "copper_mm": round(qual["copper_mm"], 1) if qual.get("copper_mm") is not None else None,
        "vias": qual.get("vias"),
        "timing_sec": total_sec,
        "turn2_applied": bool(rr.get("turn2_route_review")),
    }


def main() -> None:
    prompts = json.loads((BENCH_DIR / "prompts.json").read_text(encoding="utf-8"))
    raw_map = load_raw_finetuned_outputs()
    print(f"Loaded {len(prompts)} benchmark prompts and {len(raw_map)} raw finetuned-32b responses.")

    MF_DIR.mkdir(parents=True, exist_ok=True)

    missing = [p["board"] for p in prompts if p["board"] not in raw_map]
    if missing:
        raise RuntimeError(f"Missing raw responses for boards: {missing}")

    run_statuses: dict[str, dict] = {}
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(run_one_board, p, raw_map[p["board"]]): p["board"] for p in prompts}
        for fut in as_completed(futs):
            res = fut.result()
            b = res["board"]
            run_statuses[b] = res
            print(f"[rollout] {b:24s} -> {res['status']} ({res.get('elapsed_sec', '-') }s)")

    print(f"\nAll 23 model-first rollouts finished in {time.time() - t_all:.1f}s. Evaluating DRC + Teel...")

    mf_results: dict[str, dict] = {}
    sa_results: dict[str, dict] = {}

    baseline = json.loads((BENCH_DIR / "baseline_human_and_finetuned.json").read_text(encoding="utf-8"))
    human_rows = baseline["human_routed"]
    precomp_ft = baseline["finetuned_32b_precomputed"]

    for p in prompts:
        b = p["board"]
        st = run_statuses.get(b, {})
        mf_results[b] = evaluate_rollout_dir(b, MF_DIR / b, st.get("status"), st.get("error"))

        if b in precomp_ft:
            pre = precomp_ft[b]
            sa_results[b] = {
                "board": b,
                "blocking": pre.get("blocking"),
                "drc_errors": pre.get("drc_errors"),
                "drc_unconnected": pre.get("drc_unconnected"),
                "teel_score_100": pre.get("teel_score_100"),
                "copper_mm": pre.get("copper_mm"),
                "vias": pre.get("vias"),
                "timing_sec": pre.get("timing_sec"),
            }
        else:
            sa_results[b] = evaluate_rollout_dir(b, SA_ROLLOUTS_DIR / b)

    summary = {
        "model_first": mf_results,
        "script_assisted_baseline": sa_results,
        "human_routed": human_rows,
    }
    out_json = BENCH_DIR / "model_first_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write comparison markdown
    def _agg(rows: list[dict], name: str) -> str:
        n = len(rows)
        has_blk = [r for r in rows if r.get("blocking") is not None]
        zb = sum(1 for r in has_blk if r.get("blocking") == 0)
        drc_eval = [r for r in rows if r.get("drc_errors") is not None]
        drc0 = sum(1 for r in drc_eval if r.get("drc_errors") == 0)
        avg_drc = sum(r["drc_errors"] for r in drc_eval) / len(drc_eval) if drc_eval else 0.0
        teels = [r["teel_score_100"] for r in rows if r.get("teel_score_100") is not None]
        avg_teel = sum(teels) / len(teels) if teels else 0.0
        coppers = [r["copper_mm"] for r in rows if r.get("copper_mm") is not None]
        avg_copper = sum(coppers) / len(coppers) if coppers else 0.0
        vias = [r["vias"] for r in rows if r.get("vias") is not None]
        avg_vias = sum(vias) / len(vias) if vias else 0.0
        times = [r["timing_sec"] for r in rows if r.get("timing_sec") is not None]
        avg_time = f"{sum(times)/len(times):.1f}s" if times else "-"
        blk_str = f"{zb}/{n} ({100*zb/n:.0f}%)" if has_blk else "-"
        drc_str = f"{drc0}/{n} ({100*drc0/n:.0f}%)" if drc_eval else "-"
        return (
            f"| {name} | {n} | {blk_str} | {drc_str} | {avg_drc:.1f} | "
            f"{avg_copper:.0f} | {avg_vias:.1f} | {avg_time} | {avg_teel:.1f} |"
        )

    lines = [
        "# 23-Board Model-First vs. Script-Assisted Benchmark (`finetuned-32b`)",
        "",
        "Model-First disables Python `repair_and_enrich_groups`, strategy synthesis/clamping, "
        "smart `judge_chain` synthesis, and Turn-2 scripted rotation repair.",
        "",
        "| Mode | Boards | 0-Blocking | 0-DRC | Avg DRC Errors | Avg Copper (mm) | Avg Vias | Avg Time | Avg Teel (0-100) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        _agg(list(human_rows.values()), "**Human Ground Truth**"),
        _agg(list(sa_results.values()), "**Fine-Tuned 32B (Script-Assisted)**"),
        _agg(list(mf_results.values()), "**Fine-Tuned 32B (Model-First)**"),
        "",
        "## Per-Board Breakdown",
        "",
        "| Board | Split | Human (DRC / Teel) | Script-Assisted (blk / DRC / Teel) | Model-First (blk / DRC / Teel / status) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for p in prompts:
        b = p["board"]
        h = human_rows.get(b, {})
        sa = sa_results.get(b, {})
        mf = mf_results.get(b, {})
        lines.append(
            f"| `{b}` | {p['split']} | "
            f"{h.get('drc_errors', '-')} / {h.get('teel_score_100', '-')} | "
            f"blk={sa.get('blocking', '-')}, drc={sa.get('drc_errors', '-')}, T={sa.get('teel_score_100', '-')} | "
            f"blk={mf.get('blocking', '-')}, drc={mf.get('drc_errors', '-')}, T={mf.get('teel_score_100', '-')} ({mf.get('status')}) |"
        )

    out_md = BENCH_DIR / "model_first_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_json} and {out_md}")


if __name__ == "__main__":
    main()
