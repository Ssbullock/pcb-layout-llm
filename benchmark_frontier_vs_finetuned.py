#!/usr/bin/env python3
"""
benchmark_frontier_vs_finetuned.py
----------------------------------
23-Board Apples-to-Apples Benchmark comparing:
  1. Human Ground-Truth routed PCBs
  2. Fine-Tuned 32B Model (Qwen2.5-Coder-32B-Instruct + v7_full SFT/GRPO)
  3. Zero-Shot Base Model (Qwen2.5-Coder-32B-Instruct)
  4. Zero-Shot Frontier Models (GPT-6.1 Sol Pro, Claude Opus 5.5, Gemini 4 Carbon)

Metrics:
  - Routing Rate (0-Blocking Rate & % of boards 100% routed)
  - KiCad Native DRC (`kicad-cli pcb drc`) Error-Free Rate & Avg Errors
  - Run Time (`timing_sec` placement+routing, `llm_sec` inference)
  - Total Copper Trace Length (mm) & Via Count vs. Human
  - Teel Score (0-100)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
RUNS_DIR = WORKSPACE / "runs"
DATASET_DIR = WORKSPACE / "dataset"
BENCH_DIR = RUNS_DIR / "frontier_benchmark"
KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"

ACTIVE_MODELS = [
    "finetuned-32b",
    "gpt-6.1-sol-pro",
    "claude-opus-5.5",
    "gemini-4-carbon",
    "qwen-2.5-coder-32b-base",
]

sys.path.insert(0, str(WORKSPACE))
from gym_rollout import parse_kicad_pcb, resolve_board_paths  # noqa: E402
from pcb_design_intelligence import evaluate_physical_board_quality  # noqa: E402
from run_eval_and_grpo_pipeline import (  # noqa: E402
    extract_json_payload,
    run_single_board_rollout_with_pred_groups,
)


def load_all_prompts() -> list[dict]:
    """Load all 21 Turn-1 held-out validation boards + 2 Hydroponics boards (23 total)."""
    prompts: list[dict] = []
    val_records = [
        json.loads(line)
        for line in (DATASET_DIR / "sft_curated_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for r in val_records:
        if r.get("turn") == 1:
            prompts.append(
                {
                    "set": r["set"],
                    "board": r["board"],
                    "split": "blind_val",
                    "messages": r["messages"][:2],
                }
            )
    hydro_in = json.loads((RUNS_DIR / "hydro_modal_grpo_inputs.json").read_text(encoding="utf-8"))
    for p in hydro_in.get("val_prompts", []):
        prompts.append(
            {
                "set": p["set"],
                "board": p["board"],
                "split": "hydro",
                "messages": p["messages"][:2],
            }
        )
    return prompts


def run_kicad_drc(pcb_path: Path, out_json: Path) -> dict:
    """Run kicad-cli pcb drc (cached) and return error/unconnected counts."""
    if not pcb_path.exists():
        return {"drc_errors": None, "drc_unconnected": None}
    if not out_json.exists():
        try:
            subprocess.run(
                [KICAD_CLI, "pcb", "drc", "--format", "json", "--output", str(out_json), str(pcb_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        except Exception:
            return {"drc_errors": None, "drc_unconnected": None}
    if not out_json.exists():
        return {"drc_errors": None, "drc_unconnected": None}
    try:
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        violations = [v for v in (rep.get("violations") or []) if v.get("severity") == "error"]
        unconnected = rep.get("unconnected_items") or []
        return {"drc_errors": len(violations), "drc_unconnected": len(unconnected)}
    except Exception:
        return {"drc_errors": None, "drc_unconnected": None}


def extract_human_copper_and_vias(pcb_path: Path) -> tuple[float | None, int | None]:
    """Compute total copper trace length (mm) and via count from a routed KiCad PCB."""
    if not pcb_path.exists():
        return None, None
    try:
        import math
        pcb = parse_kicad_pcb(str(pcb_path))
        copper = sum(
            math.hypot(s.end_x - s.start_x, s.end_y - s.start_y)
            for s in (pcb.segments or [])
        )
        vias = len(pcb.vias or [])
        return round(copper, 2), vias
    except Exception:
        return None, None


def _extract_total_sec(t) -> float | None:
    if isinstance(t, (int, float)):
        return round(float(t), 1)
    if isinstance(t, dict):
        v = t.get("total_sec")
        if isinstance(v, (int, float)):
            return round(float(v), 1)
    return None


def prepare_manifest_and_human_scores() -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_all_prompts()
    (BENCH_DIR / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    drc_dir = BENCH_DIR / "human_drc"
    drc_dir.mkdir(parents=True, exist_ok=True)

    human_rows = {}
    for p in prompts:
        bname = p["board"]
        sname = p["set"]
        _, human_routed, _ = resolve_board_paths(CORPUS_DIR, bname, sname)
        if human_routed.exists():
            pq = evaluate_physical_board_quality(human_routed)
            drc = run_kicad_drc(human_routed, drc_dir / f"{bname}_drc.json")
            copper_mm, vias = extract_human_copper_and_vias(human_routed)
            human_rows[bname] = {
                "board": bname,
                "set": sname,
                "split": p["split"],
                "teel_score_100": pq.get("teel_score_100"),
                "copper_mm": copper_mm,
                "vias": vias,
                "drc_errors": drc["drc_errors"],
                "drc_unconnected": drc["drc_unconnected"],
                "slalom_traces_count": pq.get("slalom_traces_count"),
                "via_in_pad_count": pq.get("via_in_pad_count"),
                "weaving_nets_count": pq.get("weaving_nets_count"),
            }
        else:
            human_rows[bname] = {"board": bname, "error": f"missing {human_routed}"}

    # Load existing 12 fine-tuned 32B results from evaluated_boards_for_simon
    eval_dir = WORKSPACE / "evaluated_boards_for_simon"
    ft_list = json.loads((eval_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    ft_rows = {}
    for r in ft_list:
        bname = r["board"]
        rr_path = eval_dir / bname / "rollout_result.json"
        timing_sec = None
        if rr_path.exists():
            try:
                timing_sec = json.loads(rr_path.read_text(encoding="utf-8")).get("timing_sec")
            except Exception:
                pass
        ft_rows[bname] = {
            "board": bname,
            "valid_json": True,
            "blocking": r.get("blocking"),
            "vias": r.get("vias"),
            "copper_mm": r.get("copper_mm"),
            "teel_score_100": r.get("teel"),
            "slalom_traces_count": r.get("slalom"),
            "via_in_pad_count": r.get("vip"),
            "weaving_nets_count": r.get("weaving"),
            "drc_errors": r.get("drc_errors"),
            "drc_unconnected": r.get("drc_unconnected"),
            "timing_sec": _extract_total_sec(timing_sec),
            "reward": r.get("reward"),
        }

    out = {"human_routed": human_rows, "finetuned_32b_precomputed": ft_rows}
    (BENCH_DIR / "baseline_human_and_finetuned.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Prepared {len(prompts)} prompts in {BENCH_DIR / 'prompts.json'}")
    print(f"Scored {len(human_rows)} Human Ground-Truth boards and loaded {len(ft_rows)} precomputed Fine-Tuned 32B boards.")


def evaluate_saved_responses() -> None:
    prompts = json.loads((BENCH_DIR / "prompts.json").read_text(encoding="utf-8"))
    baselines = json.loads((BENCH_DIR / "baseline_human_and_finetuned.json").read_text(encoding="utf-8"))
    resp_root = BENCH_DIR / "responses"
    resp_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict]] = {}

    for model_slug in ACTIVE_MODELS:
        all_results[model_slug] = {}
        mdir = resp_root / model_slug
        print(f"\n=== Evaluating model: {model_slug} ===")
        for p in prompts:
            bname = p["board"]
            sname = p["set"]

            # Use precomputed 12 boards for finetuned-32b if not overridden in responses/
            rfile = mdir / f"{bname}.json"
            if model_slug == "finetuned-32b" and not rfile.exists():
                pre = baselines.get("finetuned_32b_precomputed", {}).get(bname)
                if pre:
                    all_results[model_slug][bname] = pre
                    print(
                        f"  {bname:24s} | [cached] blocking={pre['blocking']} | drc={pre['drc_errors']} | "
                        f"Teel={pre['teel_score_100']} | time={pre['timing_sec']}s"
                    )
                    continue

            if not rfile.exists():
                continue

            raw_obj = json.loads(rfile.read_text(encoding="utf-8"))
            raw_text = raw_obj.get("content") or raw_obj.get("raw_text") or ""
            llm_sec = raw_obj.get("llm_sec")
            payload = extract_json_payload(raw_text)
            raw_groups = payload.get("groups") if isinstance(payload, dict) else None
            schema_exact = isinstance(raw_groups, dict) and isinstance(raw_groups.get("groups"), list)
            if isinstance(raw_groups, list):
                norm = []
                for g in raw_groups:
                    if isinstance(g, dict):
                        gc = dict(g)
                        if "name" not in gc and "id" in gc:
                            gc["name"] = gc["id"]
                        norm.append(gc)
                pred_groups = {"schema": 1, "groups": norm}
            elif isinstance(raw_groups, dict) and isinstance(raw_groups.get("groups"), list):
                pred_groups = raw_groups
            else:
                all_results[model_slug][bname] = {
                    "board": bname,
                    "valid_json": False,
                    "schema_exact": False,
                    "blocking": None,
                    "drc_errors": None,
                    "teel_score_100": None,
                    "timing_sec": None,
                    "llm_sec": llm_sec,
                    "reward": None,
                }
                print(f"  {bname:24s} | INVALID JSON")
                continue

            pred_rs = payload.get("routing_strategy") if isinstance(payload.get("routing_strategy"), dict) else None
            work_dir = BENCH_DIR / "rollouts" / model_slug / bname
            res = run_single_board_rollout_with_pred_groups(
                set_name=sname,
                board_name=bname,
                pred_groups_doc=pred_groups,
                work_dir=work_dir,
                pred_routing_strategy=pred_rs,
                scattered=2,
                clustered=2,
                judge=2,
                timeout_s=240,
            )
            if not res:
                all_results[model_slug][bname] = {
                    "board": bname,
                    "valid_json": True,
                    "schema_exact": schema_exact,
                    "blocking": None,
                    "drc_errors": None,
                    "teel_score_100": None,
                    "timing_sec": 240.0,
                    "llm_sec": llm_sec,
                    "reward": None,
                }
                print(f"  {bname:24s} | ROLLOUT FAILED/TIMEOUT")
                continue

            best_seed = res.get("best_seed")
            routed_pcb = work_dir / f"s{best_seed}_fast.kicad_pcb"
            if not routed_pcb.exists():
                routed_pcb = work_dir / f"s{best_seed}.kicad_pcb"
            drc = run_kicad_drc(routed_pcb, work_dir / "drc_report.json")

            bs = res.get("best_score") or {}
            qual = bs.get("quality") or {}
            pq = res.get("physical_quality") or {}
            vias = qual.get("vias") if qual.get("vias") is not None else bs.get("vias")
            copper = qual.get("copper_mm") if qual.get("copper_mm") is not None else bs.get("copper_mm")
            reward = res.get("rl_reward")
            timing_sec = res.get("timing_sec")

            row = {
                "board": bname,
                "valid_json": True,
                "schema_exact": schema_exact,
                "blocking": bs.get("blocking"),
                "vias": vias,
                "copper_mm": round(copper, 2) if copper is not None else None,
                "drc_errors": drc["drc_errors"],
                "drc_unconnected": drc["drc_unconnected"],
                "timing_sec": _extract_total_sec(timing_sec),
                "llm_sec": round(llm_sec, 1) if llm_sec is not None else None,
                "reward": round(reward, 2) if reward is not None else None,
                "teel_score_100": pq.get("teel_score_100"),
                "slalom_traces_count": pq.get("slalom_traces_count"),
                "via_in_pad_count": pq.get("via_in_pad_count"),
                "weaving_nets_count": pq.get("weaving_nets_count"),
            }
            all_results[model_slug][bname] = row
            print(
                f"  {bname:24s} | blk={row['blocking']} | drc={row['drc_errors']} | "
                f"Teel={row['teel_score_100']} | vias={row['vias']} | time={row['timing_sec']}s"
            )

    summary = {"baselines": baselines, "models": all_results}
    (RUNS_DIR / "frontier_vs_finetuned_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Build Markdown leaderboard
    lines = ["# 23-Board PCB Layout Benchmark: Fine-Tuned 32B vs. Frontier Zero-Shot", ""]
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append(
        "| Model | Boards | Valid JSON | 100% Routed (`blk=0`) | 0-DRC Rate | Avg DRC Errors | "
        "Avg Copper (mm) | Avg Vias | Avg Rollout Time (s) | Avg Teel (0-100) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    def _avg(vals: list[float | int | None], fmt: str = ".1f") -> str:
        clean = [float(v) for v in vals if v is not None]
        return f"{sum(clean)/len(clean):{fmt}}" if clean else "-"

    h_rows = list(baselines["human_routed"].values())
    h_drc0 = sum(1 for r in h_rows if r.get("drc_errors") == 0)
    lines.append(
        f"| **Human Ground-Truth PCB** | {len(h_rows)} | - | - | "
        f"{h_drc0}/{len(h_rows)} ({100*h_drc0/len(h_rows):.0f}%) | "
        f"{_avg([r.get('drc_errors') for r in h_rows])} | "
        f"{_avg([r.get('copper_mm') for r in h_rows], '.0f')} | "
        f"{_avg([r.get('vias') for r in h_rows])} | - | "
        f"**{_avg([r.get('teel_score_100') for r in h_rows])}** |"
    )

    for mslug in ACTIVE_MODELS:
        mdict = all_results.get(mslug, {})
        rows = list(mdict.values())
        if not rows:
            continue
        vj = sum(1 for r in rows if r.get("valid_json"))
        zb = sum(1 for r in rows if r.get("blocking") == 0)
        drc0 = sum(1 for r in rows if r.get("drc_errors") == 0)
        drc_eval = sum(1 for r in rows if r.get("drc_errors") is not None)
        drc_str = f"{drc0}/{len(rows)} ({100*drc0/len(rows):.0f}%)" if drc_eval else "-"
        label = "**Fine-Tuned Qwen2.5-Coder-32B (SFT+GRPO)**" if mslug == "finetuned-32b" else f"`{mslug}` (Zero-Shot)"
        lines.append(
            f"| {label} | {len(rows)} | {vj}/{len(rows)} ({100*vj/len(rows):.0f}%) | "
            f"**{zb}/{len(rows)} ({100*zb/len(rows):.0f}%)** | {drc_str} | "
            f"{_avg([r.get('drc_errors') for r in rows])} | "
            f"{_avg([r.get('copper_mm') for r in rows], '.0f')} | "
            f"{_avg([r.get('vias') for r in rows])} | "
            f"{_avg([r.get('timing_sec') for r in rows])}s | "
            f"**{_avg([r.get('teel_score_100') for r in rows])}** |"
        )

    lines.extend(["", "## Per-Board Breakdown (Blocking / DRC Errors / Teel / Time)", ""])
    header = "| Board | Human (DRC / Teel) |"
    sep = "| :--- | :---: |"
    for mslug in ACTIVE_MODELS:
        if all_results.get(mslug):
            header += f" `{mslug}` |"
            sep += " :---: |"
    lines.extend([header, sep])

    for p in prompts:
        bname = p["board"]
        hr = baselines["human_routed"].get(bname, {})
        row_str = f"| `{bname}` | {hr.get('drc_errors', '-')} / {hr.get('teel_score_100', '-')} |"
        for mslug in ACTIVE_MODELS:
            if not all_results.get(mslug):
                continue
            mr = all_results[mslug].get(bname, {})
            if not mr:
                row_str += " - |"
            elif not mr.get("valid_json"):
                row_str += " INVALID JSON |"
            elif mr.get("blocking") is None:
                row_str += " TIMEOUT |"
            else:
                t_s = f"{mr['timing_sec']}s" if mr.get("timing_sec") is not None else "-"
                row_str += f" blk={mr['blocking']}, drc={mr.get('drc_errors', '-')}, T={mr.get('teel_score_100', '-')}, {t_s} |"
        lines.append(row_str)

    md_path = RUNS_DIR / "frontier_vs_finetuned_leaderboard.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare_manifest_and_human_scores()
    if args.eval:
        evaluate_saved_responses()


if __name__ == "__main__":
    main()
