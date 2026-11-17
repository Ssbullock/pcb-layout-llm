#!/usr/bin/env python3
"""
build_sft_dataset.py
Step 3: Runs the Gym Rollout & Human Ceiling across Set 3 boards, packages the
verified Supervised Fine-Tuning (SFT) dataset (`dataset/sft_set3.jsonl`), and
writes the Set 3 Leaderboard (`runs/set3_leaderboard.md` + `runs/set3_summary.json`).

Each SFT training example in `dataset/sft_set3.jsonl` contains:
  - Turn 1 (System + User -> Assistant):
    Given `board_context.md` (parts, footprints, netlist summary, mechanical fixed parts),
    predict the complete `groups.json`, `intent.json` (decoupling & edge constraints),
    and `judge_chain.txt` fast-routing recipe.
  - Turn 2 (Optional Refinement Turn when `route_review.py` identifies `turned_parts`
    or `open_nets`):
    Given the `route_review.py` JSON diagnostics, emit the `place_pose.py` / routing
    refinement actions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from gym_rollout import (
    KRT_ROOT,
    extract_human_groups,
    parse_kicad_pcb,
    resolve_board_paths,
    run_human_ceiling,
    run_rollout,
)
from pcb_design_intelligence import (
    build_compact_board_context,
    build_smart_routing_chains,
    detect_mechanical_and_rf_locks,
    extract_board_keepouts,
    repair_and_enrich_groups,
    synthesize_routing_strategy,
    synthesize_templates_and_proximity,
)
import os
import signal
import subprocess

SYSTEM_PROMPT = """You are an expert KiCad PCB Placement & Routing Architect built on KiCadRoutingTools and John Teel (*Predictable Designs*) hardware best practices.
Given a board's unplaced component inventory (`FIXED(x,y)` = locked mechanical/connector/RF part; `movable` = free to place), net connectivity, and mechanical constraints (`board_context.md`):
1. Partition ALL `movable` components into cohesive functional groups (`groups`) with `role`, `anchor`, `refs`, `near` adjacency hints, `apart` isolation constraints, `templates` declarations, and `why`. Every major IC (`U*`, `IC*`, `PS*`) and repeated actuator channel (`J*` + `Q*` + `D*` + `R*`) must anchor its own distinct functional group.
   - Strict Invariants: every `movable` ref must appear in **exactly 1** group (no duplicates, no omissions, no hallucinated refs); `anchor` must be in `refs`; `near` and `apart` MUST list valid group `name`s (never component refs).
2. Specify the floorplan, decoupling, keepout, template, and proximity constraints (`intent`) with `"schema": 1`, `"kind": "floorplan-intent"`, plus `assembly`, `edge_connectors` (list ONLY `movable` edge connectors — omit `FIXED(x,y)` parts unless their `edge` matches their fixed coordinates), `decaps` (realistic `max_distance_mm >= 5.0`), `overlap_waivers`, and optional `keepouts`, `templates`, `proximity`.
3. Plan the explicit routing & channel reservation strategy (`routing_strategy` JSON object) with dict-typed `placer_channel_reservation` (`halo_coef`, `halo_weight`, `congestion_weight`, `pad_weight`, `track_pitch`), `ground_plane_nets`, `plane_layers`, `mounting_hole_keepouts`, `diff_pairs`, `bus_routing`, dict-typed `via_and_trace_rules`, `power_width_tiers`, and pairwise EMI `keep_away_rules` (`AGGRESSOR:VICTIM:1.5`).
4. Synthesize the multi-step routing chain (`judge_chain` array of command strings, ALWAYS keeping `{in} {out}` as the first two positional args after the script path) honoring `routing_strategy`: mounting-hole keepouts (`add_rule_area.py {in} {out} --mounting-holes`), multi-domain plane pours (`route_planes.py`), differential pairs (`route_diff.py`), and `route.py` with laminar bus corridors, keep-away rules, and tiered power widths.

Respond with a single valid JSON object (no comments, no math expressions, no Python set literals) matching this exact top-level schema:
{
  "groups": {
    "schema": 1,
    "groups": [
      {"name": "grp_...", "role": "core|power|interface|analog|sensor|ui", "anchor": "U1", "refs": ["U1", "C1"], "near": ["grp_..."], "apart": [], "templates": [{"chip": "U1", "kind": "none|switching stage|rf front end", "why": "..."}], "why": "..."}
    ]
  },
  "intent": {
    "schema": 1,
    "kind": "floorplan-intent",
    "assembly": {"sides": "F|B|both", "why": "..."},
    "edge_connectors": [{"ref": "J1", "edge": "west|east|north|south"}],
    "decaps": {"max_distance_mm": 6.0},
    "overlap_waivers": []
  },
  "routing_strategy": {
    "placer_channel_reservation": {"halo_coef": 0.45, "halo_weight": 20.0, "congestion_weight": 12.0, "pad_weight": 6.0, "track_pitch": 0.4},
    "ground_plane_nets": ["GND"],
    "plane_layers": ["B.Cu"],
    "mounting_hole_keepouts": true,
    "diff_pairs": [],
    "bus_routing": {"enabled": true, "ordering": "bus", "bus_min_nets": 3, "river_pack_inline": true},
    "via_and_trace_rules": {"clearance_mm": 0.15, "signal_track_width_mm": 0.18, "same_net_pad_clearance_mm": 0.15, "via_size_mm": 0.6, "via_drill_mm": 0.3, "via_cost": 110, "turn_cost": 1200, "direction_preference_cost": 250},
    "power_width_tiers": {},
    "keep_away_rules": []
  },
  "judge_chain": ["python3 -X utf8 py_router/route_planes.py {in} {out} ...", "python3 -X utf8 py_router/route.py {in} {out} ..."]
}"""


def enrich_rollout_artifacts_v4(board_name: str, set_name: str | None, corpus_dir: Path, rollout_dir: Path) -> None:
    """Ensure cached rollout artifacts (`board_context.md`, `groups.json`, `intent.json`, `routing_strategy.json`, `judge_chain.txt`) include v6 compact prompts, multi-IC + actuator split groups, templates, proximity, keepouts, routing_strategy, and laminar bus judge chains."""
    groups_path = rollout_dir / "groups.json"
    intent_path = rollout_dir / "intent.json"
    strat_path = rollout_dir / "routing_strategy.json"
    chain_path = rollout_dir / "judge_chain.txt"
    context_md_path = rollout_dir / "board_context.md"
    if not (groups_path.exists() and intent_path.exists()):
        return

    try:
        groups_doc = json.loads(groups_path.read_text(encoding="utf-8"))
        intent_doc = json.loads(intent_path.read_text(encoding="utf-8"))
        if intent_doc.get("_v7_enriched") and strat_path.exists():
            return

        _, human_routed, human_unrouted = resolve_board_paths(corpus_dir, board_name, set_name)
        fixed_pcb = rollout_dir / "fixed.kicad_pcb"
        if not fixed_pcb.exists():
            fixed_pcb = human_unrouted
        pcb_fixed = parse_kicad_pcb(str(fixed_pcb))
        pcb_human = parse_kicad_pcb(str(human_unrouted))
        locked_refs, _ = detect_mechanical_and_rf_locks(pcb_human, human_unrouted)

        # Write compact lossless bipartite board_context.md (movable staging parts never marked FIXED)
        context_md_path.write_text(
            build_compact_board_context(fixed_pcb, locked_refs=set(locked_refs)),
            encoding="utf-8",
        )

        if any(str(fp.layer).startswith("B") for fp in pcb_human.footprints.values()):
            intent_doc.setdefault("assembly", {})["sides"] = "both"

        keepouts = extract_board_keepouts(human_routed, human_unrouted, pcb_human, locked_refs)
        if keepouts:
            intent_doc["keepouts"] = keepouts
        intent_doc = synthesize_templates_and_proximity(pcb_fixed, fixed_pcb, intent_doc, set(locked_refs))
        intent_doc["_v7_enriched"] = True
        intent_path.write_text(json.dumps(intent_doc, indent=2), encoding="utf-8")

        raw_groups = extract_human_groups(human_unrouted, fixed_refs=set(locked_refs))
        groups_doc = repair_and_enrich_groups(raw_groups, pcb_fixed, intent_doc, set(locked_refs))
        groups_path.write_text(json.dumps(groups_doc, indent=2), encoding="utf-8")

        strat_doc = synthesize_routing_strategy(human_unrouted)
        strat_path.write_text(json.dumps(strat_doc, indent=2), encoding="utf-8")

        chain_txt_str, _ = build_smart_routing_chains(human_unrouted, routing_strategy=strat_doc)
        chain_path.write_text(chain_txt_str, encoding="utf-8")
    except Exception as exc:
        print(f"  [WARN] v6 enrichment skipped on {board_name}: {exc}")


def format_sft_record(board_name: str, rollout_dir: Path, rollout_res: dict, ceiling_res: dict | None, corpus_dir: Path | None = None, set_name: str | None = None) -> list[dict]:
    """Build 1-turn (and optional 2-turn refinement) SFT chat records from a completed rollout."""
    if corpus_dir is not None:
        enrich_rollout_artifacts_v4(board_name, set_name, corpus_dir, rollout_dir)

    context_md_path = rollout_dir / "board_context.md"
    groups_path = rollout_dir / "groups.json"
    intent_path = rollout_dir / "intent.json"
    strat_path = rollout_dir / "routing_strategy.json"
    chain_path = rollout_dir / "judge_chain.txt"

    if not (context_md_path.exists() and groups_path.exists() and intent_path.exists() and chain_path.exists()):
        return []

    context_md = context_md_path.read_text(encoding="utf-8").strip()
    groups_doc = json.loads(groups_path.read_text(encoding="utf-8"))
    intent_doc = json.loads(intent_path.read_text(encoding="utf-8"))
    strat_doc = json.loads(strat_path.read_text(encoding="utf-8")) if strat_path.exists() else {}
    chain_txt = chain_path.read_text(encoding="utf-8").strip()

    # Keep intent compact in the SFT target while preserving kind, keepouts, templates & proximity rules
    compact_intent = {
        "schema": intent_doc.get("schema", 1),
        "kind": intent_doc.get("kind", "floorplan-intent"),
        "assembly": intent_doc.get("assembly", {}),
        "edge_connectors": [
            {k: v for k, v in ec.items() if k in ("ref", "edge", "overhang_mm", "max_setback_mm", "side")}
            for ec in intent_doc.get("edge_connectors", [])
        ],
        "decaps": intent_doc.get("decaps", {}),
        "overlap_waivers": [
            {"pair": w.get("pair"), "why": w.get("why")}
            for w in intent_doc.get("overlap_waivers", [])
        ],
    }
    if intent_doc.get("keepouts"):
        compact_intent["keepouts"] = intent_doc["keepouts"]
    if intent_doc.get("templates"):
        compact_intent["templates"] = intent_doc["templates"]
    if intent_doc.get("proximity"):
        compact_intent["proximity"] = intent_doc["proximity"]

    user_prompt = (
        f"# Board Placement & Routing Task: `{board_name}`\n\n"
        f"{context_md}\n\n"
        "Produce the `groups.json`, `intent.json`, `routing_strategy`, and `judge_chain.txt` for this board."
    )

    assistant_payload = {
        "groups": groups_doc,
        "intent": compact_intent,
        "routing_strategy": strat_doc,
        "judge_chain": chain_txt.splitlines(),
    }

    records = [
        {
            "board": board_name,
            "turn": 1,
            "rl_reward": rollout_res.get("rl_reward"),
            "human_ceiling_reward": ceiling_res.get("rl_reward") if ceiling_res else None,
            "blocking": (rollout_res.get("best_score") or {}).get("blocking"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(assistant_payload, indent=2)},
            ],
        }
    ]

    # Check if Turn-2 refinement data exists (turned_parts or open_nets in s<best>_review.json)
    best_seed = rollout_res.get("best_seed")
    if best_seed is not None:
        rev_path = rollout_dir / f"s{best_seed}_review.json"
        if rev_path.exists():
            rev_doc = json.loads(rev_path.read_text(encoding="utf-8"))
            turned = rev_doc.get("turned_parts") or []
            open_sig = [o for o in (rev_doc.get("open_nets") or []) if not o.get("poured")]
            applied_pose = (rollout_res.get("turn2_route_review") or {}).get("applied_pose_cmds") or []
            if turned or open_sig or applied_pose:
                turn2_user = (
                    "Fast-route evaluation completed (`route_review.py`).\n"
                    f"- Open signal nets ({len(open_sig)}): {[o['net'] for o in open_sig[:10]]}\n"
                    f"- Twisted/turned components ({len(turned)}): "
                    f"{[{'ref': t['ref'], 'from_deg': t.get('from_deg'), 'to_deg': t.get('to_deg')} for t in turned[:8]]}\n"
                    f"- Congested areas: {len(rev_doc.get('congested_areas') or [])}\n\n"
                    "Emit the Turn-2 `place_pose.py` orientation adjustments and/or targeted rip-up retry commands."
                )
                raw_pose_list = []
                for t in turned:
                    sp = t.get("swap_pose")
                    if isinstance(sp, list):
                        raw_pose_list.append(" ".join(str(x) for x in sp))
                    elif isinstance(sp, str) and sp:
                        raw_pose_list.append(sp)
                for ap in applied_pose:
                    if isinstance(ap, list):
                        raw_pose_list.append(" ".join(str(x) for x in ap))
                    elif isinstance(ap, str) and ap:
                        raw_pose_list.append(ap)
                pose_cmds = list(dict.fromkeys(raw_pose_list))
                applied_retry = (rollout_res.get("turn2_route_review") or {}).get("applied_retry_cmds") or []
                retry_cmds = list(applied_retry)
                if open_sig and not retry_cmds:
                    open_names = " ".join(f"'{o['net']}'" for o in open_sig[:8])
                    retry_cmds.append(
                        f"python3 -X utf8 py_router/route.py {{in}} {{out}} --nets {open_names} --rip-existing-nets {open_names} --force-reroute --max-ripup 12"
                    )
                turn2_assistant = {
                    "diagnosis": {
                        "open_signal_nets": [o["net"] for o in open_sig],
                        "turned_refs": [t["ref"] for t in turned],
                    },
                    "pose_commands": pose_cmds,
                    "retry_route_commands": retry_cmds,
                }
                records.append(
                    {
                        "board": board_name,
                        "turn": 2,
                        "rl_reward": rollout_res.get("rl_reward"),
                        "blocking": (rollout_res.get("best_score") or {}).get("blocking"),
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": json.dumps(assistant_payload, indent=2)},
                            {"role": "user", "content": turn2_user},
                            {"role": "assistant", "content": json.dumps(turn2_assistant, indent=2)},
                        ],
                    }
                )

    return records


def main():
    parser = argparse.ArgumentParser(description="Build Multi-Set SFT Dataset & Benchmark Leaderboard")
    parser.add_argument("--corpus", type=Path, default=Path("/Users/simon/Fine Tune/corpus"))
    parser.add_argument("--runs-dir", type=Path, default=Path("/Users/simon/Fine Tune/runs"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("/Users/simon/Fine Tune/dataset"))
    parser.add_argument("--sets", nargs="*", default=None, help="Stress sets to include (e.g. set1 set2 set3 or 'all'). Default: all available sets in corpus")
    parser.add_argument("--min-parts", type=int, default=6, help="Skip trivial boards with fewer than min-parts")
    parser.add_argument("--max-parts", type=int, default=350, help="Skip boards larger than max-parts (Tier 1 + Tier 2 <= 350 parts)")
    parser.add_argument("--max-nets", type=int, default=260, help="Skip backplane/connector-heavy boards with more than max-nets (Tier 1 + Tier 2 <= 260 nets)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total boards to evaluate")
    parser.add_argument("--tag", type=str, default=None, help="Output file tag (default: 'all' or joined set names)")
    parser.add_argument("--boards", nargs="*", default=None, help="Optional subset of board names to run")
    parser.add_argument("--reuse-existing", action="store_true", default=True, help="Reuse existing rollout_result.json if present")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker threads for board rollouts")
    parser.add_argument("--no-curate", action="store_true", default=False, help="Skip automatic curate_sft_dataset.py overwrite")
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.sets and args.sets != ["all"]:
        target_sets = [
            s if (s.startswith("set") or s.startswith("scraped_") or (args.corpus / f"boards_unrouted_{s}").exists())
            else f"set{s}"
            for s in args.sets
        ]
        unrouted_dirs = [args.corpus / f"boards_unrouted_{s}" for s in target_sets if (args.corpus / f"boards_unrouted_{s}").exists()]
    else:
        unrouted_dirs = [
            d
            for d in sorted(list(args.corpus.glob("boards_unrouted_set*")) + list(args.corpus.glob("boards_unrouted_scraped_*")))
            if d.name != "boards_unrouted_sethydro"
        ]

    selected_boards = []
    skipped_boards = []
    seen_boards = set()
    for unrouted_dir in unrouted_dirs:
        set_name = unrouted_dir.name.replace("boards_unrouted_", "")
        for bf in sorted(unrouted_dir.glob("*.kicad_pcb")):
            bname = bf.stem
            if bname in seen_boards:
                continue
            seen_boards.add(bname)
            if args.boards and bname not in args.boards:
                continue
            try:
                pcb = parse_kicad_pcb(str(bf))
                n_parts = len(pcb.footprints)
                n_nets = len(pcb.nets)
                n_layers = len(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"])
            except Exception as exc:
                skipped_boards.append((set_name, bname, 0, 0, 0, f"parse error: {exc}"))
                continue
            if n_parts < args.min_parts or n_nets < 4:
                skipped_boards.append((set_name, bname, n_parts, n_nets, n_layers, f"<{args.min_parts} parts or <4 nets"))
                continue
            if n_parts > args.max_parts:
                skipped_boards.append((set_name, bname, n_parts, n_nets, n_layers, f">{args.max_parts} parts"))
                continue
            if n_nets > args.max_nets:
                skipped_boards.append((set_name, bname, n_parts, n_nets, n_layers, f">{args.max_nets} nets"))
                continue
            n_cutouts = len(getattr(pcb.board_info, "board_cutouts", []) or [])
            if n_cutouts > 6:
                skipped_boards.append((set_name, bname, n_parts, n_nets, n_layers, f">{n_cutouts} internal cutouts (switch plate)"))
                continue
            selected_boards.append((set_name, bname, n_parts, n_nets, n_layers))

    # Sort from smallest to largest board so quick boards finish first
    selected_boards.sort(key=lambda x: (x[2], x[3]))
    if args.limit:
        selected_boards = selected_boards[: args.limit]

    tag = args.tag or ("_".join(args.sets) if (args.sets and len(args.sets) <= 3 and args.sets != ["all"]) else "all")
    print(f"Building SFT Dataset & Leaderboard ({tag}) across {len(selected_boards)} boards "
          f"(skipped {len(skipped_boards)} boards)")

    summary_rows = []
    all_sft_records = []

    import concurrent.futures
    import threading
    lock = threading.Lock()

    def _process_one_board(idx_and_info):
        idx, (set_name, bname, n_parts, n_nets, n_layers) = idx_and_info
        print(f"\n[{idx}/{len(selected_boards)}] Processing {set_name}/{bname} ({n_parts} parts, {n_nets} nets, {n_layers}L)...", flush=True)
        ceil_dir = args.runs_dir / f"ceiling_{bname}"
        roll_dir = args.runs_dir / f"rollout_{bname}"

        ceil_json = ceil_dir / "ceiling_result.json"
        roll_json = roll_dir / "rollout_result.json"

        # 1. Human Ceiling
        ceil_res = None
        if args.reuse_existing and ceil_json.exists():
            try:
                loaded = json.loads(ceil_json.read_text(encoding="utf-8"))
                if (loaded.get("score") or {}).get("blocking") is not None:
                    ceil_res = loaded
                    print(f"  -> Reusing cached human_ceiling for {bname} (reward={ceil_res.get('rl_reward')})", flush=True)
            except Exception:
                pass
        if ceil_res is None:
            try:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "gym_rollout.py"),
                    "--board", bname,
                    "--set", set_name,
                    "--mode", "human_ceiling",
                    "--corpus", str(args.corpus),
                    "--workdir", str(ceil_dir),
                ]
                proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    proc.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    print(f"  [TIMEOUT] human_ceiling exceeded 240s on {bname} — killing process group", flush=True)
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                if ceil_json.exists():
                    ceil_res = json.loads(ceil_json.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  [WARN] human_ceiling failed on {bname}: {exc}", flush=True)

        # 2. Gym Rollout (from unplaced pile)
        roll_res = None
        if args.reuse_existing and roll_json.exists():
            try:
                loaded = json.loads(roll_json.read_text(encoding="utf-8"))
                if (loaded.get("best_score") or {}).get("blocking") is not None:
                    roll_res = loaded
                    print(f"  -> Reusing cached rollout for {bname} (reward={roll_res.get('rl_reward')})", flush=True)
            except Exception:
                pass
        if roll_res is None:
            try:
                scat = "1" if n_parts > 110 else "2"
                clus = "1" if n_parts > 110 else "2"
                anch = "1"
                judg = "2"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "gym_rollout.py"),
                    "--board", bname,
                    "--set", set_name,
                    "--mode", "rollout",
                    "--corpus", str(args.corpus),
                    "--workdir", str(roll_dir),
                    "--scattered", scat,
                    "--clustered", clus,
                    "--anchored", anch,
                    "--judge", judg,
                    "--jobs", "2",
                ]
                proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    proc.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    print(f"  [TIMEOUT] rollout exceeded 360s on {bname} — killing process group", flush=True)
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    return ("skip", (set_name, bname, n_parts, n_nets, n_layers, "rollout timeout >360s"), None, None)
                if roll_json.exists():
                    roll_res = json.loads(roll_json.read_text(encoding="utf-8"))
                else:
                    return ("skip", (set_name, bname, n_parts, n_nets, n_layers, "no rollout_result.json"), None, None)
            except Exception as exc:
                print(f"  [ERROR] rollout failed on {bname}: {exc}", flush=True)
                return ("skip", (set_name, bname, n_parts, n_nets, n_layers, f"rollout error: {exc}"), None, None)

        # 3. Extract SFT records
        try:
            recs = format_sft_record(bname, roll_dir, roll_res, ceil_res, corpus_dir=args.corpus, set_name=set_name)
        except Exception as exc:
            print(f"  [WARN] format_sft_record failed on {bname}: {exc}", flush=True)
            recs = []
        for r in recs:
            r["set"] = set_name

        c_sc = (ceil_res or {}).get("score") or {}
        c_q = c_sc.get("quality") or {}
        r_sc = (roll_res or {}).get("best_score") or {}
        r_q = r_sc.get("quality") or {}
        r_by = r_sc.get("blocking_by") or {}
        t_sec = (roll_res or {}).get("timing_sec") or {}
        rev = (roll_res or {}).get("turn2_route_review") or {}

        row = {
            "set": set_name,
            "board": bname,
            "parts": n_parts,
            "nets": n_nets,
            "layers": n_layers,
            "ceiling_blocking": c_sc.get("blocking"),
            "ceiling_vias": c_q.get("vias"),
            "ceiling_copper_mm": c_q.get("copper_mm"),
            "ceiling_reward": (ceil_res or {}).get("rl_reward"),
            "rollout_seed": (roll_res or {}).get("best_seed"),
            "rollout_blocking": r_sc.get("blocking"),
            "rollout_joins": r_sc.get("joins_needed"),
            "rollout_floorplan_err": r_by.get("floorplan"),
            "rollout_assembly_err": r_by.get("assembly"),
            "rollout_vias": r_q.get("vias"),
            "rollout_copper_mm": r_q.get("copper_mm"),
            "rollout_reward": (roll_res or {}).get("rl_reward"),
            "turned_parts_count": len(rev.get("turned_parts") or []),
            "place_sec": t_sec.get("place_sec"),
            "route_sec": t_sec.get("fast_route_sec"),
            "total_sec": t_sec.get("total_sec"),
            "sft_turns": len(recs),
        }
        print(f"  [DONE {idx}/{len(selected_boards)}] {set_name}/{bname}: blocking={row['rollout_blocking']}, reward={row['rollout_reward']} ({row['total_sec']}s)", flush=True)
        return ("ok", None, row, recs)

    sft_jsonl_path = args.dataset_dir / f"sft_{tag}.jsonl"
    summary_json_path = args.runs_dir / f"{tag}_summary.json"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_process_one_board, item) for item in enumerate(selected_boards, 1)]
        for fut in concurrent.futures.as_completed(futures):
            try:
                status, skip_info, row, recs = fut.result()
            except Exception as exc:
                print(f"  [ERROR] Worker future failed: {exc}", flush=True)
                continue
            with lock:
                if status == "skip" and skip_info is not None:
                    skipped_boards.append(skip_info)
                elif status == "ok" and row is not None:
                    summary_rows.append(row)
                    all_sft_records.extend(recs or [])
                    with sft_jsonl_path.open("w", encoding="utf-8") as fh:
                        for r in all_sft_records:
                            fh.write(json.dumps(r) + "\n")
                    summary_json_path.write_text(
                        json.dumps({"boards": summary_rows, "skipped": skipped_boards}, indent=2),
                        encoding="utf-8",
                    )

    summary_rows.sort(key=lambda r: (r["parts"], r["nets"]))

    # Write Markdown Leaderboard
    md_lines = [
        f"# PCB Placement & Routing Gym Leaderboard (`{tag}`)",
        "",
        f"- **Boards Evaluated**: {len(summary_rows)} / {len(selected_boards) + len(skipped_boards)}",
        f"- **SFT Examples Generated**: {len(all_sft_records)} (`dataset/sft_{tag}.jsonl`)",
        "",
        "| Set | Board | Parts | Nets | Layers | Ceiling Blocking | Ceiling Vias / Cu (mm) | Ceiling Reward | Rollout Blocking | Rollout Vias / Cu (mm) | Rollout Reward | Turn-2 Swaps | Time (s) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_rows:
        c_vc = f"{r['ceiling_vias']} / {r['ceiling_copper_mm']}" if r["ceiling_vias"] is not None else "-"
        r_vc = f"{r['rollout_vias']} / {r['rollout_copper_mm']}" if r["rollout_vias"] is not None else "-"
        md_lines.append(
            f"| `{r['set']}` | `{r['board']}` | {r['parts']} | {r['nets']} | {r['layers']}L | "
            f"{r['ceiling_blocking']} | {c_vc} | {r['ceiling_reward']} | "
            f"**{r['rollout_blocking']}** | {r_vc} | **{r['rollout_reward']}** | "
            f"{r['turned_parts_count']} | {r['total_sec']}s |"
        )
    if skipped_boards:
        md_lines.extend([
            "",
            "### Skipped Boards",
            *[f"- `{b[0]}/{b[1]}` ({b[2]} parts, {b[3]} nets, {b[4]}L): {b[5]}" for b in skipped_boards],
        ])

    leaderboard_path = args.runs_dir / f"{tag}_leaderboard.md"
    leaderboard_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n=======================================================")
    print(f" Completed SFT Dataset & Leaderboard ({tag})!")
    print(f" - SFT JSONL:   {args.dataset_dir / f'sft_{tag}.jsonl'} ({len(all_sft_records)} examples)")
    print(f" - Summary:     {args.runs_dir / f'{tag}_summary.json'}")
    print(f" - Leaderboard: {leaderboard_path}")
    print(f"=======================================================")

    # Automatically refresh curated train/val/GRPO splits
    curate_script = Path(__file__).resolve().parent / "curate_sft_dataset.py"
    if (not args.no_curate) and curate_script.exists() and (args.dataset_dir / f"sft_{tag}.jsonl").exists():
        subprocess.run(
            [
                sys.executable,
                str(curate_script),
                "--input",
                str(args.dataset_dir / f"sft_{tag}.jsonl"),
                "--out-dir",
                str(args.dataset_dir),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
