#!/usr/bin/env python3
"""
gym_rollout.py
Single-Board Placement + Fast-Route RL Rollout & Evaluation Harness for KiCadRoutingTools.

Supports 3 evaluation modes:
1. `--mode baseline`: Extracts mechanical fixed/edge parts + heuristic groups from the
   human reference board (or runs a provided `--groups` and `--intent` JSON), runs
   `place_simple.py` across seeds, fast-routes via `judge_seeds.py` (`--quick 2`),
   and runs `route_review.py` to diagnose remaining bottlenecks.
2. `--mode human_ceiling`: Takes the human-placed board (stripped of copper), runs the
   exact same fast-route chain and `quick_score.py` so we know the exact upper-bound
   ceiling score on that board.
3. `--mode prompt_only`: Strips the board to an unplaced pile using `make_unplaced.py`,
   runs `board_brief.py` and `board_context.py`, and writes out the exact LLM training
   prompt (`prompt.md`) and ground-truth SFT target (`target_sft.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
for _d in ("py_router", "py_placer", "py_tools"):
    _p = str(KRT_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kicad_parser import parse_kicad_pcb, non_aperture_pads  # noqa: E402
from list_nets import find_differential_pairs, find_power_nets, read_design_rules  # noqa: E402
from placement.floorplan import emit_intent  # noqa: E402
from placement.grouping import (  # noqa: E402
    BoardView,
    analyze,
    claim_rows,
    follower_roots,
    is_ground,
    validate_doc,
)
from placement.groups import _centroid, decap_population  # noqa: E402
from placement.part_class import classify_part  # noqa: E402
from placement.roles import infer_role  # noqa: E402
from placement.seeder import stamp_locked  # noqa: E402
from placement.writer import write_placed_output  # noqa: E402

from pcb_design_intelligence import (  # noqa: E402
    build_compact_board_context,
    build_smart_routing_chains,
    detect_mechanical_and_rf_locks,
    evaluate_physical_board_quality,
    extract_board_keepouts,
    hide_fab_value_labels,
    normalize_rf_courtyard_wings,
    repair_and_enrich_groups,
    synthesize_routing_strategy,
    synthesize_templates_and_proximity,
)


import atexit
import signal

_ACTIVE_SUBPROCS: set[subprocess.Popen] = set()


def _cleanup_active_subprocs(*_args) -> None:
    for p in list(_ACTIVE_SUBPROCS):
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    _ACTIVE_SUBPROCS.clear()


def _on_term_signal(signum, _frame) -> None:
    _cleanup_active_subprocs()
    raise SystemExit(128 + signum)


atexit.register(_cleanup_active_subprocs)
try:
    signal.signal(signal.SIGTERM, _on_term_signal)
    signal.signal(signal.SIGINT, _on_term_signal)
except ValueError:
    pass


def run_krt(
    rel_script: str,
    args: list[str],
    cwd: Path = KRT_ROOT,
    quiet: bool = False,
    timeout_s: float = 180.0,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-X", "utf8", str(KRT_ROOT / rel_script)] + args
    if not quiet:
        print(f"  $ python3 {rel_script} {' '.join(args)}", flush=True)
    env = dict(os.environ)
    env["KICAD_PACK_INLINE"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _ACTIVE_SUBPROCS.add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {rel_script} exceeded {timeout_s}s — killing process group", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=stdout or "",
            stderr=f"TimeoutExpired after {timeout_s}s\n{stderr or ''}",
        )
    finally:
        _ACTIVE_SUBPROCS.discard(proc)


def extract_human_groups(pcb_path: Path, fixed_refs: set[str] | None = None) -> dict:
    """Extract a clean, check_groups-validated `groups.json` from a human reference board."""
    pcb = parse_kicad_pcb(str(pcb_path))
    view = BoardView(pcb)
    pop = decap_population(pcb)
    claims, _ = claim_rows(pcb)
    f_roots = follower_roots(claims)
    fixed_refs = fixed_refs or set()

    anchors = []
    for r in sorted(view.parts):
        cls = classify_part(pcb.footprints[r], r).name
        role = infer_role(pcb.footprints[r], r, cls)["role"]
        if (
            r.startswith(("U", "IC"))
            or cls in ("edge_receptacle", "connector_affinity")
            or role in ("regulator", "integrated circuit", "usb receptacle", "pin header")
        ):
            if cls not in ("fiducial", "mount_hole", "testpoint"):
                anchors.append(r)
    if not anchors:
        anchors = [sorted(view.parts)[0]]

    groups_map = {a: [a] for a in anchors}
    assigned = set(anchors)

    # 1. Assign crystal/claim followers to their root anchor
    for sub, root in f_roots.items():
        if sub in view.parts and sub not in assigned and root in groups_map:
            groups_map[root].append(sub)
            assigned.add(sub)

    # 2. Assign every decoupling capacitor to its closest owner IC in anchors
    for cap, (_rails, owners) in sorted(pop.items()):
        if cap in assigned or cap not in view.parts:
            continue
        valid_owners = [o for o in owners if o in groups_map]
        if valid_owners:
            pt = _centroid(pcb.footprints[cap])
            best_o = min(
                valid_owners,
                key=lambda o: math.hypot(
                    pt[0] - _centroid(pcb.footprints[o])[0],
                    pt[1] - _centroid(pcb.footprints[o])[1],
                ),
            )
            groups_map[best_o].append(cap)
            assigned.add(cap)

    # 3. Assign remaining passives/parts by signal connectivity + spatial proximity
    for r in sorted(view.parts):
        if r in assigned:
            continue
        pt = _centroid(pcb.footprints[r])
        best_a = max(
            anchors,
            key=lambda a: (
                len((view.nets_of.get(r, set()) & view.nets_of.get(a, set())) - view.rails) * 20.0
                + len(view.nets_of.get(r, set()) & view.nets_of.get(a, set())) * 2.0
                - math.hypot(
                    pt[0] - _centroid(pcb.footprints[a])[0],
                    pt[1] - _centroid(pcb.footprints[a])[1],
                )
            ),
        )
        groups_map[best_a].append(r)
        assigned.add(r)

    def _clean_group_name(ref: str) -> str:
        clean = "".join(ch if ch.isalnum() else "_" for ch in ref.lower()).strip("_")
        return f"grp_{clean}"

    # Compute inter-group signal links to populate `near` hints
    def _make_doc(gmap: dict[str, list[str]]) -> dict:
        glist = []
        for a, refs in gmap.items():
            if not refs:
                continue
            cls = classify_part(pcb.footprints[a], a).name
            role_raw = infer_role(pcb.footprints[a], a, cls)["role"]
            role = (
                "power"
                if role_raw == "regulator"
                else ("interface" if cls in ("edge_receptacle", "connector_affinity") else "core")
            )
            glist.append(
                {
                    "name": _clean_group_name(a),
                    "role": role,
                    "anchor": a if a in refs else refs[0],
                    "refs": sorted(refs),
                    "why": f"Functional group anchored by {a} ({pcb.footprints[a].value})",
                }
            )
        return {"schema": 1, "groups": glist}

    # Iteratively resolve any satellite parts pulled strongly by another group
    for _ in range(4):
        doc = _make_doc(groups_map)
        rep = analyze(doc, pcb, followers=sorted(f_roots), extra_fixed=sorted(fixed_refs))
        if not rep["pulled"]:
            break
        name_to_a = {_clean_group_name(a): a for a in groups_map}
        moved_any = False
        for p in rep["pulled"]:
            ref = p["ref"]
            from_a = name_to_a.get(p["group"])
            to_a = name_to_a.get(p["pulled_by"])
            if from_a and to_a and ref in groups_map[from_a] and ref not in pop and ref != from_a:
                groups_map[from_a].remove(ref)
                groups_map[to_a].append(ref)
                moved_any = True
        if not moved_any:
            break

    doc = _make_doc(groups_map)
    rep = analyze(doc, pcb, followers=sorted(f_roots), extra_fixed=sorted(fixed_refs))
    # Populate top `near` links from inter-group signal links
    neighbors: dict[str, list[tuple[float, str]]] = {}
    for row in rep.get("links", []):
        if row["signal_links"] >= 1.0:
            neighbors.setdefault(row["a"], []).append((row["signal_links"], row["b"]))
            neighbors.setdefault(row["b"], []).append((row["signal_links"], row["a"]))
    for g in doc["groups"]:
        nlist = sorted(neighbors.get(g["name"], []), reverse=True)[:2]
        if nlist:
            g["near"] = [b for _w, b in nlist]

    validate_doc(doc)
    return doc


def build_judge_chain(pcb_path: Path) -> str:
    """Synthesize the smart fast-route `judge_chain.txt` recipe with multi-domain pours and net-class power widths."""
    chain_txt, _ = build_smart_routing_chains(pcb_path)
    return chain_txt


KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"


def _run_kicad_drc_cached(pcb_path: Path, out_json: Path) -> tuple[int | None, int | None]:
    """Run `kicad-cli pcb drc --refill-zones` (cached) and return `(violations_count, unconnected_count)`."""
    if not pcb_path.exists():
        return None, None
    if not out_json.exists():
        try:
            subprocess.run(
                [
                    KICAD_CLI,
                    "pcb",
                    "drc",
                    "--format",
                    "json",
                    "--refill-zones",
                    "--output",
                    str(out_json),
                    str(pcb_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        except Exception:
            return None, None
    if not out_json.exists():
        return None, None
    try:
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        return len(rep.get("violations", [])), len(rep.get("unconnected_items", []))
    except Exception:
        return None, None


def _diagnose_raw_groups_invariants(
    raw_groups_doc: object,
    pcb_fixed,
    locked_refs: set[str],
) -> tuple[dict, dict, float]:
    """
    Evaluate raw `groups` structure and strict grouping invariants WITHOUT mutating the model's choices
    unless wrapping is strictly required to parse schema diagnostics.
    Returns `(normalized_or_raw_doc, diagnostics_dict, schema_invariant_penalty)`.
    """
    view = BoardView(pcb_fixed)
    all_parts = set(view.parts.keys())
    movable_parts = all_parts - set(locked_refs)

    schema_pen = 0.0
    non_object_wrapper = False
    g_list: list = []

    if isinstance(raw_groups_doc, dict) and isinstance(raw_groups_doc.get("groups"), list):
        doc_out = dict(raw_groups_doc)
        doc_out.setdefault("schema", 1)
        g_list = doc_out["groups"]
    elif isinstance(raw_groups_doc, list):
        non_object_wrapper = True
        schema_pen += 25.0
        g_list = [g for g in raw_groups_doc if isinstance(g, dict)]
        doc_out = {"schema": 1, "groups": g_list}
    elif isinstance(raw_groups_doc, dict) and raw_groups_doc and all(isinstance(v, dict) for v in raw_groups_doc.values()):
        non_object_wrapper = True
        schema_pen += 25.0
        g_list = [dict(v, name=str(k)) for k, v in raw_groups_doc.items()]
        doc_out = {"schema": 1, "groups": g_list}
    else:
        non_object_wrapper = True
        schema_pen += 60.0
        doc_out = {"schema": 1, "groups": []}

    seen_refs: set[str] = set()
    dup_refs: set[str] = set()
    hallucinated_refs: set[str] = set()
    group_names: set[str] = set()
    bad_anchors: list[str] = []

    for g in g_list:
        if not isinstance(g, dict):
            continue
        gname = g.get("name")
        if isinstance(gname, str) and gname.strip():
            group_names.add(gname.strip())
        refs = g.get("refs") or []
        if not isinstance(refs, list):
            schema_pen += 5.0
            continue
        for r in refs:
            if not isinstance(r, str):
                continue
            if r not in all_parts:
                hallucinated_refs.add(r)
            elif r in seen_refs:
                dup_refs.add(r)
            else:
                seen_refs.add(r)
        anc = g.get("anchor")
        if anc and isinstance(anc, str) and anc not in set(refs):
            bad_anchors.append(anc)

    dangling_links: list[str] = []
    for g in g_list:
        if not isinstance(g, dict):
            continue
        for field in ("near", "apart"):
            vals = g.get(field)
            if isinstance(vals, list):
                for target in vals:
                    if not isinstance(target, str) or target not in group_names:
                        dangling_links.append(str(target))

    missing_movable = sorted(movable_parts - seen_refs)
    schema_pen += (
        8.0 * len(dup_refs)
        + 6.0 * len(dangling_links)
        + 4.0 * len(missing_movable)
        + 5.0 * len(hallucinated_refs)
        + 3.0 * len(bad_anchors)
    )
    if not g_list:
        schema_pen += 40.0

    diag = {
        "non_object_wrapper": non_object_wrapper,
        "duplicate_refs": sorted(dup_refs),
        "missing_movable_refs": missing_movable,
        "dangling_links": sorted(set(dangling_links)),
        "hallucinated_refs": sorted(hallucinated_refs),
        "bad_anchors": bad_anchors,
        "schema_invariant_penalty": round(schema_pen, 2),
    }
    return doc_out, diag, round(schema_pen, 2)


def compute_rl_reward(
    score_doc: dict,
    group_warnings_penalty: float = 0.0,
    physical_quality_penalty: float = 0.0,
    schema_invariant_penalty: float = 0.0,
    excess_drc_errors: int = 0,
    drc_unconnected: int = 0,
) -> float:
    """
    Compute a scalar GRPO reward from quick_score.py output + baseline-normalized KiCad DRC +
    completion-gated physical board quality metrics + schema invariant checks:
      1. Placement defects (floorplan + assembly errors) -> heavy penalty (-50 each)
      2. Open / blocking nets -> primary penalty (-20 per blocking net, -3 per join_needed)
      3. Completion bonus for reaching DONE (blocking == 0, fp == 0, asm == 0) -> +100
      4. Baseline-normalized excess DRC (`excess_drc_errors * 3.0` + `drc_unconnected * 10.0`),
         gated softly (0.5x) until `blocking == 0` so pad-library defaults (e.g. annular width on
         footprint pins) never penalize clean routing, and only new routing violations hurt.
      5. Completion-gated & tail-compressed `physical_quality_penalty`: linear 1:1 up to 80.0
         (with 0.15x weight if blocking > 0 to avoid Teel gaming by under-routing), plus monotonic
         log tail above 80.0 so large/monster boards retain optimization gradient instead of saturating.
      6. Structural schema/invariant penalties (`schema_invariant_penalty` + `group_warnings_penalty`).
    """
    by = score_doc.get("blocking_by") or {}
    fp_err = by.get("floorplan") or 0
    asm_err = by.get("assembly") or 0
    blocking = score_doc.get("blocking")
    if blocking is None:
        # Preserve gradient across schema/floorplan failures instead of a flat -500 cliff
        return round(max(-500.0, -320.0 - float(schema_invariant_penalty) - 25.0 * (fp_err + asm_err)), 2)
    joins = score_doc.get("joins_needed") or 0
    q = score_doc.get("quality") or {}
    vias = q.get("vias") or 0
    copper_mm = q.get("copper_mm") or 0.0

    is_complete = blocking == 0 and fp_err == 0 and asm_err == 0
    done_bonus = 100.0 if is_complete else 0.0

    phys_raw = max(0.0, float(physical_quality_penalty or 0.0))
    if phys_raw <= 80.0:
        eff_phys = phys_raw
    else:
        eff_phys = 80.0 + 15.0 * math.log1p(phys_raw - 80.0)
    phys_gate = 1.0 if is_complete else 0.15

    drc_gate = 1.0 if blocking == 0 else 0.5
    drc_pen = drc_gate * (3.0 * max(0, int(excess_drc_errors or 0)) + 10.0 * max(0, int(drc_unconnected or 0)))

    reward = (
        done_bonus
        - 50.0 * (fp_err + asm_err)
        - 20.0 * blocking
        - 3.0 * joins
        - 0.20 * vias
        - 0.01 * copper_mm
        - group_warnings_penalty
        - schema_invariant_penalty
        - phys_gate * eff_phys
        - drc_pen
    )
    return round(reward, 2)


def resolve_board_paths(corpus_dir: Path, board_name: str, set_name: str | None = None) -> tuple[str, Path, Path]:
    """Find (set_name, human_routed, human_unrouted) for board_name in corpus_dir."""
    corpus_dir = corpus_dir.resolve()
    if set_name:
        if (
            set_name.startswith("set")
            or set_name.startswith("scraped_")
            or (corpus_dir / f"boards_unrouted_{set_name}").exists()
        ):
            s = set_name
        else:
            s = f"set{set_name}"
        unrouted = corpus_dir / f"boards_unrouted_{s}" / f"{board_name}.kicad_pcb"
        routed = corpus_dir / f"boards_{s}" / f"{board_name}.kicad_pcb"
        if unrouted.exists():
            return s, routed, unrouted
    for unrouted_dir in sorted(
        list(corpus_dir.glob("boards_unrouted_set*")) + list(corpus_dir.glob("boards_unrouted_scraped_*"))
    ):
        cand = unrouted_dir / f"{board_name}.kicad_pcb"
        if cand.exists():
            s = unrouted_dir.name.replace("boards_unrouted_", "")
            routed = corpus_dir / f"boards_{s}" / f"{board_name}.kicad_pcb"
            return s, routed, cand
    s = set_name or "set3"
    return s, corpus_dir / f"boards_{s}" / f"{board_name}.kicad_pcb", corpus_dir / f"boards_unrouted_{s}" / f"{board_name}.kicad_pcb"


def run_rollout(
    board_name: str,
    corpus_dir: Path,
    work_dir: Path,
    scattered_seeds: int = 2,
    clustered_seeds: int = 2,
    anchored_seeds: int = 2,
    judge_count: int = 3,
    jobs: int = 4,
    custom_groups: Path | None = None,
    custom_intent: Path | None = None,
    custom_routing_strategy: Path | dict | None = None,
    custom_judge_chain: Path | None = None,
    model_first: bool = False,
    set_name: str | None = None,
    production_finish: bool = False,
) -> dict:
    t_start = time.time()
    corpus_dir = corpus_dir.resolve()
    work_dir = work_dir.resolve()
    if isinstance(custom_groups, Path):
        custom_groups = custom_groups.resolve()
    if isinstance(custom_intent, Path):
        custom_intent = custom_intent.resolve()
    if isinstance(custom_routing_strategy, Path):
        custom_routing_strategy = custom_routing_strategy.resolve()
    resolved_set, human_routed, human_unrouted = resolve_board_paths(corpus_dir, board_name, set_name)

    if not human_unrouted.exists():
        raise FileNotFoundError(f"Unrouted board not found: {human_unrouted}")

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f" GYM ROLLOUT: {resolved_set}/{board_name} -> {work_dir}")
    print(f"=======================================================")

    # Step 1: Strip placement to an unplaced pile (`make_unplaced.py`)
    pile_pcb = work_dir / "pile.kicad_pcb"
    run_krt(
        ".claude/skills/plan-pcb-placement/scripts/make_unplaced.py",
        [str(human_unrouted), str(pile_pcb)],
    )

    # Step 2: Generate LLM Input Prompt (`brief.json` + compact `board_context.md`)
    brief_json = work_dir / "brief.json"
    context_md = work_dir / "board_context.md"
    run_krt("py_tools/board_brief.py", [str(pile_pcb), "--json", str(brief_json)], quiet=True)

    # Step 3: Prepare `fixed.kicad_pcb` + `intent.json` + `groups.json`
    pcb_human = parse_kicad_pcb(str(human_unrouted))
    pcb_pile = parse_kicad_pcb(str(pile_pcb))

    locked_refs, seat_placements = detect_mechanical_and_rf_locks(pcb_human, human_unrouted)

    fixed_pcb = work_dir / "fixed.kicad_pcb"
    if seat_placements:
        write_placed_output(str(pile_pcb), str(fixed_pcb), seat_placements, pcb_data=pcb_pile)
        stamp_locked(str(fixed_pcb), locked_refs)
    else:
        run_krt("py_router/copy_board.py", [str(pile_pcb), str(fixed_pcb)], quiet=True)
    normalize_rf_courtyard_wings(fixed_pcb)
    context_md.write_text(
        build_compact_board_context(fixed_pcb, locked_refs=set(locked_refs)),
        encoding="utf-8",
    )

    pcb_fixed = parse_kicad_pcb(str(fixed_pcb))
    intent_json = work_dir / "intent.json"
    if custom_intent and custom_intent.exists():
        shutil.copyfile(custom_intent, intent_json)
        doc_intent = json.loads(intent_json.read_text(encoding="utf-8"))
        doc_intent.setdefault("kind", "floorplan-intent")
        doc_intent.setdefault("schema", 1)
        if isinstance(doc_intent.get("decaps"), dict) and "max_distance_mm" in doc_intent["decaps"]:
            doc_intent["decaps"]["max_distance_mm"] = max(float(doc_intent["decaps"]["max_distance_mm"]), 5.0)
        if any(str(fp.layer).startswith("B") for fp in pcb_human.footprints.values()):
            doc_intent.setdefault("assembly", {})["sides"] = "both"
        intent_json.write_text(json.dumps(doc_intent, indent=2), encoding="utf-8")
    else:
        doc_intent = emit_intent(
            pcb_fixed,
            str(fixed_pcb),
            declare_classes=False,
            derive_decaps="strict",
            decaps_from=str(human_unrouted),
        )
        locked_set = set(locked_refs)
        doc_intent["edge_connectors"] = [
            c
            for c in doc_intent.get("edge_connectors", [])
            if c.get("edge") and c["ref"] not in locked_set
        ]
        if isinstance(doc_intent.get("decaps"), dict) and "max_distance_mm" in doc_intent["decaps"]:
            doc_intent["decaps"]["max_distance_mm"] = max(float(doc_intent["decaps"]["max_distance_mm"]), 5.0)
        if any(str(fp.layer).startswith("B") for fp in pcb_human.footprints.values()):
            doc_intent.setdefault("assembly", {})["sides"] = "both"

        # Extract antenna / mechanical keepout zones from reference board
        keepouts = extract_board_keepouts(human_routed, human_unrouted, pcb_human, locked_refs)
        if keepouts:
            doc_intent["keepouts"] = keepouts
        intent_json.write_text(json.dumps(doc_intent, indent=2), encoding="utf-8")

    # Waive any locked/fixed courtyard overlaps from the original design BEFORE fitting templates
    # Create a quick copy of human_unrouted with locked_refs stamped so check_assembly does not choke on the 66-part pile at (0,0)
    locked_ref_check_pcb = work_dir / "locked_check.kicad_pcb"
    shutil.copyfile(human_unrouted, locked_ref_check_pcb)
    if locked_refs:
        stamp_locked(str(locked_ref_check_pcb), locked_refs)
    normalize_rf_courtyard_wings(locked_ref_check_pcb)
    res_waive = run_krt(
        ".claude/skills/plan-pcb-placement/scripts/waive_locked_overlaps.py",
        [str(locked_ref_check_pcb), str(intent_json)],
        timeout_s=25.0,
    )
    locked_ref_check_pcb.unlink(missing_ok=True)
    print(res_waive.stdout.strip())
    doc_intent = json.loads(intent_json.read_text(encoding="utf-8"))

    # Synthesize datasheet switching-stage, power-entry, and repeated actuator channel templates
    if not (custom_intent and custom_intent.exists()):
        doc_intent = synthesize_templates_and_proximity(pcb_fixed, fixed_pcb, doc_intent, set(locked_refs))
        intent_json.write_text(json.dumps(doc_intent, indent=2), encoding="utf-8")

    # Prepare `routing_strategy.json`
    if model_first and (isinstance(custom_routing_strategy, dict) or (isinstance(custom_routing_strategy, Path) and custom_routing_strategy.exists())):
        if isinstance(custom_routing_strategy, dict):
            strat_doc = dict(custom_routing_strategy)
        else:
            strat_doc = json.loads(custom_routing_strategy.read_text(encoding="utf-8"))
        # Only fill missing top-level keys if absent; do NOT overwrite model's power_width_tiers/ground_plane_nets/rules
        if not strat_doc.get("via_and_trace_rules"):
            strat_doc["via_and_trace_rules"] = {"clearance_mm": 0.1, "signal_track_width_mm": 0.15}
    else:
        strat_doc = synthesize_routing_strategy(human_unrouted)
        phys_vtr = dict(strat_doc.get("via_and_trace_rules") or {})
        phys_pwt = dict(strat_doc.get("power_width_tiers") or {})
        phys_gp = list(strat_doc.get("ground_plane_nets") or [])
        if isinstance(custom_routing_strategy, dict):
            for k, v in custom_routing_strategy.items():
                if isinstance(v, dict) and isinstance(strat_doc.get(k), dict):
                    strat_doc[k].update(v)
                else:
                    strat_doc[k] = v
        elif isinstance(custom_routing_strategy, Path) and custom_routing_strategy.exists():
            try:
                loaded_rs = json.loads(custom_routing_strategy.read_text(encoding="utf-8"))
                if isinstance(loaded_rs, dict):
                    for k, v in loaded_rs.items():
                        if isinstance(v, dict) and isinstance(strat_doc.get(k), dict):
                            strat_doc[k].update(v)
                        else:
                            strat_doc[k] = v
            except Exception:
                pass
        # Enforce physical fine-pitch clearance/track-width ceilings and verified power-net rails
        vtr = strat_doc.setdefault("via_and_trace_rules", {})
        for k_ceil in ("clearance_mm", "signal_track_width_mm", "same_net_pad_clearance_mm", "via_size_mm", "via_drill_mm"):
            if k_ceil in phys_vtr and k_ceil in vtr:
                vtr[k_ceil] = min(float(vtr[k_ceil]), float(phys_vtr[k_ceil]))
        strat_doc["power_width_tiers"] = phys_pwt
        strat_doc["ground_plane_nets"] = phys_gp
    (work_dir / "routing_strategy.json").write_text(json.dumps(strat_doc, indent=2), encoding="utf-8")
    schema_inv_pen = 0.0
    schema_diag: dict = {}
    pcr_raw = strat_doc.get("placer_channel_reservation")
    if isinstance(pcr_raw, dict):
        pcr = pcr_raw
    else:
        if pcr_raw is not None:
            schema_inv_pen += 15.0
            schema_diag["invalid_placer_channel_reservation_type"] = type(pcr_raw).__name__
        pcr = {}

    def _safe_float(val, default: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    if model_first:
        halo_coef = _safe_float(pcr.get("halo_coef"), 0.45)
        halo_weight = _safe_float(pcr.get("halo_weight"), 20.0)
        cong_weight = _safe_float(pcr.get("congestion_weight"), 12.0)
        pad_weight = _safe_float(pcr.get("pad_weight"), 6.0)
        track_pitch = _safe_float(pcr.get("track_pitch"), 0.40)
    else:
        halo_coef = round(max(0.25, min(0.80, _safe_float(pcr.get("halo_coef"), 0.45))), 2)
        halo_weight = round(max(10.0, min(40.0, _safe_float(pcr.get("halo_weight"), 20.0))), 1)
        cong_weight = round(max(5.0, min(25.0, _safe_float(pcr.get("congestion_weight"), 12.0))), 1)
        pad_weight = round(max(2.0, min(15.0, _safe_float(pcr.get("pad_weight"), 6.0))), 1)
        track_pitch = round(max(0.30, min(0.60, _safe_float(pcr.get("track_pitch"), 0.40))), 2)

    groups_json = work_dir / "groups.json"
    group_warnings_penalty = 0.0
    if custom_groups and custom_groups.exists():
        raw_groups_doc = json.loads(custom_groups.read_text(encoding="utf-8"))
        norm_groups_doc, g_diag, g_inv_pen = _diagnose_raw_groups_invariants(
            raw_groups_doc, pcb_fixed, set(locked_refs)
        )
        schema_inv_pen += g_inv_pen
        schema_diag.update(g_diag)
        # Measure raw LLM grouping cohesion before auto-repair so GRPO rewards clean LLM groups
        try:
            raw_rep = analyze(norm_groups_doc, pcb_fixed, extra_fixed=sorted(locked_refs))
            group_warnings_penalty = round(
                min(6.0, 0.4 * len(raw_rep.get("pulled", [])) + 0.8 * len(raw_rep.get("decap_cross_group", []))),
                2,
            )
        except Exception:
            group_warnings_penalty = 6.0
        if model_first:
            groups_doc = norm_groups_doc
        else:
            groups_doc = repair_and_enrich_groups(norm_groups_doc, pcb_fixed, doc_intent, set(locked_refs))
        groups_json.write_text(json.dumps(groups_doc, indent=2), encoding="utf-8")
    else:
        raw_groups_doc = extract_human_groups(human_unrouted, fixed_refs=set(locked_refs))
        groups_doc = repair_and_enrich_groups(raw_groups_doc, pcb_fixed, doc_intent, set(locked_refs))
        groups_json.write_text(json.dumps(groups_doc, indent=2), encoding="utf-8")

    # Validate groups.json against fixed_pcb + intent.json
    res_chk = run_krt(
        "py_tools/check_groups.py",
        [str(fixed_pcb), "--groups", str(groups_json), "--intent", str(intent_json)],
    )
    print(res_chk.stdout.strip())

    # Step 4: Run `place_simple.py` across scattered + clustered + anchored seeds with IC routing channel reservation
    t_place0 = time.time()
    place_args = [
        str(fixed_pcb),
        str(work_dir / "s{seed}.kicad_pcb"),
        "--groups",
        str(groups_json),
        "--intent",
        str(intent_json),
        "--scattered",
        str(scattered_seeds),
        "--clustered",
        str(clustered_seeds),
        "--anchored",
        str(anchored_seeds),
        "--judge",
        str(judge_count),
        "--jobs",
        str(jobs),
        "--halo-coef",
        str(halo_coef),
        "--halo-weight",
        str(halo_weight),
        "--congestion-weight",
        str(cong_weight),
        "--pad-weight",
        str(pad_weight),
        "--track-pitch",
        str(track_pitch),
        "--repair-decaps",
        "--json",
        str(work_dir / "s{seed}.json"),
    ]
    res_place = run_krt("py_placer/place_simple.py", place_args, timeout_s=540.0)
    place_secs = round(time.time() - t_place0, 2)
    print(res_place.stdout.strip())
    if res_place.returncode not in (0, 4):
        print("STDERR:", res_place.stderr)

    # Step 5: Fast-route the judged seeds via `judge_seeds.py` (`--quick 2`)
    chain_txt_str, prod_steps = build_smart_routing_chains(human_unrouted, routing_strategy=strat_doc)
    if model_first and custom_judge_chain and custom_judge_chain.exists():
        raw_jc = custom_judge_chain.read_text(encoding="utf-8").strip()
        if raw_jc:
            chain_txt_str = raw_jc
    chain_txt = work_dir / "judge_chain.txt"
    chain_txt.write_text(chain_txt_str, encoding="utf-8")
    judge_json = work_dir / "judge.json"

    # If place_simple found no 100% legal seed (judge: []), pick the least-defective seed from illegal[]
    # so judge_seeds.py still fast-routes it and returns a concrete reward + route_review feedback!
    seed_override = []
    if judge_json.exists():
        jdoc = json.loads(judge_json.read_text(encoding="utf-8"))
        if not jdoc.get("judge") and jdoc.get("illegal"):
            fp_map = jdoc.get("floorplan") or {}
            asm_map = jdoc.get("assembly") or {}
            ranked_ill = sorted(
                jdoc["illegal"],
                key=lambda s: (
                    (fp_map.get(str(s)) or 0) + (asm_map.get(str(s)) or 0),
                    s,
                ),
            )
            seed_override = ["--seeds", ",".join(str(s) for s in ranked_ill[:judge_count])]

    t_route0 = time.time()
    res_judge = run_krt(
        "py_placer/judge_seeds.py",
        [
            str(judge_json),
            "--chain",
            str(chain_txt),
            "--intent",
            str(intent_json),
            "--jobs",
            str(min(jobs, 2)),
            *seed_override,
        ],
        timeout_s=240.0,
    )
    route_secs = round(time.time() - t_route0, 2)
    print(res_judge.stdout.strip())

    # Parse best seed and run Turn-2 `route_review.py` diagnostic on it
    best_seed = None
    best_score = None
    for line in res_judge.stdout.splitlines():
        if line.startswith("JSON_SUMMARY:"):
            summary_payload = json.loads(line.partition(":")[2].strip())
            best_seed = summary_payload.get("best")
            break

    review_summary = {}
    phys_quality = {}
    if best_seed is not None:
        score_file = work_dir / f"s{best_seed}_fast.score.json"
        fast_pcb = work_dir / f"s{best_seed}_fast.kicad_pcb"
        placed_seed_pcb = work_dir / f"s{best_seed}.kicad_pcb"

        # Initial score + Turn-2 `route_review.py` diagnostic on winning fast route
        if score_file.exists():
            best_score = json.loads(score_file.read_text(encoding="utf-8"))
        review_json = work_dir / f"s{best_seed}_review.json"
        if fast_pcb.exists():
            run_krt("py_tools/route_review.py", [str(fast_pcb), "--json", str(review_json)], quiet=True)
            if review_json.exists():
                rev_data = json.loads(review_json.read_text(encoding="utf-8"))
                review_summary = {
                    "open_nets": [o["net"] for o in rev_data.get("open_nets", []) if not o.get("poured")],
                    "turned_parts": [
                        {"ref": t["ref"], "cmd": t.get("swap_pose")}
                        for t in rev_data.get("turned_parts", [])
                    ],
                    "congested_areas_count": len(rev_data.get("congested_areas", [])),
                }

        # Turn-2 `place_pose.py` orientation repair when `blocking > 0` or `turned_parts` is non-empty
        cur_blk = (best_score or {}).get("blocking")
        if (not model_first) and placed_seed_pcb.exists() and ((cur_blk is not None and cur_blk > 0) or review_summary.get("turned_parts")):
            pose_ops: list[str] = []
            seen_rot_refs: set[str] = set()
            for tp in review_summary.get("turned_parts", []):
                ref_t = tp.get("ref")
                cmd_t = str(tp.get("cmd") or "")
                m_rot = re.search(r"rotate\s+(\S+)\s+(-?\d+)", cmd_t)
                if m_rot and m_rot.group(1) not in seen_rot_refs and m_rot.group(1) not in locked_refs:
                    pose_ops.extend(["rotate", m_rot.group(1), m_rot.group(2)])
                    seen_rot_refs.add(m_rot.group(1))

            # Also check ratsnest crossing / wirelength across {0, 90, 180, 270} on multi-pin ICs with open nets
            if cur_blk and cur_blk > 0:
                try:
                    pcb_s = parse_kicad_pcb(str(placed_seed_pcb))
                    fps_s = pcb_s.footprints or {}
                    pads_by_net_s: dict[int, list[tuple[str, float, float]]] = {}
                    for r_s, fp_s in fps_s.items():
                        for p_s in fp_s.pads:
                            if p_s.net_id > 0:
                                pads_by_net_s.setdefault(p_s.net_id, []).append((r_s, p_s.global_x, p_s.global_y))
                    for r_s, fp_s in fps_s.items():
                        if r_s in locked_refs or r_s in seen_rot_refs or not r_s.startswith(("U", "IC")) or len(fp_s.pads) < 16:
                            continue
                        cur_rot = int(round(float(fp_s.rotation or 0.0))) % 360
                        rel_pads = []
                        for p_s in fp_s.pads:
                            if p_s.net_id <= 0:
                                continue
                            others_pts = [(ox, oy) for orf, ox, oy in pads_by_net_s.get(p_s.net_id, []) if orf != r_s]
                            if not others_pts or len(others_pts) > 8:
                                continue
                            cx_o = sum(x for x, _ in others_pts) / len(others_pts)
                            cy_o = sum(y for _, y in others_pts) / len(others_pts)
                            rel_pads.append((p_s.global_x - fp_s.x, p_s.global_y - fp_s.y, cx_o, cy_o))
                        if len(rel_pads) < 6:
                            continue
                        best_cand_rot = cur_rot
                        cur_wl = sum(math.hypot((fp_s.x + dx) - ox, (fp_s.y + dy) - oy) for dx, dy, ox, oy in rel_pads)
                        best_wl = cur_wl
                        for delta_deg in (90, 180, 270):
                            rad = math.radians(delta_deg)
                            cs, sn = math.cos(rad), math.sin(rad)
                            wl = sum(
                                math.hypot((fp_s.x + dx * cs - dy * sn) - ox, (fp_s.y + dx * sn + dy * cs) - oy)
                                for dx, dy, ox, oy in rel_pads
                            )
                            if wl < best_wl * 0.92:
                                best_wl = wl
                                best_cand_rot = (cur_rot + delta_deg) % 360
                        if best_cand_rot != cur_rot:
                            pose_ops.extend(["rotate", r_s, str(best_cand_rot)])
                            seen_rot_refs.add(r_s)
                except Exception:
                    pass

            if pose_ops:
                t2_pcb = work_dir / f"s{best_seed}_t2.kicad_pcb"
                res_pose = run_krt(
                    "py_placer/place_pose.py",
                    [str(placed_seed_pcb), str(t2_pcb), "--intent", str(intent_json), *pose_ops],
                    quiet=True,
                )
                if res_pose.returncode == 0 and t2_pcb.exists():
                    t2_dir = work_dir / f"s{best_seed}_t2_route"
                    t2_dir.mkdir(parents=True, exist_ok=True)
                    cur_t2 = t2_pcb
                    for l_idx, line_cmd in enumerate(judge_chain_txt.strip().splitlines()):
                        step_t2_out = t2_dir / f"step_{l_idx}.kicad_pcb"
                        rendered_cmd = (
                            line_cmd.replace("{in}", f"'{cur_t2}'")
                            .replace("{out}", f"'{step_t2_out}'")
                            .replace("{dir}", f"'{t2_dir}'")
                        )
                        if rendered_cmd.startswith("python3 -X utf8 "):
                            rendered_cmd = rendered_cmd.replace("python3 -X utf8 ", f"'{sys.executable}' -X utf8 '{KRT_ROOT}/", 1)
                            # Quote script path
                            parts_c = rendered_cmd.split(" ", 3)
                            if len(parts_c) >= 4:
                                rendered_cmd = f"{parts_c[0]} {parts_c[1]} {parts_c[2]}' {parts_c[3]}"
                        subprocess.run(rendered_cmd, shell=True, cwd=str(KRT_ROOT), env={**os.environ, "PYTHONPATH": str(KRT_ROOT), "KICAD_PACK_INLINE": "1"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
                        if step_t2_out.exists():
                            cur_t2 = step_t2_out
                    t2_score_file = t2_dir / "t2.score.json"
                    if cur_t2.exists() and cur_t2 != t2_pcb:
                        run_krt("py_tools/quick_score.py", [str(cur_t2), "--intent", str(intent_json), "--baseline", str(fixed_pcb), "--json", str(t2_score_file)], quiet=True)
                        if t2_score_file.exists():
                            t2_sc = json.loads(t2_score_file.read_text(encoding="utf-8"))
                            t2_blk = t2_sc.get("blocking")
                            if t2_blk is not None and (cur_blk is None or t2_blk < cur_blk or (t2_blk == cur_blk and (t2_sc.get("quality", {}).get("vias", 9999) < (best_score or {}).get("quality", {}).get("vias", 9999)))):
                                print(f"  [Turn-2 Pose Repair] {pose_ops} improved blocking {cur_blk} -> {t2_blk}!")
                                shutil.copyfile(t2_pcb, placed_seed_pcb)
                                shutil.copyfile(cur_t2, fast_pcb)
                                shutil.copyfile(t2_score_file, score_file)
                                best_score = t2_sc
                                review_summary["applied_pose_cmds"] = [" ".join(["place_pose.py", *pose_ops])]

        # Turn-2 targeted rip-up retry when 1..6 signal/power nets remain open/broken (`joins_needed > 0`)
        cur_joins = (best_score or {}).get("joins_needed", 0)
        if fast_pcb.exists() and cur_joins and 0 < cur_joins <= 6 and review_json.exists():
            try:
                rev_data_now = json.loads(review_json.read_text(encoding="utf-8"))
                open_sig_objs = [o for o in rev_data_now.get("open_nets", []) if not o.get("poured")]
                open_sig_nets = [o["net"] for o in open_sig_objs if o.get("net")]
                open_parts = {p for o in open_sig_objs for p in (o.get("parts") or [])}
                neighbor_detours = [
                    d["net"]
                    for d in (rev_data_now.get("longest_detours") or [])[:6]
                    if d.get("net") and (set(d.get("parts") or []) & open_parts)
                ]
                retry_nets = list(dict.fromkeys(open_sig_nets + neighbor_detours))[:8]
                if retry_nets:
                    default_vtr = (synthesize_routing_strategy(human_unrouted).get("via_and_trace_rules") or {})
                    vtr_doc = (strat_doc or {}).get("via_and_trace_rules") or {}
                    clr_s = str(min(float(vtr_doc.get("clearance_mm", 0.15)), float(default_vtr.get("clearance_mm", 0.15))))
                    tw_s = str(min(float(vtr_doc.get("signal_track_width_mm", 0.18)), float(default_vtr.get("signal_track_width_mm", 0.18))))
                    vs_s = str(min(float(vtr_doc.get("via_size_mm", 0.6)), float(default_vtr.get("via_size_mm", 0.6))))
                    vd_s = str(min(float(vtr_doc.get("via_drill_mm", 0.3)), float(default_vtr.get("via_drill_mm", 0.3))))
                    t2_rip_pcb = work_dir / f"s{best_seed}_t2_rip.kicad_pcb"
                    t2_rip_sc_file = work_dir / f"s{best_seed}_t2_rip.score.json"
                    run_krt(
                        "py_router/route.py",
                        [
                            str(fast_pcb),
                            str(t2_rip_pcb),
                            "--nets",
                            *retry_nets,
                            "--rip-existing-nets",
                            *retry_nets,
                            "--force-reroute",
                            "--max-ripup",
                            "12",
                            "--clearance",
                            clr_s,
                            "--track-width",
                            tw_s,
                            "--via-size",
                            vs_s,
                            "--via-drill",
                            vd_s,
                        ],
                        quiet=True,
                        timeout_s=60.0,
                    )
                    if t2_rip_pcb.exists():
                        run_krt(
                            "py_tools/quick_score.py",
                            [str(t2_rip_pcb), "--intent", str(intent_json), "--baseline", str(fixed_pcb), "--json", str(t2_rip_sc_file)],
                            quiet=True,
                        )
                        if t2_rip_sc_file.exists():
                            rip_sc = json.loads(t2_rip_sc_file.read_text(encoding="utf-8"))
                            rip_blk = rip_sc.get("blocking")
                            old_blk = (best_score or {}).get("blocking")
                            if rip_blk is not None and (old_blk is None or rip_blk < old_blk):
                                print(f"  [Turn-2 Rip-Up Repair] rerouted {retry_nets}: blocking {old_blk} -> {rip_blk}!")
                                shutil.copyfile(t2_rip_pcb, fast_pcb)
                                shutil.copyfile(t2_rip_sc_file, score_file)
                                best_score = rip_sc
                                quoted_nets = " ".join(f"'{n}'" for n in retry_nets)
                                review_summary["applied_retry_cmds"] = [
                                    f"python3 -X utf8 py_router/route.py {{in}} {{out}} --nets {quoted_nets} --rip-existing-nets {quoted_nets} --force-reroute --max-ripup 12"
                                ]
            except Exception:
                pass

        if placed_seed_pcb.exists():
            hide_fab_value_labels(placed_seed_pcb)
            run_krt("py_placer/beautify_labels.py", [str(placed_seed_pcb), str(placed_seed_pcb)], quiet=True)
        if fast_pcb.exists():
            hide_fab_value_labels(fast_pcb)
            run_krt("py_placer/beautify_labels.py", [str(fast_pcb), str(fast_pcb)], quiet=True)

        if production_finish and placed_seed_pcb.exists():
            print(f"\n>>> Running Full Production Route & Silkscreen Polish on Winning Seed s{best_seed}...")
            cur_in = placed_seed_pcb
            for step_idx, (script_rel, step_args) in enumerate(prod_steps):
                step_out = work_dir / (
                    "final_production.kicad_pcb" if step_idx == len(prod_steps) - 1 else f"prod_step{step_idx}.kicad_pcb"
                )
                run_krt(script_rel, [str(cur_in), str(step_out), *step_args], timeout_s=300.0)
                if step_out.exists():
                    cur_in = step_out
            if (work_dir / "final_production.kicad_pcb").exists():
                hide_fab_value_labels(work_dir / "final_production.kicad_pcb")
                run_krt("py_placer/beautify_labels.py", [str(work_dir / "final_production.kicad_pcb"), str(work_dir / "final_production.kicad_pcb")], quiet=True)
                shutil.copyfile(work_dir / "final_production.kicad_pcb", fast_pcb)
                run_krt(
                    "py_tools/quick_score.py",
                    [str(fast_pcb), "--intent", str(intent_json), "--json", str(score_file)],
                    quiet=True,
                )

        if score_file.exists():
            best_score = json.loads(score_file.read_text(encoding="utf-8"))
        if fast_pcb.exists():
            phys_quality = evaluate_physical_board_quality(fast_pcb)
            res_rev = run_krt(
                "py_tools/route_review.py",
                [str(fast_pcb), "--json", str(review_json)],
                quiet=True,
            )
            if review_json.exists():
                rev_data = json.loads(review_json.read_text(encoding="utf-8"))
                applied_prev = review_summary.get("applied_pose_cmds", [])
                review_summary = {
                    "open_nets": [o["net"] for o in rev_data.get("open_nets", []) if not o.get("poured")],
                    "turned_parts": [
                        {"ref": t["ref"], "cmd": t.get("swap_pose")}
                        for t in rev_data.get("turned_parts", [])
                    ],
                    "congested_areas_count": len(rev_data.get("congested_areas", [])),
                }
                if applied_prev:
                    review_summary["applied_pose_cmds"] = applied_prev

    total_secs = round(time.time() - t_start, 2)
    phys_pen = float(phys_quality.get("physical_quality_penalty") or 0.0)
    base_drc_err, base_unconn = _run_kicad_drc_cached(fixed_pcb, work_dir / "baseline_drc.json")
    routed_drc_err, routed_unconn = (None, None)
    if best_seed is not None:
        routed_pcb_cand = work_dir / f"s{best_seed}_fast.kicad_pcb"
        if routed_pcb_cand.exists():
            routed_drc_err, routed_unconn = _run_kicad_drc_cached(routed_pcb_cand, work_dir / "routed_drc.json")
    excess_drc = (
        max(0, int(routed_drc_err) - int(base_drc_err or 0))
        if routed_drc_err is not None
        else 0
    )
    rl_reward = compute_rl_reward(
        best_score or {},
        group_warnings_penalty=group_warnings_penalty,
        physical_quality_penalty=phys_pen,
        schema_invariant_penalty=schema_inv_pen,
        excess_drc_errors=excess_drc,
        drc_unconnected=int(routed_unconn or 0),
    )

    rollout_report = {
        "board": board_name,
        "best_seed": best_seed,
        "repaired_groups_count": len(groups_doc.get("groups", [])),
        "rl_reward": rl_reward,
        "schema_diagnostics": schema_diag,
        "drc_metrics": {
            "baseline_drc_errors": base_drc_err,
            "baseline_unconnected": base_unconn,
            "routed_drc_errors": routed_drc_err,
            "excess_drc_errors": excess_drc if routed_drc_err is not None else None,
            "drc_unconnected": routed_unconn,
        },
        "best_score": {
            "blocking": best_score.get("blocking") if best_score else None,
            "blocking_by": best_score.get("blocking_by") if best_score else None,
            "joins_needed": best_score.get("joins_needed") if best_score else None,
            "quality": best_score.get("quality") if best_score else None,
        },
        "physical_quality": phys_quality,
        "turn2_route_review": review_summary,
        "timing_sec": {
            "place_sec": place_secs,
            "fast_route_sec": route_secs,
            "total_sec": total_secs,
        },
        "prompt_tokens_est": len(context_md.read_text(encoding="utf-8")) // 4,
    }
    (work_dir / "rollout_result.json").write_text(json.dumps(rollout_report, indent=2), encoding="utf-8")
    print(f"\n>>> ROLLOUT RESULT ({board_name}):")
    print(json.dumps(rollout_report, indent=2))
    return rollout_report


def run_human_ceiling(board_name: str, corpus_dir: Path, work_dir: Path, set_name: str | None = None) -> dict:
    """Fast-route the original human placement using the exact same judge_chain.txt & quick_score.py."""
    t_start = time.time()
    resolved_set, _, human_unrouted = resolve_board_paths(corpus_dir, board_name, set_name)
    if not human_unrouted.exists():
        raise FileNotFoundError(f"Unrouted board not found: {human_unrouted}")

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f" HUMAN PLACEMENT CEILING: {resolved_set}/{board_name} -> {work_dir}")
    print(f"=======================================================")

    run_krt("py_router/copy_board.py", [str(human_unrouted), str(work_dir / "s0.kicad_pcb")], quiet=True)
    intent_json = work_dir / "intent.json"
    pcb_human = parse_kicad_pcb(str(human_unrouted))
    doc_intent = emit_intent(
        pcb_human,
        str(human_unrouted),
        declare_classes=False,
        derive_decaps="strict",
        decaps_from=str(human_unrouted),
    )
    doc_intent["edge_connectors"] = [
        c for c in doc_intent.get("edge_connectors", []) if c.get("edge")
    ]
    if any(str(fp.layer).startswith("B") for fp in pcb_human.footprints.values()):
        doc_intent.setdefault("assembly", {})["sides"] = "both"
    intent_json.write_text(json.dumps(doc_intent, indent=2), encoding="utf-8")
    chain_txt = work_dir / "judge_chain.txt"
    chain_txt.write_text(build_judge_chain(human_unrouted), encoding="utf-8")
    judge_json = work_dir / "judge.json"
    judge_json.write_text(json.dumps({"judge": [0], "baseline": str(human_unrouted)}, indent=2), encoding="utf-8")

    res_judge = run_krt(
        "py_placer/judge_seeds.py",
        [
            str(judge_json),
            "--chain",
            str(chain_txt),
            "--intent",
            str(intent_json),
            "--jobs",
            "1",
        ],
    )
    print(res_judge.stdout.strip())
    score_file = work_dir / "s0_fast.score.json"
    best_score = json.loads(score_file.read_text(encoding="utf-8")) if score_file.exists() else None
    rl_reward = compute_rl_reward(best_score) if best_score else -500.0
    report = {
        "board": board_name,
        "mode": "human_ceiling",
        "rl_reward": rl_reward,
        "score": {
            "blocking": best_score.get("blocking") if best_score else None,
            "blocking_by": best_score.get("blocking_by") if best_score else None,
            "joins_needed": best_score.get("joins_needed") if best_score else None,
            "quality": best_score.get("quality") if best_score else None,
        },
        "total_sec": round(time.time() - t_start, 2),
    }
    (work_dir / "ceiling_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n>>> HUMAN CEILING RESULT ({board_name}):")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Single-Board Gym Rollout for KiCad Placement + Fast Route")
    parser.add_argument("--board", default="esp_prog", help="Board name in corpus (e.g. esp_prog, tigard, watchy)")
    parser.add_argument("--set", dest="set_name", default=None, help="Optional stress set name (e.g. set3, set1)")
    parser.add_argument("--mode", choices=["rollout", "human_ceiling"], default="rollout")
    parser.add_argument("--corpus", type=Path, default=Path("/Users/simon/Fine Tune/corpus"))
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--scattered", type=int, default=3, help="Number of scattered seeds")
    parser.add_argument("--clustered", type=int, default=3, help="Number of clustered seeds")
    parser.add_argument("--anchored", type=int, default=3, help="Number of anchored seeds")
    parser.add_argument("--judge", type=int, default=3, help="Number of top seeds to fast-route")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--groups", type=Path, default=None, help="Optional custom groups.json")
    parser.add_argument("--intent", type=Path, default=None, help="Optional custom intent.json")
    parser.add_argument("--routing-strategy", type=Path, default=None, help="Optional custom routing_strategy.json")
    parser.add_argument("--judge-chain", type=Path, default=None, help="Optional custom judge_chain.txt")
    parser.add_argument("--model-first", action="store_true", help="Honor raw model groups/intent/routing_strategy/judge_chain without Python repair/overwrites")
    parser.add_argument("--production-finish", action="store_true", help="Run full production route + silkscreen polish on winning seed")
    args = parser.parse_args()

    if args.mode == "human_ceiling":
        work_dir = args.workdir or (Path("/Users/simon/Fine Tune/runs") / f"ceiling_{args.board}")
        run_human_ceiling(board_name=args.board, corpus_dir=args.corpus, work_dir=work_dir, set_name=args.set_name)
    else:
        work_dir = args.workdir or (Path("/Users/simon/Fine Tune/runs") / f"rollout_{args.board}")
        run_rollout(
            board_name=args.board,
            corpus_dir=args.corpus,
            work_dir=work_dir,
            scattered_seeds=args.scattered,
            clustered_seeds=args.clustered,
            anchored_seeds=args.anchored,
            judge_count=args.judge,
            jobs=args.jobs,
            custom_groups=args.groups,
            custom_intent=args.intent,
            custom_routing_strategy=args.routing_strategy,
            custom_judge_chain=args.judge_chain,
            model_first=args.model_first,
            set_name=args.set_name,
            production_finish=args.production_finish,
        )


if __name__ == "__main__":
    main()
