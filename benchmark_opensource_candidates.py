#!/usr/bin/env python3
"""
benchmark_opensource_candidates.py
----------------------------------
Evaluates open-source candidate models from `runs/opensource_model_shootout/responses/`
against saved baselines (`qwen-2.5-coder-32b-base`, `finetuned-32b`, and frontier models)
across 4 verification tiers:
  1. Audit (Parse + Schema + Raw `check_groups.py` / `BoardView` diagnostics, zero ref repair)
  2. Strict Model-First Rollout (`gym_rollout.py --model-first` + `kicad-cli pcb drc` + `teel_score_100`)
  3. Sanitized Script-Assisted Rollout (`run_single_board_rollout_with_pred_groups` matching
     `benchmark_frontier_vs_finetuned.py`, plus `kicad-cli pcb drc` + `teel_score_100`)
  4. Monster-board (`hackrf_pro`) context & `check_groups` inventory check.

Outputs `/Users/simon/Fine Tune/runs/opensource_model_shootout/shootout_results.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
BENCH_DIR = WORKSPACE / "runs" / "frontier_benchmark"
SHOOTOUT_DIR = WORKSPACE / "runs" / "opensource_model_shootout"
RESP_DIR = SHOOTOUT_DIR / "responses"
AUDIT_BOARDS_DIR = SHOOTOUT_DIR / "audit_boards"
MF_DIR = SHOOTOUT_DIR / "rollouts_model_first"
ASSIST_DIR = SHOOTOUT_DIR / "rollouts_assisted"
RESULTS_JSON = SHOOTOUT_DIR / "shootout_results.json"

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
for _sub in ("py_router", "py_placer", "py_tools"):
    _p = str(KRT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from kicad_parser import parse_kicad_pcb  # noqa: E402
from placement.floorplan import IntentError, emit_intent, intent_from_dict  # noqa: E402
from placement.grouping import BoardView, GroupsError, analyze, validate_doc  # noqa: E402
from placement.seeder import stamp_locked  # noqa: E402
from placement.writer import write_placed_output  # noqa: E402

from benchmark_frontier_vs_finetuned import (  # noqa: E402
    extract_human_copper_and_vias,
    run_kicad_drc,
)
from gym_rollout import extract_human_groups, resolve_board_paths, run_krt  # noqa: E402
from pcb_design_intelligence import (  # noqa: E402
    build_compact_board_context,
    detect_mechanical_and_rf_locks,
    evaluate_physical_board_quality,
    extract_board_keepouts,
    normalize_rf_courtyard_wings,
)
from run_eval_and_grpo_pipeline import (  # noqa: E402
    extract_json_payload,
    run_single_board_rollout_with_pred_groups,
)
from run_model_first_benchmark import evaluate_rollout_dir, load_raw_finetuned_outputs  # noqa: E402

CURATED_BOARDS: list[tuple[str, str]] = [
    ("set6", "ch32v203_ev"),
    ("set2", "megadesk"),
    ("set3", "robocup_kicker"),
    ("set6", "minisolarmesh"),
    ("sethydro", "AI_Hydro_Controller"),
    ("sethydro", "HydroPCB"),
]
MONSTER_BOARD = ("set1monster", "hackrf_pro")

CANDIDATE_MODELS = [
    "qwen3-coder-30b-a3b",
    "qwen3.8-27b",
    "glm-4.7-flash",
    "deepseek-chat-v3.1",
]
BASELINE_MODELS = ["qwen-2.5-coder-32b-base", "finetuned-32b"]


def strict_json_parse(raw_text: str) -> dict | None:
    """Parse raw model output as complete JSON (allowing markdown code fences, NO truncation salvage)."""
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
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


def coerce_groups_doc_minimal(raw_groups: Any) -> tuple[dict | None, bool, bool]:
    """
    Wrap bare list/non-standard group names minimally so `validate_doc`/`check_groups.py` can inspect
    component coverage without touching `refs`, `anchor`, `near`, or `apart`.
    Returns `(doc_or_none, was_native_dict_with_groups, used_name_shim)`.
    """
    if isinstance(raw_groups, dict) and isinstance(raw_groups.get("groups"), list):
        doc = json.loads(json.dumps(raw_groups))
        shimmed = False
        for i, g in enumerate(doc["groups"]):
            if isinstance(g, dict) and not g.get("name"):
                g["name"] = str(g.get("id") or g.get("role") or f"grp_{i}").strip()
                shimmed = True
        doc.setdefault("schema", 1)
        return doc, True, shimmed

    if isinstance(raw_groups, dict) and raw_groups and all(isinstance(v, dict) for v in raw_groups.values()):
        norm = []
        for k, v in raw_groups.items():
            gc = dict(v)
            gc.setdefault("name", str(k))
            norm.append(gc)
        return {"schema": 1, "groups": norm}, False, True

    if isinstance(raw_groups, list):
        norm = []
        used = set()
        for i, g in enumerate(raw_groups):
            if not isinstance(g, dict):
                continue
            gc = dict(g)
            if not gc.get("name"):
                base = str(gc.get("id") or gc.get("role") or f"grp_{i}").strip()
                name = base if base not in used else f"{base}_{i}"
                gc["name"] = name
            used.add(gc["name"])
            norm.append(gc)
        return {"schema": 1, "groups": norm}, False, True

    return None, False, False


def ensure_audit_board(set_name: str, board_name: str) -> dict[str, Any]:
    """Create cached `fixed.kicad_pcb`, baseline `intent.json`, and movable ref set for one board."""
    bdir = AUDIT_BOARDS_DIR / board_name
    bdir.mkdir(parents=True, exist_ok=True)
    fixed_pcb = bdir / "fixed.kicad_pcb"
    intent_json = bdir / "intent.json"
    meta_json = bdir / "meta.json"

    _, human_routed, human_unrouted = resolve_board_paths(CORPUS_DIR, board_name, set_name)
    if not (fixed_pcb.exists() and intent_json.exists() and meta_json.exists()):
        pile_pcb = bdir / "pile.kicad_pcb"
        if not pile_pcb.exists():
            run_krt(".claude/skills/plan-pcb-placement/scripts/make_unplaced.py", [str(human_unrouted), str(pile_pcb)], quiet=True)
        pcb_human = parse_kicad_pcb(str(human_unrouted))
        pcb_pile = parse_kicad_pcb(str(pile_pcb))
        locked_refs, seat_placements = detect_mechanical_and_rf_locks(pcb_human, human_unrouted)
        if seat_placements:
            write_placed_output(str(pile_pcb), str(fixed_pcb), seat_placements, pcb_data=pcb_pile)
            stamp_locked(str(fixed_pcb), locked_refs)
        else:
            run_krt("py_router/copy_board.py", [str(pile_pcb), str(fixed_pcb)], quiet=True)
        normalize_rf_courtyard_wings(fixed_pcb)

        pcb_fixed = parse_kicad_pcb(str(fixed_pcb))
        doc_intent = emit_intent(
            pcb_fixed,
            str(fixed_pcb),
            declare_classes=False,
            derive_decaps="strict",
            decaps_from=str(human_unrouted),
        )
        locked_set = set(locked_refs)
        doc_intent["edge_connectors"] = [
            c for c in doc_intent.get("edge_connectors", [])
            if c.get("edge") and c["ref"] not in locked_set
        ]
        if isinstance(doc_intent.get("decaps"), dict) and "max_distance_mm" in doc_intent["decaps"]:
            doc_intent["decaps"]["max_distance_mm"] = max(float(doc_intent["decaps"]["max_distance_mm"]), 5.0)
        if any(str(fp.layer).startswith("B") for fp in pcb_human.footprints.values()):
            doc_intent.setdefault("assembly", {})["sides"] = "both"
        keepouts = extract_board_keepouts(human_routed, human_unrouted, pcb_human, locked_refs)
        if keepouts:
            doc_intent["keepouts"] = keepouts
        intent_json.write_text(json.dumps(doc_intent, indent=2), encoding="utf-8")

        view = BoardView(pcb_fixed)
        all_parts = sorted(view.parts.keys())
        movable = sorted(set(all_parts) - locked_set)
        meta_json.write_text(
            json.dumps(
                {
                    "board": board_name,
                    "set": set_name,
                    "all_parts_count": len(all_parts),
                    "locked_refs": sorted(locked_refs),
                    "movable_refs": movable,
                    "nets_count": len(pcb_fixed.nets),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    return {
        "fixed_pcb": fixed_pcb,
        "intent_json": intent_json,
        "meta": meta,
    }


def audit_single_response(model_slug: str, set_name: str, board_name: str, raw_text: str, extra_meta: dict | None = None) -> dict[str, Any]:
    """Audit raw response JSON, schema compliance, ref coverage, and `check_groups.py` diagnostics."""
    board_ctx = ensure_audit_board(set_name, board_name)
    fixed_pcb: Path = board_ctx["fixed_pcb"]
    canon_intent: Path = board_ctx["intent_json"]
    meta = board_ctx["meta"]
    movable_set = set(meta["movable_refs"])
    locked_set = set(meta["locked_refs"])
    valid_parts = movable_set | locked_set

    strict_payload = strict_json_parse(raw_text)
    salvaged_payload = extract_json_payload(raw_text) if strict_payload is None else strict_payload

    out: dict[str, Any] = {
        "model": model_slug,
        "board": board_name,
        "set": set_name,
        "strict_json_valid": strict_payload is not None,
        "salvaged_json_valid": salvaged_payload is not None,
        "truncated_finish": (extra_meta or {}).get("finish_reason") == "length",
        "llm_sec": (extra_meta or {}).get("llm_sec"),
        "prompt_variant": (extra_meta or {}).get("prompt_variant", "standard"),
        "prompt_tokens": (extra_meta or {}).get("prompt_tokens"),
        "completion_tokens": (extra_meta or {}).get("completion_tokens"),
        "groups_schema_exact": False,
        "groups_validate_doc_error": None,
        "groups_list_shim_valid": False,
        "intent_schema_exact": False,
        "intent_error": None,
        "check_groups_exit": None,
        "n_groups": 0,
        "n_movable_total": len(movable_set),
        "n_predicted_unique": 0,
        "duplicate_refs": [],
        "missing_movable": sorted(movable_set),
        "hallucinated_refs": [],
        "anchor_missing_from_refs": 0,
        "dangling_near": [],
        "dangling_apart": [],
        "decap_cross_group": [],
        "pulled_warnings": [],
    }

    payload = strict_payload or salvaged_payload
    if not isinstance(payload, dict):
        return out

    # 1. Intent exact schema check (`intent_from_dict` with kind/schema defaults matching `gym_rollout`)
    raw_intent = payload.get("intent")
    if isinstance(raw_intent, dict):
        test_intent = dict(raw_intent)
        test_intent.setdefault("kind", "floorplan-intent")
        test_intent.setdefault("schema", 1)
        try:
            intent_from_dict(test_intent)
            out["intent_schema_exact"] = True
        except Exception as exc:
            out["intent_error"] = str(exc)
    else:
        out["intent_error"] = f"intent is {type(raw_intent).__name__}"

    # 2. Exact native `groups` validation (without coercing list->dict or shimming missing names)
    raw_groups = payload.get("groups")
    if isinstance(raw_groups, dict):
        try:
            validate_doc(raw_groups)
            out["groups_schema_exact"] = True
        except Exception as exc:
            out["groups_validate_doc_error"] = str(exc)
    else:
        out["groups_validate_doc_error"] = f"top-level groups is {type(raw_groups).__name__}, expected dict"

    # 3. Minimal shim for `check_groups` & raw ref audit (zero ref repair)
    coerced_doc, _, _ = coerce_groups_doc_minimal(raw_groups)
    if coerced_doc is None:
        return out

    try:
        validate_doc(coerced_doc)
        out["groups_list_shim_valid"] = True
    except Exception as exc:
        if out["groups_validate_doc_error"] is None:
            out["groups_validate_doc_error"] = str(exc)

    groups_list = [g for g in (coerced_doc.get("groups") or []) if isinstance(g, dict)]
    out["n_groups"] = len(groups_list)
    names = {g.get("name") for g in groups_list if isinstance(g.get("name"), str) and g.get("name")}

    seen: set[str] = set()
    dups: list[str] = []
    hallucinated: list[str] = []
    anchor_bad = 0
    dangling_near: list[str] = []
    dangling_apart: list[str] = []

    for g in groups_list:
        refs = g.get("refs") if isinstance(g.get("refs"), list) else []
        for r in refs:
            if not isinstance(r, str):
                continue
            if r not in valid_parts:
                hallucinated.append(r)
            elif r in seen:
                dups.append(r)
            else:
                seen.add(r)
        anc = g.get("anchor")
        if anc and anc not in refs:
            anchor_bad += 1
        for k, bucket in (("near", dangling_near), ("apart", dangling_apart)):
            vals = g.get(k)
            if isinstance(vals, list):
                for target in vals:
                    if not isinstance(target, str) or target not in names:
                        bucket.append(f"{g.get('name')}->{target}")

    predicted_movable = seen & movable_set
    out["n_predicted_unique"] = len(seen)
    out["duplicate_refs"] = sorted(set(dups))
    out["missing_movable"] = sorted(movable_set - predicted_movable)
    out["hallucinated_refs"] = sorted(set(hallucinated))
    out["anchor_missing_from_refs"] = anchor_bad
    out["dangling_near"] = dangling_near
    out["dangling_apart"] = dangling_apart

    # 4. Run actual KiCadRoutingTools `check_groups.py` CLI on minimally wrapped `coerced_doc`
    tmp_dir = AUDIT_BOARDS_DIR / "_tmp_chk" / model_slug / board_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_groups = tmp_dir / "groups.json"
    tmp_chk_json = tmp_dir / "check_out.json"
    tmp_groups.write_text(json.dumps(coerced_doc, indent=2), encoding="utf-8")

    res_chk = run_krt(
        "py_tools/check_groups.py",
        [str(fixed_pcb), "--groups", str(tmp_groups), "--intent", str(canon_intent), "--json", str(tmp_chk_json)],
        timeout_s=30.0,
        quiet=True,
    )
    out["check_groups_exit"] = res_chk.returncode
    for line in (res_chk.stdout or "").splitlines():
        if line.startswith("JSON_SUMMARY:"):
            try:
                js = json.loads(line.partition(":")[2].strip())
                out["decap_cross_group"] = js.get("decap_cross_group", [])
                out["pulled_warnings"] = js.get("pulled", [])
            except Exception:
                pass
            break
    return out


def load_candidate_response(model_slug: str, board_name: str) -> tuple[str, dict]:
    """Load saved raw text and metadata for candidate or baseline model."""
    if model_slug == "finetuned-32b":
        ft_map = load_raw_finetuned_outputs()
        return ft_map.get(board_name, ""), {"prompt_variant": "standard"}
    if model_slug == "qwen-2.5-coder-32b-base":
        p = BENCH_DIR / "responses" / "qwen-2.5-coder-32b-base" / f"{board_name}.json"
        if not p.exists():
            return "", {}
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("content") or d.get("raw_text") or "", {"llm_sec": d.get("llm_sec"), "prompt_variant": "standard"}

    p = RESP_DIR / model_slug / f"{board_name}.json"
    if not p.exists():
        return "", {}
    d = json.loads(p.read_text(encoding="utf-8"))
    raw_log = d.get("raw_step_log") or ""
    finish_reason = None
    m_fin = re.search(r"Finish Reason:\s*(\S+)", raw_log)
    if m_fin:
        finish_reason = m_fin.group(1)
    usage = d.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt is None:
        m_pt = re.search(r"Prompt:\s*(\d+)", raw_log)
        if m_pt:
            pt = int(m_pt.group(1))
    if ct is None:
        m_ct = re.search(r"Completion:\s*(\d+)", raw_log)
        if m_ct:
            ct = int(m_ct.group(1))
    return d.get("content") or "", {
        "llm_sec": d.get("llm_sec"),
        "prompt_variant": d.get("prompt_variant", "standard"),
        "finish_reason": finish_reason,
        "prompt_tokens": pt,
        "completion_tokens": ct,
    }


def run_strict_model_first_one(model_slug: str, set_name: str, board_name: str, raw_text: str) -> dict[str, Any]:
    """Run `gym_rollout.py --model-first` cleanly using exact model outputs (cached)."""
    work_dir = MF_DIR / model_slug / board_name
    in_dir = work_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    res_json = work_dir / "rollout_result.json"
    status_file = work_dir / "shootout_status.json"

    if status_file.exists():
        cached = json.loads(status_file.read_text(encoding="utf-8"))
        if cached.get("status") != "exception":
            return cached

    payload = strict_json_parse(raw_text)
    if payload is None:
        res = {
            "board": board_name,
            "model": model_slug,
            "status": "invalid_json",
            "error": "Strict JSON parse failed (syntax error or truncated output)",
            "blocking": None,
            "drc_errors": None,
            "teel_score_100": None,
        }
        status_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res

    if not res_json.exists():
        cmd = [
            sys.executable, "-X", "utf8",
            str(WORKSPACE / "gym_rollout.py"),
            "--board", board_name,
            "--set", set_name,
            "--corpus", str(CORPUS_DIR),
            "--workdir", str(work_dir),
            "--mode", "rollout",
            "--model-first",
            "--scattered", "2",
            "--clustered", "2",
            "--anchored", "2",
            "--judge", "2",
            "--jobs", "2",
        ]
        for key, fname, flag in (
            ("groups", "groups.json", "--groups"),
            ("intent", "intent.json", "--intent"),
            ("routing_strategy", "routing_strategy.json", "--routing-strategy"),
            ("judge_chain", "judge_chain.txt", "--judge-chain"),
        ):
            val = payload.get(key)
            if val is not None and val != "":
                p = in_dir / fname
                if key == "judge_chain":
                    txt = "\n".join(str(x) for x in val) if isinstance(val, list) else str(val)
                    p.write_text(txt.strip() + "\n", encoding="utf-8")
                else:
                    p.write_text(json.dumps(val, indent=2), encoding="utf-8")
                cmd.extend([flag, str(p)])

        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(WORKSPACE), start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            stdout_data, _ = proc.communicate(timeout=360)
            (work_dir / "rollout.log").write_text(stdout_data or "", encoding="utf-8")
            run_status = "ok" if proc.returncode == 0 and res_json.exists() else "rollout_failed"
            err = None if run_status == "ok" else (stdout_data or "")[-600:]
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.communicate()
            run_status, err = "timeout", "360s timeout"
        elapsed = round(time.time() - t0, 2)
    else:
        run_status, err, elapsed = "ok", None, None

    ev = evaluate_rollout_dir(board_name, work_dir, run_status=run_status, error=err)
    ev["model"] = model_slug
    if ev.get("timing_sec") is None and elapsed is not None:
        ev["timing_sec"] = elapsed
    status_file.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    return ev


def run_assisted_one(model_slug: str, set_name: str, board_name: str, raw_text: str) -> dict[str, Any]:
    """
    Run script-assisted rollout (`run_single_board_rollout_with_pred_groups`) matching
    `benchmark_frontier_vs_finetuned.py`: applies list->dict shim + `sanitize_predicted_groups` + `repair_and_enrich_groups`.
    """
    work_dir = ASSIST_DIR / model_slug / board_name
    status_file = work_dir / "assisted_status.json"
    if status_file.exists():
        return json.loads(status_file.read_text(encoding="utf-8"))

    work_dir.mkdir(parents=True, exist_ok=True)
    payload = extract_json_payload(raw_text)
    raw_groups = payload.get("groups") if isinstance(payload, dict) else None
    coerced_groups, _, _ = coerce_groups_doc_minimal(raw_groups)

    if coerced_groups is None or not coerced_groups.get("groups"):
        res = {
            "board": board_name,
            "model": model_slug,
            "status": "invalid_json",
            "blocking": None,
            "drc_errors": None,
            "drc_unconnected": None,
            "teel_score_100": None,
            "copper_mm": None,
            "vias": None,
            "timing_sec": None,
            "rl_reward": None,
        }
        status_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res

    pred_rs = payload.get("routing_strategy") if isinstance(payload.get("routing_strategy"), dict) else None
    t0 = time.time()
    rr = run_single_board_rollout_with_pred_groups(
        set_name=set_name,
        board_name=board_name,
        pred_groups_doc=coerced_groups,
        work_dir=work_dir,
        pred_routing_strategy=pred_rs,
        scattered=2,
        clustered=2,
        judge=2,
        timeout_s=300,
    )
    elapsed = round(time.time() - t0, 2)
    if not rr:
        res = {
            "board": board_name,
            "model": model_slug,
            "status": "rollout_failed",
            "blocking": None,
            "drc_errors": None,
            "drc_unconnected": None,
            "teel_score_100": None,
            "copper_mm": None,
            "vias": None,
            "timing_sec": elapsed,
            "rl_reward": None,
        }
        status_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res

    best_seed = rr.get("best_seed")
    pcb_path = Path(rr["winner_routed_pcb"]) if rr.get("winner_routed_pcb") else (work_dir / f"s{best_seed}_fast.kicad_pcb")
    drc = run_kicad_drc(pcb_path, work_dir / "drc.json") if pcb_path.exists() else {"drc_errors": None, "drc_unconnected": None}
    pq = rr.get("physical_quality") or {}
    if "teel_score_100" not in pq and pcb_path.exists():
        try:
            pq = evaluate_physical_board_quality(pcb_path)
        except Exception:
            pq = {}

    bs = rr.get("best_score") or {}
    qual = bs.get("quality") or {}
    timing_obj = rr.get("timing_sec")
    total_sec = timing_obj.get("total_sec") if isinstance(timing_obj, dict) else (timing_obj or elapsed)
    res = {
        "board": board_name,
        "model": model_slug,
        "status": "ok",
        "blocking": bs.get("blocking"),
        "joins_needed": bs.get("joins_needed"),
        "rl_reward": rr.get("rl_reward"),
        "drc_errors": drc.get("drc_errors"),
        "drc_unconnected": drc.get("drc_unconnected"),
        "teel_score_100": pq.get("teel_score_100"),
        "copper_mm": round(qual["copper_mm"], 1) if qual.get("copper_mm") is not None else None,
        "vias": qual.get("vias"),
        "timing_sec": total_sec,
        "group_sanitization": rr.get("group_sanitization"),
    }
    status_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def inspect_monster_board() -> dict[str, Any]:
    """Verify static inventory, context sizes, and baseline `check_groups` on `hackrf_pro`."""
    set_name, board_name = MONSTER_BOARD
    ctx = ensure_audit_board(set_name, board_name)
    fixed_pcb: Path = ctx["fixed_pcb"]
    canon_intent: Path = ctx["intent_json"]
    meta = ctx["meta"]
    pcb_fixed = parse_kicad_pcb(str(fixed_pcb))
    raw_ctx = build_compact_board_context(fixed_pcb)

    # Check lossless folded char length & human heuristic `check_groups` sanity
    human_groups = extract_human_groups(fixed_pcb, fixed_refs=set(meta["locked_refs"]))
    tmp_g = AUDIT_BOARDS_DIR / board_name / "human_groups.json"
    tmp_g.write_text(json.dumps(human_groups, indent=2), encoding="utf-8")
    res_chk = run_krt(
        "py_tools/check_groups.py",
        [str(fixed_pcb), "--groups", str(tmp_g), "--intent", str(canon_intent)],
        timeout_s=45.0,
        quiet=True,
    )
    return {
        "board": board_name,
        "set": set_name,
        "all_parts": meta["all_parts_count"],
        "locked_parts": len(meta["locked_refs"]),
        "movable_parts": len(meta["movable_refs"]),
        "nets": meta["nets_count"],
        "raw_context_chars": len(raw_ctx),
        "raw_est_tokens_2chars": len(raw_ctx) // 2,
        "human_groups_count": len(human_groups.get("groups", [])),
        "human_check_groups_exit": res_chk.returncode,
        "mcp_blocked_reason": "openrouter_chat MCP silently drops user prompts >~10KB (102,378 raw chars / 39,481 lossless-folded chars)",
    }


def load_saved_baselines() -> dict[str, Any]:
    """Load exact saved metrics on the 6 curated boards from `frontier_benchmark`."""
    out: dict[str, Any] = {"model_first": {}, "assisted": {}, "human": {}}
    mf_path = BENCH_DIR / "model_first_results.json"
    if mf_path.exists():
        mf = json.loads(mf_path.read_text(encoding="utf-8"))
        for key in ("human_routed", "script_assisted_baseline", "model_first"):
            out["model_first"][key] = {
                b: mf.get(key, {}).get(b)
                for _, b in CURATED_BOARDS
                if b in (mf.get(key) or {})
            }
        out["human"] = out["model_first"].get("human_routed", {})

    fvs_path = WORKSPACE / "runs" / "frontier_vs_finetuned_summary.json"
    if fvs_path.exists():
        fvs = json.loads(fvs_path.read_text(encoding="utf-8"))
        models_map = fvs.get("models") or {}
        for mslug in ("qwen-2.5-coder-32b-base", "finetuned-32b", "gpt-6.1-sol-pro", "claude-opus-5.5", "gemini-4-carbon"):
            if mslug in models_map:
                out["assisted"][mslug] = {
                    b: models_map[mslug].get(b)
                    for _, b in CURATED_BOARDS
                    if b in (models_map[mslug] or {})
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark open-source candidate models on 6 curated PCB boards + hackrf_pro.")
    parser.add_argument("--stage", choices=["all", "audit", "model_first", "assisted"], default="all")
    parser.add_argument("--models", nargs="+", default=CANDIDATE_MODELS)
    parser.add_argument("--jobs", type=int, default=2, help="Parallel board rollout workers")
    args = parser.parse_args()

    SHOOTOUT_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}

    audit_models = list(dict.fromkeys(args.models + BASELINE_MODELS))
    audit_results: dict[str, dict[str, dict]] = existing.get("audit", {})
    if args.stage in ("all", "audit"):
        print("\n=== STAGE 1: PARSE, SCHEMA & RAW CHECK_GROUPS AUDIT ===")
        for mslug in audit_models:
            audit_results.setdefault(mslug, {})
            for sname, bname in CURATED_BOARDS:
                raw_text, meta = load_candidate_response(mslug, bname)
                if not raw_text:
                    print(f"  [{mslug:24s} | {bname:20s}] MISSING")
                    continue
                rec = audit_single_response(mslug, sname, bname, raw_text, meta)
                audit_results[mslug][bname] = rec
                print(
                    f"  {mslug:24s} | {bname:20s} | json={'S' if rec['strict_json_valid'] else ('~' if rec['salvaged_json_valid'] else 'N')} "
                    f"| g_exact={int(rec['groups_schema_exact'])} | chk={rec['check_groups_exit']} "
                    f"| dup={len(rec['duplicate_refs'])} miss={len(rec['missing_movable'])} hall={len(rec['hallucinated_refs'])} near={len(rec['dangling_near'])}"
                )

    mf_results: dict[str, dict[str, dict]] = existing.get("model_first", {})
    if args.stage in ("all", "model_first"):
        print("\n=== STAGE 2: STRICT --model-first ROLLOUTS ===")
        tasks = []
        for mslug in args.models:
            mf_results.setdefault(mslug, {})
            for sname, bname in CURATED_BOARDS:
                raw_text, _ = load_candidate_response(mslug, bname)
                if raw_text:
                    tasks.append((mslug, sname, bname, raw_text))
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = {ex.submit(run_strict_model_first_one, m, s, b, r): (m, b) for m, s, b, r in tasks}
            for fut in as_completed(futs):
                m, b = futs[fut]
                rec = fut.result()
                mf_results[m][b] = rec
                print(
                    f"  [MF] {m:20s} | {b:20s} | status={rec.get('status')} | blk={rec.get('blocking')} "
                    f"| drc={rec.get('drc_errors')} | Teel={rec.get('teel_score_100')} | t={rec.get('timing_sec')}s"
                )

    assist_results: dict[str, dict[str, dict]] = existing.get("assisted", {})
    if args.stage in ("all", "assisted"):
        print("\n=== STAGE 3: SANITIZED SCRIPT-ASSISTED ROLLOUTS ===")
        tasks = []
        for mslug in args.models:
            assist_results.setdefault(mslug, {})
            for sname, bname in CURATED_BOARDS:
                raw_text, _ = load_candidate_response(mslug, bname)
                if raw_text:
                    tasks.append((mslug, sname, bname, raw_text))
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = {ex.submit(run_assisted_one, m, s, b, r): (m, b) for m, s, b, r in tasks}
            for fut in as_completed(futs):
                m, b = futs[fut]
                rec = fut.result()
                assist_results[m][b] = rec
                print(
                    f"  [SA] {m:20s} | {b:20s} | status={rec.get('status')} | blk={rec.get('blocking')} "
                    f"| drc={rec.get('drc_errors')} | Teel={rec.get('teel_score_100')} | t={rec.get('timing_sec')}s"
                )

    monster = existing.get("monster_board") or inspect_monster_board()
    baselines = load_saved_baselines()

    summary_out = {
        "curated_boards": [b for _, b in CURATED_BOARDS],
        "candidate_models": args.models,
        "audit": audit_results,
        "model_first": mf_results,
        "assisted": assist_results,
        "baselines": baselines,
        "monster_board": monster,
    }
    RESULTS_JSON.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
