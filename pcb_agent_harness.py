#!/usr/bin/env python3
"""
pcb_agent_harness.py
--------------------
Minimal Qwen-Code / DeepSeek / OpenAI tool-calling Agent Harness for closed-loop
PCB placement and incremental routing control over KiCadRoutingTools (KRT).

Provides:
  - `TOOLS_SCHEMA`: Standard OpenAI-format JSON tool definitions for 10 tools:
      1. `get_board_overview`
      2. `inspect_subsystem`
      3. `validate_groups`
      4. `validate_floorplan`
      5. `run_placement`
      6. `route_nets`
      7. `run_routing_chain`
      8. `inspect_routing_failures`
      9. `ripup_and_reroute`
     10. `patch_layout`
     11. `undo_last_patch`
  - `PCBAgentSession`: Stateful session executing tool calls against real KRT
    binaries/scripts and `kicad-cli`, maintaining stepped board snapshots,
    sibling `.kicad_pro`/`.kicad_dru` files, and reversible turn history.
  - `run_agent_loop` & `make_openai_chat_fn`: OpenAI/Qwen/DeepSeek tool-calling
    driver supporting any compatible chat callable (live HTTP or scripted test).
"""

from __future__ import annotations

import copy
import fnmatch
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"

for _krt_sub in ("py_router", "py_placer", "py_tools"):
    _sub_str = str(KRT_ROOT / _krt_sub)
    if _sub_str not in sys.path:
        sys.path.insert(0, _sub_str)

from copy_board import SIBLING_EXTS, copy_board, copy_siblings  # noqa: E402
from kicad_parser import non_aperture_pads, parse_kicad_pcb  # noqa: E402
from list_nets import find_power_nets, read_design_rules  # noqa: E402
from placement.floorplan import (  # noqa: E402
    IntentError,
    KIND as INTENT_KIND,
    emit_intent,
    intent_from_dict,
    validate_intent,
)
from placement.grouping import (  # noqa: E402
    BoardView,
    GroupsError,
    analyze as analyze_groups,
    claim_rows,
    follower_roots,
    is_ground,
    validate_doc as validate_groups_doc,
)
from placement.groups import decap_population  # noqa: E402
from placement.part_class import classify_part  # noqa: E402
from placement.roles import infer_role  # noqa: E402
from placement.seeder import stamp_locked  # noqa: E402
from placement.writer import write_placed_output  # noqa: E402

WORKSPACE = Path("/Users/simon/Fine Tune")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from pcb_design_intelligence import (  # noqa: E402
    build_smart_routing_chains,
    detect_mechanical_and_rf_locks,
    evaluate_physical_board_quality,
    normalize_rf_courtyard_wings,
    synthesize_routing_strategy,
)


# Allowed KRT scripts for `run_routing_chain` (prevents arbitrary command execution).
_ALLOWED_CHAIN_SCRIPTS = {
    "py_router/add_rule_area.py",
    "py_router/route_planes.py",
    "py_router/route_diff.py",
    "py_router/route.py",
    "py_router/repair_planes.py",
    "py_tools/fill_for_delivery.py",
}


# ---------------------------------------------------------------------------
# OpenAI / Qwen-Code / DeepSeek JSON Tool Schemas
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_board_overview",
            "description": (
                "Return a compact summary of the PCB (bounds, copper layers, DRC design rules, "
                "power/GND net candidates, fixed/locked parts, top ICs by pin count, grouping/floorplan "
                "readiness, and routing status). Stays compact even on 500+ part boards."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_subsystem",
            "description": (
                "Inspect the localized neighborhood around a component reference (e.g. 'U1', 'J2') "
                "or net name (e.g. 'I2C_SDA', '+3.3V') up to `radius_hops` signal hops, returning "
                "local netlist connectivity, attached decoupling caps, crystal claims, connector "
                "fan-in, and pin counts without overflowing context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ic_ref_or_net": {
                        "type": "string",
                        "description": "Component reference (e.g. 'U1') or net name (e.g. 'SPI_CLK').",
                    },
                    "radius_hops": {
                        "type": "integer",
                        "description": "Signal-graph hop radius from focal part/net (1..3, default 1).",
                        "default": 1,
                    },
                    "max_parts": {
                        "type": "integer",
                        "description": "Maximum neighborhood parts to include (1..100, default 40).",
                        "default": 40,
                    },
                },
                "required": ["ic_ref_or_net"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_groups",
            "description": (
                "Validate a functional placement `groups.json` payload against the board and "
                "current intent using KiCadRoutingTools `check_groups.py`, returning exact "
                "duplicate refs, dangling `near`/`apart` references, missing movable parts, "
                "unknown refs, and grouping cohesion warnings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "groups_json": {
                        "type": "object",
                        "description": "Groups JSON object with `schema: 1` and `groups: [...]`.",
                    },
                },
                "required": ["groups_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_floorplan",
            "description": (
                "Validate a `floorplan-intent` JSON payload against the board using `check_floorplan.py`. "
                "Ensures `kind='floorplan-intent'`, validates `decap_pin_distance` / `decaps`, "
                "`keepouts`, `edge_connectors`, `blocks`, and `proximity` cleanly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_json": {
                        "type": "object",
                        "description": "Floorplan intent JSON object (`kind='floorplan-intent'`, `schema=1`).",
                    },
                },
                "required": ["intent_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_placement",
            "description": (
                "Run `place_simple.py` using the validated `groups.json` and `intent.json` without "
                "overriding them. Selects the best legal seed (or least-defective seed if none legal) "
                "and returns top seed costs, floorplan errors, assembly status, and ratsnest wirelength."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scattered": {"type": "integer", "default": 2},
                    "clustered": {"type": "integer", "default": 2},
                    "anchored": {"type": "integer", "default": 2},
                    "jobs": {"type": "integer", "default": 2},
                    "judge_count": {"type": "integer", "default": 2},
                    "repair_decaps": {"type": "boolean", "default": True},
                    "halo_coef": {"type": "number", "default": 0.45},
                    "halo_weight": {"type": "number", "default": 20.0},
                    "congestion_weight": {"type": "number", "default": 12.0},
                    "pad_weight": {"type": "number", "default": 6.0},
                    "track_pitch": {"type": "number", "default": 0.4},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_nets",
            "description": (
                "Incrementally route a subset of nets (or `['*']` for remaining unrouted nets) "
                "on the current board snapshot using `route.py` or `route_planes.py` (when "
                "`plane_layers` is provided), preserving earlier routed copper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Net names or fnmatch patterns (e.g. ['+3.3V', '+5V'] or ['*']).",
                    },
                    "layers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Copper layers to route on (e.g. ['F.Cu', 'B.Cu']).",
                    },
                    "plane_layers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "If set, creates pours via `route_planes.py` paired positionally with `nets`.",
                    },
                    "track_width": {"type": "number"},
                    "clearance": {"type": "number"},
                    "clearance_ceiling": {"type": "number", "description": "Optional cap clamping EVERY net class (including Power) to this clearance in mm."},
                    "via_size": {"type": "number"},
                    "via_drill": {"type": "number"},
                    "via_cost": {"type": "integer"},
                    "turn_cost": {"type": "integer"},
                    "direction_preference_cost": {"type": "integer"},
                    "same_net_pad_clearance": {"type": "number"},
                    "bus": {"type": "boolean", "default": False},
                    "ordering": {
                        "type": "string",
                        "enum": ["inside_out", "mps", "original", "bus"],
                    },
                    "layer_costs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Per-layer cost multipliers matching `layers` (-1 = forbidden obstacle-only).",
                    },
                    "keepout": {"type": "boolean", "default": False},
                    "keepout_layer": {"type": "string"},
                    "guide_corridor": {"type": "boolean", "default": False},
                    "guide_corridor_layer": {"type": "string"},
                    "keep_away": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keep-away rules in AGG:VICTIM:GAP format.",
                    },
                    "power_nets": {"type": "array", "items": {"type": "string"}},
                    "power_nets_widths": {"type": "array", "items": {"type": "number"}},
                    "max_ripup": {"type": "integer"},
                    "rip_existing_nets": {"type": "array", "items": {"type": "string"}},
                    "force_reroute": {"type": "boolean", "default": False},
                    "keep_input_copper": {"type": "boolean", "default": True},
                    "quick": {"type": "integer", "description": "Optional quick judge level (1 or 2). Omit for full route."},
                    "write_fill": {"type": "boolean", "default": False},
                    "add_teardrops": {"type": "boolean", "default": False},
                },
                "required": ["nets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_routing_chain",
            "description": (
                "Execute a multi-step routing strategy (`strategy_json`) or explicit list of KRT "
                "chain commands (`commands` using `{in}`, `{out}`, `{dir}`) against the current board."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_json": {
                        "type": "object",
                        "description": "Routing strategy JSON (ground_plane_nets, via_and_trace_rules, power_width_tiers, etc.).",
                    },
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional ordered list of KRT chain commands (`python3 -X utf8 py_router/... {in} {out}`).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_routing_failures",
            "description": (
                "Inspect the current board's connectivity (`blocking_by`, unrouted/broken nets with "
                "pad coordinates and bounding boxes), native `kicad-cli pcb drc --refill-zones` error "
                "breakdown, `route_review.py` congestion/detours/turned parts, and `teel_score_100` sub-metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ripup_and_reroute",
            "description": (
                "Cleanly strip copper segments/arcs/vias belonging ONLY to `nets` (preserving all other "
                "routed traces) and optionally re-route them with updated `new_params`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact net names or patterns to rip up (and re-route if `reroute=true`).",
                    },
                    "new_params": {
                        "type": "object",
                        "description": "Updated routing parameters (track_width, clearance, via_size, via_drill, max_ripup, layer_costs, ordering, bus, etc.).",
                    },
                    "reroute": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true (default), re-routes `nets` after ripping them up; if false, leaves them unrouted.",
                    },
                },
                "required": ["nets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_layout",
            "description": (
                "Apply targeted Turn-2+ layout repairs: footprint rotations/moves via `place_pose.py`, "
                "`groups.json` edits, or `intent.json` edits, with optional `re_place=true`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rotations": {
                        "type": "object",
                        "description": "Map of ref -> degrees (absolute) or {deg, relative}.",
                    },
                    "moves": {
                        "type": "object",
                        "description": "Map of ref -> [x, y] or {x, y, rot?, near?}.",
                    },
                    "group_edits": {
                        "type": "object",
                        "description": "Full groups dict or patch ({move_refs, add_groups, remove_groups, update_groups}).",
                    },
                    "intent_edits": {
                        "type": "object",
                        "description": "Full or partial intent dict to merge and validate.",
                    },
                    "re_place": {"type": "boolean", "default": False},
                    "preserve_unrelated_copper": {"type": "boolean", "default": False},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_last_patch",
            "description": (
                "Revert the most recent mutating operation (placement, routing, rip-up, or layout patch) "
                "back to the prior snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _resolve_krt_python() -> str:
    """Return a Python executable that has numpy/scipy/shapely for KRT scripts."""
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import shapely  # noqa: F401
        return sys.executable
    except Exception:
        for cand in ("/usr/bin/python3", shutil.which("python3") or ""):
            if cand and os.path.exists(cand):
                return cand
        return sys.executable


def _run_subprocess(
    cmd: List[str],
    cwd: Path = KRT_ROOT,
    timeout_s: float = 180.0,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess in its own process group with clean timeout cleanup."""
    env = dict(os.environ)
    env["KICAD_PACK_INLINE"] = "1"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout or "", stderr or "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout or "",
            f"TimeoutExpired after {timeout_s}s\n{stderr or ''}",
        )


def _run_krt(
    rel_script: str,
    args: List[str],
    timeout_s: float = 180.0,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    py_exe = _resolve_krt_python()
    cmd = [py_exe, "-X", "utf8", str(KRT_ROOT / rel_script)] + [str(a) for a in args]
    return _run_subprocess(cmd, cwd=KRT_ROOT, timeout_s=timeout_s, extra_env=extra_env)


def _extract_json_summary(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last `JSON_SUMMARY: {...}` payload from tool stdout."""
    last_obj: Optional[Dict[str, Any]] = None
    for line in (text or "").splitlines():
        if line.startswith("JSON_SUMMARY:"):
            raw = line.partition(":")[2].strip()
            try:
                last_obj = json.loads(raw)
            except Exception:
                pass
    return last_obj


def _load_json_arg(val: Union[Dict[str, Any], str, Path], label: str) -> Dict[str, Any]:
    """Load a JSON object from a dict, JSON string, or file path."""
    if isinstance(val, dict):
        return copy.deepcopy(val)
    if isinstance(val, Path):
        return json.loads(val.read_text(encoding="utf-8"))
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{"):
            return json.loads(s)
        p = Path(s)
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return json.loads(s)
    raise ValueError(f"{label} must be a JSON object, JSON string, or existing file path")


def _resolve_board_paths(
    corpus_dir: Path, board_name: str, set_name: Optional[str] = None
) -> Tuple[str, Optional[Path], Path]:
    """Locate (set_name, human_routed, human_unrouted) under corpus_dir or direct path."""
    direct = Path(board_name)
    if direct.suffix == ".kicad_pcb" and direct.exists():
        return set_name or "direct", None, direct.resolve()

    corpus_dir = corpus_dir.resolve()
    if set_name:
        s = set_name if set_name.startswith("set") else f"set{set_name}"
        unrouted = corpus_dir / f"boards_unrouted_{s}" / f"{board_name}.kicad_pcb"
        routed = corpus_dir / f"boards_{s}" / f"{board_name}.kicad_pcb"
        if unrouted.exists():
            return s, (routed if routed.exists() else None), unrouted
        if routed.exists():
            return s, routed, routed

    for unrouted_dir in sorted(corpus_dir.glob("boards_unrouted_set*")):
        cand = unrouted_dir / f"{board_name}.kicad_pcb"
        if cand.exists():
            s = unrouted_dir.name.replace("boards_unrouted_", "")
            routed = corpus_dir / f"boards_{s}" / f"{board_name}.kicad_pcb"
            return s, (routed if routed.exists() else None), cand

    for routed_dir in sorted(corpus_dir.glob("boards_set*")):
        cand = routed_dir / f"{board_name}.kicad_pcb"
        if cand.exists():
            s = routed_dir.name.replace("boards_", "")
            return s, cand, cand

    raise FileNotFoundError(
        f"Board '{board_name}' not found under {corpus_dir} (set_name={set_name!r})"
    )


def _board_has_copper(pcb) -> bool:
    routed_segs = [s for s in (pcb.segments or ()) if not getattr(s, "graphic", False) and s.net_id > 0]
    return bool(routed_segs or (pcb.vias or ()) or (pcb.zones or ()))


def _copper_counts_by_net(pcb) -> Dict[str, Dict[str, int]]:
    """Return {net_name: {'segments': int, 'vias': int, 'zones': int}}."""
    counts: Dict[str, Dict[str, int]] = {}
    nets = pcb.nets or {}
    for s in pcb.segments or ():
        if getattr(s, "graphic", False) or s.net_id <= 0:
            continue
        nm = getattr(nets.get(s.net_id), "name", None)
        if not nm:
            continue
        entry = counts.setdefault(nm, {"segments": 0, "vias": 0, "zones": 0})
        entry["segments"] += 1
    for v in pcb.vias or ():
        if v.net_id <= 0:
            continue
        nm = getattr(nets.get(v.net_id), "name", None)
        if not nm:
            continue
        entry = counts.setdefault(nm, {"segments": 0, "vias": 0, "zones": 0})
        entry["vias"] += 1
    for z in pcb.zones or ():
        nm = getattr(z, "net_name", None)
        if not nm:
            continue
        entry = counts.setdefault(nm, {"segments": 0, "vias": 0, "zones": 0})
        entry["zones"] += 1
    return counts


def _run_kicad_drc_fresh(pcb_path: Path, out_json: Path, refill_zones: bool = True) -> Dict[str, Any]:
    """Run fresh `kicad-cli pcb drc` and return error/warning/unconnected breakdown."""
    if not pcb_path.exists():
        return {"ran": False, "error": f"missing file {pcb_path}"}
    if not os.path.exists(KICAD_CLI):
        return {"ran": False, "error": f"kicad-cli not found at {KICAD_CLI}"}

    cmd = [KICAD_CLI, "pcb", "drc", "--format", "json"]
    if refill_zones:
        cmd.append("--refill-zones")
    cmd.extend(["--output", str(out_json), str(pcb_path)])
    proc = _run_subprocess(cmd, cwd=pcb_path.parent, timeout_s=60.0)
    if not out_json.exists():
        return {
            "ran": False,
            "returncode": proc.returncode,
            "error": (proc.stderr or proc.stdout or "kicad-cli produced no report").strip(),
        }
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    all_v = rep.get("violations") or []
    errs = [v for v in all_v if v.get("severity") == "error"]
    warns = [v for v in all_v if v.get("severity") == "warning"]
    unconn = rep.get("unconnected_items") or []

    by_type = Counter(v.get("type", "unknown") for v in errs)
    sample_errs = []
    for v in errs[:15]:
        sample_errs.append(
            {
                "type": v.get("type"),
                "description": v.get("description"),
                "items": [
                    {
                        "description": it.get("description"),
                        "pos": it.get("pos"),
                    }
                    for it in (v.get("items") or [])[:3]
                ],
            }
        )
    sample_unconn = []
    for u in unconn[:15]:
        sample_unconn.append(
            {
                "description": u.get("description"),
                "items": [
                    {
                        "description": it.get("description"),
                        "pos": it.get("pos"),
                    }
                    for it in (u.get("items") or [])[:3]
                ],
            }
        )
    return {
        "ran": True,
        "drc_errors": len(errs),
        "drc_warnings": len(warns),
        "drc_unconnected": len(unconn),
        "errors_by_type": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sample_errors": sample_errs,
        "sample_unconnected": sample_unconn,
        "report_path": str(out_json),
    }


def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _clean_unrouted_board_is_pile(pcb, fixed_pcb_path: Path) -> bool:
    """Return True if movable parts are still stacked at the unplaced pile origin."""
    fps = pcb.footprints or {}
    if len(fps) < 2:
        return False
    coords = Counter((round(float(fp.x or 0.0), 1), round(float(fp.y or 0.0), 1)) for fp in fps.values())
    most_common_cnt = coords.most_common(1)[0][1]
    return most_common_cnt >= max(2, int(0.5 * len(fps)))


# ---------------------------------------------------------------------------
# Stateful Session & Tool Implementations
# ---------------------------------------------------------------------------


class PCBAgentSession:
    """
    Stateful closed-loop PCB placement & routing harness wrapping KiCadRoutingTools.
    """

    def __init__(
        self,
        board_name: str,
        corpus_dir: Union[str, Path],
        work_dir: Union[str, Path],
        set_name: Optional[str] = None,
        *,
        skip_unplace: bool = False,
        model_first: bool = True,
        timeout_s: float = 240.0,
    ) -> None:
        self.board_name = Path(board_name).stem if str(board_name).endswith(".kicad_pcb") else board_name
        self.corpus_dir = Path(corpus_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.steps_dir = self.work_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.work_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.model_first = bool(model_first)
        self.timeout_s = float(timeout_s)

        resolved_set, human_routed, human_unrouted = _resolve_board_paths(
            self.corpus_dir, str(board_name), set_name
        )
        self.set_name = resolved_set
        self.human_routed = human_routed
        self.human_unrouted = human_unrouted

        self.pile_pcb = self.work_dir / "pile.kicad_pcb"
        self.fixed_pcb = self.work_dir / "fixed.kicad_pcb"
        self.groups_path = self.work_dir / "groups.json"
        self.intent_path = self.work_dir / "intent.json"
        self.starter_intent_path = self.work_dir / "starter_intent.json"
        self.routing_strategy_path = self.work_dir / "routing_strategy.json"

        self.groups_doc: Optional[Dict[str, Any]] = None
        self.groups_validated: bool = False
        self.intent_doc: Optional[Dict[str, Any]] = None
        self.intent_validated: bool = False
        self.routing_strategy_doc: Optional[Dict[str, Any]] = None

        self.locked_refs: List[str] = []
        self.placed_pcb: Optional[Path] = None
        self.current_pcb: Path = self.fixed_pcb
        self.available_seeds: Dict[int, Path] = {}
        self.best_seed: Optional[int] = None
        self.last_judge_json: Optional[Path] = None

        self.step_counter: int = 0
        self.place_run_counter: int = 0
        self.snapshots: List[Dict[str, Any]] = []
        self.tool_history: List[Dict[str, Any]] = []
        self._pcb_cache: Dict[str, Tuple[Tuple[int, int], Any, BoardView]] = {}

        self._initialize_board(skip_unplace=skip_unplace)

    # -----------------------------------------------------------------------
    # Board Initialization & Caching
    # -----------------------------------------------------------------------

    def _initialize_board(self, skip_unplace: bool = False) -> None:
        if skip_unplace:
            copy_board(str(self.human_unrouted), str(self.pile_pcb))
            copy_board(str(self.human_unrouted), str(self.fixed_pcb))
            pcb_fixed = parse_kicad_pcb(str(self.fixed_pcb))
            self.locked_refs = sorted(
                r for r, fp in (pcb_fixed.footprints or {}).items() if getattr(fp, "locked", False)
            )
            if not _clean_unrouted_board_is_pile(pcb_fixed, self.fixed_pcb):
                self.placed_pcb = self.fixed_pcb
        else:
            res_unp = _run_krt(
                ".claude/skills/plan-pcb-placement/scripts/make_unplaced.py",
                [str(self.human_unrouted), str(self.pile_pcb)],
                timeout_s=120.0,
            )
            if res_unp.returncode != 0 or not self.pile_pcb.exists():
                raise RuntimeError(
                    f"make_unplaced.py failed (rc={res_unp.returncode}): "
                    f"{res_unp.stdout}\n{res_unp.stderr}"
                )
            copy_siblings(str(self.human_unrouted), str(self.pile_pcb))

            pcb_human = parse_kicad_pcb(str(self.human_unrouted))
            pcb_pile = parse_kicad_pcb(str(self.pile_pcb))
            locked_refs, seat_placements = detect_mechanical_and_rf_locks(pcb_human, self.human_unrouted)
            self.locked_refs = sorted(set(locked_refs))

            if seat_placements:
                write_placed_output(str(self.pile_pcb), str(self.fixed_pcb), seat_placements, pcb_data=pcb_pile)
                stamp_locked(str(self.fixed_pcb), self.locked_refs)
                copy_siblings(str(self.pile_pcb), str(self.fixed_pcb))
            else:
                copy_board(str(self.pile_pcb), str(self.fixed_pcb))
            normalize_rf_courtyard_wings(self.fixed_pcb)

        pcb_fixed, _ = self._get_pcb_and_view(self.fixed_pcb)
        starter = emit_intent(pcb_fixed, str(self.fixed_pcb), declare_classes=False, derive_decaps="off")
        starter.setdefault("kind", INTENT_KIND)
        starter.setdefault("schema", 1)
        if any(str(fp.layer).startswith("B") for fp in (pcb_fixed.footprints or {}).values()):
            starter.setdefault("assembly", {})["sides"] = "both"
        self.starter_intent_path.write_text(json.dumps(starter, indent=2), encoding="utf-8")
        self.current_pcb = self.placed_pcb if self.placed_pcb is not None else self.fixed_pcb

    def _get_pcb_and_view(self, pcb_path: Optional[Path] = None) -> Tuple[Any, BoardView]:
        p = Path(pcb_path or self.current_pcb).resolve()
        st = p.stat()
        sig = (st.st_mtime_ns, st.st_size)
        cached = self._pcb_cache.get(str(p))
        if cached and cached[0] == sig:
            return cached[1], cached[2]
        pcb = parse_kicad_pcb(str(p))
        view = BoardView(pcb)
        self._pcb_cache[str(p)] = (sig, pcb, view)
        return pcb, view

    def _next_step_pcb(self, label: str) -> Path:
        self.step_counter += 1
        safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)[:40]
        return self.steps_dir / f"step_{self.step_counter:03d}_{safe_label}.kicad_pcb"

    def _save_snapshot(self, action: str) -> None:
        snap_idx = len(self.snapshots) + 1
        snap_pcb = self.snapshots_dir / f"snap_{snap_idx:03d}_current.kicad_pcb"
        copy_board(str(self.current_pcb), str(snap_pcb))
        snap_placed = None
        if self.placed_pcb and self.placed_pcb.exists():
            if self.placed_pcb.resolve() == self.current_pcb.resolve():
                snap_placed = snap_pcb
            else:
                snap_placed = self.snapshots_dir / f"snap_{snap_idx:03d}_placed.kicad_pcb"
                copy_board(str(self.placed_pcb), str(snap_placed))
        self.snapshots.append(
            {
                "snap_idx": snap_idx,
                "action": action,
                "step_counter": self.step_counter,
                "current_pcb": snap_pcb,
                "placed_pcb": snap_placed,
                "best_seed": self.best_seed,
                "groups_doc": copy.deepcopy(self.groups_doc),
                "groups_validated": self.groups_validated,
                "intent_doc": copy.deepcopy(self.intent_doc),
                "intent_validated": self.intent_validated,
                "routing_strategy_doc": copy.deepcopy(self.routing_strategy_doc),
            }
        )

    def _safe_work_path(self, p: Path) -> Path:
        rp = p.resolve()
        if self.work_dir not in rp.parents and rp != self.work_dir:
            raise ValueError(f"Refusing write outside work_dir ({self.work_dir}): {rp}")
        return rp

    # -----------------------------------------------------------------------
    # Tool Dispatcher
    # -----------------------------------------------------------------------

    def dispatch_tool(self, name: str, arguments: Union[str, Dict[str, Any], None] = None) -> Dict[str, Any]:
        """Dispatch an OpenAI tool call by name and JSON/dict arguments."""
        if arguments is None:
            args: Dict[str, Any] = {}
        elif isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = dict(arguments)
        else:
            raise TypeError(f"Unsupported arguments type: {type(arguments).__name__}")

        tool_map: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_board_overview": self.get_board_overview,
            "inspect_subsystem": self.inspect_subsystem,
            "validate_groups": self.validate_groups,
            "validate_floorplan": self.validate_floorplan,
            "run_placement": self.run_placement,
            "route_nets": self.route_nets,
            "run_routing_chain": self.run_routing_chain,
            "inspect_routing_failures": self.inspect_routing_failures,
            "ripup_and_reroute": self.ripup_and_reroute,
            "patch_layout": self.patch_layout,
            "undo_last_patch": self.undo_last_patch,
        }
        if name not in tool_map:
            return {"ok": False, "error": f"Unknown tool '{name}'. Available: {sorted(tool_map)}"}

        t0 = time.time()
        try:
            res = tool_map[name](**args)
        except Exception as exc:
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dt = round(time.time() - t0, 3)

        self.tool_history.append(
            {
                "index": len(self.tool_history) + 1,
                "tool": name,
                "arguments": args,
                "duration_sec": dt,
                "ok": bool(res.get("ok", True)) if isinstance(res, dict) else True,
            }
        )
        return res

    # -----------------------------------------------------------------------
    # Tool 1: get_board_overview
    # -----------------------------------------------------------------------

    def get_board_overview(self) -> Dict[str, Any]:
        pcb, view = self._get_pcb_and_view(self.current_pcb)
        fps = pcb.footprints or {}
        bounds = getattr(pcb.board_info, "board_bounds", None)
        layers = list(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"])

        rules_info = read_design_rules(str(self.current_pcb))
        gnd_pairs, vcc_pairs, cand_pairs = find_power_nets(pcb)
        gnd_names = {n for n, _ in gnd_pairs}
        power_names = {n for n, _ in vcc_pairs} | {view.name_of(nid) for nid in view.rails if view.name_of(nid)}

        copper_counts = _copper_counts_by_net(pcb)
        routed_segs = sum(v["segments"] for v in copper_counts.values())
        routed_vias = sum(v["vias"] for v in copper_counts.values())
        zones_count = len(pcb.zones or ())

        prefix_hist: Counter = Counter()
        fixed_list = []
        ic_candidates = []
        locked_set = set(self.locked_refs)

        for ref in sorted(view.parts.keys()):
            fp = fps[ref]
            m_pref = re.match(r"^([A-Za-z]+)", ref)
            prefix_hist[m_pref.group(1) if m_pref else "OTHER"] += 1
            cls = classify_part(fp, ref).name
            role = infer_role(fp, ref, cls)["role"]
            npads = len(non_aperture_pads(fp))
            is_locked = bool(getattr(fp, "locked", False) or ref in locked_set)
            if is_locked:
                fixed_list.append(
                    {
                        "ref": ref,
                        "value": fp.value,
                        "x": round(float(fp.x or 0.0), 2),
                        "y": round(float(fp.y or 0.0), 2),
                        "rot": round(float(fp.rotation or 0.0), 1),
                        "layer": fp.layer,
                        "pads": npads,
                    }
                )
            if cls not in ("fiducial", "mount_hole", "testpoint") and (
                npads >= 6 or ref.startswith(("U", "IC", "J", "P", "CN")) or cls in ("edge_receptacle", "connector_affinity")
            ):
                net_names = sorted(
                    {
                        view.name_of(nid)
                        for nid in view.nets_of.get(ref, set())
                        if view.name_of(nid) and not view.name_of(nid).startswith("unconnected-")
                    }
                )
                sig_sample = [n for n in net_names if n not in gnd_names and n not in power_names][:8]
                rail_sample = [n for n in net_names if n in gnd_names or n in power_names][:6]
                ic_candidates.append(
                    (
                        -npads,
                        ref,
                        {
                            "ref": ref,
                            "value": fp.value,
                            "footprint": (fp.footprint_name or "").split(":")[-1],
                            "class": cls,
                            "role": role,
                            "pads": npads,
                            "fixed": is_locked,
                            "signal_nets_sample": sig_sample,
                            "rails": rail_sample,
                        },
                    )
                )

        ic_candidates.sort(key=lambda t: (t[0], t[1]))
        top_ics = [item[2] for item in ic_candidates[:20]]

        movable_refs = sorted(set(view.parts) - view.locked - locked_set)
        grouped_refs = set()
        if isinstance(self.groups_doc, dict):
            for g in self.groups_doc.get("groups") or []:
                grouped_refs.update(g.get("refs") or [])
        ungrouped = [r for r in movable_refs if r not in grouped_refs]

        stage = "routed" if _board_has_copper(pcb) else ("placed" if self.placed_pcb else "unplaced")
        routable_nets = [
            nid
            for nid, pads in (pcb.pads_by_net or {}).items()
            if nid > 0 and view.name_of(nid) and not view.name_of(nid).startswith("unconnected-") and len(pads) >= 2
        ]

        overview: Dict[str, Any] = {
            "ok": True,
            "board_name": self.board_name,
            "set_name": self.set_name,
            "stage": stage,
            "current_pcb": str(self.current_pcb),
            "board_bounds_mm": (
                {
                    "x0": round(bounds[0], 2),
                    "y0": round(bounds[1], 2),
                    "x1": round(bounds[2], 2),
                    "y1": round(bounds[3], 2),
                    "width_mm": round(bounds[2] - bounds[0], 2),
                    "height_mm": round(bounds[3] - bounds[1], 2),
                }
                if bounds
                else None
            ),
            "copper_layers": layers,
            "layer_count": len(layers),
            "design_rules": {
                "default": (rules_info.get("classes") or {}).get("Default", {}),
                "constraints": rules_info.get("constraints") or {},
                "effective": rules_info.get("effective") or {},
                "source": rules_info.get("source"),
            },
            "counts": {
                "total_parts": len(view.parts),
                "movable_parts": len(movable_refs),
                "fixed_locked_parts": len(fixed_list),
                "total_nets": len(pcb.nets or {}),
                "routable_nets": len(routable_nets),
                "routed_segments": routed_segs,
                "routed_vias": routed_vias,
                "zones": zones_count,
            },
            "component_prefix_histogram": dict(sorted(prefix_hist.items())),
            "power_and_gnd_nets": {
                "gnd_candidates": [{"net": n, "pads": c} for n, c in gnd_pairs[:10]],
                "rail_candidates": [{"net": n, "pads": c} for n, c in vcc_pairs[:20]],
                "high_fanout_candidates": [{"net": n, "pads": c} for n, c in cand_pairs[:8]],
            },
            "fixed_locked_parts": fixed_list[:25],
            "fixed_locked_truncated": max(0, len(fixed_list) - 25),
            "top_ics": top_ics,
            "top_ics_truncated": max(0, len(ic_candidates) - len(top_ics)),
            "grouping_status": {
                "groups_validated": self.groups_validated,
                "group_count": len((self.groups_doc or {}).get("groups") or []),
                "grouped_movable_count": len(set(movable_refs) & grouped_refs),
                "ungrouped_movable_count": len(ungrouped),
                "ungrouped_sample": ungrouped[:25],
            },
            "floorplan_status": {
                "intent_validated": self.intent_validated,
                "has_intent": self.intent_path.exists(),
            },
            "routing_status": {
                "has_copper": _board_has_copper(pcb),
                "nets_with_copper": len(copper_counts),
                "routable_nets": len(routable_nets),
            },
        }
        serialized = json.dumps(overview)
        overview["context_budget"] = {
            "chars": len(serialized),
            "est_tokens": max(1, len(serialized) // 4),
        }
        return overview

    # -----------------------------------------------------------------------
    # Tool 2: inspect_subsystem
    # -----------------------------------------------------------------------

    def inspect_subsystem(
        self,
        ic_ref_or_net: str,
        radius_hops: int = 1,
        max_parts: int = 40,
    ) -> Dict[str, Any]:
        pcb, view = self._get_pcb_and_view(self.current_pcb)
        fps = pcb.footprints or {}
        radius_hops = max(1, min(3, int(radius_hops)))
        max_parts = max(1, min(100, int(max_parts)))
        query = str(ic_ref_or_net).strip()

        gnd_pairs, vcc_pairs, _ = find_power_nets(pcb)
        rail_names = {n for n, _ in gnd_pairs} | {n for n, _ in vcc_pairs} | {
            view.name_of(nid) for nid in view.rails if view.name_of(nid)
        }
        name_to_nid = {net.name: nid for nid, net in (pcb.nets or {}).items() if net.name}

        resolved_as: Optional[str] = None
        if query in fps:
            resolved_as = "part"
        elif query in name_to_nid:
            resolved_as = "net"
        else:
            part_matches = [r for r in sorted(fps) if query.lower() in r.lower()][:10]
            net_matches = [n for n in sorted(name_to_nid) if query.lower() in n.lower()][:10]
            return {
                "ok": False,
                "query": query,
                "error": f"'{query}' did not match an exact component ref or net name on {self.board_name}.",
                "part_suggestions": part_matches,
                "net_suggestions": net_matches,
            }

        def _is_traversable_signal_nid(nid: int) -> bool:
            nm = view.name_of(nid)
            if not nm or nm.startswith("unconnected-") or nm in rail_names or is_ground(nm):
                return False
            # Skip board-wide broadcast nets (>20 parts) unless explicitly queried.
            return 2 <= len(view.owners.get(nid, ())) <= 20

        dist_map: Dict[str, int] = {}
        queue: deque = deque()
        if resolved_as == "part":
            dist_map[query] = 0
            queue.append(query)
        else:
            nid0 = name_to_nid[query]
            seed_parts = sorted(view.owners.get(nid0, set()))
            # Prioritize ICs and connectors if querying a huge power rail
            seed_parts.sort(
                key=lambda r: (
                    0 if r.startswith(("U", "IC", "J", "P")) else 1,
                    -len(non_aperture_pads(fps[r])) if r in fps else 0,
                    r,
                )
            )
            for r in seed_parts[:max_parts]:
                dist_map[r] = 0
                queue.append(r)

        # Bounded BFS across signal nets only so rails never blow up the neighborhood.
        while queue:
            cur = queue.popleft()
            d = dist_map[cur]
            if d >= radius_hops:
                continue
            if resolved_as == "net" and len(dist_map) >= max_parts:
                break
            for nid in sorted(view.nets_of.get(cur, set())):
                if not _is_traversable_signal_nid(nid):
                    continue
                for nb in sorted(view.owners.get(nid, set())):
                    if nb not in dist_map and nb in fps:
                        dist_map[nb] = d + 1
                        queue.append(nb)

        # Always pull in decaps explicitly owned by focal IC(s)
        pop = decap_population(pcb)
        attached_decaps: List[Dict[str, Any]] = []
        focal_ics = {query} if resolved_as == "part" else {
            r for r in dist_map if r.startswith(("U", "IC"))
        }
        for cap, (rids, owners) in sorted(pop.items()):
            shared = [o for o in owners if o in focal_ics]
            if shared and cap in fps:
                if cap not in dist_map:
                    dist_map[cap] = 1
                attached_decaps.append(
                    {
                        "ref": cap,
                        "value": fps[cap].value,
                        "rails": [view.name_of(rid) for rid in rids if view.name_of(rid)],
                    }
                )

        # Rank & cap neighborhood parts cleanly
        ranked_parts = sorted(
            dist_map.keys(),
            key=lambda r: (
                dist_map[r],
                -(len(non_aperture_pads(fps[r])) if r in fps else 0),
                r,
            ),
        )
        kept_parts = ranked_parts[:max_parts]
        kept_set = set(kept_parts)
        truncated_parts = max(0, len(ranked_parts) - len(kept_parts))

        locked_set = set(self.locked_refs)
        copper_counts = _copper_counts_by_net(pcb)
        neighborhood_parts = []
        connector_fan_in = []

        for ref in kept_parts:
            fp = fps[ref]
            cls = classify_part(fp, ref).name
            role = infer_role(fp, ref, cls)["role"]
            npads = len(non_aperture_pads(fp))
            sig_nets = sorted(
                {
                    view.name_of(nid)
                    for nid in view.nets_of.get(ref, set())
                    if view.name_of(nid)
                    and not view.name_of(nid).startswith("unconnected-")
                    and view.name_of(nid) not in rail_names
                }
            )
            rails = sorted(
                {
                    view.name_of(nid)
                    for nid in view.nets_of.get(ref, set())
                    if view.name_of(nid) in rail_names
                }
            )
            is_fixed = bool(getattr(fp, "locked", False) or ref in locked_set)
            part_entry: Dict[str, Any] = {
                "ref": ref,
                "hop": dist_map[ref],
                "value": fp.value,
                "footprint": (fp.footprint_name or "").split(":")[-1],
                "role": role,
                "pins": npads,
                "fixed": is_fixed,
                "rails": rails[:6],
                "sig_count": len(sig_nets),
            }
            is_placed_stage = self.placed_pcb is not None and self.placed_pcb.exists()
            if is_fixed or is_placed_stage:
                part_entry["pose"] = [
                    round(float(fp.x or 0.0), 2),
                    round(float(fp.y or 0.0), 2),
                    round(float(fp.rotation or 0.0), 1),
                    fp.layer,
                ]
            neighborhood_parts.append(part_entry)
            if ref != query and (cls in ("edge_receptacle", "connector_affinity") or ref.startswith(("J", "P", "CN"))):
                if resolved_as == "part" and query in fps:
                    shared_sigs = sorted(
                        {
                            view.name_of(nid)
                            for nid in (view.nets_of.get(ref, set()) & view.nets_of.get(query, set()))
                            if view.name_of(nid) and view.name_of(nid) not in rail_names
                        }
                    )
                else:
                    shared_sigs = sig_nets[:6]
                connector_fan_in.append(
                    {
                        "connector": ref,
                        "value": fp.value,
                        "pins": npads,
                        "hop": dist_map[ref],
                        "shared_signals": shared_sigs[:8],
                    }
                )

        local_signal_nets = []
        seen_nids = set()
        for ref in kept_parts:
            for nid in view.nets_of.get(ref, set()):
                if nid in seen_nids:
                    continue
                seen_nids.add(nid)
                nm = view.name_of(nid)
                if not nm or nm.startswith("unconnected-") or nm in rail_names or is_ground(nm):
                    continue
                owners_all = sorted(view.owners.get(nid, set()))
                owners_in_scope = [o for o in owners_all if o in kept_set]
                if len(owners_in_scope) >= 2 or (resolved_as == "part" and query in owners_in_scope):
                    c = copper_counts.get(nm, {"segments": 0, "vias": 0, "zones": 0})
                    net_entry: Dict[str, Any] = {
                        "net": nm,
                        "pads": len(pcb.pads_by_net.get(nid, ())),
                        "parts": owners_in_scope[:10],
                        "total_parts": len(owners_all),
                    }
                    if c["segments"] or c["vias"]:
                        net_entry["segs"] = c["segments"]
                        net_entry["vias"] = c["vias"]
                    local_signal_nets.append(net_entry)
        local_signal_nets.sort(key=lambda x: (-len(x["parts"]), x["net"]))

        intent_doc_cur = self.intent_doc if self.intent_doc is not None else {}
        c_rows, c_notes = claim_rows(pcb, intent_doc_cur, crystals=True)
        relevant_crystals = [
            line for line in c_notes if any(r in line for r in kept_set)
        ]

        if resolved_as == "part":
            fp0 = fps[query]
            is_fixed0 = bool(getattr(fp0, "locked", False)) or query in locked_set
            is_placed_stage0 = self.placed_pcb is not None and self.placed_pcb.exists()
            focal_summary: Dict[str, Any] = {
                "type": "part",
                "ref": query,
                "value": fp0.value,
                "footprint": (fp0.footprint_name or "").split(":")[-1],
                "pin_count": len(non_aperture_pads(fp0)),
                "fixed": is_fixed0,
                "pose": (
                    [
                        round(float(fp0.x or 0.0), 3),
                        round(float(fp0.y or 0.0), 3),
                        round(float(fp0.rotation or 0.0), 1),
                        fp0.layer,
                    ]
                    if (is_fixed0 or is_placed_stage0)
                    else None
                ),
                "rails": sorted(
                    {view.name_of(nid) for nid in view.nets_of.get(query, set()) if view.name_of(nid) in rail_names}
                ),
            }
        else:
            nid0 = name_to_nid[query]
            c0 = copper_counts.get(query, {"segments": 0, "vias": 0, "zones": 0})
            focal_summary = {
                "type": "net",
                "net": query,
                "is_rail_or_gnd": query in rail_names or is_ground(query),
                "total_pads": len(pcb.pads_by_net.get(nid0, ())),
                "total_parts": len(view.owners.get(nid0, ())),
                "routed_segments": c0["segments"],
                "routed_vias": c0["vias"],
                "zones": c0["zones"],
            }

        out = {
            "ok": True,
            "query": query,
            "resolved_as": resolved_as,
            "radius_hops": radius_hops,
            "max_parts": max_parts,
            "focal_summary": focal_summary,
            "attached_decaps": attached_decaps[:24],
            "crystal_claims": relevant_crystals,
            "connector_fan_in": connector_fan_in[:16],
            "neighborhood_parts": neighborhood_parts,
            "truncated_parts_count": truncated_parts,
            "local_signal_nets": local_signal_nets[:30],
            "local_signal_nets_truncated": max(0, len(local_signal_nets) - 30),
        }
        serialized = json.dumps(out)
        out["est_tokens"] = max(1, len(serialized) // 4)
        return out

    # -----------------------------------------------------------------------
    # Tool 3: validate_groups
    # -----------------------------------------------------------------------

    def _collect_group_diagnostics(self, doc: Dict[str, Any], pcb, view: BoardView) -> Dict[str, Any]:
        """Collect comprehensive diagnostics even when schema fails early."""
        schema_errors: List[str] = []
        dangling_refs: List[Dict[str, str]] = []
        duplicate_refs: Dict[str, List[str]] = {}
        unknown_refs: List[str] = []
        missing_movable_parts: List[str] = []
        empty_groups: List[str] = []

        if not isinstance(doc, dict):
            return {
                "schema_errors": ["Top-level groups payload must be a JSON object."],
                "dangling_refs": [],
                "duplicate_refs": {},
                "unknown_refs": [],
                "missing_movable_parts": [],
                "empty_groups": [],
            }

        groups = doc.get("groups")
        if not isinstance(groups, list) or not groups:
            schema_errors.append("`groups` must be a non-empty list.")
            return {
                "schema_errors": schema_errors,
                "dangling_refs": [],
                "duplicate_refs": {},
                "unknown_refs": [],
                "missing_movable_parts": [],
                "empty_groups": [],
            }

        group_names = [g.get("name") for g in groups if isinstance(g, dict) and isinstance(g.get("name"), str)]
        name_counts = Counter(group_names)
        for nm, cnt in sorted(name_counts.items()):
            if cnt > 1:
                schema_errors.append(f"Duplicate group name '{nm}' ({cnt}x).")
        name_set = set(group_names)

        gof: Dict[str, List[str]] = {}
        for i, g in enumerate(groups):
            if not isinstance(g, dict):
                schema_errors.append(f"groups[{i}] is not an object.")
                continue
            gname = g.get("name") or f"groups[{i}]"
            refs = g.get("refs")
            if not isinstance(refs, list) or not refs:
                schema_errors.append(f"Group '{gname}' has missing or empty `refs`.")
                continue
            if not any(r in view.parts for r in refs if isinstance(r, str)):
                empty_groups.append(gname)
            for r in refs:
                if isinstance(r, str):
                    gof.setdefault(r, []).append(str(gname))
            for rel_key in ("near", "apart"):
                rel_vals = g.get(rel_key) or []
                if isinstance(rel_vals, list):
                    for other in rel_vals:
                        if other not in name_set:
                            dangling_refs.append({"group": str(gname), "field": rel_key, "target": str(other)})

        duplicate_refs = {r: gnames for r, gnames in sorted(gof.items()) if len(gnames) > 1}
        all_fps = pcb.footprints or {}
        unknown_refs = sorted(r for r in gof if r not in all_fps)

        locked_set = set(self.locked_refs)
        seated: set = set()
        if isinstance(self.intent_doc, dict):
            for fp_entry in self.intent_doc.get("fixed_poses") or []:
                if isinstance(fp_entry, dict) and fp_entry.get("ref"):
                    seated.add(str(fp_entry["ref"]))
        movable = set(view.parts) - view.locked - locked_set - seated
        missing_movable_parts = sorted(r for r in movable if r not in gof)

        try:
            validate_groups_doc(copy.deepcopy(doc), "groups")
        except GroupsError as exc:
            msg = str(exc)
            if msg not in schema_errors:
                schema_errors.append(msg)

        return {
            "schema_errors": schema_errors,
            "dangling_refs": dangling_refs,
            "duplicate_refs": duplicate_refs,
            "unknown_refs": unknown_refs,
            "missing_movable_parts": missing_movable_parts,
            "empty_groups": empty_groups,
        }

    def validate_groups(self, groups_json: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
        try:
            doc = _load_json_arg(groups_json, "groups_json")
        except Exception as exc:
            return {"ok": True, "valid": False, "schema_errors": [f"Invalid JSON: {exc}"], "errors": [str(exc)]}

        pcb_fixed, view = self._get_pcb_and_view(self.fixed_pcb)
        diag = self._collect_group_diagnostics(doc, pcb_fixed, view)

        cand_path = self.work_dir / "groups_candidate.json"
        rep_path = self.work_dir / "groups_check_report.json"
        cand_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        args = [str(self.fixed_pcb), "--groups", str(cand_path), "--json", str(rep_path)]
        intent_for_chk = self.intent_path if self.intent_path.exists() else self.starter_intent_path
        if intent_for_chk.exists():
            args.extend(["--intent", str(intent_for_chk)])

        res = _run_krt("py_tools/check_groups.py", args, timeout_s=60.0)
        summary = _extract_json_summary(res.stdout) or {}
        rep_data: Dict[str, Any] = {}
        if rep_path.exists():
            try:
                rep_data = json.loads(rep_path.read_text(encoding="utf-8"))
            except Exception:
                rep_data = {}

        errors: List[str] = list(rep_data.get("errors") or [])
        for s_err in diag["schema_errors"]:
            if s_err not in errors:
                errors.append(s_err)
        if diag["dangling_refs"]:
            for d in diag["dangling_refs"]:
                msg = f"group '{d['group']}' `{d['field']}` references missing group '{d['target']}'"
                if not any(d["target"] in e for e in errors):
                    errors.append(msg)
        if diag["duplicate_refs"] and not any("more than one group" in e for e in errors):
            errors.append(
                "refs in more than one group: "
                + ", ".join(f"{r} ({'/'.join(gs)})" for r, gs in diag["duplicate_refs"].items())
            )
        if diag["missing_movable_parts"] and not any("movable parts in no group" in e for e in errors):
            errors.append(f"movable parts in no group: {diag['missing_movable_parts']}")
        if diag["unknown_refs"] and not any("not on this board" in e for e in errors):
            errors.append(f"refs not on this board: {diag['unknown_refs']}")

        valid = res.returncode == 0 and not errors and not diag["schema_errors"]
        if valid:
            self.groups_doc = copy.deepcopy(doc)
            self.groups_doc.setdefault("schema", 1)
            self.groups_path.write_text(json.dumps(self.groups_doc, indent=2), encoding="utf-8")
            self.groups_validated = True

        warnings = list(rep_data.get("warnings") or [])
        return {
            "ok": True,
            "valid": valid,
            "exit_code": res.returncode,
            "errors": errors,
            "schema_errors": diag["schema_errors"],
            "duplicate_refs": diag["duplicate_refs"],
            "dangling_refs": diag["dangling_refs"],
            "missing_movable_parts": diag["missing_movable_parts"],
            "unknown_refs": diag["unknown_refs"],
            "empty_groups": diag["empty_groups"],
            "warnings": warnings[:30],
            "warnings_count": len(warnings),
            "summary": summary,
            "stderr": res.stderr.strip() if res.returncode == 2 else "",
            "saved_path": str(self.groups_path) if valid else None,
        }

    # -----------------------------------------------------------------------
    # Tool 4: validate_floorplan
    # -----------------------------------------------------------------------

    def validate_floorplan(self, intent_json: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
        try:
            doc = _load_json_arg(intent_json, "intent_json")
        except Exception as exc:
            return {"ok": True, "valid": False, "errors": [f"Invalid JSON: {exc}"]}

        normalized_fields: List[str] = []
        if "kind" not in doc or doc.get("kind") is None:
            doc["kind"] = INTENT_KIND
            normalized_fields.append("kind='floorplan-intent'")
        if "schema" not in doc or doc.get("schema") is None:
            doc["schema"] = 1
            normalized_fields.append("schema=1")

        errors: List[str] = []
        warnings: List[str] = []

        pcb_check_path = self.placed_pcb if (self.placed_pcb and self.placed_pcb.exists()) else self.fixed_pcb
        pcb_obj, view = self._get_pcb_and_view(pcb_check_path)

        # Structural check via KRT's own intent_from_dict & validate_intent
        parsed_intent = None
        try:
            parsed_intent = intent_from_dict(doc, source_path="intent_candidate.json")
            self_vios = validate_intent(parsed_intent)
            for v in self_vios:
                msg = f"{v.rule}: {v.message}"
                if getattr(v, "severity", "error") == "error":
                    errors.append(msg)
                else:
                    warnings.append(msg)
        except IntentError as exc:
            errors.append(str(exc))

        # Check decaps / keepouts / edge_connectors explicitly for clear LLM feedback
        decaps_cfg = doc.get("decaps") or {}
        if isinstance(decaps_cfg, dict):
            max_pin = decaps_cfg.get("max_pin_distance_mm")
            max_dist = decaps_cfg.get("max_distance_mm")
            if max_pin is not None:
                try:
                    mp_f = float(max_pin)
                    if mp_f <= 0:
                        errors.append(f"decaps.max_pin_distance_mm ({mp_f}) must be > 0.")
                    elif mp_f < 2.0:
                        warnings.append(
                            f"decaps.max_pin_distance_mm={mp_f} mm is very tight (< 2.0 mm); "
                            "pad/courtyard clearance often makes <=1.5 mm unachievable in annealing."
                        )
                except (TypeError, ValueError):
                    errors.append(f"decaps.max_pin_distance_mm must be numeric, got {max_pin!r}")
            if max_dist is not None:
                try:
                    if float(max_dist) <= 0:
                        errors.append(f"decaps.max_distance_mm ({max_dist}) must be > 0.")
                except (TypeError, ValueError):
                    errors.append(f"decaps.max_distance_mm must be numeric, got {max_dist!r}")
            for wref in decaps_cfg.get("within_radius_refs") or []:
                if wref not in (pcb_obj.footprints or {}):
                    errors.append(f"decaps.within_radius_refs names unknown part '{wref}'.")

        keepouts = doc.get("keepouts") or []
        bounds = getattr(pcb_obj.board_info, "board_bounds", None)
        if isinstance(keepouts, list):
            for idx, ko in enumerate(keepouts):
                if not isinstance(ko, dict):
                    continue
                rect = ko.get("rect")
                if rect and isinstance(rect, (list, tuple)) and len(rect) == 4 and bounds:
                    if rect[2] <= rect[0] or rect[3] <= rect[1]:
                        errors.append(f"keepouts[{idx}] has invalid rect {rect}: x1<=x0 or y1<=y0.")
                    elif rect[2] < bounds[0] or rect[0] > bounds[2] or rect[3] < bounds[1] or rect[1] > bounds[3]:
                        warnings.append(f"keepouts[{idx}] rect {rect} lies entirely outside board bounds {list(bounds)}.")

        edge_conns = doc.get("edge_connectors") or []
        locked_set = set(self.locked_refs)
        if isinstance(edge_conns, list):
            for idx, ec in enumerate(edge_conns):
                if not isinstance(ec, dict):
                    continue
                eref = ec.get("ref")
                if eref and eref not in (pcb_obj.footprints or {}):
                    errors.append(f"edge_connectors[{idx}] names unknown ref '{eref}'.")
                elif eref and eref in locked_set and ec.get("edge") and bounds:
                    fp = pcb_obj.footprints[eref]
                    raw_edge = str(ec["edge"]).strip().upper()
                    edge_norm = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}.get(raw_edge, raw_edge)
                    d_map = {
                        "W": abs(fp.x - bounds[0]),
                        "E": abs(bounds[2] - fp.x),
                        "N": abs(fp.y - bounds[1]),
                        "S": abs(bounds[3] - fp.y),
                    }
                    nearest_edge = min(d_map, key=d_map.get)
                    if edge_norm in d_map and nearest_edge != edge_norm:
                        errors.append(
                            f"edge_connector conflict: {eref} is FIXED/locked at ({fp.x:.1f}, {fp.y:.1f}) "
                            f"nearest edge '{nearest_edge}' ({d_map[nearest_edge]:.1f} mm), not '{raw_edge.lower()}'. "
                            "Omit FIXED/locked connectors from edge_connectors or set their true nearest edge."
                        )

        cand_path = self.work_dir / "intent_candidate.json"
        plan_rep_path = self.work_dir / "intent_plan_check.json"
        cand_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        plan_summary: Dict[str, Any] = {}
        plan_rc = None
        if parsed_intent is not None:
            res_plan = _run_krt(
                "py_tools/check_floorplan.py",
                [str(pcb_check_path), "--intent", str(cand_path), "--plan-only", "--json", str(plan_rep_path)],
                timeout_s=60.0,
            )
            plan_rc = res_plan.returncode
            plan_summary = _extract_json_summary(res_plan.stdout) or {}
            if plan_rep_path.exists():
                try:
                    pr = json.loads(plan_rep_path.read_text(encoding="utf-8"))
                    for pf in pr.get("plan_findings") or []:
                        msg = f"{pf.get('rule')}: {pf.get('message')}"
                        if pf.get("severity") == "error":
                            if msg not in errors:
                                errors.append(msg)
                        else:
                            if msg not in warnings:
                                warnings.append(msg)
                except Exception:
                    pass
            elif res_plan.returncode not in (0, 4):
                errors.append(f"check_floorplan --plan-only failed (rc={res_plan.returncode}): {res_plan.stderr.strip()}")

        live_grade: Optional[Dict[str, Any]] = None
        if parsed_intent is not None and not _clean_unrouted_board_is_pile(pcb_obj, pcb_check_path):
            grade_rep_path = self.work_dir / "intent_live_grade.json"
            res_live = _run_krt(
                "py_tools/check_floorplan.py",
                [str(pcb_check_path), "--intent", str(cand_path), "--json", str(grade_rep_path)],
                timeout_s=60.0,
            )
            live_sum = _extract_json_summary(res_live.stdout) or {}
            live_findings: List[Dict[str, Any]] = []
            if grade_rep_path.exists():
                try:
                    gr = json.loads(grade_rep_path.read_text(encoding="utf-8"))
                    live_findings = gr.get("violations") or gr.get("findings") or []
                except Exception:
                    pass
            live_grade = {
                "exit_code": res_live.returncode,
                "summary": live_sum,
                "findings_sample": live_findings[:15],
            }

        valid = len(errors) == 0 and (plan_rc in (0, None))
        if valid:
            self.intent_doc = copy.deepcopy(doc)
            self.intent_path.write_text(json.dumps(self.intent_doc, indent=2), encoding="utf-8")
            self.intent_validated = True

        return {
            "ok": True,
            "valid": valid,
            "kind": doc.get("kind"),
            "normalized_fields": normalized_fields,
            "errors": errors,
            "warnings": warnings[:25],
            "plan_check_summary": plan_summary,
            "live_grade_on_placed_board": live_grade,
            "decaps_check": {
                "max_distance_mm": decaps_cfg.get("max_distance_mm") if isinstance(decaps_cfg, dict) else None,
                "max_pin_distance_mm": decaps_cfg.get("max_pin_distance_mm") if isinstance(decaps_cfg, dict) else None,
                "within_radius_refs_count": len(decaps_cfg.get("within_radius_refs") or []) if isinstance(decaps_cfg, dict) else 0,
            },
            "keepouts_count": len(keepouts) if isinstance(keepouts, list) else 0,
            "edge_connectors_count": len(edge_conns) if isinstance(edge_conns, list) else 0,
            "saved_path": str(self.intent_path) if valid else None,
        }

    # -----------------------------------------------------------------------
    # Tool 5: run_placement
    # -----------------------------------------------------------------------

    def run_placement(
        self,
        scattered: int = 2,
        clustered: int = 2,
        anchored: int = 2,
        jobs: int = 2,
        judge_count: int = 2,
        repair_decaps: bool = True,
        halo_coef: float = 0.45,
        halo_weight: float = 20.0,
        congestion_weight: float = 12.0,
        pad_weight: float = 6.0,
        track_pitch: float = 0.4,
        moves: int = 0,
    ) -> Dict[str, Any]:
        if not self.groups_path.exists():
            return {
                "ok": False,
                "error": "No validated groups.json exists yet. Call `validate_groups(groups_json)` first.",
            }

        used_starter_intent = False
        intent_to_use = self.intent_path
        if not intent_to_use.exists():
            shutil.copyfile(self.starter_intent_path, self.intent_path)
            self.intent_doc = json.loads(self.intent_path.read_text(encoding="utf-8"))
            intent_to_use = self.intent_path
            used_starter_intent = True

        self._save_snapshot("run_placement")
        self.place_run_counter += 1
        place_dir = self.work_dir / f"place_run_{self.place_run_counter:02d}"
        place_dir.mkdir(parents=True, exist_ok=True)

        total_seeds = max(1, int(scattered) + int(clustered) + int(anchored))
        judge_n = max(1, min(int(judge_count), total_seeds))

        args = [
            str(self.fixed_pcb),
            str(place_dir / "s{seed}.kicad_pcb"),
            "--groups",
            str(self.groups_path),
            "--intent",
            str(intent_to_use),
            "--scattered",
            str(int(scattered)),
            "--clustered",
            str(int(clustered)),
            "--anchored",
            str(int(anchored)),
            "--judge",
            str(judge_n),
            "--jobs",
            str(max(1, int(jobs))),
            "--halo-coef",
            str(float(halo_coef)),
            "--halo-weight",
            str(float(halo_weight)),
            "--congestion-weight",
            str(float(congestion_weight)),
            "--pad-weight",
            str(float(pad_weight)),
            "--track-pitch",
            str(float(track_pitch)),
            "--json",
            str(place_dir / "s{seed}.json"),
        ]
        if int(moves) > 0:
            args.extend(["--moves", str(int(moves))])
        if repair_decaps:
            args.append("--repair-decaps")

        t0 = time.time()
        res = _run_krt("py_placer/place_simple.py", args, timeout_s=max(self.timeout_s, 360.0))
        elapsed = round(time.time() - t0, 2)

        judge_path = place_dir / "judge.json"
        judge_doc: Dict[str, Any] = {}
        if judge_path.exists():
            judge_doc = json.loads(judge_path.read_text(encoding="utf-8"))
            self.last_judge_json = judge_path

        seeds_summary: List[Dict[str, Any]] = []
        self.available_seeds = {}
        for jf in sorted(place_dir.glob("s*.json")):
            try:
                rep = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            seed_id = rep.get("seed")
            if seed_id is None:
                continue
            seed_pcb = place_dir / f"s{seed_id}.kicad_pcb"
            if seed_pcb.exists():
                self.available_seeds[int(seed_id)] = seed_pcb
            final = rep.get("final") or {}
            fp_info = rep.get("floorplan") or {}
            asm_info = rep.get("assembly") or {}
            seeds_summary.append(
                {
                    "seed": int(seed_id),
                    "start": rep.get("start"),
                    "cost": round(float(final.get("cost", 0.0)), 2),
                    "wire_mm": round(float(final.get("wire", 0.0)), 2),
                    "crossings": int(final.get("crossings", 0)),
                    "floorplan_errors": fp_info.get("errors"),
                    "floorplan_rules": fp_info.get("rules") or [],
                    "assembly_blocking": asm_info.get("blocking"),
                    "assembly_buildable": asm_info.get("buildable"),
                    "illegal_parts": rep.get("illegal") or [],
                    "decap_residual": rep.get("decap_residual") or [],
                    "seconds": rep.get("seconds"),
                }
            )

        seeds_summary.sort(
            key=lambda s: (
                1 if s["illegal_parts"] else 0,
                (s["floorplan_errors"] or 0) + (s["assembly_blocking"] or 0),
                s["cost"],
                s["seed"],
            )
        )

        judged_list = [int(s) for s in (judge_doc.get("judge") or [])]
        chosen_seed: Optional[int] = judged_list[0] if judged_list else None
        if chosen_seed is None and seeds_summary:
            chosen_seed = seeds_summary[0]["seed"]

        if chosen_seed is None or chosen_seed not in self.available_seeds:
            return {
                "ok": False,
                "returncode": res.returncode,
                "elapsed_sec": elapsed,
                "error": "place_simple produced no usable placement seed.",
                "stdout_tail": (res.stdout or "")[-1500:],
                "stderr_tail": (res.stderr or "")[-1000:],
            }

        self.best_seed = chosen_seed
        best_step = self._next_step_pcb(f"placed_s{chosen_seed}")
        copy_board(str(self.available_seeds[chosen_seed]), str(best_step))
        self.placed_pcb = best_step
        self.current_pcb = best_step

        best_meta = next((s for s in seeds_summary if s["seed"] == chosen_seed), {})
        return {
            "ok": True,
            "returncode": res.returncode,
            "elapsed_sec": elapsed,
            "used_starter_intent_fallback": used_starter_intent,
            "best_seed": chosen_seed,
            "best_pcb": str(best_step),
            "judged_seeds": judged_list,
            "illegal_seeds": judge_doc.get("illegal") or [],
            "best_cost": best_meta.get("cost"),
            "ratsnest_wirelength_mm": best_meta.get("wire_mm"),
            "airwire_crossings": best_meta.get("crossings"),
            "floorplan_errors": best_meta.get("floorplan_errors"),
            "floorplan_rules": best_meta.get("floorplan_rules"),
            "assembly_status": {
                "buildable": best_meta.get("assembly_buildable"),
                "blocking": best_meta.get("assembly_blocking"),
            },
            "seeds_summary": seeds_summary,
            "refused_poses": judge_doc.get("refused") or {},
            "oversubscribed_edges": judge_doc.get("oversubscribed") or [],
        }

    def select_seed(self, seed: int) -> Dict[str, Any]:
        """Switch active placement to another seed produced by the latest `run_placement`."""
        s = int(seed)
        if s not in self.available_seeds:
            return {"ok": False, "error": f"Seed {s} not available. Known: {sorted(self.available_seeds)}"}
        self._save_snapshot(f"select_seed_{s}")
        self.best_seed = s
        dst = self._next_step_pcb(f"placed_s{s}")
        copy_board(str(self.available_seeds[s]), str(dst))
        self.placed_pcb = dst
        self.current_pcb = dst
        return {"ok": True, "best_seed": s, "current_pcb": str(dst)}

    # -----------------------------------------------------------------------
    # Tool 6a: route_nets
    # -----------------------------------------------------------------------

    def route_nets(
        self,
        nets: Union[List[str], str],
        layers: Optional[List[str]] = None,
        track_width: Optional[float] = None,
        clearance: Optional[float] = None,
        clearance_ceiling: Optional[float] = None,
        via_size: Optional[float] = None,
        via_drill: Optional[float] = None,
        via_cost: Optional[int] = None,
        turn_cost: Optional[int] = None,
        direction_preference_cost: Optional[int] = None,
        same_net_pad_clearance: Optional[float] = None,
        bus: bool = False,
        ordering: Optional[str] = None,
        plane_layers: Optional[List[str]] = None,
        layer_costs: Optional[List[float]] = None,
        keepout: bool = False,
        keepout_layer: Optional[str] = None,
        guide_corridor: bool = False,
        guide_corridor_layer: Optional[str] = None,
        keep_away: Optional[List[str]] = None,
        power_nets: Optional[List[str]] = None,
        power_nets_widths: Optional[List[float]] = None,
        max_ripup: Optional[int] = None,
        rip_existing_nets: Optional[List[str]] = None,
        force_reroute: bool = False,
        keep_input_copper: bool = True,
        quick: Optional[int] = None,
        write_fill: bool = False,
        add_teardrops: bool = False,
        stitch_vias: bool = False,
        add_gnd_vias: bool = False,
        zone_clearance: Optional[float] = None,
        fab_tier: Optional[str] = None,
        escalation: Optional[str] = None,
    ) -> Dict[str, Any]:
        net_list = [nets] if isinstance(nets, str) else [str(n) for n in (nets or [])]
        if not net_list:
            return {"ok": False, "error": "`nets` must be a non-empty list of net names/patterns."}

        pcb_before, _ = self._get_pcb_and_view(self.current_pcb)
        had_copper_before = _board_has_copper(pcb_before)
        counts_before = _copper_counts_by_net(pcb_before)

        self._save_snapshot(f"route_nets({','.join(net_list[:4])})")
        out_pcb = self._next_step_pcb("route_planes" if plane_layers else "route")

        if plane_layers:
            if len(plane_layers) != len(net_list):
                self.snapshots.pop()
                return {
                    "ok": False,
                    "error": f"plane_layers length ({len(plane_layers)}) must match nets length ({len(net_list)}).",
                }
            args: List[str] = [str(self.current_pcb), str(out_pcb), "--nets", *net_list, "--plane-layers", *plane_layers]
            if layers:
                args.extend(["--layers", *layers])
            if layer_costs:
                args.extend(["--layer-costs", *[str(x) for x in layer_costs]])
            if track_width is not None:
                args.extend(["--track-width", str(float(track_width))])
            if clearance is not None:
                args.extend(["--clearance", str(float(clearance))])
            if clearance_ceiling is not None:
                args.extend(["--clearance-ceiling", str(float(clearance_ceiling))])
            if zone_clearance is not None:
                args.extend(["--zone-clearance", str(float(zone_clearance))])
            if via_size is not None:
                args.extend(["--via-size", str(float(via_size))])
            if via_drill is not None:
                args.extend(["--via-drill", str(float(via_drill))])
            if same_net_pad_clearance is not None:
                args.extend(["--same-net-pad-clearance", str(float(same_net_pad_clearance))])
            if stitch_vias:
                args.append("--stitch-vias")
            if add_gnd_vias:
                args.append("--add-gnd-vias")
            if add_teardrops:
                args.append("--add-teardrops")
            if fab_tier:
                args.extend(["--fab-tier", str(fab_tier)])
            if escalation:
                args.extend(["--escalation", str(escalation)])

            t0 = time.time()
            res = _run_krt("py_router/route_planes.py", args, timeout_s=self.timeout_s)
            elapsed = round(time.time() - t0, 2)
            route_json_data: Dict[str, Any] = _extract_json_summary(res.stdout) or {}
            script_used = "py_router/route_planes.py"
        else:
            route_json_path = out_pcb.with_suffix(".route.json")
            args = [str(self.current_pcb), str(out_pcb), "--nets", *net_list, "--json-out", str(route_json_path)]
            if layers:
                args.extend(["--layers", *layers])
            if layer_costs:
                args.extend(["--layer-costs", *[str(x) for x in layer_costs]])
            if track_width is not None:
                args.extend(["--track-width", str(float(track_width))])
            if clearance is not None:
                args.extend(["--clearance", str(float(clearance))])
            if clearance_ceiling is not None:
                args.extend(["--clearance-ceiling", str(float(clearance_ceiling))])
            if via_size is not None:
                args.extend(["--via-size", str(float(via_size))])
            if via_drill is not None:
                args.extend(["--via-drill", str(float(via_drill))])
            if via_cost is not None:
                args.extend(["--via-cost", str(int(via_cost))])
            if turn_cost is not None:
                args.extend(["--turn-cost", str(int(turn_cost))])
            if direction_preference_cost is not None:
                args.extend(["--direction-preference-cost", str(int(direction_preference_cost))])
            if same_net_pad_clearance is not None:
                args.extend(["--same-net-pad-clearance", str(float(same_net_pad_clearance))])
            if ordering:
                args.extend(["--ordering", str(ordering)])
            if bus:
                args.append("--bus")
            if keepout:
                args.append("--keepout")
            if keepout_layer:
                args.extend(["--keepout-layer", str(keepout_layer)])
            if guide_corridor:
                args.append("--guide-corridor")
            if guide_corridor_layer:
                args.extend(["--guide-corridor-layer", str(guide_corridor_layer)])
            if keep_away:
                args.extend(["--keep-away", *keep_away])
            if power_nets and power_nets_widths and len(power_nets) == len(power_nets_widths):
                args.extend(["--power-nets", *power_nets, "--power-nets-widths", *[str(w) for w in power_nets_widths]])
            if max_ripup is not None:
                args.extend(["--max-ripup", str(int(max_ripup))])
            if rip_existing_nets:
                args.extend(["--rip-existing-nets", *rip_existing_nets])
            if force_reroute:
                args.append("--force-reroute")
            if keep_input_copper and had_copper_before:
                args.append("--keep-input-copper")
            if quick in (1, 2):
                args.extend(["--quick", str(int(quick))])
            if write_fill:
                args.append("--write-fill")
            if add_teardrops:
                args.append("--add-teardrops")
            if fab_tier:
                args.extend(["--fab-tier", str(fab_tier)])
            if escalation:
                args.extend(["--escalation", str(escalation)])

            # Prevent collateral rip-ups of earlier routed nets unless explicitly requested.
            extra_env = {"KICAD_RIP_PREEXISTING": "0"} if (had_copper_before and not rip_existing_nets) else None

            t0 = time.time()
            res = _run_krt("py_router/route.py", args, timeout_s=self.timeout_s, extra_env=extra_env)
            elapsed = round(time.time() - t0, 2)
            route_json_data = {}
            if route_json_path.exists():
                try:
                    route_json_data = json.loads(route_json_path.read_text(encoding="utf-8"))
                except Exception:
                    route_json_data = {}
            if not route_json_data:
                route_json_data = _extract_json_summary(res.stdout) or {}
            script_used = "py_router/route.py"

        if not out_pcb.exists():
            self.snapshots.pop()
            return {
                "ok": False,
                "script": script_used,
                "returncode": res.returncode,
                "elapsed_sec": elapsed,
                "error": f"{script_used} failed to write output PCB.",
                "stdout_tail": (res.stdout or "")[-1500:],
                "stderr_tail": (res.stderr or "")[-1000:],
            }

        self.current_pcb = out_pcb
        pcb_after, _ = self._get_pcb_and_view(out_pcb)
        counts_after = _copper_counts_by_net(pcb_after)

        # Verify which pre-existing nets were untouched vs modified
        preserved_nets = []
        changed_nets = []
        for nm, prev in sorted(counts_before.items()):
            cur = counts_after.get(nm, {"segments": 0, "vias": 0, "zones": 0})
            if cur == prev:
                preserved_nets.append(nm)
            else:
                changed_nets.append({"net": nm, "before": prev, "after": cur})

        newly_routed = sorted(nm for nm in counts_after if nm not in counts_before)
        return {
            "ok": res.returncode == 0,
            "script": script_used,
            "returncode": res.returncode,
            "elapsed_sec": elapsed,
            "current_pcb": str(out_pcb),
            "requested_nets": net_list,
            "newly_routed_nets": newly_routed[:50],
            "newly_routed_count": len(newly_routed),
            "preserved_preexisting_nets_count": len(preserved_nets),
            "modified_preexisting_nets": changed_nets[:20],
            "route_summary": route_json_data,
            "stdout_tail": (res.stdout or "")[-1000:] if res.returncode != 0 else "",
            "stderr_tail": (res.stderr or "")[-600:] if res.returncode != 0 else "",
        }

    # -----------------------------------------------------------------------
    # Tool 6b: run_routing_chain
    # -----------------------------------------------------------------------

    def _build_chain_from_strategy(self, strat: Dict[str, Any]) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Convert a strategy_json into safe KRT step commands (script, args)."""
        if self.model_first and strat.get("via_and_trace_rules"):
            pcb, view = self._get_pcb_and_view(self.current_pcb)
            layers = list(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"])
            vtr = strat.get("via_and_trace_rules") or {}
            clr = float(vtr.get("clearance_mm", 0.12))
            tw = float(vtr.get("signal_track_width_mm", 0.15))
            vs = float(vtr.get("via_size_mm", 0.45))
            vd = float(vtr.get("via_drill_mm", 0.2))
            spc = float(vtr.get("same_net_pad_clearance_mm", 0.15))
            vc = int(vtr.get("via_cost", 110))
            tc = int(vtr.get("turn_cost", 1200))
            dpc = int(vtr.get("direction_preference_cost", 250))

            steps: List[List[str]] = []
            if strat.get("mounting_hole_keepouts"):
                steps.append(["py_router/add_rule_area.py", "{in}", "{out}", "--name", "MOUNT", "--mounting-holes"])

            gnd_nets = [n for n in (strat.get("ground_plane_nets") or []) if isinstance(n, str)]
            plane_layers = [l for l in (strat.get("plane_layers") or []) if l in layers]
            if gnd_nets:
                g_spec = "|".join(gnd_nets[:3])
                if not plane_layers:
                    plane_layers = [layers[1], layers[2]] if len(layers) >= 4 else ["B.Cu", "F.Cu"]
                p_nets_arg = [g_spec] * len(plane_layers)
                steps.append(
                    [
                        "py_router/route_planes.py",
                        "{in}",
                        "{out}",
                        "--nets",
                        *p_nets_arg,
                        "--plane-layers",
                        *plane_layers,
                        "--clearance",
                        str(clr),
                        "--via-size",
                        str(vs),
                        "--via-drill",
                        str(vd),
                        "--same-net-pad-clearance",
                        str(spc),
                    ]
                )

            dp_nets = [n for n in (strat.get("diff_pairs") or []) if isinstance(n, str)]
            if dp_nets:
                steps.append(
                    [
                        "py_router/route_diff.py",
                        "{in}",
                        "{out}",
                        "--nets",
                        *dp_nets,
                        "--track-width",
                        str(tw),
                        "--via-size",
                        str(vs),
                        "--via-drill",
                        str(vd),
                    ]
                )

            r_cmd = [
                "py_router/route.py",
                "{in}",
                "{out}",
                "--nets",
                "*",
                "--clearance",
                str(clr),
                "--track-width",
                str(tw),
                "--via-size",
                str(vs),
                "--via-drill",
                str(vd),
                "--same-net-pad-clearance",
                str(spc),
                "--via-cost",
                str(vc),
                "--turn-cost",
                str(tc),
                "--direction-preference-cost",
                str(dpc),
            ]
            bus_cfg = strat.get("bus_routing") or {}
            if bus_cfg.get("enabled", True):
                r_cmd.extend(["--ordering", "bus", "--bus", "--bus-min-nets", str(int(bus_cfg.get("bus_min_nets", 3)))])
            pw_tiers = strat.get("power_width_tiers") or {}
            all_board_nets = {view.name_of(nid) for nid in (pcb.nets or {}) if view.name_of(nid)}
            p_nets = [k for k in pw_tiers if k in all_board_nets]
            if p_nets:
                r_cmd.extend(["--power-nets", *p_nets, "--power-nets-widths", *[str(pw_tiers[k]) for k in p_nets]])
            ka_rules = strat.get("keep_away_rules") or []
            if ka_rules:
                r_cmd.extend(["--keep-away", *[str(x) for x in ka_rules]])
            r_cmd.extend(["--write-fill", "--json-out", "{dir}/route.json"])
            steps.append(r_cmd)
            return steps, strat

        _jc_txt, prod_steps = build_smart_routing_chains(self.current_pcb, routing_strategy=strat)
        built: List[List[str]] = []
        for script_rel, s_args in prod_steps:
            toks = [script_rel, "{in}", "{out}", *s_args]
            if script_rel == "py_router/route.py" and "--json-out" not in toks:
                toks.extend(["--json-out", "{dir}/route.json"])
            built.append(toks)
        return built, strat

    def _parse_chain_command_tokens(self, cmd_line: str) -> List[str]:
        tokens = shlex.split(cmd_line.strip())
        # Strip leading `python3 [-X utf8]` if caller provided full CLI line
        idx = 0
        if idx < len(tokens) and Path(tokens[idx]).name.startswith("python"):
            idx += 1
            while idx + 1 < len(tokens) and tokens[idx] == "-X":
                idx += 2
        tokens = tokens[idx:]
        if not tokens:
            raise ValueError(f"Empty chain command: {cmd_line!r}")
        script = tokens[0]
        if script not in _ALLOWED_CHAIN_SCRIPTS:
            raise ValueError(
                f"Disallowed chain script '{script}'. Allowed: {sorted(_ALLOWED_CHAIN_SCRIPTS)}"
            )
        return tokens

    def run_routing_chain(
        self,
        strategy_json: Optional[Union[Dict[str, Any], str, Path]] = None,
        commands: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if strategy_json is None and not commands:
            strat = synthesize_routing_strategy(self.current_pcb)
        elif strategy_json is not None:
            strat = _load_json_arg(strategy_json, "strategy_json")
        else:
            strat = self.routing_strategy_doc or {}

        parsed_steps: List[List[str]] = []
        if commands:
            for line in commands:
                s = str(line).strip()
                if s and not s.startswith("#"):
                    parsed_steps.append(self._parse_chain_command_tokens(s))
        else:
            parsed_steps, strat = self._build_chain_from_strategy(strat)

        if not parsed_steps:
            return {"ok": False, "error": "No routing chain commands to execute."}

        self._save_snapshot("run_routing_chain")
        if strat:
            self.routing_strategy_doc = copy.deepcopy(strat)
            self.routing_strategy_path.write_text(json.dumps(strat, indent=2), encoding="utf-8")

        chain_dir = self.work_dir / f"chain_step_{self.step_counter + 1:03d}"
        chain_dir.mkdir(parents=True, exist_ok=True)

        step_results = []
        cur_in = self.current_pcb
        t_all = time.time()

        for i, toks in enumerate(parsed_steps):
            script_rel = toks[0]
            step_out = self._next_step_pcb(f"chain_{i}_{Path(script_rel).stem}")
            rendered_args = [
                t.replace("{in}", str(cur_in))
                .replace("{out}", str(step_out))
                .replace("{dir}", str(chain_dir))
                for t in toks[1:]
            ]
            t0 = time.time()
            res = _run_krt(script_rel, rendered_args, timeout_s=self.timeout_s)
            dt = round(time.time() - t0, 2)
            step_ok = step_out.exists()
            step_results.append(
                {
                    "step": i + 1,
                    "script": script_rel,
                    "returncode": res.returncode,
                    "elapsed_sec": dt,
                    "output_exists": step_ok,
                    "summary": _extract_json_summary(res.stdout),
                }
            )
            if not step_ok:
                return {
                    "ok": False,
                    "error": f"Chain step {i + 1} ({script_rel}) failed to write {step_out}",
                    "steps": step_results,
                    "stdout_tail": (res.stdout or "")[-1200:],
                    "stderr_tail": (res.stderr or "")[-800:],
                }
            cur_in = step_out

        self.current_pcb = cur_in
        route_json_file = chain_dir / "route.json"
        route_summary = {}
        if route_json_file.exists():
            try:
                route_summary = json.loads(route_json_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "ok": all(s["returncode"] == 0 for s in step_results),
            "elapsed_sec": round(time.time() - t_all, 2),
            "current_pcb": str(self.current_pcb),
            "steps": step_results,
            "route_summary": route_summary,
        }

    # -----------------------------------------------------------------------
    # Tool 7: inspect_routing_failures
    # -----------------------------------------------------------------------

    def inspect_routing_failures(self) -> Dict[str, Any]:
        from board_score import score_connectivity
        from quick_score import quick_score
        from route_review import review as route_review_fn

        pcb, view = self._get_pcb_and_view(self.current_pcb)
        intent_arg = str(self.intent_path) if self.intent_path.exists() else None
        baseline_arg = str(self.fixed_pcb) if self.fixed_pcb.exists() else None

        qscore = quick_score(str(self.current_pcb), intent=intent_arg, baseline=baseline_arg)
        conn = score_connectivity(str(KRT_ROOT), str(self.current_pcb))

        failing_names: List[str] = list((qscore.get("blocking_by") or {}).get("nets") or [])
        poured_names: List[str] = list((qscore.get("poured") or {}).get("nets") or [])
        all_open_names = list(dict.fromkeys(failing_names + poured_names))

        name_to_nid = {net.name: nid for nid, net in (pcb.nets or {}).items() if net.name}
        copper_counts = _copper_counts_by_net(pcb)
        broken_detail = conn.get("broken_detail") or {}

        failing_nets_detail = []
        for nm in all_open_names[:40]:
            nid = name_to_nid.get(nm)
            pads = list((pcb.pads_by_net or {}).get(nid, ())) if nid is not None else []
            if pads:
                xs = [float(p.global_x) for p in pads]
                ys = [float(p.global_y) for p in pads]
                bbox = [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]
                span_mm = round(math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]), 2)
                pad_coords = [
                    {
                        "ref": p.component_ref,
                        "pad": str(p.pad_number),
                        "x": round(float(p.global_x), 3),
                        "y": round(float(p.global_y), 3),
                        "layers": list(getattr(p, "layers", []) or []),
                    }
                    for p in pads[:24]
                ]
            else:
                bbox = None
                span_mm = 0.0
                pad_coords = []

            c = copper_counts.get(nm, {"segments": 0, "vias": 0, "zones": 0})
            bd = broken_detail.get(nm) or {}
            failing_nets_detail.append(
                {
                    "net": nm,
                    "is_poured": nm in poured_names and nm not in failing_names,
                    "pad_count": len(pads),
                    "pads_truncated": max(0, len(pads) - len(pad_coords)),
                    "bbox_mm": bbox,
                    "span_mm": span_mm,
                    "pads": pad_coords,
                    "routed_segments": c["segments"],
                    "routed_vias": c["vias"],
                    "zones": c["zones"],
                    "disconnected_components": bd.get("components"),
                    "joins_needed": bd.get("joins_needed"),
                    "stranded_pads": bd.get("stranded_pads") or [],
                }
            )

        rev = route_review_fn(str(self.current_pcb), pcb=pcb)
        drc_path = self.steps_dir / f"{self.current_pcb.stem}.drc.json"
        drc = _run_kicad_drc_fresh(self.current_pcb, drc_path, refill_zones=True)
        phys = evaluate_physical_board_quality(self.current_pcb)

        # Actionable Teel sub-metrics (non-zero issues highlighted cleanly alongside full dict)
        actionable_teel = {
            k: v
            for k, v in phys.items()
            if k not in ("teel_score_100", "physical_quality_penalty")
            and (
                (isinstance(v, (int, float)) and v > 0 and k not in ("power_width_compliance_pct", "power_to_analog_clearance_mm", "max_detour_ratio"))
                or (k == "power_width_compliance_pct" and isinstance(v, (int, float)) and v < 95.0)
                or (k == "max_detour_ratio" and isinstance(v, (int, float)) and v > 2.0)
                or (k == "weaving_nets" and bool(v))
            )
        }

        recs: List[str] = []
        fp_cnt = (qscore.get("blocking_by") or {}).get("floorplan") or 0
        asm_cnt = (qscore.get("blocking_by") or {}).get("assembly") or 0
        if fp_cnt or asm_cnt:
            recs.append(
                f"Placement defects remain (floorplan={fp_cnt}, assembly={asm_cnt}); "
                "inspect `validate_floorplan` / `patch_layout` before finalizing routing."
            )
        if failing_names:
            recs.append(
                f"Reroute failing nets {failing_names[:6]} via `ripup_and_reroute` with "
                "larger `max_ripup` or adjusted `clearance`/`track_width`/`rip_existing_nets`."
            )
        if rev.get("turned_parts"):
            refs = [t["ref"] for t in rev["turned_parts"][:5]]
            recs.append(f"Turned 2-pad parts detected ({refs}); rotate 180 deg via `patch_layout`.")
        if drc.get("ran") and (drc.get("drc_errors") or 0) > 0:
            recs.append(
                f"KiCad DRC reported {drc['drc_errors']} error(s) ({drc.get('errors_by_type')}); "
                "rip up & reroute offending nets."
            )

        return {
            "ok": True,
            "current_pcb": str(self.current_pcb),
            "blocking": qscore.get("blocking"),
            "blocking_by": qscore.get("blocking_by"),
            "joins_needed": qscore.get("joins_needed"),
            "poured": qscore.get("poured"),
            "quality": qscore.get("quality"),
            "failing_nets_detail": failing_nets_detail,
            "kicad_drc": drc,
            "route_review": {
                "counts": rev.get("counts"),
                "most_vias": (rev.get("most_vias") or [])[:6],
                "longest_detours": (rev.get("longest_detours") or [])[:6],
                "congested_areas": (rev.get("congested_areas") or [])[:5],
                "turned_parts": (rev.get("turned_parts") or [])[:6],
            },
            "teel_score_100": phys.get("teel_score_100"),
            "physical_quality_penalty": phys.get("physical_quality_penalty"),
            "actionable_teel_submetrics": actionable_teel,
            "teel_metrics": phys,
            "recommendations": recs,
        }

    # -----------------------------------------------------------------------
    # Tool 8: ripup_and_reroute
    # -----------------------------------------------------------------------

    def ripup_and_reroute(
        self,
        nets: Union[List[str], str],
        new_params: Optional[Dict[str, Any]] = None,
        reroute: bool = True,
    ) -> Dict[str, Any]:
        net_list = [nets] if isinstance(nets, str) else [str(n) for n in (nets or [])]
        if not net_list:
            return {"ok": False, "error": "`nets` must be a non-empty list of net names or patterns."}

        pcb_before, _ = self._get_pcb_and_view(self.current_pcb)
        all_net_names = [n.name for n in (pcb_before.nets or {}).values() if n.name]
        matched_nets = sorted(
            {nm for nm in all_net_names if any(fnmatch.fnmatchcase(nm, pat) for pat in net_list)}
        )
        if not matched_nets:
            return {"ok": False, "error": f"No nets matched patterns {net_list} on {self.current_pcb.name}."}

        self._save_snapshot(f"ripup_and_reroute({','.join(matched_nets[:4])})")
        counts_before = _copper_counts_by_net(pcb_before)

        # Phase A: Cleanly strip ONLY target nets' segments, arcs, and vias via route.py --undo
        undo_pcb = self._next_step_pcb("ripup_undo")
        res_undo = _run_krt(
            "py_router/route.py",
            [str(self.current_pcb), str(undo_pcb), "--undo", "--nets", *matched_nets],
            timeout_s=90.0,
        )
        if res_undo.returncode != 0 or not undo_pcb.exists():
            self.snapshots.pop()
            return {
                "ok": False,
                "phase": "undo",
                "returncode": res_undo.returncode,
                "error": f"Failed to strip nets {matched_nets}",
                "stdout_tail": (res_undo.stdout or "")[-1000:],
                "stderr_tail": (res_undo.stderr or "")[-800:],
            }

        pcb_after_undo, _ = self._get_pcb_and_view(undo_pcb)
        counts_after_undo = _copper_counts_by_net(pcb_after_undo)

        matched_set = set(matched_nets)
        unintended_undo_diffs = []
        removed_summary: Dict[str, Dict[str, int]] = {}
        total_removed_segs = 0
        total_removed_vias = 0

        for nm, prev in sorted(counts_before.items()):
            cur = counts_after_undo.get(nm, {"segments": 0, "vias": 0, "zones": 0})
            if nm in matched_set:
                ds = prev["segments"] - cur["segments"]
                dv = prev["vias"] - cur["vias"]
                total_removed_segs += max(0, ds)
                total_removed_vias += max(0, dv)
                removed_summary[nm] = {
                    "segments_removed": ds,
                    "vias_removed": dv,
                    "zones_left": cur["zones"],
                }
            elif cur != prev:
                unintended_undo_diffs.append({"net": nm, "before": prev, "after": cur})

        preserved_other_undo = len(unintended_undo_diffs) == 0

        if not reroute:
            self.current_pcb = undo_pcb
            return {
                "ok": True,
                "rerouted": False,
                "ripped_nets": matched_nets,
                "removed_segments": total_removed_segs,
                "removed_vias": total_removed_vias,
                "removed_by_net": removed_summary,
                "preserved_other_nets_after_undo": preserved_other_undo,
                "unintended_undo_diffs": unintended_undo_diffs,
                "current_pcb": str(undo_pcb),
            }

        # Phase B: Re-route ONLY the ripped nets with updated parameters
        params = dict(new_params or {})
        rerouted_pcb = self._next_step_pcb("reroute")
        route_json_path = rerouted_pcb.with_suffix(".route.json")

        r_args: List[str] = [str(undo_pcb), str(rerouted_pcb), "--nets", *matched_nets, "--json-out", str(route_json_path)]
        if params.get("layers"):
            r_args.extend(["--layers", *[str(x) for x in params["layers"]]])
        if params.get("layer_costs"):
            r_args.extend(["--layer-costs", *[str(x) for x in params["layer_costs"]]])
        if params.get("track_width") is not None:
            r_args.extend(["--track-width", str(float(params["track_width"]))])
        if params.get("clearance") is not None:
            r_args.extend(["--clearance", str(float(params["clearance"]))])
        if params.get("clearance_ceiling") is not None:
            r_args.extend(["--clearance-ceiling", str(float(params["clearance_ceiling"]))])
        if params.get("via_size") is not None:
            r_args.extend(["--via-size", str(float(params["via_size"]))])
        if params.get("via_drill") is not None:
            r_args.extend(["--via-drill", str(float(params["via_drill"]))])
        if params.get("via_cost") is not None:
            r_args.extend(["--via-cost", str(int(params["via_cost"]))])
        if params.get("turn_cost") is not None:
            r_args.extend(["--turn-cost", str(int(params["turn_cost"]))])
        if params.get("direction_preference_cost") is not None:
            r_args.extend(["--direction-preference-cost", str(int(params["direction_preference_cost"]))])
        if params.get("same_net_pad_clearance") is not None:
            r_args.extend(["--same-net-pad-clearance", str(float(params["same_net_pad_clearance"]))])
        if params.get("ordering"):
            r_args.extend(["--ordering", str(params["ordering"])])
        if params.get("bus"):
            r_args.append("--bus")
        if params.get("max_ripup") is not None:
            r_args.extend(["--max-ripup", str(int(params["max_ripup"]))])
        if params.get("rip_existing_nets"):
            r_args.extend(["--rip-existing-nets", *[str(x) for x in params["rip_existing_nets"]]])
        if params.get("keep_input_copper", True):
            r_args.append("--keep-input-copper")
        if params.get("quick") in (1, 2):
            r_args.extend(["--quick", str(int(params["quick"]))])
        if params.get("write_fill"):
            r_args.append("--write-fill")
        if params.get("add_teardrops"):
            r_args.append("--add-teardrops")
        if params.get("power_nets") and params.get("power_nets_widths"):
            r_args.extend(
                [
                    "--power-nets",
                    *[str(x) for x in params["power_nets"]],
                    "--power-nets-widths",
                    *[str(w) for w in params["power_nets_widths"]],
                ]
            )
        if params.get("keep_away"):
            r_args.extend(["--keep-away", *[str(x) for x in params["keep_away"]]])
        if params.get("fab_tier"):
            r_args.extend(["--fab-tier", str(params["fab_tier"])])
        if params.get("escalation"):
            r_args.extend(["--escalation", str(params["escalation"])])

        extra_env = None if params.get("rip_existing_nets") else {"KICAD_RIP_PREEXISTING": "0"}
        t0 = time.time()
        res_rr = _run_krt("py_router/route.py", r_args, timeout_s=self.timeout_s, extra_env=extra_env)
        elapsed = round(time.time() - t0, 2)

        if not rerouted_pcb.exists():
            self.current_pcb = undo_pcb
            return {
                "ok": False,
                "phase": "reroute",
                "returncode": res_rr.returncode,
                "elapsed_sec": elapsed,
                "ripped_nets": matched_nets,
                "removed_segments": total_removed_segs,
                "removed_vias": total_removed_vias,
                "preserved_other_nets_after_undo": preserved_other_undo,
                "current_pcb": str(undo_pcb),
                "error": "Re-route failed to produce output PCB (ripped state kept; call `undo_last_patch` to revert).",
                "stdout_tail": (res_rr.stdout or "")[-1000:],
                "stderr_tail": (res_rr.stderr or "")[-800:],
            }

        self.current_pcb = rerouted_pcb
        pcb_after_rr, _ = self._get_pcb_and_view(rerouted_pcb)
        counts_after_rr = _copper_counts_by_net(pcb_after_rr)

        unintended_rr_diffs = []
        for nm, prev in sorted(counts_after_undo.items()):
            if nm in matched_set:
                continue
            cur = counts_after_rr.get(nm, {"segments": 0, "vias": 0, "zones": 0})
            if cur != prev:
                unintended_rr_diffs.append({"net": nm, "before": prev, "after": cur})

        route_summary = {}
        if route_json_path.exists():
            try:
                route_summary = json.loads(route_json_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not route_summary:
            route_summary = _extract_json_summary(res_rr.stdout) or {}

        return {
            "ok": res_rr.returncode == 0,
            "rerouted": True,
            "returncode": res_rr.returncode,
            "elapsed_sec": elapsed,
            "ripped_nets": matched_nets,
            "removed_segments": total_removed_segs,
            "removed_vias": total_removed_vias,
            "removed_by_net": removed_summary,
            "preserved_other_nets_after_undo": preserved_other_undo,
            "preserved_other_nets_after_reroute": len(unintended_rr_diffs) == 0,
            "unintended_reroute_diffs": unintended_rr_diffs[:15],
            "rerouted_copper_counts": {nm: counts_after_rr.get(nm, {"segments": 0, "vias": 0, "zones": 0}) for nm in matched_nets},
            "route_summary": route_summary,
            "current_pcb": str(rerouted_pcb),
        }

    # -----------------------------------------------------------------------
    # Tool 9: patch_layout
    # -----------------------------------------------------------------------

    def _apply_group_patch(self, group_edits: Dict[str, Any]) -> Dict[str, Any]:
        if "groups" in group_edits and isinstance(group_edits["groups"], list):
            cand = copy.deepcopy(group_edits)
        else:
            if self.groups_doc is None:
                return {"ok": False, "valid": False, "errors": ["No existing groups.json to patch; provide full groups_json."]}
            cand = copy.deepcopy(self.groups_doc)
            groups_by_name = {g["name"]: g for g in cand["groups"]}

            if isinstance(group_edits.get("remove_groups"), list):
                rm_set = set(group_edits["remove_groups"])
                cand["groups"] = [g for g in cand["groups"] if g["name"] not in rm_set]
                groups_by_name = {g["name"]: g for g in cand["groups"]}

            if isinstance(group_edits.get("add_groups"), list):
                for ng in group_edits["add_groups"]:
                    cand["groups"].append(copy.deepcopy(ng))
                    groups_by_name[ng["name"]] = cand["groups"][-1]

            if isinstance(group_edits.get("update_groups"), dict):
                for gname, gpatch in group_edits["update_groups"].items():
                    if gname not in groups_by_name:
                        return {"ok": False, "valid": False, "errors": [f"update_groups names unknown group '{gname}'"]}
                    groups_by_name[gname].update(copy.deepcopy(gpatch))

            if isinstance(group_edits.get("move_refs"), dict):
                for ref, target_grp in group_edits["move_refs"].items():
                    if target_grp not in groups_by_name:
                        return {"ok": False, "valid": False, "errors": [f"move_refs target group '{target_grp}' does not exist"]}
                    for g in cand["groups"]:
                        if ref in g["refs"] and g["name"] != target_grp:
                            g["refs"] = [r for r in g["refs"] if r != ref]
                            if g.get("anchor") == ref:
                                g["anchor"] = g["refs"][0] if g["refs"] else None
                    if ref not in groups_by_name[target_grp]["refs"]:
                        groups_by_name[target_grp]["refs"].append(ref)

        return self.validate_groups(cand)

    def patch_layout(
        self,
        rotations: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        moves: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        lock: Optional[List[str]] = None,
        unlock: Optional[List[str]] = None,
        group_edits: Optional[Dict[str, Any]] = None,
        intent_edits: Optional[Dict[str, Any]] = None,
        re_place: bool = False,
        preserve_unrelated_copper: bool = False,
        allow_forced_pose: bool = False,
    ) -> Dict[str, Any]:
        if not any([rotations, moves, lock, unlock, group_edits, intent_edits]):
            return {"ok": False, "error": "No patch fields supplied."}

        self._save_snapshot("patch_layout")
        results: Dict[str, Any] = {"ok": True}

        # 1. Group edits
        if group_edits is not None:
            g_res = self._apply_group_patch(group_edits)
            results["groups_validation"] = g_res
            if not g_res.get("valid"):
                self.undo_last_patch()
                results["ok"] = False
                results["error"] = "group_edits failed validation; rolled back."
                return results

        # 2. Intent edits
        if intent_edits is not None:
            if "blocks" in intent_edits and "kind" in intent_edits:
                cand_intent = copy.deepcopy(intent_edits)
            else:
                base_intent = copy.deepcopy(self.intent_doc) if self.intent_doc else json.loads(self.starter_intent_path.read_text(encoding="utf-8"))
                cand_intent = _deep_merge_dict(base_intent, intent_edits)
            i_res = self.validate_floorplan(cand_intent)
            results["intent_validation"] = i_res
            if not i_res.get("valid"):
                self.undo_last_patch()
                results["ok"] = False
                results["error"] = "intent_edits failed validation; rolled back."
                return results

        # 3. Pose edits via place_pose.py
        has_pose_ops = bool(rotations or moves or lock or unlock)
        if has_pose_ops:
            base_pcb = self.current_pcb
            pcb_cur, view_cur = self._get_pcb_and_view(base_pcb)
            had_copper = _board_has_copper(pcb_cur)

            affected_refs: set = set()
            near_calls: List[List[str]] = []
            batch_verbs: List[str] = []

            if rotations:
                rot_items = (
                    [(r, v) for r, v in rotations.items()]
                    if isinstance(rotations, dict)
                    else [(item["ref"], item) for item in rotations]
                )
                for ref, spec in rot_items:
                    affected_refs.add(str(ref))
                    if isinstance(spec, dict):
                        deg = float(spec["rot"])
                        batch_verbs.extend(["rotate", str(ref), str(deg)])
                        if spec.get("relative"):
                            batch_verbs.append("--relative")
                    else:
                        batch_verbs.extend(["rotate", str(ref), str(float(spec))])

            if moves:
                move_items = (
                    [(r, v) for r, v in moves.items()]
                    if isinstance(moves, dict)
                    else [(item["ref"], item) for item in moves]
                )
                for ref, spec in move_items:
                    affected_refs.add(str(ref))
                    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                        batch_verbs.extend(["set", str(ref), str(float(spec[0])), str(float(spec[1]))])
                        if len(spec) >= 3 and spec[2] is not None:
                            batch_verbs.extend(["--rot", str(float(spec[2]))])
                    elif isinstance(spec, dict):
                        x, y = float(spec["x"]), float(spec["y"])
                        if spec.get("near"):
                            op = ["set", str(ref), "--near", str(x), str(y)]
                            if spec.get("rot") is not None:
                                op.extend(["--rot", str(float(spec["rot"]))])
                            near_calls.append(op)
                        else:
                            batch_verbs.extend(["set", str(ref), str(x), str(y)])
                            if spec.get("rot") is not None:
                                batch_verbs.extend(["--rot", str(float(spec["rot"]))])

            if lock:
                batch_verbs.extend(["lock", *[str(r) for r in lock]])
            if unlock:
                batch_verbs.extend(["unlock", *[str(r) for r in unlock]])

            work_pcb = base_pcb
            unrouted_affected_nets: List[str] = []
            if had_copper:
                if preserve_unrelated_copper and affected_refs:
                    # Strip copper on non-zone, non-GND nets touching moved/rotated parts so tracks aren't stranded.
                    zone_net_ids = {z.net_id for z in (pcb_cur.zones or ()) if getattr(z, "net_id", 0)}
                    touching_nets = sorted(
                        {
                            view_cur.name_of(nid)
                            for ref in affected_refs
                            for nid in view_cur.nets_of.get(ref, set())
                            if view_cur.name_of(nid)
                            and nid not in zone_net_ids
                            and not is_ground(view_cur.name_of(nid))
                        }
                    )
                    if touching_nets:
                        stripped_pcb = self._next_step_pcb("patch_strip_affected")
                        res_st = _run_krt(
                            "py_router/route.py",
                            [str(work_pcb), str(stripped_pcb), "--undo", "--nets", *touching_nets],
                            timeout_s=60.0,
                        )
                        if res_st.returncode == 0 and stripped_pcb.exists():
                            work_pcb = stripped_pcb
                            unrouted_affected_nets = touching_nets
                elif self.placed_pcb and self.placed_pcb.exists():
                    work_pcb = self.placed_pcb

            pose_calls = ([batch_verbs] if batch_verbs else []) + near_calls
            pose_summaries = []
            for idx, verbs in enumerate(pose_calls):
                pose_out = self._next_step_pcb(f"pose_{idx}")
                pose_args = [str(work_pcb), str(pose_out)]
                intent_for_pose = self.intent_path if self.intent_path.exists() else self.starter_intent_path
                if intent_for_pose.exists():
                    pose_args.extend(["--intent", str(intent_for_pose)])
                if _board_has_copper(self._get_pcb_and_view(work_pcb)[0]):
                    pose_args.append("--allow-routed")
                if allow_forced_pose:
                    pose_args.append("--force")
                pose_args.extend(verbs)

                res_pose = _run_krt("py_placer/place_pose.py", pose_args, timeout_s=60.0)
                js = _extract_json_summary(res_pose.stdout) or {}
                pose_summaries.append({"returncode": res_pose.returncode, "summary": js, "verbs": verbs})
                if res_pose.returncode != 0 or not pose_out.exists():
                    self.undo_last_patch()
                    return {
                        "ok": False,
                        "error": f"place_pose.py refused or failed (rc={res_pose.returncode}); rolled back.",
                        "pose_attempts": pose_summaries,
                        "stdout": res_pose.stdout.strip(),
                        "stderr": res_pose.stderr.strip(),
                    }
                work_pcb = pose_out

            self.current_pcb = work_pcb
            if not _board_has_copper(self._get_pcb_and_view(work_pcb)[0]):
                self.placed_pcb = work_pcb
            elif self.placed_pcb and self.placed_pcb.exists() and work_pcb != self.placed_pcb:
                # Keep placed_pcb synchronized too when copper was preserved on current_pcb.
                placed_sync = self._next_step_pcb("placed_sync")
                sync_ok = True
                cur_p = self.placed_pcb
                for idx, verbs in enumerate(pose_calls):
                    nxt_p = placed_sync if idx == len(pose_calls) - 1 else self._next_step_pcb(f"placed_sync_{idx}")
                    s_args = [str(cur_p), str(nxt_p)]
                    intent_for_pose = self.intent_path if self.intent_path.exists() else self.starter_intent_path
                    if intent_for_pose.exists():
                        s_args.extend(["--intent", str(intent_for_pose)])
                    if allow_forced_pose:
                        s_args.append("--force")
                    s_args.extend(verbs)
                    r_sync = _run_krt("py_placer/place_pose.py", s_args, timeout_s=60.0)
                    if r_sync.returncode != 0 or not nxt_p.exists():
                        sync_ok = False
                        break
                    cur_p = nxt_p
                if sync_ok:
                    self.placed_pcb = cur_p

            results["pose_updates"] = pose_summaries
            results["unrouted_affected_nets"] = unrouted_affected_nets
            results["routing_reset_required"] = had_copper and not preserve_unrelated_copper

        if re_place:
            results["re_placement"] = self.run_placement()

        results["current_pcb"] = str(self.current_pcb)
        return results

    # -----------------------------------------------------------------------
    # Tool 10: undo_last_patch
    # -----------------------------------------------------------------------

    def undo_last_patch(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {"ok": False, "error": "No snapshots available to undo."}
        snap = self.snapshots.pop()
        restored_pcb = self._next_step_pcb(f"undo_snap_{snap['snap_idx']}")
        copy_board(str(snap["current_pcb"]), str(restored_pcb))
        self.current_pcb = restored_pcb

        if snap["placed_pcb"] and Path(snap["placed_pcb"]).exists():
            restored_placed = self._next_step_pcb(f"undo_placed_{snap['snap_idx']}")
            copy_board(str(snap["placed_pcb"]), str(restored_placed))
            self.placed_pcb = restored_placed
        else:
            self.placed_pcb = None

        self.best_seed = snap["best_seed"]
        self.groups_doc = copy.deepcopy(snap["groups_doc"])
        self.groups_validated = bool(snap["groups_validated"])
        if self.groups_doc is not None:
            self.groups_path.write_text(json.dumps(self.groups_doc, indent=2), encoding="utf-8")
        elif self.groups_path.exists():
            self.groups_path.unlink()

        self.intent_doc = copy.deepcopy(snap["intent_doc"])
        self.intent_validated = bool(snap["intent_validated"])
        if self.intent_doc is not None:
            self.intent_path.write_text(json.dumps(self.intent_doc, indent=2), encoding="utf-8")
        elif self.intent_path.exists():
            self.intent_path.unlink()

        self.routing_strategy_doc = copy.deepcopy(snap["routing_strategy_doc"])
        if self.routing_strategy_doc is not None:
            self.routing_strategy_path.write_text(
                json.dumps(self.routing_strategy_doc, indent=2), encoding="utf-8"
            )

        return {
            "ok": True,
            "reverted_action": snap["action"],
            "restored_snap_idx": snap["snap_idx"],
            "remaining_snapshots": len(self.snapshots),
            "current_pcb": str(self.current_pcb),
        }


# ---------------------------------------------------------------------------
# OpenAI / Qwen-Code / DeepSeek Agent Loop & HTTP Client
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = """You are an expert PCB layout and routing engineer controlling KiCadRoutingTools via tool calls.
Workflow:
1. Call `get_board_overview()` first; on larger boards, call `inspect_subsystem(...)` around key ICs and connectors.
2. Design functional groups (`validate_groups`) and floorplan constraints (`validate_floorplan` with `kind='floorplan-intent'`). Fix every error reported before proceeding.
3. Run `run_placement(...)` and verify floorplan/assembly cleanliness.
4. Route in disciplined order using `route_nets` (e.g., planes/power rails first with appropriate widths, then signals `['*']` with bus/ordering flags) or `run_routing_chain`.
5. Call `inspect_routing_failures()` to verify zero blocking nets, clean `kicad-cli` DRC, and physical Teel metrics.
6. Use `ripup_and_reroute(...)`, `patch_layout(...)`, or `undo_last_patch()` to resolve remaining failures before concluding."""


def _parse_qwen_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Fallback parser for Qwen/Hermes `<tool_call>` blocks (both JSON and `<function=><parameter=>` XML)."""
    calls: List[Dict[str, Any]] = []
    if not content:
        return calls
    for idx, m in enumerate(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)):
        inner = m.group(1).strip()
        if not inner:
            continue
        # Form 1: Hermes JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
                name = obj.get("name")
                args = obj.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                if name:
                    calls.append(
                        {
                            "id": f"xml_call_{idx + 1}",
                            "type": "function",
                            "function": {"name": name, "arguments": args_str},
                        }
                    )
                continue
            except Exception:
                pass
        # Form 2: Qwen3-Coder XML `<function=name><parameter=k>v</parameter></function>`
        fm = re.search(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", inner, re.DOTALL)
        if fm:
            fname = fm.group(1).strip()
            fbody = fm.group(2)
            params: Dict[str, Any] = {}
            for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", fbody, re.DOTALL):
                pkey = pm.group(1).strip()
                pval_raw = pm.group(2).strip()
                try:
                    pval: Any = json.loads(pval_raw)
                except Exception:
                    pval = pval_raw
                params[pkey] = pval
            calls.append(
                {
                    "id": f"qwen3_xml_call_{idx + 1}",
                    "type": "function",
                    "function": {"name": fname, "arguments": json.dumps(params)},
                }
            )
    return calls


def _mask_stale_observations(
    messages: List[Dict[str, Any]],
    keep_recent_tool_msgs: int = 4,
    max_obs_chars: int = 1600,
) -> List[Dict[str, Any]]:
    """
    Complexity-Trap observation masking: preserve the full latest tool diagnostics cleanly while
    replacing stale oversized historical tool outputs (> `max_obs_chars`) with compact summary stubs.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    protected = set(tool_indices[-keep_recent_tool_msgs:]) if keep_recent_tool_msgs > 0 else set()
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i not in protected:
            content = str(m.get("content") or "")
            if len(content) > max_obs_chars:
                ok_flag = None
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        ok_flag = parsed.get("valid", parsed.get("ok"))
                except Exception:
                    pass
                stub = json.dumps(
                    {
                        "masked_stale_observation": True,
                        "tool": m.get("name"),
                        "ok": ok_flag,
                        "original_chars": len(content),
                    }
                )
                mc = dict(m)
                mc["content"] = stub
                out.append(mc)
                continue
        out.append(m)
    return out


def _normalize_assistant_message(resp: Any) -> Dict[str, Any]:
    """Normalize OpenAI SDK objects, HTTP JSON payloads, or dicts into an assistant message dict."""
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            msg = dict(resp["choices"][0].get("message") or {})
        else:
            msg = dict(resp)
    elif hasattr(resp, "choices"):
        m = resp.choices[0].message
        raw_tcs = getattr(m, "tool_calls", None) or []
        msg = {
            "role": getattr(m, "role", "assistant") or "assistant",
            "content": getattr(m, "content", None),
            **(
                {"reasoning_content": getattr(m, "reasoning_content")}
                if getattr(m, "reasoning_content", None) is not None
                else {}
            ),
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tcs
            ],
        }
    else:
        raise TypeError(f"Unsupported chat response object: {type(resp).__name__}")

    msg.setdefault("role", "assistant")
    tcs = msg.get("tool_calls") or []
    if not tcs and isinstance(msg.get("content"), str):
        parsed = _parse_qwen_xml_tool_calls(msg["content"])
        if parsed:
            msg["tool_calls"] = parsed
    return msg


def make_openai_chat_fn(
    base_url: str,
    api_key: str,
    model: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout_s: float = 120.0,
) -> Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]]:
    """Zero-dependency OpenAI/Qwen-Code/DeepSeek `/v1/chat/completions` callable."""
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def _call(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat endpoint HTTP {exc.code}: {body[:500]}") from exc

    return _call


def run_agent_loop(
    session: PCBAgentSession,
    chat_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Any],
    *,
    user_prompt: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = 20,
    mask_stale_observations: bool = True,
    keep_recent_tool_msgs: int = 4,
    min_call_interval_s: float = 0.0,
    transcript_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute a multi-turn OpenAI-compatible tool-calling loop against `PCBAgentSession`,
    with optional Complexity-Trap observation masking and DeepSeek `reasoning_content` preservation.
    """
    if user_prompt is None:
        user_prompt = (
            f"Place and route board '{session.board_name}' cleanly (0 blocking nets, "
            "0 KiCad DRC errors, strong physical layout quality). Start with `get_board_overview`."
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    turns_executed = 0
    tool_calls_executed = 0
    final_text: Optional[str] = None
    t0 = time.time()
    last_call_ts = 0.0

    for _turn in range(max_turns):
        turns_executed += 1
        if min_call_interval_s > 0.0 and last_call_ts > 0.0:
            gap = time.time() - last_call_ts
            if gap < min_call_interval_s:
                time.sleep(min_call_interval_s - gap)
        model_messages = (
            _mask_stale_observations(messages, keep_recent_tool_msgs=keep_recent_tool_msgs)
            if mask_stale_observations
            else messages
        )
        last_call_ts = time.time()
        raw_resp = chat_fn(model_messages, TOOLS_SCHEMA)
        assistant_msg = _normalize_assistant_message(raw_resp)
        tool_calls = assistant_msg.get("tool_calls") or []

        stored_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_msg.get("content"),
        }
        if assistant_msg.get("reasoning_content") is not None:
            stored_msg["reasoning_content"] = assistant_msg["reasoning_content"]
        if tool_calls:
            stored_msg["tool_calls"] = tool_calls
        messages.append(stored_msg)

        if not tool_calls:
            final_text = assistant_msg.get("content")
            break

        for idx, tc in enumerate(tool_calls):
            tc_id = tc.get("id") or f"call_{turns_executed}_{idx + 1}"
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", "{}")
            result = session.dispatch_tool(fn_name, fn_args)
            tool_calls_executed += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "content": json.dumps(result),
                }
            )

    summary = {
        "ok": True,
        "board_name": session.board_name,
        "turns_executed": turns_executed,
        "tool_calls_executed": tool_calls_executed,
        "elapsed_sec": round(time.time() - t0, 2),
        "final_text": final_text,
        "current_pcb": str(session.current_pcb),
        "tool_history": session.tool_history,
        "messages": messages,
    }

    if transcript_path is not None:
        tp = Path(transcript_path)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "KICAD_CLI",
    "KRT_ROOT",
    "PCBAgentSession",
    "TOOLS_SCHEMA",
    "make_openai_chat_fn",
    "run_agent_loop",
]
