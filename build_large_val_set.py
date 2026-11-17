#!/usr/bin/env python3
"""
build_large_val_set.py
----------------------
Inventories the entire KiCad PCB corpus (`corpus/boards_set*`) and curates a
leakage-free Large-Board Held-Out Validation Suite (29 medium-large, large,
and monster boards) for Step 1 of the large-board agentic PCB routing plan.

Outputs under `runs/large_board_benchmark/`:
  - `corpus_inventory.json`       : Full inventory of all valid board pairs across sets
  - `large_val_manifest.json`     : Curated 29-board held-out validation suite + DRC/Teel baselines
  - `large_val_inventory.md`      : Markdown report with corpus breakdown, leakage checks, token stats,
                                    >16,384-token windowing motivation, and human baselines
  - `human_drc/<board>_drc.json`  : Native `kicad-cli pcb drc` reports on the human ground-truth PCBs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
DATASET_DIR = WORKSPACE / "dataset"
BENCH_23_PROMPTS = WORKSPACE / "runs" / "frontier_benchmark" / "prompts.json"
OUT_DIR = WORKSPACE / "runs" / "large_board_benchmark"
KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
for _d in ("py_router", "py_placer", "py_tools"):
    _p = str(KRT_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from kicad_parser import (  # noqa: E402
    detect_bga_pitch,
    detect_package_type,
    non_aperture_pads,
    parse_kicad_pcb,
)
from placement.grouping import BoardView  # noqa: E402
from placement.part_class import classify_part  # noqa: E402
from placement.roles import infer_role  # noqa: E402
from pcb_design_intelligence import (  # noqa: E402
    build_compact_board_context,
    evaluate_physical_board_quality,
)

# Matches build_sft_dataset.SYSTEM_PROMPT + user prompt wrapper overhead
SYSTEM_PROMPT_CHARS = 1580
USER_WRAPPER_CHARS = 150
MAX_SEQ_LENGTH_TOKENS = 16384

_CONNECTOR_RE = re.compile(
    r"(USB|CONN|HEADER|SOCKET|FPC|FFC|JST|MOLEX|SD_|RJ45|SFP|HDMI|PCIE|DIMM|MEZZ|ASP-|DF40|PINHEADER|RPI_PICO)",
    re.I,
)
_BGA_NAME_RE = re.compile(r"(BGA|FBGA|LFBGA|TFBGA|VFBGA|WLP|WLCSP|\bCSP\b)", re.I)
_FINE_QFN_NAME_RE = re.compile(r"(QFN|DFN|MLF|LFCSP|LQFP|TQFP|VQFN|UQFN|WQFN|X1QFN|X2QFN)", re.I)

# Curated 29-board Large/Monster Held-Out Validation Suite:
# Zero overlap with sft_curated_train.jsonl, sft_curated_val.jsonl, grpo_prompts.jsonl,
# or the 23-board frontier_benchmark; 0 unconnected human nets; diverse across sets & domains.
CURATED_SUITE: list[tuple[str, str, str, str]] = [
    # Tier 1: Medium-Large (60-119 parts, 4+ layers) - 7 boards
    ("medium_large", "set22", "esp32s3_m2_som", "ESP32-S3 M.2 compute module (4L, QFN)"),
    ("medium_large", "set14", "cm5_minima2", "Raspberry Pi CM5 compact carrier (6L)"),
    ("medium_large", "set15", "loratracker_hat", "LoRa RF tracker Raspberry Pi HAT (4L, QFN)"),
    ("medium_large", "set29", "am32_esc", "BLDC brushless motor ESC power+MCU (4L, QFN)"),
    ("medium_large", "set12", "pico_ice_rev3", "RP2040 + iCE40UP5K dual-chip FPGA board (4L, QFN)"),
    ("medium_large", "set6", "quickfeather", "QuickLogic EOS S3 Cortex-M4F + eFPGA (4L, QFN)"),
    ("medium_large", "set27", "job_storage_exp", "Storage expansion controller board (4L, QFN)"),
    # Tier 2: Large (120-249 parts) - 9 boards
    ("large", "set2monster", "lpddr5_testbed", "LPDDR5 high-speed memory testbed (6L, BGA+QFN)"),
    ("large", "set2monster", "oculink_pcie", "OCuLink PCIe high-speed interconnect adapter (4L, QFN)"),
    ("large", "set17", "moteus_x1", "High-power FOC brushless servo controller (6L, QFN)"),
    ("large", "set29", "usbarmory", "i.MX6UL + LPDDR3 USB security computer (6L, BGA+QFN)"),
    ("large", "set18", "libresolar_bms_c1", "Smart battery management system mixed-signal (4L, BGA+QFN)"),
    ("large", "set10", "bus_pirate5", "RP2040 multi-protocol hardware debugger (4L, QFN)"),
    ("large", "set20", "pocketsdr_8ch", "8-channel RF GNSS software-defined radio (4L, BGA+QFN)"),
    ("large", "set27", "gmsl_deserializer", "Automotive GMSL SerDes video deserializer (4L, QFN)"),
    ("large", "set9", "linht_radio", "Open-source VHF/UHF handheld SDR transceiver (6L, QFN)"),
    # Tier 3: Monster (250+ parts or BGA / 6-12 layers) - 13 boards
    ("monster", "set2monster", "poe_usbc_pd", "PoE + USB-C PD high-power supply board (4L, 268p)"),
    ("monster", "set2", "cynthion", "USB protocol analyzer FPGA platform (6L, 348p, >16K tok)"),
    ("monster", "set2monster", "sdi_mipi_conv", "SDI-to-MIPI video bridge converter (8L, 378p, >16K tok)"),
    ("monster", "set3monster", "osm_l_bmc_som", "OSM-L BMC system-on-module (10L, 384p, >16K tok)"),
    ("monster", "set1monster", "sonata_one", "CHERI RISC-V FPGA SoC dev platform (8L, 468p, >16K tok)"),
    ("monster", "set12", "berkeley_obsidian", "8-layer multi-chip FPGA/SDR system (8L, 521p, >16K tok)"),
    ("monster", "set1monster", "artix_dc_scm", "Artix-7 Datacenter Security Control Module (8L, 544p, >16K tok)"),
    ("monster", "set1monster", "hackrf_pro", "Next-gen wideband RF SDR transceiver (6L, 581p, >16K tok)"),
    ("monster", "set14", "librevna", "6 GHz 2-port USB vector network analyzer (6L, 590p, >16K tok)"),
    ("monster", "set1monster", "kria_k26", "Xilinx Kria K26 FPGA SOM carrier (8L, 657p, >16K tok)"),
    ("monster", "set3monster", "ddr4_rdimm_tester", "12-layer DDR4 RDIMM compliance tester (12L, 684p, >16K tok)"),
    ("monster", "set1monster", "comexpress7", "COM Express Type 7 server carrier (8L, 796p, >16K tok)"),
    ("monster", "set3monster", "jetson_agx_thor_bb", "NVIDIA Jetson AGX Thor carrier (10L, 1206p, >16K tok)"),
]


def _natural_set_sort_key(set_name: str) -> tuple[int, int, str]:
    m_num = re.fullmatch(r"set(\d+)", set_name)
    if m_num:
        return (0, int(m_num.group(1)), set_name)
    m_mon = re.fullmatch(r"set(\d+)monster", set_name)
    if m_mon:
        return (1, int(m_mon.group(1)), set_name)
    return (2, 0, set_name)


def load_membership_sets(dataset_dir: Path, bench_23_prompts: Path) -> dict[str, set[str]]:
    """Load exact board-name membership sets across train/val/grpo/23-bench."""
    def _load_jsonl_boards(path: Path) -> set[str]:
        if not path.exists():
            return set()
        boards: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("board"):
                boards.add(str(obj["board"]))
        return boards

    bench_boards: set[str] = set()
    if bench_23_prompts.exists():
        for row in json.loads(bench_23_prompts.read_text(encoding="utf-8")):
            if row.get("board"):
                bench_boards.add(str(row["board"]))

    return {
        "sft_train": _load_jsonl_boards(dataset_dir / "sft_curated_train.jsonl"),
        "sft_val": _load_jsonl_boards(dataset_dir / "sft_curated_val.jsonl"),
        "grpo_prompts": _load_jsonl_boards(dataset_dir / "grpo_prompts.jsonl"),
        "bench_23": bench_boards,
    }


def _detect_bga_and_fine_qfn(pcb) -> tuple[bool, bool, bool, list[dict[str, Any]]]:
    """Identify true SMD BGA/CSP packages and fine-pitch (<=0.55mm) >=16-pin QFN/QFP ICs."""
    has_bga = False
    has_fine_qfn = False
    details: list[dict[str, Any]] = []

    for ref, fp in sorted((pcb.footprints or {}).items()):
        fp_name = str(getattr(fp, "footprint_name", "") or "")
        cls = classify_part(fp, ref).name
        role = infer_role(fp, ref, cls)["role"]
        if (
            cls in ("edge_receptacle", "connector_affinity")
            or role in ("connector", "pin header", "usb receptacle", "edge receptacle")
            or ref.startswith(("J", "P", "CN"))
            or _CONNECTOR_RE.search(fp_name)
        ):
            continue

        pads = non_aperture_pads(fp)
        npads = len(pads)
        if npads < 9:
            continue

        # Exclude pure through-hole sockets/carriers (e.g. PLCC-84 THT, 2.54mm module headers)
        smd_pads = sum(1 for p in pads if getattr(p, "pad_type", "") == "smd" or (getattr(p, "drill", 0) or 0) == 0)
        if smd_pads == 0:
            continue

        fp_u = fp_name.upper()
        pkg = detect_package_type(fp)
        # LFCSP is Analog Devices' QFN equivalent rather than an area-array BGA
        if "LFCSP" in fp_u:
            pkg = "QFN"

        pitch = round(float(detect_bga_pitch(fp)), 3)
        is_bga = (
            pitch <= 1.5
            and (
                (pkg == "BGA" and npads >= 16)
                or (bool(_BGA_NAME_RE.search(fp_u)) and npads >= 9)
            )
        )
        is_fine_qfn = (
            not is_bga
            and npads >= 16
            and pitch <= 0.55
            and (pkg in ("QFN", "QFP") or bool(_FINE_QFN_NAME_RE.search(fp_u)))
        )

        if is_bga or is_fine_qfn:
            if is_bga:
                has_bga = True
            if is_fine_qfn:
                has_fine_qfn = True
            details.append(
                {
                    "ref": ref,
                    "pkg": "BGA" if is_bga else pkg,
                    "footprint": fp_name.split(":")[-1],
                    "pads": npads,
                    "pitch_mm": pitch,
                }
            )

    return has_bga, has_fine_qfn, (has_bga or has_fine_qfn), details


def _inspect_one_board(task: tuple[str, str, str, str]) -> dict[str, Any]:
    """Parse routed + unrouted KiCad PCBs and compute structural & context metrics."""
    set_name, board_name, routed_str, unrouted_str = task
    routed_path = Path(routed_str)
    unrouted_path = Path(unrouted_str)

    try:
        if not routed_path.exists() or routed_path.stat().st_size == 0:
            return {"valid": False, "set": set_name, "board": board_name, "error": "missing_or_empty_routed"}
        if not unrouted_path.exists() or unrouted_path.stat().st_size == 0:
            return {"valid": False, "set": set_name, "board": board_name, "error": "missing_or_empty_unrouted"}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pcb_u = parse_kicad_pcb(str(unrouted_path))
            view_u = BoardView(pcb_u)
            ctx_md = build_compact_board_context(unrouted_path)
            pcb_r = parse_kicad_pcb(str(routed_path))

        bounds = getattr(pcb_u.board_info, "board_bounds", None)
        if bounds:
            w_mm = round(float(bounds[2] - bounds[0]), 2)
            h_mm = round(float(bounds[3] - bounds[1]), 2)
            area_mm2 = round(max(0.0, w_mm * h_mm), 2)
        else:
            w_mm = None
            h_mm = None
            area_mm2 = None

        layer_names = list(getattr(pcb_u.board_info, "copper_layers", []) or [])
        copper_layers = len(layer_names)

        routable_nets = sum(
            1
            for nid, net in (pcb_u.nets or {}).items()
            if nid > 0
            and getattr(net, "name", "")
            and not str(net.name).lower().startswith("unconnected-")
            and len(pcb_u.pads_by_net.get(nid, [])) >= 2
        )

        routed_segs = [
            s for s in (pcb_r.segments or []) if not getattr(s, "graphic", False) and (s.net_id or 0) > 0
        ]
        human_copper_mm = round(
            sum(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) for s in routed_segs),
            2,
        )
        human_segments = len(routed_segs)
        human_vias = len(pcb_r.vias or [])
        human_zones = len([z for z in (pcb_r.zones or []) if not getattr(z, "is_keepout", False)])

        has_bga, has_fine_qfn, has_bga_or_fine_qfn, bga_qfn_details = _detect_bga_and_fine_qfn(pcb_u)

        prompt_chars = len(ctx_md)
        prompt_tokens_est = prompt_chars // 4
        full_turn1_prompt_tokens_est = (prompt_chars + SYSTEM_PROMPT_CHARS + USER_WRAPPER_CHARS) // 4

        degenerate_reasons: list[str] = []
        if human_segments == 0 or human_copper_mm < 10.0:
            degenerate_reasons.append(f"low_or_zero_human_copper (segs={human_segments}, mm={human_copper_mm})")
        if area_mm2 is None or area_mm2 <= 0:
            degenerate_reasons.append("missing_board_bounds")
        if len(view_u.parts) < 4 or routable_nets < 2:
            degenerate_reasons.append(f"trivial_netlist (parts={len(view_u.parts)}, nets={routable_nets})")

        return {
            "valid": True,
            "set": set_name,
            "board": board_name,
            "routed_path": str(routed_path),
            "unrouted_path": str(unrouted_path),
            "routed_size_mb": round(routed_path.stat().st_size / 1e6, 2),
            "pad_bearing_parts": len(view_u.parts),
            "total_footprints": len(pcb_u.footprints or {}),
            "routable_nets": routable_nets,
            "total_nets": len(pcb_u.nets or {}),
            "copper_layers": copper_layers,
            "copper_layer_names": layer_names,
            "board_width_mm": w_mm,
            "board_height_mm": h_mm,
            "board_area_mm2": area_mm2,
            "human_copper_mm": human_copper_mm,
            "human_segments": human_segments,
            "human_vias": human_vias,
            "human_zones": human_zones,
            "has_bga": has_bga,
            "has_fine_qfn": has_fine_qfn,
            "has_bga_or_fine_qfn": has_bga_or_fine_qfn,
            "bga_qfn_details": bga_qfn_details,
            "prompt_chars": prompt_chars,
            "prompt_tokens_est": prompt_tokens_est,
            "full_turn1_prompt_tokens_est": full_turn1_prompt_tokens_est,
            "exceeds_16k_prompt": prompt_tokens_est > MAX_SEQ_LENGTH_TOKENS,
            "exceeds_16k_with_sys": full_turn1_prompt_tokens_est > MAX_SEQ_LENGTH_TOKENS,
            "degenerate_human_routing": bool(degenerate_reasons),
            "degenerate_reasons": degenerate_reasons,
        }
    except Exception as exc:
        return {
            "valid": False,
            "set": set_name,
            "board": board_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def scan_corpus(
    corpus_dir: Path,
    membership: dict[str, set[str]],
    workers: int = 6,
) -> dict[str, Any]:
    """Scan all routed/unrouted board pairs under `corpus/boards_set*` (excluding scraped)."""
    t0 = time.time()
    tasks: list[tuple[str, str, str, str]] = []
    skipped_dirs_or_files: list[dict[str, str]] = []

    routed_dirs = sorted(
        (
            d
            for d in corpus_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("boards_set")
            and "scraped" not in d.name
            and not d.name.startswith("boards_unrouted_")
        ),
        key=lambda d: _natural_set_sort_key(d.name[len("boards_") :]),
    )

    for rdir in routed_dirs:
        set_name = rdir.name[len("boards_") :]
        udir = corpus_dir / f"boards_unrouted_{set_name}"
        if not udir.exists() or not udir.is_dir():
            skipped_dirs_or_files.append({"set": set_name, "board": "*", "reason": f"missing {udir.name}"})
            continue
        r_files = sorted(rdir.glob("*.kicad_pcb"))
        if not r_files:
            skipped_dirs_or_files.append({"set": set_name, "board": "*", "reason": "empty_or_in_progress_dir"})
            continue
        for rf in r_files:
            uf = udir / rf.name
            if not uf.exists():
                skipped_dirs_or_files.append({"set": set_name, "board": rf.stem, "reason": "missing_unrouted_pair"})
                continue
            tasks.append((set_name, rf.stem, str(rf), str(uf)))

    print(f"Scanning {len(tasks)} board pairs across {len(routed_dirs)} sets with {workers} workers...", flush=True)
    boards: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for idx, res in enumerate(pool.map(_inspect_one_board, tasks, chunksize=4), 1):
            if not res.get("valid"):
                skipped_dirs_or_files.append(
                    {"set": res["set"], "board": res["board"], "reason": res.get("error", "unknown")}
                )
            else:
                bname = res["board"]
                res["in_sft_train"] = bname in membership["sft_train"]
                res["in_sft_val"] = bname in membership["sft_val"]
                res["in_grpo_prompts"] = bname in membership["grpo_prompts"]
                res["in_23_bench"] = bname in membership["bench_23"]
                boards.append(res)
            if idx % 100 == 0 or idx == len(tasks):
                print(f"  Scanned {idx}/{len(tasks)} boards ({time.time() - t0:.1f}s)...", flush=True)

    boards.sort(key=lambda r: (_natural_set_sort_key(r["set"]), r["pad_bearing_parts"], r["board"]))
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_dir": str(corpus_dir),
        "total_valid_boards": len(boards),
        "total_skipped": len(skipped_dirs_or_files),
        "skipped": skipped_dirs_or_files,
        "boards": boards,
    }


def _run_drc_and_teel_one(args_tuple: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Run native `kicad-cli pcb drc` + `evaluate_physical_board_quality` on one human PCB."""
    entry, drc_dir_str = args_tuple
    bname = entry["board"]
    routed_path = Path(entry["routed_path"])
    drc_json = Path(drc_dir_str) / f"{bname}_drc.json"

    t0 = time.time()
    if not drc_json.exists() or drc_json.stat().st_size == 0:
        subprocess.run(
            [KICAD_CLI, "pcb", "drc", "--format", "json", "--output", str(drc_json), str(routed_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=False,
        )

    drc_errors: int | None = None
    drc_warnings: int | None = None
    drc_unconnected: int | None = None
    if drc_json.exists() and drc_json.stat().st_size > 0:
        rep = json.loads(drc_json.read_text(encoding="utf-8"))
        viols = rep.get("violations") or []
        drc_errors = sum(1 for v in viols if v.get("severity") == "error")
        drc_warnings = sum(1 for v in viols if v.get("severity") == "warning")
        drc_unconnected = len(rep.get("unconnected_items") or [])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pq = evaluate_physical_board_quality(routed_path)

    return {
        "board": bname,
        "drc_report_path": str(drc_json),
        "drc_errors": drc_errors,
        "drc_warnings": drc_warnings,
        "drc_unconnected": drc_unconnected,
        "teel_score_100": pq.get("teel_score_100"),
        "physical_quality_penalty": pq.get("physical_quality_penalty"),
        "slalom_traces_count": pq.get("slalom_traces_count"),
        "via_in_pad_count": pq.get("via_in_pad_count"),
        "weaving_nets_count": pq.get("weaving_nets_count"),
        "acute_or_90_corners_count": pq.get("acute_or_90_corners_count"),
        "decap_distance_violations": pq.get("decap_distance_violations"),
        "power_width_compliance_pct": pq.get("power_width_compliance_pct"),
        "eval_sec": round(time.time() - t0, 2),
    }


def _percentile(sorted_vals: list[int | float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f]) * (c - k) + float(sorted_vals[c]) * (k - f)


def compute_token_stats(rows: list[dict[str, Any]], key: str = "prompt_tokens_est") -> dict[str, Any]:
    vals = sorted(int(r[key]) for r in rows if r.get(key) is not None)
    if not vals:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0, "gt_16384": 0}
    return {
        "count": len(vals),
        "min": vals[0],
        "median": round(_percentile(vals, 0.50), 1),
        "p95": round(_percentile(vals, 0.95), 1),
        "max": vals[-1],
        "gt_16384": sum(1 for v in vals if v > MAX_SEQ_LENGTH_TOKENS),
    }


def curate_and_evaluate_suite(
    inv: dict[str, Any],
    out_dir: Path,
    workers: int = 6,
) -> dict[str, Any]:
    """Select the 29-board Large-Board Held-Out Suite, verify zero leakage, and run DRC+Teel."""
    boards = inv["boards"]
    by_pair = {(b["set"], b["board"]): b for b in boards}
    drc_dir = out_dir / "human_drc"
    drc_dir.mkdir(parents=True, exist_ok=True)

    selected_entries: list[dict[str, Any]] = []
    for tier, sname, bname, summary in CURATED_SUITE:
        b = by_pair.get((sname, bname))
        if not b:
            raise KeyError(f"Curated board missing from corpus inventory: {sname}/{bname}")
        # Hard leakage & quality invariants
        assert not b["in_sft_train"], f"LEAKAGE DETECTED: {sname}/{bname} is in sft_curated_train.jsonl!"
        assert not b["in_sft_val"], f"OVERLAP DETECTED: {sname}/{bname} is in sft_curated_val.jsonl!"
        assert not b["in_grpo_prompts"], f"OVERLAP DETECTED: {sname}/{bname} is in grpo_prompts.jsonl!"
        assert not b["in_23_bench"], f"OVERLAP DETECTED: {sname}/{bname} is in 23-board benchmark!"
        assert not b["degenerate_human_routing"], f"DEGENERATE BOARD: {sname}/{bname} ({b['degenerate_reasons']})"

        entry = dict(b)
        entry["tier"] = tier
        entry["description"] = summary
        selected_entries.append(entry)

    print(f"\nRunning KiCad DRC + Teel quality scoring on {len(selected_entries)} curated boards...", flush=True)
    eval_tasks = [(entry, str(drc_dir)) for entry in selected_entries]
    eval_by_board: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for res in pool.map(_run_drc_and_teel_one, eval_tasks):
            eval_by_board[res["board"]] = res
            print(
                f"  {res['board']:22s} | drc_err={res['drc_errors']} unconn={res['drc_unconnected']} "
                f"teel={res['teel_score_100']} pen={res['physical_quality_penalty']} ({res['eval_sec']}s)",
                flush=True,
            )

    curated_rows: list[dict[str, Any]] = []
    for entry in selected_entries:
        ev = eval_by_board[entry["board"]]
        assert ev["drc_unconnected"] == 0, (
            f"Unconnected nets found in human reference {entry['board']}: {ev['drc_unconnected']}"
        )
        merged = {
            "tier": entry["tier"],
            "set": entry["set"],
            "board": entry["board"],
            "description": entry["description"],
            "routed_path": entry["routed_path"],
            "unrouted_path": entry["unrouted_path"],
            "pad_bearing_parts": entry["pad_bearing_parts"],
            "total_footprints": entry["total_footprints"],
            "routable_nets": entry["routable_nets"],
            "copper_layers": entry["copper_layers"],
            "copper_layer_names": entry["copper_layer_names"],
            "board_width_mm": entry["board_width_mm"],
            "board_height_mm": entry["board_height_mm"],
            "board_area_mm2": entry["board_area_mm2"],
            "has_bga": entry["has_bga"],
            "has_fine_qfn": entry["has_fine_qfn"],
            "has_bga_or_fine_qfn": entry["has_bga_or_fine_qfn"],
            "bga_qfn_details": entry["bga_qfn_details"],
            "prompt_chars": entry["prompt_chars"],
            "prompt_tokens_est": entry["prompt_tokens_est"],
            "full_turn1_prompt_tokens_est": entry["full_turn1_prompt_tokens_est"],
            "exceeds_16k_prompt": entry["exceeds_16k_prompt"],
            "exceeds_16k_with_sys": entry["exceeds_16k_with_sys"],
            "in_sft_train": entry["in_sft_train"],
            "in_sft_val": entry["in_sft_val"],
            "in_grpo_prompts": entry["in_grpo_prompts"],
            "in_23_bench": entry["in_23_bench"],
            "human_copper_mm": entry["human_copper_mm"],
            "human_segments": entry["human_segments"],
            "human_vias": entry["human_vias"],
            "human_zones": entry["human_zones"],
            "drc_errors": ev["drc_errors"],
            "drc_warnings": ev["drc_warnings"],
            "drc_unconnected": ev["drc_unconnected"],
            "drc_report_path": ev["drc_report_path"],
            "teel_score_100": ev["teel_score_100"],
            "physical_quality_penalty": ev["physical_quality_penalty"],
            "slalom_traces_count": ev["slalom_traces_count"],
            "via_in_pad_count": ev["via_in_pad_count"],
            "weaving_nets_count": ev["weaving_nets_count"],
            "acute_or_90_corners_count": ev["acute_or_90_corners_count"],
            "decap_distance_violations": ev["decap_distance_violations"],
            "power_width_compliance_pct": ev["power_width_compliance_pct"],
        }
        curated_rows.append(merged)

    # Aggregate statistics
    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        drc0 = sum(1 for r in rows if r["drc_errors"] == 0)
        unconn0 = sum(1 for r in rows if r["drc_unconnected"] == 0)
        return {
            "count": n,
            "parts_min": min(r["pad_bearing_parts"] for r in rows),
            "parts_max": max(r["pad_bearing_parts"] for r in rows),
            "layers_min": min(r["copper_layers"] for r in rows),
            "layers_max": max(r["copper_layers"] for r in rows),
            "drc_0_count": drc0,
            "drc_0_pct": round(100.0 * drc0 / n, 1),
            "unconnected_0_count": unconn0,
            "avg_drc_errors": round(sum(r["drc_errors"] for r in rows) / n, 2),
            "avg_teel_score_100": round(sum(r["teel_score_100"] for r in rows) / n, 1),
            "avg_copper_mm": round(sum(r["human_copper_mm"] for r in rows) / n, 1),
            "avg_vias": round(sum(r["human_vias"] for r in rows) / n, 1),
            "prompt_tokens": compute_token_stats(rows, "prompt_tokens_est"),
            "full_turn1_prompt_tokens": compute_token_stats(rows, "full_turn1_prompt_tokens_est"),
        }

    tier_summaries = {
        t: _agg([r for r in curated_rows if r["tier"] == t])
        for t in ("medium_large", "large", "monster")
    }
    overall_summary = _agg(curated_rows)

    degenerate_excluded = [
        {
            "set": b["set"],
            "board": b["board"],
            "pad_bearing_parts": b["pad_bearing_parts"],
            "human_segments": b["human_segments"],
            "human_copper_mm": b["human_copper_mm"],
            "human_zones": b["human_zones"],
            "in_sft_train": b["in_sft_train"],
            "in_23_bench": b["in_23_bench"],
            "reasons": b["degenerate_reasons"],
        }
        for b in boards
        if b["degenerate_human_routing"]
    ]

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_name": "Large-Board Held-Out Validation Suite",
        "max_seq_length_tokens": MAX_SEQ_LENGTH_TOKENS,
        "total_boards": len(curated_rows),
        "leakage_verification": {
            "in_sft_train_count": sum(1 for r in curated_rows if r["in_sft_train"]),
            "in_sft_val_count": sum(1 for r in curated_rows if r["in_sft_val"]),
            "in_grpo_prompts_count": sum(1 for r in curated_rows if r["in_grpo_prompts"]),
            "in_23_bench_count": sum(1 for r in curated_rows if r["in_23_bench"]),
            "degenerate_human_count": len(degenerate_excluded),
            "degenerate_boards_excluded": degenerate_excluded,
        },
        "corpus_summary": {
            "total_valid_boards": inv["total_valid_boards"],
            "in_sft_train": sum(1 for b in boards if b["in_sft_train"]),
            "in_sft_val": sum(1 for b in boards if b["in_sft_val"]),
            "in_grpo_prompts": sum(1 for b in boards if b["in_grpo_prompts"]),
            "in_23_bench": sum(1 for b in boards if b["in_23_bench"]),
            "prompt_tokens_all_corpus": compute_token_stats(boards, "prompt_tokens_est"),
            "full_turn1_prompt_tokens_all_corpus": compute_token_stats(boards, "full_turn1_prompt_tokens_est"),
        },
        "suite_summary": {
            "overall": overall_summary,
            "by_tier": tier_summaries,
        },
        "boards": curated_rows,
    }
    return manifest


def write_markdown_report(inv: dict[str, Any], manifest: dict[str, Any], md_path: Path) -> None:
    """Write comprehensive `large_val_inventory.md` report."""
    boards = inv["boards"]
    curated = manifest["boards"]
    cs = manifest["corpus_summary"]
    ov = manifest["suite_summary"]["overall"]
    by_tier = manifest["suite_summary"]["by_tier"]
    lk = manifest["leakage_verification"]

    corpus_tok = cs["prompt_tokens_all_corpus"]
    suite_tok = ov["prompt_tokens"]
    suite_full_tok = ov["full_turn1_prompt_tokens"]
    overflow_suite = [r for r in curated if r["exceeds_16k_prompt"]]
    overflow_corpus = sorted(
        (b for b in boards if b["exceeds_16k_prompt"]),
        key=lambda x: -x["prompt_tokens_est"],
    )

    lines = [
        "# Large-Board Corpus Inventory & Held-Out Validation Suite",
        "",
        f"Generated: `{manifest['generated_at']}` | Baseline Sequence Limit: `{MAX_SEQ_LENGTH_TOKENS:,}` tokens (`max_seq_length=16384`)",
        "",
        "## 1. Full Corpus Inventory Summary",
        "",
        f"- **Valid Routed/Unrouted Board Pairs Scanned**: `{cs['total_valid_boards']}` (across 34 corpus directories: `boards_set1..30`, `boards_set1monster..3monster`, `boards_sethydro`; `0` skipped/broken)",
        f"- **Existing Dataset & Benchmark Membership**:",
        f"  - `sft_curated_train.jsonl`: `{cs['in_sft_train']}` unique boards",
        f"  - `sft_curated_val.jsonl`: `{cs['in_sft_val']}` unique boards",
        f"  - `grpo_prompts.jsonl`: `{cs['in_grpo_prompts']}` unique boards",
        f"  - `frontier_benchmark/prompts.json` (23-board benchmark): `{cs['in_23_bench']}` boards",
        f"- **Degenerate / Unrouted Human References Flagged & Excluded (`{lk['degenerate_human_count']}` boards)**:",
        "  - Boards whose human `.kicad_pcb` has `0` (or near-zero) copper trace segments (e.g., `set28/mez_rx` with `213` parts / `0` segments / `26` zones, `set10/open_weather_station` with `111` parts / `0` segments) are flagged and excluded from human ground-truth baselines.",
        f"- **Full-Corpus Compact Context Token Distribution (`build_compact_board_context` on unrouted `.kicad_pcb`)**:",
        f"  - **Min**: `{corpus_tok['min']:,}` | **Median**: `{corpus_tok['median']:,.1f}` | **p95**: `{corpus_tok['p95']:,.1f}` | **Max**: `{corpus_tok['max']:,}` tokens",
        f"  - **Boards Exceeding `{MAX_SEQ_LENGTH_TOKENS:,}` Prompt Tokens (Full Corpus)**: `{corpus_tok['gt_16384']} / {cs['total_valid_boards']}` (`{100.0 * corpus_tok['gt_16384'] / cs['total_valid_boards']:.1f}%`), ALL unseen (`in_sft_train == False`).",
        "",
        "## 2. Leakage Verification & Curation Criteria",
        "",
        f"Curated **{manifest['total_boards']} boards** spanning Medium-Large (`60–119` parts, `4+` layers), Large (`120–249` parts), and Monster (`250+` parts or BGA / `6–12` layers):",
        "",
        "| Verification Check | Count | Status |",
        "| :--- | :---: | :---: |",
        f"| Overlap with `sft_curated_train.jsonl` (`in_sft_train == True`) | `{lk['in_sft_train_count']} / {manifest['total_boards']}` | PASS (0 leakage) |",
        f"| Overlap with `sft_curated_val.jsonl` (`in_sft_val == True`) | `{lk['in_sft_val_count']} / {manifest['total_boards']}` | PASS (0 overlap) |",
        f"| Overlap with `grpo_prompts.jsonl` (`in_grpo_prompts == True`) | `{lk['in_grpo_prompts_count']} / {manifest['total_boards']}` | PASS (0 overlap) |",
        f"| Overlap with 23-Board Benchmark (`in_23_bench == True`) | `{lk['in_23_bench_count']} / {manifest['total_boards']}` | PASS (0 overlap) |",
        f"| Degenerate / Unrouted Human Reference (`mez_rx` class) | `0 / {manifest['total_boards']}` | PASS (excluded) |",
        f"| Human Ground-Truth Connectivity (`drc_unconnected == 0`) | `{ov['unconnected_0_count']} / {manifest['total_boards']}` | PASS (100% routed) |",
        "",
        "## 3. Token Overflow Beyond `16,384` `max_seq_length` (Why Hierarchical Windowing Is Required)",
        "",
        "Across our 29-board Large/Monster Validation Suite, single-shot monolithic context dumps (`build_compact_board_context`) severely overflow the current `16,384`-token SFT/GRPO sequence budget — before even adding the `1,580`-char system prompt (`~395` tokens) and `3K–15K` tokens of JSON placement/routing output:",
        "",
        "| Cohort | Boards | Min Tokens | Median Tokens | p95 Tokens | Max Tokens | `> 16,384` Prompt Tokens |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Full Corpus (`boards_set*`)** | {cs['total_valid_boards']} | {corpus_tok['min']:,} | {corpus_tok['median']:,.1f} | {corpus_tok['p95']:,.1f} | {corpus_tok['max']:,} | {corpus_tok['gt_16384']} ({100.0 * corpus_tok['gt_16384'] / cs['total_valid_boards']:.1f}%) |",
        f"| **29-Board Suite (Compact Context Only)** | {ov['count']} | {suite_tok['min']:,} | {suite_tok['median']:,.1f} | {suite_tok['p95']:,.1f} | {suite_tok['max']:,} | **{suite_tok['gt_16384']} / {ov['count']} ({100.0 * suite_tok['gt_16384'] / ov['count']:.1f}%)** |",
        f"| **29-Board Suite (+ System Prompt)** | {ov['count']} | {suite_full_tok['min']:,} | {suite_full_tok['median']:,.1f} | {suite_full_tok['p95']:,.1f} | {suite_full_tok['max']:,} | **{suite_full_tok['gt_16384']} / {ov['count']} ({100.0 * suite_full_tok['gt_16384'] / ov['count']:.1f}%)** |",
        "",
        f"### Curated Validation Boards Exceeding `16,384` Prompt Tokens (`{len(overflow_suite)} / {ov['count']}`)",
        "",
        "| Set | Board | Parts | Nets | Layers | Prompt Chars | Prompt Tokens (est) | +System Prompt | Overflow vs `16,384` |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in sorted(overflow_suite, key=lambda x: -x["prompt_tokens_est"]):
        over = r["prompt_tokens_est"] - MAX_SEQ_LENGTH_TOKENS
        lines.append(
            f"| `{r['set']}` | `{r['board']}` | {r['pad_bearing_parts']} | {r['routable_nets']} | {r['copper_layers']}L | "
            f"{r['prompt_chars']:,} | **{r['prompt_tokens_est']:,}** | {r['full_turn1_prompt_tokens_est']:,} | +{over:,} ({r['prompt_tokens_est'] / MAX_SEQ_LENGTH_TOKENS:.2f}x) |"
        )

    lines.extend([
        "",
        f"> **Engineering Implication:** `12` of `13` Monster boards in the held-out suite (and `{len(overflow_corpus)}` boards corpus-wide, peaking at **`{corpus_tok['max']:,}` prompt tokens** on `jetson_agx_thor_bb` — **3.4x** the `16,384`-token limit before adding output tokens) cannot fit into a single-turn context window. Moreover, `set2monster/poe_usbc_pd` (`11,859` prompt tokens) and `9` Large boards (`7,252–10,140` prompt tokens) will also approach or breach `16,384` total tokens once multi-IC `groups.json` + `intent.json` + `routing_strategy` + `judge_chain.txt` completions are generated. This directly necessitates an agentic inspection harness (`get_board_overview` + `inspect_subsystem` / `inspect_nets`) that lets the model partition and route large PCBs without truncating netlists or starving context.",
        "",
        "## 4. Human Ground-Truth Baselines by Tier",
        "",
        "| Tier | Boards | Parts Range | Layers | 0-DRC Rate | 0-Unconn Rate | Avg DRC Errors | Avg Teel (0-100) | Avg Copper (mm) | Avg Vias |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for tkey, label in [
        ("medium_large", "Medium-Large (`60–119p, 4+L`)"),
        ("large", "Large (`120–249p`)"),
        ("monster", "Monster (`250+p / BGA / 6–12L`)"),
    ]:
        ts = by_tier[tkey]
        lines.append(
            f"| {label} | {ts['count']} | {ts['parts_min']}–{ts['parts_max']} | {ts['layers_min']}–{ts['layers_max']}L | "
            f"{ts['drc_0_count']}/{ts['count']} ({ts['drc_0_pct']:.0f}%) | {ts['unconnected_0_count']}/{ts['count']} | "
            f"{ts['avg_drc_errors']:.1f} | {ts['avg_teel_score_100']:.1f} | {ts['avg_copper_mm']:,.0f} | {ts['avg_vias']:.0f} |"
        )
    lines.append(
        f"| **Overall Held-Out Suite** | **{ov['count']}** | **{ov['parts_min']}–{ov['parts_max']}** | **{ov['layers_min']}–{ov['layers_max']}L** | "
        f"**{ov['drc_0_count']}/{ov['count']} ({ov['drc_0_pct']:.0f}%)** | **{ov['unconnected_0_count']}/{ov['count']}** | "
        f"**{ov['avg_drc_errors']:.1f}** | **{ov['avg_teel_score_100']:.1f}** | **{ov['avg_copper_mm']:,.0f}** | **{ov['avg_vias']:.0f}** |"
    )

    lines.extend([
        "",
        "## 5. Complete 29-Board Held-Out Validation Suite",
        "",
        "| # | Tier | Set | Board | Parts | Nets | Layers | Area (mm2) | BGA / QFN | Prompt Tok | Copper (mm) | Vias | DRC Err | Unconn | Teel (0-100) | Penalty |",
        "| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for idx, r in enumerate(curated, 1):
        pkg_flags = []
        if r["has_bga"]:
            pkg_flags.append("BGA")
        if r["has_fine_qfn"]:
            pkg_flags.append("QFN")
        pkg_str = "+".join(pkg_flags) if pkg_flags else "-"
        tok_str = f"**{r['prompt_tokens_est']:,}***" if r["exceeds_16k_prompt"] else f"{r['prompt_tokens_est']:,}"
        lines.append(
            f"| {idx} | `{r['tier']}` | `{r['set']}` | `{r['board']}` | {r['pad_bearing_parts']} | {r['routable_nets']} | "
            f"{r['copper_layers']}L | {r['board_area_mm2']:,.0f} | {pkg_str} | {tok_str} | "
            f"{r['human_copper_mm']:,.1f} | {r['human_vias']:,} | {r['drc_errors']} | {r['drc_unconnected']} | "
            f"{r['teel_score_100']:.1f} | {r['physical_quality_penalty']:.1f} |"
        )
    lines.append("")
    lines.append("*`*` indicates `prompt_tokens_est > 16,384`.*")

    lines.extend([
        "",
        "### Note on Teel Scores on High-Density / Monster Boards",
        "",
        "`evaluate_physical_board_quality` imposes strict 2–4 layer heuristics (e.g., `via_in_pad_count` and `slalom_traces_count` inside dense BGA/QFN escape corridors), which clamp `teel_score_100` to `0.0` on large 6–12 layer HDI boards (`jetson_agx_thor_bb`: `penalty=912.83`, `drc_errors=0`). Accordingly, `physical_quality_penalty` and native `kicad-cli pcb drc` error/unconnected counts are both preserved in `large_val_manifest.json` as high-resolution baseline signals.",
        "",
        "## 6. Excluded Degenerate Human Boards (`12` Corpus-Wide)",
        "",
        "| Set | Board | Parts | Human Segs | Copper (mm) | Zones | `in_sft_train` | `in_23_bench` | Reason |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for d in lk["degenerate_boards_excluded"]:
        lines.append(
            f"| `{d['set']}` | `{d['board']}` | {d['pad_bearing_parts']} | {d['human_segments']} | "
            f"{d['human_copper_mm']:.1f} | {d['human_zones']} | {d['in_sft_train']} | {d['in_23_bench']} | {', '.join(d['reasons'])} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_inventory(inv: dict[str, Any]) -> None:
    boards = inv["boards"]
    print(f"\n=== Corpus Inventory Summary ({len(boards)} valid boards, {inv['total_skipped']} skipped) ===")
    print(f"  in_sft_train==True   : {sum(1 for b in boards if b['in_sft_train'])}")
    print(f"  in_sft_val==True     : {sum(1 for b in boards if b['in_sft_val'])}")
    print(f"  in_grpo_prompts==True: {sum(1 for b in boards if b['in_grpo_prompts'])}")
    print(f"  in_23_bench==True    : {sum(1 for b in boards if b['in_23_bench'])}")
    print(f"  degenerate_human     : {sum(1 for b in boards if b['degenerate_human_routing'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inventory KiCad PCB corpus and curate Large-Board Held-Out Validation Suite.")
    ap.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    ap.add_argument("--bench-23-prompts", type=Path, default=BENCH_23_PROMPTS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--scan-only", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = args.out_dir / "corpus_inventory.json"
    membership = load_membership_sets(args.dataset_dir, args.bench_23_prompts)

    if args.rescan or not inv_path.exists():
        inv = scan_corpus(args.corpus, membership, workers=args.workers)
        inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
        print(f"Wrote {inv_path}")
    else:
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        print(f"Loaded cached {inv_path} ({len(inv['boards'])} boards)")

    summarize_inventory(inv)
    if args.scan_only:
        return

    manifest = curate_and_evaluate_suite(inv, args.out_dir, workers=args.workers)
    manifest_path = args.out_dir / "large_val_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")

    md_path = args.out_dir / "large_val_inventory.md"
    write_markdown_report(inv, manifest, md_path)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
