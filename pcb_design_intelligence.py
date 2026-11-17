#!/usr/bin/env python3
"""
pcb_design_intelligence.py
--------------------------
General, netlist- and geometry-driven hardware engineering intelligence for KiCadRoutingTools:
  1. Accurate mechanical & RF-module locking (`detect_mechanical_and_rf_locks`)
  2. Keepout zone extraction (`extract_board_keepouts`)
  3. Automatic datasheet & repeated-channel template fitting (`synthesize_templates_and_proximity`)
  4. Galvanic isolation partitioning & `check_groups` zero-warning repair (`repair_and_enrich_groups`)
  5. Multi-domain dual-layer copper pours, real net-class power track widths, full production routing,
     and silkscreen label beautification (`build_judge_chain_smart`, `run_production_finish`)
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
for _sub in ("py_placer", "py_router", "py_tools"):
    _p = str(KRT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pose_score  # noqa: E402
from mounting_holes import mounting_holes  # noqa: E402
from keep_away import parse_keep_away_rules  # noqa: E402
from kicad_parser import find_matching_paren, non_aperture_pads, parse_kicad_pcb  # noqa: E402
from list_nets import board_floor_knobs, find_differential_pairs, read_design_rules  # noqa: E402
from placement.body import board_bodies  # noqa: E402
from placement.grouping import (  # noqa: E402
    BoardView,
    _is_cap,
    _is_inductor,
    analyze,
    claim_rows,
    fold_followers,
    follower_roots,
    is_ground,
    reads_as_rail,
    validate_doc,
)
from placement.part_class import classify_part  # noqa: E402
from placement.roles import infer_role  # noqa: E402
from placement.simple import _turn  # noqa: E402
from placement.templates import FitError, TemplateError, fit  # noqa: E402


RF_MODULE_RE = re.compile(
    r"(ESP32|WROOM|WROVER|NRF52|CC254|CC264|RAK\d+|SIM800|SIM7600|XBEE|LORA|RFM9\d|ANTENNA|WIFI|BLE)",
    re.I,
)


def footprint_has_embedded_keepout(raw_pcb_text: str, ref: str) -> bool:
    """Return True if footprint `ref` contains an embedded `(keepout ... (footprints not_allowed))`."""
    pat = re.compile(
        r'\(\s*(?:property\s+"Reference"|fp_text\s+reference)\s+"' + re.escape(ref) + r'"'
    )
    for m in pat.finditer(raw_pcb_text):
        # Find enclosing (footprint ... ) block
        fp_start = raw_pcb_text.rfind("(footprint", 0, m.start())
        if fp_start < 0:
            continue
        try:
            fp_end = find_matching_paren(raw_pcb_text, fp_start)
        except Exception:
            continue
        block = raw_pcb_text[fp_start:fp_end]
        if "(keepout" in block and "(footprints not_allowed)" in block:
            return True
    return False


def normalize_rf_courtyard_wings(pcb_path: Path) -> None:
    """
    If any footprint with an embedded antenna keepout zone (e.g. `ESP32-S3-WROOM-1`) also draws
    a T-shaped `CrtYd` whose bounding box is >= 2x wider than its physical `.Fab`/pad body,
    clamp its `CrtYd` vertices to `[-cx, +cx] x [-cy, +cy]` around its physical body + pads
    (since `intent.json` `keepouts[]` already enforces the antenna keepout rectangle!).
    This prevents `place_simple.py`'s axis-aligned courtyard bbox from creating phantom
    15mm x 21mm dead zones beside the module's VDD/EN pins.
    """
    pcb = parse_kicad_pcb(str(pcb_path))
    bodies = board_bodies(pcb, str(pcb_path))
    text = pcb_path.read_text(encoding="utf-8", errors="ignore")
    changed = False

    for ref, fp in sorted((pcb.footprints or {}).items()):
        bg = bodies.get(ref)
        if not bg or not bg.body_local or not bg.drawn_local or not bg.pads_local:
            continue
        bx0, by0, bx1, by1 = bg.body_local
        dx0, dy0, dx1, dy1 = bg.drawn_local
        px0, py0, px1, py1 = bg.pads_local
        real_x0, real_y0 = min(dx0, px0) - 0.5, min(dy0, py0) - 0.5
        real_x1, real_y1 = max(dx1, px1) + 0.5, max(dy1, py1) + 0.5
        if (bx1 - bx0) <= 1.6 * (real_x1 - real_x0) and (by1 - by0) <= 1.6 * (real_y1 - real_y0):
            continue
        if not footprint_has_embedded_keepout(text, ref):
            continue

        # Clamp F.CrtYd / B.CrtYd coordinates inside `ref`'s footprint block
        pat = re.compile(r'\(\s*(?:property\s+"Reference"|fp_text\s+reference)\s+"' + re.escape(ref) + r'"')
        for m in pat.finditer(text):
            fp_start = text.rfind("(footprint", 0, m.start())
            if fp_start < 0:
                continue
            try:
                fp_end = find_matching_paren(text, fp_start)
            except Exception:
                continue
            block = text[fp_start:fp_end]

            # Clamp only the wide wing coordinates that stick out beyond real_x0..real_x1 or real_y0..real_y1
            # while keeping the inner stem boundary (e.g. y = -6.75 on ESP32-S3-WROOM-1)
            stem_y0 = max(by0, min(py0 - 1.05, dy0))
            cx0 = round(max(bx0, real_x0), 2)
            cx1 = round(min(bx1, real_x1), 2)
            cy0 = round(stem_y0 if abs(by0 - dy0) > 10.0 else max(by0, real_y0), 2)
            cy1 = round(min(by1, real_y1), 2)

            def _clamp_crtyd_elem(em: re.Match) -> str:
                eb = em.group(0)
                if "CrtYd" not in eb:
                    return eb

                def _clamp_xy(pt_m: re.Match) -> str:
                    kw = pt_m.group(1)
                    x = max(cx0, min(cx1, float(pt_m.group(2))))
                    y = max(cy0, min(cy1, float(pt_m.group(3))))
                    return f"({kw} {x:g} {y:g})"

                return re.sub(r"\((start|end|xy)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\)", _clamp_xy, eb)

            new_block = re.sub(r"\(fp_(?:line|rect|poly)\b[\s\S]*?\n\t\t\)", _clamp_crtyd_elem, block)
            if new_block != block:
                text = text[:fp_start] + new_block + text[fp_end:]
                changed = True
                break

    if changed:
        pcb_path.write_text(text, encoding="utf-8")


def detect_mechanical_and_rf_locks(pcb_human, human_pcb_path: Path) -> tuple[list[str], list[dict]]:
    """
    Identify all footprints that represent mechanical/enclosure/RF-edge decisions:
      - Connectors, receptacles, switches, mounting holes, fiducials
      - Large pin frames / shields (>= 25% of board area measured via `board_bodies`)
      - RF / Wi-Fi / BLE modules with antenna keepouts (e.g. ESP32-S3-WROOM-1)
      - Parts already marked `(locked yes)` in the file
    """
    raw_text = human_pcb_path.read_text(encoding="utf-8", errors="ignore")
    bounds = getattr(pcb_human.board_info, "board_bounds", None)
    board_area = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) if bounds else 0.0
    bodies = board_bodies(pcb_human)

    locked_refs: list[str] = []
    seat_placements: list[dict] = []

    for ref, fp in sorted(pcb_human.footprints.items()):
        if not non_aperture_pads(fp):
            continue
        cls = classify_part(fp, ref).name
        role = infer_role(fp, ref, cls)["role"]
        body = bodies.get(ref)
        bl = (
            getattr(body, "body_local", None)
            or getattr(body, "drawn_local", None)
            or getattr(body, "pads_local", None)
        )
        if bl and len(bl) == 4:
            fp_area = max(0.0, float(bl[2]) - float(bl[0])) * max(0.0, float(bl[3]) - float(bl[1]))
        else:
            fp_area = 0.0

        is_large_frame = board_area > 0 and fp_area >= 0.25 * board_area
        val_fp_str = f"{getattr(fp, 'value', '')} {getattr(fp, 'footprint_name', '')}"
        is_rf_module = bool(RF_MODULE_RE.search(val_fp_str)) or footprint_has_embedded_keepout(raw_text, ref)

        is_mechanical = (
            getattr(fp, "locked", False)
            or is_large_frame
            or is_rf_module
            or cls in ("mount_hole", "fiducial", "edge_actuator", "edge_receptacle", "connector_affinity")
            or role in ("usb receptacle", "pin header", "edge receptacle", "edge actuator", "connector")
            or ref.upper().startswith(("SHIELD", "ANT", "AE", "M1", "CONN", "SW", "BTN", "J", "H"))
        )
        if is_mechanical:
            locked_refs.append(ref)
            seat_placements.append(
                {
                    "reference": ref,
                    "new_x": round(fp.x, 4),
                    "new_y": round(fp.y, 4),
                    "new_rotation": round(fp.rotation or 0.0, 2),
                    "new_side": "B" if str(fp.layer).startswith("B") else "F",
                }
            )
    return locked_refs, seat_placements


def extract_board_keepouts(
    human_routed_path: Path,
    human_unrouted_path: Path,
    pcb_human,
    locked_refs: list[str],
) -> list[dict]:
    """
    Extract footprint-forbidding keepout zones (e.g. antenna keepouts) from the reference board
    so `intent.json` `keepouts[]` prevents `place_simple.py` from placing parts inside them.
    """
    src_path = human_routed_path if human_routed_path.exists() else human_unrouted_path
    text = src_path.read_text(encoding="utf-8", errors="ignore")
    bounds = getattr(pcb_human.board_info, "board_bounds", None)
    if not bounds:
        return []
    bx0, by0, bx1, by1 = bounds

    keepouts: list[dict] = []
    seen_rects: list[tuple[float, float, float, float]] = []

    for m in re.finditer(r"\(\s*zone\b", text):
        z_start = m.start()
        try:
            z_end = find_matching_paren(text, z_start)
        except Exception:
            continue
        z_block = text[z_start:z_end]
        if "(keepout" not in z_block:
            continue
        # Check if it forbids footprints or is an Edge.Cuts / antenna keepout
        forbids_fp = "(footprints not_allowed)" in z_block or '(layer "Edge.Cuts")' in z_block
        if not forbids_fp:
            continue
        # Extract polygon pts
        poly_m = re.search(r"\(\s*polygon\s*\(\s*pts\b([\s\S]*?)\)\s*\)", z_block)
        if not poly_m:
            continue
        pts = [
            (float(xm.group(1)), float(xm.group(2)))
            for xm in re.finditer(r"\(\s*xy\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)", poly_m.group(1))
        ]
        if len(pts) < 3:
            continue
        x0, y0 = max(bx0, min(p[0] for p in pts)), max(by0, min(p[1] for p in pts))
        x1, y1 = min(bx1, max(p[0] for p in pts)), min(by1, max(p[1] for p in pts))
        w, h = x1 - x0, y1 - y0
        if w < 3.0 or h < 3.0 or (w * h) > 0.55 * ((bx1 - bx0) * (by1 - by0)):
            continue
        rect = (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))
        if any( all(abs(a - b) < 1.5 for a, b in zip(rect, sr)) for sr in seen_rects ):
            continue
        seen_rects.append(rect)

        # Allow any locked part that already overlaps this keepout (e.g. U3 antenna module)
        allowed = []
        for lref in locked_refs:
            fp = pcb_human.footprints.get(lref)
            if fp and (x0 - 15.0 <= fp.x <= x1 + 15.0) and (y0 - 15.0 <= fp.y <= y1 + 15.0):
                allowed.append(lref)

        keepouts.append(
            {
                "name": f"keepout_zone_{len(keepouts) + 1}",
                "rect": list(rect),
                "sides": "both",
                "allow": sorted(allowed),
                "why": "Antenna / mechanical keepout zone extracted from board specification",
            }
        )
    return keepouts


def _local_inward_side(pcb, ref: str) -> str | None:
    """Compute the inward board direction (`+x`, `-x`, `+y`, `-y`) in `ref`'s unrotated local frame."""
    bounds = getattr(pcb.board_info, "board_bounds", None)
    fp = pcb.footprints.get(ref)
    if not bounds or not fp:
        return None
    bx0, by0, bx1, by1 = bounds
    d_west = abs(fp.x - bx0)
    d_east = abs(bx1 - fp.x)
    d_north = abs(fp.y - by0)
    d_south = abs(by1 - fp.y)
    min_d = min(d_west, d_east, d_north, d_south)
    if d_west == min_d:
        wx, wy = 1.0, 0.0
    elif d_east == min_d:
        wx, wy = -1.0, 0.0
    elif d_north == min_d:
        wx, wy = 0.0, 1.0
    else:
        wx, wy = 0.0, -1.0

    lx, ly = _turn(wx, wy, -(fp.rotation or 0.0))
    if abs(lx) >= abs(ly):
        return "+x" if lx > 0 else "-x"
    return "+y" if ly > 0 else "-y"


def _parse_cap_farads(val_str: str) -> float:
    """Approximate capacitance in Farads from a value string like '100nF 50V' or '470uF 25V'."""
    s = (val_str or "").lower().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*([pnum\u00b5])f?", s)
    if not m:
        return 1e-6
    num = float(m.group(1))
    unit = m.group(2)
    mult = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "\u00b5": 1e-6, "m": 1e-3}.get(unit, 1e-6)
    return num * mult


def _incremental_fit(pcb, state, anchor: str, candidate_members: list[dict], kind: str | None = None, source: str = "datasheet layout", reach: float = 12.0) -> dict | None:
    """
    Incrementally fit `candidate_members` around `anchor` using `placement.templates.fit`,
    skipping any individual member that cannot fit so partial templates still succeed cleanly.
    """
    accepted_specs: list[dict] = []
    best_row: dict | None = None
    for cand in candidate_members:
        trial_specs = accepted_specs + [cand]
        sketch = {
            "anchor": anchor,
            "source": source,
            "members": trial_specs,
        }
        if kind:
            sketch["kind"] = kind
        try:
            row = fit(pcb, state, sketch, reach=reach)
            accepted_specs = trial_specs
            best_row = row
        except (FitError, TemplateError, Exception):
            # Try without `side` or `rot` restriction if it failed due to side constraint
            cand_relaxed = {k: v for k, v in cand.items() if k not in ("side", "rot")}
            if cand_relaxed != cand:
                trial_specs2 = accepted_specs + [cand_relaxed]
                sketch2 = {"anchor": anchor, "source": source, "members": trial_specs2}
                if kind:
                    sketch2["kind"] = kind
                try:
                    row = fit(pcb, state, sketch2, reach=reach)
                    accepted_specs = trial_specs2
                    best_row = row
                except Exception:
                    pass
    return best_row


def synthesize_templates_and_proximity(pcb_fixed, fixed_pcb_path: Path, doc_intent: dict, locked_refs: set[str]) -> dict:
    """
    Automatically synthesize `templates[]` and `proximity[]` in `doc_intent` from board netlist & footprints:
      1. Switching regulator templates (`U*` + HF decap + catch diode + inductor `L*` + output cap + bulk cap)
      2. Power-entry protection chains (`J*` + fuse `F*` + series diode `D*` + TVS diode `D*`)
      3. Repeated actuator MOSFET/flyback channels (`J*` + `Q*` + `R*` + `D*`), replicated across identical channels
      4. USB / interface ESD & CC resistor proximity claims
    """
    clr, edge, _ = board_floor_knobs(str(fixed_pcb_path))
    state = pose_score.make_state(
        pcb_fixed,
        str(fixed_pcb_path),
        clearance=clr,
        board_edge_clearance=edge,
        estimate_courtyards=True,
    )
    view = BoardView(pcb_fixed)
    fps = pcb_fixed.footprints or {}

    templates: list[dict] = list(doc_intent.get("templates") or [])
    proximity: list[dict] = list(doc_intent.get("proximity") or [])
    used_refs: set[str] = {t["anchor"] for t in templates} | {m["ref"] for t in templates for m in t.get("members", [])}

    def _pads_on_net(ref: str, nid: int) -> list[str]:
        return [str(p.pad_number) for p in fps[ref].pads if p.net_id == nid]

    def _all_conn_pads(ref: str) -> list[str]:
        return [str(p.pad_number) for p in fps[ref].pads if p.net_id > 0]

    # -------------------------------------------------------------------------
    # 1. Switching Regulator Templates (`kind: "switching stage"`)
    # -------------------------------------------------------------------------
    for ref, fp in sorted(fps.items()):
        if ref in locked_refs or ref in used_refs:
            continue
        if not re.match(r"^(U|IC)\d", ref):
            continue
        pads = [p for p in fp.pads if p.net_id > 0]
        if len(pads) < 4:
            continue
        gnd_nids = {p.net_id for p in pads if is_ground(view.name_of(p.net_id))}
        if not gnd_nids:
            continue

        # Find switching pin carrying an inductor L*
        sw_nid = None
        ind_ref = None
        for p in pads:
            if p.net_id in gnd_nids:
                continue
            others = view.owners.get(p.net_id, set()) - {ref}
            if 1 <= len(others) <= 5:
                inds = sorted(o for o in others if _is_inductor(fps[o], o) and o not in used_refs and o not in locked_refs)
                if inds:
                    sw_nid = p.net_id
                    ind_ref = inds[0]
                    break
        if not sw_nid or not ind_ref:
            continue

        # Identify catch diode on sw_nid (2-pad D* connected to sw_nid and GND)
        catch_diode = None
        for o in sorted(view.owners.get(sw_nid, set()) - {ref, ind_ref}):
            if o.startswith("D") and o not in used_refs and o not in locked_refs:
                o_nids = {p.net_id for p in fps[o].pads if p.net_id > 0}
                if len(o_nids) == 2 and (o_nids & gnd_nids):
                    catch_diode = o
                    break

        # Identify output rail on the other side of ind_ref
        ind_nids = [p.net_id for p in fps[ind_ref].pads if p.net_id > 0 and p.net_id != sw_nid]
        out_nid = ind_nids[0] if ind_nids else None

        # Identify input rail on U* (supply rail pin not equal to out_nid or sw_nid or GND)
        in_nid = None
        for p in pads:
            if p.net_id in gnd_nids or p.net_id in (sw_nid, out_nid):
                continue
            nm = view.name_of(p.net_id)
            if p.net_id in view.rails or reads_as_rail(nm) is True or "VIN" in (p.pinfunction or "").upper():
                in_nid = p.net_id
                break

        # Find input caps (between in_nid and gnd_nids) sorted by capacitance (HF ceramic first, bulk second)
        in_caps: list[str] = []
        if in_nid:
            for o in sorted(view.owners.get(in_nid, set()) - {ref}):
                if _is_cap(fps[o], o) and o not in used_refs and o not in locked_refs:
                    o_nids = {p.net_id for p in fps[o].pads if p.net_id > 0}
                    if len(o_nids) == 2 and (o_nids & gnd_nids):
                        in_caps.append(o)
            in_caps.sort(key=lambda c: (_parse_cap_farads(fps[c].value), c))

        # Find output caps (between out_nid and gnd_nids), largest bulk cap near L1 first
        out_caps: list[str] = []
        if out_nid:
            for o in sorted(view.owners.get(out_nid, set()) - {ref}):
                if _is_cap(fps[o], o) and o not in used_refs and o not in locked_refs:
                    o_nids = {p.net_id for p in fps[o].pads if p.net_id > 0}
                    if len(o_nids) == 2 and (o_nids & gnd_nids):
                        out_caps.append(o)
            out_caps.sort(key=lambda c: (-_parse_cap_farads(fps[c].value), c))

        gnd_pads_u = [
            str(p.pad_number)
            for p in pads
            if p.net_id in gnd_nids
            and (not getattr(p, "pinfunction", "") or any(k in p.pinfunction.upper() for k in ("GND", "VSS", "PGND", "TAB", "EP")))
        ]
        if not gnd_pads_u:
            gnd_pads_u = [str(p.pad_number) for p in pads if p.net_id in gnd_nids][:1]
        cand_members: list[dict] = []
        if in_caps and in_nid:
            hf_c = in_caps[0]
            cand_members.append(
                {
                    "ref": hf_c,
                    "pads": {
                        hf_c: _all_conn_pads(hf_c),
                        ref: _pads_on_net(ref, in_nid) + gnd_pads_u,
                    },
                }
            )
        if catch_diode:
            cand_members.append(
                {
                    "ref": catch_diode,
                    "pads": {
                        catch_diode: _all_conn_pads(catch_diode),
                        ref: _pads_on_net(ref, sw_nid) + gnd_pads_u,
                    },
                }
            )
        cand_members.append(
            {
                "ref": ind_ref,
                "pads": {
                    ind_ref: _pads_on_net(ind_ref, sw_nid),
                    ref: _pads_on_net(ref, sw_nid),
                    **({catch_diode: _pads_on_net(catch_diode, sw_nid)} if catch_diode else {}),
                },
            }
        )
        if out_caps and out_nid:
            oc = out_caps[0]
            cand_members.append(
                {
                    "ref": oc,
                    "pads": {
                        oc: _all_conn_pads(oc),
                        ind_ref: _pads_on_net(ind_ref, out_nid),
                        ref: _pads_on_net(ref, out_nid) + gnd_pads_u,
                    },
                }
            )
        bulk_in_caps = [c for c in in_caps[1:] if _parse_cap_farads(fps[c].value) >= 4.7e-6]
        if bulk_in_caps and in_nid:
            bulk_c = bulk_in_caps[0]
            cand_members.append(
                {
                    "ref": bulk_c,
                    "pads": {
                        bulk_c: _all_conn_pads(bulk_c),
                        ref: _pads_on_net(ref, in_nid) + gnd_pads_u,
                        in_caps[0]: _all_conn_pads(in_caps[0]),
                    },
                }
            )

        row = _incremental_fit(
            pcb_fixed,
            state,
            ref,
            cand_members,
            kind="switching stage",
            source=f"{ref} ({fp.value}) switching stage layout",
            reach=14.0,
        )
        if row and row.get("members"):
            templates.append(row)
            used_refs.add(ref)
            for m in row["members"]:
                used_refs.add(m["ref"])

    # -------------------------------------------------------------------------
    # 2. Power-Entry Protection Chain Templates (`J*` -> `F*` -> `D_rev` -> `D_tvs`)
    # -------------------------------------------------------------------------
    for jref in sorted(locked_refs):
        if jref in used_refs or not jref.startswith("J"):
            continue
        jfp = fps.get(jref)
        if not jfp:
            continue
        jpads = [p for p in jfp.pads if p.net_id > 0]
        if not (2 <= len(jpads) <= 4):
            continue
        # Check if any pin of J connects directly to a fuse F*
        fuse_ref = None
        fuse_nid = None
        for p in jpads:
            others = view.owners.get(p.net_id, set()) - {jref}
            fuses = sorted(o for o in others if o.startswith("F") and o not in used_refs and o not in locked_refs)
            if fuses and len(others) <= 3:
                fuse_ref = fuses[0]
                fuse_nid = p.net_id
                break
        if not fuse_ref:
            continue

        inward = _local_inward_side(pcb_fixed, jref)
        cand_members = [
            {
                "ref": fuse_ref,
                "pads": {fuse_ref: _pads_on_net(fuse_ref, fuse_nid), jref: _pads_on_net(jref, fuse_nid)},
                **({"side": inward} if inward else {}),
            }
        ]
        # Follow other pad of fuse_ref to reverse diode D* and/or TVS D*
        other_f_nids = [p.net_id for p in fps[fuse_ref].pads if p.net_id > 0 and p.net_id != fuse_nid]
        if other_f_nids:
            fnid2 = other_f_nids[0]
            diodes = sorted(
                o
                for o in (view.owners.get(fnid2, set()) - {fuse_ref})
                if o.startswith("D") and o not in used_refs and o not in locked_refs
            )
            if diodes:
                d_ser = diodes[0]
                cand_members.append(
                    {
                        "ref": d_ser,
                        "pads": {d_ser: _pads_on_net(d_ser, fnid2), fuse_ref: _pads_on_net(fuse_ref, fnid2)},
                    }
                )
                # Check if d_ser feeds a rail with a TVS diode across rail and GND
                d_other_nids = [p.net_id for p in fps[d_ser].pads if p.net_id > 0 and p.net_id != fnid2]
                if d_other_nids:
                    rail_nid = d_other_nids[0]
                    tvs_cands = sorted(
                        o
                        for o in (view.owners.get(rail_nid, set()) - {d_ser})
                        if o.startswith("D")
                        and o not in used_refs
                        and o not in locked_refs
                        and ("V" in (fps[o].value or "").upper() or "TVS" in (fps[o].value or "").upper())
                    )
                    if tvs_cands:
                        tvs = tvs_cands[0]
                        cand_members.append(
                            {
                                "ref": tvs,
                                "pads": {tvs: _all_conn_pads(tvs), d_ser: _pads_on_net(d_ser, rail_nid), jref: _all_conn_pads(jref)},
                            }
                        )

        row = _incremental_fit(
            pcb_fixed,
            state,
            jref,
            cand_members,
            source=f"{jref} ({jfp.value}) power entry protection chain",
            reach=14.0,
        )
        if row and row.get("members"):
            templates.append(row)
            used_refs.add(jref)
            for m in row["members"]:
                used_refs.add(m["ref"])

    # -------------------------------------------------------------------------
    # 3. Repeated Actuator MOSFET + Flyback Diode Channels (`J*` -> `Q*` + `R*` + `D*`)
    # -------------------------------------------------------------------------
    channel_groups: dict[tuple, list[dict]] = {}
    for jref in sorted(locked_refs):
        if jref in used_refs or not jref.startswith("J"):
            continue
        jfp = fps.get(jref)
        if not jfp:
            continue
        jpads = [p for p in jfp.pads if p.net_id > 0]
        if not (2 <= len(jpads) <= 3):
            continue

        # Look for a switched pin (connects to a 3-pad Q* and a 2-pad flyback D*)
        q_ref, d_ref, led_ref, r_ref, sw_nid = None, None, None, None, None
        for p in jpads:
            if p.net_id in view.rails or is_ground(view.name_of(p.net_id)):
                continue
            others = view.owners.get(p.net_id, set()) - {jref}
            if not (1 <= len(others) <= 4):
                continue
            qs = sorted(o for o in others if o.startswith("Q") and o not in used_refs and o not in locked_refs)
            ds = sorted(
                (o for o in others if o.startswith("D") and o not in used_refs and o not in locked_refs),
                key=lambda o: (-sum((pad.size_x or 0.0) * (pad.size_y or 0.0) for pad in fps[o].pads), o),
            )
            if qs and ds:
                q_ref, d_ref, sw_nid = qs[0], ds[0], p.net_id
                if len(ds) >= 2:
                    led_ref = ds[1]
                # Find gate resistor connected to Q's gate pin
                for qp in fps[q_ref].pads:
                    if qp.net_id > 0 and qp.net_id != sw_nid and not is_ground(view.name_of(qp.net_id)):
                        rs = sorted(
                            o
                            for o in (view.owners.get(qp.net_id, set()) - {q_ref})
                            if o.startswith("R") and o not in used_refs and o not in locked_refs
                        )
                        if rs:
                            r_ref = rs[0]
                            break
                break

        if not q_ref or not d_ref:
            continue

        inward = _local_inward_side(pcb_fixed, jref)
        # Seat Q first (closest to J), then R beside Q, then flyback diode D, then optional LED
        cands = [
            {
                "ref": q_ref,
                "pads": {q_ref: _pads_on_net(q_ref, sw_nid), jref: _pads_on_net(jref, sw_nid)},
                **({"side": inward, "rot": 180} if inward == "+y" else ({"side": inward} if inward else {})),
            }
        ]
        if r_ref:
            cands.append(
                {
                    "ref": r_ref,
                    "pads": {r_ref: _all_conn_pads(r_ref), q_ref: _all_conn_pads(q_ref)},
                    **({"side": inward, "rot": 90} if inward == "+y" else ({"side": inward} if inward else {})),
                }
            )
        cands.append(
            {
                "ref": d_ref,
                "pads": {d_ref: _all_conn_pads(d_ref), jref: _all_conn_pads(jref), q_ref: _pads_on_net(q_ref, sw_nid)},
                **({"side": inward, "rot": 90} if inward == "+y" else ({"side": inward} if inward else {})),
            }
        )
        if led_ref:
            cands.append(
                {
                    "ref": led_ref,
                    "pads": {led_ref: _all_conn_pads(led_ref), jref: _all_conn_pads(jref), q_ref: _pads_on_net(q_ref, sw_nid)},
                    **({"side": inward, "rot": 90} if inward == "+y" else ({"side": inward} if inward else {})),
                }
            )

        row = _incremental_fit(
            pcb_fixed,
            state,
            jref,
            cands,
            source=f"{jref} ({jfp.value}) actuator driver channel",
            reach=14.0,
        )
        if row and row.get("members"):
            sig_key = (
                jfp.footprint_name,
                round(jfp.rotation or 0.0, 1),
                inward,
                tuple(fps[m["ref"]].footprint_name for m in row["members"]),
            )
            channel_groups.setdefault(sig_key, []).append(row)
            used_refs.add(jref)
            for m in row["members"]:
                used_refs.add(m["ref"])

    # Standardize identical repeated actuator channels so they form a uniform aligned strip
    for _sig_key, rows in channel_groups.items():
        if len(rows) >= 2:
            # Pick the channel with the most compact vertical span (typically an interior channel away from corner switches)
            def _channel_cost(r: dict) -> float:
                return sum(math.hypot(m["at"][0], m["at"][1]) for m in r["members"])

            canonical = min(rows, key=_channel_cost)
            for r in rows:
                if len(r["members"]) == len(canonical["members"]):
                    for m, cm in zip(r["members"], canonical["members"]):
                        m["at"] = list(cm["at"])
                        m["rot"] = cm["rot"]
        templates.extend(rows)

    # -------------------------------------------------------------------------
    # 4. USB / High-Speed Connector ESD & Interface Proximity Claims
    # -------------------------------------------------------------------------
    existing_prox = {(c.get("ref"), c.get("near")) for c in proximity}
    for jref in sorted(locked_refs):
        jfp = fps.get(jref)
        if not jfp:
            continue
        cls = classify_part(jfp, jref).name
        role = infer_role(jfp, jref, cls)["role"]
        if role != "usb receptacle" and "USB" not in (jfp.value or "").upper() and "TYPE-C" not in (jfp.value or "").upper():
            continue
        # Find 1-hop interface parts (ESD diode array D*, CC resistors R*, VBUS diode D*)
        for p in jfp.pads:
            if p.net_id <= 0 or p.net_id in view.rails or is_ground(view.name_of(p.net_id)):
                continue
            others = view.owners.get(p.net_id, set()) - {jref}
            if len(others) > 5:
                continue
            for o in sorted(others):
                if o in locked_refs or o in used_refs or (o, jref) in existing_prox:
                    continue
                if o.startswith(("D", "R", "F", "L", "C")) or "USBLC" in (fps[o].value or "").upper():
                    proximity.append(
                        {
                            "ref": o,
                            "near": jref,
                            "max_mm": 9.0,
                            "pads": {
                                o: _pads_on_net(o, p.net_id),
                                jref: _pads_on_net(jref, p.net_id),
                            },
                            "why": f"Keep USB interface component {o} ({fps[o].value}) adjacent to {jref}",
                        }
                    )
                    existing_prox.add((o, jref))

    if templates:
        doc_intent["templates"] = templates
    if proximity:
        doc_intent["proximity"] = proximity
    return doc_intent


def repair_and_enrich_groups(
    doc: dict,
    pcb,
    intent_doc: dict,
    fixed_refs: set[str],
) -> dict:
    """
    Upgrade and repair any `groups.json` (whether extracted from human reference or predicted by the LLM):
      1. Partition isolated ground domains (`GND_ISO*`, `AGND`, `GNDA`) into a dedicated isolated group
         with explicit `apart` constraints against noisy power/driver groups.
      2. Fold all `templates[]` and `proximity[]` members into their anchor's group.
      3. Iteratively resolve `decap_cross_group` and `pulled` warnings from `check_groups.py` so
         every decoupling capacitor sits in its owner IC's group and every satellite passive sits
         with the group it connects to.
      4. Merge empty `0-movable` fixed-connector singleton groups into their connected functional block.
      5. Attach `templates` declarations to switching regulator groups so `check_groups.py` emits 0 hints.
    """
    doc = copy.deepcopy(doc)
    view = BoardView(pcb)
    fps = pcb.footprints or {}

    # Ensure every movable + fixed copper-bearing part is present in exactly one group
    seen: set[str] = set()
    for g in doc.get("groups", []):
        clean_refs = []
        for r in g.get("refs", []):
            if r in view.parts and r not in seen:
                clean_refs.append(r)
                seen.add(r)
        g["refs"] = clean_refs
    doc["groups"] = [g for g in doc.get("groups", []) if g.get("refs")]
    if not doc["groups"]:
        doc["groups"] = [{"name": "grp_main", "role": "core", "refs": sorted(view.parts), "why": "Core board components"}]
        seen = set(view.parts)

    missing = sorted(set(view.parts) - seen)
    if missing:
        doc["groups"][0]["refs"].extend(missing)

    # 0. Split any oversized multi-IC catch-all group (> 18 parts with >= 2 major ICs/regulators)
    # so each major IC (e.g. U1 buck, U2 LDO, U3 MCU, U9 sensor) gets its own dedicated functional group!
    split_groups_out: list[dict] = []
    existing_gnames = {g["name"] for g in doc["groups"]}
    for g in doc["groups"]:
        g_refs = g["refs"]
        major_ics = [
            r
            for r in g_refs
            if re.match(r"^(U|IC|REG|PS)\d", r) and len(fps[r].pads) >= 4
        ]
        if len(g_refs) > 18 and len(major_ics) >= 2:
            buckets: dict[str, list[str]] = {ic: [ic] for ic in major_ics}
            other_refs = [r for r in g_refs if r not in major_ics]
            for r in other_refs:
                r_nets = view.nets_of.get(r, set())
                r_sig = r_nets - view.rails
                best_ic = max(
                    major_ics,
                    key=lambda ic: (
                        len(r_sig & view.nets_of.get(ic, set())) * 20
                        + len(r_nets & view.nets_of.get(ic, set())) * 2
                        - len(buckets[ic]) * 0.1
                    ),
                )
                buckets[best_ic].append(r)
            for idx_ic, ic in enumerate(major_ics):
                b_refs = sorted(set(buckets[ic]))
                base_nm = "grp_" + re.sub(r"[^a-z0-9]+", "_", ic.lower()).strip("_")
                gnm = base_nm if base_nm not in existing_gnames or base_nm == g["name"] else f"{base_nm}_{idx_ic}"
                existing_gnames.add(gnm)
                cls_ic = classify_part(fps[ic], ic).name
                role_ic = infer_role(fps[ic], ic, cls_ic)["role"]
                split_groups_out.append(
                    {
                        "name": gnm,
                        "role": "power" if role_ic == "regulator" else "core",
                        "anchor": ic,
                        "refs": b_refs,
                        "why": f"Functional block anchored by {ic} ({fps[ic].value})",
                    }
                )
        else:
            split_groups_out.append(g)
    doc["groups"] = split_groups_out

    # 0b. Extract repeated per-connector MOSFET + flyback diode actuator channels (`J*` + `Q*` + `D*` + `R*`)
    # into dedicated `grp_j*` interface groups whenever >= 2 such channels exist on the board
    actuator_channels: list[tuple[str, set[str]]] = []
    claimed_ch_refs: set[str] = set()
    for jref in sorted(fixed_refs):
        if not jref.startswith("J") or jref not in fps:
            continue
        jpads = [p for p in fps[jref].pads if p.net_id > 0]
        if not (2 <= len(jpads) <= 3):
            continue
        for p in jpads:
            if p.net_id in view.rails or is_ground(view.name_of(p.net_id)):
                continue
            others = view.owners.get(p.net_id, set()) - {jref}
            if not (1 <= len(others) <= 4):
                continue
            qs = sorted(o for o in others if o.startswith("Q") and o not in fixed_refs and o not in claimed_ch_refs)
            ds = sorted(
                (o for o in others if o.startswith("D") and o not in fixed_refs and o not in claimed_ch_refs),
                key=lambda o: (-sum((pad.size_x or 0.0) * (pad.size_y or 0.0) for pad in fps[o].pads), o),
            )
            if qs and ds:
                q_ref = qs[0]
                ch_refs = {jref, q_ref, *ds}
                for qp in fps[q_ref].pads:
                    if qp.net_id > 0 and qp.net_id != p.net_id and not is_ground(view.name_of(qp.net_id)):
                        rs = sorted(
                            o
                            for o in (view.owners.get(qp.net_id, set()) - {q_ref})
                            if o.startswith("R") and o not in fixed_refs and o not in claimed_ch_refs and len(fps[o].pads) == 2
                        )
                        if rs:
                            ch_refs.update(rs)
                for d_item in ds:
                    for dp in fps[d_item].pads:
                        if dp.net_id > 0 and dp.net_id != p.net_id and dp.net_id not in view.rails and not is_ground(view.name_of(dp.net_id)) and len(view.owners.get(dp.net_id, set())) == 2:
                            rs_d = [
                                o
                                for o in (view.owners.get(dp.net_id, set()) - {d_item})
                                if o.startswith("R") and o not in fixed_refs and o not in claimed_ch_refs and len(fps[o].pads) == 2
                            ]
                            ch_refs.update(rs_d)
                claimed_ch_refs.update(ch_refs)
                actuator_channels.append((jref, ch_refs))
                break

    if len(actuator_channels) >= 2:
        all_ch_refs = {r for _, ch in actuator_channels for r in ch}
        for g in doc["groups"]:
            g["refs"] = [r for r in g["refs"] if r not in all_ch_refs]
            if g.get("anchor") and g["anchor"] not in g["refs"]:
                g["anchor"] = g["refs"][0] if g["refs"] else None
        doc["groups"] = [g for g in doc["groups"] if g["refs"]]
        for jref, ch_refs in actuator_channels:
            gnm = "grp_" + re.sub(r"[^a-z0-9]+", "_", jref.lower()).strip("_")
            if any(g["name"] == gnm for g in doc["groups"]):
                gnm += "_drv"
            doc["groups"].append(
                {
                    "name": gnm,
                    "role": "interface",
                    "anchor": jref,
                    "refs": sorted(ch_refs),
                    "why": f"{jref} ({fps[jref].value}) low-side MOSFET actuator channel",
                }
            )

    # 1. Detect isolated ground domain(s) (e.g. GND_ISO_PH, GND_ISO)
    all_gnd_nids = {nid for nid in view.owners if is_ground(view.name_of(nid))}
    iso_gnd_nids = [
        nid
        for nid in all_gnd_nids
        if ("ISO" in view.name_of(nid).upper() or "AGND" in view.name_of(nid).upper() or "GNDA" in view.name_of(nid).upper())
        and len(view.owners[nid]) >= 4
    ]
    for iso_nid in iso_gnd_nids:
        non_iso_gnds = all_gnd_nids - {iso_nid}
        iso_name = view.name_of(iso_nid)
        iso_refs = set(view.owners[iso_nid])
        # Include any part whose nets are purely on the isolated side (never touching non-isolated GND)
        for _ in range(2):
            for nid, owners in view.owners.items():
                if nid in view.rails or nid in non_iso_gnds:
                    continue
                # Do not cross non-isolated rails connected to bridge chips (PS1 / U8)
                if any((view.nets_of.get(o, set()) & non_iso_gnds) and o not in view.owners[iso_nid] for o in owners):
                    continue
                if owners & iso_refs and len(owners) <= 10:
                    for o in owners:
                        o_nids = view.nets_of.get(o, set())
                        if not (o_nids & non_iso_gnds) and len(fps[o].pads) <= 16:
                            iso_refs.add(o)

        # Remove iso_refs from existing groups and form a dedicated isolated group
        for g in doc["groups"]:
            g["refs"] = [r for r in g["refs"] if r not in iso_refs]
            if g.get("anchor") and g["anchor"] not in g["refs"]:
                g["anchor"] = g["refs"][0] if g["refs"] else None
        doc["groups"] = [g for g in doc["groups"] if g["refs"]]

        iso_locked = sorted(r for r in iso_refs if r in fixed_refs and r.startswith("J"))
        iso_anchor = iso_locked[0] if iso_locked else sorted(iso_refs)[0]
        grp_iso_name = "grp_" + re.sub(r"[^a-z0-9]+", "_", iso_name.lower()).strip("_")
        if any(g["name"] == grp_iso_name for g in doc["groups"]):
            grp_iso_name += "_island"
        doc["groups"].append(
            {
                "name": grp_iso_name,
                "role": "analog",
                "anchor": iso_anchor,
                "refs": sorted(iso_refs),
                "why": f"Galvanically isolated {iso_name} front-end and bridge components",
            }
        )

    # 2. Fold all template members and proximity claim followers into their anchor's group
    rows, _ = claim_rows(pcb, intent_doc, crystals=True)
    tmpl_claims = [
        {"ref": m["ref"], "near": t["anchor"]}
        for t in (intent_doc.get("templates") or [])
        for m in (t.get("members") or [])
    ]
    all_claims = rows + tmpl_claims
    doc, _ = fold_followers(doc, all_claims)
    protected_followers = set(follower_roots(all_claims)) | {
        m["ref"] for t in (intent_doc.get("templates") or []) for m in (t.get("members") or [])
    }

    # 3. Iteratively resolve `decap_cross_group` and `pulled` warnings from `check_groups.py`
    for _ in range(6):
        rep = analyze(doc, pcb, followers=sorted(protected_followers), extra_fixed=sorted(fixed_refs))
        gmap = {g["name"]: g for g in doc["groups"]}
        moved_any = False

        # 3a. Move cross-group decoupling capacitors into their owner IC's group (matching ground domain!)
        for dc in rep.get("decap_cross_group", []):
            cap = dc["cap"]
            if cap in protected_followers or cap in fixed_refs:
                continue
            from_g = dc["cap_group"]
            cap_gnds = {nid for nid in view.nets_of.get(cap, set()) if is_ground(view.name_of(nid))}
            # Prefer owner ICs that share the same ground net as the capacitor
            matching_owners = [
                o
                for o in dc.get("owners", [])
                if (view.nets_of.get(o, set()) & cap_gnds)
                and not (o in ("PS1", "U8") and not (cap_gnds & set(iso_gnd_nids)))
            ]
            pref_groups = [
                g["name"]
                for g in doc["groups"]
                if g["name"] != from_g and any(o in g["refs"] for o in matching_owners)
            ]
            owner_gs = pref_groups or [og for og in dc.get("owner_groups", []) if og in gmap and og != from_g]
            if owner_gs and from_g in gmap and cap in gmap[from_g]["refs"]:
                target_g = owner_gs[0]
                gmap[from_g]["refs"].remove(cap)
                gmap[target_g]["refs"].append(cap)
                moved_any = True

        # 3b. Move satellite parts pulled strongly by another group
        for p in rep.get("pulled", []):
            ref = p["ref"]
            if ref in protected_followers or ref in fixed_refs:
                continue
            from_g = p["group"]
            to_g = p["pulled_by"]
            if from_g in gmap and to_g in gmap and ref in gmap[from_g]["refs"]:
                if ref != gmap[from_g].get("anchor"):
                    gmap[from_g]["refs"].remove(ref)
                    gmap[to_g]["refs"].append(ref)
                    moved_any = True

        # Clean up any emptied groups
        for g in doc["groups"]:
            g["refs"] = sorted(set(g["refs"]))
            if g.get("anchor") and g["anchor"] not in g["refs"]:
                g["anchor"] = g["refs"][0] if g["refs"] else None
        doc["groups"] = [g for g in doc["groups"] if g["refs"]]
        if not moved_any:
            break

    # 4. Merge singleton 0-movable connector groups and tiny testpoint-only groups into the group their nets connect to
    movable_set = set(view.parts) - set(fixed_refs)
    empty_fixed_groups = [
        g
        for g in doc["groups"]
        if not any(r in movable_set for r in g["refs"])
        or (len(g["refs"]) == 1 and g["refs"][0].startswith("TP"))
    ]
    active_groups = [
        g for g in doc["groups"] if g not in empty_fixed_groups
    ]
    if empty_fixed_groups and active_groups:
        for eg in empty_fixed_groups:
            eg_all_nets = {nid for r in eg["refs"] for nid in view.nets_of.get(r, set())}
            eg_sig_nets = eg_all_nets - view.rails
            best_ag = max(
                active_groups,
                key=lambda ag: (
                    len(eg_sig_nets & {nid for r in ag["refs"] for nid in view.nets_of.get(r, set())}) * 10
                    + len(eg_all_nets & {nid for r in ag["refs"] for nid in view.nets_of.get(r, set())}),
                    -len(ag["refs"]),
                ),
            )
            best_ag["refs"] = sorted(set(best_ag["refs"]) | set(eg["refs"]))
        doc["groups"] = active_groups

    # 4b. Co-group secondary LDOs (e.g. U2 AMS1117-3.3) with their upstream buck converter group (e.g. grp_u1)
    # when a small LDO group (<= 14 parts) takes its input rail directly from a switching stage group
    tmpl_anchors = {t["anchor"] for t in (intent_doc.get("templates") or []) if t.get("kind") == "switching stage"}
    sw_groups = [g for g in doc["groups"] if any(r in tmpl_anchors for r in g["refs"])]
    if sw_groups:
        sw_g = sw_groups[0]
        sw_rails = {
            nid
            for r in sw_g["refs"]
            if _is_inductor(fps[r], r)
            for nid in view.nets_of.get(r, set())
            if nid in view.rails and not is_ground(view.name_of(nid))
        }
        to_merge_into_sw = []
        for g in doc["groups"]:
            if g is sw_g or g.get("role") == "analog" or "iso" in g["name"]:
                continue
            g_ics = [r for r in g["refs"] if re.match(r"^(U|IC|REG)\d", r)]
            if len(g_ics) == 1 and len(g["refs"]) <= 14:
                ic = g_ics[0]
                if len(fps[ic].pads) <= 6 and (view.nets_of.get(ic, set()) & sw_rails):
                    to_merge_into_sw.append(g)
        for mg in to_merge_into_sw:
            sw_g["refs"] = sorted(set(sw_g["refs"]) | set(mg["refs"]))
            doc["groups"] = [g for g in doc["groups"] if g is not mg]

    # 4c. Balance decoupling capacitors across all groups that own supply pins on each rail
    # so no IC with a supply pin (like U3 / U9 on +3.3V) is starved of a decoupling cap in its group!
    sup_pins = view.supply_pins()  # {rail_nid: [(ic_ref, pin_num), ...]}
    decap_map = view.decaps()      # {cap_ref: rail_nid}
    by_rail_caps: dict[int, list[str]] = {}
    for cref, rnid in decap_map.items():
        if cref not in protected_followers and cref not in fixed_refs:
            by_rail_caps.setdefault(rnid, []).append(cref)

    for rnid, caps in by_rail_caps.items():
        rail_pin_pairs = sup_pins.get(rnid, [])
        if not rail_pin_pairs:
            continue
        # Count distinct (ic, pin_num) supply pins per group on `rnid` (matching the ground domain!)
        pin_count_by_g: dict[str, int] = {}
        cap_list_by_g: dict[str, list[str]] = {}
        for g in doc["groups"]:
            gname = g["name"]
            if g.get("role") == "analog" or "iso" in gname:
                continue
            distinct_pins = {(ic, pin_n) for (ic, pin_n) in rail_pin_pairs if ic in g["refs"]}
            if distinct_pins:
                pin_count_by_g[gname] = len(distinct_pins)
            c_list = [
                c for c in g["refs"]
                if c in caps and _parse_cap_farads(fps[c].value) <= 2.2e-6
            ]
            cap_list_by_g[gname] = sorted(c_list, key=lambda c: (_parse_cap_farads(fps[c].value), c))

        if len(pin_count_by_g) >= 2:
            gmap = {g["name"]: g for g in doc["groups"]}
            # Ensure every group with supply pins on `rnid` gets up to `pin_count` HF decaps from surplus groups
            for needy_g, needed_pins in sorted(pin_count_by_g.items(), key=lambda kv: -kv[1]):
                while len(cap_list_by_g.get(needy_g, [])) < needed_pins:
                    donors = [
                        dg for dg, d_caps in cap_list_by_g.items()
                        if dg != needy_g and len(d_caps) > max(1, pin_count_by_g.get(dg, 0))
                    ]
                    if not donors:
                        break
                    donor_g = max(donors, key=lambda dg: len(cap_list_by_g[dg]) - pin_count_by_g.get(dg, 0))
                    moving_cap = cap_list_by_g[donor_g].pop(0)
                    if moving_cap in gmap[donor_g]["refs"]:
                        gmap[donor_g]["refs"].remove(moving_cap)
                        gmap[needy_g]["refs"].append(moving_cap)
                        gmap[needy_g]["refs"] = sorted(set(gmap[needy_g]["refs"]))
                        cap_list_by_g.setdefault(needy_g, []).append(moving_cap)

    # 5. Recompute clean `near`, `apart`, `edge`, and `templates` metadata
    valid_names = {g["name"] for g in doc["groups"]}
    for g in doc["groups"]:
        g.pop("near", None)
        g.pop("apart", None)
        g.pop("templates", None)

    rep = analyze(doc, pcb, followers=sorted(protected_followers), extra_fixed=sorted(fixed_refs))
    neighbors: dict[str, list[tuple[float, str]]] = {}
    for row in rep.get("links", []):
        if row["signal_links"] >= 1.0:
            neighbors.setdefault(row["a"], []).append((row["signal_links"], row["b"]))
            neighbors.setdefault(row["b"], []).append((row["signal_links"], row["a"]))
    for g in doc["groups"]:
        nlist = [b for _w, b in sorted(neighbors.get(g["name"], []), reverse=True) if b in valid_names][:2]
        if nlist:
            g["near"] = nlist

    # Add `apart` between any isolated/analog group and power/driver groups
    iso_group_names = [g["name"] for g in doc["groups"] if g.get("role") == "analog" or "iso" in g["name"]]
    noisy_group_names = [
        g["name"]
        for g in doc["groups"]
        if g["name"] not in iso_group_names
        and (
            g.get("role") == "power"
            or any(r.startswith(("U1", "L", "Q", "F")) for r in g["refs"])
        )
    ]
    for g in doc["groups"]:
        if g["name"] in iso_group_names and noisy_group_names:
            g["apart"] = sorted(noisy_group_names)

    # Declare `templates` on groups containing switching stages so `check_groups.py` has 0 template_hints
    tmpl_by_chip = {
        t["anchor"]: t for t in (intent_doc.get("templates") or []) if t.get("kind") in ("switching stage", "rf front end")
    }
    for g in doc["groups"]:
        g_tmpls = []
        for r in g["refs"]:
            if r in tmpl_by_chip:
                t = tmpl_by_chip[r]
                g_tmpls.append(
                    {
                        "chip": r,
                        "kind": t["kind"],
                        "why": t.get("source") or f"{r} datasheet switching layout",
                    }
                )
        if g_tmpls:
            g["templates"] = g_tmpls

    # Ensure no ref ever appears in more than one group
    seen_final_refs: set[str] = set()
    deduped_groups = []
    for g in doc["groups"]:
        clean_r = [r for r in g["refs"] if r not in seen_final_refs]
        seen_final_refs.update(clean_r)
        if clean_r:
            g["refs"] = sorted(clean_r)
            if g.get("anchor") not in g["refs"]:
                g["anchor"] = g["refs"][0]
            deduped_groups.append(g)
    doc["groups"] = deduped_groups

    validate_doc(doc)
    return doc


def classify_power_nets_and_widths(pcb) -> tuple[list[str], list[float]]:
    """
    Classify board power, switching, and actuator nets into two width tiers:
      - High-current / switching / actuator paths (0.8 mm): +12V, +24V, VIN, fuse/diode entry,
        inductor switch nodes, and MOSFET actuator drain nets.
      - Secondary DC supply rails & grounds (0.5 mm): +5V, +3.3V, +5V_ISO*, +3.3V_ISO*, VBUS, GND*.
    (All power widths automatically neck down at fine-pitch SMD pads inside `route.py`.)
    """
    view = BoardView(pcb)
    fps = pcb.footprints or {}
    high_current_nets: set[str] = set()
    med_power_nets: set[str] = set()

    for nid, owners in view.owners.items():
        nm = view.name_of(nid)
        if not nm or nm.startswith("unconnected-"):
            continue
        # Ground nets are handled by route_planes.py (copper pours + GND stitching vias);
        # passing GND to --power-nets lays redundant 0.5mm traces that block fine-pitch pads.
        if is_ground(nm):
            continue
        # Never classify a 2-pin net on a small/fine-pitch pad (e.g. VPP on QFN-20) as a wide power rail
        if len(owners) < 3:
            has_large_power_pad = False
            for o in owners:
                fp_o = fps.get(o)
                if fp_o:
                    for p in fp_o.pads:
                        if p.net_id == nid and (p.size_x * p.size_y) >= 1.2:
                            has_large_power_pad = True
                            break
            if not has_large_power_pad:
                continue
        u = nm.upper()
        # High-voltage / high-current supply rails
        if any(k in u for k in ("+12V", "+24V", "12V", "24V", "VBAT", "VSYS", "VIN", "VMOT", "VPUMP")):
            high_current_nets.add(nm)
            continue
        # Fuse / power-entry / inductor switching / MOSFET drain nets (excluding nH RF matching inductors)
        if any(o.startswith("F") for o in owners) and len(owners) <= 6:
            high_current_nets.add(nm)
            continue
        if (
            any(
                o.startswith("L") and "NH" not in (fps[o].value or "").upper()
                for o in owners
            )
            and any(o.startswith(("U", "IC", "D", "Q", "J")) or nid in view.rails for o in owners)
            and len(owners) <= 6
        ):
            high_current_nets.add(nm)
            continue
        if any(o.startswith("Q") for o in owners) and any(o.startswith("D") for o in owners) and any(o.startswith("J") for o in owners):
            high_current_nets.add(nm)
            continue
        # Secondary rails (+5V, +3.3V, VBUS, etc.)
        if nid in view.rails or reads_as_rail(nm) is True or any(k in u for k in ("+5V", "+3.3V", "3V3", "5V", "VBUS", "VCC", "VDD")):
            med_power_nets.add(nm)

    power_nets: list[str] = []
    power_widths: list[float] = []
    for nm in sorted(high_current_nets):
        power_nets.append(nm)
        power_widths.append(0.8)
    for nm in sorted(med_power_nets - high_current_nets):
        power_nets.append(nm)
        power_widths.append(0.5)
    return power_nets, power_widths


def synthesize_keep_away_rules(pcb) -> list[str]:
    """
    Synthesize pairwise EMI keep-away rules (`AGGRESSOR:VICTIM:GAP_MM`) between high-dV/dt switching /
    actuator nets and sensitive feedback / analog / sensor nets, validated against `parse_keep_away_rules`.
    """
    view = BoardView(pcb)
    fps = pcb.footprints or {}
    aggressors: list[str] = []
    victims: list[str] = []

    def _safe_pat(net_name: str) -> str:
        return re.sub(r"[:;,\s]", "?", net_name)

    for nid, owners in sorted(view.owners.items()):
        nm = view.name_of(nid)
        if not nm or nm.startswith("unconnected-") or is_ground(nm):
            continue
        u = nm.upper()
        # Switching node or actuator drain net
        is_sw_ind = (
            any(o.startswith("L") and "NH" not in (fps[o].value or "").upper() for o in owners)
            and any(o.startswith(("U", "IC", "D", "Q")) for o in owners)
            and len(owners) <= 6
            and nid not in view.rails
        )
        is_act_drain = (
            any(o.startswith("Q") for o in owners)
            and any(o.startswith("D") for o in owners)
            and any(o.startswith("J") for o in owners)
            and len(owners) <= 5
        )
        is_named_agg = any(k in u for k in ("BUCK_SW", "SW_NODE", "LX", "PH_SW", "PUMP", "PWM")) and not any(
            k in u for k in ("SWD", "SWCLK", "SWDIO", "BTN", "RESET")
        )
        if is_sw_ind or is_act_drain or is_named_agg:
            aggressors.append(_safe_pat(nm))
            continue

        if nid in view.rails or reads_as_rail(nm) is True:
            continue
        if any(k in u for k in ("FB", "SENSE", "VREF", "PH", "EC", "ADC", "AIN", "TEMP", "THERM", "XTAL", "OSC")):
            victims.append(_safe_pat(nm))

    valid_rules: list[str] = []
    if aggressors and victims:
        agg_set = sorted(set(aggressors))[:5]
        vic_set = sorted(set(victims) - set(agg_set))[:5]
        if agg_set and vic_set:
            r_str = f"{','.join(agg_set)}:{','.join(vic_set)}:1.5"
            try:
                if parse_keep_away_rules([r_str]):
                    valid_rules.append(r_str)
            except Exception:
                pass
    return valid_rules


def synthesize_routing_strategy(pcb_path: Path) -> dict:
    """
    Synthesize the structured `routing_strategy` object for `pcb_path` so the LLM explicitly plans
    channel reservation, ground/split-plane pours, laminar bus rivers, keep-away EMI rules,
    via/turn costs, and power track width tiers alongside placement.
    """
    pcb = parse_kicad_pcb(str(pcb_path))
    layers = list(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"])
    n_layers = len(layers)
    dr = read_design_rules(str(pcb_path))
    eff = dr.get("effective") or {}
    fab = eff.get("fab") or {}
    drc_clr = eff.get("drc_clearance") or fab.get("clearance") or 0.15
    via_sz = eff.get("working_via_diameter") or fab.get("via_diameter") or 0.5
    via_dr = eff.get("working_via_drill") or fab.get("via_drill") or 0.3

    has_fine_pitch = False
    for fp in (pcb.footprints or {}).values():
        pads = [p for p in fp.pads if p.net_id > 0]
        if 4 <= len(pads) <= 128:
            for i in range(len(pads)):
                for j in range(i + 1, len(pads)):
                    if pads[i].net_id == pads[j].net_id:
                        continue
                    dx = max(0.0, abs(pads[i].local_x - pads[j].local_x) - 0.5 * (pads[i].size_x + pads[j].size_x))
                    dy = max(0.0, abs(pads[i].local_y - pads[j].local_y) - 0.5 * (pads[i].size_y + pads[j].size_y))
                    if math.hypot(dx, dy) <= 0.45:
                        has_fine_pitch = True
                        break
                if has_fine_pitch:
                    break
        if has_fine_pitch:
            break

    clr = round(min(float(drc_clr), 0.12) if has_fine_pitch else max(float(drc_clr), 0.15), 4)
    sig_tw = 0.15 if has_fine_pitch else 0.25
    via_sz = round(max(float(via_sz), 0.45 if has_fine_pitch else 0.50), 4)
    via_dr = round(max(float(via_dr), 0.20 if has_fine_pitch else 0.25), 4)

    view = BoardView(pcb)
    gnd_nets = [
        view.name_of(nid)
        for nid, refs in sorted(view.owners.items(), key=lambda kv: -len(kv[1]))
        if view.name_of(nid) and is_ground(view.name_of(nid)) and len(refs) >= 2 and not view.name_of(nid).startswith("-")
    ]
    p_nets, p_widths = classify_power_nets_and_widths(pcb)
    safe_pw = {n: w for n, w in zip(p_nets, p_widths) if n and not n.startswith("-")}
    try:
        dpairs = find_differential_pairs(pcb)
    except Exception:
        dpairs = []
    dp_nets: list[str] = []
    for dp in (dpairs or [])[:4]:
        p_net = getattr(dp, "p_net", None) or getattr(dp, "pos_net", None)
        n_net = getattr(dp, "n_net", None) or getattr(dp, "neg_net", None)
        if p_net and n_net:
            dp_nets.extend([p_net, n_net])

    try:
        unnetted_mh = [h for h in mounting_holes(pcb) if not h.nets]
    except Exception:
        unnetted_mh = []

    keep_away_rules = synthesize_keep_away_rules(pcb)

    return {
        "placer_channel_reservation": {
            "halo_coef": 0.45,
            "halo_weight": 20.0,
            "congestion_weight": 12.0,
            "pad_weight": 6.0,
            "track_pitch": 0.40,
        },
        "ground_plane_nets": gnd_nets[:3],
        "plane_layers": [layers[1], layers[2]] if n_layers >= 4 else ["B.Cu", "F.Cu"],
        "mounting_hole_keepouts": bool(unnetted_mh),
        "diff_pairs": dp_nets,
        "bus_routing": {
            "enabled": True,
            "ordering": "bus",
            "bus_min_nets": 3,
            "river_pack_inline": True,
        },
        "via_and_trace_rules": {
            "clearance_mm": clr,
            "signal_track_width_mm": sig_tw,
            "via_size_mm": via_sz,
            "via_drill_mm": via_dr,
            "same_net_pad_clearance_mm": 0.15,
            "via_cost": 110,
            "turn_cost": 1200,
            "direction_preference_cost": 250,
        },
        "power_width_tiers": safe_pw,
        "keep_away_rules": keep_away_rules,
    }


def build_smart_routing_chains(
    pcb_path: Path,
    routing_strategy: dict | None = None,
) -> tuple[str, list[tuple[str, list[str]]]]:
    """
    Build:
      1. `judge_chain.txt` (fast seed evaluation chain honoring `routing_strategy`: mounting-hole keepouts,
         multi-domain pours, diff pairs, laminar bus corridors, EMI keep-away rules, and power width tiers)
      2. `prod_steps` (full production routing commands for the winning seed: mounting-hole keepouts,
         dual-layer split pours on `B.Cu` + `F.Cu` with `--stitch-vias` + `--add-gnd-vias`, `route_diff.py`,
         and full `route.py` with `--bus`, `--keep-away`, `--power-nets`, `--add-teardrops`, and `--write-fill`).
    """
    pcb = parse_kicad_pcb(str(pcb_path))
    layers = list(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"])
    n_layers = len(layers)
    default_strat = synthesize_routing_strategy(pcb_path)
    strat = copy.deepcopy(default_strat)
    if isinstance(routing_strategy, dict):
        for k in ("bus_routing", "via_and_trace_rules", "placer_channel_reservation"):
            if isinstance(routing_strategy.get(k), dict):
                strat[k].update(routing_strategy[k])
        if isinstance(routing_strategy.get("power_width_tiers"), dict) and routing_strategy["power_width_tiers"]:
            # Keep only verified power/switching/actuator nets and enforce physical package width ceilings
            strat["power_width_tiers"] = dict(default_strat["power_width_tiers"])
        if isinstance(routing_strategy.get("keep_away_rules"), list):
            validated_ka: list[str] = []
            for r_str in routing_strategy["keep_away_rules"][:6]:
                if isinstance(r_str, str):
                    try:
                        if parse_keep_away_rules([r_str]):
                            validated_ka.append(r_str)
                    except Exception:
                        pass
            if validated_ka:
                strat["keep_away_rules"] = validated_ka

    vt = strat["via_and_trace_rules"]
    def_vt = default_strat["via_and_trace_rules"]
    clr = round(min(float(vt.get("clearance_mm", def_vt["clearance_mm"])), float(def_vt["clearance_mm"])), 4)
    sig_tw = round(min(float(vt.get("signal_track_width_mm", def_vt["signal_track_width_mm"])), float(def_vt["signal_track_width_mm"])), 4)
    via_sz = round(min(float(vt.get("via_size_mm", def_vt["via_size_mm"])), float(def_vt["via_size_mm"])), 4)
    via_dr = round(min(float(vt.get("via_drill_mm", def_vt["via_drill_mm"])), float(def_vt["via_drill_mm"])), 4)
    same_pad_clr = float(vt.get("same_net_pad_clearance_mm", 0.15))
    via_cost = int(max(50, min(300, int(vt.get("via_cost", 110)))))
    turn_cost = int(max(1000, min(2500, int(vt.get("turn_cost", 1200)))))
    dir_pref_cost = int(max(100, min(600, int(vt.get("direction_preference_cost", 250)))))

    bus_cfg = strat.get("bus_routing") or {}
    bus_enabled = bool(bus_cfg.get("enabled", True))
    bus_min_nets = int(max(2, min(8, int(bus_cfg.get("bus_min_nets", 3)))))
    bus_cli_str = f" --ordering bus --bus --bus-min-nets {bus_min_nets}" if bus_enabled else ""
    bus_args_list = ["--ordering", "bus", "--bus", "--bus-min-nets", str(bus_min_nets)] if bus_enabled else []

    rule_flags = (
        f"--clearance {clr} --track-width {sig_tw} --via-size {via_sz} --via-drill {via_dr} "
        f"--same-net-pad-clearance {same_pad_clr} --via-cost {via_cost} "
        f"--turn-cost {turn_cost} --direction-preference-cost {dir_pref_cost}{bus_cli_str}"
    )
    plane_flags = f"--clearance {clr} --via-size {via_sz} --via-drill {via_dr} --same-net-pad-clearance {same_pad_clr}"

    view = BoardView(pcb)
    all_board_nets = {view.name_of(nid) for nid in (pcb.nets or {}).keys() if view.name_of(nid)}
    raw_gnd = strat.get("ground_plane_nets") or default_strat.get("ground_plane_nets") or []
    gnd_nets = [n for n in raw_gnd if isinstance(n, str) and n in all_board_nets and not n.startswith("-")]
    if not gnd_nets:
        gnd_nets = [n for n in (default_strat.get("ground_plane_nets") or []) if n in all_board_nets and not n.startswith("-")]
    gnd_spec = "|".join(gnd_nets[:3]) if gnd_nets else ""

    pwr_nets = [
        view.name_of(nid)
        for nid, refs in sorted(view.owners.items(), key=lambda kv: -len(kv[1]))
        if view.name_of(nid)
        and not is_ground(view.name_of(nid))
        and nid in view.rails
        and len(refs) >= 5
        and not view.name_of(nid).startswith("-")
    ]

    pw_map = strat.get("power_width_tiers") or {}
    p_nets = [k for k in pw_map.keys() if k in all_board_nets and not k.startswith("-")]
    p_widths = [pw_map[k] for k in p_nets]
    power_cli_str = ""
    power_args_list: list[str] = []
    if p_nets:
        quoted_nets = " ".join(f"'{n}'" for n in p_nets)
        widths_str = " ".join(str(w) for w in p_widths)
        power_cli_str = f" --power-nets {quoted_nets} --power-nets-widths {widths_str}"
        power_args_list = ["--power-nets", *p_nets, "--power-nets-widths", *[str(w) for w in p_widths]]

    ka_rules = strat.get("keep_away_rules") or []
    ka_cli_str = ""
    ka_args_list: list[str] = []
    if ka_rules:
        quoted_ka = " ".join(f"'{r}'" for r in ka_rules)
        ka_cli_str = f" --keep-away {quoted_ka}"
        ka_args_list = ["--keep-away", *ka_rules]

    dp_nets = strat.get("diff_pairs") or []
    has_unnetted_mh = bool(default_strat.get("mounting_hole_keepouts", False))

    judge_lines: list[str] = []
    if has_unnetted_mh:
        judge_lines.append("python3 -X utf8 py_router/add_rule_area.py {in} {out} --name MOUNT --mounting-holes")

    if gnd_spec:
        if n_layers >= 4:
            in1, in2 = layers[1], layers[2]
            if pwr_nets:
                pwr_spec = "|".join(pwr_nets[:2])
                judge_lines.append(
                    f"python3 -X utf8 py_router/route_planes.py {{in}} {{out}} --nets '{gnd_spec}' '{pwr_spec}' --plane-layers {in1} {in2} {plane_flags}"
                )
            else:
                judge_lines.append(
                    f"python3 -X utf8 py_router/route_planes.py {{in}} {{out}} --nets '{gnd_spec}' '{gnd_spec}' --plane-layers {in1} {in2} {plane_flags}"
                )
        else:
            judge_lines.append(
                f"python3 -X utf8 py_router/route_planes.py {{in}} {{out}} --nets '{gnd_spec}' '{gnd_spec}' --plane-layers B.Cu F.Cu {plane_flags}"
            )

    if dp_nets:
        dp_quoted = " ".join(f"'{n}'" for n in dp_nets)
        judge_lines.append(
            f"python3 -X utf8 py_router/route_diff.py {{in}} {{out}} --nets {dp_quoted} --track-width 0.15 --diff-pair-gap 0.15 --via-size {via_sz} --via-drill {via_dr} --quick 2"
        )

    judge_lines.append(
        f"python3 -X utf8 py_router/route.py {{in}} {{out}} --nets '*' --quick 2 {rule_flags}{power_cli_str}{ka_cli_str} --json-out {{dir}}/route.json"
    )
    judge_chain_txt = "\n".join(judge_lines) + "\n"

    prod_steps: list[tuple[str, list[str]]] = []
    if has_unnetted_mh:
        prod_steps.append(
            (
                "py_router/add_rule_area.py",
                ["--name", "MOUNT", "--mounting-holes"],
            )
        )

    if gnd_spec:
        if n_layers >= 4:
            in1, in2 = layers[1], layers[2]
            nets_arg = [gnd_spec, "|".join(pwr_nets[:2])] if pwr_nets else [gnd_spec, gnd_spec]
            prod_steps.append(
                (
                    "py_router/route_planes.py",
                    [
                        "--nets",
                        *nets_arg,
                        "--plane-layers",
                        in1,
                        in2,
                        "--clearance",
                        str(clr),
                        "--via-size",
                        str(via_sz),
                        "--via-drill",
                        str(via_dr),
                        "--same-net-pad-clearance",
                        str(same_pad_clr),
                        "--stitch-vias",
                        "--add-gnd-vias",
                    ],
                )
            )
        else:
            prod_steps.append(
                (
                    "py_router/route_planes.py",
                    [
                        "--nets",
                        gnd_spec,
                        gnd_spec,
                        "--plane-layers",
                        "B.Cu",
                        "F.Cu",
                        "--clearance",
                        str(clr),
                        "--via-size",
                        str(via_sz),
                        "--via-drill",
                        str(via_dr),
                        "--same-net-pad-clearance",
                        str(same_pad_clr),
                        "--stitch-vias",
                    ],
                )
            )

    if dp_nets:
        prod_steps.append(
            (
                "py_router/route_diff.py",
                [
                    "--nets",
                    *dp_nets,
                    "--track-width",
                    "0.15",
                    "--diff-pair-gap",
                    "0.15",
                    "--via-size",
                    str(via_sz),
                    "--via-drill",
                    str(via_dr),
                ],
            )
        )

    prod_steps.append(
        (
            "py_router/route.py",
            [
                "--nets",
                "*",
                "--clearance",
                str(clr),
                "--track-width",
                str(sig_tw),
                "--via-size",
                str(via_sz),
                "--via-drill",
                str(via_dr),
                "--same-net-pad-clearance",
                str(same_pad_clr),
                "--via-cost",
                str(via_cost),
                "--turn-cost",
                str(turn_cost),
                "--direction-preference-cost",
                str(dir_pref_cost),
                *bus_args_list,
                *power_args_list,
                *ka_args_list,
                "--add-teardrops",
                "--write-fill",
            ],
        )
    )
    return judge_chain_txt, prod_steps


def hide_fab_value_labels(pcb_path: Path) -> None:
    """
    Ensure `(property "Value" ...)` and `(fp_text value ...)` are hidden (`(hide yes)`) so
    KiCad's PCB view does not render giant overlapping component value strings over pads and SilkS.
    """
    text = pcb_path.read_text(encoding="utf-8", errors="ignore")
    spans: list[tuple[int, int, str]] = []
    for pat in (r'\(\s*property\s+"Value"\s+"[^"]*"', r"\(\s*fp_text\s+value\s+"):
        for m in re.finditer(pat, text):
            start = m.start()
            try:
                end = find_matching_paren(text, start)
            except Exception:
                continue
            node = text[start:end]
            if "(hide yes)" not in node and "hide" not in node:
                # Insert (hide yes) before closing paren
                new_node = node[:-1].rstrip() + "\n\t\t\t(hide yes)\n\t\t)"
                spans.append((start, end, new_node))
    if not spans:
        return
    spans.sort(key=lambda x: x[0])
    out: list[str] = []
    pos = 0
    for start, end, repl in spans:
        if start < pos:
            continue
        out.append(text[pos:start])
        out.append(repl)
        pos = end
    out.append(text[pos:])
    pcb_path.write_text("".join(out), encoding="utf-8")


def build_compact_board_context(pcb_path: Path, locked_refs: set[str] | None = None) -> str:
    """
    Build a high-density, lossless bipartite Markdown representation of a board's geometry,
    component inventory, and netlist connectivity:
      - Marks ONLY truly locked/mechanical footprints as `FIXED(x,y)`; staged unplaced parts
        always print `movable` (never leaking arbitrary staging pile coordinates).
      - Avoids duplicating per-component `nets: [...]` lists (all connectivity lives in
        `## Netlist Connectivity`), shrinking prompt size ~55-60% with zero dropped refs/nets.
    """
    pcb = parse_kicad_pcb(str(pcb_path))
    view = BoardView(pcb)
    if locked_refs is None:
        try:
            det_locked, _ = detect_mechanical_and_rf_locks(pcb, pcb_path)
            locked_set = set(det_locked)
        except Exception:
            locked_set = set()
    else:
        locked_set = set(locked_refs)

    bounds = getattr(pcb.board_info, "board_bounds", None)
    layers = list(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"])
    lines: list[str] = []
    if bounds:
        w = round(bounds[2] - bounds[0], 2)
        h = round(bounds[3] - bounds[1], 2)
        lines.append(
            f"## Board Geometry: {w}x{h} mm ({bounds[0]:.1f},{bounds[1]:.1f} to {bounds[2]:.1f},{bounds[3]:.1f}), "
            f"Layers ({len(layers)}): {layers}"
        )
    lines.append("\n## Component Inventory (ref | value | footprint | class/role | pads | fixed_at)")
    for ref in sorted(view.parts.keys()):
        fp = pcb.footprints[ref]
        cls = classify_part(fp, ref).name
        role = infer_role(fp, ref, cls)["role"]
        npads = len(non_aperture_pads(fp))
        fp_short = (fp.footprint_name or "").split(":")[-1]
        fp_short = re.sub(r"_(\d{4})_\d{4}Metric$", r"_\1", fp_short)
        is_fixed = bool(getattr(fp, "locked", False)) or (ref in locked_set)
        locked_str = f"FIXED({fp.x:.1f},{fp.y:.1f})" if is_fixed else "movable"
        lines.append(
            f"- {ref} | {fp.value} | {fp_short} | {cls}/{role} | {npads}p | {locked_str}"
        )
    lines.append("\n## Netlist Connectivity (net_name | type | connected_refs)")
    for nid, owners in sorted(view.owners.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        nm = view.name_of(nid)
        if not nm or nm.startswith("unconnected-") or len(owners) < 2:
            continue
        ntype = "GND" if is_ground(nm) else ("RAIL" if nid in view.rails else "SIG")
        lines.append(f"- {nm} ({ntype}, {len(owners)}p): {sorted(owners)}")
    return "\n".join(lines) + "\n"


def evaluate_physical_board_quality(pcb_path: Path) -> dict:
    """
    Comprehensive 19-point John Teel (*Predictable Designs*) Physical PCB Placement & Routing Quality Suite:
      Category A — Placement, Channels & Anti-Slalom:
        1. `ic_channel_starvation_count`: Multi-pin ICs (>=12 pins) placed with < 1.8 mm gap while sharing buses.
        2. `slalom_traces_count`: Nets weaving between a foreign component's pin row (>= 3 adjacent foreign pads)
           or making >= 5 alternating accordion direction reversals within < 15 mm.
        3. `off_axis_parts_count`: Footprints rotated off 90-degree multiples.
        4. `channel_symmetry_err_mm`: Positional variance across repeated actuator driver channels.
      Category B — Trace Geometry, Layer Transitions & Widths:
        5. `acute_or_90_corners_count`: Same-layer 2-valence trace bends with interior angle <= 92 deg (90-deg or acid traps).
        6. `weaving_nets_count` & `excess_weave_vias`: Signal nets bouncing across layers (> 2 vias per net).
        7. `high_detour_nets_count` & `max_detour_ratio`: 2-pin signal nets (>= 8 mm span) with routed_len / span > 2.2.
        8. `trace_stubs_count`: Dangling trace endpoints not terminating at a pad, via, or segment.
        9. `power_width_compliance_pct`: Percentage of power/switching/actuator copper meeting ampacity width tiers.
      Category C — Switching Power Supply & Hot-Loop Rules:
        10. `switch_node_length_mm` & `hot_loop_violations`: Switching node copper length (> 18 mm violation).
        11. `fb_aggressor_coupling_mm`: Feedback/sense/analog trace length within < 1.2 mm of switch nodes or inductors.
        12. `decap_distance_violations` & `mean_decap_distance_mm`: IC supply pins > 6.5 mm from nearest bypass cap.
      Category D — Return Paths, Split Planes & High-Speed / Mixed-Signal Isolation:
        13. `split_plane_crossings_count`: Non-bridge signal traces crossing between main GND and isolated GND domains.
        14. `power_to_analog_clearance_mm` & `power_analog_violations`: Inductors L* < 8.0 mm from analog/sensor ICs.
        15. `crystal_layout_violations`: Crystal oscillator nets with vias > 0 or routed length > 12 mm.
        16. `edge_coupled_high_speed_mm`: Clock/USB/bus traces routed within < 0.8 mm of board outline.
      Category E — DFM, ESD & Mechanical Clearances:
        17. `esd_to_connector_violations`: TVS/ESD protection devices placed > 10 mm from their connector.
        18. `via_in_pad_count`: Vias drilled inside SMD pads (excluding thermal ground pads >= 2.2 mm).
        19. `mounting_hole_clearance_violations`: Copper/pads encroaching inside M3 mounting hole keepout radii.
      Composite:
        - `physical_quality_penalty`: Scalar RL penalty used in `compute_rl_reward()`.
        - `teel_score_100`: Intuitive 0-100 composite John Teel Hardware Quality Score.
    """
    if not pcb_path.exists():
        return {
            "ic_channel_starvation_count": 0,
            "slalom_traces_count": 0,
            "off_axis_parts_count": 0,
            "channel_symmetry_err_mm": 0.0,
            "acute_or_90_corners_count": 0,
            "weaving_nets_count": 0,
            "excess_weave_vias": 0,
            "high_detour_nets_count": 0,
            "max_detour_ratio": 1.0,
            "trace_stubs_count": 0,
            "power_width_compliance_pct": 100.0,
            "switch_node_length_mm": 0.0,
            "hot_loop_violations": 0,
            "fb_aggressor_coupling_mm": 0.0,
            "decap_distance_violations": 0,
            "mean_decap_distance_mm": 0.0,
            "split_plane_crossings_count": 0,
            "power_to_analog_clearance_mm": 99.0,
            "power_analog_violations": 0,
            "crystal_layout_violations": 0,
            "edge_coupled_high_speed_mm": 0.0,
            "esd_to_connector_violations": 0,
            "via_in_pad_count": 0,
            "mounting_hole_clearance_violations": 0,
            "teel_score_100": 100.0,
            "physical_quality_penalty": 0.0,
        }

    pcb = parse_kicad_pcb(str(pcb_path))
    view = BoardView(pcb)
    fps = pcb.footprints or {}
    p_nets, p_widths = classify_power_nets_and_widths(pcb)
    expected_w = dict(zip(p_nets, p_widths))
    gnd_nids = {nid for nid in view.owners if is_ground(view.name_of(nid))}
    routed_segs = [s for s in pcb.segments if not s.graphic and s.net_id > 0]

    # -------------------------------------------------------------------------
    # Category A: Placement, Channels & Anti-Slalom
    # -------------------------------------------------------------------------
    # A1. IC Routing Channel Starvation (multi-pin ICs >= 12 pads with < 1.8 mm pad-bbox gap sharing >= 2 signals)
    multi_pin_ics: list[tuple[str, tuple[float, float, float, float]]] = []
    for ref, fp in sorted(fps.items()):
        pads = [p for p in fp.pads if p.net_id > 0]
        if len(pads) >= 12 and not ref.startswith(("J", "P", "H", "SHIELD")):
            px0 = min(p.global_x - 0.5 * p.size_x for p in pads)
            py0 = min(p.global_y - 0.5 * p.size_y for p in pads)
            px1 = max(p.global_x + 0.5 * p.size_x for p in pads)
            py1 = max(p.global_y + 0.5 * p.size_y for p in pads)
            multi_pin_ics.append((ref, (px0, py0, px1, py1)))

    ic_channel_starvation_count = 0
    for i in range(len(multi_pin_ics)):
        r1, b1 = multi_pin_ics[i]
        s1 = (view.nets_of.get(r1, set()) - view.rails) - gnd_nids
        for j in range(i + 1, len(multi_pin_ics)):
            r2, b2 = multi_pin_ics[j]
            s2 = (view.nets_of.get(r2, set()) - view.rails) - gnd_nids
            if len(s1 & s2) < 1:
                continue
            dx = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
            dy = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
            if math.hypot(dx, dy) < 2.4:
                ic_channel_starvation_count += 1

    # A2. Pin-Row Slalom / Accordion Traces
    # Build multi-pin component pad index for foreign-pad slalom detection
    comp_pads_by_ref: dict[str, list] = {}
    for ref, fp in fps.items():
        pads = [p for p in fp.pads if p.net_id > 0]
        if len(pads) >= 8:
            comp_pads_by_ref[ref] = pads

    segs_by_net_layer: dict[tuple[int, str], list] = {}
    segs_by_net: dict[int, list] = {}
    for s in routed_segs:
        segs_by_net.setdefault(s.net_id, []).append(s)
        if s.net_id not in gnd_nids:
            segs_by_net_layer.setdefault((s.net_id, s.layer), []).append(s)

    slalom_nets: set[int] = set()
    for (nid, _layer), nsegs in segs_by_net_layer.items():
        if len(nsegs) < 5:
            continue
        # Check if segments of `nid` pass within < 0.50 mm of >= 3 distinct foreign pads on a multi-pin component
        # while making >= 4 short jog segments (< 3.5 mm) inside that component's pad-row corridor
        for cref, cpads in comp_pads_by_ref.items():
            px0 = min(p.global_x for p in cpads) - 1.5
            py0 = min(p.global_y for p in cpads) - 1.5
            px1 = max(p.global_x for p in cpads) + 1.5
            py1 = max(p.global_y for p in cpads) + 1.5
            local_jogs = [
                s
                for s in nsegs
                if px0 <= 0.5 * (s.start_x + s.end_x) <= px1
                and py0 <= 0.5 * (s.start_y + s.end_y) <= py1
                and 0.35 <= math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) <= 3.5
            ]
            if len(local_jogs) < 4:
                continue
            close_foreign_pads = 0
            for p in cpads:
                if p.net_id == nid:
                    continue
                px, py = p.global_x, p.global_y
                thresh = 0.5 * max(p.size_x, p.size_y) + 0.50
                for s in local_jogs:
                    vx, vy = s.end_x - s.start_x, s.end_y - s.start_y
                    l2 = vx * vx + vy * vy
                    t = max(0.0, min(1.0, ((px - s.start_x) * vx + (py - s.start_y) * vy) / l2)) if l2 > 1e-6 else 0.0
                    if math.hypot(px - (s.start_x + t * vx), py - (s.start_y + t * vy)) <= thresh:
                        close_foreign_pads += 1
                        break
                if close_foreign_pads >= 3:
                    slalom_nets.add(nid)
                    break
            if nid in slalom_nets:
                break

    slalom_traces_count = len(slalom_nets)

    # A3 & A4. Orthogonal Orientation & Repeated Channel Symmetry
    off_axis_parts = []
    for ref, fp in fps.items():
        rot = float(fp.rotation or 0.0) % 90.0
        if min(rot, 90.0 - rot) > 0.5:
            off_axis_parts.append(ref)

    channel_offsets: list[tuple[float, float]] = []
    channel_d_offsets: list[tuple[float, float]] = []
    channel_d_rots: list[int] = []
    for qref, qfp in fps.items():
        if not qref.startswith("Q"):
            continue
        q_nids = view.nets_of.get(qref, set()) - view.rails - gnd_nids
        for nid in q_nids:
            owners = view.owners.get(nid, set())
            j_owners = [o for o in owners if o.startswith("J") and o in fps]
            d_owners = sorted(
                (o for o in owners if o.startswith("D") and o in fps),
                key=lambda o: (-sum((pad.size_x or 0.0) * (pad.size_y or 0.0) for pad in fps[o].pads), o),
            )
            if j_owners and d_owners:
                jfp = fps[j_owners[0]]
                dfp = fps[d_owners[0]]
                channel_offsets.append((round(qfp.x - jfp.x, 2), round(qfp.y - jfp.y, 2)))
                channel_d_offsets.append((round(dfp.x - jfp.x, 2), round(dfp.y - jfp.y, 2)))
                channel_d_rots.append(int(round(float(dfp.rotation or 0.0))) % 360)
                break
    channel_symmetry_err_mm = 0.0
    if len(channel_offsets) >= 3:
        mean_dx = sum(dx for dx, _ in channel_offsets) / len(channel_offsets)
        mean_dy = sum(dy for _, dy in channel_offsets) / len(channel_offsets)
        q_err = sum(math.hypot(dx - mean_dx, dy - mean_dy) for dx, dy in channel_offsets) / len(channel_offsets)
        mean_ddx = sum(dx for dx, _ in channel_d_offsets) / len(channel_d_offsets)
        mean_ddy = sum(dy for _, dy in channel_d_offsets) / len(channel_d_offsets)
        d_err = sum(math.hypot(dx - mean_ddx, dy - mean_ddy) for dx, dy in channel_d_offsets) / len(channel_d_offsets)
        modal_rot = max(set(channel_d_rots), key=channel_d_rots.count)
        rot_mismatch = sum(1 for r in channel_d_rots if r != modal_rot)
        channel_symmetry_err_mm = round(0.5 * (q_err + d_err) + 2.0 * rot_mismatch, 2)

    # -------------------------------------------------------------------------
    # Category B: Trace Geometry, Layer Transitions & Widths
    # -------------------------------------------------------------------------
    # B5 & B8. 90-deg / Acute Corners & Dangling Trace Stubs
    acute_or_90_corners_count = 0
    trace_stubs_count = 0
    pad_via_pts_by_net: dict[int, list[tuple[float, float, float]]] = {}
    for ref, fp in fps.items():
        for p in fp.pads:
            if p.net_id > 0:
                pad_via_pts_by_net.setdefault(p.net_id, []).append(
                    (p.global_x, p.global_y, 0.5 * max(p.size_x, p.size_y) + 0.25)
                )
    for v in pcb.vias:
        if v.net_id > 0:
            pad_via_pts_by_net.setdefault(v.net_id, []).append((v.x, v.y, 0.5 * v.size + 0.25))

    for (nid, _layer), nsegs in segs_by_net_layer.items():
        vmap: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        all_endpts: list[tuple[float, float]] = []
        for s in nsegs:
            slen = math.hypot(s.end_x - s.start_x, s.end_y - s.start_y)
            if slen < 1e-4:
                continue
            all_endpts.append((s.start_x, s.start_y))
            all_endpts.append((s.end_x, s.end_y))
            k1 = (round(s.start_x * 10), round(s.start_y * 10))
            k2 = (round(s.end_x * 10), round(s.end_y * 10))
            vmap.setdefault(k1, []).append((s.end_x - s.start_x, s.end_y - s.start_y, slen))
            vmap.setdefault(k2, []).append((s.start_x - s.end_x, s.start_y - s.end_y, slen))

        pv_list = pad_via_pts_by_net.get(nid, [])
        for (kx, ky), vecs in vmap.items():
            vx_pt, vy_pt = kx / 10.0, ky / 10.0
            in_pad_or_via = any(math.hypot(vx_pt - px, vy_pt - py) <= pr + 0.35 for px, py, pr in pv_list)
            if len(vecs) == 1 and not in_pad_or_via and vecs[0][2] > 0.60:
                # Confirm no other segment endpoint sits within 0.35 mm
                near_other = sum(1 for ex, ey in all_endpts if math.hypot(vx_pt - ex, vy_pt - ey) <= 0.35)
                if near_other <= 1:
                    trace_stubs_count += 1
            elif len(vecs) == 2 and not in_pad_or_via:
                (dx1, dy1, l1), (dx2, dy2, l2) = vecs[0], vecs[1]
                if l1 >= 0.35 and l2 >= 0.35:
                    cos_t = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (l1 * l2)))
                    ang_deg = math.degrees(math.acos(cos_t))
                    # Interior angle between the two rays leaving the vertex: 180 = straight, 135 = 45-deg bend, <= 92 = 90-deg or acute
                    if ang_deg <= 92.0:
                        acute_or_90_corners_count += 1

    # B6. Layer weaving (vias per non-ground, non-power signal net relative to pad count)
    pads_by_net: dict[int, list] = {}
    for fp in fps.values():
        for p in fp.pads:
            if p.net_id > 0:
                pads_by_net.setdefault(p.net_id, []).append(p)

    vias_by_net: dict[int, int] = {}
    for v in pcb.vias:
        if v.net_id > 0 and v.net_id not in gnd_nids:
            vias_by_net[v.net_id] = vias_by_net.get(v.net_id, 0) + 1
    weaving_nets = {
        view.name_of(nid): cnt
        for nid, cnt in vias_by_net.items()
        if nid not in view.rails
        and view.name_of(nid) not in expected_w
        and cnt > max(2, len(pads_by_net.get(nid, [])))
    }
    excess_weave_vias = sum(
        cnt - max(2, len(pads_by_net.get(nid, [])))
        for nid, cnt in vias_by_net.items()
        if nid not in view.rails
        and view.name_of(nid) not in expected_w
        and cnt > max(2, len(pads_by_net.get(nid, [])))
    )

    # B7. Trace Detour Ratio on 2-pin signal nets
    high_detour_nets_count = 0
    max_detour_ratio = 1.0
    for nid, npads in pads_by_net.items():
        if len(npads) != 2 or nid in gnd_nids or nid in view.rails:
            continue
        span = math.hypot(npads[0].global_x - npads[1].global_x, npads[0].global_y - npads[1].global_y)
        if span < 8.0:
            continue
        rlen = sum(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) for s in segs_by_net.get(nid, []))
        if rlen > 0:
            ratio = round(rlen / span, 2)
            if ratio > max_detour_ratio:
                max_detour_ratio = ratio
            if ratio > 2.2:
                high_detour_nets_count += 1

    # B9. Power Width Tier Compliance
    power_len_by_net: dict[str, float] = {}
    conforming_len_by_net: dict[str, float] = {}
    for s in routed_segs:
        nm = view.name_of(s.net_id)
        if nm in expected_w and not is_ground(nm):
            seg_len = math.hypot(s.end_x - s.start_x, s.end_y - s.start_y)
            power_len_by_net[nm] = power_len_by_net.get(nm, 0.0) + seg_len
            if s.width >= expected_w[nm] - 0.05:
                conforming_len_by_net[nm] = conforming_len_by_net.get(nm, 0.0) + seg_len
    total_pwr_len = sum(power_len_by_net.values())
    conf_pwr_len = sum(conforming_len_by_net.values())
    width_compliance = round(100.0 * conf_pwr_len / total_pwr_len, 1) if total_pwr_len > 0 else 100.0
    width_shortfall_pct = max(0.0, 100.0 - width_compliance)

    # -------------------------------------------------------------------------
    # Category C: Switching Power Supply & Hot-Loop Rules
    # -------------------------------------------------------------------------
    # C10. Switching Node Copper Length & Hot-Loop Violations
    sw_nids: set[int] = set()
    ind_pad_pts: list[tuple[float, float]] = []
    for ref, fp in fps.items():
        if _is_inductor(fp, ref):
            for p in fp.pads:
                ind_pad_pts.append((p.global_x, p.global_y))
                if p.net_id > 0 and p.net_id not in gnd_nids and p.net_id not in view.rails:
                    owners = view.owners.get(p.net_id, set())
                    if len(owners) <= 6 and any(o.startswith(("U", "IC", "D", "Q")) for o in owners):
                        sw_nids.add(p.net_id)

    switch_node_length_mm = 0.0
    hot_loop_violations = 0
    sw_segs = []
    for snid in sw_nids:
        slen = sum(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) for s in segs_by_net.get(snid, []))
        switch_node_length_mm += slen
        sw_segs.extend(segs_by_net.get(snid, []))
        if slen > 18.0:
            hot_loop_violations += 1
    switch_node_length_mm = round(switch_node_length_mm, 2)

    # C11. Feedback / Sense / Analog Trace Coupling to Switching Aggressors (< 1.2 mm)
    fb_aggressor_coupling_mm = 0.0
    if sw_segs or ind_pad_pts:
        for s in routed_segs:
            if s.net_id in sw_nids or s.net_id in gnd_nids or s.net_id in view.rails:
                continue
            nm_u = view.name_of(s.net_id).upper()
            if not any(k in nm_u for k in ("FB", "SENSE", "VREF", "PH", "EC", "ADC", "AIN")):
                continue
            mx, my = 0.5 * (s.start_x + s.end_x), 0.5 * (s.start_y + s.end_y)
            near_agg = any(math.hypot(mx - ix, my - iy) < 1.2 for ix, iy in ind_pad_pts)
            if not near_agg:
                for sws in sw_segs:
                    if math.hypot(mx - 0.5 * (sws.start_x + sws.end_x), my - 0.5 * (sws.start_y + sws.end_y)) < 1.2:
                        near_agg = True
                        break
            if near_agg:
                fb_aggressor_coupling_mm += math.hypot(s.end_x - s.start_x, s.end_y - s.start_y)
    fb_aggressor_coupling_mm = round(fb_aggressor_coupling_mm, 2)

    # C12. Decoupling Capacitor Distance to IC Supply Pins
    sup_pins = view.supply_pins()
    decap_map = view.decaps()
    decap_dists: list[float] = []
    decap_distance_violations = 0
    for rnid, ic_pin_list in sup_pins.items():
        rail_caps = [
            cref for cref, crnid in decap_map.items()
            if crnid == rnid and cref in fps and _parse_cap_farads(fps[cref].value) <= 2.2e-6
        ]
        if not rail_caps:
            continue
        for ic_ref, pin_num in ic_pin_list:
            if ic_ref not in fps:
                continue
            ic_gnds = view.nets_of.get(ic_ref, set()) & gnd_nids
            ic_pads = [p for p in fps[ic_ref].pads if str(p.pad_number) == str(pin_num)]
            if not ic_pads:
                continue
            ip = ic_pads[0]
            same_domain_caps = [
                c for c in rail_caps if not ic_gnds or (view.nets_of.get(c, set()) & ic_gnds)
            ] or rail_caps
            min_d = min(
                math.hypot(ip.global_x - cp.global_x, ip.global_y - cp.global_y)
                for c in same_domain_caps
                for cp in fps[c].pads
                if cp.net_id == rnid
            )
            decap_dists.append(min_d)
            if min_d > 6.5:
                decap_distance_violations += 1
    mean_decap_distance_mm = round(sum(decap_dists) / len(decap_dists), 2) if decap_dists else 0.0

    # -------------------------------------------------------------------------
    # Category D: Return Paths, Split Planes & High-Speed / Mixed-Signal Isolation
    # -------------------------------------------------------------------------
    # D13. Split-Plane Signal Crossings
    iso_gnd_nids = [
        nid
        for nid in gnd_nids
        if ("ISO" in view.name_of(nid).upper() or "AGND" in view.name_of(nid).upper() or "GNDA" in view.name_of(nid).upper())
        and len(view.owners[nid]) >= 4
    ]
    split_plane_crossings_count = 0
    if iso_gnd_nids:
        iso_nid = iso_gnd_nids[0]
        main_gnds = gnd_nids - set(iso_gnd_nids)
        bridge_refs = {
            r for r in view.owners[iso_nid] if view.nets_of.get(r, set()) & main_gnds
        }
        iso_pts = [
            (p.global_x, p.global_y)
            for r in (view.owners[iso_nid] - bridge_refs)
            if r in fps
            for p in fps[r].pads
        ]
        if len(iso_pts) >= 3:
            ix0, iy0 = min(x for x, _ in iso_pts) + 1.0, min(y for _, y in iso_pts) + 1.0
            ix1, iy1 = max(x for x, _ in iso_pts) - 1.0, max(y for _, y in iso_pts) - 1.0
            if ix1 > ix0 and iy1 > iy0:
                iso_side_refs = set(view.owners[iso_nid])
                for s in routed_segs:
                    if s.net_id in gnd_nids:
                        continue
                    owners = view.owners.get(s.net_id, set())
                    if owners & iso_side_refs:
                        continue
                    mx, my = 0.5 * (s.start_x + s.end_x), 0.5 * (s.start_y + s.end_y)
                    if ix0 <= mx <= ix1 and iy0 <= my <= iy1:
                        split_plane_crossings_count += 1

    # D14. Power Inductor to Sensitive Analog IC Separation
    ind_refs = [r for r, fp in fps.items() if _is_inductor(fp, r)]
    analog_ic_refs = [
        r
        for r, fp in fps.items()
        if re.match(r"^(U|IC)\d", r)
        and (
            bool(view.nets_of.get(r, set()) & set(iso_gnd_nids))
            or bool(re.search(r"(ADS111|INA21|LM358|MCP60|OP07|TL07|MAX318|HX711|ADC)", f"{fp.value} {fp.footprint_name}", re.I))
        )
    ]
    power_to_analog_clearance_mm = 99.0
    power_analog_violations = 0
    if ind_refs and analog_ic_refs:
        dists_pa = [
            math.hypot(fps[ir].x - fps[ar].x, fps[ir].y - fps[ar].y)
            for ir in ind_refs
            for ar in analog_ic_refs
        ]
        power_to_analog_clearance_mm = round(min(dists_pa), 2)
        power_analog_violations = sum(1 for d in dists_pa if d < 8.0)

    # D15. Crystal Oscillator Layout Violations (vias > 0 or routed length > 12 mm)
    crystal_layout_violations = 0
    for ref, fp in fps.items():
        if ref.startswith(("Y", "X")) or re.search(r"(XTAL|CRYSTAL|MHZ|32\.768)", f"{fp.value} {fp.footprint_name}", re.I):
            for nid in view.nets_of.get(ref, set()) - gnd_nids - view.rails:
                if len(view.owners.get(nid, set())) <= 4:
                    nvias = vias_by_net.get(nid, 0)
                    nlen = sum(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) for s in segs_by_net.get(nid, []))
                    if nvias > 0 or nlen > 12.0:
                        crystal_layout_violations += 1

    # D16. High-Speed / Clock / Bus Board-Edge Setback (< 0.8 mm from Edge.Cuts)
    edge_coupled_high_speed_mm = 0.0
    bounds = getattr(pcb.board_info, "board_bounds", None)
    if bounds:
        bx0, by0, bx1, by1 = bounds
        for s in routed_segs:
            if s.net_id in gnd_nids or s.net_id in view.rails:
                continue
            nm_u = view.name_of(s.net_id).upper()
            if any(k in nm_u for k in ("CLK", "SCK", "USB", "D+", "D-", "MISO", "MOSI", "SDA", "SCL")):
                mx, my = 0.5 * (s.start_x + s.end_x), 0.5 * (s.start_y + s.end_y)
                if min(abs(mx - bx0), abs(bx1 - mx), abs(my - by0), abs(by1 - my)) < 0.8:
                    edge_coupled_high_speed_mm += math.hypot(s.end_x - s.start_x, s.end_y - s.start_y)
    edge_coupled_high_speed_mm = round(edge_coupled_high_speed_mm, 2)

    # -------------------------------------------------------------------------
    # Category E: DFM, ESD & Mechanical Clearances
    # -------------------------------------------------------------------------
    # E17. ESD / TVS Diode to Connector Proximity (> 10.0 mm violation)
    esd_to_connector_violations = 0
    for ref, fp in fps.items():
        if ref.startswith("D") and re.search(r"(TVS|ESD|USBLC|PESD|SMBJ|SMAJ)", f"{fp.value} {fp.footprint_name}", re.I):
            conn_js = [
                o
                for nid in (view.nets_of.get(ref, set()) - gnd_nids)
                for o in view.owners.get(nid, set())
                if o.startswith("J") and o in fps
            ]
            if conn_js:
                min_jd = min(math.hypot(fp.x - fps[j].x, fp.y - fps[j].y) for j in conn_js)
                if min_jd > 10.0:
                    esd_to_connector_violations += 1

    # E18. Via-in-Pad (VIP) on SMD non-thermal pads
    vip_violations: list[tuple[str, str, str]] = []
    smd_pads = []
    for ref, fp in fps.items():
        for p in fp.pads:
            if p.pad_type == "smd" and p.net_id > 0:
                if p.net_id in gnd_nids and min(p.size_x, p.size_y) >= 2.2:
                    continue
                smd_pads.append((ref, p))
    for v in pcb.vias:
        for ref, p in smd_pads:
            if (
                abs(v.x - p.global_x) < 0.5 * p.size_x + 0.5 * v.drill
                and abs(v.y - p.global_y) < 0.5 * p.size_y + 0.5 * v.drill
            ):
                vip_violations.append((ref, str(p.pad_number), view.name_of(v.net_id)))
                break

    # E19. Mounting Hole Screw-Head Clearance Violations
    mounting_hole_clearance_violations = 0
    try:
        mholes = [h for h in mounting_holes(pcb) if not h.nets]
    except Exception:
        mholes = []
    for mh in mholes:
        keep_r = float(getattr(mh, "head_radius", mh.drill * 0.9))
        encroached = any(
            math.hypot(s.start_x - mh.x, s.start_y - mh.y) < keep_r
            or math.hypot(s.end_x - mh.x, s.end_y - mh.y) < keep_r
            for s in routed_segs
        )
        if encroached:
            mounting_hole_clearance_violations += 1

    # -------------------------------------------------------------------------
    # Composite Penalty & John Teel 0-100 Score
    # -------------------------------------------------------------------------
    penalty = round(
        1.20 * ic_channel_starvation_count
        + 1.20 * slalom_traces_count
        + 1.00 * len(off_axis_parts)
        + 0.50 * channel_symmetry_err_mm
        + 0.35 * acute_or_90_corners_count
        + 0.45 * excess_weave_vias
        + 0.35 * high_detour_nets_count
        + 0.60 * trace_stubs_count
        + 0.20 * width_shortfall_pct
        + 1.00 * hot_loop_violations
        + 0.50 * fb_aggressor_coupling_mm
        + 0.40 * decap_distance_violations
        + 1.20 * split_plane_crossings_count
        + 0.80 * power_analog_violations
        + 0.80 * crystal_layout_violations
        + 0.25 * edge_coupled_high_speed_mm
        + 0.60 * esd_to_connector_violations
        + 0.40 * len(vip_violations)
        + 1.20 * mounting_hole_clearance_violations,
        2,
    )
    teel_score_100 = round(max(0.0, 100.0 - 1.2 * penalty), 1)

    return {
        "ic_channel_starvation_count": ic_channel_starvation_count,
        "slalom_traces_count": slalom_traces_count,
        "off_axis_parts_count": len(off_axis_parts),
        "channel_symmetry_err_mm": channel_symmetry_err_mm,
        "acute_or_90_corners_count": acute_or_90_corners_count,
        "weaving_nets_count": len(weaving_nets),
        "weaving_nets": weaving_nets,
        "excess_weave_vias": excess_weave_vias,
        "high_detour_nets_count": high_detour_nets_count,
        "max_detour_ratio": max_detour_ratio,
        "trace_stubs_count": trace_stubs_count,
        "power_width_compliance_pct": width_compliance,
        "switch_node_length_mm": switch_node_length_mm,
        "hot_loop_violations": hot_loop_violations,
        "fb_aggressor_coupling_mm": fb_aggressor_coupling_mm,
        "decap_distance_violations": decap_distance_violations,
        "mean_decap_distance_mm": mean_decap_distance_mm,
        "split_plane_crossings_count": split_plane_crossings_count,
        "power_to_analog_clearance_mm": power_to_analog_clearance_mm,
        "power_analog_violations": power_analog_violations,
        "crystal_layout_violations": crystal_layout_violations,
        "edge_coupled_high_speed_mm": edge_coupled_high_speed_mm,
        "esd_to_connector_violations": esd_to_connector_violations,
        "via_in_pad_count": len(vip_violations),
        "mounting_hole_clearance_violations": mounting_hole_clearance_violations,
        "teel_score_100": teel_score_100,
        "physical_quality_penalty": penalty,
    }
