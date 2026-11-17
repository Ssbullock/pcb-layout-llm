#!/usr/bin/env python3
"""
make_ai_pcb_movie.py
--------------------
Builds a 3-act MP4 visualization of our 32B PCB AI pipeline on a board:
  Act 1 (Pre-Placement): Animated 32B LLM architectural plan (`groups.json` hulls,
         `near` adjacency springs, `keep_away_rules`, `power_width_tiers`, and
         `placer_channel_reservation` weights) overlaid on the board outline.
  Act 2 (Placement): `py_placer/place_movie.py` showing the pile flying onto the
         board, tinted by the 32B model's functional groups, and annealing with
         the model's channel reservation weights.
  Act 3 (Routing): `py_router/route_movie.py` with `KICAD_ROUTE_TRACE=1` showing
         every Rust A* trace, rip-up, copper pour, and teardrop landing net-by-net.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
for _d in ("py_router", "py_placer", "py_tools"):
    _p = str(KRT_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kicad_parser import parse_kicad_pcb  # noqa: E402
from pcb_design_intelligence import build_smart_routing_chains  # noqa: E402

PALETTE = [
    (78, 168, 222),
    (82, 183, 136),
    (244, 162, 97),
    (231, 111, 81),
    (181, 131, 141),
    (168, 218, 220),
    (233, 196, 106),
    (144, 190, 109),
    (155, 93, 229),
    (241, 91, 181),
    (0, 187, 249),
    (254, 228, 64),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_preplacement_act(
    run_dir: Path,
    out_mp4: Path,
    size: int = 1000,
    fps: int = 12,
) -> Path:
    """Render Act 1: 32B LLM architectural plan & weight visualization."""
    res = json.loads((run_dir / "rollout_result.json").read_text(encoding="utf-8"))
    best_seed = res["best_seed"]
    placed_pcb_path = run_dir / f"s{best_seed}.kicad_pcb"
    fixed_pcb_path = run_dir / "fixed.kicad_pcb"
    groups_doc = json.loads((run_dir / "groups.json").read_text(encoding="utf-8"))
    strat_doc = json.loads((run_dir / "routing_strategy.json").read_text(encoding="utf-8"))

    pcb_placed = parse_kicad_pcb(str(placed_pcb_path))
    pcb_fixed = parse_kicad_pcb(str(fixed_pcb_path))
    bounds = pcb_placed.board_info.board_bounds
    bx0, by0, bx1, by1 = bounds
    bw, bh = bx1 - bx0, by1 - by0

    pad_px = 70
    header_h = 110
    footer_h = 80
    scale = (size - 2 * pad_px) / max(bw, bh)
    w_px = int(round(bw * scale + 2 * pad_px))
    h_px = int(round(bh * scale + 2 * pad_px + header_h + footer_h))
    # Ensure even dimensions for h264
    w_px += w_px % 2
    h_px += h_px % 2

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = int(round(pad_px + (x - bx0) * scale))
        py = int(round(header_h + pad_px + (y - by0) * scale))
        return px, py

    groups = groups_doc.get("groups", [])
    grp_by_name = {g["name"]: g for g in groups}
    grp_colors = {g["name"]: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    # Precompute group target centroids from placed board & unplaced fixed positions
    grp_centers: dict[str, tuple[float, float]] = {}
    for g in groups:
        xs, ys = [], []
        for r in g.get("refs", []):
            fp = pcb_placed.footprints.get(r)
            if fp:
                xs.append(fp.x)
                ys.append(fp.y)
        if xs:
            grp_centers[g["name"]] = (sum(xs) / len(xs), sum(ys) / len(ys))

    font_title = _load_font(22)
    font_sub = _load_font(15)
    font_lbl = _load_font(13)

    pcr = strat_doc.get("placer_channel_reservation") or {}
    vtr = strat_doc.get("via_and_trace_rules") or {}
    gnd_nets = ", ".join(strat_doc.get("ground_plane_nets") or [])
    ka_rules = "; ".join(strat_doc.get("keep_away_rules") or ["none"])
    pw_cnt = len(strat_doc.get("power_width_tiers") or {})

    total_frames = int(fps * 6.0)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w_px}x{h_px}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for f_idx in range(total_frames):
        t = f_idx / max(1, total_frames - 1)
        img = Image.new("RGB", (w_px, h_px), (18, 22, 28))
        draw = ImageDraw.Draw(img, "RGBA")

        # Header banner
        draw.rectangle([0, 0, w_px, header_h], fill=(26, 32, 42))
        draw.text((24, 16), f"ACT 1 — 32B LLM Pre-Placement Plan: {res['board']}", fill=(240, 246, 252), font=font_title)
        draw.text(
            (24, 48),
            f"Channel Weights: halo={pcr.get('halo_coef')} / w={pcr.get('halo_weight')}  cong_w={pcr.get('congestion_weight')}  "
            f"via_cost={vtr.get('via_cost')}  turn_cost={vtr.get('turn_cost')}",
            fill=(140, 200, 255),
            font=font_sub,
        )
        draw.text(
            (24, 74),
            f"Planes: [{gnd_nets}]   Power Tiers: {pw_cnt} nets   EMI Keep-Away: {ka_rules}",
            fill=(160, 220, 175),
            font=font_sub,
        )

        # Board outline
        x0p, y0p = to_px(bx0, by0)
        x1p, y1p = to_px(bx1, by1)
        draw.rectangle([x0p, y0p, x1p, y1p], fill=(28, 34, 40), outline=(180, 190, 205), width=2)

        # Fixed footprints
        for ref, fp in pcb_fixed.footprints.items():
            if not getattr(fp, "locked", False):
                continue
            px, py = to_px(fp.x, fp.y)
            draw.rectangle([px - 6, py - 6, px + 6, py + 6], fill=(90, 100, 115, 160), outline=(140, 150, 165))

        # Progressive reveal of LLM functional groups
        n_show = max(1, min(len(groups), int(math.ceil(t * 1.45 * len(groups)))))
        for g in groups[:n_show]:
            gname = g["name"]
            col = grp_colors[gname]
            pts = []
            for r in g.get("refs", []):
                fp = pcb_placed.footprints.get(r)
                if fp:
                    pts.append(to_px(fp.x, fp.y))
            if pts:
                gx0 = min(p[0] for p in pts) - 16
                gy0 = min(p[1] for p in pts) - 16
                gx1 = max(p[0] for p in pts) + 16
                gy1 = max(p[1] for p in pts) + 16
                draw.rounded_rectangle([gx0, gy0, gx1, gy1], radius=10, fill=(*col, 42), outline=(*col, 210), width=2)
                for px, py in pts:
                    draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(*col, 230))
                anchor = g.get("anchor", gname)
                draw.text((gx0 + 4, max(header_h + 4, gy0 - 16)), f"{gname} ({anchor})", fill=(*col, 255), font=font_lbl)

        # Progressive reveal of `near` adjacency springs (after t > 0.45)
        if t > 0.45:
            spring_alpha = int(min(220, 220 * (t - 0.45) / 0.35))
            seen_edges = set()
            for g in groups[:n_show]:
                c1 = grp_centers.get(g["name"])
                if not c1:
                    continue
                for nb in g.get("near", []):
                    edge = tuple(sorted((g["name"], nb)))
                    if edge in seen_edges or nb not in grp_centers:
                        continue
                    seen_edges.add(edge)
                    p1 = to_px(*c1)
                    p2 = to_px(*grp_centers[nb])
                    draw.line([p1, p2], fill=(255, 215, 100, spring_alpha), width=2)

        # Footer caption
        draw.rectangle([0, h_px - footer_h, w_px, h_px], fill=(26, 32, 42))
        stage_txt = (
            "1/3: Partitioning netlist into 1-IC functional groups..."
            if t < 0.45
            else ("2/3: Binding 'near' adjacency springs & keep-away rules..." if t < 0.78 else "3/3: Locking channel reservation weights -> starting placement anneal")
        )
        draw.text((24, h_px - footer_h + 26), stage_txt, fill=(230, 235, 240), font=font_sub)

        proc.stdin.write(img.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rendering Act 1")
    return out_mp4


def render_placement_act(run_dir: Path, out_mp4: Path, size: int = 1000, fps: int = 12) -> Path:
    """Render Act 2: `py_placer/place_movie.py` with LLM groups and channel weights."""
    res = json.loads((run_dir / "rollout_result.json").read_text(encoding="utf-8"))
    best_seed = res["best_seed"]
    strat_doc = json.loads((run_dir / "routing_strategy.json").read_text(encoding="utf-8"))
    pcr = strat_doc.get("placer_channel_reservation") or {}

    intent_clean = {k: v for k, v in json.loads((run_dir / "intent.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    clean_intent_path = out_mp4.parent / "movie_intent.json"
    clean_intent_path.write_text(json.dumps(intent_clean, indent=2), encoding="utf-8")

    placed_tmp = run_dir / f"movie_placed_s{best_seed}.kicad_pcb"
    cmd = [
        sys.executable, "-X", "utf8",
        str(KRT_ROOT / "py_placer/place_movie.py"),
        str(run_dir / "fixed.kicad_pcb"),
        str(placed_tmp),
        "--groups", str(run_dir / "groups.json"),
        "--intent", str(clean_intent_path),
        "--seed", str(best_seed),
        "--frames", "120",
        "--speedup", "4",
        "--size", str(size),
        "--fps", str(fps),
        "--halo-coef", str(pcr.get("halo_coef", 0.45)),
        "--halo-weight", str(pcr.get("halo_weight", 20.0)),
        "--congestion-weight", str(pcr.get("congestion_weight", 12.0)),
        "--pad-weight", str(pcr.get("pad_weight", 6.0)),
        "--track-pitch", str(pcr.get("track_pitch", 0.40)),
        "--movie", str(out_mp4),
    ]
    print("Running Act 2 (place_movie.py):", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(KRT_ROOT), check=True)
    return out_mp4


def render_routing_act(run_dir: Path, out_mp4: Path, size: int = 1000, fps: int = 12) -> Path:
    """Render Act 3: Re-run winning seed production steps with `KICAD_ROUTE_TRACE=1` and film via `route_movie.py`."""
    res = json.loads((run_dir / "rollout_result.json").read_text(encoding="utf-8"))
    best_seed = res["best_seed"]
    strat_doc = json.loads((run_dir / "routing_strategy.json").read_text(encoding="utf-8"))
    seed_pcb = run_dir / f"s{best_seed}.kicad_pcb"
    _, prod_steps = build_smart_routing_chains(seed_pcb, routing_strategy=strat_doc)

    movie_route_dir = run_dir / "movie_route_trace"
    if movie_route_dir.exists():
        shutil.rmtree(movie_route_dir)
    movie_route_dir.mkdir(parents=True)

    env = dict(os.environ)
    env["KICAD_ROUTE_TRACE"] = "1"

    cur_in = movie_route_dir / "step0_placed.kicad_pcb"
    shutil.copyfile(seed_pcb, cur_in)
    pro_src = run_dir / "final_production.kicad_pro"
    if pro_src.exists():
        shutil.copyfile(pro_src, cur_in.with_suffix(".kicad_pro"))

    chain_pcbs = [cur_in]
    for idx, (rel_script, step_args) in enumerate(prod_steps, 1):
        step_out = movie_route_dir / f"step{idx}.kicad_pcb"
        cmd = [sys.executable, "-X", "utf8", str(KRT_ROOT / rel_script), str(cur_in), str(step_out), *step_args]
        print(f"Running traced routing step {idx}/{len(prod_steps)}: {rel_script}", flush=True)
        subprocess.run(cmd, cwd=str(KRT_ROOT), env=env, check=True)
        chain_pcbs.append(step_out)
        cur_in = step_out

    movie_cmd = [
        sys.executable, "-X", "utf8",
        str(KRT_ROOT / "py_router/route_movie.py"),
        *[str(p) for p in chain_pcbs],
        "-o", str(out_mp4),
        "--size", str(size),
        "--fps", str(fps),
        "--max-frames", "480",
    ]
    print("Running Act 3 (route_movie.py):", " ".join(movie_cmd), flush=True)
    subprocess.run(movie_cmd, cwd=str(KRT_ROOT), check=True)
    return out_mp4


def concat_acts(acts: list[Path], out_mp4: Path, target_w: int = 1000, target_h: int = 1000, fps: int = 12) -> Path:
    """Scale/pad all acts to identical dimensions and concatenate into one MP4."""
    norm_acts = []
    for i, act in enumerate(acts):
        norm = act.with_name(f"norm_{i}_{act.name}")
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(act),
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=12161c,fps={fps},setsar=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(norm),
        ]
        subprocess.run(cmd, check=True)
        norm_acts.append(norm)

    concat_list = out_mp4.with_suffix(".txt")
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in norm_acts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", "-movflags", "+faststart", str(out_mp4)],
        check=True,
    )
    concat_list.unlink(missing_ok=True)
    for p in norm_acts:
        p.unlink(missing_ok=True)
    return out_mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("/Users/simon/Fine Tune/runs/hydro_grpo_llm_AI_Hydro_Controller"))
    ap.add_argument("--out", type=Path, default=Path("/Users/simon/Fine Tune/evaluated_boards_for_simon/AI_Hydro_Controller/AI_Hydro_Controller_ai_movie.mp4"))
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.out = args.out.resolve()

    work = args.run_dir / "movie_build"
    work.mkdir(parents=True, exist_ok=True)

    act1 = render_preplacement_act(args.run_dir, work / "act1_preplacement.mp4", size=args.size, fps=args.fps)
    act2 = render_placement_act(args.run_dir, work / "act2_placement.mp4", size=args.size, fps=args.fps)
    act3 = render_routing_act(args.run_dir, work / "act3_routing.mp4", size=args.size, fps=args.fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    concat_acts([act1, act2, act3], args.out, target_w=args.size, target_h=args.size, fps=args.fps)
    print(f"\nDONE! Wrote combined 3-act AI PCB movie to: {args.out}")


if __name__ == "__main__":
    main()
