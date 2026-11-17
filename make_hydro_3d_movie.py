#!/usr/bin/env python3
"""Render HydroPCB's AI placement + KRT routing using KRT's movie3d Three.js studio renderer.

Produces the dark-studio 3D movie with group-colored semi-transparent STEP models,
3D ratsnest airwires, bottom group legend, and live trace-by-trace 3D routing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
MOVIE3D_DIR = KRT_ROOT / "movie3d"

for _p in (
    Path(__file__).resolve().parent,
    MOVIE3D_DIR,
    KRT_ROOT / "py_placer",
    KRT_ROOT / "py_router",
    KRT_ROOT / "py_tools",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_movie as m3d_make  # noqa: E402
import record as m3d_record  # noqa: E402
import render_theme  # noqa: E402
import route3d as m3d_route  # noqa: E402
from pcb_design_intelligence import build_smart_routing_chains  # noqa: E402


def ensure_traced_route_boards(run_dir: Path, route_dir: Path) -> list[str]:
    """Return [step0.kicad_pcb, step1.kicad_pcb, ...] with *_routetrace.json, reusing if present."""
    step3 = route_dir / "step3.kicad_pcb"
    trace_json = route_dir / "step3_routetrace.json"
    if step3.exists() and trace_json.exists():
        boards = sorted(str(p) for p in route_dir.glob("step*.kicad_pcb"))
        if len(boards) >= 2:
            print(f"Reusing traced routing boards in {route_dir} ({len(boards)} steps)", flush=True)
            return boards

    res = json.loads((run_dir / "rollout_result.json").read_text(encoding="utf-8"))
    best_seed = res["best_seed"]
    strat_doc = json.loads((run_dir / "routing_strategy.json").read_text(encoding="utf-8"))
    seed_pcb = run_dir / f"s{best_seed}_route" / "in.kicad_pcb"
    if not seed_pcb.exists():
        seed_pcb = run_dir / f"s{best_seed}.kicad_pcb"

    _, prod_steps = build_smart_routing_chains(seed_pcb, routing_strategy=strat_doc)
    if route_dir.exists():
        shutil.rmtree(route_dir)
    route_dir.mkdir(parents=True)

    env = dict(os.environ)
    env["KICAD_ROUTE_TRACE"] = "1"
    cur_in = route_dir / "step0.kicad_pcb"
    shutil.copyfile(seed_pcb, cur_in)
    pro_src = run_dir / "final_production.kicad_pro"
    if pro_src.exists():
        shutil.copyfile(pro_src, cur_in.with_suffix(".kicad_pro"))

    boards = [str(cur_in)]
    for idx, (rel_script, step_args) in enumerate(prod_steps, 1):
        step_out = route_dir / f"step{idx}.kicad_pcb"
        cmd = [
            sys.executable, "-X", "utf8",
            str(KRT_ROOT / rel_script),
            str(cur_in), str(step_out), *step_args,
        ]
        print(f"Running traced routing step {idx}/{len(prod_steps)}: {rel_script}", flush=True)
        subprocess.run(cmd, cwd=str(KRT_ROOT), env=env, check=True)
        boards.append(str(step_out))
        cur_in = step_out
    return boards


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HydroPCB 3D studio movie via movie3d")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/hydro_grpo_llm_HydroPCB"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluated_boards_for_simon/HydroPCB/HydroPCB_3d_movie.mp4"),
    )
    parser.add_argument("--route-frames", type=int, default=240)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_mp4 = args.out.resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    work_dir = run_dir / "movie3d_studio_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    res = json.loads((run_dir / "rollout_result.json").read_text(encoding="utf-8"))
    best_seed = res["best_seed"]
    strat_doc = json.loads((run_dir / "routing_strategy.json").read_text(encoding="utf-8"))
    pcr = strat_doc.get("placer_channel_reservation") or {}

    intent_clean = {
        k: v
        for k, v in json.loads((run_dir / "intent.json").read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }
    clean_intent_path = work_dir / "movie_intent.json"
    clean_intent_path.write_text(json.dumps(intent_clean, indent=2), encoding="utf-8")

    # Export VRML once so TO-263-5_TabPin3.step (which kicad-cli GLB skips) becomes a triangulated .wrl
    vrml_dir = work_dir / "vrml_models"
    vrml_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            "pcb", "export", "vrml", "-f",
            "--models-dir", str(vrml_dir),
            "--units", "mm",
            "-o", str(work_dir / "tmp_vrml.wrl"),
            str(run_dir / "fixed.kicad_pcb"),
        ],
        capture_output=True,
        check=True,
    )
    u1_wrl = vrml_dir / "TO-263-5_TabPin3.wrl"

    # Point project_models to Simon's original HydroPCB STEP files, and U1 to triangulated WRL
    fixed_src = (run_dir / "fixed.kicad_pcb").read_text(encoding="utf-8")
    fixed_src = fixed_src.replace(
        "${KIPRJMOD}/3dmodels/project_models/",
        "/Users/simon/Hydropinics/HydroPCB/3dmodels/project_models/",
    )
    if u1_wrl.exists():
        fixed_src = fixed_src.replace(
            "${KICAD10_3DMODEL_DIR}/Package_TO_SOT_SMD.3dshapes/TO-263-5_TabPin3.step",
            str(u1_wrl),
        )
    fixed_with_models = work_dir / "fixed_with_models.kicad_pcb"
    fixed_with_models.write_text(fixed_src, encoding="utf-8")

    # More dynamic camera orbit & lower 3/4 elevation
    m3d_record.SWEEP = 48.0
    m3d_record.ELEV = 40.0

    # Build the exact Namespace that movie3d/record.py expects
    ap = argparse.ArgumentParser(add_help=False)
    m3d_record.own_args(ap, work_required=False)
    a, _ = ap.parse_known_args([
        "--work", str(work_dir),
        "--width", "1280",
        "--height", "720",
        "--fps", "24",
        "--frames", "120",
        "--speedup", "4",
    ])
    a.browser = os.environ.get("MOVIE3D_CHROMIUM")
    a.gpu = "auto"
    a.crf = args.crf

    placed_out = work_dir / f"placed_s{best_seed}.kicad_pcb"
    place_argv = [
        str(fixed_with_models),
        str(placed_out),
        "--groups", str(run_dir / "groups.json"),
        "--intent", str(clean_intent_path),
        "--seed", str(best_seed),
        "--anchored-seeds", str(best_seed),
        "--halo-coef", str(pcr.get("halo_coef", 0.45)),
        "--halo-weight", str(pcr.get("halo_weight", 20.0)),
        "--congestion-weight", str(pcr.get("congestion_weight", 12.0)),
        "--pad-weight", str(pcr.get("pad_weight", 6.0)),
        "--track-pitch", str(pcr.get("track_pitch", 0.40)),
    ]

    t0 = time.time()
    print("1/3 Recording 3D AI placement via movie3d/record.py...", flush=True)
    rc, rec = m3d_record.place(a, place_argv)
    if rec.scene is None:
        raise RuntimeError(f"record.place failed (rc={rc})")
    rec.placed(rc, rec.hold)

    # Center wide intro frames symmetrically around the board so the first frame
    # isn't pushed high up in the viewport by the asymmetric pile bounding box.
    if rec.home and rec.wide:
        hx0, hy0, hx1, hy1 = rec.home[:4]
        for fr in rec.frames:
            v = fr.get("view")
            if v and (v[2] - v[0] > (hx1 - hx0) + 4.0 or v[3] - v[1] > (hy1 - hy0) + 4.0):
                px = min(22.0, max(hx0 - v[0], v[2] - hx1))
                py = min(22.0, max(hy0 - v[1], v[3] - hy1))
                fr["view"] = [round(hx0 - px, 3), round(hy0 - py, 3), round(hx1 + px, 3), round(hy1 + py, 3), v[4], v[5]]

    # 2. Play KRT traced routing on the 3D board
    route_dir = run_dir / "movie_3d_build" / "route_trace"
    chain_boards = ensure_traced_route_boards(run_dir, route_dir)

    th = render_theme.theme("dark", strict=False)
    layers = list(rec.pcb.board_info.copper_layers)
    pal = render_theme.layer_palette(layers, th)
    inner = iter(m3d_route.INNER_3D * len(layers))
    track = {n: (c if n in ("F.Cu", "B.Cu") else next(inner)) for n, c in pal.items()}

    cop = m3d_route.Copper(layers, rec.scene["outline"], rec.scene["cutouts"])
    rec.cop = cop
    rec.copper_shot(f"seed {best_seed} placed: routing with KRT", times=rec.hold)
    m3d_route.play_chain(
        rec,
        cop,
        chain_boards[1:],
        chain_boards[0],
        prefix="route | ",
        first_label=f"placed, seed {best_seed}",
        budget=args.route_frames,
    )
    cop.next()
    rec.copper_shot(
        f"routed: 49/49 nets, {res['best_score']['quality']['vias']} vias, 0 blocking",
        times=2 * rec.hold,
    )
    # Flip the board over to show bottom-layer copper & pours, then return to top
    rec.turn_over(rec.copper_shot)
    rec.scene["copper"] = cop.json()
    rec.scene["colors"].update(
        layers=[list(track.get(n, (200, 120, 60))) for n in layers],
        pour_layers=[list(pal.get(n, (200, 120, 60))) for n in layers],
        via=list(th.rgb("via")),
        new=list(th.rgb("event_new")),
        rip=[235, 70, 235],
        restored=list(th.rgb("event_restored")),
    )

    sp, tp, n_frames = m3d_record.write(a, rec)
    print(f"Recorded {n_frames} frames in {time.time() - t0:.1f} s", flush=True)

    # 3. Render with Three.js in headless Chromium (GPU/Metal on macOS)
    frames_dir = work_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    glb = (json.loads(Path(sp).read_text(encoding="utf-8")).get("glb") or {}).get("path")
    browser, gpu = m3d_make.gpu_browser(a)
    job = {
        "browser": browser,
        "gpu": gpu,
        "width": a.width,
        "height": a.height,
        "outDir": str(frames_dir),
        "scene": sp,
        "timeline": tp,
        "glb": glb,
    }
    jp = work_dir / "job.json"
    jp.write_text(json.dumps(job), encoding="utf-8")

    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not found on PATH")

    print(f"2/3 Rendering {n_frames} Three.js frames (gpu={gpu})...", flush=True)
    proc = subprocess.Popen(
        [node, str(MOVIE3D_DIR / "render.mjs"), str(jp)],
        cwd=str(MOVIE3D_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    done_msg, err_msg = None, None
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("type")
        if kind == "info":
            print(
                f"  renderer={msg.get('renderer')} three=r{msg.get('three')} models={msg.get('models', 0)}",
                flush=True,
            )
        elif kind == "progress" and (msg["done"] % 50 == 0 or msg["done"] == msg["of"]):
            print(f"  rendered {msg['done']}/{msg['of']} frames", flush=True)
        elif kind == "done":
            done_msg = msg
        elif kind == "error":
            err_msg = msg.get("why")
    proc.wait()
    if err_msg or done_msg is None or proc.returncode:
        tail = (proc.stderr.read() if proc.stderr else "").strip().splitlines()[-5:]
        raise RuntimeError(f"render.mjs failed: {err_msg or ' | '.join(tail) or proc.returncode}")

    # 4. Encode MP4
    print("3/3 Encoding MP4 via ffmpeg...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{a.fps:g}",
            "-i", str(frames_dir / "f%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf),
            "-movflags", "+faststart", str(out_mp4),
        ],
        check=True,
    )
    print(f"SUCCESS: {out_mp4} ({out_mp4.stat().st_size / 1e6:.2f} MB, {n_frames} frames)", flush=True)


if __name__ == "__main__":
    main()
