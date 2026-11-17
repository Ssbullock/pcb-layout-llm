#!/usr/bin/env python3
"""
fetch_and_prep_all_sets.py
--------------------------
Downloads and normalizes all stress sets from KiCadRoutingTools/tests/stress
into /Users/simon/Fine Tune/corpus:
  - boards_set<N>/          (human-placed, human-routed reference + .kicad_pro/.kicad_dru)
  - boards_unrouted_set<N>/ (human-placed, copper-stripped + .kicad_pro/.kicad_dru)

Includes set1..set30 (and optionally --include-monsters for set1monster..set3monster).
Enforces polite throttling between GitHub raw downloads and skips boards already prepped.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
KRT_STRESS = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools/tests/stress")
KPY = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"
PREP_PY = KRT_STRESS / "prep_set2.py"
SILK_PY = KRT_STRESS / "set_silk_ignore.py"

# Silk-dense boards where silkscreen DRC is demoted to 'ignore'
SILK_IGNORE_BOARDS = {"lily58"}


def fetch_board_from_manifest_entry(entry: dict, src_dir: Path, delay_s: float = 2.5) -> Path | None:
    """Download a single board + sibling .kicad_pro/.kicad_dru into src_dir."""
    file_name = Path(entry["file"]).name
    dest = src_dir / file_name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if entry.get("archive_url"):
        time.sleep(delay_s)
        with tempfile.TemporaryDirectory() as td:
            zp = Path(td) / "src.zip"
            r = subprocess.run(
                ["curl", "-sL", "--fail", entry["archive_url"], "-o", str(zp)],
                capture_output=True,
            )
            if r.returncode != 0 or not zp.exists() or zp.stat().st_size == 0:
                return None
            try:
                with zipfile.ZipFile(zp) as z:
                    z.extractall(Path(td) / "x")
            except zipfile.BadZipFile:
                return None
            member = Path(td) / "x" / entry["archive_member"]
            if not member.exists():
                return None
            shutil.copy(member, dest)
            for ext in (".kicad_pro", ".kicad_dru"):
                sib = member.with_suffix(ext)
                if sib.exists():
                    shutil.copy(sib, dest.with_suffix(ext))
        return dest

    raw_url = entry.get("raw_url")
    if not raw_url:
        return None

    time.sleep(delay_s)
    r = subprocess.run(
        ["curl", "-sL", "--fail", raw_url, "-o", str(dest)],
        capture_output=True,
    )
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        if dest.exists():
            dest.unlink()
        return None

    # Fetch sibling .kicad_pro and .kicad_dru if they exist on raw.githubusercontent.com
    if raw_url.endswith(".kicad_pcb"):
        base_url = raw_url[: -len(".kicad_pcb")]
        for ext in (".kicad_pro", ".kicad_dru"):
            sib = dest.with_suffix(ext)
            subprocess.run(
                ["curl", "-sL", "--fail", base_url + ext, "-o", str(sib)],
                capture_output=True,
            )
            if sib.exists() and sib.stat().st_size == 0:
                sib.unlink()

    return dest


def process_set1(delay_s: float = 2.5) -> tuple[int, int]:
    """Fetch and prep set1 directly from manifest_set1.json using prep_set2.py."""
    manifest_path = KRT_STRESS / "manifest_set1.json"
    boards = json.loads(manifest_path.read_text())
    src_dir = CORPUS_DIR / "sources" / "github_set1"
    routed_dir = CORPUS_DIR / "boards_set1"
    unrouted_dir = CORPUS_DIR / "boards_unrouted_set1"
    src_dir.mkdir(parents=True, exist_ok=True)
    routed_dir.mkdir(parents=True, exist_ok=True)
    unrouted_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    for b in boards:
        short_name = b.get("short_name") or Path(b["file"]).stem
        routed_pcb = routed_dir / f"{short_name}.kicad_pcb"
        unrouted_pcb = unrouted_dir / f"{short_name}.kicad_pcb"
        if routed_pcb.exists() and unrouted_pcb.exists() and unrouted_pcb.stat().st_size > 0:
            print(f"  [SKIP] set1/{short_name} (already prepped)", flush=True)
            ok_count += 1
            continue

        src_pcb = fetch_board_from_manifest_entry(b, src_dir, delay_s=delay_s)
        if not src_pcb:
            print(f"  [FAIL FETCH] set1/{short_name} <- {b.get('raw_url')}", flush=True)
            continue

        print(f"  [PREP] set1/{short_name} ({src_pcb.stat().st_size // 1024} KB)...", flush=True)
        r = subprocess.run(
            [KPY, str(PREP_PY), str(src_pcb), str(routed_pcb), str(unrouted_pcb)],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and routed_pcb.exists() and unrouted_pcb.exists():
            for ext in (".kicad_pro", ".kicad_dru"):
                sib = src_pcb.with_suffix(ext)
                if sib.exists():
                    shutil.copy(sib, routed_dir / f"{short_name}{ext}")
                    shutil.copy(sib, unrouted_dir / f"{short_name}{ext}")
            ok_count += 1
        else:
            print(f"  [FAIL PREP] set1/{short_name}", flush=True)

    return ok_count, len(boards)


def process_standard_set(set_name: str, delay_s: float = 2.5) -> tuple[int, int]:
    """Fetch sources from manifest_<set_name>.json with throttling, then run prep_<set_name>.sh."""
    manifest_path = KRT_STRESS / f"manifest_{set_name}.json"
    prep_sh = KRT_STRESS / f"prep_{set_name}.sh"
    boards = json.loads(manifest_path.read_text())

    routed_dir = CORPUS_DIR / f"boards_{set_name}"
    unrouted_dir = CORPUS_DIR / f"boards_unrouted_{set_name}"
    src_dir = CORPUS_DIR / "sources" / f"github_{set_name}"
    src_dir.mkdir(parents=True, exist_ok=True)
    routed_dir.mkdir(parents=True, exist_ok=True)
    unrouted_dir.mkdir(parents=True, exist_ok=True)

    existing_unrouted = list(unrouted_dir.glob("*.kicad_pcb"))
    if len(existing_unrouted) >= len(boards):
        print(f"[{set_name}] Already complete ({len(existing_unrouted)}/{len(boards)} boards).", flush=True)
        return len(existing_unrouted), len(boards)

    print(f"[{set_name}] Fetching {len(boards)} boards into {src_dir}...", flush=True)
    fetched = 0
    for b in boards:
        dest = fetch_board_from_manifest_entry(b, src_dir, delay_s=delay_s)
        if dest:
            fetched += 1
            print(f"  OK fetch {b.get('repo', dest.name):40} ({dest.stat().st_size // 1024} KB)", flush=True)
        else:
            print(f"  FAIL fetch {b.get('repo', '?')} <- {b.get('raw_url', b.get('archive_url'))}", flush=True)

    print(f"[{set_name}] Running {prep_sh.name} ({fetched}/{len(boards)} fetched)...", flush=True)
    env = os.environ.copy()
    env["STRESS_DIR"] = str(CORPUS_DIR)
    env["KICAD_PYTHON"] = KPY
    subprocess.run(["bash", str(prep_sh)], cwd=str(KRT_STRESS), env=env)

    final_unrouted = list(unrouted_dir.glob("*.kicad_pcb"))
    print(f"[{set_name}] Done: {len(final_unrouted)}/{len(boards)} boards ready in {unrouted_dir.name}/", flush=True)
    return len(final_unrouted), len(boards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", help="Specific sets to run (e.g. set1 set2 set4). Default: set1..set30")
    ap.add_argument("--include-monsters", action="store_true", help="Also fetch/prep set1monster..set3monster")
    ap.add_argument("--delay", type=float, default=2.5, help="Delay in seconds between GitHub downloads")
    args = ap.parse_args()

    if args.sets:
        target_sets = args.sets
    else:
        target_sets = [f"set{i}" for i in range(1, 31)]
        if args.include_monsters:
            target_sets += ["set1monster", "set2monster", "set3monster"]

    summary = {}
    total_ok = 0
    total_expected = 0

    for s in target_sets:
        t0 = time.time()
        print(f"\n============================================================", flush=True)
        print(f"Processing {s} ...", flush=True)
        print(f"============================================================", flush=True)
        if s == "set1":
            ok, total = process_set1(delay_s=args.delay)
        else:
            ok, total = process_standard_set(s, delay_s=args.delay)
        elapsed = time.time() - t0
        summary[s] = {"ok": ok, "total": total, "elapsed_s": round(elapsed, 1)}
        total_ok += ok
        total_expected += total
        print(f"--> {s}: {ok}/{total} boards ready ({elapsed:.1f}s)", flush=True)

    status_file = CORPUS_DIR / "corpus_manifest_summary.json"
    status_file.write_text(json.dumps({
        "total_ready": total_ok,
        "total_expected": total_expected,
        "sets": summary,
    }, indent=2))
    print(f"\n============================================================", flush=True)
    print(f"ALL SETS COMPLETE: {total_ok}/{total_expected} boards ready in {CORPUS_DIR}", flush=True)
    print(f"Summary written to {status_file}", flush=True)


if __name__ == "__main__":
    main()
