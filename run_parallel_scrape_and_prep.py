#!/usr/bin/env python3
"""
run_parallel_scrape_and_prep.py
-------------------------------
Continuous Parallel Producer-Consumer Pipeline:
  - Worker A (Producer): Runs `scrape_kicad_repos.py` (BoardRepo 9.6K+ sitemap +
    GitHub date-sharded search + curated catalogs/orgs; respects heypcb.ai robots.txt
    `ai-train=no` exclusion and >=2.85s request throttling) until valid corpus reaches
    `--target-corpus-boards` (default: 1000).
  - Worker B (Consumer Pool): Concurrently watches ALL existing `boards_unrouted_set*`
    AND newly admitted `boards_unrouted_scraped_*` boards, skips degenerate + held-out
    Large-Val boards, runs `gym_rollout.py` + `format_sft_record` in parallel, merges
    into `dataset/sft_v8.jsonl`, and periodically refreshes curated splits via
    `curate_sft_dataset.py`.

Safe to interrupt & `--resume`: all scraper manifests, rollout_result.json caches,
and merged JSONL records persist atomically on disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
RUNS_DIR = WORKSPACE / "runs"
DATASET_DIR = WORKSPACE / "dataset"
STATUS_PATH = RUNS_DIR / "parallel_1000_status.json"
META_CACHE_PATH = RUNS_DIR / "board_meta_cache.json"
SFT_BASE_PATH = DATASET_DIR / "sft_v7.jsonl"
SFT_OUT_PATH = DATASET_DIR / "sft_v8.jsonl"

sys.path.insert(0, str(WORKSPACE))
from build_sft_dataset import format_sft_record  # noqa: E402
from curate_sft_dataset import load_exclusion_sets  # noqa: E402
from gym_rollout import parse_kicad_pcb  # noqa: E402


def count_valid_corpus_boards(degenerate: set[str]) -> tuple[int, int, int]:
    """Return `(total_files, valid_non_degen, scraped_prepped)`."""
    seen_all: set[str] = set()
    scraped_cnt = 0
    for d in sorted(list(CORPUS_DIR.glob("boards_unrouted_set*")) + list(CORPUS_DIR.glob("boards_unrouted_scraped_*"))):
        is_scr = d.name.startswith("boards_unrouted_scraped_")
        for f in d.glob("*.kicad_pcb"):
            if f.stat().st_size > 0:
                seen_all.add(f.stem)
                if is_scr:
                    scraped_cnt += 1
    valid_non_degen = len(seen_all - degenerate)
    return len(seen_all), valid_non_degen, scraped_cnt


_BOARD_META_CACHE: dict[str, tuple[float, int, tuple[int, int, int, int] | None]] = {}
_CACHE_LOADED = False


def _ensure_meta_cache_loaded() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    if META_CACHE_PATH.exists():
        try:
            raw = json.loads(META_CACHE_PATH.read_text(encoding="utf-8"))
            for k, v in (raw.get("entries") or {}).items():
                if isinstance(v, list) and len(v) == 3:
                    m, s, meta = v
                    _BOARD_META_CACHE[k] = (float(m), int(s), tuple(meta) if meta else None)
        except Exception:
            pass


def _save_meta_cache() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = META_CACHE_PATH.with_suffix(".tmp")
    serial = {k: [v[0], v[1], list(v[2]) if v[2] else None] for k, v in _BOARD_META_CACHE.items()}
    tmp.write_text(json.dumps({"version": 1, "entries": serial}), encoding="utf-8")
    os.replace(tmp, META_CACHE_PATH)


def list_candidate_boards(
    degenerate: set[str],
    large_val: set[str],
    max_parts: int,
    max_nets: int,
    min_parts: int = 6,
    skip_boards: set[str] | None = None,
) -> list[tuple[str, str, int, int, int]]:
    """Return all eligible `(set_name, board_name, parts, nets, layers)` across corpus."""
    _ensure_meta_cache_loaded()
    cache_dirty = False
    items: list[tuple[str, str, int, int, int]] = []
    seen: set[str] = set()
    dirs = sorted(list(CORPUS_DIR.glob("boards_unrouted_scraped_*")) + list(CORPUS_DIR.glob("boards_unrouted_set*")))
    for udir in dirs:
        if udir.name == "boards_unrouted_sethydro":
            continue
        sname = udir.name.replace("boards_unrouted_", "")
        for bf in sorted(udir.glob("*.kicad_pcb")):
            bname = bf.stem
            if bname in seen or bname in degenerate or bname in large_val or (skip_boards and bname in skip_boards):
                continue
            seen.add(bname)
            try:
                st = bf.stat()
                key = str(bf)
                cached = _BOARD_META_CACHE.get(key)
                if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
                    meta = cached[2]
                else:
                    pcb = parse_kicad_pcb(key)
                    meta = (
                        len(pcb.footprints),
                        len(pcb.nets),
                        len(getattr(pcb.board_info, "copper_layers", []) or ["F.Cu", "B.Cu"]),
                        len(getattr(pcb.board_info, "board_cutouts", []) or []),
                    )
                    _BOARD_META_CACHE[key] = (st.st_mtime, st.st_size, meta)
                    cache_dirty = True
            except Exception:
                _BOARD_META_CACHE[str(bf)] = (0.0, 0, None)
                cache_dirty = True
                continue
            if meta is None:
                continue
            n_parts, n_nets, n_layers, n_cutouts = meta
            if n_parts < min_parts or n_nets < 4 or n_parts > max_parts or n_nets > max_nets or n_cutouts > 6:
                continue
            # Also require corresponding routed board exists
            routed_f = CORPUS_DIR / f"boards_{sname}" / f"{bname}.kicad_pcb"
            if not routed_f.exists():
                continue
            items.append((sname, bname, n_parts, n_nets, n_layers))
    if cache_dirty:
        _save_meta_cache()
    # Prioritize newly scraped boards first, then smallest-to-largest backlog
    items.sort(key=lambda x: (0 if x[0].startswith("scraped_") else 1, x[2], x[3]))
    return items


_ACTIVE_PREP_PROCS: set[subprocess.Popen] = set()
_ACTIVE_PREP_LOCK = threading.Lock()


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Terminate `gym_rollout.py` gracefully so its signal handler kills nested `run_krt` groups, then force-kill."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=3.0)
    except Exception:
        pass


def _prune_dir_intermediates(workdir: Path) -> None:
    """Remove bulky `.kicad_pcb` and `*_route/` intermediates while keeping JSON/MD metadata."""
    if not workdir.exists():
        return
    for sub in list(workdir.iterdir()):
        if sub.is_dir() and "_route" in sub.name:
            shutil.rmtree(sub, ignore_errors=True)
        elif sub.is_file() and sub.suffix == ".kicad_pcb":
            try:
                sub.unlink()
            except Exception:
                pass


def prep_one_board(
    sname: str,
    bname: str,
    n_parts: int,
    n_nets: int,
    reuse_existing: bool = True,
) -> tuple[str, str, list[dict], str]:
    """Run ceiling + rollout (with caching) and format SFT record(s)."""
    ceil_dir = RUNS_DIR / f"ceiling_{bname}"
    roll_dir = RUNS_DIR / f"rollout_{bname}"
    ceil_json = ceil_dir / "ceiling_result.json"
    roll_json = roll_dir / "rollout_result.json"

    ceil_res = None
    if reuse_existing and ceil_json.exists():
        try:
            loaded = json.loads(ceil_json.read_text(encoding="utf-8"))
            if (loaded.get("score") or {}).get("blocking") is not None:
                ceil_res = loaded
        except Exception:
            pass
    if ceil_res is None:
        try:
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                str(WORKSPACE / "gym_rollout.py"),
                "--board", bname,
                "--set", sname,
                "--mode", "human_ceiling",
                "--corpus", str(CORPUS_DIR),
                "--workdir", str(ceil_dir),
            ]
            proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with _ACTIVE_PREP_LOCK:
                _ACTIVE_PREP_PROCS.add(proc)
            try:
                proc.wait(timeout=420)
            except subprocess.TimeoutExpired:
                _kill_proc_tree(proc)
            finally:
                with _ACTIVE_PREP_LOCK:
                    _ACTIVE_PREP_PROCS.discard(proc)
            if ceil_json.exists():
                ceil_res = json.loads(ceil_json.read_text(encoding="utf-8"))
                if (ceil_res.get("score") or {}).get("blocking") is not None:
                    _prune_dir_intermediates(ceil_dir)
        except Exception:
            pass

    roll_res = None
    if reuse_existing and roll_json.exists():
        try:
            loaded = json.loads(roll_json.read_text(encoding="utf-8"))
            if (loaded.get("best_score") or {}).get("blocking") is not None:
                roll_res = loaded
        except Exception:
            pass
    if roll_res is None:
        scat = "1"
        clus = "1"
        anch = "1" if n_parts <= 110 else "0"
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(WORKSPACE / "gym_rollout.py"),
            "--board", bname,
            "--set", sname,
            "--mode", "rollout",
            "--corpus", str(CORPUS_DIR),
            "--workdir", str(roll_dir),
            "--scattered", scat,
            "--clustered", clus,
            "--anchored", anch,
            "--judge", "2",
            "--jobs", "2",
        ]
        proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with _ACTIVE_PREP_LOCK:
            _ACTIVE_PREP_PROCS.add(proc)
        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            _kill_proc_tree(proc)
            _prune_dir_intermediates(roll_dir)
            return (sname, bname, [], "timeout")
        finally:
            with _ACTIVE_PREP_LOCK:
                _ACTIVE_PREP_PROCS.discard(proc)
        if roll_json.exists():
            roll_res = json.loads(roll_json.read_text(encoding="utf-8"))

    if not roll_res:
        _prune_dir_intermediates(roll_dir)
        return (sname, bname, [], "no_rollout_result")

    try:
        recs = format_sft_record(bname, roll_dir, roll_res, ceil_res, corpus_dir=CORPUS_DIR, set_name=sname)
        for r in recs:
            r["set"] = sname
        blk = (roll_res.get("best_score") or {}).get("blocking")
        _prune_dir_intermediates(ceil_dir)
        _prune_dir_intermediates(roll_dir)
        return (sname, bname, recs, f"ok:blk={blk}:rw={roll_res.get('rl_reward')}")
    except Exception as exc:
        return (sname, bname, [], f"format_err:{exc}")


def merge_and_curate(records_by_key: dict[tuple[str, int], dict]) -> dict:
    tmp = SFT_OUT_PATH.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in records_by_key.values()) + "\n", encoding="utf-8")
    os.replace(tmp, SFT_OUT_PATH)
    subprocess.run(
        [sys.executable, "-X", "utf8", str(WORKSPACE / "curate_sft_dataset.py"), "--input", str(SFT_OUT_PATH), "--out-dir", str(DATASET_DIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    rep_path = DATASET_DIR / "curation_report.json"
    if rep_path.exists():
        return json.loads(rep_path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run parallel KiCad scraping (to 1000+ boards) and SFT/GRPO prep concurrently")
    ap.add_argument("--target-corpus-boards", type=int, default=1000, help="Target non-degenerate validated corpus boards")
    ap.add_argument("--prep-workers", type=int, default=3, help="Concurrent rollout prep workers")
    ap.add_argument("--max-parts", type=int, default=500, help="Max part count for rollout prep")
    ap.add_argument("--max-nets", type=int, default=380, help="Max net count for rollout prep")
    ap.add_argument("--delay", type=float, default=2.85, help="Strict HTTP/GitHub throttle (>=2.8s)")
    ap.add_argument("--min-tier", choices=["small", "medium", "large", "monster"], default="medium")
    ap.add_argument("--max-per-repo", type=int, default=4)
    ap.add_argument("--cooldown-sec", type=int, default=120, help="Cooldown if GitHub search rate limits pause scraper")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    degenerate, large_val, _ = load_exclusion_sets()

    # Load existing merged SFT pool (sft_v8.jsonl or sft_v7.jsonl)
    records_by_key: dict[tuple[str, int], dict] = {}
    seed_file = SFT_OUT_PATH if SFT_OUT_PATH.exists() else SFT_BASE_PATH
    if seed_file.exists():
        for line in seed_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records_by_key[(str(r.get("board")), int(r.get("turn", 1)))] = r
    attempted_boards: set[str] = {b for b, _t in records_by_key}

    stop_scraper = threading.Event()
    scraper_state: dict = {"runs": 0, "done": False, "last_rc": None, "proc": None}

    def _cleanup_children(*_args) -> None:
        stop_scraper.set()
        proc = scraper_state.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
        with _ACTIVE_PREP_LOCK:
            procs = list(_ACTIVE_PREP_PROCS)
        for p in procs:
            if p.poll() is None:
                _kill_proc_tree(p)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _cleanup_children)
    signal.signal(signal.SIGINT, _cleanup_children)

    def _scraper_supervisor() -> None:
        while not stop_scraper.is_set():
            _, valid_cnt, scraped_cnt = count_valid_corpus_boards(degenerate)
            if valid_cnt >= args.target_corpus_boards:
                print(f"[SCRAPER] Reached target valid corpus boards: {valid_cnt}/{args.target_corpus_boards}!", flush=True)
                scraper_state["done"] = True
                break
            scraper_state["runs"] += 1
            needed_scraped = max(scraped_cnt + 1, args.target_corpus_boards - (valid_cnt - scraped_cnt))
            # Relax min-tier to 'small' automatically if we exhausted medium+ pending repos
            cur_min_tier = args.min_tier
            mf_path = WORKSPACE / "runs" / "scraper_logs" / "scraper_manifest.json"
            if mf_path.exists():
                try:
                    mf = json.loads(mf_path.read_text(encoding="utf-8"))
                    pend = sum(1 for r in (mf.get("repos") or {}).values() if r.get("status") == "pending")
                    if pend == 0 and "boardrepo:sitemap_xml" in (mf.get("completed_queries") or []):
                        cur_min_tier = "small"
                except Exception:
                    pass
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                str(WORKSPACE / "scrape_kicad_repos.py"),
                "--resume",
                "--skip-discovery-if-pending",
                "--sources", "boardrepo", "github_search", "curated_lists", "curated_orgs",
                "--discovery-cap", "12000",
                "--min-tier", cur_min_tier,
                "--max-per-repo", str(args.max_per_repo),
                "--max-repos", "0",
                "--target-corpus-total", str(args.target_corpus_boards),
                "--target-boards", str(needed_scraped),
                "--delay", str(max(2.85, args.delay)),
            ]
            print(f"[SCRAPER] Launching pass #{scraper_state['runs']} (valid={valid_cnt}/{args.target_corpus_boards}, min_tier={cur_min_tier})...", flush=True)
            scraper_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"}
            proc = subprocess.Popen(cmd, cwd=str(WORKSPACE), env=scraper_env, start_new_session=True)
            scraper_state["proc"] = proc
            rc = proc.wait()
            scraper_state["proc"] = None
            scraper_state["last_rc"] = rc
            _, valid_after, _ = count_valid_corpus_boards(degenerate)
            if valid_after >= args.target_corpus_boards:
                scraper_state["done"] = True
                break
            if stop_scraper.is_set():
                break
            print(f"[SCRAPER] Pass ended (rc={rc}, valid={valid_after}/{args.target_corpus_boards}); cooling down {args.cooldown_sec}s before resuming...", flush=True)
            time.sleep(args.cooldown_sec)

    scraper_thread = threading.Thread(target=_scraper_supervisor, daemon=True)
    scraper_thread.start()

    prepped_this_session = 0
    in_flight: dict[concurrent.futures.Future, tuple[str, str]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.prep_workers)) as pool:
        last_merge_cnt = 0
        last_status_ts = 0.0
        while True:
            # Reap completed prep futures
            done_futs = [f for f in list(in_flight) if f.done()]
            for fut in done_futs:
                sn, bn = in_flight.pop(fut)
                try:
                    _, _, recs, msg = fut.result()
                    if recs:
                        for r in recs:
                            records_by_key[(str(r.get("board")), int(r.get("turn", 1)))] = r
                        prepped_this_session += 1
                    print(f"[PREP #{prepped_this_session}] {sn}/{bn} -> {msg} (total_sft_recs={len(records_by_key)})", flush=True)
                except Exception as exc:
                    print(f"[PREP-ERR] {sn}/{bn}: {exc}", flush=True)

            # Enqueue newly discovered / backlog boards up to worker capacity
            cands = list_candidate_boards(degenerate, large_val, args.max_parts, args.max_nets, skip_boards=attempted_boards)
            for sname, bname, np_cnt, nn_cnt, _nl in cands:
                if len(in_flight) >= args.prep_workers:
                    break
                if bname in attempted_boards:
                    continue
                attempted_boards.add(bname)
                fut = pool.submit(prep_one_board, sname, bname, np_cnt, nn_cnt, True)
                in_flight[fut] = (sname, bname)

            # Periodically merge + curate + write live status JSON
            now = time.time()
            if (prepped_this_session - last_merge_cnt >= 5) or (now - last_status_ts >= 30.0):
                cur_rep = {}
                if prepped_this_session > last_merge_cnt:
                    cur_rep = merge_and_curate(records_by_key)
                    last_merge_cnt = prepped_this_session
                elif (DATASET_DIR / "curation_report.json").exists():
                    cur_rep = json.loads((DATASET_DIR / "curation_report.json").read_text(encoding="utf-8"))

                total_files, valid_corpus, scraped_cnt = count_valid_corpus_boards(degenerate)
                status = {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "target_corpus_boards": args.target_corpus_boards,
                    "corpus_total_files": total_files,
                    "corpus_valid_non_degenerate": valid_corpus,
                    "scraped_prepped_count": scraped_cnt,
                    "scraper_done": scraper_state["done"],
                    "scraper_passes": scraper_state["runs"],
                    "sft_raw_records": len(records_by_key),
                    "sft_unique_boards": len({b for b, _ in records_by_key}),
                    "prep_in_flight": [f"{sn}/{bn}" for sn, bn in in_flight.values()],
                    "prep_eligible_remaining": sum(1 for _sn, bn, *_ in cands if bn not in attempted_boards),
                    "curation": {k: cur_rep.get(k) for k in ("curated_train", "curated_val", "grpo_board_prompts", "excluded_degenerate", "excluded_large_val")},
                }
                tmp_st = STATUS_PATH.with_suffix(".tmp")
                tmp_st.write_text(json.dumps(status, indent=2), encoding="utf-8")
                os.replace(tmp_st, STATUS_PATH)
                last_status_ts = now

            if scraper_state["done"] and not in_flight:
                remaining = [bn for _sn, bn, *_ in cands if bn not in attempted_boards]
                if not remaining:
                    break
            time.sleep(3.0)

    stop_scraper.set()
    scraper_thread.join(timeout=10.0)
    final_cur = merge_and_curate(records_by_key)
    total_files, valid_corpus, scraped_cnt = count_valid_corpus_boards(degenerate)
    print(f"\n=== ALL DONE: valid_corpus={valid_corpus}/{args.target_corpus_boards}, scraped={scraped_cnt}, sft_records={len(records_by_key)} ===", flush=True)
    print(json.dumps(final_cur, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
