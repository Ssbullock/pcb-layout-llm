#!/usr/bin/env python3
"""
curate_sft_dataset.py
---------------------
Step 4A: Filters raw rollouts (`dataset/sft_all.jsonl` + `runs/rollout_*`) into a
high-reward, rejection-sampled SFT + GRPO dataset for cloud GPU training
(Unsloth / TRL / Axolotl / HF Trainer).

Rejection-Sampling Filter Criteria:
  A rollout is admitted to the curated SFT training split ONLY if:
    1. `blocking == 0` (100% routed connectivity, 0 floorplan/assembly errors), OR
    2. `blocking <= 2` AND `rl_reward >= human_ceiling_reward` (beat or tied the
       human designer's placement on a hard board).
  And the assistant's JSON payload parses cleanly with valid `groups`, `intent`,
  and `judge_chain` keys.

Outputs written to `/Users/simon/Fine Tune/dataset/`:
  - `sft_curated_train.jsonl` : High-reward SFT training examples (90% split by board)
  - `sft_curated_val.jsonl`   : Held-out validation examples (10% split by board, no board leakage)
  - `grpo_prompts.jsonl`      : All valid board prompts + ground-truth metadata for Stage-2 GRPO RL
  - `curation_report.json`    : Detailed statistics on acceptance rate, token lengths, and reward distribution
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_INVENTORY_PATH = WORKSPACE / "runs" / "large_board_benchmark" / "corpus_inventory.json"
LARGE_VAL_MANIFEST_PATH = WORKSPACE / "runs" / "large_board_benchmark" / "large_val_manifest.json"
FRONTIER_PROMPTS_PATH = WORKSPACE / "runs" / "frontier_benchmark" / "prompts.json"

# Degenerate / near-zero-copper human boards flagged during corpus audit (must never enter train or GRPO)
DEFAULT_DEGENERATE_BOARDS = {
    "picodvi",
    "qmtech_k7",
    "open_weather_station",
    "g473_gen1",
    "mbira_esp_midi",
    "triad_actuator",
    "vigor_frontpanel",
    "openjoint",
    "cherrycam",
    "sx1262_re_433",
    "bms_sensor",
    "mez_rx",
}


def load_exclusion_sets() -> tuple[set[str], set[str], set[str]]:
    """Return `(degenerate_boards, large_val_boards, frontier_bench_boards)`."""
    degen = set(DEFAULT_DEGENERATE_BOARDS)
    if CORPUS_INVENTORY_PATH.exists():
        try:
            inv = json.loads(CORPUS_INVENTORY_PATH.read_text(encoding="utf-8"))
            rows = inv.get("boards", []) if isinstance(inv, dict) else inv
            for row in rows:
                if isinstance(row, dict) and row.get("degenerate_human_routing"):
                    degen.add(str(row.get("board")))
        except Exception:
            pass

    large_val: set[str] = set()
    if LARGE_VAL_MANIFEST_PATH.exists():
        try:
            lv = json.loads(LARGE_VAL_MANIFEST_PATH.read_text(encoding="utf-8"))
            entries = lv.get("boards", []) if isinstance(lv, dict) else lv
            for row in entries:
                if isinstance(row, dict) and row.get("board"):
                    large_val.add(str(row["board"]))
        except Exception:
            pass

    frontier_bench: set[str] = set()
    if FRONTIER_PROMPTS_PATH.exists():
        try:
            fp = json.loads(FRONTIER_PROMPTS_PATH.read_text(encoding="utf-8"))
            for row in fp:
                if isinstance(row, dict) and row.get("board") and row.get("split") == "blind_val":
                    frontier_bench.add(str(row["board"]))
        except Exception:
            pass

    return degen, large_val, frontier_bench


def is_high_quality_record(rec: dict) -> tuple[bool, str]:
    """Return (accepted, reason) for a record from sft_all.jsonl."""
    reward = rec.get("rl_reward")
    ceil = rec.get("human_ceiling_reward")
    blocking = rec.get("blocking")
    turn = rec.get("turn", 1)

    if reward is None or reward <= -400:
        return False, "failed_or_timeout"

    # Validate JSON structure of assistant response(s)
    msgs = rec.get("messages") or []
    if len(msgs) < 3:
        return False, "malformed_messages"

    try:
        t1_payload = json.loads(msgs[2]["content"])
        if not all(k in t1_payload for k in ("groups", "intent", "judge_chain")):
            return False, "missing_t1_keys"
        if not t1_payload["groups"].get("groups"):
            return False, "empty_groups"
    except Exception:
        return False, "invalid_t1_json"

    if turn == 2:
        # Turn 2 teaches diagnostic recovery (`route_review.py` -> `place_pose.py` / `--rip-blocker-nets`)
        # Admit Turn-2 examples whenever the rollout succeeded or had a small number of actionable open nets/swaps
        if blocking is not None and blocking <= 5:
            return True, f"turn2_diagnostic (blocking={blocking})"
        return False, f"turn2_too_many_blocking ({blocking})"

    # Turn 1 rejection sampling:
    if blocking == 0:
        return True, "zero_blocking"
    if blocking is not None and blocking <= 2 and ceil is not None and reward >= ceil:
        return True, f"beat_human_ceiling (blocking={blocking}, reward={reward} >= {ceil})"

    return False, f"suboptimal_rollout (blocking={blocking}, reward={reward}, ceiling={ceil})"


def board_is_val(board_name: str, val_ratio: float = 0.10) -> bool:
    """Deterministic hash split by board name so Turn 1 and Turn 2 of the same board never leak across splits."""
    h = int(hashlib.md5(board_name.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000) < int(val_ratio * 1000)


def main():
    ap = argparse.ArgumentParser(description="Curate high-reward SFT and GRPO datasets from sft_all.jsonl")
    ap.add_argument("--input", type=Path, default=Path("/Users/simon/Fine Tune/dataset/sft_all.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/Users/simon/Fine Tune/dataset"))
    ap.add_argument("--val-ratio", type=float, default=0.10)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    raw_records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    degenerate_boards, large_val_boards, frontier_bench_boards = load_exclusion_sets()

    train_records = []
    val_records = []
    grpo_prompts = []
    rejected = []
    reasons_count = {}
    excluded_degenerate = 0
    excluded_large_val = 0

    for rec in raw_records:
        bname = rec.get("board", "unknown")
        turn = rec.get("turn", 1)

        if bname in degenerate_boards:
            excluded_degenerate += 1
            reasons_count["excluded_degenerate_human_ref"] = reasons_count.get("excluded_degenerate_human_ref", 0) + 1
            rejected.append({"board": bname, "turn": turn, "reason": "excluded_degenerate_human_ref"})
            continue

        if bname in large_val_boards:
            excluded_large_val += 1
            reasons_count["excluded_large_val_holdout"] = reasons_count.get("excluded_large_val_holdout", 0) + 1
            rejected.append({"board": bname, "turn": turn, "reason": "excluded_large_val_holdout"})
            continue

        # Every non-held-out Turn-1 board with a valid prompt is eligible for Stage-2 GRPO exploration
        if turn == 1 and rec.get("messages") and len(rec["messages"]) >= 2 and bname not in frontier_bench_boards:
            grpo_prompts.append({
                "set": rec.get("set", "unknown"),
                "board": bname,
                "baseline_rollout_reward": rec.get("rl_reward"),
                "human_ceiling_reward": rec.get("human_ceiling_reward"),
                "baseline_blocking": rec.get("blocking"),
                "prompt": rec["messages"][:2],  # system + user
            })

        ok, reason = is_high_quality_record(rec)
        reasons_count[reason] = reasons_count.get(reason, 0) + 1
        if not ok:
            rejected.append({"board": bname, "turn": turn, "reason": reason})
            continue

        # Normalize assistant JSON envelope (`kind="floorplan-intent"`, `groups.schema=1`)
        try:
            t1_obj = json.loads(rec["messages"][2]["content"])
            if isinstance(t1_obj.get("groups"), dict):
                t1_obj["groups"].setdefault("schema", 1)
            if isinstance(t1_obj.get("intent"), dict):
                t1_obj["intent"].setdefault("schema", 1)
                t1_obj["intent"].setdefault("kind", "floorplan-intent")
            rec["messages"][2]["content"] = json.dumps(t1_obj, indent=2)
        except Exception:
            pass

        if bname in frontier_bench_boards or board_is_val(bname, args.val_ratio):
            val_records.append(rec)
        else:
            train_records.append(rec)

    # Guarantee at least 2 validation examples if dataset is non-empty
    if not val_records and len(train_records) >= 6:
        val_records = train_records[-2:]
        train_records = train_records[:-2]

    train_path = args.out_dir / "sft_curated_train.jsonl"
    val_path = args.out_dir / "sft_curated_val.jsonl"
    grpo_path = args.out_dir / "grpo_prompts.jsonl"
    report_path = args.out_dir / "curation_report.json"

    with train_path.open("w", encoding="utf-8") as fh:
        for r in train_records:
            fh.write(json.dumps(r) + "\n")

    with val_path.open("w", encoding="utf-8") as fh:
        for r in val_records:
            fh.write(json.dumps(r) + "\n")

    with grpo_path.open("w", encoding="utf-8") as fh:
        for p in grpo_prompts:
            fh.write(json.dumps(p) + "\n")

    # Compute token estimates (~4 chars/token)
    token_lengths = [
        sum(len(m.get("content", "")) for m in r["messages"]) // 4
        for r in (train_records + val_records)
    ]

    report = {
        "raw_examples": len(raw_records),
        "curated_total": len(train_records) + len(val_records),
        "curated_train": len(train_records),
        "curated_val": len(val_records),
        "grpo_board_prompts": len(grpo_prompts),
        "excluded_degenerate": excluded_degenerate,
        "excluded_large_val": excluded_large_val,
        "rejected_count": len(rejected),
        "acceptance_rate_pct": round(100.0 * (len(train_records) + len(val_records)) / max(len(raw_records), 1), 1),
        "token_length_stats": {
            "min": min(token_lengths) if token_lengths else 0,
            "median": sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0,
            "max": max(token_lengths) if token_lengths else 0,
        },
        "reasons_breakdown": reasons_count,
        "rejected_boards": rejected,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== SFT & GRPO Dataset Curation Complete ===")
    print(f"  Raw examples read     : {len(raw_records)}")
    print(f"  Curated SFT kept      : {len(train_records) + len(val_records)} ({report['acceptance_rate_pct']}%)")
    print(f"    -> Train split      : {train_path} ({len(train_records)} examples)")
    print(f"    -> Val split        : {val_path} ({len(val_records)} examples)")
    print(f"  Stage-2 GRPO prompts  : {grpo_path} ({len(grpo_prompts)} boards)")
    print(f"  Token length (est.)   : min={report['token_length_stats']['min']}, "
          f"median={report['token_length_stats']['median']}, max={report['token_length_stats']['max']}")


if __name__ == "__main__":
    main()
