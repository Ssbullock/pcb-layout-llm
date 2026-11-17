#!/usr/bin/env python3
"""
make_benchmark_chart.py
-----------------------
Renders a clean, minimal 2400x1350 (16:9) editorial benchmark chart
from `runs/frontier_vs_finetuned_summary.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path("/Users/simon/Fine Tune")
SUMMARY_PATH = WORKSPACE / "runs/frontier_vs_finetuned_summary.json"
OUT_PATH = WORKSPACE / "runs/frontier_vs_finetuned_chart.png"

S = 2
W, H = 2400 * S, 1350 * S

BG = (248, 250, 252)          # #F8FAFC
CARD_BG = (255, 255, 255)     # #FFFFFF
CARD_BORDER = (226, 232, 240) # #E2E8F0
INK_PRI = (15, 23, 42)        # #0F172A
INK_SEC = (100, 116, 139)     # #64748B
INK_MUTED = (148, 163, 184)   # #94A3B8
GRID_COL = (241, 245, 249)    # #F1F5F9
TRACK_BG = (241, 245, 249)    # #F1F5F9

ACCENT_HERO = (37, 99, 235)   # #2563EB
ACCENT_HUMAN = (71, 85, 105)  # #475569
BAR_BASE = (148, 163, 184)    # #94A3B8
BAR_ZERO = (203, 213, 225)    # #CBD5E1

LABELS = {
    "finetuned-32b": "Fine-Tuned 32B",
    "gemini-4-carbon": "Gemini 4 Carbon",
    "claude-opus-5.5": "Claude Opus 5.5",
    "gpt-6.1-sol-pro": "GPT-6.1 Sol Pro",
    "qwen-2.5-coder-32b-base": "Base 32B",
    "human": "Human",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size * S)
            except Exception:
                pass
    return ImageFont.load_default()


def bar_color(slug: str) -> tuple[int, int, int]:
    if slug == "finetuned-32b":
        return ACCENT_HERO
    if slug == "human":
        return ACCENT_HUMAN
    if slug == "qwen-2.5-coder-32b-base":
        return BAR_ZERO
    return BAR_BASE


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    bars: list[tuple[str, str, float, str]],
    max_val: float,
    ticks: list[tuple[float, str]],
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=16 * S, fill=CARD_BG, outline=CARD_BORDER, width=1 * S)

    draw.text((x0 + 36 * S, y0 + 28 * S), title, font=fonts["panel_title"], fill=INK_PRI)

    label_x = x0 + 36 * S
    bar_x0 = x0 + 285 * S
    bar_x1 = x1 - 165 * S
    bar_w_max = bar_x1 - bar_x0

    top_y = y0 + 86 * S
    axis_y = y1 - 40 * S
    avail_h = axis_y - top_y
    n = len(bars)
    row_step = avail_h / max(n, 1)
    bar_h = min(38 * S, int(row_step * 0.56))

    for tick_val, tick_lbl in ticks:
        gx = int(bar_x0 + bar_w_max * (tick_val / max_val))
        draw.line([(gx, top_y), (gx, axis_y - 4 * S)], fill=GRID_COL, width=1 * S)
        tw = draw.textlength(tick_lbl, font=fonts["tick"])
        draw.text((gx - tw / 2, axis_y + 6 * S), tick_lbl, font=fonts["tick"], fill=INK_MUTED)

    for idx, (slug, label, val, val_str) in enumerate(bars):
        cy = int(top_y + idx * row_step + row_step * 0.5)
        by0 = cy - bar_h // 2
        by1 = cy + bar_h // 2

        is_hero = slug == "finetuned-32b"
        col = bar_color(slug)

        lbl_font = fonts["label_bold"] if is_hero else fonts["label"]
        lbl_col = INK_PRI if is_hero else ACCENT_HUMAN
        draw.text((label_x, cy - 13 * S), label, font=lbl_font, fill=lbl_col)

        draw.rounded_rectangle([bar_x0, by0, bar_x1, by1], radius=6 * S, fill=TRACK_BG)

        w = max(8 * S, int(round(bar_w_max * (val / max_val)))) if val > 0 else 0
        if w > 0:
            draw.rounded_rectangle([bar_x0, by0, bar_x0 + w, by1], radius=6 * S, fill=col)

        val_x = bar_x0 + w + 14 * S
        val_font = fonts["val_bold"] if is_hero else fonts["val"]
        val_col = ACCENT_HERO if is_hero else INK_PRI
        draw.text((val_x, cy - 13 * S), val_str, font=val_font, fill=val_col)


def main() -> None:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    models = data["models"]
    human_rows = list(data["baselines"]["human_routed"].values())
    n_boards = len(human_rows)

    def stats_for(slug: str) -> dict:
        rows = list(models[slug].values())
        vj = sum(1 for r in rows if r.get("valid_json"))
        zb = sum(1 for r in rows if r.get("blocking") == 0)
        drc0 = sum(1 for r in rows if r.get("drc_errors") == 0)
        times = [r["timing_sec"] for r in rows if r.get("timing_sec") is not None]
        avg_t = sum(times) / len(times) if times else None
        return {
            "vj_pct": 100.0 * vj / len(rows),
            "vj_cnt": vj,
            "zb_pct": 100.0 * zb / len(rows),
            "zb_cnt": zb,
            "drc0_pct": 100.0 * drc0 / len(rows),
            "drc0_cnt": drc0,
            "avg_t": avg_t,
        }

    st = {s: stats_for(s) for s in ["finetuned-32b", "gemini-4-carbon", "claude-opus-5.5", "gpt-6.1-sol-pro", "qwen-2.5-coder-32b-base"]}
    h_drc0 = sum(1 for r in human_rows if r.get("drc_errors") == 0)
    h_drc0_pct = 100.0 * h_drc0 / n_boards

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": load_font(44, bold=True),
        "sub": load_font(23, bold=False),
        "panel_title": load_font(29, bold=True),
        "label": load_font(23, bold=False),
        "label_bold": load_font(23, bold=True),
        "val": load_font(23, bold=False),
        "val_bold": load_font(23, bold=True),
        "tick": load_font(17, bold=False),
    }

    # Clean header
    draw.rectangle([70 * S, 46 * S, 76 * S, 128 * S], fill=ACCENT_HERO)
    draw.text((96 * S, 44 * S), "Fine-Tuned Qwen2.5-Coder-32B vs. Frontier Models", font=fonts["title"], fill=INK_PRI)
    draw.text((96 * S, 102 * S), "23 unseen KiCad PCBs", font=fonts["sub"], fill=INK_SEC)

    # 2x2 grid
    pad_x = 70 * S
    gap = 36 * S
    pw = (W - 2 * pad_x - gap) // 2
    top_y = 166 * S
    bot_y = H - 56 * S
    ph = (bot_y - top_y - gap) // 2

    pct_ticks = [(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]
    time_ticks = [(0, "0s"), (30, "30s"), (60, "60s"), (90, "90s"), (120, "120s")]

    p1_bars = [
        (s, LABELS[s], st[s]["zb_pct"], f"{st[s]['zb_pct']:.0f}%  ({st[s]['zb_cnt']}/23)")
        for s in ["finetuned-32b", "gemini-4-carbon", "claude-opus-5.5", "gpt-6.1-sol-pro", "qwen-2.5-coder-32b-base"]
    ]
    draw_panel(draw, (pad_x, top_y, pad_x + pw, top_y + ph), "100% Routed Rate", p1_bars, 100.0, pct_ticks, fonts)

    p2_bars = [
        ("finetuned-32b", LABELS["finetuned-32b"], st["finetuned-32b"]["drc0_pct"], f"{st['finetuned-32b']['drc0_pct']:.0f}%  ({st['finetuned-32b']['drc0_cnt']}/23)"),
        ("gemini-4-carbon", LABELS["gemini-4-carbon"], st["gemini-4-carbon"]["drc0_pct"], f"{st['gemini-4-carbon']['drc0_pct']:.0f}%  ({st['gemini-4-carbon']['drc0_cnt']}/23)"),
        ("claude-opus-5.5", LABELS["claude-opus-5.5"], st["claude-opus-5.5"]["drc0_pct"], f"{st['claude-opus-5.5']['drc0_pct']:.0f}%  ({st['claude-opus-5.5']['drc0_cnt']}/23)"),
        ("human", LABELS["human"], h_drc0_pct, f"{h_drc0_pct:.0f}%  ({h_drc0}/23)"),
        ("gpt-6.1-sol-pro", LABELS["gpt-6.1-sol-pro"], st["gpt-6.1-sol-pro"]["drc0_pct"], f"{st['gpt-6.1-sol-pro']['drc0_pct']:.0f}%  ({st['gpt-6.1-sol-pro']['drc0_cnt']}/23)"),
    ]
    draw_panel(draw, (pad_x + pw + gap, top_y, pad_x + 2 * pw + gap, top_y + ph), "KiCad 0-DRC Rate", p2_bars, 100.0, pct_ticks, fonts)

    p3_bars = [
        (s, LABELS[s], st[s]["vj_pct"], f"{st[s]['vj_pct']:.0f}%  ({st[s]['vj_cnt']}/23)")
        for s in ["finetuned-32b", "gemini-4-carbon", "claude-opus-5.5", "gpt-6.1-sol-pro", "qwen-2.5-coder-32b-base"]
    ]
    draw_panel(draw, (pad_x, top_y + ph + gap, pad_x + pw, top_y + 2 * ph + gap), "Valid Schema Rate", p3_bars, 100.0, pct_ticks, fonts)

    p4_bars = [
        (s, LABELS[s], st[s]["avg_t"] or 0.0, f"{st[s]['avg_t']:.1f}s")
        for s in ["finetuned-32b", "gemini-4-carbon", "gpt-6.1-sol-pro", "claude-opus-5.5"]
    ]
    draw_panel(draw, (pad_x + pw + gap, top_y + ph + gap, pad_x + 2 * pw + gap, top_y + 2 * ph + gap), "Avg Routing Time", p4_bars, 125.0, time_ticks, fonts)

    final_img = img.resize((2400, 1350), Image.Resampling.LANCZOS)
    final_img.save(OUT_PATH, quality=98)
    print(f"Saved clean chart to {OUT_PATH}")


if __name__ == "__main__":
    main()
