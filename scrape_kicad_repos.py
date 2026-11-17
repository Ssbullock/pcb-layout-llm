#!/usr/bin/env python3
"""
scrape_kicad_repos.py
---------------------
KiCad repository scraper, 3-layer deduplicator, structural/routing validator,
and corpus-expansion pipeline for Step 1 of the PCB training plan.

Reuses / mirrors KiCadRoutingTools (KRT) stress-corpus standards:
  - validate_candidate.py : KiCad 6/7/8/9/10 format, valid Edge.Cuts,
    >= 15 pad-bearing footprints, >= 20 routable nets, >= 2 copper layers,
    human-routed copper (nonzero segments, non-WIP coverage)
  - dedupe_score.py       : 0.5*Jaccard(fp_names) + 0.5*Jaccard(nets) < 0.30
    plus source repo/URL, schematic/footprint UUID lineage, geometry+ref, SHA256
    against ALL existing boards in /Users/simon/Fine Tune/corpus/boards_set*
  - prep_set2.py          : KiCad pcbnew normalization -> boards_scraped_set1/
    and copper-stripped unrouted boards -> boards_unrouted_scraped_set1/

Strict Rate Limit Rule:
  Enforces >= 2.8 seconds sleep between EVERY GitHub web/API/git/raw request,
  supports checkpointed resumption via runs/scraper_logs/scraper_manifest.json,
  and handles HTTP 403/429 rate-limit backoff gracefully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Workspace & Tool Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
SOURCES_DIR = CORPUS_DIR / "sources" / "scraped_github"
STAGING_DIR = SOURCES_DIR / "_staging"
ROUTED_DIR = CORPUS_DIR / "boards_scraped_set1"
UNROUTED_DIR = CORPUS_DIR / "boards_unrouted_scraped_set1"

LOGS_DIR = WORKSPACE / "runs" / "scraper_logs"
MANIFEST_PATH = LOGS_DIR / "scraper_manifest.json"
SUMMARY_PATH = LOGS_DIR / "scrape_summary.json"
EVENTS_PATH = LOGS_DIR / "scrape_events.jsonl"
DEDUPE_CACHE_PATH = LOGS_DIR / "corpus_dedupe_cache.json"

KRT_ROOT = Path("/Users/simon/.gemini/config/tools/KiCadRoutingTools")
KRT_STRESS = KRT_ROOT / "tests" / "stress"
PREP_PY = KRT_STRESS / "prep_set2.py"
KPY = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"

for _d in (str(KRT_STRESS), str(KRT_ROOT / "py_router"), str(KRT_ROOT / "py_placer"), str(KRT_ROOT / "py_tools")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import dedupe_score as krt_dedupe  # noqa: E402
from kicad_parser import parse_kicad_pcb  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & Regexes
# ---------------------------------------------------------------------------
MIN_RATE_LIMIT_DELAY_S = 2.8
MIN_KICAD_VERSION = 20211014  # KiCad 6.0+
MIN_PAD_BEARING_FPS = 15
MIN_ROUTABLE_NETS = 20
MIN_SEGMENTS = 10
DEDUPE_BAR = 0.30
MAX_FILE_BYTES = 120 * 1024 * 1024
MIN_FILE_BYTES = 5 * 1024

VERSION_RE = re.compile(r"\(version\s+(\d+)\)")
PATH_UUID_RE = re.compile(r'\(path\s+"([^"]+)"\)')
FP_BLOCK_START_RE = re.compile(r"\((?:footprint|module)\s+")
FP_UUID_RE = re.compile(r'\((?:uuid|tstamp)\s+"?([0-9a-fA-F-]{8,})"?\)')
FP_REF_NEW_RE = re.compile(r'\(property\s+"Reference"\s+"([^"]+)"')
FP_REF_OLD_RE = re.compile(r'\(fp_text\s+reference\s+"?([^"\s()]+)"?')
SEGMENT_OR_ARC_RE = re.compile(r"\((?:segment|arc)\s+")
EDGE_LAYER_RE = re.compile(r'\(layer\s+"?Edge\.Cuts"?\)')
XY_PAIR_RE = re.compile(r"\((?:start|end|mid|center|xy)\s+([-\d.]+)\s+([-\d.]+)\)")

EXCLUDE_PATH_RE = re.compile(
    r"(?i)(?:^|/)(?:-?backups?|autosave[^/]*|tests?|fixtures?|testdata|test_data|unit_?tests?|spec|"
    r"demos?|templates?|examples?|samples?|qa|fuzz|debug|docs?|resources?|assets?|benchmarks?)/"
    r"|(?:-backups?/|autosave|panel(?:ized)?\.kicad_pcb$|(?:switch_?)?plate\.kicad_pcb$|(?:top_?|bottom_?)?case\.kicad_pcb$)"
)

SOFTWARE_REPO_RE = re.compile(
    r"(?i)\b(kicad[-_ ]?plugin|kicad[-_ ]?addon|kicad[-_ ]?librar(?:y|ies)|footprint[-_ ]?(?:generator|lib)|"
    r"symbol[-_ ]?librar|3d[-_ ]?models|svg2shenzhen|pcb[-_ ]?parser|kicad[-_ ]?scripts?|kicad[-_ ]?cli|"
    r"gerber[-_ ]?viewer|drc[-_ ]?checker|kicad[-_ ]?templates?|kicad[-_ ]?mcp|mcp[-_ ]?server|"
    r"pcbdraw|kikit|kicanvas|freerouting|skidl|tscircuit|pcb2blender|autorouter|awesome[-_ ])\b"
)

EXCLUDED_REPOS_OR_OWNERS = {
    "kicad",
    "tscircuit",
    "freerouting",
    "yaqwsx",
    "theacodes",
    "devbisme",
    "bouni",
    "easyw",
    "30350n",
    "sethhillbrand",
    "mixelpixx",
    "aklofas",
    "drandyhaas/kicadroutingtools",
    "espressif/kicad-libraries",
    "daprice/keyswitches.pretty",
    "kitspace/awesome-electronics",
    "way2pramil/awesome-kicad-projects",
}

# Curated open-source hardware catalogs & prolific open-hardware search queries
CURATED_CATALOG_URLS = [
    ("kitspace_awesome_electronics", "https://raw.githubusercontent.com/kitspace/awesome-electronics/master/README.md"),
    ("way2pramil_awesome_kicad", "https://raw.githubusercontent.com/way2pramil/awesome-kicad-projects/main/README.md"),
]

GITHUB_SEARCH_QUERIES = [
    # Modern KiCad 6/7/8/9/10 hardware queries (pushed:>2022-03-01, fork:false)
    ("topic_kicad_pcb", "topic:kicad-pcb pushed:>2022-06-01 fork:false stars:>2"),
    ("kicad_8_pcb", '"KiCad 8" pcb pushed:>2023-06-01 fork:false stars:>1'),
    ("kicad_7_pcb", '"KiCad 7" pcb pushed:>2022-06-01 fork:false stars:>1'),
    ("kicad_fpga", "kicad fpga pcb pushed:>2022-03-01 fork:false stars:>2"),
    ("kicad_stm32", "kicad stm32 pcb pushed:>2022-03-01 fork:false stars:>2"),
    ("kicad_rp2040", "kicad rp2040 pcb pushed:>2022-03-01 fork:false stars:>2"),
    ("kicad_esp32", "kicad esp32 pcb pushed:>2022-03-01 fork:false stars:>2"),
    ("kicad_4layer", 'kicad "4-layer" pushed:>2022-03-01 fork:false stars:>1'),
    ("kicad_6layer", 'kicad "6-layer" pushed:>2022-03-01 fork:false stars:>1'),
    ("kicad_ddr3", "kicad ddr3 pushed:>2022-03-01 fork:false stars:>1"),
    ("kicad_bga", "kicad bga pushed:>2022-03-01 fork:false stars:>1"),
    ("kicad_motor_bldc", "kicad bldc motor pushed:>2022-03-01 fork:false stars:>1"),
    ("kicad_sdr_rf", "kicad sdr rf pushed:>2022-03-01 fork:false stars:>1"),
    ("kicad_usb_pd", 'kicad "usb-pd" pushed:>2022-03-01 fork:false stars:>1'),
    ("kicad_can_bus", "kicad can-bus pushed:>2022-03-01 fork:false stars:>1"),
    ("kicad_eurorack", "kicad eurorack pushed:>2022-03-01 fork:false stars:>2"),
    ("kicad_flight_ctrl", 'kicad "flight controller" pushed:>2022-03-01 fork:false stars:>1'),
    ("topic_kicad_hardware", "topic:kicad topic:hardware pushed:>2022-06-01 fork:false stars:>3"),
    ("topic_open_hardware", "topic:open-hardware kicad pushed:>2022-06-01 fork:false stars:>3"),
    # Date-windowed shards (bypasses GitHub's 1000-result cap per single query)
    ("kicad_pcb_2025_2026", "topic:kicad-pcb pushed:>=2025-01-01 fork:false"),
    ("kicad_pcb_2024", "topic:kicad-pcb pushed:2024-01-01..2024-12-31 fork:false"),
    ("kicad_pcb_2023", "topic:kicad-pcb pushed:2023-01-01..2023-12-31 fork:false"),
    ("kicad_pcb_2022", "topic:kicad-pcb pushed:2022-01-01..2022-12-31 fork:false"),
    ("kicad_hw_2024_lowstar", "topic:kicad pushed:>=2024-01-01 fork:false stars:0..3"),
    ("kicad_hw_2023_lowstar", "topic:kicad pushed:2023-01-01..2023-12-31 fork:false stars:0..3"),
]

# Non-repo top-level URL segments in https://boardrepo.com/sitemap.xml
BOARDREPO_SKIP_OWNERS = {
    "about", "blog", "brand", "browse", "changelog", "compare", "component", "components",
    "connect", "contact", "docs", "guides", "news", "packages", "pricing", "privacy",
    "sync", "tag", "tags", "teams", "terms", "trust", "wol",
}
BOARDREPO_PRIORITY_OWNERS = {
    "antmicro", "olimex", "electroniccats", "greatscottgadgets", "mlab-modules",
    "prusa3d", "newaetech", "tinyvision-ai-inc", "librespacefoundation",
    "1bitsquared", "machdyne", "mntmn", "pine64", "badgeteam",
    "opendrone-hw", "skiselev", "ploopyco", "boardoza", "zenitesolar",
}
BOARDREPO_EAGLE_HEAVY_OWNERS = {
    "adafruit", "sparkfun", "solderedelectronics",
}

CURATED_ORG_QUERIES = [
    ("org_olimex", "org:OLIMEX kicad"),
    ("org_mntmn", "org:mntmn kicad"),
    ("org_machdyne", "org:machdyne kicad"),
    ("org_pine64", "org:pine64 kicad"),
    ("org_antmicro", "org:antmicro kicad"),
    ("org_librespace", "org:librespacefoundation kicad"),
    ("org_1bitsquared", "org:1bitsquared kicad"),
    ("org_newaetech", "org:newaetech kicad"),
    ("org_tinyvision", "org:tinyvision-ai-inc kicad"),
    ("org_bigtreetech", "org:bigtreetech pcb"),
    ("org_bitcraze", "org:bitcraze hardware"),
    ("org_simplefoc", "org:simplefoc kicad"),
    ("org_openephys", "org:open-ephys kicad"),
    ("org_enjoy_digital", "org:enjoy-digital hardware"),
    ("org_lambdaconcept", "org:lambdaconcept kicad"),
    ("org_unexpectedmaker", "org:UnexpectedMaker pcb"),
    ("org_meshtastic", "org:meshtastic pcb"),
    ("org_voron", "org:VoronDesign pcb"),
    ("org_greatscott", "org:greatscottgadgets hardware"),
    ("org_glasgow", "org:GlasgowEmbedded kicad"),
    ("org_electroniccats", "org:ElectronicCats kicad"),
    ("org_soldered", "org:SolderedElectronics kicad"),
]

NON_HW_REPO_RE = re.compile(
    r"(?i)\b(firmware|bootloader|sdk|documentation|docs|website|jupyter|tutorial|workshop|guide|examples?)\b"
)
HW_KEYWORD_RE = re.compile(r"(?i)\b(kicad|pcb|hardware|schematic|layout|board|rev\d|open-?hardware)\b")

TIER_ORDER = {"small": 0, "medium": 1, "large": 2, "monster": 3}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(event_type: str, **payload: Any) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utc_now(), "event": event_type, **payload}
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


class RateLimitExhausted(RuntimeError):
    def __init__(self, wait_s: float, message: str):
        super().__init__(message)
        self.wait_s = wait_s


# ---------------------------------------------------------------------------
# Strict Throttled Client (>= 2.8s between EVERY GitHub/web/git request)
# ---------------------------------------------------------------------------
class ThrottledClient:
    def __init__(self, min_delay_s: float = 2.85, max_rate_limit_wait_s: float = 120.0):
        self.min_delay_s = max(MIN_RATE_LIMIT_DELAY_S, float(min_delay_s))
        self.max_rate_limit_wait_s = float(max_rate_limit_wait_s)
        self.last_req_end_ts: float | None = None
        self.total_requests = 0
        self.total_sleep_s = 0.0
        self.min_observed_gap_s: float | None = None
        self.rate_limit_hits = 0
        self.token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        self.core_remaining: int | None = None

    def wait_slot(self, label: str) -> float:
        """Strictly wait >= min_delay_s since the previous request completed."""
        now = time.monotonic()
        slept = 0.0
        if self.last_req_end_ts is not None:
            gap = now - self.last_req_end_ts
            if gap < self.min_delay_s:
                slept = self.min_delay_s - gap
                time.sleep(slept)
                self.total_sleep_s += slept
            actual_gap = time.monotonic() - self.last_req_end_ts
            if self.min_observed_gap_s is None or actual_gap < self.min_observed_gap_s:
                self.min_observed_gap_s = actual_gap
        return slept

    def _mark_done(self) -> None:
        self.last_req_end_ts = time.monotonic()
        self.total_requests += 1

    def http_get(self, url: str, timeout: int = 30, max_retries: int = 3) -> tuple[int, dict[str, str], bytes]:
        headers = {
            "User-Agent": "KiCadCorpusScraper/1.0 (OpenHardwareResearch)",
            "Accept": "application/vnd.github+json" if "api.github.com" in url else "*/*",
        }
        if self.token and "api.github.com" in url:
            headers["Authorization"] = f"Bearer {self.token}"

        for attempt in range(max_retries + 1):
            self.wait_slot(url)
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    hdrs = {k.lower(): v for k, v in resp.headers.items()}
                    self._mark_done()
                    if "api.github.com" in url and "/search/" not in url and "x-ratelimit-remaining" in hdrs:
                        try:
                            self.core_remaining = int(hdrs["x-ratelimit-remaining"])
                        except ValueError:
                            pass
                    return resp.status, hdrs, body
            except urllib.error.HTTPError as e:
                hdrs = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
                self._mark_done()
                if e.code in (403, 429, 503):
                    self.rate_limit_hits += 1
                    retry_after = hdrs.get("retry-after")
                    rem = hdrs.get("x-ratelimit-remaining")
                    reset = hdrs.get("x-ratelimit-reset")
                    if retry_after and retry_after.isdigit():
                        wait_s = float(int(retry_after) + 2)
                    elif rem == "0" and reset and reset.isdigit():
                        wait_s = max(5.0, float(int(reset) - int(time.time()) + 3))
                    else:
                        wait_s = min(90.0, 15.0 * (2 ** attempt))
                    wait_s = max(self.min_delay_s, wait_s)
                    log_event("rate_limit_backoff", url=url, status=e.code, wait_s=round(wait_s, 1), attempt=attempt)
                    if attempt >= max_retries or wait_s > self.max_rate_limit_wait_s:
                        raise RateLimitExhausted(wait_s, f"HTTP {e.code} on {url} (wait={wait_s:.0f}s)")
                    print(f"  [RATE-LIMIT] HTTP {e.code} on {url[:70]} -> sleeping {wait_s:.1f}s...", flush=True)
                    time.sleep(wait_s)
                    self.total_sleep_s += wait_s
                    self.last_req_end_ts = time.monotonic()
                    continue
                if e.code == 404:
                    return 404, hdrs, b""
                if attempt < max_retries and e.code >= 500:
                    time.sleep(self.min_delay_s)
                    continue
                return e.code, hdrs, b""
            except Exception as e:
                self._mark_done()
                log_event("http_error", url=url, error=str(e), attempt=attempt)
                if attempt >= max_retries:
                    return 0, {}, b""
                time.sleep(self.min_delay_s)
        return 0, {}, b""

    def download_file(self, url: str, dest: Path, timeout: int = 60) -> bool:
        status, _, body = self.http_get(url, timeout=timeout)
        if status != 200 or not body:
            return False
        # Guard against HTML error pages saved as .kicad_pcb
        head = body[:256].lstrip().lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return True

    def list_repo_tree(self, repo: str, default_branch: str | None = None) -> tuple[str | None, str | None, list[str], str]:
        """Return (branch, commit_sha, file_paths, method_or_err).

        Uses GitHub Trees API when GITHUB_TOKEN is available (fast + avoids git overhead),
        otherwise uses blobless shallow `git clone --filter=blob:none --no-checkout`
        which bypasses the 60 req/hr unauthenticated REST core limit while still
        honoring the >= 2.8s throttle.
        """
        if self.token and (self.core_remaining is None or self.core_remaining > 15) and default_branch:
            api_url = f"https://api.github.com/repos/{repo}/git/trees/{urllib.parse.quote(default_branch)}?recursive=1"
            st, _, body = self.http_get(api_url)
            if st == 200 and body:
                try:
                    data = json.loads(body.decode("utf-8", "replace"))
                    sha = data.get("sha")
                    paths = [t["path"] for t in data.get("tree", []) if t.get("type") == "blob" and "path" in t]
                    return default_branch, sha, paths, "api_tree"
                except Exception:
                    pass

        # Fallback: blobless shallow clone (downloads only commit + tree metadata, 0 blobs)
        self.wait_slot(f"git:{repo}")
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        with tempfile.TemporaryDirectory(prefix="kicad_tree_") as td:
            cmd = [
                "git", "clone", "--depth", "1", "--filter=blob:none",
                "--no-checkout", "--quiet",
                f"https://github.com/{repo}.git", td,
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=45, env=env)
                self._mark_done()
            except subprocess.TimeoutExpired:
                self._mark_done()
                return None, None, [], "git_clone_timeout"
            if r.returncode != 0:
                err = (r.stderr or "").strip().splitlines()[-1] if r.stderr else f"rc={r.returncode}"
                return None, None, [], f"git_clone_failed:{err[:100]}"
            try:
                sha = subprocess.check_output(["git", "-C", td, "rev-parse", "HEAD"], text=True, timeout=10).strip()
                br_res = subprocess.run(["git", "-C", td, "symbolic-ref", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
                branch = br_res.stdout.strip() if br_res.returncode == 0 else (default_branch or "HEAD")
                # IMPORTANT: never pass --long on a blobless clone (it triggers 1 network fetch per blob)
                ls_out = subprocess.check_output(["git", "-C", td, "ls-tree", "-r", "--name-only", "HEAD"], text=True, timeout=15)
                paths = [line.strip() for line in ls_out.splitlines() if line.strip()]
                return branch, sha, paths, "git_blobless"
            except Exception as e:
                return None, None, [], f"git_tree_err:{e}"

    def stats(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "min_delay_configured_s": self.min_delay_s,
            "min_gap_observed_s": round(self.min_observed_gap_s, 3) if self.min_observed_gap_s is not None else None,
            "total_sleep_s": round(self.total_sleep_s, 2),
            "rate_limit_hits": self.rate_limit_hits,
            "authenticated": bool(self.token),
        }


# ---------------------------------------------------------------------------
# 3-Layer Deduplication Index Against All ~474 Existing Corpus Boards
# ---------------------------------------------------------------------------
def extract_board_signatures(path: Path, dims_override: tuple[float, float] | None = None) -> dict[str, Any]:
    """Extract dedupe_score tokens, schematic/footprint UUIDs, refs, geometry, and sha256."""
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")

    fps = sorted(
        {m.lower() for m in krt_dedupe.FP_QUOTED_RE.findall(text)}
        | {m.lower() for m in krt_dedupe.FP_BARE_RE.findall(text)}
    )
    nets = sorted({m.lower() for m in krt_dedupe.NET_RE.findall(text)} - krt_dedupe.GENERIC_NETS)

    # Schematic symbol path UUIDs (e.g. "/5d9fa4e3-.../3b114248-...")
    sym_uuids: set[str] = set()
    for p in PATH_UUID_RE.findall(text):
        for seg in p.strip("/").split("/"):
            if len(seg) >= 12 and seg != "00000000-0000-0000-0000-000000000000":
                sym_uuids.add(seg.lower())

    # Footprint-level UUIDs & reference designators (scan only footprint headers, not segments)
    fp_uuids: set[str] = set()
    refs: set[str] = set()
    for m in FP_BLOCK_START_RE.finditer(text):
        chunk = text[m.start():m.start() + 900]
        um = FP_UUID_RE.search(chunk)
        if um:
            u = um.group(1).lower()
            if u != "00000000-0000-0000-0000-000000000000":
                fp_uuids.add(u)
        rm = FP_REF_NEW_RE.search(chunk) or FP_REF_OLD_RE.search(chunk)
        if rm:
            r = rm.group(1).strip()
            if r and not r.startswith("#") and r != "REF**":
                refs.add(r.upper())

    dims_mm = dims_override
    if dims_mm is None:
        xs: list[float] = []
        ys: list[float] = []
        n_text = len(text)
        for m in EDGE_LAYER_RE.finditer(text):
            i = m.start() - 1
            min_i = max(0, m.start() - 250_000)
            depth = 0
            while i >= min_i:
                ch = text[i]
                if ch == ")":
                    depth += 1
                elif ch == "(":
                    if depth == 0:
                        break
                    depth -= 1
                i -= 1
            if i < min_i or not text.startswith(("(gr_line", "(gr_rect", "(gr_arc", "(gr_circle", "(gr_poly"), i):
                continue
            j = i + 1
            depth = 1
            max_j = min(n_text, i + 250_000)
            while j < max_j and depth > 0:
                ch = text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            blk = text[i:j]
            for xm in XY_PAIR_RE.finditer(blk):
                try:
                    xs.append(float(xm.group(1)))
                    ys.append(float(xm.group(2)))
                except ValueError:
                    pass
        if xs and ys:
            dims_mm = (round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2))

    return {
        "fp_tokens": fps,
        "net_tokens": nets,
        "sym_uuids": sorted(sym_uuids),
        "fp_uuids": sorted(fp_uuids),
        "refs": sorted(refs),
        "n_refs": len(refs),
        "dims_mm": list(dims_mm) if dims_mm else None,
        "sha256": sha256,
    }


def _scraped_cache_path() -> Path:
    return LOGS_DIR / "scraped_dedupe_cache.json"


def _load_scraped_sig_cache() -> dict[str, Any]:
    p = _scraped_cache_path()
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if raw.get("version") == 1:
                return dict(raw.get("entries") or {})
        except Exception:
            pass
    return {}


def _save_scraped_sig_cache(entries: dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    p = _scraped_cache_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "count": len(entries), "entries": entries}), encoding="utf-8")
    os.replace(tmp, p)


def _extract_scraped_sig_cached(key: str, path: Path, cache: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    st = path.stat()
    mtime, size = int(st.st_mtime), st.st_size
    c = cache.get(key)
    if c and c.get("mtime") == mtime and c.get("size") == size:
        return c, False
    sig = extract_board_signatures(path)
    sig["key"] = key
    sig["mtime"] = mtime
    sig["size"] = size
    cache[key] = sig
    return sig, True


def _jaccard_sets(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CorpusDedupeIndex:
    """Zero-overlap deduplication against all existing corpus boards and manifests."""

    def __init__(self) -> None:
        self.existing_repos: set[str] = set()
        self.existing_urls: set[str] = set()
        self.existing_short_names: set[str] = set()
        self.boards: list[dict[str, Any]] = []
        self._load_manifests()
        self._load_or_build_board_cache()

    def _load_manifests(self) -> None:
        for mf in sorted(KRT_STRESS.glob("manifest_*.json")):
            try:
                entries = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for e in entries:
                repo = (e.get("repo") or "").strip().lower()
                if repo:
                    self.existing_repos.add(repo)
                for k in ("github_url", "raw_url", "archive_url"):
                    u = (e.get(k) or "").strip().lower()
                    if u:
                        self.existing_urls.add(u)
                sn = (e.get("short_name") or "").strip().lower()
                if sn:
                    self.existing_short_names.add(sn)
                if e.get("file"):
                    self.existing_short_names.add(Path(e["file"]).stem.lower())

        # Also record all existing board stems in corpus/boards_set* (incl. boards_sethydro)
        for d in sorted(CORPUS_DIR.glob("boards_set*")):
            for p in d.glob("*.kicad_pcb"):
                self.existing_short_names.add(p.stem.lower())

    def _load_or_build_board_cache(self) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        existing_files = sorted(
            p for d in sorted(CORPUS_DIR.glob("boards_set*")) for p in d.glob("*.kicad_pcb")
        )
        file_meta = {
            f"{p.parent.name}/{p.stem}": (str(p), int(p.stat().st_mtime), p.stat().st_size)
            for p in existing_files
        }

        cached: dict[str, Any] = {}
        if DEDUPE_CACHE_PATH.exists():
            try:
                raw_cache = json.loads(DEDUPE_CACHE_PATH.read_text(encoding="utf-8"))
                if raw_cache.get("version") == 1:
                    cached = raw_cache.get("entries", {})
            except Exception:
                cached = {}

        updated_entries: dict[str, Any] = {}
        rebuilt = 0
        for key, (fpath, mtime, size) in file_meta.items():
            c = cached.get(key)
            if c and c.get("mtime") == mtime and c.get("size") == size:
                updated_entries[key] = c
            else:
                sig = extract_board_signatures(Path(fpath))
                sig["key"] = key
                sig["mtime"] = mtime
                sig["size"] = size
                updated_entries[key] = sig
                rebuilt += 1

        if rebuilt > 0 or not DEDUPE_CACHE_PATH.exists():
            tmp = DEDUPE_CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"version": 1, "count": len(updated_entries), "entries": updated_entries}),
                encoding="utf-8",
            )
            os.replace(tmp, DEDUPE_CACHE_PATH)

        self.boards = [self._hydrate(v) for v in updated_entries.values()]
        print(
            f"[DEDUPE] Indexed {len(self.boards)} corpus boards from boards_set* "
            f"(rebuilt={rebuilt}, known_repos={len(self.existing_repos)}, known_urls={len(self.existing_urls)})",
            flush=True,
        )

    @staticmethod
    def _hydrate(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": entry["key"],
            "fp_set": set(entry["fp_tokens"]),
            "net_set": set(entry["net_tokens"]),
            "sym_uuids": set(entry["sym_uuids"]),
            "fp_uuids": set(entry["fp_uuids"]),
            "refs": set(entry["refs"]),
            "n_refs": entry["n_refs"],
            "dims_mm": tuple(entry["dims_mm"]) if entry.get("dims_mm") else None,
            "sha256": entry["sha256"],
        }

    def add_admitted_board(self, key: str, sig: dict[str, Any], repo: str, urls: list[str], short_name: str) -> None:
        if repo:
            self.existing_repos.add(repo.lower())
        for u in urls:
            if u:
                self.existing_urls.add(u.lower())
        self.existing_short_names.add(short_name.lower())
        sig_copy = dict(sig)
        sig_copy["key"] = key
        self.boards.append(self._hydrate(sig_copy))

    def is_known_repo(self, repo: str, parent_repo: str | None = None) -> tuple[bool, str | None]:
        rl = repo.strip().lower()
        if rl in self.existing_repos:
            return True, f"repo_in_corpus:{repo}"
        if parent_repo and parent_repo.strip().lower() in self.existing_repos:
            return True, f"parent_repo_in_corpus:{parent_repo}"
        return False, None

    def check_candidate(
        self,
        pcb_path: Path,
        repo: str,
        urls: list[str],
        dims_override: tuple[float, float] | None = None,
    ) -> tuple[bool, str | None, dict[str, Any], dict[str, Any]]:
        """Check candidate against all indexed boards.

        Returns (is_duplicate, reason, details, extracted_sig).
        """
        if repo.lower() in self.existing_repos:
            return True, "dup_source_repo", {"repo": repo}, {}
        for u in urls:
            if u and u.lower() in self.existing_urls:
                return True, "dup_source_url", {"url": u}, {}

        sig = extract_board_signatures(pcb_path, dims_override=dims_override)
        cand = self._hydrate({**sig, "key": pcb_path.stem})

        best_score = 0.0
        best_match = ""
        best_sym_overlap = 0
        best_fp_overlap = 0

        for b in self.boards:
            if cand["sha256"] == b["sha256"]:
                return True, f"dup_sha256:{b['key']}", {"match": b["key"], "score": 1.0}, sig

            # 1. KRT dedupe_score.py metric
            s = 0.5 * _jaccard_sets(cand["fp_set"], b["fp_set"]) + 0.5 * _jaccard_sets(cand["net_set"], b["net_set"])
            if s > best_score:
                best_score = s
                best_match = b["key"]

            # 2. UUID lineage check (schematic symbol path UUIDs + footprint UUIDs)
            sym_shared = len(cand["sym_uuids"] & b["sym_uuids"]) if (cand["sym_uuids"] and b["sym_uuids"]) else 0
            fp_shared = len(cand["fp_uuids"] & b["fp_uuids"]) if (cand["fp_uuids"] and b["fp_uuids"]) else 0
            if sym_shared > best_sym_overlap:
                best_sym_overlap = sym_shared
            if fp_shared > best_fp_overlap:
                best_fp_overlap = fp_shared

            if sym_shared >= 5 or fp_shared >= 5:
                return (
                    True,
                    f"dup_uuid_lineage:{b['key']}",
                    {
                        "match": b["key"],
                        "dedupe_score": round(s, 4),
                        "sym_uuids_shared": sym_shared,
                        "fp_uuids_shared": fp_shared,
                    },
                    sig,
                )

            # 3. Geometry + reference designator signature
            if cand["dims_mm"] and b["dims_mm"] and cand["n_refs"] >= 15 and b["n_refs"] >= 15:
                dw = abs(cand["dims_mm"][0] - b["dims_mm"][0])
                dh = abs(cand["dims_mm"][1] - b["dims_mm"][1])
                # Also check 90-degree rotated match
                dw_rot = abs(cand["dims_mm"][0] - b["dims_mm"][1])
                dh_rot = abs(cand["dims_mm"][1] - b["dims_mm"][0])
                dim_close = (dw <= 0.6 and dh <= 0.6) or (dw_rot <= 0.6 and dh_rot <= 0.6)
                if dim_close and abs(cand["n_refs"] - b["n_refs"]) <= 3:
                    ref_j = _jaccard_sets(cand["refs"], b["refs"])
                    if ref_j >= 0.85:
                        return (
                            True,
                            f"dup_geometry_ref:{b['key']}",
                            {"match": b["key"], "ref_jaccard": round(ref_j, 4), "dedupe_score": round(s, 4)},
                            sig,
                        )

        if best_score >= DEDUPE_BAR:
            return (
                True,
                f"dup_dedupe_score:{best_match}",
                {
                    "match": best_match,
                    "dedupe_score": round(best_score, 4),
                    "sym_uuids_shared": best_sym_overlap,
                    "fp_uuids_shared": best_fp_overlap,
                },
                sig,
            )

        return (
            False,
            None,
            {
                "best_match": best_match,
                "dedupe_score": round(best_score, 4),
                "sym_uuids_shared": best_sym_overlap,
                "fp_uuids_shared": best_fp_overlap,
            },
            sig,
        )


# ---------------------------------------------------------------------------
# KiCad Candidate Validator (pcbnew subprocess, mirrors + extends KRT)
# ---------------------------------------------------------------------------
PCBNEW_VALIDATE_SCRIPT = r"""
import json, os, re, sys
from collections import defaultdict
import pcbnew

sys.path.insert(0, sys.argv[2])
from prep_set2 import strip_zero_length_edge_cuts
strip_zero_length_edge_cuts(sys.argv[1])

BGA_RE = re.compile(r"(?:^|[^a-z0-9])(t?f?bga|lfbga|vfbga|wfbga|ufbga|wlcsp|bga[-_]?\d+)(?:[^a-z0-9]|$)", re.I)
CONN_REF_RE = re.compile(r"^(?:J|P|CN|CONN|X|H)\d", re.I)

try:
    b = pcbnew.LoadBoard(sys.argv[1])
except Exception as e:
    print(json.dumps({"error": f"pcbnew_load:{type(e).__name__}:{str(e)[:120]}"}))
    os._exit(0)

copper_layers = int(b.GetCopperLayerCount())
fps = list(b.GetFootprints())
bbox = b.GetBoardEdgesBoundingBox()
width_mm = round(bbox.GetWidth() / 1e6, 2)
height_mm = round(bbox.GetHeight() / 1e6, 2)
has_outline = bbox.GetWidth() > 1_000_000 and bbox.GetHeight() > 1_000_000

outlines = pcbnew.SHAPE_POLY_SET()
poly_ok = bool(b.GetBoardPolygonOutlines(outlines, True))
outline_count = int(outlines.OutlineCount()) if poly_ok else 0
hole_count = sum(int(outlines.HoleCount(i)) for i in range(outline_count)) if poly_ok else 0

x0, y0, x1, y1 = bbox.GetLeft(), bbox.GetTop(), bbox.GetRight(), bbox.GetBottom()
off_board = 0
pad_bearing_fps = 0
max_pads = 0
has_bga = False
total_net_pads = 0
padcount = defaultdict(int)

for fp in fps:
    p = fp.GetPosition()
    if has_outline and (p.x < x0 or p.x > x1 or p.y < y0 or p.y > y1):
        off_board += 1
    pads = fp.Pads()
    npads = len(pads)
    if npads > 0:
        pad_bearing_fps += 1
    max_pads = max(max_pads, npads)
    ref = fp.GetReference() or ""
    descr = fp.GetLibDescription() if hasattr(fp, "GetLibDescription") else ""
    fpid = f"{fp.GetFPIDAsString()} {descr} {fp.GetValue()}"
    if npads >= 36 and not CONN_REF_RE.match(ref):
        all_smd = all(pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD for pad in pads)
        if all_smd and BGA_RE.search(fpid):
            has_bga = True
        elif all_smd and npads >= 100:
            smd_pts = [(pad.GetPosition().x, pad.GetPosition().y) for pad in pads]
            xs = [pt[0] for pt in smd_pts]; ys = [pt[1] for pt in smd_pts]
            px0, px1 = min(xs), max(xs); py0, py1 = min(ys), max(ys)
            mx = 0.12 * (px1 - px0); my = 0.12 * (py1 - py0)
            interior = sum(1 for x, y in smd_pts if (px0 + mx < x < px1 - mx) and (py0 + my < y < py1 - my))
            pad_sizes = {(pad.GetSize().x, pad.GetSize().y) for pad in pads}
            if interior >= 25 and len(pad_sizes) <= 3:
                has_bga = True
    for pad in pads:
        nc = pad.GetNetCode()
        if nc > 0:
            padcount[nc] += 1
            total_net_pads += 1

routable_nets = {nc for nc, c in padcount.items() if c >= 2}

tracks = list(b.GetTracks())
segments = 0
vias = 0
track_net_ids = set()
for t in tracks:
    if t.Type() == pcbnew.PCB_VIA_T:
        vias += 1
    else:
        segments += 1
        nc = t.GetNetCode()
        if nc > 0:
            track_net_ids.add(nc)

zones = [z for z in b.Zones() if not z.GetIsRuleArea()]
zone_net_ids = {z.GetNetCode() for z in zones if z.GetNetCode() > 0}
routed_net_ids = track_net_ids | zone_net_ids
routed_cov = len(routable_nets & routed_net_ids) / max(1, len(routable_nets))
non_zone_routable = routable_nets - zone_net_ids
non_zone_cov = (
    len(non_zone_routable & track_net_ids) / len(non_zone_routable)
    if non_zone_routable else 1.0
)

b.BuildConnectivity()
unconnected_count = int(b.GetConnectivity().GetUnconnectedCount(False))

print(json.dumps({
    "copper_layers": copper_layers,
    "footprints": len(fps),
    "pad_bearing_fps": pad_bearing_fps,
    "off_board_fps": off_board,
    "has_outline": bool(has_outline),
    "poly_outline_ok": bool(poly_ok and outline_count >= 1),
    "outline_count": outline_count,
    "hole_count": hole_count,
    "width_mm": width_mm,
    "height_mm": height_mm,
    "routable_nets": len(routable_nets),
    "max_pads": max_pads,
    "has_bga": has_bga,
    "total_net_pads": total_net_pads,
    "segments": segments,
    "vias": vias,
    "zones": len(zones),
    "routed_cov": round(routed_cov, 4),
    "non_zone_cov": round(non_zone_cov, 4),
    "unconnected_count": unconnected_count,
}))
sys.stdout.flush()
os._exit(0)
"""


def classify_tier(layers: int, pad_fps: int, total_fps: int, routable_nets: int, has_bga: bool, max_pads: int) -> tuple[str, str]:
    """Return (corpus_tier, krt_tier).

    corpus_tier follows Step 1 stratification:
      - monster : >= 200 parts OR BGA OR >= 6 layers
      - large   : >= 80 parts OR >= 4 layers
      - medium  : >= 35 parts OR >= 40 routable nets
      - small   : >= 15 pad-bearing footprints
    krt_tier mirrors validate_candidate.classify().
    """
    if pad_fps >= 200 or total_fps >= 200 or has_bga or layers >= 6:
        corpus_tier = "monster"
    elif pad_fps >= 80 or total_fps >= 80 or layers >= 4:
        corpus_tier = "large"
    elif pad_fps >= 35 or routable_nets >= 40:
        corpus_tier = "medium"
    else:
        corpus_tier = "small"

    big_bga = max_pads >= 200
    if layers > 8 or total_fps >= 500:
        krt_tier = "monster"
    elif big_bga or layers >= 6 or total_fps >= 120 or routable_nets >= 250:
        krt_tier = "hard"
    elif layers >= 4 or total_fps >= 45 or routable_nets >= 80:
        krt_tier = "medium"
    else:
        krt_tier = "easy"

    return corpus_tier, krt_tier


def validate_candidate_board(pcb_path: Path) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate format, geometry, footprint count, Edge.Cuts, and human routing."""
    if not pcb_path.exists():
        return False, "missing_file", {}
    sz = pcb_path.stat().st_size
    if sz < MIN_FILE_BYTES:
        return False, "file_too_small", {"size_bytes": sz}
    if sz > MAX_FILE_BYTES:
        return False, "file_too_large", {"size_bytes": sz}

    text_head = pcb_path.read_text(encoding="utf-8", errors="replace")[:2048]
    if "(kicad_pcb" not in text_head:
        return False, "not_kicad_pcb", {}
    vm = VERSION_RE.search(text_head)
    kicad_version = int(vm.group(1)) if vm else None
    if kicad_version is None or kicad_version < MIN_KICAD_VERSION:
        return False, f"pre_kicad_6:{kicad_version}", {"kicad_version": kicad_version}

    # Fast pre-check before launching KiCad pcbnew subprocess
    full_text = pcb_path.read_text(encoding="utf-8", errors="replace")
    if not SEGMENT_OR_ARC_RE.search(full_text):
        return False, "unrouted_zero_segments", {"kicad_version": kicad_version}

    # Strip zero-length Edge.Cuts on a temp copy inside KiCad Python (matches prep_set2.py)
    with tempfile.TemporaryDirectory(prefix="val_pcb_") as td:
        tmp_pcb = Path(td) / pcb_path.name
        shutil.copy(pcb_path, tmp_pcb)
        pro = pcb_path.with_suffix(".kicad_pro")
        if pro.exists():
            shutil.copy(pro, tmp_pcb.with_suffix(".kicad_pro"))

        try:
            r = subprocess.run(
                [KPY, "-c", PCBNEW_VALIDATE_SCRIPT, str(tmp_pcb), str(KRT_STRESS)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "pcbnew_timeout", {"kicad_version": kicad_version}

        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip().startswith("{")]
        if not lines:
            err = (r.stderr or "").strip().splitlines()[-1] if r.stderr else "no_output"
            return False, f"pcbnew_no_json:{err[:100]}", {"kicad_version": kicad_version}

        try:
            m = json.loads(lines[-1])
        except json.JSONDecodeError:
            return False, "pcbnew_bad_json", {"kicad_version": kicad_version}

    if "error" in m:
        return False, m["error"], {"kicad_version": kicad_version}

    m["kicad_version"] = kicad_version
    layers = m["copper_layers"]
    fps = m["footprints"]
    pad_fps = m["pad_bearing_fps"]
    routable = m["routable_nets"]
    off_board = m["off_board_fps"]

    if layers < 2:
        return False, f"copper_layers<{2}:{layers}", m
    if not m["has_outline"] or not m["poly_outline_ok"]:
        return False, "invalid_or_missing_edge_cuts", m
    if m["hole_count"] > 6:
        return False, f"mechanical_plate_cutouts:{m['hole_count']}", m
    if off_board > max(2, 0.10 * max(1, fps)):
        return False, f"off_board_footprints:{off_board}/{fps}", m
    if pad_fps < MIN_PAD_BEARING_FPS:
        return False, f"pad_bearing_fps<{MIN_PAD_BEARING_FPS}:{pad_fps}", m
    if routable < MIN_ROUTABLE_NETS:
        return False, f"routable_nets<{MIN_ROUTABLE_NETS}:{routable}", m

    # Human-routed & anti-WIP checks
    segs = m["segments"]
    if segs < MIN_SEGMENTS:
        return False, f"unrouted_few_segments:{segs}", m
    if segs < max(MIN_SEGMENTS, int(0.35 * routable)):
        return False, f"wip_low_segment_density:{segs}_for_{routable}_nets", m
    if m["routed_cov"] < 0.72:
        return False, f"wip_low_net_coverage:{m['routed_cov']}", m
    if m["non_zone_cov"] < 0.75:
        return False, f"wip_low_signal_coverage:{m['non_zone_cov']}", m

    unconn = m["unconnected_count"]
    near_complete = unconn <= max(8, int(0.05 * routable))
    if not near_complete:
        if m["zones"] == 0 and unconn > max(15, int(0.12 * max(1, m["total_net_pads"]))):
            return False, f"wip_unconnected_ratsnest:{unconn}", m
        if m["routed_cov"] < 0.82 or m["non_zone_cov"] < 0.85:
            return False, f"wip_incomplete_routing:cov={m['routed_cov']},unconn={unconn}", m

    tier, krt_tier = classify_tier(
        layers=layers,
        pad_fps=pad_fps,
        total_fps=fps,
        routable_nets=routable,
        has_bga=m["has_bga"],
        max_pads=m["max_pads"],
    )
    m["tier"] = tier
    m["krt_tier"] = krt_tier
    return True, None, m


# ---------------------------------------------------------------------------
# Normalization & Copper Stripping (prep_set2.py + kicad_parser verification)
# ---------------------------------------------------------------------------
def prep_and_verify_board(src_pcb: Path, short_name: str) -> tuple[bool, str | None, dict[str, Any]]:
    ROUTED_DIR.mkdir(parents=True, exist_ok=True)
    UNROUTED_DIR.mkdir(parents=True, exist_ok=True)

    routed_pcb = ROUTED_DIR / f"{short_name}.kicad_pcb"
    unrouted_pcb = UNROUTED_DIR / f"{short_name}.kicad_pcb"

    try:
        r = subprocess.run(
            [KPY, str(PREP_PY), str(src_pcb), str(routed_pcb), str(unrouted_pcb)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "prep_timeout", {}

    if r.returncode != 0 or not routed_pcb.exists() or not unrouted_pcb.exists():
        err = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["prep_nonzero_exit"]
        for p in (routed_pcb, unrouted_pcb):
            if p.exists():
                p.unlink()
        return False, f"prep_failed:{err[0][:100]}", {}

    if routed_pcb.stat().st_size == 0 or unrouted_pcb.stat().st_size == 0:
        return False, "prep_empty_output", {}

    # Copy siblings explicitly if present
    for ext in (".kicad_pro", ".kicad_dru"):
        sib = src_pcb.with_suffix(ext)
        if sib.exists() and sib.stat().st_size > 20:
            shutil.copy(sib, ROUTED_DIR / f"{short_name}{ext}")
            shutil.copy(sib, UNROUTED_DIR / f"{short_name}{ext}")

    # Verify stripped unrouted board parses cleanly with KRT's kicad_parser
    def _cleanup_prepped() -> None:
        for ext in (".kicad_pcb", ".kicad_pro", ".kicad_dru"):
            for d in (ROUTED_DIR, UNROUTED_DIR):
                p = d / f"{short_name}{ext}"
                if p.exists():
                    p.unlink()

    try:
        pcb = parse_kicad_pcb(str(unrouted_pcb))
        non_graphic_segs = [s for s in pcb.segments if not getattr(s, "graphic", False)]
        if len(non_graphic_segs) != 0 or len(pcb.vias) != 0:
            _cleanup_prepped()
            return False, f"strip_incomplete:segs={len(non_graphic_segs)},vias={len(pcb.vias)}", {}
        if len(pcb.footprints) < MIN_PAD_BEARING_FPS:
            _cleanup_prepped()
            return False, f"parsed_fps_too_few:{len(pcb.footprints)}", {}
        if not pcb.board_info.board_bounds:
            _cleanup_prepped()
            return False, "parsed_missing_board_bounds", {}
        cutouts = len(getattr(pcb.board_info, "board_cutouts", []) or [])
        if cutouts > 6:
            _cleanup_prepped()
            return False, f"parsed_plate_cutouts:{cutouts}", {}
        return True, None, {
            "routed_kb": routed_pcb.stat().st_size // 1024,
            "unrouted_kb": unrouted_pcb.stat().st_size // 1024,
            "parsed_fps": len(pcb.footprints),
            "parsed_nets": len(pcb.nets),
            "parsed_layers": len(getattr(pcb.board_info, "copper_layers", []) or []),
            "parsed_cutouts": cutouts,
            "graphic_copper_segs": len(pcb.segments) - len(non_graphic_segs),
        }
    except Exception as e:
        _cleanup_prepped()
        return False, f"kicad_parser_failed:{type(e).__name__}:{str(e)[:100]}", {}


# ---------------------------------------------------------------------------
# Helper Functions (Slugs, Revision Grouping, Repo Discovery)
# ---------------------------------------------------------------------------
REV_SUFFIX_RE = re.compile(r"(?:[-_.]?(?:rev|r|v)\d+(?:\.\d+)*[a-z]?)$", re.I)


def make_short_name(repo: str, pcb_path: str, used_names: set[str]) -> str:
    owner, rname = repo.split("/", 1)
    stem = Path(pcb_path).stem
    # Prefer repo name if stem is generic like 'pcb' / 'board' / 'main' / 'hardware'
    generic_stems = {"pcb", "board", "main", "hardware", "kicad", "layout", "project", "rev1", "v1"}
    base = rname if stem.lower() in generic_stems else stem
    if len(base) < 4:
        base = f"{rname}_{stem}"
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")[:56] or "board"
    if slug not in used_names:
        return slug
    owner_slug = re.sub(r"[^a-z0-9]+", "_", owner.lower()).strip("_")[:16]
    cand = f"{owner_slug}_{slug}"[:64]
    if cand not in used_names:
        return cand
    i = 2
    while f"{cand}_{i}" in used_names:
        i += 1
    return f"{cand}_{i}"


def select_candidate_pcbs(paths: list[str], max_per_repo: int) -> tuple[list[str], int]:
    """Filter backups/fixtures/plates and deduplicate revision variants within one repo."""
    all_pcbs = [p for p in paths if p.lower().endswith(".kicad_pcb")]
    filtered: list[str] = []
    for p in all_pcbs:
        if EXCLUDE_PATH_RE.search(p):
            continue
        filtered.append(p)

    # Group by (parent_dir, stem_without_rev) so we keep the latest revision per board
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in filtered:
        pp = Path(p)
        norm_stem = REV_SUFFIX_RE.sub("", pp.stem.lower())
        groups[(str(pp.parent), norm_stem)].append(p)

    chosen: list[str] = []
    for key in sorted(groups):
        variants = sorted(groups[key])
        chosen.append(variants[-1])  # highest lex/rev version

    # Prefer paths closer to root / hardware dirs, then alphabetical
    def _rank(p: str) -> tuple[int, int, str]:
        pl = p.lower()
        hw_bonus = 0 if any(k in pl for k in ("hw/", "hardware/", "pcb/", "kicad/", "board/")) else 1
        return (hw_bonus, p.count("/"), p)

    chosen.sort(key=_rank)
    return chosen[:max_per_repo], len(all_pcbs) - len(filtered)


def discover_repos(
    client: ThrottledClient,
    sources: list[str],
    manifest: dict[str, Any],
    dedupe_idx: CorpusDedupeIndex,
    search_pages: int = 1,
    discovery_cap: int = 700,
    repos_file: Path | None = None,
    allow_forks: bool = False,
) -> dict[str, dict[str, Any]]:
    """Discover candidate GitHub repositories from API searches, curated lists, OSHWA, and org queries."""
    discovered: dict[str, dict[str, Any]] = manifest.setdefault("repos", {})
    completed_queries: set[str] = set(manifest.setdefault("completed_queries", []))

    def _add_repo(full_name: str, source_tag: str, meta: dict[str, Any] | None = None) -> None:
        if "/" not in full_name:
            return
        full_name = full_name.strip().removesuffix(".git")
        owner, rname = full_name.split("/", 1)
        if not owner or not rname or owner.lower() in {"topics", "orgs", "users", "explore", "sponsors"}:
            return
        key = full_name.lower()
        if owner.lower() in EXCLUDED_REPOS_OR_OWNERS or key in EXCLUDED_REPOS_OR_OWNERS:
            return
        meta = meta or {}
        if meta.get("archived"):
            return
        if not allow_forks and meta.get("fork"):
            return
        topics_str = " ".join(meta.get("topics") or [])
        desc = f"{full_name} {meta.get('description') or ''} {topics_str}"
        if SOFTWARE_REPO_RE.search(desc):
            return
        pushed_at = (meta.get("pushed_at") or "")[:10]
        if pushed_at and pushed_at < "2021-10-14":
            return
        if NON_HW_REPO_RE.search(rname) and not HW_KEYWORD_RE.search(desc):
            return
        known, _ = dedupe_idx.is_known_repo(full_name)
        if known:
            if key not in discovered:
                discovered[key] = {
                    "repo": full_name,
                    "source": source_tag,
                    "status": "skipped_existing_corpus",
                    "default_branch": meta.get("default_branch"),
                    "stars": meta.get("stargazers_count"),
                }
            return
        if key not in discovered:
            discovered[key] = {
                "repo": full_name,
                "source": source_tag,
                "status": "pending",
                "default_branch": meta.get("default_branch"),
                "pushed_at": pushed_at or None,
                "stars": meta.get("stargazers_count"),
                "topics": meta.get("topics", []),
                "license": (meta.get("license") or {}).get("spdx_id") if isinstance(meta.get("license"), dict) else None,
            }

    if "curated_lists" in sources:
        for name, url in CURATED_CATALOG_URLS:
            qkey = f"catalog:{name}"
            if qkey in completed_queries:
                continue
            print(f"[DISCOVER] Fetching curated catalog {name}...", flush=True)
            try:
                st, _, body = client.http_get(url)
            except RateLimitExhausted as e:
                print(f"  [WARN] Rate limit during catalog {name}: {e}", flush=True)
                break
            if st == 200 and body:
                text = body.decode("utf-8", "replace")
                links = re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
                for lk in links:
                    lk = lk.split("#")[0].split("?")[0].rstrip(").,;\"'")
                    if lk.count("/") == 1:
                        _add_repo(lk, qkey)
                completed_queries.add(qkey)
                print(f"  -> Extracted {len(set(links))} GitHub links from {name}", flush=True)

    # BoardRepo public open-hardware catalog (sitemap.xml allowed per robots.txt).
    # Note: Circuit World / heypcb.ai sets `Content-Signal: ai-train=no` + `Disallow: /explore,/world,/api/`
    # in https://heypcb.ai/robots.txt and is intentionally excluded out of compliance.
    if "boardrepo" in sources:
        qkey = "boardrepo:sitemap_xml"
        if qkey not in completed_queries:
            print("[DISCOVER] Fetching BoardRepo sitemap catalog (https://boardrepo.com/sitemap.xml)...", flush=True)
            try:
                st, _, body = client.http_get("https://boardrepo.com/sitemap.xml", timeout=45)
                if st == 200 and body:
                    xml = body.decode("utf-8", "replace")
                    locs = re.findall(r"<loc>([^<]+)</loc>", xml)
                    added_br = 0
                    for u in locs:
                        if "boardrepo.com/" not in u:
                            continue
                        rel = u.split("boardrepo.com/", 1)[-1].strip("/")
                        parts = rel.split("/")
                        if len(parts) != 2:
                            continue
                        own, rp = parts[0].strip(), parts[1].strip()
                        if not own or not rp or own.lower() in BOARDREPO_SKIP_OWNERS:
                            continue
                        slug = f"{own}/{rp}"
                        before = len(discovered)
                        _add_repo(slug, "boardrepo:sitemap")
                        if len(discovered) > before:
                            added_br += 1
                    completed_queries.add(qkey)
                    print(f"  -> BoardRepo sitemap parsed {len(locs)} URLs, added {added_br} new repos (pool={len(discovered)})", flush=True)
                else:
                    print(f"  [WARN] BoardRepo sitemap HTTP {st}", flush=True)
            except Exception as e:
                print(f"  [WARN] BoardRepo sitemap fetch error: {e}", flush=True)

    search_batches: list[tuple[str, str]] = []
    if "github_search" in sources and "curated_orgs" in sources:
        # Interleave general search queries and prolific org queries so both run before discovery_cap
        max_len = max(len(GITHUB_SEARCH_QUERIES), len(CURATED_ORG_QUERIES))
        for idx in range(max_len):
            if idx < len(GITHUB_SEARCH_QUERIES):
                k, q = GITHUB_SEARCH_QUERIES[idx]
                search_batches.append((f"search:{k}", q))
            if idx < len(CURATED_ORG_QUERIES):
                k, q = CURATED_ORG_QUERIES[idx]
                search_batches.append((f"org:{k}", q))
    elif "github_search" in sources:
        search_batches.extend([(f"search:{k}", q) for k, q in GITHUB_SEARCH_QUERIES])
    elif "curated_orgs" in sources:
        search_batches.extend([(f"org:{k}", q) for k, q in CURATED_ORG_QUERIES])

    if "oshwa" in sources:
        oshwa_key = os.environ.get("OSHWA_API_KEY", "").strip()
        qkey = "oshwa:api"
        if oshwa_key and qkey not in completed_queries:
            print("[DISCOVER] Querying OSHWA Certification API...", flush=True)
            req = urllib.request.Request(
                "https://certificationapi.oshwa.org/api/projects?limit=250",
                headers={"Authorization": f"Bearer {oshwa_key}", "User-Agent": "KiCadCorpusScraper/1.0"},
            )
            client.wait_slot("oshwa_api")
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    client._mark_done()
                    payload = json.loads(resp.read().decode("utf-8", "replace"))
                    items = payload.get("items", payload if isinstance(payload, list) else [])
                    for item in items:
                        for field in ("documentationUrl", "projectWebsite", "repositoryUrl", "github"):
                            val = str(item.get(field) or "")
                            for lk in re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", val):
                                _add_repo(lk, "oshwa:api")
                    completed_queries.add(qkey)
            except Exception as e:
                client._mark_done()
                print(f"  [WARN] OSHWA API failed: {e}", flush=True)
        elif not oshwa_key:
            print("[DISCOVER] Skipping OSHWA API (set OSHWA_API_KEY env var to enable authenticated OSHWA feed).", flush=True)

    hit_search_rate_limit = False
    for qtag, query in search_batches:
        pending_cnt = sum(1 for r in discovered.values() if r.get("status") == "pending")
        if discovery_cap > 0 and pending_cnt >= discovery_cap:
            print(f"[DISCOVER] Reached --discovery-cap={discovery_cap} pending repos ({pending_cnt}); starting harvest.", flush=True)
            break
        for page in range(1, search_pages + 1):
            qkey = f"{qtag}:p{page}"
            if qkey in completed_queries:
                continue
            encoded_q = urllib.parse.quote(query)
            url = (
                f"https://api.github.com/search/repositories"
                f"?q={encoded_q}&sort=stars&order=desc&per_page=100&page={page}"
            )
            print(f"[DISCOVER] Query {qkey} ({query})...", flush=True)
            try:
                st, _, body = client.http_get(url)
            except RateLimitExhausted as e:
                print(f"  [PAUSE] Search rate limit hit ({e}); continuing with discovered pool.", flush=True)
                hit_search_rate_limit = True
                break
            if st != 200 or not body:
                print(f"  [WARN] Query {qkey} returned status {st}", flush=True)
                break
            data = json.loads(body.decode("utf-8", "replace"))
            items = data.get("items", [])
            for item in items:
                _add_repo(item["full_name"], qtag, item)
            completed_queries.add(qkey)
            print(f"  -> {len(items)} repos returned (total discovered pool={len(discovered)})", flush=True)
            if len(items) < 100:
                break
        if hit_search_rate_limit:
            break

    if repos_file and repos_file.exists():
        for line in repos_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                _add_repo(line, f"file:{repos_file.name}")

    manifest["completed_queries"] = sorted(completed_queries)
    return discovered


# ---------------------------------------------------------------------------
# Manifest & Summary Persistence
# ---------------------------------------------------------------------------
def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed_queries": [],
        "repos": {},
        "boards": {},
    }


def save_manifest(manifest: dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = utc_now()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)


def build_and_write_summary(manifest: dict[str, Any], client: ThrottledClient) -> dict[str, Any]:
    repos = manifest.get("repos", {})
    boards = manifest.get("boards", {})

    repo_status_counts: dict[str, int] = defaultdict(int)
    for r in repos.values():
        repo_status_counts[r.get("status", "unknown")] += 1

    board_status_counts: dict[str, int] = defaultdict(int)
    reject_reasons: dict[str, int] = defaultdict(int)
    dedupe_reasons: dict[str, int] = defaultdict(int)
    tier_counts_prepped: dict[str, int] = defaultdict(int)
    tier_counts_validated: dict[str, int] = defaultdict(int)
    krt_tier_counts_prepped: dict[str, int] = defaultdict(int)
    source_breakdown: dict[str, int] = defaultdict(int)

    prepped_boards = []
    for bid, b in sorted(boards.items()):
        st = b.get("status", "unknown")
        board_status_counts[st] += 1
        reason = b.get("reject_reason")
        if reason:
            key = reason.split(":")[0]
            if st == "duplicate":
                dedupe_reasons[key] += 1
            elif st in ("rejected", "prep_failed", "download_failed", "filtered_tier"):
                reject_reasons[key] += 1
        if b.get("validated"):
            tier_counts_validated[b.get("tier", "unknown")] += 1
        if st == "prepped":
            tier = b.get("tier", "unknown")
            tier_counts_prepped[tier] += 1
            krt_tier_counts_prepped[b.get("krt_tier", "unknown")] += 1
            source_breakdown[b.get("discovery_source", "unknown")] += 1
            prepped_boards.append({
                "short_name": b["short_name"],
                "repo": b["repo"],
                "path": b["path"],
                "sha": b.get("sha"),
                "raw_url": b.get("raw_url"),
                "tier": tier,
                "krt_tier": b.get("krt_tier"),
                "layers": (b.get("metrics") or {}).get("copper_layers"),
                "pad_bearing_fps": (b.get("metrics") or {}).get("pad_bearing_fps"),
                "routable_nets": (b.get("metrics") or {}).get("routable_nets"),
                "segments": (b.get("metrics") or {}).get("segments"),
                "has_bga": (b.get("metrics") or {}).get("has_bga"),
                "has_kicad_pro": b.get("has_kicad_pro", False),
                "has_kicad_dru": b.get("has_kicad_dru", False),
                "dedupe_score": b.get("dedupe_best_score"),
                "dedupe_match": b.get("dedupe_best_match"),
                "routed_path": b.get("routed_path"),
                "unrouted_path": b.get("unrouted_path"),
            })

    summary = {
        "generated_at": utc_now(),
        "paths": {
            "scraper_script": str(WORKSPACE / "scrape_kicad_repos.py"),
            "sources_dir": str(SOURCES_DIR),
            "routed_dir": str(ROUTED_DIR),
            "unrouted_dir": str(UNROUTED_DIR),
            "manifest": str(MANIFEST_PATH),
            "events_log": str(EVENTS_PATH),
        },
        "funnel": {
            "repos_discovered_total": len(repos),
            "repos_skipped_existing_corpus": repo_status_counts.get("skipped_existing_corpus", 0),
            "repos_visited": sum(
                v for k, v in repo_status_counts.items() if k not in ("pending", "skipped_existing_corpus")
            ),
            "repos_with_kicad_pcb": sum(1 for r in repos.values() if r.get("selected_pcbs", 0) > 0),
            "repos_status_breakdown": dict(sorted(repo_status_counts.items())),
            "candidate_boards_total": len(boards),
            "candidate_boards_downloaded": sum(1 for b in boards.values() if b.get("downloaded")),
            "candidate_boards_deduped_passed": sum(
                1 for b in boards.values() if b.get("downloaded") and b.get("status") != "duplicate"
            ),
            "candidate_boards_duplicates_dropped": board_status_counts.get("duplicate", 0),
            "candidate_boards_validated": sum(1 for b in boards.values() if b.get("validated")),
            "candidate_boards_prepped": board_status_counts.get("prepped", 0),
            "board_status_breakdown": dict(sorted(board_status_counts.items())),
        },
        "dedupe_breakdown": dict(sorted(dedupe_reasons.items())),
        "filter_reject_breakdown": dict(sorted(reject_reasons.items())),
        "tier_breakdown_prepped": {
            "small": tier_counts_prepped.get("small", 0),
            "medium": tier_counts_prepped.get("medium", 0),
            "large": tier_counts_prepped.get("large", 0),
            "monster": tier_counts_prepped.get("monster", 0),
        },
        "krt_tier_breakdown_prepped": dict(sorted(krt_tier_counts_prepped.items())),
        "tier_breakdown_validated": dict(sorted(tier_counts_validated.items())),
        "discovery_source_breakdown_prepped": dict(sorted(source_breakdown.items())),
        "rate_limit_stats": client.stats(),
        "prepped_boards": prepped_boards,
    }
    tmp = SUMMARY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, SUMMARY_PATH)
    return summary


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> int:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    ROUTED_DIR.mkdir(parents=True, exist_ok=True)
    UNROUTED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    dedupe_idx = CorpusDedupeIndex()
    if args.verify_corpus_only:
        return 0

    if args.dry_run_dedupe:
        cand = Path(args.dry_run_dedupe)
        is_dup, reason, details, _ = dedupe_idx.check_candidate(cand, repo="", urls=[])
        ok, vreason, metrics = validate_candidate_board(cand)
        print(json.dumps({
            "file": str(cand),
            "duplicate": is_dup,
            "dup_reason": reason,
            "dup_details": details,
            "valid": ok,
            "validate_reason": vreason,
            "metrics": metrics,
        }, indent=2))
        return 0

    manifest = load_manifest() if args.resume else {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed_queries": [],
        "repos": {},
        "boards": {},
    }

    # Re-register already-prepped scraped boards into dedupe_idx so --resume preserves intra-corpus dedupe
    used_names = set(dedupe_idx.existing_short_names)
    scraped_sig_cache = _load_scraped_sig_cache()
    scraped_sig_dirty = False
    for b in manifest.get("boards", {}).values():
        if b.get("status") == "prepped":
            sn = b["short_name"]
            used_names.add(sn.lower())
            routed_p = Path(b["routed_path"])
            if routed_p.exists():
                key = f"scraped_set1/{sn}"
                sig, changed = _extract_scraped_sig_cached(key, routed_p, scraped_sig_cache)
                if changed:
                    scraped_sig_dirty = True
                dedupe_idx.add_admitted_board(
                    key,
                    sig,
                    repo=b.get("repo", ""),
                    urls=[b.get("raw_url", "")],
                    short_name=sn,
                )
    if scraped_sig_dirty:
        _save_scraped_sig_cache(scraped_sig_cache)

    client = ThrottledClient(min_delay_s=args.delay, max_rate_limit_wait_s=args.max_rate_limit_wait)

    # Optionally skip heavy search discovery if pending repos already exist,
    # but always ingest single-fetch catalogs (like boardrepo:sitemap_xml) if not completed yet.
    pending_repos = [r for r in manifest.get("repos", {}).values() if r.get("status") == "pending"]
    completed_q = set(manifest.get("completed_queries", []))
    if not (args.skip_discovery_if_pending and len(pending_repos) >= 30):
        discover_repos(
            client=client,
            sources=args.sources,
            manifest=manifest,
            dedupe_idx=dedupe_idx,
            search_pages=args.search_pages,
            discovery_cap=args.discovery_cap,
            repos_file=args.repos_file,
            allow_forks=args.allow_forks,
        )
        save_manifest(manifest)
    elif "boardrepo" in args.sources and "boardrepo:sitemap_xml" not in completed_q:
        discover_repos(
            client=client,
            sources=["boardrepo"],
            manifest=manifest,
            dedupe_idx=dedupe_idx,
            search_pages=args.search_pages,
            discovery_cap=args.discovery_cap,
            repos_file=args.repos_file,
            allow_forks=args.allow_forks,
        )
        save_manifest(manifest)

    # Process repos in batches
    repos_dict = manifest["repos"]
    boards_dict = manifest["boards"]

    def _repo_priority(item: dict[str, Any]) -> tuple[int, int, str]:
        src = item.get("source", "")
        repo_l = (item.get("repo") or "").lower()
        owner = (repo_l.split("/", 1)[0])
        # Deprioritize predominantly Eagle/Altium vendor catalogs behind KiCad-native repos
        if owner in BOARDREPO_EAGLE_HEAVY_OWNERS or "eagle" in repo_l or "altium" in repo_l:
            return (5, -(item.get("stars") or 15), item["repo"])
        # Prioritize modern KiCad topic/hardware searches & prolific KiCad hardware orgs
        if src.startswith("search:topic_kicad_pcb") or src.startswith("search:kicad_8") or src.startswith("search:kicad_7"):
            tier_p = 0
        elif (
            src.startswith("search:kicad_")
            or src == "catalog:kitspace_awesome_electronics"
            or src.startswith("org:")
            or (src == "boardrepo:sitemap" and owner in BOARDREPO_PRIORITY_OWNERS)
            or (
                src == "boardrepo:sitemap"
                and any(k in repo_l for k in ("kicad", "stm32", "esp32", "rp2040", "fpga", "carrier", "flight", "can"))
            )
        ):
            tier_p = 1
        elif src.startswith("search:") or src == "boardrepo:sitemap":
            tier_p = 2
        else:
            tier_p = 3
        stars = item.get("stars") or 15
        if stars > 2500:
            tier_p = 4
        return (tier_p, -stars, item["repo"])

    pending = sorted((r for r in repos_dict.values() if r.get("status") == "pending"), key=_repo_priority)
    already_prepped = sum(1 for b in boards_dict.values() if b.get("status") == "prepped")
    existing_corpus_count = len(dedupe_idx.boards) - already_prepped
    eff_target_scraped = args.target_boards
    if getattr(args, "target_corpus_total", 0) > 0:
        needed_from_scraper = max(0, args.target_corpus_total - existing_corpus_count)
        eff_target_scraped = max(eff_target_scraped, needed_from_scraper) if eff_target_scraped > 0 else needed_from_scraper
    small_count = sum(1 for b in boards_dict.values() if b.get("status") == "prepped" and b.get("tier") == "small")
    prepped_this_run = 0
    repos_visited_this_run = 0

    print(
        f"\n[HARVEST] Pending repos={len(pending)} | existing_corpus={existing_corpus_count} | already_scraped={already_prepped} "
        f"| target_scraped={eff_target_scraped or 'unlimited'} (corpus={existing_corpus_count + already_prepped}/{getattr(args, 'target_corpus_total', 0) or '-'}) | min_delay={client.min_delay_s:.2f}s",
        flush=True,
    )

    for repo_rec in pending:
        if args.max_repos > 0 and repos_visited_this_run >= args.max_repos:
            print(f"[STOP] Reached --max-repos={args.max_repos}", flush=True)
            break
        if eff_target_scraped > 0 and already_prepped >= eff_target_scraped:
            print(f"[STOP] Reached target prepped boards={already_prepped}/{eff_target_scraped}", flush=True)
            break

        repo = repo_rec["repo"]
        repos_visited_this_run += 1
        print(f"\n[{repos_visited_this_run}] Scanning repo {repo} (src={repo_rec.get('source')})...", flush=True)

        known, why_known = dedupe_idx.is_known_repo(repo)
        if known:
            repo_rec["status"] = "skipped_existing_corpus"
            repo_rec["note"] = why_known
            save_manifest(manifest)
            continue

        try:
            branch, sha, paths, tree_method = client.list_repo_tree(repo, repo_rec.get("default_branch"))
        except RateLimitExhausted as e:
            print(f"[PAUSE] Rate limit exhausted while listing {repo}: {e}", flush=True)
            save_manifest(manifest)
            break

        repo_rec["branch"] = branch
        repo_rec["sha"] = sha
        repo_rec["tree_method"] = tree_method
        repo_rec["visited_at"] = utc_now()

        if not paths:
            repo_rec["status"] = "tree_error" if tree_method.startswith("git_") else "no_files"
            repo_rec["note"] = tree_method
            save_manifest(manifest)
            continue

        selected_pcbs, prefiltered_count = select_candidate_pcbs(paths, max_per_repo=args.max_per_repo)
        repo_rec["total_pcbs_in_tree"] = sum(1 for p in paths if p.lower().endswith(".kicad_pcb"))
        repo_rec["prefiltered_pcbs"] = prefiltered_count
        repo_rec["selected_pcbs"] = len(selected_pcbs)

        if not selected_pcbs:
            repo_rec["status"] = "no_eligible_kicad_pcb"
            save_manifest(manifest)
            continue

        path_set = set(paths)
        admitted_from_repo = 0

        for rel_pcb in selected_pcbs:
            if eff_target_scraped > 0 and already_prepped >= eff_target_scraped:
                break

            board_id = f"{repo}::{rel_pcb}"
            if board_id in boards_dict and boards_dict[board_id].get("status") in ("prepped", "duplicate", "rejected", "filtered_tier"):
                continue

            ref_spec = sha or branch or "HEAD"
            raw_url = f"https://raw.githubusercontent.com/{repo}/{ref_spec}/{urllib.parse.quote(rel_pcb)}"
            github_url = f"https://github.com/{repo}/blob/{ref_spec}/{urllib.parse.quote(rel_pcb)}"

            short_name = make_short_name(repo, rel_pcb, used_names)
            stage_pcb = STAGING_DIR / f"{short_name}.kicad_pcb"
            brec: dict[str, Any] = {
                "board_id": board_id,
                "short_name": short_name,
                "repo": repo,
                "path": rel_pcb,
                "branch": branch,
                "sha": sha,
                "raw_url": raw_url,
                "github_url": github_url,
                "discovery_source": repo_rec.get("source"),
                "downloaded": False,
                "validated": False,
                "status": "discovered",
                "updated_at": utc_now(),
            }

            try:
                ok_dl = client.download_file(raw_url, stage_pcb)
            except RateLimitExhausted as e:
                print(f"[PAUSE] Rate limit on raw download {raw_url}: {e}", flush=True)
                save_manifest(manifest)
                build_and_write_summary(manifest, client)
                return 0

            if not ok_dl:
                brec["status"] = "download_failed"
                brec["reject_reason"] = "raw_download_failed"
                boards_dict[board_id] = brec
                log_event("board_download_failed", board_id=board_id, raw_url=raw_url)
                continue

            brec["downloaded"] = True
            brec["size_kb"] = stage_pcb.stat().st_size // 1024

            # Fetch sibling .kicad_pro / .kicad_dru ONLY if present in the repo tree!
            base_rel = rel_pcb[: -len(".kicad_pcb")]
            for ext, flag_key in ((".kicad_pro", "has_kicad_pro"), (".kicad_dru", "has_kicad_dru")):
                sib_rel = base_rel + ext
                if sib_rel in path_set:
                    sib_url = f"https://raw.githubusercontent.com/{repo}/{ref_spec}/{urllib.parse.quote(sib_rel)}"
                    try:
                        if client.download_file(sib_url, stage_pcb.with_suffix(ext)):
                            brec[flag_key] = True
                    except RateLimitExhausted:
                        pass

            def _cleanup_staging() -> None:
                for ext in (".kicad_pcb", ".kicad_pro", ".kicad_dru"):
                    sp = stage_pcb.with_suffix(ext)
                    if sp.exists():
                        sp.unlink()

            # Step 1: Validate format, Edge.Cuts, >= 15 pad-bearing fps, human-routed
            ok_val, vreason, metrics = validate_candidate_board(stage_pcb)
            brec["metrics"] = metrics
            if not ok_val:
                brec["status"] = "rejected"
                brec["reject_reason"] = vreason
                boards_dict[board_id] = brec
                _cleanup_staging()
                print(f"  [REJECT] {rel_pcb} -> {vreason}", flush=True)
                log_event("board_rejected", board_id=board_id, reason=vreason)
                continue

            brec["validated"] = True
            brec["tier"] = metrics["tier"]
            brec["krt_tier"] = metrics["krt_tier"]

            # Optional tier filter / small cap so harvest spans medium/large/monster cleanly
            if TIER_ORDER.get(brec["tier"], 0) < TIER_ORDER.get(args.min_tier, 0):
                brec["status"] = "filtered_tier"
                brec["reject_reason"] = f"below_min_tier:{brec['tier']}"
                boards_dict[board_id] = brec
                _cleanup_staging()
                print(f"  [FILTER] {rel_pcb} -> tier={brec['tier']} < min_tier={args.min_tier}", flush=True)
                continue

            if brec["tier"] == "small" and args.max_small >= 0 and small_count >= args.max_small:
                brec["status"] = "filtered_tier"
                brec["reject_reason"] = f"small_tier_cap_reached:{small_count}"
                boards_dict[board_id] = brec
                _cleanup_staging()
                print(f"  [FILTER] {rel_pcb} -> small tier cap reached ({small_count})", flush=True)
                continue

            # Step 2: 3-Layer Deduplication against ALL 474 existing boards + already-prepped scraped boards
            dims_ov = (metrics["width_mm"], metrics["height_mm"])
            is_dup, dreason, ddetails, sig = dedupe_idx.check_candidate(
                stage_pcb, repo=repo, urls=[raw_url, github_url], dims_override=dims_ov
            )
            brec["dedupe_best_score"] = ddetails.get("dedupe_score")
            brec["dedupe_best_match"] = ddetails.get("best_match") or ddetails.get("match")
            brec["dedupe_details"] = ddetails

            if is_dup:
                brec["status"] = "duplicate"
                brec["reject_reason"] = dreason
                boards_dict[board_id] = brec
                _cleanup_staging()
                print(f"  [DUP] {rel_pcb} -> {dreason} ({ddetails})", flush=True)
                log_event("board_duplicate", board_id=board_id, reason=dreason, details=ddetails)
                continue

            # Step 3: Promote to sources/scraped_github/ and normalize+strip via prep_set2.py
            used_names.add(short_name.lower())
            final_src = SOURCES_DIR / f"{short_name}.kicad_pcb"
            shutil.move(str(stage_pcb), str(final_src))
            for ext in (".kicad_pro", ".kicad_dru"):
                ss = stage_pcb.with_suffix(ext)
                if ss.exists():
                    shutil.move(str(ss), str(final_src.with_suffix(ext)))

            ok_prep, preason, prep_info = prep_and_verify_board(final_src, short_name)
            if not ok_prep:
                brec["status"] = "prep_failed"
                brec["reject_reason"] = preason
                boards_dict[board_id] = brec
                for ext in (".kicad_pcb", ".kicad_pro", ".kicad_dru"):
                    sp = final_src.with_suffix(ext)
                    if sp.exists():
                        sp.unlink()
                print(f"  [PREP-FAIL] {short_name} -> {preason}", flush=True)
                log_event("board_prep_failed", board_id=board_id, reason=preason)
                continue

            brec["status"] = "prepped"
            brec["local_src"] = str(final_src)
            brec["routed_path"] = str(ROUTED_DIR / f"{short_name}.kicad_pcb")
            brec["unrouted_path"] = str(UNROUTED_DIR / f"{short_name}.kicad_pcb")
            brec["prep_info"] = prep_info
            boards_dict[board_id] = brec

            dedupe_idx.add_admitted_board(
                f"scraped_set1/{short_name}",
                sig,
                repo=repo,
                urls=[raw_url, github_url],
                short_name=short_name,
            )

            already_prepped += 1
            prepped_this_run += 1
            admitted_from_repo += 1
            if brec["tier"] == "small":
                small_count += 1

            print(
                f"  [ADMIT #{already_prepped}] {short_name:32s} tier={brec['tier']:7s} "
                f"krt={brec['krt_tier']:7s} L={metrics['copper_layers']} fps={metrics['pad_bearing_fps']:3d} "
                f"nets={metrics['routable_nets']:3d} segs={metrics['segments']:5d} "
                f"dedupe={brec['dedupe_best_score']:.3f}",
                flush=True,
            )
            log_event(
                "board_admitted",
                board_id=board_id,
                short_name=short_name,
                tier=brec["tier"],
                krt_tier=brec["krt_tier"],
                dedupe_score=brec["dedupe_best_score"],
            )

        repo_rec["status"] = "visited_admitted" if admitted_from_repo > 0 else "visited_none_admitted"
        repo_rec["admitted_count"] = admitted_from_repo
        save_manifest(manifest)
        build_and_write_summary(manifest, client)

    save_manifest(manifest)
    summary = build_and_write_summary(manifest, client)
    print("\n=== Scrape Run Complete ===", flush=True)
    print(json.dumps({
        "prepped_this_run": prepped_this_run,
        "total_prepped_on_disk": summary["funnel"]["candidate_boards_prepped"],
        "funnel": summary["funnel"],
        "tier_breakdown_prepped": summary["tier_breakdown_prepped"],
        "dedupe_breakdown": summary["dedupe_breakdown"],
        "filter_reject_breakdown": summary["filter_reject_breakdown"],
        "rate_limit_stats": summary["rate_limit_stats"],
    }, indent=2), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Throttled KiCad PCB corpus scraper, deduplicator, validator, and prepper")
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["boardrepo", "github_search", "curated_lists", "curated_orgs"],
        choices=["boardrepo", "github_search", "curated_lists", "curated_orgs", "oshwa"],
        help="Discovery sources to run",
    )
    ap.add_argument("--repos-file", type=Path, default=None, help="Optional file of owner/repo lines to include")
    ap.add_argument("--search-pages", type=int, default=1, help="Pages per GitHub search query (100 repos/page)")
    ap.add_argument("--discovery-cap", type=int, default=12000, help="Stop querying search API once N pending repos are queued (0 = unlimited)")
    ap.add_argument("--target-boards", type=int, default=30, help="Stop after N total scraped boards (0 = unlimited)")
    ap.add_argument("--target-corpus-total", type=int, default=0, help="Optional total corpus target (existing + scraped >= N, e.g. 1000)")
    ap.add_argument("--max-repos", type=int, default=120, help="Max pending repos to inspect per run (0 = unlimited)")
    ap.add_argument("--max-per-repo", type=int, default=2, help="Max candidate .kicad_pcb files to inspect per repo")
    ap.add_argument("--delay", type=float, default=2.85, help="Seconds between EVERY network request (clamped >= 2.8s)")
    ap.add_argument("--max-rate-limit-wait", type=float, default=120.0, help="Max seconds to sleep on 403/429 before checkpointing")
    ap.add_argument("--min-tier", choices=["small", "medium", "large", "monster"], default="small", help="Minimum tier to admit")
    ap.add_argument("--max-small", type=int, default=3, help="Max small-tier boards to admit (-1 = unlimited)")
    ap.add_argument("--allow-forks", action="store_true", help="Include forked GitHub repos (default: skip forks)")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from manifest (default)")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh manifest")
    ap.add_argument("--skip-discovery-if-pending", action="store_true", help="Use pending repos in manifest before querying more")
    ap.add_argument("--verify-corpus-only", action="store_true", help="Only build/verify corpus dedupe index")
    ap.add_argument("--dry-run-dedupe", type=str, default=None, help="Run dedupe + validation check on one local .kicad_pcb")
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(run_pipeline(parse_args()))
