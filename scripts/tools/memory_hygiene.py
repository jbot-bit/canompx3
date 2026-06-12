#!/usr/bin/env python3
"""Read-only memory-budget + baton clear-tier report.

Two jobs, both READ-ONLY (this tool NEVER writes or deletes anything):

1. **Budget report.** The auto-memory `MEMORY.md` index is loaded into every
   session under the official cutoff: the first **200 lines OR 25 KB**, whichever
   binds first (https://code.claude.com/docs/en/memory). Over budget, the index
   *tail* is silently invisible — not lost (the corpus is intact, topic files
   load on-demand), but un-recalled. This reports lines/bytes vs 200 / 24,400,
   the first silently-dropped line, and over-long index lines (> 200 chars) that
   inflate the byte count (avg ~355 chars/line today vs the ~200 guideline).

2. **Baton clear-tier worklist.** Tiers `project_*.md` batons into READY /
   LANDED-BUT-OPEN / UNVERIFIED so the operator can decide which resume-batons
   are safe to clear. The repo never autonomously deletes a baton — this tool
   only RANKS and, with `--print-clear`, emits a fully-COMMENTED paste-ready
   `rm` + `sed` block for READY-tier batons (operator runs it, or doesn't).

   **READY gate (operator-confirmed primary-SHA design):** a strict
   "all-quoted-SHAs-merged" gate yields READY=0 (batons quote recovery/orphan/
   peer SHAs; one unmerged flips the whole baton). So READY uses the *primary*
   SHA — the headline SHA (first quoted, OR adjacent to `origin/main=` /
   `DONE+PUSHED`) — and requires (a) that SHA proven on origin/main AND (b) no
   open-work marker (`NEXT=`/`▶`/`RESUME`/`TODO`/`OWED`/`PENDING`/`BLOCKED`/
   `NOT STARTED`). All SHAs are shown as evidence; the operator decides.

Reuse (institutional-rigor §4 — no re-encode): git/regex/path logic is imported
BY PATH from `.claude/hooks/_memory_capture.py` (`_git`, `_sha_on_origin_main`,
`_SHA_RE`, HOME-based `MEMORY_DIR`). That module is not on sys.path, so it is
loaded via importlib — the same idiom the companion tests use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Official load-budget constants (memory docs). The byte budget binds first for
# this repo's index. 24,400 is the repo's own soft target (< the 25 KB hard cut)
# already cited in MEMORY.md's self-warning ("limit: 24.4KB").
# --------------------------------------------------------------------------- #
LINE_BUDGET = 200
BYTE_BUDGET = 24_400  # repo soft target, under the official 25 KB hard cutoff
HARD_BYTE_CUTOFF = 25_000  # official 25 KB load cutoff (first-dropped-line calc)
OVERLONG_LINE_CHARS = 200  # the ~200-char/line index guideline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CAPTURE_MODULE = PROJECT_ROOT / ".claude" / "hooks" / "_memory_capture.py"


def _load_capture_module():
    """Load `_memory_capture.py` by path (it is not importable normally).

    Mirrors the importlib idiom in tests/test_hooks/test_baton_staleness.py.
    Raises on failure — the tool cannot tier batons without the git reuse, so a
    hard failure here is correct (this is a CLI tool, not a fail-open hook).
    """
    spec = importlib.util.spec_from_file_location("_memory_capture", _CAPTURE_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_CAPTURE_MODULE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CAP = _load_capture_module()
_sha_on_origin_main = _CAP._sha_on_origin_main
_SHA_RE = _CAP._SHA_RE
MEMORY_DIR: Path = _CAP.MEMORY_DIR
MEMORY_MD = MEMORY_DIR / "MEMORY.md"

# --------------------------------------------------------------------------- #
# Baton tiering patterns. Open-work markers veto READY regardless of SHA state —
# a baton that still describes pending work is not clearable even if its code
# landed. Kept here (consumer-local) rather than in _memory_capture because the
# capture hook's _LIVE_STATUS_RE is description-line-scoped; the clear gate scans
# the whole baton body, so the marker set is broader and tool-specific.
# --------------------------------------------------------------------------- #
_OPEN_WORK_RE = re.compile(
    r"NEXT\s*=|(?<![A-Za-z])▶|\bRESUME\b|\bTODO\b|\bOWED\b|\bPENDING\b|"
    r"\bBLOCKED\b|\bNOT\s+STARTED\b|\bIN\s+PROGRESS\b",
    re.IGNORECASE,
)
# Headline-SHA anchors: a SHA adjacent to one of these is the primary SHA even if
# it is not the first SHA in the file.
_HEADLINE_ANCHOR_RE = re.compile(
    r"(?:origin/main\s*=\s*|DONE\+PUSHED[^\n]*?|PUSHED[^\n]*?)`?([0-9a-f]{7,40})`?",
    re.IGNORECASE,
)


def _first_dropped_line(raw_bytes: bytes) -> tuple[int, int] | None:
    """Return (line_no, kind) of the first line silently dropped at load.

    kind: 0 = byte cutoff reached first, 1 = line cutoff reached first.
    Returns None if the whole index loads (within both budgets). line_no is
    1-based — the FIRST line the session never sees.

    Walks the RAW on-disk bytes via `splitlines(keepends=True)` so each physical
    line carries its TRUE terminator (`\\r\\n`, `\\n`, or none on a final
    unterminated line). The earlier `+1`-per-line heuristic undercounted CRLF
    files (it assumed a 1-byte terminator), which is the false-PASS direction for
    a budget warning — fixed to count the file as the loader actually sees it.
    """
    running_bytes = 0
    for idx, physical in enumerate(raw_bytes.splitlines(keepends=True), start=1):
        running_bytes += len(physical)
        if running_bytes > HARD_BYTE_CUTOFF:
            return (idx, 0)
        if idx > LINE_BUDGET:
            return (idx, 1)
    return None


def _overlong_lines(lines: list[str]) -> list[tuple[int, int]]:
    """1-based (line_no, char_len) for index lines exceeding the char guideline.

    Blank lines and section headers are skipped — the guideline targets the
    long bullet pointers that inflate the byte count, not headers.
    """
    out: list[tuple[int, int]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) > OVERLONG_LINE_CHARS:
            out.append((idx, len(line)))
    return out


def budget_report() -> dict:
    """Read-only budget metrics for MEMORY.md. Never raises on a missing file."""
    if not MEMORY_MD.is_file():
        return {
            "exists": False,
            "path": str(MEMORY_MD),
        }
    raw_bytes = MEMORY_MD.read_bytes()
    raw = raw_bytes.decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    # Count the file as the loader sees it — RAW on-disk bytes, NOT the
    # CRLF-stripped re-encode of splitlines() (which undercounts CRLF files by
    # one byte per line: the false-PASS direction for a budget guard).
    n_bytes = len(raw_bytes)
    dropped = _first_dropped_line(raw_bytes)
    overlong = _overlong_lines(lines)
    return {
        "exists": True,
        "path": str(MEMORY_MD),
        "lines": len(lines),
        "bytes": n_bytes,
        "line_budget": LINE_BUDGET,
        "byte_budget": BYTE_BUDGET,
        "over_line_budget": len(lines) > LINE_BUDGET,
        "over_byte_budget": n_bytes > BYTE_BUDGET,
        "first_dropped": ({"line": dropped[0], "by": "bytes" if dropped[1] == 0 else "lines"} if dropped else None),
        "overlong_lines": [{"line": ln, "chars": c} for ln, c in overlong],
    }


# --------------------------------------------------------------------------- #
# Baton tiering
# --------------------------------------------------------------------------- #
def _all_shas(text: str) -> list[str]:
    """Unique SHAs in order of first appearance (7-40 hex, _memory_capture regex).

    Drops all-digit tokens: a 7+ digit decimal is not a git SHA in practice and
    the merge-base check would waste a git call returning False.
    """
    seen: dict[str, None] = {}
    for sha in _SHA_RE.findall(text):
        if sha.isdigit():
            continue
        seen.setdefault(sha, None)
    return list(seen.keys())


def _primary_sha(text: str, all_shas: list[str]) -> str | None:
    """The headline SHA: anchored (origin/main=/DONE+PUSHED) first, else first SHA."""
    m = _HEADLINE_ANCHOR_RE.search(text)
    if m and not m.group(1).isdigit():
        return m.group(1)
    return all_shas[0] if all_shas else None


def tier_baton(path: Path) -> dict:
    """Tier one project baton. READY / LANDED-BUT-OPEN / UNVERIFIED + evidence.

    READY  = primary SHA proven on origin/main AND no open-work marker.
    LANDED-BUT-OPEN = primary SHA on origin/main BUT an open-work marker present.
    UNVERIFIED = no SHA, or the primary SHA is not provably on origin/main.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    shas = _all_shas(text)
    primary = _primary_sha(text, shas)
    has_open_marker = bool(_OPEN_WORK_RE.search(text))
    primary_merged = bool(primary) and _sha_on_origin_main(primary)
    # Evidence: which of the quoted SHAs are individually on origin/main.
    merged_shas = [s for s in shas if _sha_on_origin_main(s)]

    if primary_merged and not has_open_marker:
        tier = "READY"
    elif primary_merged and has_open_marker:
        tier = "LANDED-BUT-OPEN"
    else:
        tier = "UNVERIFIED"

    return {
        "file": path.name,
        "tier": tier,
        "primary_sha": primary,
        "primary_merged": primary_merged,
        "has_open_marker": has_open_marker,
        "shas": shas,
        "merged_shas": merged_shas,
    }


def tier_all_batons() -> list[dict]:
    """Tier every project_*.md baton in MEMORY_DIR. Sorted READY-first then name."""
    if not MEMORY_DIR.is_dir():
        return []
    order = {"READY": 0, "LANDED-BUT-OPEN": 1, "UNVERIFIED": 2}
    out = [tier_baton(p) for p in sorted(MEMORY_DIR.glob("project_*.md"))]
    out.sort(key=lambda r: (order.get(r["tier"], 9), r["file"]))
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_report(budget: dict, batons: list[dict]) -> str:
    lines: list[str] = []
    lines.append("=== MEMORY.md budget ===")
    if not budget.get("exists"):
        lines.append(f"MEMORY.md not found at {budget['path']}")
    else:
        b = budget
        line_flag = "OVER" if b["over_line_budget"] else "ok"
        byte_flag = "OVER" if b["over_byte_budget"] else "ok"
        lines.append(f"path:  {b['path']}")
        lines.append(f"lines: {b['lines']} / {b['line_budget']}  [{line_flag}]")
        lines.append(f"bytes: {b['bytes']} / {b['byte_budget']}  [{byte_flag}]")
        if b["first_dropped"]:
            d = b["first_dropped"]
            lines.append(f"  -> first silently-dropped line: {d['line']} (cutoff hit by {d['by']})")
        else:
            lines.append("  -> whole index loads (within both budgets)")
        if b["overlong_lines"]:
            lines.append(
                f"  -> {len(b['overlong_lines'])} over-long line(s) (> {OVERLONG_LINE_CHARS} chars) inflating bytes:"
            )
            for ol in b["overlong_lines"][:20]:
                lines.append(f"       line {ol['line']}: {ol['chars']} chars")
            if len(b["overlong_lines"]) > 20:
                lines.append(f"       ... +{len(b['overlong_lines']) - 20} more")

    lines.append("")
    lines.append("=== project batons — clear tiers ===")
    counts = {"READY": 0, "LANDED-BUT-OPEN": 0, "UNVERIFIED": 0}
    for r in batons:
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1
    lines.append(
        f"READY={counts['READY']}  "
        f"LANDED-BUT-OPEN={counts['LANDED-BUT-OPEN']}  "
        f"UNVERIFIED={counts['UNVERIFIED']}  (total {len(batons)})"
    )
    lines.append("")
    for r in batons:
        sha = r["primary_sha"] or "—"
        marker = "open-work" if r["has_open_marker"] else "no-open-marker"
        lines.append(f"[{r['tier']:<15}] {r['file']}")
        lines.append(
            f"    primary={sha} merged={r['primary_merged']} {marker} "
            f"| {len(r['merged_shas'])}/{len(r['shas'])} SHAs on origin/main"
        )
    return "\n".join(lines)


def _render_clear_block(batons: list[dict]) -> str:
    """Fully-COMMENTED paste-ready rm + index-trim block for READY batons ONLY.

    Every line is a comment — nothing executes if pasted accidentally. The
    operator must un-comment deliberately. Never includes LANDED-BUT-OPEN or
    UNVERIFIED batons.
    """
    ready = [r for r in batons if r["tier"] == "READY"]
    out: list[str] = []
    out.append("# ============================================================")
    out.append("# MEMORY CLEAR BLOCK — READY-tier batons only (operator-run)")
    out.append("# Every line is COMMENTED. Review, then un-comment to execute.")
    out.append("# This tool NEVER deletes; you do, deliberately.")
    out.append("# ============================================================")
    if not ready:
        out.append("# (no READY-tier batons — nothing safe to clear)")
        return "\n".join(out)
    out.append(f"# {len(ready)} READY baton(s):")
    for r in ready:
        sha = r["primary_sha"] or "?"
        out.append(f"#   primary SHA {sha} proven on origin/main, no open-work marker")
        out.append(f"#   rm '{MEMORY_DIR / r['file']}'")
    out.append("#")
    out.append("# Then remove each baton's pointer line from MEMORY.md by hand")
    out.append("# (search the filename stem; the index is hand-curated).")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only MEMORY.md budget + baton clear-tier report.")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument(
        "--print-clear",
        action="store_true",
        help="emit a COMMENTED paste-ready clear block for READY-tier batons",
    )
    args = parser.parse_args(argv)

    budget = budget_report()
    batons = tier_all_batons()

    if args.json:
        print(json.dumps({"budget": budget, "batons": batons}, indent=2))
        return 0

    print(_render_report(budget, batons))
    if args.print_clear:
        print("")
        print(_render_clear_block(batons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
