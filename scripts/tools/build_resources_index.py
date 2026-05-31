#!/usr/bin/env python3
"""Build resources/INDEX.md — a cheap navigation manifest for the local corpus.

Why this exists
---------------
`resources/` carries the institutional PDF library (Lopez de Prado, Harvey-Liu,
Carver, Chan, Aronson, Harris microstructure, CUSUM/SR, FDR papers, plus the
ProjectX API spec and prop-firm rules). Existing rules *require* grounding in
these local sources before citing training memory (CLAUDE.md § Local Academic /
Project-Source Grounding Rule; institutional-rigor.md § 7), but nothing made the
check cheap — so grounding meant either a 24-PDF sweep or skipping it.

This indexer makes grounding a single small-file read. It emits a manifest that
maps each resource to: a topic, key terms (from the first text page), and — most
importantly — whether a CURATED, page-cited extract already exists in
`docs/institutional/literature/`. That literature dir is the canonical citation
source; INDEX.md points at it rather than duplicating it.

Cheapness contract
------------------
- One text page per PDF (first page with extractable text), capped — never a
  full-document parse. PyMuPDF (``fitz``) only; if unavailable, the entry still
  lists title + literature cross-link (degrade, don't fail).
- mtime guard: ``--check`` exits 0 (fresh) when INDEX.md is newer than every
  resource, so a hook can call it without rebuilding every session.
- Pure-ish: extraction is isolated; the manifest assembly is deterministic given
  the file set.

Usage
-----
    python scripts/tools/build_resources_index.py            # build/refresh
    python scripts/tools/build_resources_index.py --check     # exit 0 if fresh, 1 if stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / "resources"
LITERATURE_DIR = PROJECT_ROOT / "docs" / "institutional" / "literature"
INDEX_PATH = RESOURCES_DIR / "INDEX.md"

# Document extensions we index at the resources/ top level.
DOC_EXTS = {".pdf", ".md", ".epub", ".txt"}

# Hand-curated topic + literature-extract crosswalk. Keyed by a substring of the
# resource filename (lowercased). The extract stem points into
# docs/institutional/literature/<stem>.md — the CANONICAL citation source. Where
# an extract exists, cite FROM it (with page numbers), not from the raw PDF and
# never from training memory. Keep this table in sync when a new extract lands;
# unknown resources still get indexed (topic="(uncurated)") so nothing is hidden.
CROSSWALK: dict[str, tuple[str, str | None]] = {
    "benjamini": ("Multiple-testing / FDR (BH-1995 primary)", "benjamini_hochberg_1995_fdr"),
    "two_million": ("Multiple-testing — t≥3.79 empirical bound", "chordia_et_al_2018_two_million_strategies"),
    "pseudo-mathematics": (
        "Multiple-testing — MinBTL bound (caps brute-force ≤300)",
        "bailey_et_al_2013_pseudo_mathematics",
    ),
    "deflated-sharpe": ("Multiple-testing — Deflated Sharpe Ratio", "bailey_lopez_de_prado_2014_deflated_sharpe"),
    "false-strategy": ("Multiple-testing — False Strategy Theorem", "lopez_de_prado_bailey_2018_false_strategy"),
    "man_overfitting": ("Backtest overfitting (Man/2015)", None),
    "backtesting_dukepeople": ("Backtesting / Sharpe haircut (Harvey-Liu)", "harvey_liu_2015_backtesting"),
    "evidence_based_technical": ("Data-snooping / EBTA (Aronson)", "aronson_2007_ebta_data_snooping"),
    "lopez_de_prado_ml": ("ML for Asset Managers — theory-first, CPCV", "lopez_de_prado_2020_ml_for_asset_managers"),
    "algorithmic_trading_chan": (
        "Backtest method / look-ahead / TOC (Chan 2013)",
        "chan_2013_ch1_backtesting_lookahead",
    ),
    "quantitative_trading_chan": (
        "Intraday sessions / regime (Chan 2008/09)",
        "chan_2009_ch1_intraday_session_handling",
    ),
    "harris": (
        "Market microstructure — stop cascades, adverse selection (Harris)",
        "harris_2002_trading_exchanges_microstructure",
    ),
    "carver": (
        "Vol targeting / portfolio construction / sizing (Carver)",
        "carver_2015_volatility_targeting_position_sizing",
    ),
    "building_reliable": (
        "Trade-system reliability (Fitschen) — ORB premise",
        "fitschen_2013_path_of_least_resistance",
    ),
    "real_time_strategy_monitoring": (
        "Live monitoring — Shiryaev-Roberts CUSUM",
        "pepelyshev_polunchenko_2015_cusum_sr",
    ),
    "rober prado": ("Trading-strategy optimization (Pardo)", None),
    "projectx_api": ("Broker — ProjectX/TopstepX API spec (live execution)", None),
    "prop-firm-official": ("Prop-firm official rules (account death / MLL)", None),
    "ml4am_code_companion": (
        "ML4AM code companion (Lopez de Prado notebooks)",
        "lopez_de_prado_2020_ml_for_asset_managers",
    ),
    "harris_ocr": (
        "Harris microstructure — OCR full text (searchable)",
        "harris_2002_trading_exchanges_microstructure",
    ),
    # Value-area / auction-market-theory corpus — UNREVIEWED 2026 preprints
    # (theory_grant:false; require t>=3.79). Closed gaps G1-G3: extract existed,
    # source PDF was located-but-uncommitted and is now copied into resources/
    # (untracked — CI-safe mode unaffected; strict-local verifies page count).
    "howard_2026_value_area": (
        "Value-area breakout / stop methodology (Howard 2026, UNREVIEWED preprint)",
        "howard_2026_value_area_breakouts_es",
    ),
    "tolusic_2026_amt": (
        "Auction-market-theory inventory dynamics (Tolusic 2026, UNREVIEWED preprint)",
        "tolusic_2026_amt_inventory_dynamics",
    ),
    "advances_in_financial_machine_learning": (
        "Advances in Financial Machine Learning — full book (Lopez de Prado 2018)",
        "lopez_de_prado_2018_afml_ch_3_7_8",
    ),
}


def _topic_for(name: str) -> tuple[str, str | None]:
    low = name.lower()
    for key, (topic, stem) in CROSSWALK.items():
        if key in low:
            # Only return a literature stem if the extract file actually exists.
            if stem and not (LITERATURE_DIR / f"{stem}.md").exists():
                return topic, None
            return topic, stem
    return "(uncurated — extract TOC + 3 mid pages before dismissing)", None


def _first_text_terms(pdf_path: Path, max_chars: int = 600) -> str:
    """Cheap key-term snippet: first page that yields text, squeezed to one line.

    PyMuPDF only; returns "" on any failure (degrade, don't fail). Never parses
    more than the first few pages — this is a navigation hint, not a summary.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page in doc[:3]:  # at most 3 pages
                # "text" mode always returns str; coerce defensively for the type
                # checker (fitz is untyped and get_text() is overloaded).
                text = str(page.get_text("text") or "")
                if text.strip():
                    return re.sub(r"\s+", " ", text.strip())[:max_chars]
        return ""
    except Exception:
        return ""


def _iter_resources() -> list[Path]:
    """Top-level documents + the two known subdir corpora, sorted by name."""
    if not RESOURCES_DIR.is_dir():
        return []
    items: list[Path] = []
    for p in RESOURCES_DIR.iterdir():
        if p.name == "INDEX.md":
            continue
        if p.is_file() and p.suffix.lower() in DOC_EXTS:
            items.append(p)
        elif p.is_dir():
            items.append(p)  # subdir corpus (ml4am_code_companion, harris_ocr)
    return sorted(items, key=lambda p: p.name.lower())


def _is_fresh() -> bool:
    """True when INDEX.md exists and is newer than every resource (mtime guard)."""
    if not INDEX_PATH.exists():
        return False
    index_mtime = INDEX_PATH.stat().st_mtime
    for p in _iter_resources():
        try:
            if p.stat().st_mtime > index_mtime:
                return False
        except OSError:
            return False
    return True


def build_index() -> str:
    resources = _iter_resources()
    lines: list[str] = [
        "# Resources Index — local corpus navigation manifest",
        "",
        "> **Generated** by `scripts/tools/build_resources_index.py`. Do not edit by hand.",
        "> Rebuild: `python scripts/tools/build_resources_index.py`.",
        "",
        "**Grounding rule (CLAUDE.md § Local Academic / Project-Source Grounding):**",
        "before citing training memory on any topic below, ground in the local source.",
        "Where a **curated extract** exists in `docs/institutional/literature/`, that",
        "extract is the **canonical citation source** (page-cited) — fetch it via the",
        "`research-catalog` MCP (`get_literature_excerpt`), do NOT paraphrase the raw PDF",
        "from memory. If no local source supports a claim, say **UNSUPPORTED**.",
        "",
        f"**{len(resources)} resources indexed.**",
        "",
        "| Resource | Topic | Curated extract (canonical cite) | Key terms (first page) |",
        "|---|---|---|---|",
    ]
    for p in resources:
        rel = p.name + ("/" if p.is_dir() else "")
        topic, stem = _topic_for(p.name)
        if stem:
            cite = f"`docs/institutional/literature/{stem}.md`"
        else:
            cite = "—"
        terms = ""
        if p.is_file() and p.suffix.lower() == ".pdf":
            snippet = _first_text_terms(p)
            # keep the cell short — first ~140 chars, pipe-escaped
            terms = snippet[:140].replace("|", "\\|") if snippet else "(no extractable text — OCR/scan)"
        elif p.is_dir():
            terms = "(subdir corpus — see its README / page files)"
        elif p.suffix.lower() == ".md":
            terms = "(markdown — read directly)"
        lines.append(f"| `{rel}` | {topic} | {cite} | {terms} |")

    lines.extend(
        [
            "",
            "## How to ground cheaply",
            "",
            "1. Find the row for your topic above.",
            "2. If it has a **curated extract**, cite from that file (page numbers included).",
            "3. If not (`—`), extract the TOC + 3 mid-document pages from the PDF before",
            "   characterizing or dismissing it (terminology differs across sources — see",
            "   institutional-rigor.md § 7 extract-before-dismiss rule).",
            "4. Never cite a resource you have not opened this session as if you read it.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build resources/INDEX.md navigation manifest.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if INDEX.md is fresh (newer than all resources), 1 if stale/missing. No write.",
    )
    args = parser.parse_args(argv)

    if not RESOURCES_DIR.is_dir():
        print("resources/ not found — nothing to index.", file=sys.stderr)
        return 0  # fail-open: no corpus is not an error

    if args.check:
        fresh = _is_fresh()
        print(f"resources/INDEX.md: {'fresh' if fresh else 'STALE (rebuild needed)'}")
        return 0 if fresh else 1

    content = build_index()
    # newline="\n" disables Windows text-mode CRLF translation so the committed
    # INDEX stays LF regardless of the OS that regenerates it (a Windows regen
    # otherwise silently flips the whole file to CRLF — a spurious full-file diff).
    INDEX_PATH.write_text(content, encoding="utf-8", newline="\n")
    n = content.count("\n| `")
    print(f"Wrote {INDEX_PATH.relative_to(PROJECT_ROOT)} ({n} resources indexed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
