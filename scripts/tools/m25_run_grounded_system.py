#!/usr/bin/env python
"""Rerun grounded system improvement with 128K output + nb resources."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.m25_audit import load_api_key, audit, ARCHITECTURE_CONTEXT

api_key = load_api_key()

# ── Extract nb resources ──────────────────────────────────────────────
NB_DIR = Path(r"C:\Users\joshd\OneDrive\Desktop\Organisation\nb resources")

TARGETS = {
    "Robert Carver - Systematic Trading.pdf": {
        "search": [
            "position size",
            "kelly",
            "trailing stop",
            "exit",
            "portfolio",
            "volatility target",
            "instrument weight",
            "forecast",
            "drawdown",
            "risk",
            "sharpe",
            "diversification",
        ],
        "max_pages": 25,
    },
    "Algorithmic_Trading_Chan.pdf": {
        "search": [
            "exit",
            "stop loss",
            "trailing",
            "position size",
            "kelly",
            "portfolio",
            "regime",
            "drawdown",
            "risk",
            "sharpe",
        ],
        "max_pages": 25,
    },
    "Building_Reliable_Trading_Systems.pdf": {
        "search": [
            "exit",
            "profit target",
            "trailing",
            "scale out",
            "partial",
            "stop",
            "position",
            "risk",
            "robust",
            "walk-forward",
        ],
        "max_pages": 25,
    },
    "Quantitative_Trading_Chan_2008.pdf": {
        "search": [
            "kelly",
            "position",
            "exit",
            "drawdown",
            "risk management",
            "sharpe",
            "backtest",
            "regime",
        ],
        "max_pages": 20,
    },
    "Lopez_de_Prado_ML_for_Asset_Managers.pdf": {
        "search": [
            "position size",
            "bet size",
            "meta-label",
            "exit",
            "portfolio",
            "feature importance",
            "walk-forward",
            "purge",
            "embargo",
        ],
        "max_pages": 20,
    },
    "real_time_strategy_monitoring_cusum.pdf": {
        "search": [
            "cusum",
            "monitoring",
            "regime",
            "decay",
            "drift",
            "performance",
            "sequential",
            "change point",
        ],
        "max_pages": 15,
    },
    "deflated-sharpe.pdf": {
        "search": [
            "sharpe",
            "deflated",
            "multiple testing",
            "haircut",
            "backtest",
            "expected maximum",
        ],
        "max_pages": 15,
    },
}

all_extracts = []
for fname, config in TARGETS.items():
    fpath = NB_DIR / fname
    if not fpath.exists():
        continue
    doc = fitz.open(str(fpath))
    relevant_pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().lower()
        if any(term in text for term in config["search"]):
            relevant_pages.append(page_num)
    relevant_pages = relevant_pages[: config["max_pages"]]
    if relevant_pages:
        extract = f"### {fname} ({len(relevant_pages)} relevant pages)\n"
        for pn in relevant_pages:
            page = doc[pn]
            text = page.get_text()
            if len(text) > 3000:
                text = text[:3000] + "... [truncated]"
            extract += f"\n--- Page {pn + 1} ---\n{text}\n"
        all_extracts.append(extract)
        print(f"  {fname}: {len(relevant_pages)} pages", file=sys.stderr)
    doc.close()

nb_content = "\n\n===\n\n".join(all_extracts)
if len(nb_content) > 150000:
    nb_content = nb_content[:150000] + "\n\n... [TRUNCATED for API limits]"
print(f"NB extracts: {len(nb_content):,} chars", file=sys.stderr)

# ── Load system files ─────────────────────────────────────────────────
KEY_FILES = [
    "CLAUDE.md",
    "TRADING_RULES.md",
    "RESEARCH_RULES.md",
    "MARKET_PLAYBOOK.md",
    "trading_app/config.py",
    "trading_app/live_config.py",
    "pipeline/cost_model.py",
    "trading_app/outcome_builder.py",
    "trading_app/execution_engine.py",
    "trading_app/paper_trader.py",
    "trading_app/entry_rules.py",
    "pipeline/dst.py",
    "pipeline/build_daily_features.py",
    "trading_app/ml/config.py",
    "trading_app/ml/predict_live.py",
]

file_parts = []
for f in KEY_FILES:
    p = PROJECT_ROOT / f
    if p.exists():
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if len(lines) > 1500:
            content = "\n".join(lines[:1500]) + f"\n\n... [TRUNCATED - {len(lines)} total lines]"
        file_parts.append(f"### {f} ({len(lines)} lines)\n```\n{content}\n```")

system_files = "\n\n---\n\n".join(file_parts)

# ── Build prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""\
You are the Chief Investment Officer at a $5B systematic macro fund. You have deployed \
ORB breakout systems across futures markets for 15 years. You have read ALL of the academic \
references provided below and are applying their frameworks to THIS specific system.

You have UNLIMITED output space. Be EXHAUSTIVE. Show formulas, pseudocode, implementation sketches.
Do not summarize — EXPAND. Every recommendation must have academic citation, implementation formula,
and validation protocol.

{ARCHITECTURE_CONTEXT}

---

**SYSTEM CONTEXT**

This is a complete ORB breakout trading system for 4 micro futures (MGC, MNQ, MES, M2K).
Current state: ~735 FDR-validated strategies, 12 ML-enabled sessions, +251.9R honest OOS from ML overlay.
The system works. Strategies are validated. ML adds value.

What has been tried and KILLED (do NOT suggest):
- E0 entry (artifact, purged), DOW filters (0 BH survivors), Calendar overlays (0 BH survivors)
- Pre-break compression (90+ tests, 0 BH survivors), Break quality bars (look-ahead/no signal)
- Non-ORB strategies (540 tests, 0 FDR survivors), Dead instruments (MCL/SIL/M6E/MBT)
- RSI, MACD, Bollinger, MA crossovers, Stochastics (guilty until proven per RESEARCH_RULES)

---

**YOUR TASK: ACADEMICALLY-GROUNDED TOTAL SYSTEM IMPROVEMENT BRIEF**

For EVERY recommendation, you MUST:
1. **Cite the specific academic source** (book, chapter, page if possible) from the literature provided
2. **Quote or paraphrase the author's key argument**
3. **State the author's WARNING/CAVEAT** about the approach
4. **Provide the EXACT implementation formula** adapted to our variables (atr_20, orb_size, pnl_r, etc.)
5. **Write pseudocode** for the validation experiment
6. **State expected impact** grounded in the literature's reported effect sizes
7. **State what could go wrong**

Cover ALL of these areas:

## 1. EXIT OPTIMIZATION (Grounded in Murray, Carver, Chan)
- C8 cap optimization (is 8R right?)
- C3 trailing stop lookback (2-bar vs 3-bar vs 5-bar)
- Partial profit taking / scale-out at 1R/2R
- Breakeven stop (conditional on winner speed)
- Session-specific exits (T80 extension)
- Time-based exits (academic framework)

## 2. POSITION SIZING (Grounded in Carver Ch.9-10, Chan, de Prado)
- Volatility-targeted sizing (Carver's framework)
- Kelly criterion — when safe vs dangerous
- Half-Kelly and fractional Kelly
- ATR-normalized position sizing formula
- Risk budget allocation across instruments

## 3. PORTFOLIO CONSTRUCTION (Grounded in Carver Ch.11, de Prado)
- Correlation-aware position limits
- Instrument weighting (equal risk vs optimized)
- Diversification multiplier
- Cross-instrument hedging
- Drawdown-triggered de-leveraging

## 4. REGIME DETECTION & ADAPTATION (Grounded in literature)
- Volatility regime filters (beyond G-gates)
- Trend/mean-reversion regime identification
- CUSUM monitoring for strategy decay
- Dynamic strategy selection by regime

## 5. RISK MANAGEMENT (Grounded in all sources)
- Daily/weekly loss limits
- Maximum drawdown triggers
- Tail risk protection
- Correlation spike detection

## 6. ML IMPROVEMENTS (Grounded in de Prado)
- LightGBM vs RF (academic comparison)
- Bet sizing from meta-label probabilities (de Prado Ch.10)
- Feature importance stability (de Prado)
- Walk-forward for ML (CPCV adequacy)

## 7. THE +100R ROADMAP
Top 10 improvements ranked by ROI. Each with:
- Academic source and citation
- Expected R gain (realistic)
- Implementation pseudocode
- Validation experiment design
- What the literature warns could go wrong

CRITICAL: Every recommendation must cite a specific source from the provided literature.
If the literature WARNS AGAINST something, say so clearly — the literature wins over intuition.
Do NOT suggest things already tried and killed.
"""

user_content = (
    "## ACADEMIC LITERATURE (from our reference library)\n\n"
    + nb_content
    + "\n\n---\n\n## CURRENT SYSTEM CONFIGURATION\n\n"
    + system_files
)

total_chars = len(user_content) + len(SYSTEM_PROMPT)
print(f"Total payload: {total_chars:,} chars (~{total_chars // 4:,} tokens)", file=sys.stderr)
print("Sending grounded system improvement prompt to M2.5 (128K output)...", file=sys.stderr)

result = audit(user_content, SYSTEM_PROMPT, api_key, include_context=False, timeout=600.0)

ts = datetime.now().strftime("%Y%m%d_%H%M")
out_dir = PROJECT_ROOT / "research" / "output"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"m25_grounded_system_128k_{ts}.md"
out.write_text(
    f"# M2.5 Academically-Grounded System Improvement Brief (128K output)\n"
    f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n"
    f"**Sources:** 7 academic books/papers + full system codebase\n"
    f"**Max tokens:** 131072\n\n---\n\n" + result,
    encoding="utf-8",
)
print(f"\nSaved: {out}", file=sys.stderr)
print(f"Output length: {len(result):,} chars", file=sys.stderr)
