# Project Overview

**Updated: 2026-03-22.** This is the AI context doc for external AI tools (Gemini, ChatGPT, etc.).

This project is a **Quantitative Trading System for Micro Futures** — a self-contained data pipeline and backtesting engine for Opening Range Breakout (ORB) strategies.

**Active instruments (Mar 2026):** MGC (Micro Gold), MNQ (Micro Nasdaq), MES (Micro S&P 500).
**Dead for ORB:** MCL, SIL, M6E, MBT, M2K (all tested, 0 validated strategies).

## Core Components:

1.  **Data Pipeline (`pipeline/`):**
    *   Ingests Databento DBN data, transforms 1-minute bars into 5-minute bars, builds daily trading features (ORBs, sessions, RSI, ATR), ensures database integrity through drift checks.
    *   Technology: Python, `databento`, `DuckDB` (embedded analytical database).
    *   Orchestration: `pipeline/run_pipeline.py` runs the full flow (ingest, build_5m, build_features, audit).

2.  **Trading Application (`trading_app/`):**
    *   Core logic for defining, validating, and executing quantitative trading strategies. Processes market data bar-by-bar.
    *   **Execution Engine (`trading_app/execution_engine.py`):** State machine — ORB detection, signal confirmation, trade entry/exit for 3 entry models:
        *   **E1:** Market-On-Next-Bar — honest conservative baseline. No backtest biases. Industry standard.
        *   **E2:** Stop-Market at ORB level + 1 tick slippage — honest aggressive entry. Includes fakeout fills. Industry standard for breakout backtesting.
        *   **E3:** Limit-At-ORB retrace — soft-retired. Fill-on-touch bias, adverse selection risk.
        *   **E0: PURGED (Feb 2026).** Had 3 compounding optimistic biases (fill-on-touch, fakeout exclusion, fill-bar wins). Won 33/33 combos = artifact. Replaced by E2.
    *   **Strategy Management:** Discovery (grid search ~105K combos), validation (walk-forward + BH FDR), portfolio construction, risk management.
    *   **Backtesting:** Historical replay via `trading_app/paper_trader.py`.

3.  **Sessions (10 total, all event-based):**
    All sessions resolve times dynamically per-day from `pipeline/dst.py` SESSION_CATALOG, eliminating DST contamination. Old fixed-clock sessions have been replaced.
    *   CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, US_DATA_830, NYSE_OPEN, US_DATA_1000, CME_PRECLOSE, COMEX_SETTLE, NYSE_CLOSE.

## Validation Pipeline:

*   **Grid Search:** ~105,000 strategy combinations (3 apertures x 6 RR x 5 CB x 13 filters x 3 entry models x 10 sessions x 3 instruments).
*   **Walk-forward validation:** 3-year train / 1-year test, rolling. WFE > 50% required.
*   **BH FDR:** Benjamini-Hochberg correction across all ~105K tests.
*   **Classification:** CORE (>=100 trades), REGIME (30-99), INVALID (<30). min_sample=30.
*   **Current state (Mar 22 2026):** 832 validated (MGC 20, MES 81, MNQ 731). 17 live-tradeable strategies.

## What Works:

1.  ORB size IS the edge — strip the size filter and ALL edges die. G4+ minimum.
2.  MNQ E2 is the ONLY instrument x entry model with positive UNFILTERED baseline.
3.  MGC/MES need G5+ size filters on select sessions.
4.  TOKYO_OPEN is LONG-ONLY (short breakouts are negative).
5.  Negative session correlation provides real portfolio diversification.

## Confirmed NO-GOs:

No filter/L-filters, MGC/MES unfiltered, M2K/MCL/SIL/M6E/MBT (all 0 validated), E0 (3 biases), ML on negative baselines (p=0.350), non-ORB strategies (30 archetypes tested), calendar cascade, cross-asset lead-lag, prior-day context, breakeven trails, VWAP overlay, RSI reversion, gap fade.

## Key Technologies:

*   **Language:** Python 3.13+
*   **Data:** `databento`, `duckdb`, `pandas`, `numpy`, `pyarrow`
*   **Testing:** `pytest`, `pytest-xdist`
*   **Linting:** `ruff`, `pyright`
*   **CI/CD:** GitHub Actions

## Key Documents:

| Document | What It Covers |
|----------|---------------|
| `TRADING_RULES.md` | Sessions, entry models, cost models, edge summary, NO-GOs |
| `RESEARCH_RULES.md` | Statistical methodology, sample sizes, significance, mechanism test |
| `docs/STRATEGY_BLUEPRINT.md` | Research test sequence, gates, NO-GO registry |
| `CLAUDE.md` | Code architecture, guardrails, design principles |
| `docs/ARCHITECTURE.md` | Data flow, price sources, commands, classification rules |
| `HANDOFF.md` | Current session state (changes every session) |

## Building and Running:

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Pipeline
python pipeline/run_pipeline.py --instrument MGC --start 2016-02-01 --end 2026-02-04

# Strategy discovery + validation
python trading_app/outcome_builder.py --instrument MGC --start 2021-02-05 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_validator.py --instrument MGC --min-sample 30

# Tests & guardrails
python -m pytest tests/ -x -q
python pipeline/check_drift.py
```
