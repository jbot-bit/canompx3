#!/usr/bin/env python
"""
MiniMax M2.5 independent audit tool.

Second-opinion code review: sends files or queries to MiniMax M2.5
for independent analysis, separate from Claude.

IMPORTANT: M2.5 findings are UNVERIFIED SUGGESTIONS with a ~70% false
positive rate. Always cross-reference with Claude Code before acting.
CLAUDE.md is the authority — M2.5 cannot override it.

Usage:
    # Review a single file
    python scripts/tools/m25_audit.py pipeline/build_daily_features.py

    # Review multiple files
    python scripts/tools/m25_audit.py trading_app/outcome_builder.py trading_app/strategy_validator.py

    # Custom audit prompt
    python scripts/tools/m25_audit.py pipeline/dst.py --prompt "Check for DST bugs and timezone issues"

    # Preset audit modes
    python scripts/tools/m25_audit.py trading_app/outcome_builder.py --mode bias
    python scripts/tools/m25_audit.py pipeline/build_bars_5m.py --mode joins
    python scripts/tools/m25_audit.py trading_app/strategy_validator.py --mode bugs

    # Save output to file
    python scripts/tools/m25_audit.py pipeline/ingest_dbn.py --output audit_result.md

    # DEEP MODE: 3-turn reasoning + auto cross-file context (~3 API calls, ~30% FP rate)
    python scripts/tools/m25_audit.py trading_app/outcome_builder.py --deep --mode bias
    python scripts/tools/m25_audit.py pipeline/build_daily_features.py --deep --mode joins

    # DISCOVERY MODE: what features/improvements are missing from this code?
    python scripts/tools/m25_audit.py trading_app/ml/meta_label.py --mode discovery
    python scripts/tools/m25_audit.py pipeline/cost_model.py --mode discovery

    # PLAN MODE: 4-turn implementation planning for a new feature (~4 API calls)
    # Pass feature description as --plan, and relevant existing files as args
    python scripts/tools/m25_audit.py trading_app/entry_rules.py --plan "Add trailing stop to E2"
    python scripts/tools/m25_audit.py pipeline/build_daily_features.py trading_app/outcome_builder.py --plan "Add VWAP deviation feature to daily_features"

Setup:
    Set MINIMAX_API_KEY in your .env or environment:
        export MINIMAX_API_KEY=your-key-here
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────
API_URL = "https://api.minimax.io/v1/chat/completions"
MODEL_STANDARD = "MiniMax-M2.5"  # Deep analysis, improvements mode
MODEL_FAST = "MiniMax-M2.5"  # Flash variants not on api.minimax.io plan; same model as standard
MAX_CONTEXT = 200000  # M2.5 total context window (input + output) in tokens
MAX_TOKENS = 131072  # 128K default — auto-reduced if input is large
API_TIMEOUT_STANDARD = 600.0  # seconds — standard model, large files
API_TIMEOUT_FAST = 120.0  # seconds — Lightning is much quicker

# ── Call budget counter ──────────────────────────────────────────────
# Tracks daily API calls in ~/.m25_budget.json so you know where you stand.
# 100 calls per 5-hour window is the Claude Code integration limit.
import json as _json
from datetime import date as _date
from pathlib import Path as _Path

_BUDGET_FILE = _Path.home() / ".m25_budget.json"
_BUDGET_WINDOW = 100  # calls per window


def _increment_call_counter() -> tuple[int, int]:
    """Increment today's call count. Returns (today_count, window_remaining)."""
    today = str(_date.today())
    try:
        data = _json.loads(_BUDGET_FILE.read_text()) if _BUDGET_FILE.exists() else {}
    except Exception:
        data = {}
    data[today] = data.get(today, 0) + 1
    # Prune old dates (keep last 7 days)
    keys = sorted(data.keys())
    if len(keys) > 7:
        for k in keys[:-7]:
            del data[k]
    try:
        _BUDGET_FILE.write_text(_json.dumps(data))
    except Exception:
        pass
    return data[today], max(0, _BUDGET_WINDOW - data[today])


def show_budget() -> None:
    """Print today's call usage."""
    today = str(_date.today())
    try:
        data = _json.loads(_BUDGET_FILE.read_text()) if _BUDGET_FILE.exists() else {}
    except Exception:
        data = {}
    used = data.get(today, 0)
    remaining = max(0, _BUDGET_WINDOW - used)
    print(f"M2.5 budget today: {used} used / {remaining} remaining (window: {_BUDGET_WINDOW})")
    # Show last 3 days for context
    for d in sorted(data.keys())[-3:]:
        marker = " ← today" if d == today else ""
        print(f"  {d}: {data[d]} calls{marker}")


# ── Architecture context ────────────────────────────────────────────
# Prepended to every audit to prevent M2.5's known false positive patterns.
# These are FACTS about the codebase that M2.5 consistently gets wrong
# because it can't trace cross-file architecture.
ARCHITECTURE_CONTEXT = """\
IMPORTANT — Read ALL of this before analysing any file.
This is a multi-instrument futures ORB (Opening Range Breakout) trading pipeline.
The facts below prevent false positives that arise from single-file analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 — TECHNOLOGY PATTERNS (these are correct design, not bugs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DuckDB replacement scans
   WHY: DuckDB natively references in-scope pandas DataFrames in SQL
   (`SELECT * FROM chunk_df` with no con.register()). This is documented
   DuckDB behaviour, chosen deliberately for zero-copy performance.
   WRONG to flag: "chunk_df is not registered as a DuckDB table"
   CORRECT: this is valid DuckDB syntax.

2. Multi-stage statistical pipeline
   WHY: outcome_builder.py pre-computes ALL parameter combinations (grid
   search). Benjamini-Hochberg FDR correction is applied DOWNSTREAM in
   strategy_validator.py. The grid search is not snooping — the correction
   lives in a different file by design (separation of concerns).
   WRONG to flag: "grid search without multiple-testing correction"
   CORRECT: only flag if strategy_validator.py itself is missing BH FDR.

3. ML 4-gate quality system
   WHY: The ML overlay uses four gates (delta_r >= 0, CPCV AUC >= 0.50,
   test AUC > 0.52, skip rate <= 85%) as a combined system. No single gate
   is sufficient alone — together they guard against overfitting.
   WRONG to flag: "delta_r >= 0 is too lenient a threshold"
   CORRECT: evaluate only if ALL four gates are simultaneously trivially met.

4. Fail-open ML design
   WHY: When ML models are absent or fail, the system takes all trades
   (fail-open). In a trading system, a missed trade is a realised loss;
   a false positive (taking a bad trade) is merely a potential loss. The
   asymmetry intentionally favours participation over caution.
   WRONG to flag: "system should fail-closed if ML model is missing"

5. atexit exception handling
   WHY: `except Exception: pass` inside atexit handlers is correct Python.
   Raising during interpreter shutdown produces spurious noise tracebacks
   with no actionable effect. This is the standard pattern for cleanup code.
   WRONG to flag: "silent exception swallowed in atexit"

6. -999.0 NaN sentinel
   WHY: sklearn RandomForest cannot handle NaN. -999.0 is an intentional
   domain sentinel: for level-proximity features it means "no prior level
   exists" (a meaningful signal). All real feature values are in 0–5 range,
   so -999.0 is safely out-of-band and distinguishable.
   WRONG to flag: "fillna(-999.0) will corrupt feature distributions"

7. DELETE + INSERT idempotency
   WHY: Every write operation deletes rows for the target date range then
   re-inserts. This makes all pipeline stages safe to re-run without
   duplicates. It is intentional, not inefficient.
   WRONG to flag: "DELETE before INSERT is redundant; use UPSERT"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2 — ENTRY MODELS & STRATEGY STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entry models in this system:
- E1: market order after confirm bar (honest conservative baseline) — ACTIVE
- E2: stop-market at ORB high/low (industry-standard) — ACTIVE, dominant
- E0: limit order at ORB boundary — PURGED Feb 2026 (3 structural biases
  confirmed: fill-on-touch artefact, fakeout exclusion, fill-bar wins).
  E0 absence is correct. Any reference to E0 as "missing" is a false positive.
- E3: wider-stop variant — SOFT-RETIRED. Still present in DB for history;
  retire_e3_strategies.py removes promoted E3 after each validator run.

Active instruments: MGC, MNQ, MES, M2K (micro futures).
Dead instruments (zero ORB edge, validated by research): MCL, SIL, M6E, MBT.
WRONG to flag: "MCL not included — possible survivorship bias"
CORRECT: dead instruments were tested; their absence is a research conclusion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3 — PIPELINE ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

One-way dependency: pipeline/ → trading_app/ (never reversed).
If you see trading_app importing from pipeline/, that is correct.
If you see pipeline/ importing from trading_app/, that is a real bug.

Session architecture: ALL sessions are dynamic/event-based, resolved per-day
from pipeline/dst.py SESSION_CATALOG (e.g. CME_REOPEN, TOKYO_OPEN, NYSE_OPEN).
There are no hardcoded clock times for sessions. DST contamination was fully
resolved in Feb 2026. The old fixed-clock sessions (0900/1800/0030/2300) were
replaced. References to fixed session times in SESSION_WINDOWS in
build_daily_features.py are Brisbane-time approximations for stats display only.
WRONG to flag: "hardcoded session time 09:00 — will break under DST"

daily_features JOIN invariant: daily_features has 3 rows per (trading_day,
symbol) — one per orb_minutes (5, 15, 30). Any JOIN with orb_outcomes MUST
include orb_minutes in the ON clause or row count triples. This IS a real bug
if the join is missing orb_minutes. For LAG() / window functions, always check
for WHERE d.orb_minutes = 5 inside the CTE to prevent cross-aperture leakage.

Cost model: ALL P&L calculations deduct round-trip friction (commission +
spread + slippage) via to_r_multiple() in pipeline/cost_model.py.
WRONG to flag: "no transaction costs applied" — they are, in cost_model.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4 — TRADING DOMAIN RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

G-filter semantics: ORB_G4 / ORB_G5 / ORB_G6 / ORB_G8 mean ORB size >= N
POINTS (not ATR multiples, not percentages). Values like 4, 6, 8 are not
magic numbers — they are instrument-appropriate point thresholds.
WRONG to flag: "magic number 6 in ORB_G6 filter — should be a named constant"

Strategy classification thresholds (from config.py, intentional):
- CORE: N >= 100 trades — standalone portfolio weight
- REGIME: N 30–99 — conditional overlay / signal only
- INVALID: N < 30 — not tradeable
Low trade counts under strict G6/G8 filters are EXPECTED behaviour, not bugs.
WRONG to flag: "strategy has only 47 trades — insufficient sample size"
CORRECT: check if N < 30 (INVALID) or 30–99 (REGIME, which is valid as overlay).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5 — CONFIRMED RESEARCH NO-GOs (already tested, not worth re-suggesting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These have been rigorously tested and rejected. Do not recommend them:
- Day-of-week filters (DOW): 0 BH FDR survivors across all instruments.
- Calendar overlays (NFP/OPEX/FOMC): 0 BH FDR survivors at q=0.10.
- Non-ORB strategies (RSI, MACD, MA crossovers, Bollinger): 540 tests, 0 FDR survivors.
- Pre-break bar compression as a filter: rejected across all instruments/sessions.
- NODBL filter (no-double-break): removed Feb 2026 after 6 strategies proved artefacts.
- E0 entry model: purged. Any suggestion to restore limit-order fills is wrong.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only flag issues that are REAL given all the above facts.
If a concern is addressed in a file you have NOT been shown, say so explicitly
rather than flagging it as unhandled.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

AUDIT_MODES = {
    "general": (
        "You are a senior quant developer reviewing code for a futures trading pipeline.\n\n"
        "Structure your review as:\n"
        "1. **WELL-DONE** — specific praise for sound engineering decisions\n"
        "2. **FINDINGS** — real issues with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "3. **RECOMMENDATIONS** — concrete fixes ranked by impact\n\n"
        "Check for:\n"
        "- Bugs, logic errors, off-by-one errors\n"
        "- Error handling gaps (silent failures, swallowed exceptions)\n"
        "- Code quality issues (dead code, unreachable branches)\n"
        "- Resource management (unclosed connections, file handles)\n\n"
        "Be specific. Cite line numbers. Only flag real issues, not style preferences.\n"
        "Do NOT flag patterns explained in the architecture context above."
    ),
    "bias": (
        "You are the Head of Quantitative Research at a systematic futures fund.\n\n"
        "Structure your review as:\n"
        "1. **WELL-DONE** — specific praise for bias prevention measures already in place\n"
        "2. **FINDINGS** — real bias risks with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "3. **RECOMMENDATIONS** — concrete improvements ranked by impact\n\n"
        "Audit for:\n"
        "- LOOK-AHEAD BIAS: Features derived from post-entry information\n"
        "- SURVIVORSHIP BIAS: Ignoring dead instruments when drawing conclusions\n"
        "- DATA SNOOPING: Multiple hypothesis testing without BH FDR correction "
        "(NOTE: BH FDR IS applied in strategy_validator.py — check if the file "
        "you're reviewing is upstream of that correction before flagging)\n"
        "- OVERFITTING: Too few samples (N<30) or too many free parameters\n"
        "- TRANSACTION COST ILLUSION: Ignoring spread, slippage, or commission "
        "(NOTE: cost_model.py handles this — check before flagging)\n\n"
        "This is a futures ORB breakout system. daily_features has 3 rows per "
        "(trading_day, symbol) — one per orb_minutes (5, 15, 30).\n\n"
        "Be specific. Cite line numbers. Distinguish between issues in THIS file "
        "vs issues handled in OTHER files you cannot see."
    ),
    "joins": (
        "You are a database expert auditing SQL queries in a trading pipeline.\n\n"
        "Structure your review as:\n"
        "1. **SAFE PATTERNS** — correctly implemented SQL patterns\n"
        "2. **FINDINGS** — real SQL issues with severity\n"
        "3. **RECOMMENDATIONS** — fixes with corrected SQL\n\n"
        "Check for:\n"
        "- JOIN correctness: daily_features has 3 rows per (trading_day, symbol) — "
        "one per orb_minutes. Any JOIN MUST include orb_minutes or rows triple.\n"
        "- Missing WHERE clauses that could cause row explosion\n"
        "- LAG()/window functions without proper PARTITION BY or WHERE orb_minutes = 5\n"
        "- Aggregations that could double-count due to JOIN fan-out\n"
        "- DELETE+INSERT idempotency — verify the DELETE range matches the INSERT range\n\n"
        "NOTE: This project uses DuckDB, which can reference in-scope pandas "
        "DataFrames directly in SQL (replacement scans). Do NOT flag this as a bug.\n\n"
        "Cite line numbers. Show the problematic SQL and the fix."
    ),
    "bugs": (
        "You are a Python expert doing a thorough bug hunt on a trading system.\n\n"
        "Structure your review as:\n"
        "1. **WELL-DONE** — good defensive coding patterns you observe\n"
        "2. **FINDINGS** — real bugs with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "3. **RECOMMENDATIONS** — fixes ranked by impact\n\n"
        "Check for:\n"
        "- Type errors, None handling, missing edge cases\n"
        "- Off-by-one errors in date ranges, slicing, indexing\n"
        "- Timezone bugs (UTC vs local, naive vs aware datetimes)\n"
        "- Resource leaks (unclosed files, connections)\n"
        "- Subprocess calls without return code checks\n"
        "- Silent failures (except: pass, catch-all exception handlers)\n"
        "- Variable shadowing, mutation of shared state\n\n"
        "IMPORTANT: `except Exception: pass` in atexit handlers is correct Python "
        "for shutdown cleanup. Do NOT flag this pattern.\n\n"
        "Be specific. Cite line numbers. Only flag actual bugs, not style."
    ),
    "improvements": (
        "You are the Head of Quantitative Research at a systematic macro fund "
        "managing $2B AUM. You hold a PhD in Financial Engineering and have 18 years "
        "building production ML/quant systems for systematic trading.\n\n"
        "Structure your review as graded sections:\n\n"
        "For each section, provide:\n"
        "- GRADE (A/B/C/D/F) with 1-sentence justification\n"
        "- SPECIFIC FINDINGS (cite file:line where applicable)\n"
        "- RECOMMENDED IMPROVEMENTS ranked by impact/effort ratio\n"
        "- RISK RATING for each finding (CRITICAL / HIGH / MEDIUM / LOW)\n\n"
        "Evaluate:\n"
        "1. CODE QUALITY — architecture, error handling, testability\n"
        "2. STATISTICAL RIGOR — bias prevention, validation methodology\n"
        "3. PRODUCTION READINESS — monitoring, alerting, failure modes\n"
        "4. INSTITUTIONAL GAPS — what would a Bloomberg/Two Sigma system have?\n\n"
        "End with: OVERALL ASSESSMENT and top 3 highest-ROI improvements.\n\n"
        "Be specific. Distinguish between what's already well-done, genuine risks, "
        "and concrete improvements. Do NOT recommend things already implemented "
        "(check the architecture context above first)."
    ),
    "discovery": (
        "You are a senior quant researcher and systems architect. Your job is to read "
        "this code and identify HIGH-VALUE OPPORTUNITIES that don't yet exist.\n\n"
        "Structure your output as:\n\n"
        "## 1. WHAT EXISTS (orient yourself first)\n"
        "One paragraph summarising what this code currently does and its quality.\n\n"
        "## 2. FEATURE OPPORTUNITIES\n"
        "For each opportunity:\n"
        "- **Name**: concise label\n"
        "- **Gap**: what's missing or suboptimal\n"
        "- **Value**: why this matters for trading edge or system quality\n"
        "- **Effort**: S/M/L (Small = 1 file change, Medium = 2-3 files, Large = new module)\n"
        "- **Files to touch**: specific files/functions\n"
        "- **Risk**: what could go wrong if implemented naively\n\n"
        "## 3. PRIORITY RANKING\n"
        "Top 3 opportunities ranked by value/effort ratio.\n\n"
        "RULES:\n"
        "- Only suggest things genuinely absent from the current code\n"
        "- Be concrete: name the function to add, the column to compute, the check to insert\n"
        "- Do NOT suggest what the architecture context says is intentionally absent or dead\n"
        "- Distinguish 'trading edge improvement' from 'engineering quality improvement'"
    ),
    "architect": (
        "You are a senior quant systems architect. You have been given code from an "
        "existing trading pipeline. Your job is to design an implementation plan for "
        "a NEW FEATURE or CHANGE, working within the existing architecture.\n\n"
        "Structure your output as:\n\n"
        "## 1. CURRENT STATE ANALYSIS\n"
        "What exists today that is relevant to this feature. What would need to change.\n\n"
        "## 2. DESIGN\n"
        "- Data model: what new columns/tables/fields are needed\n"
        "- Interface: function signatures with types\n"
        "- Data flow: where does data come from, what transforms happen, where does it go\n"
        "- Integration points: which existing functions call or are called by the new code\n\n"
        "## 3. IMPLEMENTATION STEPS (ordered)\n"
        "Numbered steps with: file to edit, function to add/change, what exactly changes.\n"
        "Be specific enough that a developer can execute each step without ambiguity.\n\n"
        "## 4. RISKS AND GUARDRAILS\n"
        "- What could silently go wrong (data leaks, join traps, timezone issues)\n"
        "- What tests are needed\n"
        "- What drift checks or assertions should be added\n\n"
        "RULES:\n"
        "- Respect the one-way dependency: pipeline/ → trading_app/ (never reversed)\n"
        "- Never suggest hardcoding values that belong in canonical sources\n"
        "- Every DB write must be idempotent (DELETE+INSERT pattern)\n"
        "- Every new join on daily_features MUST include orb_minutes"
    ),
    "gaps": (
        "You are a senior quant systems architect reviewing MULTIPLE files from the "
        "same trading pipeline simultaneously.\n\n"
        "Your task is CROSS-FILE GAP ANALYSIS — finding issues that are invisible when "
        "reviewing files one at a time. Single-file reviewers cannot see these.\n\n"
        "Structure your review as:\n"
        "1. **INTERFACE CONTRACTS** — do caller assumptions match callee behaviour? "
        "(e.g. column names, return types, date ranges passed between files)\n"
        "2. **INVARIANT GAPS** — rules enforced in one file but silently violated in "
        "another (e.g. orb_minutes join rule, filter_type naming, UTC timestamps)\n"
        "3. **MISSING GUARDS** — error that file A relies on file B to catch, but B "
        "doesn't (e.g. division by zero only safe if upstream guarantees N>0)\n"
        "4. **DATA FLOW BREAKS** — a value produced in file A that file B consumes "
        "incorrectly (wrong column, wrong aggregation level, wrong timezone)\n"
        "5. **DRIFT RISKS** — patterns that will silently break if one file changes "
        "without the other being updated\n\n"
        "IMPORTANT: Only flag cross-file issues. Do NOT repeat single-file bugs "
        "that are obvious from reading one file alone — those are covered by other modes.\n\n"
        "Be specific. Cite file:line for BOTH sides of each cross-file issue."
    ),
}


def gather_runtime_context(primary_paths: list[str]) -> str:
    """Gather runtime context that M2.5 cannot access on its own.

    M2.5 is a blind API call — no tools, no DB, no git. This function
    pre-gathers the context it would need to avoid false positives:
    - DB schema for tables referenced in the code
    - Current config values (entry models, filters, sessions, costs)
    - Test file contents for the target files
    - Recent git diff for the target files (last 5 commits)
    """
    import subprocess

    project_root = Path(__file__).parent.parent.parent
    parts: list[str] = []

    # ── 1. DB schema (if target touches DB) ──────────────────────────
    db_keywords = {
        "duckdb",
        "gold.db",
        "GOLD_DB",
        "bars_1m",
        "bars_5m",
        "daily_features",
        "orb_outcomes",
        "validated_setups",
        "experimental_strategies",
        "edge_families",
    }
    touches_db = False
    for p in primary_paths:
        path = Path(p) if Path(p).exists() else project_root / p
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")[:5000]
            if any(kw in content for kw in db_keywords):
                touches_db = True
                break

    if touches_db:
        try:
            db_path = project_root / "gold.db"
            if db_path.exists():
                import duckdb

                con = duckdb.connect(str(db_path), read_only=True)
                tables = con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
                ).fetchall()
                schema_lines = []
                for (tbl,) in tables:
                    cols = con.execute(
                        f"SELECT column_name, data_type FROM information_schema.columns "
                        f"WHERE table_name='{tbl}' ORDER BY ordinal_position"
                    ).fetchall()
                    col_str = ", ".join(f"{c} {t}" for c, t in cols)
                    count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    schema_lines.append(f"  {tbl} ({count:,} rows): {col_str}")
                con.close()
                parts.append(
                    "## DATABASE SCHEMA (live from gold.db)\n"
                    "M2.5 cannot query the DB. Use this to verify column names, "
                    "types, and table relationships.\n\n" + "\n".join(schema_lines)
                )
        except Exception as e:
            parts.append(f"## DATABASE SCHEMA\n(Could not read: {e})")

    # ── 2. Current config values ─────────────────────────────────────
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from trading_app.config import ENTRY_MODELS, FILTER_TYPES, ORB_MINUTES_OPTIONS, "
                    "CLASSIFICATION_THRESHOLDS, CONFIRM_BARS_OPTIONS, RR_TARGET_OPTIONS\n"
                    "print('ENTRY_MODELS:', sorted(ENTRY_MODELS))\n"
                    "print('FILTER_TYPES:', sorted(FILTER_TYPES))\n"
                    "print('ORB_MINUTES:', sorted(ORB_MINUTES_OPTIONS))\n"
                    "print('CONFIRM_BARS:', sorted(CONFIRM_BARS_OPTIONS))\n"
                    "print('RR_TARGETS:', sorted(RR_TARGET_OPTIONS))\n"
                    "print('CLASSIFICATION:', CLASSIFICATION_THRESHOLDS)\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(project_root),
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("## CURRENT CONFIG VALUES (live from trading_app/config.py)\n" + result.stdout.strip())
    except Exception:
        pass

    # ── 3. Active instruments + cost specs ───────────────────────────
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS\n"
                    "from pipeline.cost_model import COST_SPECS\n"
                    "print('ACTIVE_INSTRUMENTS:', sorted(ACTIVE_ORB_INSTRUMENTS))\n"
                    "for sym, spec in sorted(COST_SPECS.items()):\n"
                    "    print(f'  {sym}: {spec}')\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(project_root),
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("## ACTIVE INSTRUMENTS & COST SPECS (live)\n" + result.stdout.strip())
    except Exception:
        pass

    # ── 4. Session catalog summary ───────────────────────────────────
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pipeline.dst import SESSION_CATALOG\n"
                    "import datetime\n"
                    "# Show resolved Brisbane times for EST and EDT\n"
                    "est_day = datetime.date(2025, 1, 15)  # US EST\n"
                    "edt_day = datetime.date(2025, 6, 15)  # US EDT\n"
                    "print('Sessions (resolved Brisbane times, EST vs EDT):')\n"
                    "for name in sorted(SESSION_CATALOG):\n"
                    "    info = SESSION_CATALOG[name]\n"
                    "    event = info.get('event', 'unknown')\n"
                    "    bg = info.get('break_group', '?')\n"
                    "    resolver = info.get('resolver')\n"
                    "    if resolver:\n"
                    "        try:\n"
                    "            est_hm = resolver(est_day)\n"
                    "            edt_hm = resolver(edt_day)\n"
                    "            est_str = f'{est_hm[0]:02d}:{est_hm[1]:02d}'\n"
                    "            edt_str = f'{edt_hm[0]:02d}:{edt_hm[1]:02d}'\n"
                    "            shift = '' if est_str == edt_str else f' (shifts to {edt_str} in EDT)'\n"
                    "            print(f'  {name}: {est_str} Brisbane (EST){shift} | group={bg} | {event}')\n"
                    "        except Exception as e:\n"
                    "            print(f'  {name}: resolver error: {e} | group={bg} | {event}')\n"
                    "    else:\n"
                    "        print(f'  {name}: no resolver | group={bg} | {event}')\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(project_root),
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(
                "## SESSION CATALOG (live from pipeline/dst.py)\n"
                "These are ORB session START times in Brisbane time.\n"
                "US DST shifts some sessions by 1 hour.\n\n" + result.stdout.strip()
            )
    except Exception:
        pass

    # ── 5. Test files for target modules ─────────────────────────────
    test_content_parts = []
    for p in primary_paths:
        stem = Path(p).stem
        # Search common test locations
        for test_dir in ["tests/test_pipeline", "tests/test_trading_app", "tests"]:
            test_path = project_root / test_dir / f"test_{stem}.py"
            if test_path.exists():
                content = test_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                if len(lines) > 300:
                    content = "\n".join(lines[:300])
                    content += f"\n... [TRUNCATED at 300/{len(lines)} lines]"
                test_content_parts.append(
                    f"### Test file: {test_path.relative_to(project_root).as_posix()}\n```python\n{content}\n```"
                )
                break

    if test_content_parts:
        parts.append(
            "## TEST FILES (M2.5 cannot see these without injection)\n"
            "Use these to understand what IS tested vs what is NOT.\n\n" + "\n\n".join(test_content_parts)
        )

    # ── 6. Recent git history for target files ───────────────────────
    git_parts = []
    for p in primary_paths:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5", "--", p],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(project_root),
            )
            if result.returncode == 0 and result.stdout.strip():
                git_parts.append(f"  {p}:\n    " + "\n    ".join(result.stdout.strip().splitlines()))
        except Exception:
            pass

    if git_parts:
        parts.append("## RECENT GIT HISTORY (last 5 commits per file)\n" + "\n".join(git_parts))

    if not parts:
        return ""

    return (
        "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "RUNTIME CONTEXT (auto-gathered — M2.5 cannot access tools)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(parts)
        + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )


def build_diff_content(files: list[str], ref: str = "HEAD") -> str | None:
    """Build a focused diff view for M2.5 review.

    Instead of sending whole files, sends only changed hunks + 15 lines
    of surrounding context. M2.5 is strongest at reviewing small, focused
    changes — this plays to that strength.

    Returns None if no diff is available (new files, no changes, etc.).
    """
    import subprocess

    project_root = Path(__file__).parent.parent.parent
    diff_parts = []

    for f in files:
        try:
            # Get the diff with context
            result = subprocess.run(
                ["git", "diff", "-U15", ref, "--", f],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(project_root),
            )
            if result.returncode == 0 and result.stdout.strip():
                diff_parts.append(f"### DIFF: {f} (vs {ref})\n```diff\n{result.stdout.strip()}\n```")
            else:
                # Try staged diff
                result = subprocess.run(
                    ["git", "diff", "-U15", "--cached", "--", f],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(project_root),
                )
                if result.returncode == 0 and result.stdout.strip():
                    diff_parts.append(f"### DIFF (staged): {f} (vs {ref})\n```diff\n{result.stdout.strip()}\n```")
        except Exception:
            pass

    if not diff_parts:
        return None

    return (
        "## CODE CHANGES TO REVIEW\n"
        "Focus your audit on THESE CHANGES specifically. The surrounding context "
        "is provided for understanding, but your findings should relate to the "
        "changed lines (marked with + and -).\n\n" + "\n\n".join(diff_parts)
    )


def triage_output(raw_output: str) -> str:
    """Post-process M2.5 output to extract only actionable findings.

    Strips FALSE POSITIVE sections, keeps TRUE and WORTH EXPLORING.
    M2.5 has a ~41% FP rate — this filters the noise.
    """
    lines = raw_output.splitlines()
    result_lines: list[str] = []
    skip_section = False

    for line in lines:
        upper = line.upper().strip()

        # Detect verdict markers
        if "**FALSE POSITIVE**" in upper or "FALSE POSITIVE" in upper:
            skip_section = True
            continue
        if "**TRUE**" in upper or "**TRUE BUG**" in upper:
            skip_section = False
        if "**WORTH EXPLORING**" in upper:
            skip_section = False
            pass

        # Keep section headers and non-finding content
        if line.startswith("## ") or line.startswith("# "):
            skip_section = False
            # Skip "FALSE POSITIVE" headers
            if "FALSE POSITIVE" in upper:
                skip_section = True
                continue

        # Keep WELL-DONE, FINDINGS, RECOMMENDATIONS, SUMMARY headers
        if any(
            kw in upper
            for kw in [
                "WELL-DONE",
                "FINDING",
                "RECOMMENDATION",
                "SUMMARY",
                "OVERALL",
                "VERDICT",
                "TRUE FINDING",
                "WORTH EXPLORING",
                "CRITICAL",
                "HIGH",
                "MEDIUM",
            ]
        ):
            if "FALSE" not in upper:
                skip_section = False

        if not skip_section:
            result_lines.append(line)

    triaged = "\n".join(result_lines).strip()

    # Count findings for summary
    true_count = raw_output.upper().count("**TRUE**") + raw_output.upper().count("**TRUE BUG**")
    fp_count = raw_output.upper().count("**FALSE POSITIVE**")
    explore_count = raw_output.upper().count("**WORTH EXPLORING**")

    if true_count or fp_count or explore_count:
        summary = (
            f"\n\n--- TRIAGE SUMMARY ---\n"
            f"TRUE findings: {true_count} | "
            f"FALSE POSITIVEs filtered: {fp_count} | "
            f"WORTH EXPLORING: {explore_count}"
        )
        triaged += summary

    return triaged


def find_related_files(primary_paths: list[str], max_extra: int = 4) -> dict[str, str]:
    """Auto-detect project files imported by primary_paths and return their contents.

    M2.5's #1 false positive source is cross-file blindness — it flags "missing"
    behaviour that lives in a different module. This pulls in the files that the
    target code actually imports so M2.5 has the full picture.

    Returns: {relative_path: content} for related files (not the primary files).
    """
    import re as _re

    project_root = Path(__file__).parent.parent.parent

    # Map importable module names to relative file paths in this project
    module_map: dict[str, str] = {}
    for package in ("pipeline", "trading_app", "trading_app/ml"):
        pkg_dir = project_root / package.replace("/", "\\")
        if not pkg_dir.exists():
            pkg_dir = project_root / package
        for py_file in pkg_dir.glob("*.py") if pkg_dir.exists() else []:
            rel = py_file.relative_to(project_root).as_posix()
            # Register as both "package.module" and bare "module"
            mod_name = py_file.stem
            pkg_name = package.replace("/", ".")
            module_map[f"{pkg_name}.{mod_name}"] = rel
            module_map[mod_name] = rel

    # Parse imports from every primary file
    candidates: dict[str, int] = {}  # rel_path -> reference count
    for p in primary_paths:
        path = Path(p)
        if not path.exists():
            path = project_root / p
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        # Match: from pipeline.foo import ..., from trading_app.ml.bar import ...
        for m in _re.finditer(r"(?:from|import)\s+([\w.]+)", src):
            name = m.group(1)
            # Try full name and last two parts (e.g. "trading_app.config" -> "config")
            for key in (name, name.split(".")[-1], ".".join(name.split(".")[-2:])):
                if key in module_map:
                    rel = module_map[key]
                    if rel not in [Path(p).as_posix() for p in primary_paths]:
                        candidates[rel] = candidates.get(rel, 0) + 1

    # Sort by reference count (most-imported first), cap at max_extra
    ranked = sorted(candidates, key=lambda r: -candidates[r])[:max_extra]

    result: dict[str, str] = {}
    for rel in ranked:
        full = project_root / rel
        if full.exists():
            content = full.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > 500:
                # Keep first 500 lines for related files — enough for interfaces
                content = "\n".join(lines[:500])
                content += f"\n\n... [TRUNCATED at 500 lines — {len(lines)} total]"
            result[rel] = content
    return result


def _call_api(
    messages: list[dict],
    api_key: str,
    *,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> str:
    """Single raw API call. Returns the assistant message content."""
    payload = {
        "model": MODEL_STANDARD,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            break
        except httpx.ReadTimeout:
            if attempt == 0:
                timeout = min(timeout * 2, 900.0)
            else:
                raise
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def audit_deep(
    primary_paths: list[str],
    mode: str,
    api_key: str,
    *,
    verbose: bool = False,
    enrich: bool = False,
) -> str:
    """Multi-turn structured audit that mirrors how Claude reasons.

    Instead of one-shot "find bugs", this runs 3 turns:

      Turn 1 — UNDERSTAND: Describe what the code does, its purpose, key
               patterns, and any intentional design choices. No judgements yet.

      Turn 2 — HYPOTHESISE: Given your understanding + the architecture
               context, list potential issues. For each, state WHY you think
               it might be a problem and what evidence you need to confirm it.

      Turn 3 — VERIFY + VERDICT: For each hypothesis, quote the specific
               code that proves or disproves it. Mark each as TRUE / FALSE
               POSITIVE. Only report TRUE findings in the final output.

    Also auto-injects related files so M2.5 has cross-file context.
    Uses ~3 API calls per run.
    """
    project_root = Path(__file__).parent.parent.parent

    # Build primary file content
    primary_content_parts = []
    for p in primary_paths:
        path = Path(p)
        if not path.exists():
            path = project_root / p
        if not path.exists():
            print(f"WARNING: File not found: {p}", file=sys.stderr)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if len(lines) > 2000:
            content = "\n".join(lines[:2000])
            content += f"\n\n... [TRUNCATED — {len(lines)} total lines]"
        primary_content_parts.append(f"### PRIMARY FILE: {p}\n```python\n{content}\n```")

    if not primary_content_parts:
        raise ValueError("No readable primary files.")

    primary_content = "\n\n".join(primary_content_parts)

    # Auto-inject related files
    related = find_related_files(primary_paths)
    if related:
        related_parts = [
            f"### RELATED FILE (imported by primary): {rel}\n```python\n{content}\n```"
            for rel, content in related.items()
        ]
        context_block = (
            "\n\n---\n## CROSS-FILE CONTEXT (auto-injected)\n"
            "These files are imported by the primary file(s). Use them to verify "
            "cross-module claims before flagging anything.\n\n" + "\n\n".join(related_parts)
        )
        if verbose:
            print(
                f"  Auto-context: {len(related)} related file(s): {list(related)!r}",
                file=sys.stderr,
            )
    else:
        context_block = ""

    mode_instruction = AUDIT_MODES.get(mode, AUDIT_MODES["general"])
    system = (
        f"{ARCHITECTURE_CONTEXT}\n---\n\n{mode_instruction}\n\n"
        "CRITICAL RULE: You MUST complete all three turns (UNDERSTAND → HYPOTHESISE "
        "→ VERIFY). Do NOT skip to conclusions. In Turn 3 you MUST quote actual code "
        "lines that prove or disprove each hypothesis. If you cannot find code evidence "
        "for a finding, it is a FALSE POSITIVE."
    )

    full_code = primary_content + context_block

    # Enrich with runtime context if requested
    if enrich:
        if verbose:
            print("  Gathering runtime context (DB, config, tests, git)...", file=sys.stderr)
        runtime_ctx = gather_runtime_context(primary_paths)
        if runtime_ctx:
            full_code = full_code + "\n\n" + runtime_ctx

    # ── Turn 1: Understand ───────────────────────────────────────────
    t1_prompt = (
        "## TURN 1 — UNDERSTAND\n\n"
        "Read the code carefully. Do NOT flag any issues yet.\n"
        "Describe:\n"
        "1. What this code does and its purpose in the pipeline\n"
        "2. Key patterns and design choices (and why they might be intentional)\n"
        "3. What the code ASSUMES about its inputs and environment\n"
        "4. What other modules it depends on and what it expects from them\n\n"
        "Be factual. No bug-hunting yet.\n\n" + full_code
    )

    if verbose:
        print("  [deep] Turn 1: Understanding code...", file=sys.stderr)
    _increment_call_counter()
    msgs: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": t1_prompt},
    ]
    t1_response = _call_api(msgs, api_key, max_tokens=4096, timeout=180.0)
    msgs.append({"role": "assistant", "content": t1_response})

    # ── Turn 2: Hypothesise ──────────────────────────────────────────
    t2_prompt = (
        "## TURN 2 — HYPOTHESISE\n\n"
        "Based on your understanding from Turn 1, list potential issues.\n"
        "For EACH hypothesis:\n"
        "- State the concern precisely (what could go wrong, under what conditions)\n"
        "- State what code evidence would CONFIRM it as a real bug\n"
        "- State what code evidence would REFUTE it (i.e. an existing guard)\n"
        "- Rate your initial confidence: HIGH / MEDIUM / LOW\n\n"
        "Use the cross-file context provided — if a guard exists in a related file, "
        "say so and lower your confidence accordingly.\n\n"
        "List 3–8 hypotheses maximum. Quality over quantity."
    )

    if verbose:
        print("  [deep] Turn 2: Forming hypotheses...", file=sys.stderr)
    _increment_call_counter()
    msgs.append({"role": "user", "content": t2_prompt})
    t2_response = _call_api(msgs, api_key, max_tokens=4096, timeout=180.0)
    msgs.append({"role": "assistant", "content": t2_response})

    # ── Turn 3: Verify + Verdict ─────────────────────────────────────
    t3_prompt = (
        "## TURN 3 — VERIFY + VERDICT\n\n"
        "For EACH hypothesis from Turn 2:\n"
        "1. Quote the EXACT code line(s) that confirm or refute it\n"
        "2. Mark as: **TRUE** (real bug) | **FALSE POSITIVE** | **WORTH EXPLORING** (not a bug, but an improvement)\n"
        "3. If TRUE: describe the minimal fix\n"
        "4. If FALSE POSITIVE: explain what existing guard makes it safe\n\n"
        "Then write a FINAL REPORT:\n"
        "- Only TRUE findings with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "- WORTH EXPLORING list (improvements, not bugs)\n"
        "- Summary line: 'X TRUE findings, Y false positives, Z improvements'\n\n"
        "HARD RULE: If you cannot quote specific code evidence for a TRUE finding, "
        "downgrade it to FALSE POSITIVE. Unverified claims are not findings."
    )

    if verbose:
        print("  [deep] Turn 3: Verifying + final verdict...", file=sys.stderr)
    _increment_call_counter()
    msgs.append({"role": "user", "content": t3_prompt})
    t3_response = _call_api(msgs, api_key, max_tokens=8192, timeout=300.0)

    # Return only Turn 3 (the verified verdict) — that's all the caller needs
    return t3_response


def audit_plan(
    feature_description: str,
    context_paths: list[str],
    api_key: str,
    *,
    verbose: bool = False,
    enrich: bool = False,
) -> str:
    """4-turn implementation planning session for a new feature.

    Given a feature description and relevant existing files, produces a
    concrete implementation plan. Mirrors how a senior engineer approaches
    a new feature: orient → design → detail → validate.

      Turn 1 — ORIENT: Understand the current codebase relevant to this
               feature. What exists, what's close, what's missing.

      Turn 2 — DESIGN: High-level design — data model, interfaces, data
               flow, integration points. No code yet.

      Turn 3 — DETAIL: Concrete implementation steps — specific files,
               functions, line-level changes. Enough to execute without
               ambiguity.

      Turn 4 — VALIDATE: Risks, guardrails, tests needed, drift checks.
               What would silently break if done wrong.

    Args:
        feature_description: Plain-English description of the feature to build.
        context_paths: Existing files relevant to the feature (auto-extended
                       with detected related files).
        api_key: MiniMax API key.
        verbose: Print turn progress to stderr.

    Uses ~4 API calls.
    """
    project_root = Path(__file__).parent.parent.parent

    # Build context from provided files + auto-detected related files
    all_paths = list(context_paths)
    related = find_related_files(context_paths, max_extra=3)
    if verbose and related:
        print(f"  Auto-context: {list(related)!r}", file=sys.stderr)

    content_parts = []
    for p in all_paths:
        path = Path(p)
        if not path.exists():
            path = project_root / p
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if len(lines) > 1500:
            content = "\n".join(lines[:1500]) + f"\n\n... [TRUNCATED at 1500/{len(lines)} lines]"
        content_parts.append(f"### EXISTING FILE: {p}\n```python\n{content}\n```")

    for rel, content in related.items():
        if rel not in all_paths:
            content_parts.append(f"### RELATED FILE: {rel}\n```python\n{content}\n```")

    codebase_block = "\n\n".join(content_parts) if content_parts else "(no files provided)"

    # Enrich with runtime context if requested
    if enrich:
        if verbose:
            print("  Gathering runtime context (DB, config, tests, git)...", file=sys.stderr)
        runtime_ctx = gather_runtime_context(all_paths)
        if runtime_ctx:
            codebase_block = codebase_block + "\n\n" + runtime_ctx

    system = (
        f"{ARCHITECTURE_CONTEXT}\n---\n\n"
        "You are designing a new feature for this production trading pipeline. "
        "You MUST respect existing architecture conventions exactly as described above. "
        "You MUST NOT suggest changes that violate the one-way dependency, "
        "hardcode canonical values, or break idempotency."
    )

    feature_block = f"## Feature Request\n\n{feature_description}"

    # ── Turn 1: Orient ───────────────────────────────────────────────
    t1_prompt = (
        "## TURN 1 — ORIENT\n\n"
        f"{feature_block}\n\n"
        "Read the existing code below. Describe:\n"
        "1. What already exists that is relevant to this feature\n"
        "2. What functions/modules/data structures are closest to what's needed\n"
        "3. What is genuinely missing (the gap this feature fills)\n"
        "4. Any constraints the existing architecture imposes on how this can be built\n\n"
        "Do NOT design anything yet. Just orient yourself.\n\n" + codebase_block
    )

    if verbose:
        print("  [plan] Turn 1: Orienting in codebase...", file=sys.stderr)
    _increment_call_counter()
    msgs: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": t1_prompt},
    ]
    t1 = _call_api(msgs, api_key, max_tokens=3000, timeout=180.0)
    msgs.append({"role": "assistant", "content": t1})

    # ── Turn 2: Design ───────────────────────────────────────────────
    t2_prompt = (
        "## TURN 2 — DESIGN\n\n"
        "Based on your orientation, design the high-level architecture:\n\n"
        "1. **Data model**: What new columns, tables, or fields are needed? "
        "What type, what units, what nullable?\n"
        "2. **Interfaces**: Function signatures with full type annotations. "
        "Where do they live (which module)?\n"
        "3. **Data flow**: Step by step — where does data enter, what transforms "
        "happen, where does it go?\n"
        "4. **Integration**: Which existing functions call or are called by your new code?\n\n"
        "No implementation code yet — just design. Be specific about types and names."
    )

    if verbose:
        print("  [plan] Turn 2: Designing architecture...", file=sys.stderr)
    _increment_call_counter()
    msgs.append({"role": "user", "content": t2_prompt})
    t2 = _call_api(msgs, api_key, max_tokens=4000, timeout=180.0)
    msgs.append({"role": "assistant", "content": t2})

    # ── Turn 3: Detail ───────────────────────────────────────────────
    t3_prompt = (
        "## TURN 3 — IMPLEMENTATION STEPS\n\n"
        "Write the concrete implementation plan — ordered steps a developer "
        "can execute exactly as written:\n\n"
        "For each step:\n"
        "- **File**: exact path\n"
        "- **Where**: function name or line reference\n"
        "- **What**: the exact change (new function body, new column in SQL, "
        "new parameter, etc.)\n"
        "- **Why**: one sentence on why this step is needed\n\n"
        "Include the actual code/SQL for non-trivial steps. "
        "Steps should be atomic — each independently committable. "
        "Order them so earlier steps don't depend on later ones."
    )

    if verbose:
        print("  [plan] Turn 3: Detailing implementation steps...", file=sys.stderr)
    _increment_call_counter()
    msgs.append({"role": "user", "content": t3_prompt})
    t3 = _call_api(msgs, api_key, max_tokens=6000, timeout=300.0)
    msgs.append({"role": "assistant", "content": t3})

    # ── Turn 4: Validate ─────────────────────────────────────────────
    t4_prompt = (
        "## TURN 4 — RISKS AND GUARDRAILS\n\n"
        "Review your implementation plan for things that could go wrong:\n\n"
        "1. **Silent failure modes**: What would produce wrong results without "
        "raising an error? (join traps, timezone bugs, off-by-one date ranges, "
        "look-ahead data leaks)\n"
        "2. **Tests needed**: Specific pytest test cases with their assertion logic\n"
        "3. **Drift checks**: Should a new check be added to `pipeline/check_drift.py`? "
        "If yes, what does it assert?\n"
        "4. **Rebuild requirements**: Does this change require re-running "
        "outcome_builder, strategy_validator, or build_edge_families? In what order?\n"
        "5. **Rollback plan**: If this causes issues in production, how do you undo it?\n\n"
        "Then write a **FINAL PLAN SUMMARY** — a single block a developer can "
        "paste into a task tracker:\n"
        "```\n"
        "Feature: <name>\n"
        "Files: <list>\n"
        "Steps: <count>\n"
        "Estimated effort: S/M/L\n"
        "Requires rebuild: yes/no (which stages)\n"
        "Key risk: <one sentence>\n"
        "```"
    )

    if verbose:
        print("  [plan] Turn 4: Risk assessment + final summary...", file=sys.stderr)
    _increment_call_counter()
    msgs.append({"role": "user", "content": t4_prompt})
    t4 = _call_api(msgs, api_key, max_tokens=5000, timeout=300.0)

    # Assemble full plan document
    sections = [
        f"# Implementation Plan: {feature_description[:80]}\n",
        "---\n## Turn 1 — Current State\n",
        t1,
        "\n---\n## Turn 2 — Design\n",
        t2,
        "\n---\n## Turn 3 — Implementation Steps\n",
        t3,
        "\n---\n## Turn 4 — Risks & Guardrails\n",
        t4,
    ]
    return "\n".join(sections)


def load_api_key() -> str:
    """Load MiniMax API key from env or .env file."""
    load_dotenv()
    key = os.environ.get("MINIMAX_API_KEY", "")
    if not key:
        print("ERROR: MINIMAX_API_KEY not set.", file=sys.stderr)
        print("  Set it in your .env file or run:", file=sys.stderr)
        print("    export MINIMAX_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)
    return key


def read_files(paths: list[str]) -> str:
    """Read and concatenate files with headers."""
    parts = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            # Try relative to project root
            path = Path(__file__).parent.parent.parent / p
        if not path.exists():
            print(f"WARNING: File not found: {p}", file=sys.stderr)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files
        lines = content.splitlines()
        if len(lines) > 2000:
            content = "\n".join(lines[:2000])
            content += f"\n\n... [TRUNCATED — {len(lines)} total lines, showing first 2000]"
        parts.append(f"### File: {p}\n```python\n{content}\n```")
    if not parts:
        print("ERROR: No readable files provided.", file=sys.stderr)
        sys.exit(1)
    return "\n\n".join(parts)


def audit(
    file_content: str,
    system_prompt: str,
    api_key: str,
    user_prompt: str | None = None,
    *,
    include_context: bool = True,
    fast: bool = False,
    timeout: float | None = None,
) -> str:
    """Send code to MiniMax M2.5 for review.

    Args:
        file_content: Code to audit (concatenated file contents).
        system_prompt: The system prompt / audit mode instruction.
        api_key: MiniMax API key.
        user_prompt: Optional additional user instruction prepended to file content.
        include_context: Prepend ARCHITECTURE_CONTEXT to system prompt (default True).
            Set False for custom prompts that provide their own context.
        fast: Use MiniMax-M2.5-Lightning (3-5x faster, same accuracy). Default False.
        timeout: API timeout in seconds. Defaults to model-appropriate value.
    """
    if user_prompt:
        full_prompt = f"{user_prompt}\n\n{file_content}"
    else:
        full_prompt = file_content

    # Prepend architecture context to reduce false positives.
    # Append official M2.5 token-budget hint last (per MiniMax best-practices docs:
    # "M2.5 may terminate tasks early when approaching context capacity thresholds").
    TOKEN_BUDGET_HINT = (
        "\n\nThis is a thorough but bounded task. Keep your total output within the "
        "available context window. Prioritise completeness for high-severity findings; "
        "be concise for LOW findings. Do not truncate mid-review — finish every section."
    )
    full_system = system_prompt
    if include_context:
        full_system = f"{ARCHITECTURE_CONTEXT}\n---\n\n{system_prompt}"
    full_system += TOKEN_BUDGET_HINT

    # Auto-size output tokens: estimate input tokens, leave rest for output
    input_chars = len(full_system) + len(full_prompt)
    input_tokens_est = input_chars // 4  # ~4 chars/token heuristic
    available_output = MAX_CONTEXT - input_tokens_est - 1000  # 1K safety margin
    effective_max_tokens = max(4096, min(MAX_TOKENS, available_output))

    if available_output < MAX_TOKENS:
        print(
            f"  Auto-sized output: {effective_max_tokens:,} tokens "
            f"(input ~{input_tokens_est:,}, context limit {MAX_CONTEXT:,})",
            file=sys.stderr,
        )

    model = MODEL_FAST if fast else MODEL_STANDARD
    effective_timeout = timeout or (API_TIMEOUT_FAST if fast else API_TIMEOUT_STANDARD)

    # Track call budget
    used, remaining = _increment_call_counter()
    if remaining <= 10:
        print(f"  ⚠ Budget warning: {remaining} calls remaining today", file=sys.stderr)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": full_prompt},
        ],
        "max_tokens": effective_max_tokens,
        "temperature": 0.1,  # Low temp for precise analysis
    }

    for attempt in range(2):
        try:
            with httpx.Client(timeout=effective_timeout) as client:
                resp = client.post(
                    API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            break
        except httpx.ReadTimeout:
            if attempt == 0:
                print("  TIMEOUT — retrying with extended timeout...", file=sys.stderr)
                effective_timeout = min(effective_timeout * 2, 900.0)
            else:
                raise

    if resp.status_code != 200:
        print(f"ERROR: API returned {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax M2.5 independent code audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modes: general, bias, joins, bugs, improvements, gaps",
    )
    parser.add_argument("files", nargs="*", help="Files to audit")
    parser.add_argument(
        "--mode",
        choices=list(AUDIT_MODES.keys()),
        default="general",
        help="Preset audit mode (default: general). New: discovery, architect",
    )
    parser.add_argument("--prompt", help="Custom audit prompt (overrides --mode)")
    parser.add_argument("--output", "-o", help="Save output to file")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use MiniMax-M2.5-Lightning (3-5x faster, good for quick scans)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Multi-turn audit: understand → hypothesise → verify. "
            "Auto-injects related files. ~3 API calls. Reduces FP rate ~72%%→30%%."
        ),
    )
    parser.add_argument(
        "--plan",
        metavar="FEATURE",
        help=(
            "4-turn implementation planning mode. Pass a feature description in quotes. "
            "Provide relevant existing files as positional args. "
            "Produces: current state → design → implementation steps → risks. "
            "Example: --plan 'Add trailing stop to E2 entry model' trading_app/entry_rules.py"
        ),
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "Auto-gather runtime context (DB schema, config values, test files, "
            "git history) and inject into the prompt. M2.5 cannot call tools — "
            "this is the workaround. Recommended for deep audits and presets."
        ),
    )
    parser.add_argument(
        "--diff",
        metavar="REF",
        nargs="?",
        const="HEAD",
        help=(
            "Diff-only mode: show only changed lines + 15 lines context from "
            "git diff REF (default HEAD). M2.5 is strongest reviewing small "
            "focused changes, not whole files. Use with --enrich for full context."
        ),
    )
    parser.add_argument(
        "--triage",
        action="store_true",
        help=(
            "Post-process output: extract only TRUE findings and WORTH EXPLORING, "
            "strip FALSE POSITIVEs. Reduces noise from M2.5's ~41%% FP rate."
        ),
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        help="Show today's call usage and exit",
    )
    args = parser.parse_args()

    if args.budget:
        show_budget()
        return

    api_key = load_api_key()

    # ── Plan mode ────────────────────────────────────────────────────
    if args.plan:
        context_files = args.files or []
        print(
            f"Planning: '{args.plan[:60]}' with {len(context_files)} context file(s)...",
            file=sys.stderr,
        )
        result = audit_plan(args.plan, context_files, api_key, verbose=True, enrich=args.enrich)

    # ── Deep audit mode ──────────────────────────────────────────────
    elif args.deep:
        if not args.files:
            parser.error("--deep requires at least one file")
        mode = args.mode if not args.prompt else "general"
        print(
            f"Deep audit: {len(args.files)} file(s) [{mode}] — 3-turn + auto-context...",
            file=sys.stderr,
        )
        result = audit_deep(args.files, mode, api_key, verbose=True, enrich=args.enrich)

    # ── Standard single-turn mode ────────────────────────────────────
    else:
        if not args.files:
            parser.error("at least one file is required (or use --budget / --plan)")

        # Diff mode: send only changed hunks + context instead of whole files
        if args.diff is not None:
            diff_content = build_diff_content(args.files, ref=args.diff)
            if diff_content:
                file_content = diff_content
                # Still include full file for reference but after the diff
                file_content += "\n\n## FULL FILE REFERENCE (for context only)\n"
                file_content += read_files(args.files)
                print(
                    f"  Diff mode: reviewing changes vs {args.diff}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  No diff found vs {args.diff}, falling back to full file",
                    file=sys.stderr,
                )
                file_content = read_files(args.files)
        else:
            file_content = read_files(args.files)

        if args.enrich:
            print("  Gathering runtime context (DB, config, tests, git)...", file=sys.stderr)
            runtime_ctx = gather_runtime_context(args.files)
            if runtime_ctx:
                file_content = file_content + "\n\n" + runtime_ctx
        system_prompt = args.prompt if args.prompt else AUDIT_MODES[args.mode]
        print(
            f"Auditing {len(args.files)} file(s) [M2.5/{args.mode}]...",
            file=sys.stderr,
        )
        result = audit(file_content, system_prompt, api_key, fast=args.fast)

    # Post-process: triage to strip false positives
    if args.triage:
        result = triage_output(result)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        sys.stdout.buffer.write((result + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
