# CLAUDE.md

Guidance for Claude Code working with this repository.

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Supports MGC (Micro Gold), MNQ (Micro Nasdaq), MCL (Micro Crude), MES (Micro S&P 500).

**CRITICAL: Price Data Source (MGC)**
Raw data contains GC (full-size Gold) which has better 1m bar coverage than MGC. Pipeline ingests GC bars, stores under `symbol='MGC'` (same price, same exchange), uses MGC cost model. The `source_symbol` column records the actual GC contract.

**For instruments, cost models, sessions, entry models, and all trading logic → see `TRADING_RULES.md`.**
**For research methodology, statistical standards, and market structure knowledge → see `RESEARCH_RULES.md`.**

---

## Document Authority

| Document | Scope | Conflict Rule |
|----------|-------|---------------|
| `CLAUDE.md` | Code structure, commands, guardrails, AI behavior | Wins for code decisions |
| `TRADING_RULES.md` | Trading rules, sessions, filters, research findings, NO-GOs | Wins for trading logic |
| `RESEARCH_RULES.md` | Research methodology, statistical standards, trading lens, market structure | Wins for research/analysis decisions |
| `ROADMAP.md` | Planned features, phase status | Updated on phase completion |
| `REPO_MAP.md` | Module index, file inventory | Auto-generated (`python scripts/tools/gen_repo_map.py`) — never hand-edit |
| `docs/STRATEGY_DISCOVERY_AUDIT.md` | Strategy discovery system deep-dive | Reference only |
| `docs/RESEARCH_ARCHIVE.md` | Research findings, NO-GO archive, alternative strategy results | Supplements TRADING_RULES.md |
| `CANONICAL_*.txt` | Frozen specs | Read-only; live code is truth |

**Conflict resolution:**
- Code behavior → CLAUDE.md wins
- Trading logic → TRADING_RULES.md wins
- Research methodology → RESEARCH_RULES.md wins
- File inventory → REPO_MAP.md (not this file)
- CANONICAL_*.txt are frozen; the live code is truth

---

## Architecture

### Data Flow

```
Databento .dbn.zst files
  → pipeline/ingest_dbn.py (validate, select front contract)
  → gold.db:bars_1m (1-minute OHLCV, UTC timestamps)
  → pipeline/build_bars_5m.py (deterministic 5m aggregation)
  → gold.db:bars_5m
  → pipeline/build_daily_features.py (ORBs, sessions, RSI, ATR)
  → gold.db:daily_features

  → trading_app/outcome_builder.py (pre-compute trade outcomes)
  → gold.db:orb_outcomes
  → trading_app/strategy_discovery.py (grid search)
  → gold.db:experimental_strategies
  → trading_app/strategy_validator.py (multi-phase validation + walk-forward)
  → gold.db:validated_setups
  → scripts/tools/build_edge_families.py (cluster by trade-day hash)
  → gold.db:edge_families
```

### Key Design Principles
- **Fail-closed:** Any validation failure aborts immediately
- **Idempotent:** All operations safe to re-run (INSERT OR REPLACE / DELETE+INSERT)
- **Pre-computed outcomes:** 689K rows computed once, reused for all discovery
- **One-way dependency:** pipeline/ → trading_app/ (never reversed)

### Time & Calendar Model
- Local timezone: `Australia/Brisbane` (UTC+10, no DST)
- Trading day: 09:00 local → next 09:00 local
- Bars before 09:00 assigned to PREVIOUS trading day
- All DB timestamps are UTC (`TIMESTAMPTZ`)

### CRITICAL: DST Contamination in Fixed Sessions (Feb 2026 Finding)

**Problem:** Four fixed sessions (0900, 1800, 0030, 2300) have their relationship to market events change with DST. Three (0900/1800/0030) align with their event in winter but miss by 1 hour in summer. 2300 is a special case — it NEVER aligns with the US data release but sits on opposite sides of it depending on DST. Every metric computed on these sessions (avgR, Sharpe, WR, totR) is a blended average of two different market contexts.

**Which sessions are affected:**

| Session | Winter (std time) | Summer (DST) | Shift source |
|---------|------------------|--------------|-------------|
| 0900 | = CME open (5PM CST = 23:00 UTC = 09:00 Bris) | CME opened at 0800 Bris (5PM CDT = 22:00 UTC) | US DST |
| 1800 | = London open (8AM GMT = 08:00 UTC = 18:00 Bris) | London opened at 1700 Bris (8AM BST = 07:00 UTC) | UK DST |
| 0030 | = US equity open (9:30AM EST = 14:30 UTC = 00:30 Bris) | US equity opened at 2330 Bris (9:30AM EDT = 13:30 UTC) | US DST |
| 2300 | 30min BEFORE US data (8:30 EST = 13:30 UTC; 2300 = 13:00 UTC) | 30min AFTER US data (8:30 EDT = 12:30 UTC; 2300 = 13:00 UTC) | US DST |

**2300 NOTE:** Unlike 0900/1800/0030 which align with their event in winter, 2300 Brisbane (13:00 UTC) NEVER catches the US data release (8:30 ET). It's always 30min off — but DST flips which side: pre-data in winter, post-data in summer. Volume data confirms: summer has 76-90% MORE volume (data already released, market reacting). The winter/summer split is still meaningful and the `"US"` classification in `dst.py` is correct.

**Which sessions are CLEAN (no DST issue):**
- 1000 — Tokyo open. Japan has NO DST. Always aligned.
- 1100 — Singapore open. No DST. Always aligned.
- 1130 — HK/Shanghai open. No DST. Always aligned.
- All dynamic sessions (CME_OPEN, LONDON_OPEN, US_EQUITY_OPEN, US_DATA_OPEN, US_POST_EQUITY, CME_CLOSE) — resolvers adjust per-day.

**What's contaminated:**
- `daily_features` columns for 0900/1800/0030/2300 ORBs
- `orb_outcomes` rows for those sessions
- `experimental_strategies` and `validated_setups` for those sessions
- `edge_families` containing those sessions
- ALL research findings about 0900/1800 in TRADING_RULES.md
- Hypothesis tests (H1-H5) that used those sessions
- Cross-instrument portfolio analysis at those sessions

**What's NOT contaminated:**
- Everything at 1000, 1100, 1130 (Asia sessions — no DST anywhere)
- Dynamic session results (CME_OPEN, LONDON_OPEN, etc.)
- Pipeline data integrity (bars_1m, bars_5m are raw UTC data — correct)
- The ORB computation itself is correct — it computes the ORB at the stated clock time. The issue is that the stated clock time maps to different market events depending on DST.

**DST remediation status (Feb 2026 — DONE):**
- ✅ Validator split: DST columns on both strategy tables, auto-migrated by `init_trading_app_schema()`.
- ✅ Revalidation: 1272 strategies — 275 STABLE, 155 WINTER-DOM, 130 SUMMER-DOM. No validated broken. CSV: `research/output/dst_strategy_revalidation.csv`.
- ✅ Volume analysis: event-driven edges confirmed. `research/output/volume_dst_findings.md`.
- ✅ Time scan: `research/research_orb_time_scan.py`. New candidates all rejected.
- ✅ DST columns live in production gold.db (942 validated_setups, 464 with DST splits; 12,996 experimental, 2,304 with DST splits).

**937 validated strategies exist** across all instruments. 2300: 4 (MGC G8+). 0030: 44 (MES 31, MNQ 13). Do NOT deprecate.

**Rule for all future research:** ANY analysis touching sessions 0900/1800/0030/2300 MUST split by DST regime (US for 0900/0030/2300; UK for 1800) and report both halves. Blended numbers are misleading.

---

## Database Location & Workflow (CRITICAL)

**ONE database** (`gold.db`) at `<project>/gold.db` — local disk, no cloud sync.

For long-running jobs, you can optionally use `C:\db\gold.db` as a scratch copy:
```bash
# Optional: copy to scratch location for crash safety
cp "C:\Users\joshd\canompx3\gold.db" "C:\db\gold.db"
export DUCKDB_PATH=C:/db/gold.db
python trading_app/strategy_discovery.py --instrument MGC
# Copy back when done
cp "C:\db\gold.db" "C:\Users\joshd\canompx3\gold.db"
```

**Rules:**
- NEVER run two write processes against the same DuckDB file simultaneously
- `pipeline/paths.py` reads `DUCKDB_PATH` env var to override default path

---

## Key Commands

```bash
# Database
python pipeline/init_db.py                    # Create schema
python pipeline/init_db.py --force            # Drop + recreate (DESTROYS DATA)

# Ingestion
python pipeline/ingest_dbn.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/ingest_dbn.py --instrument MGC --resume
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python scripts/infra/run_parallel_ingest.py --instrument MGC --db-path C:/db/gold.db

# Features
python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-12-31

# Trading App
python trading_app/outcome_builder.py --instrument MGC --start 2021-02-05 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_validator.py --instrument MGC --min-sample 50
python trading_app/paper_trader.py --instrument MGC --start 2025-01-01 --end 2025-12-31
python -m trading_app.live_config --db-path C:/db/gold.db

# Guardrails
python pipeline/check_drift.py               # Drift detection
python -m pytest tests/ -v                    # Full test suite
python -m pytest tests/ -x -q                # Stop on first failure
python pipeline/health_check.py              # All-in-one health check

# Reports
python pipeline/dashboard.py                 # Generate dashboard.html
python scripts/reports/report_edge_portfolio.py      # Edge family portfolio report
```

---

## Guardrails

### 1. Pre-Commit Hook (`.githooks/pre-commit`)
Runs drift check + fast tests before every commit.
Setup: `git config core.hooksPath .githooks`

### 2. Drift Detection (`pipeline/check_drift.py`)
Static analysis catching: hardcoded symbols, performance anti-patterns, import direction violations, connection leaks, schema mismatches, timezone hygiene, analytical honesty, CLAUDE.md size cap. Run `python pipeline/check_drift.py` to see current check count and results.

### 3. Claude Code Hooks
- Pipeline file edit → drift check
- Schema edit → schema test
- Any .py edit → test suite

### 4. GitHub Actions CI (`.github/workflows/ci.yml`)
On push/PR: drift check + pure-function tests + schema tests.

### 5. Validation Gates (Built Into Pipeline)
- **Ingestion:** 7 gates (DBN schema, UTC proof, outright filter, OHLCV sanity, PK safety, merge integrity, honesty gates)
- **5m Aggregation:** 4 gates (no dupes, alignment, OHLCV sanity, volume non-negative)
- **Checkpoint:** JSONL append-only, supports `--resume` and `--retry-failed`

---

## Configuration (.env)

```
DATABENTO_API_KEY=...         # Required for backfills
DUCKDB_PATH=gold.db           # Override DB location
SYMBOL=MGC                    # Default instrument
TZ_LOCAL=Australia/Brisbane   # Local timezone
```

---

## Strategy Classification Rules (FIX5 — MANDATORY)

### Trade Day Invariant
A valid trade day requires BOTH:
1. A break occurred (outcome exists in `orb_outcomes`)
2. The strategy's `filter_type` makes the day eligible (per `daily_features`)

`orb_outcomes` contains ALL break-days regardless of filter. Portfolio overlay MUST only write `pnl_r` on eligible days. Low trade counts under strict filters (G6/G8) are EXPECTED behavior, not bugs.

### Classification Thresholds (from `config.py`)
| Class | Min Samples | Usage |
|-------|------------|-------|
| **CORE** | >= 100 | Standalone portfolio weight |
| **REGIME** | 30-99 | Conditional overlay / signal only |
| **INVALID** | < 30 | Not tradeable |

### Behavioral Rules
1. NEVER treat "low trade count" alone as evidence of a bug
2. ALWAYS verify `trade_days <= eligible_days` before investigating
3. If `trade_days > eligible_days` → assume corruption until proven otherwise
4. Do NOT suggest "fixing" filters to increase sample size
5. NEVER recommend REGIME strategies as standalone trading systems
6. For trading logic (filters, entry models, edge zones, Sharpe formulas) → see `TRADING_RULES.md`

---

## What's NOT Built Yet

See `ROADMAP.md` for current phase status and planned features.

**Do NOT reference unbuilt features in code or tests. Build guardrails for what exists.**
