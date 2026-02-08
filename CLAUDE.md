# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Gold (MGC) Data Pipeline — builds a clean, replayable local dataset for Micro Gold futures (MGC) trading research and backtesting from Databento DBN files.

**CRITICAL: Price Data Source**
Raw data files contain GC (full-size Gold futures) which has ~40-70% more 1-minute bars than MGC. The pipeline ingests **GC bars** for accurate ORB construction, stores them under `symbol='MGC'` (prices are identical — same underlying, same exchange), and uses the **MGC cost model** ($10/point, $8.40 RT friction) for all trading math. The `source_symbol` column records the actual GC contract used (e.g., GCJ1, GCG5).

**Target ORBs**: 09:00, 10:00, 11:00 (primary), 18:00, 23:00, 00:30 (secondary)

---

## What Exists (Current State)

### Pipeline Code (`pipeline/`)

| File | LOC | Purpose |
|------|-----|---------|
| `ingest_dbn_mgc.py` | 924 | MGC-specific DBN ingestion with 7 validation gates |
| `ingest_dbn.py` | 525 | Generic multi-instrument wrapper |
| `build_bars_5m.py` | 352 | Deterministic 5m bar aggregation from 1m bars |
| `run_pipeline.py` | 264 | Pipeline orchestrator (ingest → 5m → features → audit) |
| `check_drift.py` | 220 | Static analysis drift detector (7 checks) |
| `init_db.py` | 132 | Database schema initialization |
| `asset_configs.py` | 111 | Per-instrument config (MGC, MNQ, NQ) |
| `check_db.py` | 84 | Database inspection tool |
| `dashboard.py` | 450 | Self-contained HTML report generator (7 panels) |
| `health_check.py` | 100 | Quick all-in-one health check CLI |
| `build_daily_features.py` | 480 | Daily features builder (ORBs, sessions, RSI, outcomes) |
| `cost_model.py` | 120 | Canonical cost model (MGC friction, R-multiples, stress test) |
| `paths.py` | 21 | Canonical path constants |

### Root Scripts

| File | Purpose |
|------|---------|
| `run_backfill_overnight.py` | Crash-recovery retry wrapper |

### Reference Documents

| File | Purpose |
|------|---------|
| `CANONICAL_LOGIC.txt` | Trading logic specification (R-multiples, entry rules, cost model) |
| `CANONICAL_backfill_dbn_mgc_rules.txt` | Ingestion rules (fail-closed, chunked, checkpointed) |
| `CANONICAL_backfill_dbn_mgc_rules_addon.txt` | Advanced ingestion patterns |

### Data Files (gitignored)

- `DB/GOLD_DB_FULLSIZE/` — 1,555 daily `.dbn.zst` files (2021-02-05 to 2026-02-04)
- `DB/gold_db_fullsize_2016-2021/` — 1,557 daily `.dbn.zst` files (2016-02-01 to 2021-02-03)
- `gold.db` — DuckDB database (created by `init_db.py`, ~10 years of data)

---

## Architecture

### Data Flow

```
Databento DBN files (.dbn.zst)
  → pipeline/ingest_dbn_mgc.py (validate, filter outrights, choose front contract)
  → gold.db:bars_1m (1-minute OHLCV, UTC timestamps)
  → pipeline/build_bars_5m.py (deterministic aggregation)
  → gold.db:bars_5m (5-minute OHLCV, fully rebuildable)
  → pipeline/build_daily_features.py (ORBs, sessions, RSI, outcomes)
  → gold.db:daily_features (one row per trading day)
```

### Database Schema (DuckDB)

**bars_1m** (primary raw data):
- Columns: `ts_utc`, `symbol`, `source_symbol`, `open`, `high`, `low`, `close`, `volume`
- Primary key: `(symbol, ts_utc)`
- `symbol`: 'MGC' (continuous logical symbol)
- `source_symbol`: actual contract (e.g., 'MGCG4', 'MGCM4')

**bars_5m** (derived):
- Same columns as bars_1m
- Deterministically aggregated from bars_1m
- Bucket = floor(epoch(ts)/300)*300
- Fully rebuildable at any time

**daily_features** (one row per trading day per instrument):
- Primary key: `(symbol, trading_day)`
- `trading_day`: DATE (09:00 Brisbane boundary)
- `bar_count_1m`: INTEGER (bars in trading day)
- Session stats: `session_{asia,london,ny}_{high,low}` (DOUBLE)
- RSI: `rsi_14_at_0900` (DOUBLE, Wilder's 14-period on 5m closes)
- 6 ORBs x 8 columns each: `orb_{0900,1000,1100,1800,2300,0030}_{high,low,size,break_dir,break_ts,outcome,mae_r,mfe_r}`
- Built by `pipeline/build_daily_features.py` (idempotent, configurable --orb-minutes)

### Time & Calendar Model (CRITICAL)

- Local timezone: `Australia/Brisbane` (UTC+10, no DST)
- Trading day: **09:00 local → next 09:00 local**
- Bars before 09:00 local assigned to PREVIOUS trading day
- All timestamps in database are UTC (`TIMESTAMPTZ`)
- Expected 1-minute count per full weekday: ~1440 rows

### Futures Contract Handling

- Raw data contains GC (full-size Gold) and MGC (Micro Gold) bars
- **Pipeline uses GC outrights** for price data (better bar coverage for accurate ORBs)
- Databento returns real contracts (GCG4, GCM4, etc.)
- Pipeline selects front/most liquid GC contract per day (highest volume, excludes spreads)
- Deterministic tiebreak: earliest expiry → lexicographic smallest
- Stores under `symbol='MGC'` with `source_symbol=actual GC contract`
- Cost model uses MGC specs ($10/point) — you trade MGC, GC is just the price source

---

## Key Commands

### Database Operations

```bash
python pipeline/init_db.py              # Create schema (bars_1m, bars_5m)
python pipeline/init_db.py --force      # Drop and recreate (WARNING: destroys data)
python pipeline/check_db.py             # Inspect database contents
```

### Ingestion

```bash
# Ingest MGC DBN into bars_1m
python pipeline/ingest_dbn_mgc.py --start 2024-01-01 --end 2024-12-31
python pipeline/ingest_dbn_mgc.py --resume          # Resume from checkpoint
python pipeline/ingest_dbn_mgc.py --dry-run          # Validate only

# Full pipeline (ingest → 5m → audit)
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31
```

### 5-Minute Bar Aggregation

```bash
python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --dry-run
```

### Daily Features

```bash
python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --orb-minutes 15
python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --dry-run
```

### Guardrails

```bash
python pipeline/check_drift.py                        # Drift detection (19 checks)
python -m pytest tests/ -v                             # Full test suite (~500 tests)
python -m pytest tests/ -x -q                          # Fast test run (stop on first fail)
python pipeline/health_check.py                        # All-in-one health check
```

### Dashboard

```bash
python pipeline/dashboard.py                           # Generate dashboard.html
python pipeline/dashboard.py --output report.html      # Custom output path
```

### Testing

```bash
python -m pytest tests/ -v                             # All tests
python -m pytest tests/test_validation.py -v           # Specific test file
python -m pytest tests/ -x -q --timeout=30             # Quick run with timeout
```

---

## Guardrails (Automatic)

### 1. Git Pre-Commit Hook (`.githooks/pre-commit`)
Runs automatically before every commit:
- Drift check (`check_drift.py`)
- Fast pure-function tests
- Syntax validation on changed files
Setup: `git config core.hooksPath .githooks`

### 2. Drift Detection (`pipeline/check_drift.py`)
Static analysis that catches:
1. Hardcoded 'MGC' SQL in generic pipeline code
2. `.apply()`/`.iterrows()` on large data (performance anti-pattern)
3. Writes to non-bars_1m tables in ingest scripts
4. Schema-query table name mismatches
5. Import cycles between pipeline modules
6. Hardcoded absolute Windows paths
7. DuckDB connection leaks (missing close/finally/atexit)

### 3. Claude Code Hooks
Auto-run checks when editing pipeline files:
- Pipeline edit → drift check
- Schema edit → schema test
- Any .py edit → test suite

### 4. GitHub Actions CI (`.github/workflows/ci.yml`)
On push/PR to main:
- Drift check
- Pure-function tests
- Schema + aggregation tests

---

## Validation Gates (Built Into Pipeline)

The pipeline uses **fail-closed** design throughout. Any validation failure aborts immediately.

### Ingestion Gates (ingest_dbn_mgc.py):
1. DBN schema must be `ohlcv-1m`
2. Timestamps must be UTC (proven, not assumed)
3. Outright filter (exclude spreads)
4. OHLCV validation (NaN, infinite, non-positive, high/low consistency)
5. PK safety (no duplicate ts_utc per trading day)
6. Merge integrity (no dupes/NULLs after INSERT)
7. Final honesty gates (global type/dupe/NULL checks)

### 5m Aggregation Gates (build_bars_5m.py):
1. No duplicate (symbol, ts_utc)
2. All timestamps 5-minute aligned
3. OHLCV sanity (high >= low, etc.)
4. Volume non-negative

### Checkpoint System:
- JSONL append-only (never edits/deletes records)
- Supports resume (`--resume`) and retry (`--retry-failed`)
- Keyed by date range + source file identity

---

## Idempotency & Resume

All operations are safe to re-run:
- Ingestion: `INSERT OR REPLACE` on primary key
- 5m aggregation: DELETE then INSERT for date range
- No duplicate rows possible

---

## Configuration (.env)

Required (for Databento backfills):
- `DATABENTO_API_KEY`

Defaults (override if needed):
- `DUCKDB_PATH`: `gold.db`
- `SYMBOL`: `MGC`
- `TZ_LOCAL`: `Australia/Brisbane`

---

## Important Notes

1. **Data files**: Two directories of daily `.dbn.zst` files covering 10 years:
   - `DB/GOLD_DB_FULLSIZE/` — 1,555 files (2021-02-05 to 2026-02-04)
   - `DB/gold_db_fullsize_2016-2021/` — 1,557 files (2016-02-01 to 2021-02-03)
   Pipeline currently expects a single concatenated file — `ingest_dbn_daily.py` handles individual daily files. Use `--data-dir` to point at either directory.

2. **Contract selection**: Uses GC (full-size Gold) outrights for price data — better 1m bar coverage than MGC. Automatically handles futures rolls by selecting most liquid GC contract per day (highest volume, excluding spreads). Prices are identical to MGC (same underlying).

3. **5-minute bars**: Always rebuilt from 1-minute bars. Never manually edit bars_5m.

4. **Weekend/holiday handling**: Missing data stored as NULL. Scripts will not crash on days without data.

5. **Minimum start date**: Dataset starts 2016-02-01. GC has full bar coverage from this date. Two data directories cover 2016-2021 and 2021-2026 with zero overlap.

6. **Schema migration**: If you have old data, wipe and rebuild:
   ```bash
   python pipeline/init_db.py --force
   # Then re-run ingestion
   ```

---

## Strategy Classification Rules (FIX5 — MANDATORY)

### Trade Day Invariant
A valid trade day requires BOTH:
1. A break occurred (outcome exists in `orb_outcomes`)
2. The strategy's `filter_type` makes the day eligible (per `daily_features`)

`orb_outcomes` contains ALL break-days regardless of filter. The portfolio overlay
MUST only write `pnl_r` on eligible days (`series == 0.0`). Low trade counts under
strict filters (G6/G8) are EXPECTED behavior, not bugs.

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
6. G6/G8 filters are volatility regime detectors — evaluate for conditional uplift, drawdown reduction, crisis alpha

### Where Edge Lives
- Edge requires G4+ ORB size filter (NO_FILTER and L-filters have negative ExpR)
- E1 for momentum sessions (0900/1000), E3 for retrace sessions (1800/2300)
- ORB size is THE edge: <4pt = house wins, 4-10pt = breakeven+, >10pt = strong
- 2021 is structurally different (tiny ORBs) — exclude from validation

---

## What's NOT Built Yet

See `ROADMAP.md` for planned features including:
- Phase 6e: Monitoring & alerting (live vs backtest drift detection)

**Do NOT reference unbuilt features in code or tests. Build guardrails for what exists.**
