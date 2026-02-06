# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Gold (MGC) Data Pipeline — builds a clean, replayable local dataset of Micro Gold futures (MGC) from Databento DBN files for ORB-based trading research and backtesting.

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

- `DB/GOLD_DB_FULLSIZE/` — 1,559 daily `.dbn.zst` files (2021-02-05 to 2026-02-04)
- `gold.db` — DuckDB database (created by `init_db.py`)

---

## Architecture

### Data Flow

```
Databento DBN files (.dbn.zst)
  → pipeline/ingest_dbn_mgc.py (validate, filter outrights, choose front contract)
  → gold.db:bars_1m (1-minute OHLCV, UTC timestamps)
  → pipeline/build_bars_5m.py (deterministic aggregation)
  → gold.db:bars_5m (5-minute OHLCV, fully rebuildable)
  → [NOT YET BUILT: daily_features, ORB calculations]
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

### Time & Calendar Model (CRITICAL)

- Local timezone: `Australia/Brisbane` (UTC+10, no DST)
- Trading day: **09:00 local → next 09:00 local**
- Bars before 09:00 local assigned to PREVIOUS trading day
- All timestamps in database are UTC (`TIMESTAMPTZ`)
- Expected 1-minute count per full weekday: ~1440 rows

### Futures Contract Handling

- `MGC1!` = continuous front-month symbol (charting/broker convention)
- Databento returns real contracts (MGCG4, MGCM4, etc.)
- Pipeline selects front/most liquid contract per day (highest volume, excludes spreads)
- Deterministic tiebreak: earliest expiry → lexicographic smallest
- Stores under `symbol='MGC'` with `source_symbol=actual contract`

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

### Guardrails

```bash
python pipeline/check_drift.py                        # Drift detection (7 checks)
python -m pytest tests/ -v                             # Full test suite
python -m pytest tests/ -x -q                          # Fast test run (stop on first fail)
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

1. **Data files**: 1,559 daily `.dbn.zst` files in `DB/GOLD_DB_FULLSIZE/`. Pipeline currently expects a single concatenated file — `ingest_dbn_daily.py` handles individual daily files.

2. **Contract selection**: Automatically handles futures rolls by selecting most liquid contract per day (highest volume, excluding spreads with '-' in symbol).

3. **5-minute bars**: Always rebuilt from 1-minute bars. Never manually edit bars_5m.

4. **Weekend/holiday handling**: Missing data stored as NULL. Scripts will not crash on days without data.

5. **Minimum start date**: MGC data before 2019-01-01 has only 5-28% bar coverage and is unusable for ORB calculations. Pipeline enforces this cutoff.

6. **Schema migration**: If you have old data, wipe and rebuild:
   ```bash
   python pipeline/init_db.py --force
   # Then re-run ingestion
   ```

---

## What's NOT Built Yet

See `ROADMAP.md` for planned features including:
- `daily_features` table (ORB calculations, session stats)
- `trading_app/` module (strategy detection, execution)
- `validated_setups` table (production strategies)
- Cost model (`pipeline/cost_model.py`)
- Strategy validation framework

**Do NOT reference these unbuilt features in code or tests. Build guardrails for what exists.**
