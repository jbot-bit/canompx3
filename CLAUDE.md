# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Supports MGC (Micro Gold), MNQ (Micro Nasdaq), MCL (Micro Crude), MES (Micro S&P 500).

**CRITICAL: Price Data Source (MGC)**
Raw data files contain GC (full-size Gold futures) which has ~40-70% more 1-minute bars than MGC. The pipeline ingests **GC bars** for accurate ORB construction, stores them under `symbol='MGC'` (prices are identical — same underlying, same exchange), and uses the **MGC cost model** ($10/point, $8.40 RT friction) for all trading math. The `source_symbol` column records the actual GC contract used (e.g., GCJ1, GCG5).

**Instruments & Cost Models:**
| Instrument | $/point | RT Friction | Data Coverage |
|------------|---------|-------------|---------------|
| MGC | $10 | $8.40 | 2016-02-01 to 2026-02-14 (10yr) |
| MNQ | $2 | $2.74 | 2024-02-04 to 2026-02-14 (2yr) |
| MCL | $1 | $2.10 | 2021-07-11 to 2026-02-14 (NO EDGE) |
| MES | $1.25 | $2.10 | 2024-02-12 to 2026-02-14 (2yr) |

**ORB Sessions (11 total):**
- Fixed: 0900, 1000, 1100, 1130, 1800, 2300, 0030
- Dynamic (DST-aware): CME_OPEN, US_EQUITY_OPEN, US_DATA_OPEN, LONDON_OPEN
- Aliases: TOKYO_OPEN → 1000, HK_SG_OPEN → 1130
- Per-asset enabled sessions configured in `pipeline/asset_configs.py`

---

## Document Authority

| Document | Status | Scope | Conflict Rule |
|----------|--------|-------|---------------|
| `CLAUDE.md` | **AUTHORITY** | Code structure, commands, guardrails, AI behavior | Wins for all code decisions |
| `TRADING_RULES.md` | **AUTHORITY** | Trading rules, research findings, NO-GOs | Wins for all trading logic |
| `TRADING_PLAN.md` | Live | Current positions, sizing, session logic | Operational (changes frequently) |
| `CANONICAL_LOGIC.txt` | Frozen | Cost model, R-multiples, entry rules | Read-only spec; code in config.py is the live version |
| `CANONICAL_backfill_*.txt` | Frozen | Ingestion logic spec | Read-only spec; code in ingest_dbn_mgc.py is live |
| `ROADMAP.md` | Live | Planned features, phase status | Updated on phase completion |
| `REPO_MAP.md` | Auto-generated | Module index, CLI entry points | Never hand-edit; `python scripts/gen_repo_map.py` |
| `MARKET_PLAYBOOK.md` | Reference | Session playbooks, market structure | Supplements TRADING_RULES.md |
| `docs/ai-context/GEMINI.md` | External AI | Context for Gemini | Not authoritative for Claude |
| `docs/ai-context/LOCAL_MODEL_CONTEXT.md` | External AI | Context for local models | Not authoritative for Claude |

**Conflict resolution:** If two documents disagree:
- Code behavior -> CLAUDE.md wins
- Trading logic -> TRADING_RULES.md wins
- CANONICAL_*.txt are frozen specs; the live code is truth

---

## What Exists (Current State)

### Pipeline Code (`pipeline/`)

| File | LOC | Purpose |
|------|-----|---------|
| `ingest_dbn_mgc.py` | 925 | MGC-specific DBN ingestion with 7 validation gates |
| `ingest_dbn.py` | 525 | Generic multi-instrument wrapper |
| `build_bars_5m.py` | 352 | Deterministic 5m bar aggregation from 1m bars |
| `build_daily_features.py` | 1012 | Daily features builder (11 ORBs, sessions, RSI, outcomes) |
| `run_pipeline.py` | 284 | Pipeline orchestrator (ingest → 5m → features → audit) |
| `check_drift.py` | 1160 | Static analysis drift detector (21 checks) |
| `init_db.py` | 251 | Database schema initialization |
| `cost_model.py` | 278 | Canonical cost model (multi-instrument friction, R-multiples, stress test) |
| `dashboard.py` | 786 | Self-contained HTML report generator (7 panels) |
| `asset_configs.py` | 151 | Per-instrument config (MGC, MNQ, MCL, MES) + enabled sessions |
| `dst.py` | 262 | DST detection + dynamic session resolvers (SESSION_CATALOG) |
| `health_check.py` | 152 | Quick all-in-one health check CLI |
| `check_db.py` | 102 | Database inspection tool |
| `paths.py` | 39 | Canonical path constants |

### Trading App (`trading_app/`)

| File | LOC | Purpose |
|------|-----|---------|
| `config.py` | 296 | 12 ORB size filters + NO_FILTER + VolumeFilter, ENTRY_MODELS |
| `execution_engine.py` | 867 | Bar-by-bar state machine (ARMED → CONFIRMING → ENTERED → EXITED) |
| `portfolio.py` | 918 | Diversified strategy selection, position sizing, family dedup |
| `outcome_builder.py` | 729 | Pre-compute outcomes for RR x CB x EM grid |
| `strategy_discovery.py` | 563 | Bulk-load grid search across 5,148 combos per instrument |
| `strategy_validator.py` | 345 | 7-phase validation (incl. Phase 4b walk-forward) + risk floor + stress test |
| `walkforward.py` | 175 | Anchored walk-forward OOS validation (Phase 4b gate) |
| `paper_trader.py` | 433 | Historical replay with journal + risk management |
| `db_manager.py` | 411 | Schema for 6 trading_app tables + family head helpers |
| `entry_rules.py` | 260 | detect_confirm + resolve_entry (E1/E3) |
| `risk_manager.py` | 172 | Circuit breaker, max concurrent/daily limits |
| `execution_spec.py` | 84 | ExecutionSpec dataclass with entry_model field |
| `setup_detector.py` | 83 | Filter daily_features by conditions |
| `strategy_fitness.py` | 524 | 3-layer fitness: structural + rolling regime + decay monitoring |
| `rolling_portfolio.py` | 569 | Rolling window stability scoring + family aggregation |
| `live_config.py` | 477 | Declarative live portfolio: core (always-on) + regime-gated strategies |

### Nested ORB Research (`trading_app/nested/`)

| File | LOC | Purpose |
|------|-----|---------|
| `schema.py` | 289 | 3 new tables (nested_outcomes, nested_strategies, nested_validated) |
| `builder.py` | 390 | Resample 1m→5m + build nested outcomes (15/30m ORB + 5m entry bars) |
| `discovery.py` | 283 | Strategy discovery on nested tables |
| `validator.py` | 180 | Validation on nested tables |
| `compare.py` | 261 | A/B comparison tool (baseline vs nested) |
| `audit_outcomes.py` | 389 | Independent outcome verification |

### Research Archive (`research/`)

Completed research scripts, moved from `scripts/`. Not part of production pipeline.

| File | Purpose |
|------|---------|
| `analyze_adx_filter.py` | ADX trend filter overlay analysis |
| `analyze_double_break.py` | Double-break reversal entry research |
| `analyze_first_half_hour.py` | First-half-hour momentum analysis |
| `analyze_gap_fade.py` | Overnight gap fade strategy |
| `analyze_overlay_filters.py` | Combined overlay filter comparison |
| `analyze_prior_day_hl.py` | Prior day high/low level research |
| `analyze_rsi_reversion.py` | RSI mean-reversion analysis |
| `analyze_session_fade.py` | Session fade strategy research |
| `analyze_vwap_pullback.py` | VWAP pullback entry analysis |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `report_edge_portfolio.py` | Edge family portfolio report (per-trade + daily ledger stats) |
| `build_edge_families.py` | Cluster validated strategies by trade-day hash |
| `backfill_strategy_trade_days.py` | Populate strategy_trade_days ground truth table |
| `run_backfill_overnight.py` | Crash-recovery retry wrapper |

### Reference Documents

| File | Purpose |
|------|---------|
| `CANONICAL_LOGIC.txt` | Trading logic specification (R-multiples, entry rules, cost model) |
| `CANONICAL_backfill_dbn_mgc_rules.txt` | Ingestion rules (fail-closed, chunked, checkpointed) |
| `CANONICAL_backfill_dbn_mgc_rules_addon.txt` | Advanced ingestion patterns |
| `TRADING_RULES.md` | Single source of truth for all trading rules and research findings |
| `docs/STRATEGY_DISCOVERY_AUDIT.md` | Comprehensive strategy discovery system audit |

### Data Files (gitignored)

- `DB/GOLD_DB_FULLSIZE/` — 1,555 daily `.dbn.zst` files (2021-02-05 to 2026-02-04)
- `DB/gold_db_fullsize_2016-2021/` — 1,557 daily `.dbn.zst` files (2016-02-01 to 2021-02-03)
- `gold.db` — DuckDB database (created by `init_db.py`, ~10 years of data)

### Database Location & Workflow (CRITICAL)

There is **ONE database** (`gold.db`), but it may exist in two locations:

| Location | Purpose | When to use |
|----------|---------|-------------|
| `<project>/gold.db` | Master copy (OneDrive-synced) | Read-only queries, dashboard, fitness checks |
| `C:\db\gold.db` | Working copy (local disk) | **ALL heavy write operations** |

**Why?** OneDrive sync conflicts with DuckDB write locks, causing 10x slowdown and lock errors.

**Workflow for heavy jobs** (discovery, validation, rolling eval, outcome_builder):
```bash
# 1. Copy master to working location
cmd /c copy "C:\Users\joshd\OneDrive\Desktop\Canompx3\gold.db" "C:\db\gold.db"

# 2. Run job against working copy (use --db-path or DUCKDB_PATH env var)
set DUCKDB_PATH=C:\db\gold.db
python trading_app/strategy_discovery.py --instrument MGC

# 3. Copy results back to master
cmd /c copy "C:\db\gold.db" "C:\Users\joshd\OneDrive\Desktop\Canompx3\gold.db"
```

**Rules:**
- NEVER run two write processes against the same DuckDB file simultaneously
- NEVER run long write jobs against the OneDrive path
- `pipeline/paths.py` reads `DUCKDB_PATH` env var — set it to override the default path
- After copying back, the master is up-to-date and both copies are identical

---

## Architecture

### Data Flow

```
Databento DBN files (.dbn.zst)
  → pipeline/ingest_dbn_mgc.py (validate, filter outrights, choose front contract)
  → gold.db:bars_1m (1-minute OHLCV, UTC timestamps)
  → pipeline/build_bars_5m.py (deterministic aggregation)
  → gold.db:bars_5m (5-minute OHLCV, fully rebuildable)
  → pipeline/build_daily_features.py (ORBs, sessions, RSI)
  → gold.db:daily_features (one row per trading day per orb_minutes)

  → trading_app/outcome_builder.py (pre-compute all trade outcomes)
  → gold.db:orb_outcomes (MGC 133K + MNQ 126K + MCL 126K + MES 145K)
  → trading_app/strategy_discovery.py (grid search 5,148 combos per instrument)
  → gold.db:experimental_strategies (MGC 3,276 + MNQ 2,664 + MCL 1,800 + MES 3,744)
  → trading_app/strategy_validator.py (7-phase validation incl. Phase 4b walk-forward)
  → gold.db:validated_setups (MGC 216 + MNQ 610 + MCL 0 + MES 198)
  → scripts/build_edge_families.py (cluster by trade-day hash)
  → gold.db:edge_families (215 families from 1,024 validated strategies)
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

**daily_features** (one row per trading day per instrument per orb_minutes):
- Primary key: `(symbol, trading_day, orb_minutes)`
- `trading_day`: DATE (09:00 Brisbane boundary)
- `orb_minutes`: INTEGER (5, 15, or 30)
- `bar_count_1m`: INTEGER (bars in trading day)
- Session stats: `session_{asia,london,ny}_{high,low}` (DOUBLE)
- RSI: `rsi_14_at_0900` (DOUBLE, Wilder's 14-period on 5m closes)
- Daily OHLC: `daily_open`, `daily_high`, `daily_low`, `daily_close` (DOUBLE)
- Overnight gap: `gap_open_points` (DOUBLE, today's open - previous day's close)
- Volatility: `atr_20` (DOUBLE, 20-day SMA of True Range — regime detection)
- 11 ORBs x 9 columns each: `orb_{0900,1000,1100,1130,1800,2300,0030,CME_OPEN,US_EQUITY_OPEN,US_DATA_OPEN,LONDON_OPEN}_{high,low,size,break_dir,break_ts,outcome,mae_r,mfe_r,double_break}`
- 118 columns total (base features + 11 ORB sessions x 9 cols)
- Built by `pipeline/build_daily_features.py` (idempotent, configurable --orb-minutes)

**orb_outcomes** (pre-computed trade outcomes):
- One row per (day, ORB, RR, CB, entry_model) combination
- Per-instrument: MGC 133K, MNQ 126K, MCL 126K, MES 145K (2yr outcome windows)
- Used by strategy_discovery.py for bulk backtesting

**experimental_strategies** (grid search results):
- 5,148 combos full grid (E1: 11 ORBs x 6 RRs x 5 CBs x 13 filters = 4,290 + E3: 11x6x1x13 = 858)
- Per-asset: MGC 3,276 | MNQ 2,664 | MCL 1,800 | MES 3,744
- Metrics: sample_size, win_rate, expectancy_r, sharpe_ratio, max_drawdown_r, yearly_results

**validated_setups** (strategies passing validation):
- 1,024 validated strategies (MGC 216, MNQ 610, MES 198, MCL 0)
- Only G4+ ORB size filters have positive ExpR
- Columns: `family_hash` (TEXT), `is_family_head` (BOOLEAN)

**strategy_trade_days** (ground truth post-filter trade days):
- One row per (strategy_id, trading_day) — all entry days (win+loss+early_exit+scratch)
- 332,390 rows across all instruments
- Used by edge family clustering (MD5 hash of sorted trade days)

**edge_families** (strategy clustering by trade-day hash):
- 215 unique families from 1,024 validated strategies
- Per-instrument: MGC 48, MNQ 119, MES 48
- Head election: median expectancy_r within each family
- Robustness status: ROBUST, PURGED, WHITELISTED

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
python pipeline/check_drift.py                        # Drift detection (21 checks)
python -m pytest tests/ -v                             # Full test suite (1,072 tests)
python -m pytest tests/ -x -q                          # Fast test run (stop on first fail)
python pipeline/health_check.py                        # All-in-one health check
```

### Dashboard

```bash
python pipeline/dashboard.py                           # Generate dashboard.html
python pipeline/dashboard.py --output report.html      # Custom output path
```

### Trading App

```bash
python trading_app/outcome_builder.py --instrument MGC --start 2021-02-05 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_validator.py --instrument MGC --min-sample 50
python trading_app/strategy_validator.py --instrument MGC --no-walkforward
python trading_app/strategy_validator.py --instrument MNQ --wf-min-windows 2  # relaxed for 2yr data
python trading_app/paper_trader.py --instrument MGC --start 2025-01-01 --end 2025-12-31
python -m trading_app.live_config --db-path C:/db/gold.db     # Show live portfolio
python -m trading_app.live_config --db-path C:/db/gold.db --output live_portfolio.json
```

### Nested ORB Research

```bash
python -m trading_app.nested.builder --instrument MGC --orb-minutes 15 30
python -m trading_app.nested.discovery --instrument MGC
python -m trading_app.nested.validator --instrument MGC --min-sample 200
python -m trading_app.nested.compare --instrument MGC
```

### Testing

```bash
python -m pytest tests/ -v                             # All 1,072 tests
python -m pytest tests/test_trading_app/ -v            # Trading app tests only
python -m pytest tests/ -x -q                          # Fast run (stop on first fail)
```

---

## Guardrails (Automatic)

### 1. Git Pre-Commit Hook (`.githooks/pre-commit`)
Runs automatically before every commit:
- Drift check (`check_drift.py`)
- Fast pure-function tests
- Syntax validation on changed files
Setup: `git config core.hooksPath .githooks`

### 2. Drift Detection (`pipeline/check_drift.py` — 21 checks)
Static analysis that catches:
1. Hardcoded 'MGC' SQL in generic pipeline code
2. `.apply()`/`.iterrows()` on large data (performance anti-pattern)
3. Writes to non-bars_1m tables in ingest scripts
4. Schema-query table name mismatches (pipeline/)
5. Import cycles between pipeline modules
6. Hardcoded absolute Windows paths
7. DuckDB connection leaks (missing close/finally/atexit)
8. Pipeline → trading_app import direction (one-way dependency)
9. Trading app connection cleanup
10. Trading app hardcoded paths
11. Trading app connection leaks
12. Config filter_type sync enforcement
13. ENTRY_MODELS sync enforcement
14. Entry price sanity (no entry_price = ORB level without E3 guard)
15. Nested → production import isolation
16. Nested import validation (no cross-contamination)
17. Nested production table write guard (blocks SQL writes to orb_outcomes etc.)
18. Schema-query table name mismatches (trading_app/)
19. Timezone hygiene (blocks pytz imports and hardcoded timedelta(hours=10))
20. MarketState/scoring/cascade read-only SQL guard
21. Analytical honesty guard (sharpe_ann in discovery + view_strategies)

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

### Annualized Sharpe
- Per-trade Sharpe alone is MEANINGLESS without trade frequency
- Annualized Sharpe = per_trade_sharpe * sqrt(trades_per_year)
- Minimum bar: ShANN >= 0.5 with 150+ trades/year
- Strong: ShANN >= 0.8
- Institutional: ShANN >= 1.0

### Filter Deduplication
- Many filter variants (G2/G3/G4/NO_FILTER) produce the SAME trade set
- Always report unique trade count: group by (session, EM, RR, CB)
- Only G4+ filters meaningfully filter (>5% filter rate)
- G2/G3 on most sessions = cosmetic label, not real filtering

---

## What's NOT Built Yet

See `ROADMAP.md` for planned features:
- **Phase 6e**: Monitoring & alerting (live vs backtest drift detection)
- **orb_outcomes rebuild**: R1 fill-bar exit logic is in code but stored outcomes (689K rows) need full rebuild to apply
- **orb_outcomes 2016-2020 backfill**: outcomes only cover 2021-2026; bars_1m data exists back to 2016

**Do NOT reference unbuilt features in code or tests. Build guardrails for what exists.**
