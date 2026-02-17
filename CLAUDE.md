# CLAUDE.md

Guidance for Claude Code working with this repository.

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Supports MGC (Micro Gold), MNQ (Micro Nasdaq), MCL (Micro Crude), MES (Micro S&P 500).

**CRITICAL: Price Data Source (MGC)**
Raw data contains GC (full-size Gold) which has better 1m bar coverage than MGC. Pipeline ingests GC bars, stores under `symbol='MGC'` (same price, same exchange), uses MGC cost model. The `source_symbol` column records the actual GC contract.

**For instruments, cost models, sessions, entry models, and all trading logic → see `TRADING_RULES.md`.**

---

## Document Authority

| Document | Scope | Conflict Rule |
|----------|-------|---------------|
| `CLAUDE.md` | Code structure, commands, guardrails, AI behavior | Wins for code decisions |
| `TRADING_RULES.md` | Trading rules, sessions, filters, research findings, NO-GOs | Wins for trading logic |
| `ROADMAP.md` | Planned features, phase status | Updated on phase completion |
| `REPO_MAP.md` | Module index, file inventory | Auto-generated (`python scripts/tools/gen_repo_map.py`) — never hand-edit |
| `docs/STRATEGY_DISCOVERY_AUDIT.md` | Strategy discovery system deep-dive | Reference only |
| `docs/RESEARCH_ARCHIVE.md` | Research findings, NO-GO archive, alternative strategy results | Supplements TRADING_RULES.md |
| `CANONICAL_*.txt` | Frozen specs | Read-only; live code is truth |

**Conflict resolution:**
- Code behavior → CLAUDE.md wins
- Trading logic → TRADING_RULES.md wins
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
