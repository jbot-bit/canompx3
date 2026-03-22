# Architecture & Commands Reference

## Data Flow

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

## Price Data Sources (Full-Size → Micro Mapping)

Several instruments use full-size contract data for better 1m bar coverage. Full-size and micro contracts trade at identical prices on the same exchange — only the multiplier differs. Pipeline stores under the micro symbol; `source_symbol` records the actual contract.

**Active instruments (Mar 2026): MGC, MNQ, MES.** Dead for ORB: M2K, MCL, SIL, M6E, MBT.

| Stored Symbol | Source Contracts | Reason | Cost Model | Status |
|--------------|-----------------|--------|------------|--------|
| MGC | GC (full gold) | Better 1m coverage than MGC | MGC ($10/pt) | **ACTIVE** |
| MES | ES (pre-Feb 2024), then native MES | ES has data back to 2019 | MES ($5/pt) | **ACTIVE** |
| MNQ | MNQ (native micro) | No mapping needed | MNQ ($2/pt) | **ACTIVE** |
| M2K | RTY (E-mini Russell) | RTY has better coverage | M2K ($5/pt) | DEAD — 0/18 families survive null test (Mar 2026) |
| M6E | 6E (full EUR/USD) | 6E has better coverage | M6E ($12,500/pt) | DEAD — 0/2064 validated (Feb 2026) |
| SIL | SI (full silver) | SI has better coverage | SIL ($1,000/pt) | DEAD — 0/432 validated (Feb 2026) |
| MCL | CL (full crude oil) | Better 1m coverage than MCL | MCL ($100/pt) | DEAD — 0 validated |

## Key Commands

```bash
# Guardrails (run frequently)
python pipeline/check_drift.py               # Drift detection (count self-reported at runtime)
python -m pytest tests/ -x -q                # Fast test suite
python pipeline/health_check.py              # All-in-one health check

# Database
python pipeline/init_db.py                    # Create schema
python pipeline/init_db.py --force            # Drop + recreate (DESTROYS DATA)

# Pipeline (typical flow: ingest → 5m bars → daily features)
python pipeline/ingest_dbn.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/ingest_dbn.py --instrument MGC --resume
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-12-31
python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-12-31

# Pipeline Status & Rebuild
python scripts/tools/pipeline_status.py --status              # Staleness for all instruments
python scripts/tools/pipeline_status.py --rebuild --instrument MGC  # Rebuild stale steps
python scripts/tools/pipeline_status.py --rebuild-all         # All stale instruments
python scripts/tools/pipeline_status.py --resume --instrument MGC   # Resume failed rebuild

# Trading App
python trading_app/paper_trader.py --instrument MGC --start 2025-01-01 --end 2025-12-31
python -m trading_app.live_config --db-path C:/db/gold.db
python pipeline/dashboard.py                 # Generate dashboard.html
python scripts/reports/report_edge_portfolio.py      # Edge family portfolio report

# Tooling
ruff format pipeline/ trading_app/ ui/ scripts/ tests/  # Format all code
ruff check pipeline/ trading_app/ ui/ scripts/           # Lint all code
ruff check --fix pipeline/ trading_app/ ui/ scripts/     # Auto-fix lint issues
pyright                                                   # Type check (basic mode)
uv sync --frozen                                          # Install from lock file
uv lock                                                   # Regenerate lock file
pip-audit --desc on                                       # Security scan
```

For outcome rebuild, validation, and edge family workflows, use slash commands:
`/validate-instrument MGC`, `/rebuild-outcomes MGC`, `/health-check`

## Configuration (.env)

```
DATABENTO_API_KEY=...         # Required for backfills
DUCKDB_PATH=gold.db           # Override DB location
SYMBOL=MGC                    # Default instrument
TZ_LOCAL=Australia/Brisbane   # Local timezone
```

## Strategy Classification Rules (FIX5)

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

See CLAUDE.md "Strategy Classification — Behavioral Rules" for the 7 non-negotiable rules.
