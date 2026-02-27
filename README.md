# Canompx3 — Multi-Instrument Futures ORB Trading Pipeline

Self-contained data pipeline and backtesting engine for Opening Range Breakout (ORB) strategies on micro futures (MGC, MNQ, MES, M2K). 10 years of 1-minute bar data, 1,251 validated strategies across 5/15/30m ORB apertures, 37 drift checks, fully automated guardrails.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python pipeline/init_db.py

# Ingest raw data (requires .dbn.zst files in DB/)
python pipeline/run_pipeline.py --instrument MGC --start 2016-02-01 --end 2026-02-04

# Run strategy discovery + validation
python trading_app/outcome_builder.py --instrument MGC --start 2021-02-05 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_validator.py --instrument MGC --min-sample 50

# View results
python trading_app/view_strategies.py
python trading_app/view_strategies.py --orb 0900 --sort sharpe
python pipeline/dashboard.py
```

## Project Structure

```
pipeline/           Data pipeline (ingest, aggregate, features, validation)
trading_app/        Trading engine (strategies, execution, portfolio, risk)
trading_app/nested/ Nested ORB research (15m/30m ORB + 5m entry bars)
tests/              655 tests (all passing)
scripts/            Utilities (backup, parallel ingest)
docs/               Plans, archives, analysis documents
.githooks/          Pre-commit hook (lint + drift + tests + syntax)
.github/workflows/  CI pipeline (GitHub Actions)
```

## Key Commands

```bash
python trading_app/view_strategies.py --summary    # Strategy overview
python pipeline/check_drift.py                     # 37 static analysis checks
python -m pytest tests/ -x -q                      # Full test suite
python pipeline/dashboard.py                       # Generate HTML dashboard
python scripts/backup_db.py                        # Backup gold.db
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for full architecture, schema, time model, and guardrails documentation.

See [ROADMAP.md](ROADMAP.md) for development phases and status.

## Guardrails

Every commit is gated by:
1. **Ruff lint** -- catches unused imports, undefined names, unreachable code
2. **37 drift checks** -- architecture isolation, config sync, timezone hygiene
3. **600+ fast tests** -- pipeline, trading app, nested ORB
4. **Syntax validation** -- on all staged .py files

CI runs the same checks on push/PR to main.
