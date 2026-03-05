# Validation Workflow

## Correct Validator Flags
```bash
# Walk-forward enabled for all instruments (Mar 2026).
# All 4 active instruments have 5+ years — sufficient for WF.
# MGC uses WF_START_OVERRIDE=2022-01-01 in config.py (regime shift).
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 \
  --no-regime-waivers --min-years-positive-pct 0.75
```

## Critical Notes
- **All instruments use WF testing** (Mar 2026). MGC (10yr), MES (7yr), MNQ (5yr), M2K (5yr) all have sufficient data. `--no-walkforward` is only for debugging/one-off analysis.
- **MGC WF_START_OVERRIDE=2022-01-01** in config.py — skips pre-2022 low-ATR regime for WF windows only.
- **`--min-years-positive-pct` is dead code** when regime waivers are enabled (default). It lives in an `else` branch only reached with `--no-regime-waivers`. Always pair them.
- **PowerShell breaks `--` flags** — always run validators from bash
- **DB path:** `pipeline/paths.py` auto-loads `.env` -> `DUCKDB_PATH`. No manual env var needed.

## Full Rebuild Chain
When outcome_builder changes (e.g., C8/C3 exit rules):
1. `python trading_app/outcome_builder.py --instrument MGC --force --orb-minutes 5`  # repeat for 15, 30
2. `python trading_app/strategy_discovery.py --instrument MGC --orb-minutes 5`  # repeat for 15, 30
3. `python trading_app/strategy_validator.py --instrument MGC --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75`
4. `python scripts/migrations/retire_e3_strategies.py`  # validator promotes E3; this fixes it
5. `python scripts/tools/build_edge_families.py --instrument MGC`
6. `python scripts/tools/gen_repo_map.py`  # auto-regen file inventory
7. `python pipeline/health_check.py`  # post-rebuild validation gate
8. `python scripts/tools/sync_pinecone.py`  # sync knowledge base

Or use the wrapper: `bash scripts/tools/run_rebuild_with_sync.sh MGC`

## Strategy Classification
| Class | Min Samples | Usage |
|-------|------------|-------|
| CORE | >= 100 | Standalone portfolio weight |
| REGIME | 30-99 | Conditional overlay / signal only |
| INVALID | < 30 | Not tradeable |

Never treat "low trade count" alone as evidence of a bug. G6/G8 filters producing low N is expected.
