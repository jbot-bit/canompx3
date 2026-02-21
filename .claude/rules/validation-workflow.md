# Validation Workflow

## Correct Validator Flags
```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 \
  --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
```

## Critical Notes
- **Always use `--no-walkforward` for MNQ** — only 2 years of data, never has 3+ WF windows
- **`--min-years-positive-pct` is dead code** when regime waivers are enabled (default). It lives in an `else` branch only reached with `--no-regime-waivers`. Always pair them.
- **PowerShell breaks `--` flags** — always run validators from bash
- **DB path:** `pipeline/paths.py` auto-loads `.env` -> `DUCKDB_PATH`. No manual env var needed.

## Full Rebuild Chain
When outcome_builder changes (e.g., C8/C3 exit rules):
1. `python trading_app/outcome_builder.py --instrument MGC --force --start 2021-02-05 --end 2026-02-04`  # adjust --end to latest ingested date
2. `python trading_app/strategy_discovery.py --instrument MGC`
3. `python trading_app/strategy_validator.py --instrument MGC --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward`
4. `python scripts/tools/build_edge_families.py --instrument MGC`

## Strategy Classification
| Class | Min Samples | Usage |
|-------|------------|-------|
| CORE | >= 100 | Standalone portfolio weight |
| REGIME | 30-99 | Conditional overlay / signal only |
| INVALID | < 30 | Not tradeable |

Never treat "low trade count" alone as evidence of a bug. G6/G8 filters producing low N is expected.
