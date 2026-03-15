---
name: validate-instrument
description: Validate strategies for instrument (default MGC). Uses the correct flag combination.
disable-model-invocation: true
---
Validate strategies for instrument $ARGUMENTS (default MGC). Uses the correct flag combination.

Use when: "validate", "run validation", "validate [instrument]", "re-validate", "strategy validation"

Run this exact command from bash (not PowerShell):

```bash
python trading_app/strategy_validator.py --instrument $ARGUMENTS --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75
```

After validation completes, run mandatory steps:
```bash
python scripts/migrations/retire_e3_strategies.py   # E3 soft-retire (validator marks E3 as active — this fixes it)
python scripts/tools/build_edge_families.py --instrument $ARGUMENTS
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
```

Important:
- Walk-forward enabled for all instruments (Mar 2026). All have 5+ years of data. MGC uses WF_START_OVERRIDE=2022-01-01 in config.py.
- `--min-years-positive-pct` is dead code without `--no-regime-waivers` — always pair them
- DB path resolved via `pipeline.paths.GOLD_DB_PATH` (reads `DUCKDB_PATH` from `.env`). `C:/db/gold.db` is the SCRATCH copy, not canonical.
- See `docs/plans/2026-02-27-rebuild-audit-playbook.md` for full rebuild procedures
