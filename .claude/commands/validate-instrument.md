Validate strategies for instrument $ARGUMENTS (default MGC). Uses the correct flag combination.

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
- `--no-walkforward` for MNQ — currently run without WF; ~5yr data means WF is feasible (re-run without flag to enable)
- `--min-years-positive-pct` is dead code without `--no-regime-waivers` — always pair them
- Use scratch DB (`C:/db/gold.db`) for long runs, copy back when done
- See `docs/plans/2026-02-27-rebuild-audit-playbook.md` for full rebuild procedures
