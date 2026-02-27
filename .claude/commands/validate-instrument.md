Validate strategies for instrument $ARGUMENTS (default MGC). Uses the correct flag combination.

Run this exact command from bash (not PowerShell):

```bash
python trading_app/strategy_validator.py --instrument $ARGUMENTS --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
```

After validation completes, run mandatory audit gates:
```bash
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
```

Important:
- Always use `--no-walkforward` for MNQ (only 2 years of data)
- `--min-years-positive-pct` is dead code without `--no-regime-waivers` — always pair them
- Use scratch DB (`C:/db/gold.db`) for long runs, copy back when done
- See `docs/plans/2026-02-27-rebuild-audit-playbook.md` for full rebuild procedures
