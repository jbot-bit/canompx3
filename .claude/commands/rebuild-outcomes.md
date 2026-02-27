Full rebuild chain for instrument $ARGUMENTS (default MGC). Use after outcome_builder changes (e.g., new exit rules).

Run these steps sequentially from bash:

1. Rebuild outcomes with --force (adjust --end to latest ingested date):
```bash
python trading_app/outcome_builder.py --instrument $ARGUMENTS --force --start 2021-02-05 --end 2026-02-04
```

2. Re-run strategy discovery:
```bash
python trading_app/strategy_discovery.py --instrument $ARGUMENTS
```

3. Re-validate:
```bash
python trading_app/strategy_validator.py --instrument $ARGUMENTS --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
```

4. Rebuild edge families:
```bash
python scripts/tools/build_edge_families.py --instrument $ARGUMENTS
```

5. Run mandatory audit gates:
```bash
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
python -m pytest tests/ -x -q
```

Important: Use scratch DB for long runs. Each step depends on the previous one completing successfully. Never skip the audit gates — see `docs/plans/2026-02-27-rebuild-audit-playbook.md`.
