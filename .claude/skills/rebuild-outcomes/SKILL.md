---
name: rebuild-outcomes
description: Full rebuild chain for instrument (default MGC). Use after outcome_builder changes (e.g., new exit rules).
disable-model-invocation: true
---
Full rebuild chain for instrument $ARGUMENTS (default MGC). Use after outcome_builder changes (e.g., new exit rules).

Use when: "rebuild", "rebuild outcomes", "full rebuild", "rebuild [instrument]", "outcome_builder changed", "re-run the chain"

Run these steps sequentially from bash:

1. Rebuild outcomes with --force for all apertures:
```bash
for OM in 5 15 30; do
    python trading_app/outcome_builder.py --instrument $ARGUMENTS --force --orb-minutes $OM
done
```

2. Re-run strategy discovery for all apertures:
```bash
for OM in 5 15 30; do
    python trading_app/strategy_discovery.py --instrument $ARGUMENTS --orb-minutes $OM
done
```

3. Re-validate (walk-forward enabled for all instruments):
```bash
python trading_app/strategy_validator.py --instrument $ARGUMENTS --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75
```

4. Rebuild edge families:
```bash
python scripts/migrations/retire_e3_strategies.py
python scripts/tools/build_edge_families.py --instrument $ARGUMENTS
```

5. Run mandatory audit gates:
```bash
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
python -m pytest tests/ -x -q
```

Or use the wrapper: `bash scripts/tools/run_rebuild_with_sync.sh $ARGUMENTS`

Important: Each step depends on the previous one completing successfully. Never skip the audit gates.
