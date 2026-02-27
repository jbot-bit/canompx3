Run the full project health check suite. No arguments needed.

Run these checks:

1. Drift detection (static analysis — 37 checks):
```bash
python pipeline/check_drift.py
```

2. Data integrity audit (17 checks):
```bash
python scripts/tools/audit_integrity.py
```

3. Fast test suite (stop on first failure):
```bash
python -m pytest tests/ -x -q
```

4. Pipeline health check (runs all of the above + deps/DB/hooks):
```bash
python pipeline/health_check.py
```

Report results for each step. If drift, integrity, or tests fail, investigate before proceeding. See `docs/plans/2026-02-27-rebuild-audit-playbook.md` for recovery procedures.
