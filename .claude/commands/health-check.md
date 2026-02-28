Run the full project health check suite. No arguments needed.

Run these checks:

1. Drift detection (static analysis — check count self-reported at runtime):
```bash
python pipeline/check_drift.py
```

2. Data integrity audit (check count self-reported at runtime):
```bash
python scripts/tools/audit_integrity.py
```

3. Behavioral audit (anti-pattern scanner):
```bash
python scripts/tools/audit_behavioral.py
```

4. Fast test suite (stop on first failure):
```bash
python -m pytest tests/ -x -q
```

5. Pipeline health check (runs all of the above + deps/DB/hooks):
```bash
python pipeline/health_check.py
```

Report results for each step. If drift, integrity, or tests fail, investigate before proceeding. See `docs/plans/2026-02-27-rebuild-audit-playbook.md` for recovery procedures.
