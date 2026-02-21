Run the full project health check suite. No arguments needed.

Run these checks:

1. Drift detection (static analysis — 28+ checks):
```bash
python pipeline/check_drift.py
```

2. Fast test suite (stop on first failure):
```bash
python -m pytest tests/ -x -q
```

3. Pipeline health check:
```bash
python pipeline/health_check.py
```

Report results for each step. If drift or tests fail, investigate before proceeding.
