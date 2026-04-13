---
task: "Fix VWAP filter drift failures: route in get_filters_for_grid + family_rr_locks"
mode: IMPLEMENTATION
stage: 1
total_stages: 1
scope_lock:
  - trading_app/config.py
  - tests/test_trading_app/test_config.py
blast_radius:
  - trading_app/config.py (get_filters_for_grid routing)
acceptance:
  - "PYTHONPATH=. python pipeline/check_drift.py — checks 41 and 60 PASS"
  - "python -m pytest tests/test_trading_app/test_config.py -q — all pass"
---
