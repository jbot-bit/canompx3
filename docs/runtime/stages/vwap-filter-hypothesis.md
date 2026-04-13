---
task: "Add VWAP breakout direction gate filter + pre-registered hypotheses"
mode: IMPLEMENTATION
stage: 1
total_stages: 2
scope_lock:
  - trading_app/config.py
  - tests/test_trading_app/test_config.py
  - docs/audit/hypotheses/2026-04-13-mnq-vwap-us-data-1000.yaml
  - docs/audit/hypotheses/2026-04-13-mnq-vwap-cme-preclose.yaml
  - scripts/tmp/mes_relative_ib_research.py
blast_radius:
  - trading_app/config.py (ALL_FILTERS, _HYPOTHESIS_SCOPED_FILTERS)
acceptance:
  - "python -m pytest tests/test_trading_app/test_config_filters.py -x -q"
  - "VWAP_MID_ALIGNED and VWAP_BP_ALIGNED in ALL_FILTERS"
  - "python pipeline/check_drift.py passes (PYTHONPATH=.)"
  - "Both hypothesis YAML files parseable"
  - "Pass-rate pre-flight: >10% and <90% for both combos"
---
