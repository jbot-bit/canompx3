---
mode: IMPLEMENTATION
task: Fix 3 vanilla Claude compat issues (PyYAML dep, hardcoded instruments, relative path)
scope_lock:
  - pyproject.toml
  - scripts/databento_daily.py
  - scripts/tools/migrate_fairness_audit.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - pyyaml in pyproject.toml deps
  - databento_daily.py uses ACTIVE_ORB_INSTRUMENTS guard
  - migrate_fairness_audit.py uses absolute path
  - scripts still import cleanly
---
