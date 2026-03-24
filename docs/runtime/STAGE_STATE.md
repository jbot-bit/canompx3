---
mode: IMPLEMENTATION
task: Fix 2 Critical + 4 Important from code review
scope_lock:
  - .gitignore
  - trading_app/live_config.py
  - scripts/reports/monitor_paper_forward.py
stage: 1/1
acceptance: |
  Accidental files removed + gitignored.
  Inlined stats replaced with @research-source refs.
  t-test min N raised to 30.
  EUROPE_FLOW/COMEX_SETTLE comments updated with 10yr caveat.
  Drift checks pass.
---
