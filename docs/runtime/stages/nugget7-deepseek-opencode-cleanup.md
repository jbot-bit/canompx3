---
task: "NUGGET 7 — Remove DeepSeek/OpenCode dead tooling"
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
---

## Blast Radius

- pipeline/check_drift.py — remove 3 dead check functions + 3 CHECKS registrations; no production logic changed
- tests/test_pipeline/test_check_drift.py — remove 2 dead test classes + 2 dead imports; mirrors check_drift.py removals
- 10 DeepSeek/OpenCode files already deleted via git rm (no pipeline/trading_app logic)
- Reads: none. Writes: none to DB or runtime.
