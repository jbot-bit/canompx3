---
task: Ralph Loop iter 225 — projectx auth silent exception in _validate_or_login
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/projectx/auth.py
updated: 2026-05-31T00:00:00
agent: ralph
---

## Blast Radius
- trading_app/live/projectx/auth.py — bind exception in except clause at line 123, add %s to log.warning; zero logic change
- No test file changes needed (existing 3 tests cover the path; diagnostic-only change)
