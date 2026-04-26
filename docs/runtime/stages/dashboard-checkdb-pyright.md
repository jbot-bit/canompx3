---
slug: dashboard-checkdb-pyright
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Drive pipeline/dashboard.py (30 errs) and pipeline/check_db.py (9 errs) to 0 pyright errors. All 39 errors are the same `fetchone()[0]` Optional-subscript pattern. Inline type-ignore × 39.
---

# Stage: dashboard + check_db pyright cleanup

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/dashboard.py
  - pipeline/check_db.py
  - docs/runtime/stages/dashboard-checkdb-pyright.md

## Why

Same uniform pattern as build_daily_features.py / check_drift.py / build_bars_5m.py: `con.execute(SELECT ... ).fetchone()[0]` flagged by pyright because `fetchone()` is typed Optional. SQL semantics guarantee ≥1 row from these COUNT/scalar queries.

## Blast Radius

- 2 files modified, ~39 single-line additions of `# type: ignore[index]`.
- No behavioral change — `# type: ignore` erased at runtime.
- Reversibility: single commit.
