---
slug: check-drift-pyright-narrow
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Drive pipeline/check_drift.py to 0 pyright errors. All 15 errors are the same pattern — `con.execute(SELECT ...).fetchone()[0]` flagged because pyright types fetchone() Optional. SQL guarantees ≥1 row from COUNT/SELECT. Inline type-ignore × 15.
---

# Stage: check_drift pyright cleanup

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/check_drift.py
  - docs/runtime/stages/check-drift-pyright-narrow.md

## Why

Pipeline drift system is the truth-layer enforcement gate that runs on every commit (113 checks). 15 pyright errors in this file all of the same `fetchone()[0]` None-subscript pattern. Same resolution as `build_daily_features.py` Cluster F: type-ignore on the SQL count queries. SQL semantics guarantee at least one row from `SELECT COUNT(*) FROM ...` — pyright's Optional-aware return type is overly conservative.

## Blast Radius

- **Files modified:** `pipeline/check_drift.py` only.
- **Diff size:** ~15 single-line additions (`# type: ignore[index]`).
- **Tests:** drift system has no unit tests; smoke test = `python pipeline/check_drift.py` exits 0.
- **Production behavior change:** zero — `# type: ignore` is erased at runtime.
- **Reversibility:** single commit; revert via `git revert`.

## Method

Read each error line; if the call is exactly `.fetchone()[0]` (SQL count/scalar pattern), apply the inline ignore. Anything that looks like an actual `Optional` lookup gets a real `is None` guard, not a suppression.
