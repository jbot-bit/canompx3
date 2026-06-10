---
task: Fix refresh_data build boundary to delegate to canonical trading-day window (kills the Check-79-vs-refresh_data mismatch that silently skipped a complete trading day)
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/refresh_data.py
  - tests/test_tools/test_refresh_data.py
---

## Blast Radius

- `scripts/tools/refresh_data.py` — modifies the build-boundary computation in
  `refresh_instrument`. Adds a module-level `_now_utc()` (testable seam) and a
  `_last_complete_trading_day(now_utc)` helper that delegates to canonical
  `pipeline.dst.compute_trading_day_from_timestamp` /
  `compute_trading_day_utc_range`. The Databento availability clamp
  (`api_end`/`yesterday` as *fetch* range) is UNTOUCHED. Build boundary becomes
  `build_end = min(_last_complete_trading_day(now), last_date)`. One in-repo
  caller (`main` at :354) — signature unchanged, no caller edits needed.
  Reads: Databento metadata (unchanged), gold.db (read-only via get_last_bar_date).
  Writes: none directly (delegates to existing builder subprocesses, idempotent
  DELETE+INSERT).
- `tests/test_tools/test_refresh_data.py` — companion tests. Existing tests
  asserted `build_end == today-1` (the buggy calendar behavior) and are updated
  to assert the canonical trading-day boundary. New regression tests for the
  06-09 just-closed-day case, in-progress-window case, the 09:00/23:00 UTC edge,
  and the `min(boundary, last_date)` clamp.
- No schema change. No canonical-source (pipeline/dst, cost_model, etc.) change —
  this DELEGATES to canonical dst, it does not modify it.
- Does NOT touch Check 79 (`pipeline/check_drift.py:5011`) — it is correct.

## Why

Two modules answered "what is the last complete trading day?" with two different
formulas. Check 79 used canonical `compute_trading_day_utc_range` (window fully
elapsed); refresh_data used calendar `today-1`. When Databento availability
clamped to 06-09, refresh built only through 06-08, silently dropping the
complete 06-09 trading day → Check 79 demanded the build → blocked an unrelated
code commit. Fix = delegate refresh_data's build boundary to pipeline.dst so the
two can never disagree (institutional-rigor §4, §10).
