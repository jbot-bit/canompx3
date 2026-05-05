---
task: Add drift check to prevent third recurrence of orb_minutes=5 hardcode in scoring paths
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_aperture.py
---

## Blast Radius

- `pipeline/check_drift.py` — add new check `check_aperture_hardcode_in_scoring_paths()` registered in `CHECKS` list at the end of the file. Greps `trading_app/paper_*.py` and `trading_app/lane_*.py` for `WHERE\s+(symbol|s)?\.?\s*=?\s*\?\s+AND\s+orb_minutes\s*=\s*5` patterns (and equivalent forms). Allows opt-out via `# canonical-cte-guard:` comment on the same or previous line — for cases like `_get_trading_days` (DISTINCT dedup) and `_get_median_atr_20` (non-aperture scalar). Allows opt-out via `# session-regime-gate:` comment for the deliberate `lane_allocator.py:454` case.
- `tests/test_pipeline/test_check_drift_aperture.py` — new test file. 4 tests: (a) check runs on a fixture file with a violation → returns error; (b) check runs on a fixture with `# canonical-cte-guard:` annotation → passes; (c) check runs on the live `trading_app/` dir → passes (post PR #231 + PR #232); (d) check is registered in CHECKS list.
- Reads: `trading_app/` source files (read-only).
- Writes: none.
- Affects: pre-commit and `pipeline/check_drift.py` runs gain one more enforcement gate. Adds ~30s to drift-check runtime (single grep over ~10 files).

## Why

PR #189 fixed the class bug in `lane_allocator.py`. PR #231 fixed the recurrence in `paper_trade_logger.py`. PR #232 fixed the recurrence in `paper_trader.py`. Three recurrences of the same fingerprint = architectural smell. Per institutional-rigor rule "If review cycles keep finding new divergences, the architecture is wrong — stop patching. Name the root cause, propose the structural change."

Root cause: there is no canonical "load aperture-correct daily_features for a lane" helper; every lane-iterating consumer rolls its own SQL with literal `WHERE orb_minutes=5`. Long-term refactor would extract a `load_features_for_lane(con, lane)` helper. Short-term defense (this PR): a drift check that fails fast on the syntactic pattern in scoring-path files.

The check must NOT flag canonical CTE-Guard dedup (per `.claude/rules/daily-features-joins.md`), which appears in research scripts, allocator session-regime gates, and DISTINCT/aggregate reads of non-aperture columns. Discrimination by file scope (only `paper_*.py` + `lane_*.py`) + opt-out comment annotation handles both classes.

## Acceptance

- New check registered, fires on synthetic violation, exempts annotated lines.
- `python pipeline/check_drift.py` returns same exit code as pre-change (post PR #231 + PR #232 main has no remaining violations in scope).
- 4 unit tests pass.
