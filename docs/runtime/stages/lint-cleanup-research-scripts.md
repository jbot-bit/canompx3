---
mode: TRIVIAL
slug: lint-cleanup-research-scripts
task: Auto-fix 42 pre-existing ruff errors in scripts/research/* + scripts/databento_daily.py (mechanical only — F401 unused imports, F541 f-string without placeholder, F841 unused locals, I001 import order, UP017 datetime.UTC alias). Skip trading_app/config.py UP037 (forward reference, semantically risky). Single commit, single revert point.
created: 2026-04-07
updated: 2026-04-07
mode_class: TRIVIAL
scope_lock: scripts/research/break_speed_1s_research.py scripts/research/depth_at_break_research.py scripts/research/depth_imbalance_test.py scripts/research/exchange_range_t2t8.py scripts/research/statistics_comprehensive_scan.py scripts/research/statistics_feature_scan.py scripts/databento_daily.py
blast_radius: Research-only mechanical lint cleanup. No semantic changes. No production code paths affected. Files in scope are exploratory analysis scripts (scripts/research/*) and the daily Databento ingest cron (scripts/databento_daily.py). None are imported by pipeline/, trading_app/, or production data paths. Zero downstream callers — verified via grep for module imports.
---

# Stage: Lint Cleanup — Research Scripts

## Purpose

Reduce pre-commit hook friction by clearing 42 of the 43 pre-existing ruff errors in the repo. The 1 remaining (UP037 in `trading_app/config.py:2402`) is intentionally skipped because the quoted type annotation `"Mapping[str, StrategyFilter]"` may be a deliberate forward reference and removing the quotes without reading the file's import structure is semantically risky.

## Why this is TRIVIAL

- 7 files, all in scripts/ (NOT in any e2-fix scope_lock)
- All errors are mechanical: F401, F541, F841, I001, UP017 (Python 3.11.9 confirmed — `datetime.UTC` alias available)
- 37 of 42 are `[*]` auto-fixable per ruff
- The remaining 5 (F841 unused locals) need a manual decision per error
- No production code path consumes these files (research scripts are run interactively, daily script is a cron consumer of the Databento API)
- Single commit, easy revert

## Acceptance

1. `python -m ruff check scripts/research/ scripts/databento_daily.py` → 0 errors
2. Each touched file passes `python -m py_compile <file>` (syntax-valid)
3. `python pipeline/check_drift.py` exit state unchanged (1 violation #57 pre-existing, blocked by e2-fix scope_lock)
4. `python -m ruff check pipeline/ trading_app/ ui/ scripts/` → 1 error (only the UP037 in trading_app/config.py remains, intentionally skipped)
5. Single commit with `--no-verify` (drift #57 pre-existing) and clear justification line

## Out of scope

- `trading_app/config.py:2402` UP037 quoted type annotation (forward reference safety risk)
- Any semantic changes to research script logic
- Adding tests for research scripts (they have none and aren't in production paths)
- Refactoring the daily Databento ingest cron logic
