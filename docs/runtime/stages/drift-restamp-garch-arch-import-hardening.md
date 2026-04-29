---
task: drift-restamp-garch-arch-import-hardening
mode: IMPLEMENTATION
scope_lock:
  - pipeline/build_daily_features.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
blast_radius: |
  pipeline/build_daily_features.py:837 — ImportError except branch only; log level
  rises DEBUG→WARN. No data-shape change. No call-site change. Function signature
  unchanged. pipeline/check_drift.py — pure additive: new function
  check_garch_dependency_importable() + new entry in registry tuple list.
  tests/test_pipeline/test_check_drift.py — one new targeted test for the new check.
agent: claude
updated: 2026-04-30
---

# Drift restamp — GARCH arch-import hardening

**Status:** IMPLEMENTATION (1/1)
**Date:** 2026-04-30
**Worktree:** `canompx3-data-drift-restamp` on `session/joshd-data-drift-restamp`
**Owner:** Claude (this session)

## Why this stage exists

Two drift checks fired on `origin/main`:

- **Check 48** — 10 active VALIDATOR_NATIVE strategies have stale `validated_setups.last_trade_day` lagging the canonical recompute by 2-26 trading days. Already fixed by running `scripts/migrations/backfill_validated_trade_windows.py` (no code change).
- **Check 65** — 9 NULL `garch_forecast_vol` rows on 2026-04-28 across MES/MGC/MNQ × O5/O15/O30. Root cause: `arch>=8.0.0` (a hard pyproject.toml dep) was missing from the canonical venv `C:\Users\joshd\canompx3\.venv`. `pipeline.build_daily_features.compute_garch_forecast` swallows `ImportError` at `logger.debug` level and returns `None`, so every daily build silently NULLed GARCH on the new row without any operator-visible signal.

Already done outside this stage scope:
- `pip install 'arch>=8.0.0'` into the canonical venv.
- Extended `scripts/tools/backfill_garch.py` to also write `garch_forecast_vol_pct` (canonical helper `_prior_rank_pct` from `build_daily_features`). Re-ran for MES/MGC/MNQ on 2026-04-28+2026-04-29; Check 65 now passes (verified via `python pipeline/check_drift.py`).

Two hardening edits remain — both touch NEVER_TRIVIAL files — and need this stage:

1. `pipeline/build_daily_features.py` — change the `compute_garch_forecast` ImportError swallow from `logger.debug` to `logger.warning` so future venv drift is visible in `logs/daily_refresh.log`.
2. `pipeline/check_drift.py` — add `check_garch_dependency_importable()` that fails-closed if `arch` cannot be imported. Register in the check tuple list.

## Scope Lock

- pipeline/build_daily_features.py
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift.py

## Blast Radius

- `pipeline/build_daily_features.py:837` (the ImportError except branch). The runtime behavior is unchanged on the happy path (arch present); on the sad path the log level rises from DEBUG to WARN. No data-shape change. No call-site change. Function signature unchanged.
- `pipeline/check_drift.py` — pure additive. New function + new entry in the registry tuple. Cannot break existing checks.
- One new test in `tests/test_pipeline/test_check_drift.py` to guard the new check.

## Acceptance criteria

- [x] `arch>=8.0.0` installed and verified via `python -c 'import arch; print(arch.__version__)'`
- [x] 2026-04-28 + 2026-04-29 GARCH columns populated for MES/MGC/MNQ across O5/O15/O30
- [x] 10 stale `validated_setups` rows refreshed
- [x] `compute_garch_forecast` ImportError now logs WARN with remediation
- [x] New `check_garch_dependency_importable()` registered and passing
- [x] `pipeline/check_drift.py` shows Check 48 and Check 65 PASSED
- [x] Targeted tests added and passing
- [x] Self-review pass
