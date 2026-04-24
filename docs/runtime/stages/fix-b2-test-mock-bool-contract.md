## Stage: fix-b2-test-mock-bool-contract

mode: IMPLEMENTATION
date: 2026-04-25
scope_lock:
  - tests/test_trading_app/test_bot_dashboard.py
  - docs/runtime/stages/fix-b2-test-mock-bool-contract.md

## Blast Radius

- `tests/test_trading_app/test_bot_dashboard.py::test_preflight_helper_opens_no_duckdb_connection` — line 614 mock stub updated from `lambda *a, **k: None` to `lambda *a, **k: True`.
- Zero production code touched.
- Grep confirms exactly one test in the repo stubs `notify()` with a fixed return value (this one). Other `patch("trading_app.live.notifications.notify", ...)` sites either use `side_effect=` (raise) or `MagicMock` defaults (truthy) and are unaffected.

## Why

Commit `5be52bdc` (B2 fix, 2026-04-24) changed `trading_app/live/notifications.py::notify()` from `-> None` to `-> bool` and updated `scripts/run_live_session.py::_run_lightweight_component_self_tests()` preflight check 5 to read the return value. `test_preflight_helper_opens_no_duckdb_connection` is a regression guard for the 45f50916 journal-lock fix (asserts preflight opens no duckdb connections). Its `notify` mock returns `None` under the legacy contract — post-B2, `None` is falsy → `results["notifications"] = False` → `assert results["notifications"] is True` fails, even though the test's real assertion (zero duckdb.connect calls) still passes.

The test's intent is DB-leak detection, not Telegram connectivity. Mock must reflect the new bool contract: return `True` = "notify succeeded, move on".

Per `.claude/rules/institutional-rigor.md` § 2 ("after any fix, review the fix") — B2 landed last session without a repo-wide grep for test stubs matching the old contract. This fix closes that miss.

## Verification

1. `PYTHONPATH=. python -m pytest tests/test_trading_app/test_bot_dashboard.py -q` — full file green.
2. `PYTHONPATH=. python -m pytest tests/test_trading_app/ -q` — full trading_app suite green.
3. `PYTHONPATH=. python -m pytest tests/test_trading_app/test_notifications.py -q` — B2-specific tests still green (no regression).
4. `PYTHONPATH=. python pipeline/check_drift.py` — no NEW failures.

## Out of scope

- Rewiring B2 fix or preflight contract (both correct as-is).
- Adding new tests — existing regression guard remains intact, only its mock is corrected.
- Fixing `notify` docstring or stage doc for `live-b2-notifications-bool-return` (already archived in commit `af4897e3`).

## Commit

`fix(test): align test_preflight mock with post-B2 notify() bool contract`
