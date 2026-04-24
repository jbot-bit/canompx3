# Stage: live-b2-notifications-bool-return

mode: IMPLEMENTATION
date: 2026-04-24
scope_lock:
  - trading_app/live/notifications.py
  - scripts/run_live_session.py
  - tests/test_trading_app/test_notifications.py
  - docs/runtime/stages/live-b2-notifications-bool-return.md

## Blast Radius

- `trading_app/live/notifications.py::notify()` signature returns `bool` instead of `None`. Existing callers ignore the return value (verified via `grep -nE "= notify\(|if notify\("` returning 0 matches in `trading_app/`). Additive change; no behavioral regression.
- `scripts/run_live_session.py::_run_lightweight_component_self_tests()` preflight check 5 now reads the return value and sets `results["notifications"] = False` on failure. Loud instead of silent.
- New test file `tests/test_trading_app/test_notifications.py`: 3 tests exercising success / raise / never-raises contract.
- Stage doc under `docs/runtime/stages/`.

## Why

`trading_app/live/notifications.py::notify()` has been swallowing every exception and returning `None`. Preflight check 5 wrapped the call in its own `try/except` and set `results["notifications"] = True` in the success branch — but since `notify()` cannot raise, the success branch always fired. Net result: a misconfigured Telegram (wrong token, wrong chat_id, network drop) silently drops every live notification and the preflight self-test never flags it.

Per `.claude/rules/institutional-rigor.md` rule 6 ("no silent failures") and `.claude/rules/integrity-guardian.md` rule 3 ("never catch Exception and return success in health/audit paths"), the fix is to have `notify()` return a bool so the preflight can distinguish a working Telegram pipe from a silently broken one.

## Verification

1. `pytest tests/test_trading_app/test_notifications.py tests/test_trading_app/test_instance_lock.py -q` — new test file green + no regression.
2. `python scripts/run_live_session.py --instrument MNQ --preflight` — check 5 now shows real status.
3. `python pipeline/check_drift.py` — no NEW failures (pre-existing check #4 false positive on `work_queue.py` and check #59 MNQ daily_features gap are known, untouched).
4. `grep -n "def notify" trading_app/live/notifications.py` — exactly one definition, `-> bool` return type.

## Out of scope

- Fixing the underlying Telegram config (separate task; this just makes it detectable).
- `_notify()` wrapper in session_orchestrator.py (already wraps `notify()` with its own try/except; unchanged).
- Per-callsite return-value propagation (20+ call sites, all log-then-continue; not changing).

## Commit

`fix(live): notify() returns bool so preflight self-test can detect broken Telegram`
