## Stage: ralph-iter-179-pass-one-hardening

mode: IMPLEMENTATION
date: 2026-04-25
branch: ralph/crit-high-burndown-v5.2
scope_lock:
  - trading_app/live/session_orchestrator.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_session_safety_state.py
  - docs/runtime/stages/ralph-iter-179-pass-one-hardening.md
  - docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md

## Blast Radius

- `trading_app/live/session_orchestrator.py` — adds `_shutdown_task` async helper near `run`; retrofits four cancel callsites in the `finally` block of `async def run`. Helper is additive; retrofit is behavior-preserving with the upgrade of bounded-timeout await + critical-log on hang.
- `pipeline/check_drift.py` — adds drift check 114 enforcing C1 kill-switch guards remain at `_on_bar` and `_handle_event` ENTRY branch in `session_orchestrator.py`. Registers in `CHECKS` tuple. Bumps total expected count 113 → 114.
- `tests/test_trading_app/test_session_orchestrator.py` — adds T4 replacement test that calls real `_handle_event` (not a mock) with an EXIT event under `_kill_switch_fired=True`, asserts the EXIT branch executes (passthrough). Closes audit gap S2 from iter 178.
- `tests/test_trading_app/test_session_safety_state.py` — adds R3 cross-restart test instantiating `SessionSafetyState` twice on the same `tmp_path` file, asserting `last_connected_at` round-trips through disk. Closes audit gap S1 from iter 178.

Zero touch on mes-debt branch files. Zero touch on Codex helpers introduced in `cddc8afd`. Zero touch on `_check_trading_day_rollover` idempotency or `_fire_kill_switch` persistence (audit do-not-touch list).

## Why

Plan v5.2 Pass One. Two motivations combined:
1. **Structural hardening.** S3 from iter 178 audit: cancel-without-await pattern is inherited by every new `asyncio.create_task` in the orchestrator. R4 / R5 / F7 will each add another leak unless the pattern is collapsed into a named helper. Make the right thing the default; retrofit the four existing callsites.
2. **Drift-check regression prevention.** C1 (iter 174 audit) was caught because `_handle_event` lacked a guard that mirrored `_on_bar`. Without enforcement, a future refactor can re-open the race. Drift check 114 fails-closed in CI if either guard goes missing or the C1 guard is widened to a blanket (which would break EOD wind-down per audit do-not-touch).

Plus two test-only fixes routed in from iter 178 audit (CONDITIONAL verdict): R3 cross-restart round-trip and T4 EXIT-passthrough using real `_handle_event`. Both close INFERRED-not-MEASURED gaps.

## Verification

1. `PYTHONPATH=. python -m pytest tests/test_trading_app/test_session_orchestrator.py -q` — full file green, including 3 new shutdown-helper tests + T4 replacement test.
2. `PYTHONPATH=. python -m pytest tests/test_trading_app/test_session_safety_state.py -q` — full file green, including new R3 cross-restart test.
3. `PYTHONPATH=. python pipeline/check_drift.py` — total 114 checks, zero violations.
4. Pre-commit eight of eight on the burndown worktree.
5. Commit message cites institutional-rigor § 6 (S3 helper closes silent-failure pattern) and § 3 (refactor when pattern of bugs appears) and the iter 178 audit-gate rule.
