---
task: "START_BOT Stages 3/4/5 — V3 assert→raise, V5 shutdown clear_state, Gate #13 account routing fix"
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
blast_radius: |
  bot_dashboard.py — 3 surgical edits:
  (1) line 85: LIVE_PILOT_ACCOUNT_ID 23055112 → 21944866 (comment sync)
  (2) lines 2972-2975: assert → raise RuntimeError (live path only, never executed on signal/demo)
  (3) line 3000-3001: dead ternary → simple assignment (account_id is not None guaranteed by raise above)
  (4) lines 196-ish: clear_state() call in _lifespan shutdown after child-process loop
  Callers: START_BOT.bat already uses 21944866 (no change needed).
  Tests: test_trading_app/ — run after. No schema changes. No pipeline/ touch.
---

## Stage 0 — Repro / ground truth

### V3 assert→raise
- `bot_dashboard.py:2972-2975`: `assert account_id is not None` — asserts strip under Python `-O`; live path needs guaranteed enforcement.
- `bot_dashboard.py:3000-3001`: ternary `(account_id if account_id is not None else LIVE_PILOT_ACCOUNT_ID) if mode == "live" else None` — after the assert/raise above, `account_id` cannot be None on the live path; the fallback arm is dead.

### Gate #13 account routing
- `START_BOT.bat` uses `21944866` (EXPRESS) on lines 125 and 184 — correct per operator intent.
- `bot_dashboard.py:85` has `LIVE_PILOT_ACCOUNT_ID = 23055112` (50K Combine) as zero-arg fallback.
- Operator confirmed: `21944866` (EXPRESS-V2) is the intended live account. Fix the constant.

### V5 shutdown clear_state
- `_lifespan` shutdown (lines 158-196) cancels SSE watchers and terminates child processes but never calls `clear_state()`.
- `clear_state` is already imported and used in two other lifespan paths (line 135, 1652).
- Add guarded call after the child-process loop (line 196).

## Scope Lock
- `trading_app/live/bot_dashboard.py` only
