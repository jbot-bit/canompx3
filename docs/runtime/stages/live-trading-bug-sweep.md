---
mode: IMPLEMENTATION
slug: live-trading-bug-sweep
task: Fix real runtime bugs in live trading path — Windows zombie PID detection, type protocol hygiene for Rithmic/Tradovate subclass attribute access, session_orchestrator signal_only None guards
created: 2026-04-07
updated: 2026-04-07
stage: 1
of: 1
scope_lock:
  - trading_app/live/instance_lock.py
  - trading_app/live/broker_base.py
  - trading_app/live/rithmic/auth.py
  - trading_app/live/rithmic/order_router.py
  - trading_app/live/rithmic/contracts.py
  - trading_app/live/rithmic/positions.py
  - trading_app/live/tradovate/auth.py
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_instance_lock.py
blast_radius: (1) instance_lock._is_pid_alive currently treats zombie Windows PIDs as alive because OpenProcess returns a valid handle for exited processes (verified: PID 4188 returns handle with exit_code=0 != STILL_ACTIVE=259). Real runtime bug — bot refuses to restart after crash until lock file manually deleted. Fix adds GetExitCodeProcess check. Test added covering the zombie case. (2) BrokerAuth base type in rithmic/* files needs concrete RithmicAuth annotation so self.auth.client / self.auth.run_async type-checks. Runtime unchanged (the attribute access already works because concrete auth has these), just type hygiene. (3) Same pattern for tradovate/order_router.py:49 accessing self.auth.base_url. (4) session_orchestrator signal_only=True sets order_router=None; 15 pyright errors flag None.submit/etc in live-only methods. Fix: add explicit assertion at entry of live-only methods — raises clear error if accidentally called in signal mode (currently crashes with AttributeError deep inside). Blast radius zero for working code paths (assertion always passes when called correctly).
---

# Stage: Live Trading Bug Sweep

## Purpose

Three real quality issues in the live trading path:

1. **BUG (HIGH):** `instance_lock._is_pid_alive` on Windows returns True for zombie PIDs. After bot crash, the operator cannot restart the bot without manually deleting the lock file. Verified: `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, 4188)` returns handle 336 with `GetExitCodeProcess → 0` (not `STILL_ACTIVE=259`).

2. **TYPE HYGIENE (MED):** Rithmic and Tradovate order routers, contracts, and positions modules access subclass-specific attributes (`client`, `run_async`, `base_url`) through the `BrokerAuth` base type. 17 pyright errors. Runtime-safe because the auth object IS the concrete subclass, but the type annotations lie about the type, which means pyright can't catch real bugs in these files.

3. **DEFENSIVE (LOW):** `session_orchestrator.order_router` is `None` in `signal_only=True` mode. 15 pyright errors flag `.submit`, `.build_order_spec`, etc. on the potentially-None router. Currently protected by control flow but AttributeError on violation is a cryptic "NoneType has no attribute submit" deep in a retry loop. Fix: assertion at method entry with clear message.

## Files touched

1. `trading_app/live/instance_lock.py` — add `GetExitCodeProcess` check to `_is_pid_alive` on Windows
2. `tests/test_trading_app/test_instance_lock.py` — add test covering the zombie case
3. `trading_app/live/rithmic/order_router.py` — annotate `self.auth` as concrete `RithmicAuth`
4. `trading_app/live/rithmic/contracts.py` — same
5. `trading_app/live/rithmic/positions.py` — same
6. `trading_app/live/tradovate/order_router.py` — annotate `self.auth` as concrete `TradovateAuth`
7. `trading_app/live/session_orchestrator.py` — add assertion guards on `order_router` in live-only methods
8. `trading_app/live/broker_base.py` — only if needed for protocol hygiene (likely NOT touched)

## Acceptance Criteria

1. **Zombie PID fix verified:** manually injected test creates a lock file with a dead PID, `_is_pid_alive(dead_pid)` returns False.
2. **test_instance_lock.py full suite passes** including new zombie test.
3. **Full trading_app tests pass** — zero regressions.
4. **Pyright errors reduced:** rithmic/* errors drop from 17 to 0. session_orchestrator None.attr errors drop from 15 to 0. Tradovate base_url error drops to 0. Net: 57 → ~5 remaining (truly unavoidable false positives).
5. **Drift check passes** (77/77).
6. **No changes to file paths in other terminal's scope_lock** (canonical-filter-self-description) — verified by git diff after commit.
7. **No behavior change in healthy bot operation** — assertions are no-ops when `signal_only=False`.

## Canonical discipline

- No changes to `config.py`, `eligibility/*`, `pipeline/check_drift.py` (other terminal's scope).
- No new abstractions — use precise type annotations, not protocol hierarchy changes.
- `BrokerAuth` base class stays unchanged. Concrete classes annotated directly in consumer modules.

## Commit plan

1. `fix(instance_lock): Windows zombie PID detection — check GetExitCodeProcess`
2. `fix(rithmic): annotate auth as RithmicAuth for type-safe client/run_async access`
3. `fix(tradovate): annotate auth as TradovateAuth for type-safe base_url access`
4. `fix(session_orchestrator): assert order_router non-None in live-only methods`

Each commit must pass drift check + relevant tests before moving on.
