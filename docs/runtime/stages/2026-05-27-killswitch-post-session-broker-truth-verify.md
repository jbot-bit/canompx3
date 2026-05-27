---
task: Close Gap #2 — kill-switch / post_session position-abandonment race. Replace the unconditional flag-trust EOD-close skip in post_session() with a broker-truth-verified skip using query_open().
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
---

## Blast Radius

- trading_app/live/session_orchestrator.py — restructures the `post_session()` kill-switch skip block (~:3766-3770). One conditional. Reuses `self.positions.query_open(account_id)` (canonical broker-truth read, same pattern as startup orphan check at :670) and `self._positions.active_positions()` (local fallback). No signature change; no change to the no-kill-switch path. Also corrects the misleading `_fire_kill_switch` docstring (:1045).
- tests/test_trading_app/test_session_orchestrator.py — 4 new tests in TestKillSwitch. Existing `:1264` happy-path test must pass unchanged (routes through broker-flat branch — FakePositions returns no orphans).
- Reads: none from gold.db. Writes: none to gold.db. Live order path only when a position is genuinely open at EOD (strictly safer than current abandon-silently behavior).
- Composes with committed fix 99958a51 (no-loop feed-dead _notify); no overlap.
- query_open already exercised at startup (:670); ProjectX/Tradovate/Rithmic impls exist. No new adapter surface.

## Decision logic (when self._kill_switch_fired is True)

1. Signal-only / no order_router / no positions → skip query (no live position possible); skip EOD close as today.
2. Query broker via `self.positions.query_open(account_id)`:
   - broker flat AND local tracker flat → skip EOD close (preserves no-duplicate-close goal; common correct case).
   - broker shows open position → do NOT skip; MANUAL CLOSE REQUIRED _notify + CRITICAL log; run _close_all.
   - NotImplementedError → fall back to local active_positions(); if active → attempt close + alert; else skip.
   - generic Exception → CRITICAL + _notify; do NOT skip (fail-closed).

## Verification
- pytest -k "TestKillSwitch or post_session" all pass incl. 4 new + unchanged :1264.
- Mutation-proof: revert new branch, confirm test_post_session_attempts_close_when_broker_shows_open FAILS, restore.
- python pipeline/check_drift.py → 165/0.
- full orchestrator suite no regression.
