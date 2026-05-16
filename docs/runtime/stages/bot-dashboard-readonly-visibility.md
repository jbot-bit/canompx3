---
task: "Bot dashboard read-only operator visibility — surface broker auth, bracket/fill_poller probes, fill-poll counters, last order/fill, slippage placeholder"
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_state.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_bot_dashboard.py
---

## Context

Monday live debut needs operator visibility into broker-edge health WITHOUT touching execution. Phase 1 (preflight reliability — bracket/fill_poller probes + safeguard narrowing) is in-flight on `main` working tree. Phase 2 (this stage) was non-goal'd from Phase 1 and is now in its own worktree off `origin/main`. Read-only contract: the snapshot writer produces JSON; the dashboard reads it; no new strategy decisions, no execution changes, no broker calls from the dashboard process.

## Blast Radius

- `trading_app/live/session_orchestrator.py` — add a private `_write_live_health_snapshot()` method called from the existing main loop tick (NOT from a new background task — reuses an already-running cadence) AND once at end of preflight. Snapshot fields:
  - `auth_healthy` (bool) — from broker_components ProjectXAuth `.is_healthy` property; `None` if components missing/unknown
  - `brackets_probe` / `fill_poller_probe` (bool) — last preflight `results["brackets"]` / `results["fill_poller"]`
  - `fill_polls_run`, `fill_polls_confirmed`, `fill_polls_failed` — copied from `self._stats` (already in-memory at session_orchestrator.py:280-282)
  - `last_order_status` (str | None) — most recent broker response status if tracked; else `None`
  - `last_fill_price` (float | None) — from active PositionRecord.fill_entry_price OR most recent live_trades row
  - `realized_slippage_pts` (float | None) — only emitted if a non-null value exists in current session's TradeRecord; else omit key entirely (placeholder rule)
  - `snapshot_ts_utc` (ISO-8601 string)
  - `broker_status` (str: "unknown" | "ok" | "degraded") — fail-closed: `"unknown"` when components is None
  - Writer is best-effort (catch + log.warning, never raise into trading loop)
- `trading_app/live/bot_dashboard.py` — extend `_build_operator_payload()` to call a new helper `_read_live_health_snapshot()` that reads `runtime/state/live_health.json` (path resolved relative to project root via existing `paths` module convention). On `FileNotFoundError` / `json.JSONDecodeError` / missing keys → payload gets `live_health: {"status": "unknown", "reason": "<reason>"}` (fail-closed). On success → payload gets `live_health: {...validated fields...}`.
- `trading_app/live/bot_state.py` — NO write changes. Read-only: if dashboard needs to surface `live_health` next to other state, the new key is additive in the operator payload, not in `bot_state.json`. Keeps the two state files semantically separate (bot_state = mode/lanes; live_health = broker-edge health).
- HTML rendering: existing `bot_dashboard.html` already consumes payload keys via the operator-state render function. Adding new keys is JS-additive in a follow-up; THIS STAGE only guarantees the Python payload contains the keys. (HTML JS is outside scope_lock to keep diff bounded; static rendering proves the data, dashboard team can pretty-print after.)
- Reads: `runtime/state/live_health.json` (new file written by orchestrator). Writes: same file. No DB, no schema, no canonical-source changes. No broker API calls from dashboard process.
- Capital-class risk: NONE. Writer is best-effort + caught; failure to snapshot does not affect order routing, risk, or fill polling. Reader is fail-closed (broker status = "unknown" surfaces to operator UI).

## Non-goals (explicit)

- HTML/JS dashboard rendering of the new card (follow-up stage; payload is the contract here).
- Persisting `live_health.json` history (this snapshot is current-tick-only, overwrite-on-write).
- Any change to fill-polling cadence, broker reconnect logic, or kill-switch behavior.
- Any change to `bot_state.json` schema.
- Any change to Phase 1 preflight files (`scripts/run_live_session.py`) — those edits live on the main worktree.

## Verification

1. `python pipeline/check_drift.py` from the worktree root — all gates pass.
2. `pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_bot_dashboard.py -x -q` — all pass, including new tests:
   - `test_live_health_snapshot_writes_expected_keys`
   - `test_live_health_snapshot_fail_closed_on_missing_components`
   - `test_dashboard_payload_includes_live_health_when_file_present`
   - `test_dashboard_payload_live_health_unknown_when_file_missing`
   - `test_dashboard_payload_live_health_unknown_when_file_corrupt`
3. Manual: write a fixture `runtime/state/live_health.json` with known values, call `_build_operator_payload("default")` in a REPL, confirm `payload["live_health"]` matches fixture.
4. Branch is `session/joshd-bot-dashboard-readonly` off `origin/main`. Worktree at `.worktrees/bot-dashboard-readonly` (locked). Phase 1 main-worktree state untouched.
