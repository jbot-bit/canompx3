---
task: Live-broker resilience hardening — institutional baseline (Stages 1+2 in this PR)
mode: CLOSED
stage: 1
total_stages: 5
worktree: C:/Users/joshd/canompx3/.worktrees/live-broker-resilience
branch: feat/live-broker-resilience
closed_note: |
  Full staged chain is on main via `77b3feb1` (`feat(live): broker HTTP
  resilience + circuit breaker + kill-switch SLA (Stages 1-5) (#301)`), with
  Stage 3/4 docs also closed after the 2026-05-23 Tradovate equity-age follow-up.
  Current 2026-05-24 verification: `./.venv-wsl/bin/python -m pytest
  tests/test_trading_app/test_http_client.py
  tests/test_trading_app/test_order_idempotency.py
  tests/test_trading_app/test_equity_age_watchdog.py
  tests/test_trading_app/test_orchestrator_circuit_wiring.py
  tests/test_trading_app/test_circuit_breaker.py
  tests/test_trading_app/test_projectx_positions.py
  tests/test_trading_app/test_tradovate.py -q` => 118 passed, 1 warning
  (`asyncio` executor join warning in the stale-equity watchdog test).
---

## Scope Lock

- trading_app/live/http_client.py
- trading_app/live/projectx/auth.py
- trading_app/live/projectx/order_router.py
- trading_app/live/projectx/contract_resolver.py
- trading_app/live/projectx/positions.py
- trading_app/live/tradovate/http.py
- trading_app/live/bot_dashboard.py
- trading_app/live/trade_journal.py
- pipeline/migrations/
- tests/test_trading_app/test_http_client.py
- tests/test_trading_app/test_order_idempotency.py
- docs/runtime/stages/live-broker-resilience.md

## Blast Radius

- trading_app/live/http_client.py — NEW canonical HTTP client. Replaces 5 bespoke retry implementations.
- trading_app/live/projectx/auth.py — _login_with_retry collapses into client; 401 refresh path added.
- trading_app/live/projectx/order_router.py — places orders. Adds client_order_id (idempotency key); migrates to client's classifier. Highest blast radius file in repo — duplicate-order bug class lives here.
- trading_app/live/projectx/contract_resolver.py — idempotent read, wrap in READ_POLICY.
- trading_app/live/projectx/positions.py — query_equity must STOP returning silent None on transient failure. Returns last-good + age or raises typed error.
- trading_app/live/tradovate/http.py — collapse into client (thin wrapper preserved).
- trading_app/live/bot_dashboard.py — _fetch_accounts_for_connection uses READ_POLICY with deadline. (Dashboard equity-fetch is what failed tonight.)
- trading_app/live/trade_journal.py — add client_order_id column. Versioned migration in pipeline/migrations/.
- Reads: live broker APIs (read-only side effects). Writes: trade_journal schema migration (idempotent), order placement (idempotent via client_order_id).
- Not touched in this stage: session_orchestrator, circuit_breaker (later stages), Rithmic (deferred), risk_manager, sizing.

## Why this stage

Tonight's TopStepX equity-fetch surfaced two failures in 2 min — read-timeout + TCP RST — both unhandled. No retry on equity/positions; no idempotency keys on orders means a retried place after broker-side accept = duplicate contract. Stage 1+2 close both gaps in one mergeable unit; splitting would create a window where retries exist without idempotency keys (strictly worse).

## Done criteria

- BrokerHTTPClient covers A-G classifier with bounded backoff and deadline propagation.
- All listed call sites migrated; no `requests.post(` or `requests.get(` to broker endpoints outside the client.
- Order placement carries client_order_id; trade_journal persists it BEFORE the HTTP call.
- Reconcile path via /Order/searchOpen on retry.
- New tests: test_http_client.py (8 cases minimum), test_order_idempotency.py (4 scenarios).
- pipeline/check_drift.py green.
- Targeted pytest green.
