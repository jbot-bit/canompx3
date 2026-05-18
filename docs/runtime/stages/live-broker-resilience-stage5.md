---
task: Live-broker resilience hardening — Stage 5 (drift check + doctrine + base-class fail-open cleanup)
mode: IMPLEMENTATION
stage: 5
total_stages: 5
worktree: C:/Users/joshd/canompx3/.worktrees/live-broker-resilience
branch: feat/live-broker-resilience
---

## Scope Lock

- pipeline/check_drift.py
- trading_app/live/broker_base.py
- trading_app/live/session_orchestrator.py
- trading_app/live/tradovate/auth.py
- docs/runtime/decision-ledger.md
- tests/test_pipeline/test_check_drift_broker_endpoints.py
- tests/test_trading_app/test_broker_base.py
- docs/runtime/stages/live-broker-resilience-stage5.md

## Blast Radius

- `pipeline/check_drift.py` — adds `check_no_direct_requests_to_broker_endpoints(trading_app_dir)`. Scans `trading_app/live/projectx/` and `trading_app/live/tradovate/` for `requests.{get,post,request,put,delete,patch}(` patterns. Allows the canonical client (`trading_app/live/http_client.py`) and explicit allowlisted lines. Registered as Check #156. Idempotent file-system scan; zero DB dependency.
- `trading_app/live/broker_base.py` — `BrokerPositions.query_equity_with_age(account_id)` default returns `EquityReading(value=None, age_s=0.0, source="missing")`. Imports `EquityReading` from `http_client` (existing dataclass, used by ProjectXPositions). The base default closes the Stage 3 fail-open gap: adapters that do not override (Tradovate, Rithmic) now return a typed "missing" reading instead of being absent from the API. The orchestrator's SLA gate behavior is preserved — a `source="missing"` reading is treated identically to "no query_equity_with_age method" today (no kill switch fires because `age_s=0.0` is below SLA).
- `trading_app/live/session_orchestrator.py` — `_broker_equity_stale()` removes the `getattr(self.positions, "query_equity_with_age", None)` ducktype check and calls the base-class default directly. Still short-circuits on `source="missing"` (the result for adapters without an override) — institutionally equivalent to the prior fail-open semantics but routed through the typed API rather than `hasattr`.
- `trading_app/live/tradovate/auth.py` — `_login` and `_renew_or_login` migrated from direct `requests.post(...)` to `BrokerHTTPClient.post_json(...)`. Mirrors `projectx/auth.py` pattern (Stage 1+2 baseline). Two call sites; each becomes ~5 lines instead of ~13. Caught by the new drift check on first run — pre-existing gap that Stage 5 closes for free.
- `docs/runtime/decision-ledger.md` — appends entry `live-broker-http-client-canonical-2026-05-18` naming `trading_app/live/http_client.py` as the single sanctioned HTTP surface for broker endpoints, with the drift check (#156) as the enforcement mechanism.
- Tests:
  - `tests/test_pipeline/test_check_drift_broker_endpoints.py` (NEW) — covers: clean codebase passes; injected `requests.post(...)` in `projectx/order_router.py` fails; allowlist exempts `trading_app/live/http_client.py`; allowlist exempts non-broker modules. Mutation-proof per institutional-rigor § 11.
  - `tests/test_trading_app/test_broker_base.py` — extends with test verifying `BrokerPositions.query_equity_with_age()` default returns `EquityReading(value=None, age_s=0.0, source="missing")`.
- Reads: filesystem scan of `trading_app/live/projectx/` and `trading_app/live/tradovate/`. No DB, no broker API.
- Not touched: production HTTP call paths, circuit breaker, http_client.py itself, no migration.

## Why this stage

Stages 1-4 introduced a canonical HTTP client (Stage 1+2) and wired its failure_hook to the orchestrator's circuit breaker (Stage 4). Without a drift check, the next "let me just curl this real quick" PR would silently re-introduce a bespoke HTTP path that bypasses the retry budget, idempotency keys, AND the circuit breaker. Stage 5 closes the doctrine gap.

Concurrently, the Stage 3 fail-open for adapters without `query_equity_with_age` lives at the orchestrator layer (`getattr(..., None)`). That is an architectural smell — the orchestrator now knows about the adapter-coverage gap. Per institutional-rigor § 4 (delegate to canonical sources), the gap belongs on the base class as a typed default, not at the consumer's `getattr` site.

## Done criteria

DONE (this commit, partial close under context pressure):

- `check_no_direct_requests_to_broker_endpoints` registered in `pipeline/check_drift.py::CHECKS` as Check #156.
- Drift check executes against the current tree and passes (Stages 1+2 migrated every projectx call site; this commit migrates `tradovate/auth.py:_login` and `:_renew_or_login` to `BrokerHTTPClient.post_json` — the new check caught both as pre-existing gaps).
- Mutation-probe DONE via injection: added literal `requests.get("https://example.com/canary")` to `trading_app/live/projectx/positions.py:25`, re-ran drift → caught with full diagnostic message (`positions.py:25: direct requests.get( call — must route through trading_app.live.http_client.BrokerHTTPClient...`); reverted; check returned to PASS. Check 156 has teeth.
- `BrokerPositions.query_equity_with_age()` default returns `EquityReading(value=None, age_s=0.0, source="missing")` via `TYPE_CHECKING` forward ref. `ProjectXPositions.query_equity_with_age` override is unchanged (it provides the real implementation).
- 274/274 targeted tests pass (test_circuit_breaker + test_http_client + test_orchestrator_circuit_wiring + test_equity_age_watchdog + test_broker_base + test_session_orchestrator).

CARRY-OVER (deferred for next session):

- `SessionOrchestrator._broker_equity_stale()` simplification: replace the `getattr(self.positions, "query_equity_with_age", None)` ducktype check with a direct call now that the base class provides the default. Functionally equivalent today (`getattr` finds the inherited default → returns source="missing" → fail-open identical) — purely a code-cleanliness follow-up. Defer because it touches `session_orchestrator.py` which carries adversarial-audit-gate weight.
- `docs/runtime/decision-ledger.md` entry naming `trading_app/live/http_client.py` as the single sanctioned HTTP surface for broker endpoints.
- `tests/test_pipeline/test_check_drift_broker_endpoints.py` (NEW) — formalize the injection probe as a permanent regression test (covers: clean codebase passes; injected `requests.post(...)` fails; allowlist exempts http_client.py; allowlist exempts non-broker modules). Injection probe was executed manually this session; formal test pins it.
- `tests/test_trading_app/test_broker_base.py` extension: verify `BrokerPositions.query_equity_with_age()` default returns `EquityReading(value=None, age_s=0.0, source="missing")`.
