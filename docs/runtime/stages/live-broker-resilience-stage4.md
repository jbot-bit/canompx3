---
task: Live-broker resilience hardening — Stage 4 (circuit-breaker wiring + operator surface)
mode: IMPLEMENTATION
stage: 4
total_stages: 5
worktree: C:/Users/joshd/canompx3/.worktrees/live-broker-resilience
branch: feat/live-broker-resilience
---

## Scope Lock

- trading_app/live/circuit_breaker.py
- trading_app/live/broker_base.py
- trading_app/live/projectx/auth.py
- trading_app/live/projectx/order_router.py
- trading_app/live/projectx/positions.py
- trading_app/live/projectx/contract_resolver.py
- trading_app/live/tradovate/http.py
- trading_app/live/session_orchestrator.py
- trading_app/live/bot_state.py
- tests/test_trading_app/test_circuit_breaker.py
- tests/test_trading_app/test_http_client.py
- tests/test_trading_app/test_orchestrator_circuit_wiring.py
- tests/test_trading_app/test_session_orchestrator.py
- docs/runtime/stages/live-broker-resilience-stage4.md

## Blast Radius

- `circuit_breaker.py` — `record_failure()` gains optional `error_class: str | None = None` (back-compat default). New `last_error_class` field surfaces the most-recent failure class for the operator dashboard. Existing call sites (`session_orchestrator.py:2111`, `:2428`) keep working without modification.
- `broker_base.py` — `BrokerAuth` ABC gains a concrete `failure_hook` attribute (defaults to `None`) on a `__init__` that subclasses can call via `super().__init__()`. Adapters that build their `BrokerHTTPClient` lazily read this attribute to wire the breaker. No method-signature changes.
- `projectx/{auth,order_router,positions,contract_resolver}.py` — each `BrokerHTTPClient(...)` construction site reads `getattr(auth, "failure_hook", None)` and passes it as `failure_hook=` when non-None. Auth itself reads `self.failure_hook` (post-`super().__init__()`) at the same point.
- `tradovate/http.py` — module-level `client = BrokerHTTPClient(base_url="", name="tradovate")` factory function refactored to accept an optional `failure_hook` argument so tradovate adapter can be wired the same way (mirrors projectx pattern; future-ready).
- `session_orchestrator.py` — `_circuit_breaker` construction MOVES from line 858 to before component construction (~line 322); `self.auth.failure_hook = self._circuit_breaker` is set before `contracts_cls(...)`, `router_cls(...)`, `positions_cls(...)` so they see the wired hook at their `__init__`. `_broker_status_payload()` gains `circuit_open`, `consecutive_failures`, `last_error_class` fields.
- `bot_state.py` — no production code change; only consumes the broker_status payload as today. The new fields ride through `_sanitize_for_state` without modification (all JSON-safe primitives).
- Tests: `test_circuit_breaker.py` gains 4 tests covering the new arg back-compat, the last_error_class tracking, the default None preserving prior semantics, and the broker_status fields surfaced via the dataclass. `test_http_client.py` gains 1 test asserting the client passes the error class through to a custom hook. `test_orchestrator_circuit_wiring.py` is a new file verifying the orchestrator wires `auth.failure_hook = self._circuit_breaker` before component construction.
- Reads: no broker API changes; no DB changes. Writes: bot_state.json gains three new fields (additive, dashboard-tolerant).
- Not touched: pipeline/, trading_app/risk_manager.py, ExecutionEngine, the actual order-submit path semantics.

## Why this stage

Stages 1+2 introduced `BrokerHTTPClient.failure_hook` as the per-HTTP-call observability point. Stage 4 closes the wiring gap: today the only failure hook is `_NoopFailureHook` because no component passes one in. Meanwhile, the orchestrator's `_circuit_breaker` only sees `record_failure()` at the *exit-submit* layer (`:2111`) and the *entry-submit* layer (`:2428`) — both far downstream of the actual HTTP failure. By the time those record, the broker has already returned five consecutive 5xxs to the client retry loop.

Wiring `BrokerHTTPClient.failure_hook → orchestrator._circuit_breaker` means the breaker opens on classified HTTP failures at the moment they happen, not five trades later. The dashboard then surfaces `circuit_open=true` to the operator before any order is even attempted on a degraded broker.

## Done criteria

- `CircuitBreaker.record_failure()` accepts optional `error_class: str | None = None`. Old call sites pass no arg and still work (verified by `test_circuit_breaker.py`).
- `CircuitBreaker.last_error_class: str | None` field tracks most-recent error_class (None until first classified failure; resets to None on `record_success`).
- Every `BrokerHTTPClient(...)` construction in `trading_app/live/projectx/` and `trading_app/live/tradovate/` reads `getattr(auth, "failure_hook", None)` and threads it through as `failure_hook=`.
- `BrokerAuth.__init__()` exists and sets `self.failure_hook: Any = None`; all concrete auth subclasses call `super().__init__()`.
- `SessionOrchestrator.__init__()` constructs `self._circuit_breaker` BEFORE component construction, sets `self.auth.failure_hook = self._circuit_breaker` BEFORE `contracts_cls(...)`, and the previous line-858 construction is removed (single-source).
- `_broker_status_payload()` returns three new fields: `circuit_open: bool`, `consecutive_failures: int`, `last_error_class: str | None`.
- `pipeline/check_drift.py` green.
- Targeted pytest green: test_circuit_breaker, test_http_client, test_session_orchestrator, and the new test_orchestrator_circuit_wiring.
- Self-review pass before commit.
