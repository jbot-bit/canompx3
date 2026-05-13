---
task: Implement verify_bracket_legs() on Tradovate order router; flip has_queryable_bracket_legs() to True
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/broker_base.py
  - trading_app/live/projectx/order_router.py
  - trading_app/live/copy_order_router.py
  - tests/test_trading_app/test_tradovate_bracket_legs.py
  - tests/test_trading_app/test_tradovate.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_projectx_router.py
  - tests/test_trading_app/test_copy_order_router.py
---

## Blast Radius

- `trading_app/live/tradovate/order_router.py` — modified: implements `verify_bracket_legs()` and flips `has_queryable_bracket_legs()` to True. Consumer chain: `session_orchestrator.py:2426-2431` dispatches based on the flag; `copy_order_router.py:204-216` already delegates to `self.primary` so fan-out auto-inherits the new path.
- `trading_app/live/session_orchestrator.py` — modified at the bracket-verification exception path (lines 2455-2469): replaces unconditional `[order_id+1, order_id+2]` fallback (a ProjectX-specific sequential-ID convention) with a capability-gated branch reading `supports_sequential_bracket_ids()`. Per adversarial-audit-gate finding: previously the Tradovate `RateLimitExhausted` path would store guessed IDs that don't correspond to real Tradovate orders, since Tradovate API IDs are not sequential from entry. Added to scope mid-stage per institutional-rigor § 2 ("after any fix, review the fix") — auditor verdict was CONDITIONAL pending this change.
- `tests/test_trading_app/test_tradovate_bracket_legs.py` — new unit tests covering the verify_bracket_legs identification contract.
- `tests/test_trading_app/test_tradovate.py` — refreshed stale `has_queryable_bracket_legs_false` test; will add a `supports_sequential_bracket_ids` assertion to confirm Tradovate returns False (default from broker_base).

Reads: none (no DB or live broker calls; tests stub `query_open_orders`).
Writes: none.
Affects: Tradovate broker path + the shared session_orchestrator exception branch. ProjectX/Rithmic behavior preserved via the new capability hook (default False on the base class; ProjectX override returns True). tradeify_50k_type_b activation is OUT OF SCOPE for this stage (Stage 3b).

## Mode

IMPLEMENTATION. No production-code edits outside scope_lock. No `auth.py`, `http.py`, `copy_order_router.py`, `session_orchestrator.py`, or `prop_profiles.py` changes in this stage.

## Acceptance

1. All 4+ unit tests pass with output shown.
2. Mutation proof: locally setting `has_queryable_bracket_legs()` back to False makes the flag test fail; restoring makes it pass. Locally swapping `(stop_id, target_id)` return order makes the identification test fail; restoring makes it pass.
3. `python pipeline/check_drift.py` exits 0.
4. `grep -rn "TODO(tradovate-activation)" trading_app/live/tradovate/` returns 0 lines (was 1).
5. No other files in scope_lock touched. No `git add` outside scope_lock.
6. Adversarial-audit gate dispatched per `.claude/rules/adversarial-audit-gate.md` before Stage 3a.

## Stage 3a — Tradovate F4 emergency-flatten parity tests (test-only)

Adds Tradovate-shaped parity for the six existing ProjectX-shaped emergency-flatten tests. Proves the orchestrator's kill-switch flow is broker-agnostic by exercising the same `_emergency_flatten` path against a `FakeTradovateRouter` double that mirrors `TradovateOrderRouter`'s capability flags (`supports_native_brackets=True`, `has_queryable_bracket_legs=True`, `supports_sequential_bracket_ids=False`) and wire-field shape (`orderQty`, large-int order IDs).

### Stage 3a scope_lock

- `tests/test_trading_app/test_session_orchestrator.py` — added `class FakeTradovateRouter` + `class TestEmergencyFlattenTradovateParity` (6 tests). No edits to existing tests.

No production-code edits. Stage 3a is test-only coverage; the orchestrator and router are unchanged from Stages 1/2.

### Stage 3a acceptance

1. All 6 parity tests pass:
   - `test_emergency_flatten_tradovate_signal_only_logs_manual_close`
   - `test_emergency_flatten_tradovate_retries_on_failure`
   - `test_emergency_flatten_tradovate_all_retries_fail_logs_manual`
   - `test_emergency_flatten_tradovate_uses_correct_qty` (asserts `orderQty=3` flows through `build_exit_spec(qty=record.contracts)`)
   - `test_emergency_flatten_tradovate_multiple_positions`
   - `test_emergency_flatten_tradovate_cancels_bracket_legs_before_exit` (R2-C4 invariant: every `cancel` precedes the first `submit`)
2. Full `tests/test_trading_app/test_session_orchestrator.py` passes (214 tests).
3. Mutation proofs (BOTH required per institutional-rigor § 2):
   - **Test 4** (`test_emergency_flatten_tradovate_uses_correct_qty`): changing `session_orchestrator.py:2704` from `qty=record.contracts` to `qty=1` makes BOTH the ProjectX-shaped test (`TestKillSwitch::test_emergency_flatten_uses_correct_qty`) AND the Tradovate parity test fail with `assert 1 == 3`. This proves the parity test exercises the same orchestrator contract via a different router shape — not a tautology.
   - **Test 6** (`test_emergency_flatten_tradovate_cancels_bracket_legs_before_exit`): commenting out the R2-C4 bracket-cancel block at `session_orchestrator.py:2680-2692` makes only this Tradovate parity test fail with `assert [] == [234567001, 234567002]` (no ProjectX-shaped equivalent exists in the existing suite — this is new coverage, not parity). Production code reverted to clean state after each proof; `git diff` clean.
4. `grep -c "TradovateOrderRouter\|FakeTradovateRouter" tests/test_trading_app/test_session_orchestrator.py` returns ≥6 hits (parity class + fake landed).
5. `python pipeline/check_drift.py` — 124 code-integrity checks pass. **Single FAILED check is Check 64 (daily_features row integrity)**: MGC trading_day `2026-05-11` has 1 row instead of 3 (missing 2 of the 3 aperture rows for orb_minutes ∈ {5, 15, 30}). Root-cause query: `SELECT trading_day, COUNT(*) FROM daily_features WHERE symbol LIKE 'MGC%' GROUP BY trading_day, symbol HAVING COUNT(*) != 3` returns exactly that one row. Confirmed failing on `origin/main` worktree (pre-existing data-state, not introduced by Stage 3a). Likely cause: partial `build_daily_features` run for 2026-05-11 that didn't complete all three apertures. Out of scope for Stage 3a (test-only). Backlog: rebuild that one MGC trading day's daily_features entries for the missing apertures.

### Why a concrete fake, not `Mock(spec=TradovateOrderRouter)`

The plan suggested `Mock(spec=...)`. The implementation uses a concrete `FakeTradovateRouter` to match the file's established convention (`FakeRouter`, `FakeAuth`, `FakePositions` at lines 124-186). Concrete fakes catch attribute typos at access time the same way `spec=` would, and compose cleanly with the autouse `_inline_executor_offloads` fixture without injecting Mock-specific call semantics into the executor inline-call path. Per `.claude/rules/institutional-rigor.md` § 4 (delegate to canonical sources, never re-encode): the fake delegates to the `BrokerRouter` interface shape rather than encoding a parallel mock surface.

Out of scope for Stage 3a (deferred to subsequent stages):

- Stage 3b: activation of `tradeify_50k_type_b` paper-only.
- Stage 4: PR open.
