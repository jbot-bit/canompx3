---
mode: IMPLEMENTATION
slug: bracket-verification-flag
task: Add has_queryable_bracket_legs() to BrokerRouter so the session_orchestrator caller can distinguish "broker has separately-queryable bracket leg orders" (ProjectX) from "broker uses native atomic server-side brackets with no separate legs" (Rithmic) or "broker has legs but query is not implemented yet" (Tradovate). Eliminates the latent false "BRACKET LEGS MISSING" alarm that would fire for Rithmic and Tradovate when they are activated. Also fix the orphaned rithmic/auth.py:68 pyright import error with a targeted type-ignore.
created: 2026-04-07
updated: 2026-04-07
stage: 1
of: 1
scope_lock:
  - trading_app/live/broker_base.py
  - trading_app/live/projectx/order_router.py
  - trading_app/live/rithmic/order_router.py
  - trading_app/live/rithmic/auth.py
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/copy_order_router.py
  - trading_app/live/broker_dispatcher.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_copy_order_router.py
  - tests/test_trading_app/test_session_orchestrator.py  # mid-stage scope expansion: FakeBracketRouter fixture needs has_queryable_bracket_legs() stub for the new interface
blast_radius: Adds one new method `has_queryable_bracket_legs() -> bool` to BrokerRouter with default True (conservative). Rithmic overrides to False (atomic server-side brackets — no separate SL/TP order IDs to query). Tradovate overrides to False with TODO comment (placeOSO creates child orders that WOULD be queryable but verify implementation is deferred until activation). ProjectX gets explicit True override for clarity. CopyOrderRouter and broker_dispatcher delegate to primary. session_orchestrator.py:1683-1713 wraps the verify_bracket_legs call in `if self.order_router.has_queryable_bracket_legs():` — when False, skip verification entirely (no alarm, empty bracket_order_ids list is fine for cancel paths since native-bracket brokers don't need explicit client-side leg cancels). Zero runtime change for the currently-active TopStep+CopyOrderRouter+ProjectX path (primary.has_queryable_bracket_legs() = True, verify still runs). Also adds `# pyright: ignore[reportMissingImports]` to rithmic/auth.py:68 where `from async_rithmic import ...` lives (third-party package with no type stubs, lazy import inside _ensure_connected so not loaded at module import time).
---

# Stage: Bracket Verification Flag + Rithmic Pyright Stub

## Purpose

Close two items flagged as "deferred, not blocking" at the end of the live-trading-bug-sweep stage:

1. **(ITEM 1 — design fix)** The `verify_bracket_legs` silent-default regression from commit 73178a8 was patched for CopyOrderRouter in commit cb4abb9 (delegates to primary). But the underlying design flaw remains: BrokerRouter's default `(None, None)` return is interpreted as "BRACKET LEGS MISSING" critical alarm by session_orchestrator.py:1699-1711. This is wrong for any broker whose native brackets do NOT have separately-queryable leg orders — Rithmic (atomic server-side) and Tradovate (we have not implemented the query yet).

   Today, the latent bug is dormant because Rithmic and Tradovate are inactive in production. But both are in the action queue — Rithmic activation specifically flagged in MEMORY.md as durable-scaling lane #5. Activating either broker would cause the false critical alarm to fire on every entry, drowning the operator in false Telegram alerts.

2. **(ITEM 2 — pyright cleanup)** `rithmic/auth.py:68` has a `from async_rithmic import OrderPlacement, RithmicClient, SysInfraType` lazy import that pyright flags as `reportMissingImports` because the package has no type stubs and may not be installed in dev environments. Adding `# pyright: ignore[reportMissingImports]` with a comment explaining the lazy-import rationale closes the last pyright error in the stage scope from the prior stage.

## Design for Item 1

### The root question
"Can the session_orchestrator trust that `verify_bracket_legs` returning `(None, None)` means something actionable?"

Current answer (post-73178a8): NO — it could mean "broker has atomic brackets" OR "verify failed" OR "missing legs". The caller conflates all three into a critical alarm.

### The clean fix
Add a BrokerRouter method that declares whether bracket legs are queryable as separate orders:

```python
def has_queryable_bracket_legs(self) -> bool:
    """True when this broker's brackets are distinct, separately-queryable orders
    (e.g. ProjectX AutoBracket creates entry_id+1 for SL, entry_id+2 for TP).
    False when brackets are atomic with the entry and managed server-side
    (e.g. Rithmic native brackets have no separately-queryable leg orders).

    When False, session_orchestrator skips verify_bracket_legs entirely —
    the broker manages bracket cancellation on exit order submission.
    """
    return True  # conservative default: assume verifiable
```

### Per-broker override matrix

| Broker | Override | Reason |
|---|---|---|
| ProjectX | explicit `True` | AutoBracket creates separate child orders (entry_id+1 for SL, entry_id+2 for TP) — fully queryable. Matches current behavior. |
| Rithmic | `False` | Native server-side brackets are atomic with the entry submission. No separate SL/TP order IDs exist to query. (None, None) is semantically correct but the caller must skip the check. |
| Tradovate | `False` WITH TODO | placeOSO creates bracket1/bracket2 child orders that SHOULD be queryable via the same API. But we have not implemented the query yet. False + TODO comment prevents false alarms today; when Tradovate activation is imminent we flip to True and implement the query. |
| CopyOrderRouter | delegate to primary | Same pattern as supports_native_brackets at L129-131 |
| broker_dispatcher | delegate to primary | Same pattern |

### Session orchestrator caller update

Current (session_orchestrator.py:1683-1713):
```python
if _bracket_merged:
    record = self._positions.get(event.strategy_id)
    if record is not None and order_id:
        try:
            sl_id, tp_id = self.order_router.verify_bracket_legs(...)
            if sl_id and tp_id:
                record.bracket_order_ids = [sl_id, tp_id]
                log.info("BRACKET VERIFIED: ...")
            else:
                log.critical("BRACKET LEGS MISSING ...")  # ← FALSE ALARM for Rithmic/Tradovate
                ...
        except Exception as e:
            log.error(...)
            record.bracket_order_ids = [order_id + 1, order_id + 2]  # sequential fallback
```

After:
```python
if _bracket_merged:
    record = self._positions.get(event.strategy_id)
    if record is not None and order_id:
        if not self.order_router.has_queryable_bracket_legs():
            # Native atomic brackets (e.g. Rithmic) — no separately-queryable legs.
            # Broker manages cancellation on exit order submission.
            log.debug("Native atomic brackets for %s — no leg IDs to track", event.strategy_id)
        else:
            try:
                sl_id, tp_id = self.order_router.verify_bracket_legs(...)
                if sl_id and tp_id:
                    record.bracket_order_ids = [sl_id, tp_id]
                    log.info("BRACKET VERIFIED: ...")
                else:
                    log.critical("BRACKET LEGS MISSING ...")
                    ...
            except Exception as e:
                log.error(...)
                record.bracket_order_ids = [order_id + 1, order_id + 2]
```

### Test coverage

1. `test_copy_order_router.py` — add `test_has_queryable_bracket_legs_delegates_to_primary` (mirrors existing supports_native_brackets delegation test at L205-210)
2. ProjectX/Rithmic/Tradovate order_router test files — verify the per-broker return value matches expectation

Session orchestrator test coverage for the new skip path is deferred — no test fixture currently exercises _bracket_merged=True with has_queryable_bracket_legs=False, and adding one would require constructing a second FakeBrokerComponents variant. The new code path is proven by direct reading and the per-broker unit test.

## Design for Item 2

Single-line addition at `rithmic/auth.py:68`:

```python
# async_rithmic is a third-party package with no type stubs and is
# intentionally lazy-imported inside _ensure_connected() to avoid
# import errors when users aren't running the Rithmic adapter.
from async_rithmic import OrderPlacement, RithmicClient, SysInfraType  # pyright: ignore[reportMissingImports]
```

No code behavior change. Only silences the one remaining pyright error in the live trading stack scope.

## Acceptance Criteria

1. `BrokerRouter.has_queryable_bracket_legs()` exists with default True and docstring.
2. ProjectX, Rithmic, Tradovate, CopyOrderRouter, broker_dispatcher all have correct overrides/delegations.
3. session_orchestrator.py:1683 wraps verify_bracket_legs in the new flag check.
4. test_copy_order_router.py::test_has_queryable_bracket_legs_delegates_to_primary PASS.
5. Full tests/test_trading_app/ suite: 2194+ passed, 0 failures.
6. Pyright trading_app/live/: 0 errors in session_orchestrator.py, rithmic/, tradovate/, copy_order_router.py, broker_base.py (may still show pre-existing errors in bar_aggregator, bot_dashboard, projectx/auth, projectx/data_feed — out of scope).
7. Drift check: same as before (77 PASS / 1 pre-existing Check 57 FAIL for MGC daily_features — deferred to e2 worktree).
8. No behavior change in active TopStep+CopyOrderRouter+ProjectX path (ProjectX returns True → verify runs as before → cb4abb9 delegate still works correctly).

## Scope discipline

- NOT touching session_orchestrator.py beyond the narrow wrap at L1683.
- NOT implementing Tradovate's real verify_bracket_legs query (deferred to activation stage).
- NOT changing the CopyOrderRouter.verify_bracket_legs delegate from cb4abb9 (still needed for ProjectX path).
- NOT touching session_orchestrator.py test fixture (would drag in scope creep).
- NOT attempting to fix Check 57 (MGC daily_features) — that's e2 worktree's scope.

## Commit plan

1. `feat(broker_base): add has_queryable_bracket_legs() flag + overrides per broker`
2. `fix(session_orch): skip bracket leg verification for brokers with atomic native brackets`
3. `test(copy_router): delegation test for has_queryable_bracket_legs`
4. `fix(rithmic/auth): pyright ignore for lazy async_rithmic import`

Possibly collapsed into 1-2 commits depending on review convenience.
