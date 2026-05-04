---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ProjectX Bracket Order Design

**Date:** 2026-03-16
**Status:** Approved (4TP auto-proceed)
**Scope:** Atomic bracket orders for E2 ORB trading via ProjectX API

## Problem

`build_bracket_spec()` returns `{"orders": [...]}` — a format that doesn't exist in the ProjectX API.
Broker-side brackets have been **silently failing**: `_submit_bracket()` catches the API error, logs a warning,
and positions fall back to software-managed exits. This leaves positions unprotected against crashes.

## Solution

Attach `stopLossBracket` and `takeProfitBracket` fields to the entry order payload.
The exchange creates bracket child orders atomically when the entry fills.

### API Format (from official ProjectX docs)

```json
{
  "accountId": 123,
  "contractId": "CON.F.US.MGC.M26",
  "type": 4,
  "side": 0,
  "size": 1,
  "stopPrice": 2950.0,
  "stopLossBracket": {"ticks": 100, "type": 4},
  "takeProfitBracket": {"ticks": 200, "type": 1}
}
```

- `ticks` = integer offset from fill price in minimum price increments
- Exchange creates OCO bracket (stop + limit target) when entry fills
- Single `/api/Order/place` call — same endpoint, added fields

### Tick Size Source

`pipeline.cost_model.COST_SPECS[instrument].tick_size` — canonical source.

| Instrument | tick_size | Example: 10pt risk → ticks |
|------------|-----------|---------------------------|
| MGC | 0.10 | 100 ticks |
| MNQ | 0.25 | 40 ticks |
| MES | 0.25 | 40 ticks |
| M2K | 0.10 | 100 ticks |

### Architecture

```
BEFORE (broken):
  submit(entry) → fill → build_bracket_spec() → submit(bracket) → API 400 → silent fail

AFTER (atomic):
  build_entry_spec → build_bracket_spec → merge_bracket_into_entry → submit(combined)
  → exchange creates entry + OCO bracket atomically
```

## Files Changed

| File | Change |
|------|--------|
| `trading_app/live/broker_base.py` | Add `merge_bracket_into_entry()` default (3 lines) |
| `trading_app/live/projectx/order_router.py` | Fix bracket spec (ticks), add merge, tick_size in __init__ |
| `trading_app/live/session_orchestrator.py` | Build bracket before entry submit, merge, skip post-fill bracket |
| `tests/test_trading_app/test_projectx_router.py` | 6 new tests |

## Risks

- **Tick size mismatch:** Mitigated by canonical COST_SPECS
- **API rejects bracket fields:** Confirmed in official docs; test with demo post-deploy
- **Bracket child IDs:** Store parent orderId; cancel parent → cancels children
- **Backward compat:** Tradovate `supports_native_brackets() == False` → merge is no-op

## Failure Modes

- Combined submit fails → no entry, no bracket → **fail-closed** ✓
- Bracket fields ignored → same as current (software exits work) → no regression
- Tick off by 1 → 0.10-0.25 pts error → negligible
