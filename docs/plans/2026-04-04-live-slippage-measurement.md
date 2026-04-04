# Design Plan 2: Live Slippage Measurement System

**Date:** 2026-04-04
**Priority:** HIGH — single biggest unknown for production confidence
**Status:** DESIGN (awaiting implementation)

---

## What & Why

The cost model assumes flat slippage (MNQ: 1 tick/$0.50, MGC: $2.00, MES: $1.25).
MGC tbbo pilot showed actual mean=6.75 ticks vs 1 modeled. MNQ pilot hasn't run yet.
If actual slippage > 2x modeled, ALL backtest ExpR values are overstated.

This system measures, reports, and alerts on actual vs modeled slippage from live trades.

## Existing Infrastructure (Already Built)

| Component | File | What Exists |
|-----------|------|-------------|
| Trade journal schema | `trading_app/live/trade_journal.py:21-46` | `engine_entry`, `fill_entry`, `engine_exit`, `fill_exit`, `slippage_pts` columns |
| Entry slippage calc | `trading_app/live/position_tracker.py:125-156` | `fill_entry - engine_entry` on E2 fill |
| Exit slippage calc | `trading_app/live/session_orchestrator.py:1157-1173` | Ad-hoc `exit_fill - engine_exit` |
| Session multipliers | `pipeline/cost_model.py:180-264` | `SESSION_SLIPPAGE_MULT` dict (live only, not backtest) |
| Fill validation | `session_orchestrator.py:706-734` | `_validate_fill_price()` — NaN/inf/deviation checks |
| Break-even analysis | `scripts/tools/slippage_scenario.py` | Per-lane extra ticks to zero |
| Live journal DB | `live_journal.db` | Separate DuckDB, `live_trades` table |

## Gaps to Fill

### Gap 1: Exit Fill Retrieval (CRITICAL)
- Entry fills have a dedicated poller (E2 stop-market → `update_entry_fill()`)
- Exit fills only captured when broker explicitly returns `fill_exit`
- Market exits (stop/target hits) often record `engine_exit` as actual, leaving `fill_exit` NULL
- **Fix:** Add exit fill polling for ALL exit types (bracket target/stop fills)

### Gap 2: Slippage Aggregation Report (NEW)
- No script aggregates slippage by session, entry_model, instrument, direction
- No comparison of actual vs `SESSION_SLIPPAGE_MULT` predictions
- **Fix:** New `scripts/tools/slippage_report.py`

### Gap 3: Kill Switch Threshold (NEW)
- `cost_model.py:83` defines kill criterion: "actual avg slippage > 2x modeled"
- No automated check enforces this
- **Fix:** Add threshold alert in the slippage report + pre-session check

---

## Implementation Plan

### Stage 1: Exit Fill Polling (session_orchestrator.py)

**Scope:** `trading_app/live/session_orchestrator.py`

After bracket order fills (target or stop), query the broker for actual fill price:

```python
# In _handle_exit_fill() or equivalent exit completion path:
if exit_fill_price is not None:
    self.journal.record_exit(
        trade_id=record.journal_trade_id,
        fill_exit=exit_fill_price,  # ACTUAL broker fill
        engine_exit=event.price,     # MODEL exit
        slippage_pts=entry_slippage + (exit_fill_price - event.price),
        ...
    )
```

For ProjectX: bracket fills already return fill price via `searchOpen` polling.
For Tradovate: exit order status endpoint returns `avgFillPrice`.

**Key constraint:** Fail-open — if fill retrieval fails, log CRITICAL but record `engine_exit` as fallback. Never block trade flow.

### Stage 2: Slippage Report Script (NEW FILE)

**File:** `scripts/tools/slippage_report.py`

```
Usage: python -m scripts.tools.slippage_report [--days 30] [--instrument MNQ]
```

Queries `live_journal.db` → `live_trades` table:

```sql
SELECT
    instrument,
    -- Extract session from strategy_id (e.g., "MNQ_NYSE_OPEN_E2..." → "NYSE_OPEN")
    strategy_id,
    entry_model,
    COUNT(*) as n_trades,
    AVG(fill_entry - engine_entry) as avg_entry_slip_pts,
    AVG(fill_exit - engine_exit) as avg_exit_slip_pts,
    AVG(slippage_pts) as avg_total_slip_pts,
    STDDEV(slippage_pts) as std_slip_pts,
    MAX(ABS(slippage_pts)) as max_slip_pts,
    -- Convert to dollars
    AVG(slippage_pts) * point_value as avg_slip_dollars,
    -- Compare to modeled
    modeled_slippage_dollars,
    AVG(slippage_pts) * point_value / modeled_slippage_dollars as actual_vs_model_ratio
FROM live_trades
WHERE exited_at IS NOT NULL
  AND fill_entry IS NOT NULL
  AND trading_day >= CURRENT_DATE - INTERVAL ? DAY
GROUP BY instrument, strategy_id, entry_model
ORDER BY actual_vs_model_ratio DESC
```

Output sections:
1. **Per-Session Summary** — avg slippage vs modeled, ratio, N
2. **Kill Switch Check** — any session with ratio > 2.0x → RED ALERT
3. **E2 Tick Analysis** — `(fill_entry - orb_level) / tick_size` histogram
4. **Direction Bias** — long vs short slippage (detects asymmetric fills)
5. **Time-of-Day Pattern** — Asian vs US session slippage comparison

### Stage 3: Pre-Session Slippage Gate

**File:** `trading_app/pre_session_check.py` (add check)

New advisory check in existing pre-session flow:

```python
def check_slippage_health(instrument: str, lookback_days: int = 14) -> CheckResult:
    """Compare recent actual slippage to cost model.

    WARN if ratio > 1.5x modeled
    ALERT if ratio > 2.0x modeled (kill criterion from cost_model.py:83)
    """
```

### Stage 4: SESSION_SLIPPAGE_MULT Feedback

After 50+ trades per session accumulate in `live_journal.db`:
- Report recommended multiplier updates based on actual data
- Do NOT auto-update production `cost_model.py` — human review required
- Output format: "TOKYO_OPEN current=1.0x, measured=1.4x (N=52, p=0.03)"

---

## Files to Touch

| File | Change | Type |
|------|--------|------|
| `trading_app/live/session_orchestrator.py` | Exit fill polling for bracket fills | MODIFY |
| `trading_app/live/trade_journal.py` | No changes needed (schema already has fill_exit) | NONE |
| `scripts/tools/slippage_report.py` | NEW — aggregation + kill switch check | CREATE |
| `trading_app/pre_session_check.py` | Add slippage health advisory check | MODIFY |
| `tests/test_trading_app/test_slippage_report.py` | NEW — test report queries | CREATE |

## Blast Radius

- `session_orchestrator.py` is the live trading kernel — changes must be minimal, fail-open
- `pre_session_check.py` is advisory-only — safe to extend
- New script is read-only against `live_journal.db` — zero risk

## Acceptance Criteria

1. Exit fills captured for >90% of completed trades (non-NULL fill_exit)
2. `slippage_report.py` runs against live_journal.db and produces per-session breakdown
3. Kill switch flags any session with avg slippage > 2x modeled
4. All tests pass, drift checks clean
