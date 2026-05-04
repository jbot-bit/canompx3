---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Design Plan 4: Trade-by-Trade Regime Detection

**Date:** 2026-04-04
**Priority:** MEDIUM — monthly rebalance is 30-day blind spot
**Status:** DESIGN (awaiting implementation)
**Depends on:** Plan 3 (Alert Engine) for notification delivery

---

## What & Why

Current regime detection operates on a **monthly rebalance cycle**:
1. `lane_allocator.py` computes 6-month trailing session regime (HOT/COLD)
2. Writes `docs/runtime/lane_allocation.json`
3. `session_orchestrator.py` loads it ONCE at session start
4. Strategies in `paused` list are blocked from entry

**Problem:** If a regime breaks mid-month, 30+ days pass before the system responds.
A strategy could accumulate 20+ losing trades before the next rebalance catches it.

**Solution:** Wire the existing `CUSUMMonitor` (dormant in `cusum_monitor.py`) into the
session orchestrator for real-time per-strategy drift detection.

## Existing Infrastructure

| Component | File | Status |
|-----------|------|--------|
| `CUSUMMonitor` class | `trading_app/live/cusum_monitor.py:17-62` | COMPLETE but orphaned |
| SPRT monitor | `trading_app/sprt_monitor.py:70-92` | Reporting only |
| Session regime gate | `session_orchestrator.py:155-174` | Loads JSON once at start |
| Entry gate check | `session_orchestrator.py:1455-1464` | Checks `_regime_paused` set |
| Lane allocator | `trading_app/lane_allocator.py:383-424` | Monthly: HOT/COLD per session |
| Strategy fitness | `trading_app/strategy_fitness.py:104-134` | FIT/WATCH/DECAY/STALE |
| Trade journal | `trading_app/live/trade_journal.py` | `cusum_alarm` column exists |

### Key Insight from Backtest (lane_allocator.py:396-403)
Individual strategy month streaks are NOISE. Session regime is the signal.
Trade-by-trade detection must detect **per-strategy cumulative drift**, but the
RESPONSE should be per-session (pause the session, not just one strategy).

---

## Design Decisions

### Per-Strategy vs Per-Session Monitoring

**Monitor per-strategy, aggregate per-session.**
- Each strategy gets its own `CUSUMMonitor` instance
- When ANY strategy in a session triggers CUSUM alarm → flag the entire session
- Rationale: session regime is the proven gate (backtest evidence), but individual
  strategies provide the fastest signal of regime change

### Threshold Selection

The existing `CUSUMMonitor` uses `threshold=4.0` (4σ accumulated drift).
This is conservative — catches genuine regime breaks within ~100-trade windows.

For live trading with 5-10 trades/session/month:
- **4σ threshold** → alarm after ~20-40 trades (2-4 months). TOO SLOW.
- **3σ threshold** → alarm after ~10-20 trades (1-2 months). GOOD for live.
- **2σ threshold** → alarm after ~5-10 trades (<1 month). FALSE POSITIVE risk.

**Recommendation:** Use `threshold=3.0` for live, keep `4.0` as default.
Make configurable per instantiation.

### Response: Pause vs Alert

When CUSUM fires:
1. **Immediate:** Fire WARNING alert via alert engine (Plan 3)
2. **Do NOT auto-pause.** Regime decisions require human review.
3. **Log to trade journal** — `cusum_alarm=True` on the triggering trade
4. **Dashboard display** — show drift severity meter per strategy

**Rationale:** Auto-pausing creates a new risk — false CUSUM alarms could pull
strategies that are just experiencing normal variance. Monthly rebalance is the
proven mechanism; CUSUM is an EARLY WARNING, not an auto-pilot.

---

## Implementation Plan

### Stage 1: Wire CUSUMMonitor into SessionOrchestrator

**File:** `trading_app/live/session_orchestrator.py`

```python
# In __init__ or on_trading_day_start():
from trading_app.live.cusum_monitor import CUSUMMonitor

# Create one monitor per active strategy
self._cusum_monitors: dict[str, CUSUMMonitor] = {}
for strategy in self._portfolio.strategies:
    self._cusum_monitors[strategy.strategy_id] = CUSUMMonitor(
        expected_r=strategy.backtest_expr,  # From validated_setups or lane config
        std_r=strategy.backtest_std_r or 1.0,  # Conservative default
        threshold=3.0,  # Live threshold (more responsive than default 4.0)
    )
```

### Stage 2: Feed Trade Results to CUSUM

```python
# After trade exit (in _handle_trade_exit or equivalent):
monitor = self._cusum_monitors.get(trade.strategy_id)
if monitor:
    alarm_fired = monitor.update(trade.actual_r)
    if alarm_fired:
        # 1. Alert via engine (Plan 3)
        self._alert_engine.fire(Alert(
            level="WARNING",
            category="cusum_drift",
            message=f"{trade.strategy_id}: CUSUM alarm at {monitor.drift_severity:.1f}σ "
                    f"after {monitor.n_trades} trades",
            strategy_id=trade.strategy_id,
            data={"drift_sigma": monitor.drift_severity, "n_trades": monitor.n_trades},
        ))
        # 2. Mark in journal
        cusum_alarm = True  # Passed to journal.record_exit()
```

### Stage 3: Backtest Stats Loader

**File:** `trading_app/live/cusum_monitor.py` (add helper)

```python
def load_backtest_stats(strategy_id: str, con: duckdb.DuckDBPyConnection) -> tuple[float, float]:
    """Load expected_r and std_r from orb_outcomes for a validated strategy.

    Returns (expected_r, std_r) for CUSUMMonitor initialization.
    Falls back to (0.10, 1.0) if data unavailable.
    """
    # Parse strategy_id to extract dimensions
    # Query orb_outcomes with filter applied via daily_features
    # Return (mean(pnl_r), stddev(pnl_r))
```

This must properly apply the strategy's filter (join daily_features on triple key).

### Stage 4: Dashboard Drift Display

**File:** `trading_app/live/bot_dashboard.py` + `.html`

Add per-strategy drift meter to dashboard status:

```json
{
  "strategies": [
    {
      "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER",
      "cusum_drift_sigma": 1.7,
      "cusum_n_trades": 12,
      "cusum_status": "OK"  // or "ALARM"
    }
  ]
}
```

Visual: horizontal bar showing drift severity (green < 2σ, yellow 2-3σ, red > 3σ).

### Stage 5: Session-Level Aggregation (FUTURE — v2)

After individual CUSUM works, add session-level aggregation:
- If 2+ strategies in the same session fire CUSUM → escalate to CRITICAL
- Generate "SESSION COLD SIGNAL" advisory
- Pre-populate next month's rebalance with suggested pauses

This is v2 — don't build until v1 (per-strategy CUSUM) has run for 1+ months.

---

## Files to Touch

| File | Change | Type |
|------|--------|------|
| `trading_app/live/session_orchestrator.py` | Init CUSUM monitors + feed trades | MODIFY |
| `trading_app/live/cusum_monitor.py` | Add `load_backtest_stats()` helper | MODIFY |
| `trading_app/live/bot_dashboard.py` | Add drift data to status JSON | MODIFY |
| `trading_app/live/bot_dashboard.html` | Add drift meter visualization | MODIFY |
| `tests/test_trading_app/test_cusum_integration.py` | NEW — test CUSUM wiring | CREATE |

## Blast Radius

- `session_orchestrator.py` — additive only (new dict + 1 post-trade hook). No change to entry/exit logic.
- `cusum_monitor.py` — adding a helper function, existing class untouched
- Dashboard changes are additive (new JSON fields + UI panel)
- **CUSUM does NOT auto-pause strategies** — it only fires alerts. Entry gate unchanged.

## Acceptance Criteria

1. Each active strategy gets a `CUSUMMonitor` instance at session start
2. Every trade exit feeds `actual_r` to the monitor
3. CUSUM alarm fires WARNING alert when drift > 3σ
4. `cusum_alarm=True` recorded in `live_trades` journal on triggering trade
5. Dashboard shows per-strategy drift severity
6. No auto-pause behavior — human review required
7. Tests verify: alarm fires at threshold, resets on clear(), backtest stats load correctly
