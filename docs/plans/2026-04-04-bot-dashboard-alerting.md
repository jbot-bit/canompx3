# Design Plan 3: Bot Dashboard Alerting System

**Date:** 2026-04-04
**Priority:** MEDIUM — flying blind during live sessions without alerts
**Status:** DESIGN (awaiting implementation)

---

## What & Why

`bot_dashboard.py` + `bot_dashboard.html` show live session status (lanes, trades, P&L).
But there are NO alert thresholds — no regime shift alerts, no DD breach warnings,
no win rate divergence notifications. The operator has to manually watch the dashboard
to catch problems. During overnight sessions (CME_REOPEN at 09:00 AEST), nobody is watching.

## Existing Infrastructure

| Component | File | What Exists |
|-----------|------|-------------|
| Dashboard web UI | `trading_app/live/bot_dashboard.py` + `.html` | HTTP server, JSON status endpoint, HTML display |
| Risk manager | `trading_app/risk_manager.py` | Circuit breaker, position limits, DD tracking |
| CUSUM monitor | `trading_app/live/cusum_monitor.py` | Per-strategy drift detection (DORMANT — not wired in) |
| SPRT monitor | `trading_app/sprt_monitor.py` | Sequential test for lane degradation (reporting only) |
| Account HWM tracker | `trading_app/account_hwm_tracker.py` | Daily/weekly loss limits, DD monitoring |
| Pre-session checks | `trading_app/pre_session_check.py` | Advisory checks before session start |
| Performance monitor | `trading_app/live/performance_monitor.py` | Live trade tracking, P&L computation |
| Trade journal | `trading_app/live/trade_journal.py` | `cusum_alarm` boolean column already in schema |

## Alert Categories

### Category A: CRITICAL (immediate action required)

| Alert | Trigger | Data Source | Latency |
|-------|---------|-------------|---------|
| **Circuit Breaker Trip** | Cumulative daily loss > firm limit | `risk_manager.py` | Real-time (per trade) |
| **DD Breach Warning** | Trailing DD within 20% of firm max | `account_hwm_tracker.py` | Real-time (per trade) |
| **Feed Silence** | No bars received for 5+ minutes during session | `session_orchestrator.py` watchdog | Real-time |
| **Position Orphan** | Entry confirmed but no bracket placed | `position_tracker.py` | Real-time |

### Category B: WARNING (review within session)

| Alert | Trigger | Data Source | Latency |
|-------|---------|-------------|---------|
| **CUSUM Alarm** | Cumulative drift > 4σ below expected | `cusum_monitor.py` | Per-trade |
| **Loss Streak** | 5+ consecutive losses on any lane | `sprt_monitor.compute_streak()` | Per-trade |
| **Slippage Spike** | Single trade slippage > 3x modeled | `trade_journal.py` | Per-trade |
| **SPRT Degraded** | Lane log-LR < -2.079 | `sprt_monitor.py` | Per-trade |

### Category C: ADVISORY (review at session end)

| Alert | Trigger | Data Source | Latency |
|-------|---------|-------------|---------|
| **Win Rate Divergence** | Session WR deviates > 10% from backtest | `performance_monitor.py` | End of session |
| **Regime Shift Signal** | Session ExpR turns negative (6mo trailing) | `lane_allocator.py` | Daily |
| **Consistency Warning** | Best day approaching 40% of total (Bulenox) | `consistency_tracker.py` | Daily |

---

## Implementation Plan

### Stage 1: Alert Engine (NEW MODULE)

**File:** `trading_app/live/alert_engine.py`

Core design: event-driven, stateless per check, in-process with orchestrator.

```python
@dataclass(frozen=True)
class Alert:
    level: Literal["CRITICAL", "WARNING", "ADVISORY"]
    category: str           # "circuit_breaker", "cusum", "slippage", etc.
    message: str
    strategy_id: str | None
    timestamp: datetime
    data: dict              # Arbitrary payload for display

class AlertEngine:
    def __init__(self, channels: list[AlertChannel]):
        self._channels = channels
        self._suppressed: dict[str, datetime] = {}  # dedup key → last fired

    def fire(self, alert: Alert, dedup_key: str | None = None,
             cooldown_minutes: int = 30) -> None:
        """Fire alert to all channels. Dedup within cooldown window."""
        if dedup_key and self._is_suppressed(dedup_key, cooldown_minutes):
            return
        for channel in self._channels:
            channel.send(alert)
        if dedup_key:
            self._suppressed[dedup_key] = alert.timestamp
```

### Stage 2: Notification Channels

```python
class AlertChannel(Protocol):
    def send(self, alert: Alert) -> None: ...

class LogChannel(AlertChannel):
    """Always-on — writes to session log file."""
    def send(self, alert: Alert) -> None:
        log.warning("ALERT [%s] %s: %s", alert.level, alert.category, alert.message)

class DashboardChannel(AlertChannel):
    """Pushes to bot_dashboard via shared state."""
    def send(self, alert: Alert) -> None:
        # Append to dashboard's alert list (displayed in UI)
        self._dashboard_state.alerts.append(alert)

class FileChannel(AlertChannel):
    """Writes JSON alerts to docs/runtime/alerts.jsonl for external consumption."""
    def send(self, alert: Alert) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(alert), default=str) + "\n")

# FUTURE (not in v1): WebhookChannel for Discord/Telegram/email
```

### Stage 3: Wire into Session Orchestrator

**File:** `trading_app/live/session_orchestrator.py`

Minimal integration points (3 hooks):

```python
# 1. After trade exit (on_trade_complete or equivalent):
self._alert_engine.check_post_trade(trade_record)

# 2. On watchdog tick (every 60s):
self._alert_engine.check_heartbeat(last_bar_time, now)

# 3. On circuit breaker trip (risk_manager callback):
self._alert_engine.fire(Alert(level="CRITICAL", category="circuit_breaker", ...))
```

### Stage 4: Dashboard Alert Panel

**File:** `trading_app/live/bot_dashboard.html`

Add alert panel to existing dashboard:
- Scrolling alert log (newest first)
- Color-coded: red=CRITICAL, yellow=WARNING, blue=ADVISORY
- Dismissible (marks as acknowledged)
- Persists in `docs/runtime/alerts.jsonl` for post-session review

### Stage 5: Post-Trade Alert Checks

**File:** `trading_app/live/alert_engine.py` (method)

```python
def check_post_trade(self, trade: TradeRecord) -> None:
    """Run all per-trade alert checks."""
    # 1. Slippage spike
    if trade.slippage_pts and abs(trade.slippage_pts) > 3 * modeled_slippage:
        self.fire(Alert(level="WARNING", category="slippage_spike", ...))

    # 2. Loss streak (query recent trades from performance_monitor)
    streak = self._perf_monitor.current_streak(trade.strategy_id)
    if streak >= 8:
        self.fire(Alert(level="WARNING", category="loss_streak_8", ...))
    elif streak >= 5:
        self.fire(Alert(level="ADVISORY", category="loss_streak_5", ...))

    # 3. CUSUM (delegate to cusum_monitor — see Plan 4)
    # 4. Circuit breaker proximity
    dd_remaining = self._risk_manager.dd_remaining()
    if dd_remaining < 0.20 * self._risk_manager.max_dd:
        self.fire(Alert(level="CRITICAL", category="dd_proximity", ...))
```

---

## Files to Touch

| File | Change | Type |
|------|--------|------|
| `trading_app/live/alert_engine.py` | NEW — alert engine + channels + checks | CREATE |
| `trading_app/live/session_orchestrator.py` | Wire 3 alert hooks (post-trade, heartbeat, circuit breaker) | MODIFY |
| `trading_app/live/bot_dashboard.py` | Add `/alerts` JSON endpoint | MODIFY |
| `trading_app/live/bot_dashboard.html` | Add alert panel UI | MODIFY |
| `tests/test_trading_app/test_alert_engine.py` | NEW — unit tests for alert firing, dedup, channels | CREATE |

## Blast Radius

- `session_orchestrator.py` — 3 minimal hooks, fail-open (alert failure must never block trading)
- `bot_dashboard.py/html` — additive UI, no breaking changes
- New module `alert_engine.py` — isolated, no imports into pipeline/

## Acceptance Criteria

1. CRITICAL alerts (circuit breaker, DD breach, feed silence) fire within 1 trade/60s
2. WARNING alerts (CUSUM, loss streak, slippage) fire per-trade with 30-min cooldown dedup
3. Dashboard shows scrolling alert log with color coding
4. `docs/runtime/alerts.jsonl` persists alert history
5. All alerts fail-open — alert engine crash never blocks trading
6. Tests cover: alert firing, dedup cooldown, channel dispatch, all 3 alert levels
