# Plan: Live Trading Observability — Kill Silent Failures

**Goal:** Every component in the live trading stack must be VERIFIABLE. If something breaks, the operator KNOWS within seconds, not never.
**Priority:** CRITICAL — this blocks going live with real money.

## The Problem

7 commits added live trading features using a "never raises" pattern (`except Exception: pass`). This prevents crashes but creates invisible failures. The operator has NO WAY to distinguish "working" from "silently broken."

### Silent Failure Inventory

| Component | What Can Fail Silently | Impact If Broken | Current Detection |
|-----------|----------------------|------------------|-------------------|
| `_notify()` | Import fail, bad token, bad chat ID, network error | Operator blind to all critical events | **NONE** |
| `_fill_poller()` | NotImplementedError, poller task dies, query always fails | PENDING_ENTRY stuck forever, position never confirmed | WARNING log only |
| `_submit_bracket()` | Wrong API format, auth expired, bracket rejected | No crash protection — user thinks they're protected but aren't | WARNING log only |
| `_cancel_brackets()` | Cancel fails, bracket fills during race window | Double position (bracket fills + exit creates opposite) | WARNING log only |
| `was_stopped` detection | Property returns wrong value on specific feed impl | Session dies or loops forever | **NONE** |
| Orchestrator reconnect | Auth refresh fails silently, backoff logic wrong | Session dies after first disconnect | Notification (circular) |
| Trading day rollover | `_handle_event` fails during rollover | Positions left open at broker across trading days | CRITICAL log only |

### The Circular Dependency

Telegram notifications alert about failures → but if Telegram itself fails → nothing alerts about Telegram failing → operator is blind.

---

## Phase 1: Startup Self-Test (Effort: 30 min, ROI: 10x)

**Solve the circular dependency.** On session start, verify every component works BEFORE accepting any bars.

### 1A: Notification Self-Test

Add to `__init__` after `_notify(f"Session started...")`:

```python
def _verify_notifications(self) -> bool:
    """Send test notification and verify it was sent. Returns False if broken."""
    try:
        from trading_app.live.notifications import notify
        # notify() returns None and swallows exceptions — we need to test deeper
        from scripts.infra.telegram_feed import send_telegram
        result = send_telegram(f"[{self.instrument}] SELF-TEST: notifications working")
        if not result:
            log.critical("NOTIFICATION SELF-TEST FAILED — send_telegram returned False")
            return False
        log.info("Notification self-test passed")
        return True
    except Exception as e:
        log.critical("NOTIFICATION SELF-TEST FAILED: %s", e)
        return False
```

Call during `__init__` or at start of `run()`. If it fails:
- Log CRITICAL (visible in console)
- Set `self._notifications_broken = True`
- Print to STDOUT: `!!! NOTIFICATIONS ARE BROKEN — you will not receive alerts !!!`
- Do NOT abort session — but operator must be watching console

### 1B: Bracket Order Self-Test

```python
def _verify_brackets(self) -> bool:
    """Verify bracket order support is real, not just claimed."""
    if self.order_router is None:
        return True  # signal-only, no brackets needed
    if not self.order_router.supports_native_brackets():
        log.info("Broker does not support brackets — no crash protection")
        return True
    # Dry-run: build a bracket spec and verify it's valid JSON
    try:
        spec = self.order_router.build_bracket_spec(
            direction="long", symbol="TEST", entry_price=100.0,
            stop_price=99.0, target_price=102.0, qty=1,
        )
        if spec is None:
            log.warning("build_bracket_spec returned None despite supports_native_brackets=True")
            return False
        log.info("Bracket spec self-test passed: %s", spec)
        return True
    except Exception as e:
        log.critical("BRACKET SELF-TEST FAILED: %s", e)
        return False
```

### 1C: Fill Poller Self-Test

```python
def _verify_fill_poller(self) -> bool:
    """Verify broker supports order status queries."""
    if self.order_router is None or self.signal_only:
        return True
    try:
        self.order_router.query_order_status(0)  # dummy ID
        return True  # if it doesn't raise NotImplementedError, it's implemented
    except NotImplementedError:
        log.warning("Broker does not support query_order_status — fill poller disabled")
        return False
    except Exception:
        return True  # other errors (e.g. 404) mean the endpoint exists
```

### 1D: Wire Into Preflight

The `--preflight` mode already exists. Add these 3 checks to it. Print a clear pass/fail summary:

```
SELF-TEST RESULTS:
  Notifications:  PASS / FAIL (details)
  Brackets:       PASS / FAIL / N/A (signal-only)
  Fill Poller:    PASS / FAIL / N/A (signal-only)
  Feed Connect:   PASS / FAIL
```

---

## Phase 2: Session Heartbeat (Effort: 20 min, ROI: 5x)

**Problem:** If the session is running but nothing is happening (no bars, no trades), the operator has no signal that it's alive.

Add a periodic heartbeat notification (every 30 min):

```python
async def _heartbeat_notifier(self) -> None:
    """Send periodic alive notification. Proves notifications work continuously."""
    while True:
        await asyncio.sleep(1800)  # 30 minutes
        n_bars = self._bar_count  # track in _on_bar
        n_trades = len(self.monitor.trades) if self.monitor else 0
        self._notify(f"Heartbeat: {n_bars} bars, {n_trades} trades, poller={'ON' if self._poller_active else 'OFF'}")
```

Start alongside watchdog and fill poller in `run()`.

**Why this solves the circular dependency:** If you stop receiving heartbeats, notifications are broken. The ABSENCE of the heartbeat IS the alert.

---

## Phase 3: Observability Counters (Effort: 30 min, ROI: 4x)

Track success/failure counts for every silent-failure component. Report in EOD summary.

```python
@dataclass
class SessionStats:
    notifications_sent: int = 0
    notifications_failed: int = 0
    brackets_submitted: int = 0
    brackets_failed: int = 0
    bracket_cancels_ok: int = 0
    bracket_cancels_failed: int = 0
    fill_polls_run: int = 0
    fill_polls_confirmed: int = 0
    fill_polls_failed: int = 0
    reconnect_attempts: int = 0
    bars_received: int = 0
    events_processed: int = 0
```

Increment in each component. In `_notify()`:
```python
def _notify(self, message: str) -> None:
    try:
        from trading_app.live.notifications import notify
        notify(self.instrument, message)
        self._stats.notifications_sent += 1
    except Exception:
        self._stats.notifications_failed += 1
```

In `post_session()` EOD summary:
```python
log.info("SESSION STATS: %s", self._stats)
self._notify(f"EOD Stats: {self._stats.bars_received} bars, "
             f"{self._stats.notifications_sent}/{self._stats.notifications_sent + self._stats.notifications_failed} notifs OK, "
             f"{self._stats.brackets_submitted}/{self._stats.brackets_submitted + self._stats.brackets_failed} brackets OK")
```

---

## Phase 4: Upgrade `_notify()` From Fire-and-Forget to Verified (Effort: 15 min, ROI: 3x)

Current `_notify()` catches ALL exceptions and does nothing. Change to:

```python
def _notify(self, message: str) -> None:
    """Send notification. Logs failure but never raises."""
    try:
        from trading_app.live.notifications import notify
        notify(self.instrument, message)
        self._stats.notifications_sent += 1
    except Exception as e:
        self._stats.notifications_failed += 1
        log.error("Notification failed (will not retry): %s", e)
        if self._stats.notifications_failed == 1:
            # First failure — print to console so operator sees it
            print(f"!!! NOTIFICATION FAILURE: {e} — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID !!!")
```

Still never raises. But now failures are:
1. Counted
2. Logged as ERROR (not swallowed)
3. First failure prints to STDOUT (operator sees it even without log monitoring)

---

## Phase 5: Fill Poller Activity Logging (Effort: 10 min, ROI: 2x)

Current fill poller logs nothing when it has nothing to do. Add:

```python
# At end of each poll cycle
if pending_count > 0:
    log.info("Fill poller: checked %d pending orders", pending_count)
    self._stats.fill_polls_run += 1
```

And a flag `self._poller_active = True` set in the poller loop, checked by heartbeat.

---

## Implementation Order

1. **Phase 1** (startup self-test) — highest ROI, catches broken config before any real money at risk
2. **Phase 2** (heartbeat) — solves circular dependency, operator knows session is alive
3. **Phase 3** (counters) — quantifies silent failures in EOD summary
4. **Phase 4** (upgrade _notify) — first failure visible to operator
5. **Phase 5** (fill poller logging) — smallest gap, lowest priority

## Verification

After each phase:
1. `python -m pytest tests/test_trading_app/test_session_orchestrator.py -x -q`
2. `python pipeline/check_drift.py`
3. Manual: `python scripts/run_live_session.py --instrument MGC --preflight` — verify self-test output
4. Manual: Send test notification with bad token — verify FAILURE is visible

---

## What This Does NOT Cover

- **Broker API format verification** — ProjectX bracket spec format is speculative. Only real API testing or docs will confirm it. This plan makes the failure VISIBLE, not fix it.
- **Network partition handling** — if the machine loses internet, everything fails. That's an infra problem, not a code problem.
- **Log aggregation/monitoring** — long-term, logs should go somewhere persistent (file, cloud). For V1, console + Telegram is sufficient if heartbeat works.
