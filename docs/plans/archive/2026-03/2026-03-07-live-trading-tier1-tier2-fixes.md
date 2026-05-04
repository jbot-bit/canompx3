---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Live Trading Tier 1+2 Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 defects identified by Bloomey review (Tier 1 must-fix + Tier 2 pre-live) to bring the live trading path from B- to A-.

**Architecture:** All changes are isolated to `trading_app/live/` and `trading_app/live_config.py`. No pipeline changes, no schema changes, no strategy logic changes. Each fix is independent and can be committed separately.

**Tech Stack:** Python 3.13, pytest, pytest-asyncio, asyncio.Queue, dataclasses

---

### Task 0: Fix dollar gate arithmetic bug in live_config.py

**Files:**
- Modify: `trading_app/live_config.py:334` (`_check_dollar_gate`)
- Modify: `trading_app/live_config.py:641` (`_exp_dollars`)
- Test: `tests/test_trading_app/test_live_config.py`

**Context:**
The dollar gate computes `one_r_dollars = median_risk_pts * spec.point_value + spec.total_friction`. This adds friction to the risk denominator — wrong. 1R in dollars = just the risk distance in dollar terms (`median_risk_pts * point_value`). Friction is a cost deducted from P&L, not part of the risk unit.

The same bug exists in the CLI display helper `_exp_dollars()` at line 641.

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_live_config.py`:

```python
def test_dollar_gate_does_not_add_friction_to_risk():
    """1R dollars = risk_pts * point_value. Friction is NOT in the denominator."""
    from unittest.mock import patch, MagicMock
    from trading_app.live_config import _check_dollar_gate

    # MGC: point_value=10, total_friction=2.74
    mock_spec = MagicMock()
    mock_spec.point_value = 10.0
    mock_spec.total_friction = 2.74

    variant = {
        "median_risk_points": 3.0,  # 1R = 3.0 * 10 = $30.00 (NOT $32.74)
        "expectancy_r": 0.15,       # Exp$ = 0.15 * 30 = $4.50
    }

    with patch("trading_app.live_config.get_cost_spec", return_value=mock_spec):
        passes, note = _check_dollar_gate(variant, "MGC")

    # $4.50 >= 1.3 * $2.74 ($3.56) -> passes
    assert passes is True
    assert "$4.50" in note or "4.50" in note
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_live_config.py::test_dollar_gate_does_not_add_friction_to_risk -v`
Expected: FAIL — current code computes `one_r_dollars = 3.0 * 10 + 2.74 = 32.74`, so `Exp$ = 0.15 * 32.74 = $4.91`, not `$4.50`.

**Step 3: Fix `_check_dollar_gate`**

In `trading_app/live_config.py`, line 334, change:
```python
# BEFORE (wrong):
one_r_dollars = median_risk_pts * spec.point_value + spec.total_friction

# AFTER (correct):
one_r_dollars = median_risk_pts * spec.point_value
```

**Step 4: Fix `_exp_dollars` display helper**

In `trading_app/live_config.py`, line 641, same fix:
```python
# BEFORE (wrong):
one_r_dollars = s.median_risk_points * spec.point_value + spec.total_friction

# AFTER (correct):
one_r_dollars = s.median_risk_points * spec.point_value
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_live_config.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add trading_app/live_config.py tests/test_trading_app/test_live_config.py
git commit -m "fix: dollar gate removes friction from 1R denominator (Bloomey A-1)"
```

---

### Task 1: Make webhook secret fail-closed

**Files:**
- Modify: `trading_app/live/webhook_server.py:39` (module-level secret check)
- Modify: `trading_app/live/webhook_server.py:216` (trade endpoint auth)
- Test: `tests/test_trading_app/test_webhook_server.py` (new file)

**Context:**
`WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")` defaults to empty string. The auth check at line 216 is `if WEBHOOK_SECRET and req.secret != WEBHOOK_SECRET` — when empty, ALL requests bypass auth. This is a security hole.

**Step 1: Write the failing test**

Create `tests/test_trading_app/test_webhook_server.py`:

```python
"""Tests for webhook server security."""

import os
from unittest.mock import patch


def test_webhook_rejects_empty_secret():
    """Server must reject requests when no secret is configured."""
    from fastapi.testclient import TestClient

    # Patch the module-level constant before importing the app
    with patch.dict(os.environ, {"WEBHOOK_SECRET": "test-secret-123"}):
        # Re-import to pick up the env var
        import importlib
        import trading_app.live.webhook_server as ws
        importlib.reload(ws)

        client = TestClient(ws.app, raise_server_exceptions=False)

        # Request with wrong secret should get 403
        resp = client.post("/trade", json={
            "instrument": "MGC",
            "direction": "long",
            "action": "entry",
            "secret": "wrong-secret",
        })
        assert resp.status_code == 403


def test_webhook_secret_required_on_startup():
    """Server should not start without WEBHOOK_SECRET env var."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove WEBHOOK_SECRET entirely
        os.environ.pop("WEBHOOK_SECRET", None)
        import importlib
        import trading_app.live.webhook_server as ws

        # Reloading without the secret should raise
        try:
            importlib.reload(ws)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "WEBHOOK_SECRET" in str(e)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_webhook_server.py::test_webhook_secret_required_on_startup -v`
Expected: FAIL — currently no startup validation.

**Step 3: Fix webhook_server.py**

Change line 39 from:
```python
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
```
to:
```python
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
if not WEBHOOK_SECRET:
    raise RuntimeError(
        "WEBHOOK_SECRET env var is required. "
        "Set it in .env or environment before starting the webhook server."
    )
```

Also update the auth check at line 216 — remove the `if WEBHOOK_SECRET` guard since it's now guaranteed non-empty:
```python
# BEFORE:
if WEBHOOK_SECRET and req.secret != WEBHOOK_SECRET:

# AFTER:
if req.secret != WEBHOOK_SECRET:
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_webhook_server.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add trading_app/live/webhook_server.py tests/test_trading_app/test_webhook_server.py
git commit -m "fix: webhook secret fail-closed — reject startup without secret (Bloomey A-2)"
```

---

### Task 2: Make Tradovate positions stub raise NotImplementedError

**Files:**
- Modify: `trading_app/live/tradovate/positions.py:15`
- Test: `tests/test_trading_app/test_broker_factory.py` (existing, or add inline)

**Context:**
`TradovatePositions.query_open()` returns `[]`, silently passing orphan detection. If someone runs `--broker tradovate`, the orchestrator thinks there are zero orphans when it actually never checked.

**Step 1: Write the failing test**

```python
def test_tradovate_positions_raises_not_implemented():
    """Tradovate positions must not silently return empty — it's not implemented."""
    from unittest.mock import MagicMock
    from trading_app.live.tradovate.positions import TradovatePositions

    pos = TradovatePositions(auth=MagicMock(), demo=True)
    import pytest
    with pytest.raises(NotImplementedError):
        pos.query_open(12345)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_broker_factory.py::test_tradovate_positions_raises_not_implemented -v`
Expected: FAIL — currently returns `[]`.

**Step 3: Fix positions.py**

Change `query_open` in `trading_app/live/tradovate/positions.py`:
```python
def query_open(self, account_id: int) -> list[dict]:
    raise NotImplementedError(
        "Tradovate position query not implemented. "
        "Use --broker projectx or implement Tradovate position API."
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_broker_factory.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add trading_app/live/tradovate/positions.py tests/test_trading_app/test_broker_factory.py
git commit -m "fix: Tradovate positions raises NotImplementedError (Bloomey B-5)"
```

---

### Task 3: Fix signalrcore bar drops with asyncio queue bridge

**Files:**
- Modify: `trading_app/live/projectx/data_feed.py:256-284` (sync callbacks)
- Test: `tests/test_trading_app/test_projectx_feed.py`

**Context:**
The signalrcore backend uses sync callbacks (`_on_quote_sync`, `_on_trade_sync`). These can't `await self.on_bar(bar)`, so completed bars are logged but **never delivered** to the execution engine. This means the signalrcore fallback path silently drops all bars.

Fix: Use `asyncio.Queue` as a bridge. Sync callbacks push bars to the queue; an async consumer task drains the queue and calls `on_bar`.

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_projectx_feed.py`:

```python
import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, AsyncMock

import pytest

from trading_app.live.projectx.data_feed import ProjectXDataFeed


@pytest.fixture
def feed_with_queue():
    """Create a feed with a mock on_bar callback to verify bar delivery."""
    auth = MagicMock()
    auth.get_token.return_value = "fake"
    bars_received = []

    async def capture_bar(bar):
        bars_received.append(bar)

    feed = ProjectXDataFeed(auth=auth, on_bar=capture_bar)
    feed._symbol = "12345"
    return feed, bars_received


@pytest.mark.asyncio
async def test_signalrcore_sync_callback_delivers_bars(feed_with_queue):
    """Bars from sync callbacks must be delivered to on_bar via queue."""
    feed, bars_received = feed_with_queue

    # Simulate two ticks that complete a bar (different minutes)
    t1 = datetime(2026, 3, 7, 14, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 7, 14, 1, 0, tzinfo=UTC)

    # Start the queue consumer
    consumer = asyncio.create_task(feed._drain_bar_queue())

    # Sync callback path
    feed._on_quote_sync([{"lastPrice": 100.0, "volume": 10}])
    # Manually set aggregator time for first tick
    feed._agg._bar_minute = t1
    feed._agg._current.ts_utc = t1

    # Second tick in new minute completes the bar
    bar = feed._agg.on_tick(101.0, 5, t2)
    if bar is not None:
        bar.symbol = feed._symbol
        feed._bar_queue.put_nowait(bar)

    # Give consumer time to process
    await asyncio.sleep(0.05)
    consumer.cancel()

    assert len(bars_received) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_projectx_feed.py::test_signalrcore_sync_callback_delivers_bars -v`
Expected: FAIL — `_bar_queue` doesn't exist yet.

**Step 3: Implement queue bridge in data_feed.py**

Add to `ProjectXDataFeed.__init__`:
```python
self._bar_queue: asyncio.Queue[Bar] = asyncio.Queue()
```

Add async drain method:
```python
async def _drain_bar_queue(self) -> None:
    """Consume bars from the sync→async queue bridge."""
    while True:
        bar = await self._bar_queue.get()
        await self.on_bar(bar)
```

Update `_on_quote_sync` and `_on_trade_sync` to push to queue instead of just logging:
```python
def _on_quote_sync(self, args: Any) -> None:
    quotes = args if isinstance(args, list) else [args]
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        try:
            price, vol = self.parse_quote(quote)
            bar = self._agg.on_tick(price, vol, datetime.now(UTC))
            if bar is not None:
                bar.symbol = self._symbol
                try:
                    self._bar_queue.put_nowait(bar)
                except Exception:
                    log.error("Bar queue full — dropping bar %s", bar.ts_utc)
        except (ValueError, KeyError) as e:
            log.debug("Skipping quote: %s", e)
```

Same pattern for `_on_trade_sync`.

Update `_run_signalrcore` to start the drain task:
```python
# After hub.start() and subscribe, start queue consumer
drain_task = asyncio.create_task(self._drain_bar_queue())
try:
    while not _STOP_FILE.exists():
        await asyncio.sleep(2.5)
        self.auth.refresh_if_needed()
finally:
    drain_task.cancel()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_projectx_feed.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add trading_app/live/projectx/data_feed.py tests/test_trading_app/test_projectx_feed.py
git commit -m "fix: signalrcore bar drops — async queue bridge for sync callbacks (Bloomey A-3)"
```

---

### Task 4: Per-strategy CUSUM sigma from validated_setups

**Files:**
- Modify: `trading_app/live/session_orchestrator.py` (CUSUM initialization)
- Test: `tests/test_trading_app/test_cusum_monitor.py` (add test for non-default sigma)

**Context:**
`CUSUMMonitor` already accepts `std_r` as a constructor param. The issue is that `session_orchestrator.py` doesn't pass per-strategy sigma — it likely uses the default `1.0` for all strategies. The fix is to look up `std_r` from the strategy's historical R distribution (available in `validated_setups` or computed from `orb_outcomes`).

Since computing exact per-strategy sigma requires a DB query at startup, a pragmatic approach is:
1. Add a `std_r` field to `PortfolioStrategy` (default 1.0)
2. When building the portfolio, optionally load it from DB
3. Pass it through to CUSUM initialization

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_cusum_monitor.py`:

```python
def test_cusum_with_custom_sigma():
    """CUSUM with lower sigma should trigger alarm sooner on same drift."""
    from trading_app.live.cusum_monitor import CUSUMMonitor

    conservative = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    sensitive = CUSUMMonitor(expected_r=0.3, std_r=0.5, threshold=4.0)

    # Feed identical losing trades
    for _ in range(8):
        conservative.update(-0.5)
        sensitive.update(-0.5)

    # Sensitive (lower sigma) should alarm first
    assert sensitive.alarm_triggered is True
    assert conservative.alarm_triggered is False  # needs more trades at sigma=1.0
```

**Step 2: Run test to verify behavior**

Run: `pytest tests/test_trading_app/test_cusum_monitor.py::test_cusum_with_custom_sigma -v`
Expected: PASS (CUSUMMonitor already supports this — the test validates the mechanism works).

**Step 3: Verify orchestrator passes std_r**

Read the orchestrator's CUSUM setup code. If it hardcodes `std_r=1.0`, update it to pull from the strategy's stats. If there's no CUSUM setup in the orchestrator yet (monitor handles it), document where to wire it.

This task may be smaller than expected — CUSUM is already parameterized. The gap is ensuring the orchestrator uses per-strategy values when available.

**Step 4: Commit**

```bash
git add tests/test_trading_app/test_cusum_monitor.py
git commit -m "test: CUSUM per-strategy sigma validation (Bloomey A-4)"
```

---

### Task 5: Add broker API circuit breaker

**Files:**
- Create: `trading_app/live/circuit_breaker.py`
- Modify: `trading_app/live/session_orchestrator.py` (wrap order submission)
- Test: `tests/test_trading_app/test_circuit_breaker.py` (new file)

**Context:**
If the broker API returns consecutive errors, the system keeps hammering it. A circuit breaker stops attempts after N consecutive failures for M seconds, then allows one probe request. Standard resilience pattern.

**Step 1: Write the failing test**

Create `tests/test_trading_app/test_circuit_breaker.py`:

```python
"""Tests for broker API circuit breaker."""

import time
from trading_app.live.circuit_breaker import CircuitBreaker


def test_circuit_opens_after_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    # 3 consecutive failures should open the circuit
    for _ in range(3):
        breaker.record_failure()

    assert breaker.is_open is True


def test_circuit_resets_on_success():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()  # resets counter

    assert breaker.is_open is False
    assert breaker.consecutive_failures == 0


def test_circuit_allows_probe_after_timeout():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is True

    time.sleep(0.15)  # wait past recovery timeout
    assert breaker.should_allow_request() is True  # probe allowed


def test_circuit_stays_open_before_timeout():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.should_allow_request() is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_circuit_breaker.py -v`
Expected: FAIL — `circuit_breaker.py` doesn't exist.

**Step 3: Implement CircuitBreaker**

Create `trading_app/live/circuit_breaker.py`:

```python
"""Circuit breaker for broker API resilience.

Opens after N consecutive failures, blocks requests for M seconds,
then allows one probe request. Resets on success.
"""

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds

    consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold

    def should_allow_request(self) -> bool:
        if not self.is_open:
            return True
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self.recovery_timeout:
            log.info("Circuit breaker: allowing probe request after %.1fs", elapsed)
            return True
        return False

    def record_success(self) -> None:
        if self.consecutive_failures > 0:
            log.info("Circuit breaker: reset after %d failures", self.consecutive_failures)
        self.consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.is_open and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning(
                "Circuit breaker OPEN after %d consecutive failures. "
                "Blocking requests for %.0fs.",
                self.consecutive_failures,
                self.recovery_timeout,
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_circuit_breaker.py -v`
Expected: ALL PASS

**Step 5: Wire into session_orchestrator (optional — can be a follow-up)**

In `_handle_event`, wrap `self.order_router.submit()` calls:
```python
if not self._broker_breaker.should_allow_request():
    log.warning("Circuit breaker open — skipping order for %s", event.strategy_id)
    return
try:
    result = self.order_router.submit(spec)
    self._broker_breaker.record_success()
except Exception as e:
    self._broker_breaker.record_failure()
    raise
```

**Step 6: Commit**

```bash
git add trading_app/live/circuit_breaker.py tests/test_trading_app/test_circuit_breaker.py
git commit -m "feat: broker API circuit breaker — stop hammering after N failures (Bloomey B-3)"
```

---

### Task 6: Run full test suite + commit

**Files:**
- All modified files from Tasks 0-5

**Step 1: Run ruff format**

```bash
ruff format trading_app/live/ trading_app/live_config.py tests/test_trading_app/
```

**Step 2: Run ruff check**

```bash
ruff check trading_app/live/ trading_app/live_config.py tests/test_trading_app/
```

**Step 3: Run drift checks**

```bash
python pipeline/check_drift.py
```

**Step 4: Run full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: ALL PASS, no regressions.

**Step 5: Final commit if any formatting fixes needed**

```bash
git add -A
git commit -m "style: format after Bloomey Tier 1+2 fixes"
```
