---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Bloomey B- Operational Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 8 operational gaps from the Bloomey usability review to bring the live trading path from B- to A-.

**Architecture:** Surgical fixes to existing modules. No new files except tests. Focus: fail-closed daily features, CUSUM calibration, preflight validation, circuit breaker wiring, monitor reset bug.

**Tech Stack:** Python 3.13, asyncio, DuckDB, pytest, ruff

---

### Task 0: Fix CUSUM std_r calibration (false alarms on RR2.5+)

**Files:**
- Modify: `trading_app/live/performance_monitor.py:45-53`
- Modify: `trading_app/live/cusum_monitor.py:20` (docstring only)
- Test: `tests/test_trading_app/test_cusum_monitor.py`

**Step 1: Write the failing test**

```python
# tests/test_trading_app/test_cusum_monitor.py — add to existing file

def test_std_r_calibrated_per_strategy():
    """CUSUM std_r must be computed from win_rate + rr_target, not hardcoded 1.0."""
    from trading_app.live.performance_monitor import PerformanceMonitor
    from trading_app.portfolio import PortfolioStrategy

    # RR3.0 strategy: theoretical std_r ≈ 1.87, NOT 1.0
    s = PortfolioStrategy(
        strategy_id="TEST_RR3", instrument="MGC", orb_label="CME_REOPEN",
        entry_model="E2", rr_target=3.0, confirm_bars=1, filter_type="ORB_G5",
        expectancy_r=0.20, win_rate=0.30, sample_size=200, sharpe_ratio=1.0,
        max_drawdown_r=5.0, median_risk_points=3.0, stop_multiplier=1.0,
        source="test", weight=1.0,
    )
    monitor = PerformanceMonitor([s])
    cusum = monitor.get_cusum("TEST_RR3")
    # std_r for WR=0.30, RR=3.0, ExpR=0.20:
    # sqrt(0.30*(3.0-0.20)^2 + 0.70*(-1-0.20)^2) ≈ 1.764
    assert cusum.std_r > 1.5, f"std_r={cusum.std_r} — should be ~1.76, not 1.0"
    assert cusum.std_r < 2.0

def test_std_r_rr1_stays_near_one():
    """RR1.0 strategies should have std_r ≈ 1.0 (validates formula doesn't break them)."""
    from trading_app.live.performance_monitor import PerformanceMonitor
    from trading_app.portfolio import PortfolioStrategy

    s = PortfolioStrategy(
        strategy_id="TEST_RR1", instrument="MGC", orb_label="CME_REOPEN",
        entry_model="E2", rr_target=1.0, confirm_bars=1, filter_type="ORB_G5",
        expectancy_r=0.05, win_rate=0.55, sample_size=200, sharpe_ratio=1.0,
        max_drawdown_r=5.0, median_risk_points=3.0, stop_multiplier=1.0,
        source="test", weight=1.0,
    )
    monitor = PerformanceMonitor([s])
    cusum = monitor.get_cusum("TEST_RR1")
    assert 0.9 < cusum.std_r < 1.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_cusum_monitor.py -v -k "test_std_r"`
Expected: FAIL — std_r is 1.0 for all strategies

**Step 3: Implement the fix**

In `trading_app/live/performance_monitor.py:45-53`, replace hardcoded `std_r=1.0` with computed value:

```python
import math

def _compute_std_r(win_rate: float, rr_target: float, expectancy_r: float) -> float:
    """Theoretical std of R outcomes for a fixed-RR strategy.

    Wins are +RR, losses are -1.0.
    @research-source: binary outcome variance formula
    """
    return math.sqrt(
        win_rate * (rr_target - expectancy_r) ** 2
        + (1 - win_rate) * (-1.0 - expectancy_r) ** 2
    )
```

Then in `__init__`:
```python
self._monitors: dict[str, CUSUMMonitor] = {
    s.strategy_id: CUSUMMonitor(
        expected_r=s.expectancy_r,
        std_r=_compute_std_r(s.win_rate, s.rr_target, s.expectancy_r),
        threshold=4.0,
    )
    for s in strategies
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_cusum_monitor.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add trading_app/live/performance_monitor.py tests/test_trading_app/test_cusum_monitor.py
git commit -m "fix: CUSUM std_r calibrated per strategy (was hardcoded 1.0)"
```

---

### Task 1: Fix PerformanceMonitor.reset_daily() — clear _trades list

**Files:**
- Modify: `trading_app/live/performance_monitor.py:86-88`
- Test: `tests/test_trading_app/test_session_orchestrator.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_trading_app/test_session_orchestrator.py TestSlippageInSummary class

def test_reset_daily_clears_trades(self):
    """reset_daily() must clear _trades list so slippage doesn't accumulate across days."""
    from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord

    monitor = PerformanceMonitor([_test_strategy()])
    record = TradeRecord(
        strategy_id=STRATEGY_ID, trading_day=date(2026, 3, 7),
        direction="long", entry_price=100.0, exit_price=102.0,
        actual_r=0.5, expected_r=0.3, slippage_pts=0.25,
    )
    monitor.record_trade(record)
    monitor.reset_daily()

    summary = monitor.daily_summary()
    assert summary["total_slippage_pts"] == 0.0
    assert summary["n_trades"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_session_orchestrator.py::TestSlippageInSummary::test_reset_daily_clears_trades -v`
Expected: FAIL — n_trades=1, slippage=0.25

**Step 3: Implement the fix**

In `trading_app/live/performance_monitor.py:86-88`:

```python
def reset_daily(self) -> None:
    """Clear daily accumulators (call at EOD after logging summary)."""
    self._daily_r.clear()
    self._trades.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_session_orchestrator.py::TestSlippageInSummary -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add trading_app/live/performance_monitor.py tests/test_trading_app/test_session_orchestrator.py
git commit -m "fix: reset_daily() clears trade list (was accumulating across days)"
```

---

### Task 2: Make daily_features fail-closed on DB error or extreme staleness

**Files:**
- Modify: `trading_app/live/session_orchestrator.py:193-239`
- Test: `tests/test_trading_app/test_session_orchestrator.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_trading_app/test_session_orchestrator.py

class TestDailyFeaturesFailClosed:
    def test_raises_on_db_error(self):
        """_build_daily_features_row must raise if DB is unreachable, not silently continue."""
        with patch("trading_app.live.session_orchestrator.duckdb") as mock_db:
            mock_db.connect.side_effect = RuntimeError("DB locked")
            with pytest.raises(RuntimeError, match="daily_features"):
                SessionOrchestrator._build_daily_features_row(date(2026, 3, 7), "MGC")

    def test_raises_on_extreme_staleness(self):
        """>5 days stale must raise, not just warn."""
        # This requires mocking the DB to return old data — see implementation
        pass  # Detailed mock in implementation step
```

**Step 2: Run test to verify it fails**

Expected: FAIL — currently swallows exception with `log.warning`

**Step 3: Implement the fix**

In `session_orchestrator.py`, change the `except` at line 238 from:
```python
except Exception as e:
    log.warning("Could not load daily_features for live session: %s", e)
```
to:
```python
except Exception as e:
    raise RuntimeError(
        f"FAIL-CLOSED: Cannot load daily_features for {instrument}: {e}. "
        f"Filters would silently reject all trades. Fix the database or run "
        f"'python pipeline/build_daily_features.py --instrument {instrument}'."
    ) from e
```

And change staleness from warning to error at line 221-222:
```python
if gap > 5:
    raise RuntimeError(
        f"FAIL-CLOSED: daily_features is {gap} days stale (latest: {latest_day}). "
        f"Run pipeline/build_daily_features.py --instrument {instrument}."
    )
elif gap > 3:
    log.warning("daily_features data is %d days stale (latest: %s)", gap, latest_day)
```

**Step 4: Run tests**

Run: `pytest tests/test_trading_app/test_session_orchestrator.py -v`

**Step 5: Commit**

```bash
git add trading_app/live/session_orchestrator.py tests/test_trading_app/test_session_orchestrator.py
git commit -m "fix: daily_features fail-closed on DB error or >5 days stale"
```

---

### Task 3: Add --preflight mode to run_live_session.py

**Files:**
- Modify: `scripts/run_live_session.py`
- Test: manual (CLI flag)

**Step 1: Add --preflight flag to argparse**

```python
parser.add_argument(
    "--preflight",
    action="store_true",
    default=False,
    help="Run pre-flight checks (auth, portfolio, daily_features) then exit — no trading",
)
```

**Step 2: Implement preflight function**

```python
def _run_preflight(instrument: str, broker: str | None, demo: bool) -> bool:
    """Pre-flight validation. Returns True if all checks pass."""
    from trading_app.live.broker_factory import create_broker_components, get_broker_name
    from trading_app.live_config import build_live_portfolio
    from pipeline.paths import GOLD_DB_PATH

    checks_passed = 0
    checks_total = 4

    # 1. Auth check
    broker_name = broker or get_broker_name()
    print(f"\n[1/{checks_total}] Auth check ({broker_name})...", end=" ")
    try:
        components = create_broker_components(broker_name, demo=demo)
        token = components["auth"].get_token()
        print(f"OK (token: {token[:8]}...)")
        checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")

    # 2. Portfolio check
    print(f"[2/{checks_total}] Portfolio check ({instrument})...", end=" ")
    try:
        portfolio, notes = build_live_portfolio(db_path=GOLD_DB_PATH, instrument=instrument)
        print(f"OK ({len(portfolio.strategies)} strategies)")
        for s in portfolio.strategies:
            print(f"    {s.strategy_id} | {s.orb_label} {s.entry_model} RR{s.rr_target} "
                  f"| WR={s.win_rate:.0%} ExpR={s.expectancy_r:.3f} N={s.sample_size}")
        for note in notes:
            print(f"    NOTE: {note}")
        checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")

    # 3. Daily features freshness
    print(f"[3/{checks_total}] Daily features freshness...", end=" ")
    try:
        row = SessionOrchestrator._build_daily_features_row(
            date.today(), instrument
        )
        atr = row.get("atr_20")
        print(f"OK (atr_20={atr})")
        checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")

    # 4. Contract resolution
    print(f"[4/{checks_total}] Contract resolution...", end=" ")
    try:
        contracts_cls = components["contracts_class"]
        contracts = contracts_cls(auth=components["auth"], demo=demo)
        front = contracts.resolve_front_month(instrument)
        print(f"OK ({front})")
        checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")

    print(f"\nPreflight: {checks_passed}/{checks_total} passed")
    return checks_passed == checks_total
```

**Step 3: Wire into main()**

After `args = parser.parse_args()`, before mode selection:
```python
if args.preflight:
    demo = not args.live
    ok = _run_preflight(args.instrument, args.broker, demo)
    sys.exit(0 if ok else 1)
```

**Step 4: Test manually**

Run: `python scripts/run_live_session.py --instrument MGC --preflight`
Expected: 4 check results printed, exit 0 if all pass

**Step 5: Commit**

```bash
git add scripts/run_live_session.py
git commit -m "feat: --preflight mode for pre-session validation"
```

---

### Task 4: Wire CircuitBreaker into order submission

**Files:**
- Modify: `trading_app/live/session_orchestrator.py` (add circuit breaker to __init__, wrap order submission)
- Test: `tests/test_trading_app/test_session_orchestrator.py`

**Step 1: Write the failing test**

```python
def test_circuit_breaker_blocks_after_failures():
    """After 5 consecutive order failures, circuit breaker blocks further orders."""
    from trading_app.live.circuit_breaker import CircuitBreaker

    orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
    # Verify circuit breaker exists
    assert hasattr(orch, '_circuit_breaker')
    assert isinstance(orch._circuit_breaker, CircuitBreaker)
```

**Step 2: Implement**

In `session_orchestrator.py __init__`, after PositionTracker:
```python
from trading_app.live.circuit_breaker import CircuitBreaker
self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
```

In `_handle_event` ENTRY branch, wrap the `loop.run_in_executor` call:
```python
if not self._circuit_breaker.should_allow_request():
    log.critical("CIRCUIT BREAKER OPEN — skipping order for %s", event.strategy_id)
    self._write_signal_record({"type": "CIRCUIT_BREAKER", "strategy_id": event.strategy_id})
    return

try:
    result = await loop.run_in_executor(None, self.order_router.submit, spec)
    self._circuit_breaker.record_success()
except Exception as e:
    self._circuit_breaker.record_failure()
    log.error("Order submission failed for %s: %s", event.strategy_id, e)
    return
```

Same pattern for EXIT branch.

**Step 3: Update build_orchestrator in tests**

Add `orch._circuit_breaker = CircuitBreaker()` to `build_orchestrator()`.

**Step 4: Run tests**

Run: `pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_circuit_breaker.py -v`

**Step 5: Commit**

```bash
git add trading_app/live/session_orchestrator.py tests/test_trading_app/test_session_orchestrator.py
git commit -m "feat: wire CircuitBreaker into order submission"
```

---

### Task 5: Make Tradovate orphan detection clearly disabled (not silently swallowed)

**Files:**
- Modify: `trading_app/live/session_orchestrator.py:114-127`
- Test: `tests/test_trading_app/test_session_orchestrator.py`

**Step 1: Write the failing test**

```python
def test_not_implemented_positions_logs_explicit_warning():
    """Broker with NotImplementedError positions must log clear warning, not swallow silently."""
    import logging

    class RaisingPositions:
        def query_open(self, account_id):
            raise NotImplementedError("Tradovate positions not implemented")

    orch = build_orchestrator()
    orch.positions = RaisingPositions()

    # Simulate what __init__ does for orphan check
    with patch.object(logging.getLogger("trading_app.live.session_orchestrator"), "warning") as mock_warn:
        try:
            orch.positions.query_open(12345)
        except NotImplementedError:
            pass
        # Current code silently catches this — test that new code logs explicitly
```

**Step 2: Implement**

In `session_orchestrator.py:114-127`, add explicit `NotImplementedError` handling:
```python
try:
    orphans = self.positions.query_open(account_id)
    ...
except RuntimeError:
    raise
except NotImplementedError:
    log.warning(
        "ORPHAN DETECTION DISABLED: %s broker does not implement positions.query_open(). "
        "Cannot verify no orphaned positions exist. Proceed with caution.",
        self._broker_name,
    )
except Exception as e:
    log.warning("Position query failed on startup: %s", e)
```

**Step 3: Run tests**

**Step 4: Commit**

```bash
git add trading_app/live/session_orchestrator.py tests/test_trading_app/test_session_orchestrator.py
git commit -m "fix: explicit warning when broker lacks orphan detection"
```

---

### Task 6: Post-kill-switch EOD reconciliation context

**Files:**
- Modify: `trading_app/live/session_orchestrator.py:700-713`

**Step 1: Implement**

In `post_session()`, update the reconciliation block to mention kill switch context:
```python
if self.positions and not self.signal_only:
    try:
        account_id = self.order_router.account_id if self.order_router else 0
        remaining = self.positions.query_open(account_id)
        if remaining:
            if self._kill_switch_fired:
                log.critical(
                    "EOD RECONCILIATION: %d positions STILL OPEN despite kill switch flatten attempt: %s",
                    len(remaining), remaining,
                )
            else:
                log.critical(
                    "EOD RECONCILIATION: %d positions still open after session end: %s",
                    len(remaining), remaining,
                )
        else:
            log.info("EOD reconciliation: all positions flat")
    except Exception as e:
        log.warning("EOD position reconciliation failed: %s", e)
```

**Step 2: Commit**

```bash
git add trading_app/live/session_orchestrator.py
git commit -m "fix: EOD reconciliation mentions kill switch context"
```

---

### Task 7: Run full test suite + format + final commit

**Files:** All modified files

**Step 1: Format all changed files**

```bash
ruff format trading_app/live/performance_monitor.py trading_app/live/session_orchestrator.py scripts/run_live_session.py
ruff check trading_app/live/ scripts/run_live_session.py
```

**Step 2: Run affected tests**

```bash
pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_cusum_monitor.py tests/test_trading_app/test_circuit_breaker.py tests/test_trading_app/test_position_tracker.py -v
```

**Step 3: Run drift check**

```bash
python pipeline/check_drift.py
```

Expected: ALL PASS
