# Phase 6: Live Trading Preparation — Design Document

**Date**: 2026-02-08
**Status**: IMPLEMENTED (6a-6d DONE, 6e TODO). Full codebase audit completed.
**Dependencies**: Phase 5b complete (entry models + win PnL fix + DB rebuild)

---

## Executive Summary

Phase 6 transforms the backtested strategy pipeline into a live-tradeable system.
Five sub-phases, each independently testable, each with clear gates before advancing.

**Hard constraint**: No live money until paper trading proves the system matches backtest expectations.

---

## Sub-Phase Architecture

```
Phase 6a: Portfolio Construction     ← SELECT which strategies to trade
Phase 6b: Execution Engine           ← HOW to detect and execute trades
Phase 6c: Risk Management            ← LIMITS and circuit breakers
Phase 6d: Paper Trading              ← PROVE the system works forward
Phase 6e: Monitoring & Alerting      ← DETECT when reality diverges from backtest
```

Each sub-phase produces a working module with tests. No sub-phase depends on
a later one (strict dependency order: 6a → 6b → 6c → 6d → 6e).

---

## Phase 6a: Portfolio Construction

### Purpose
Select a diversified subset of validated strategies for live trading.
Size positions. Estimate daily capital requirements.

### Module: `trading_app/portfolio.py`

### Key Functions

```python
@dataclass(frozen=True)
class PortfolioStrategy:
    strategy_id: str
    instrument: str
    orb_label: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str
    weight: float           # allocation weight (0.0–1.0)
    max_contracts: int      # position limit for this strategy
    expectancy_r: float
    win_rate: float
    sample_size: int

@dataclass(frozen=True)
class Portfolio:
    name: str
    strategies: list[PortfolioStrategy]
    max_concurrent_positions: int
    max_daily_loss_r: float
    account_equity: float
    risk_per_trade_pct: float

def build_portfolio(
    db_path: Path,
    instrument: str,
    max_strategies: int = 20,
    min_expectancy_r: float = 0.10,
    diversify_by: list[str] = ["orb_label", "entry_model"],
) -> Portfolio:
    """Select strategies from validated_setups, diversified across dimensions."""

def compute_position_size(
    portfolio: Portfolio,
    strategy: PortfolioStrategy,
    cost_spec: CostSpec,
    risk_points: float,
) -> int:
    """Contracts = (equity * risk%) / (risk_points * point_value)."""

def estimate_daily_capital(portfolio: Portfolio, cost_spec: CostSpec) -> dict:
    """Estimate margin, max concurrent risk, worst-case daily draw."""

def correlation_matrix(db_path: Path, strategy_ids: list[str]) -> pd.DataFrame:
    """Daily R-series correlation between strategies from orb_outcomes."""
```

### Selection Criteria (ranked)
1. ExpR > min threshold (e.g., 0.15R)
2. All years positive (already enforced by validation)
3. Stress test passed (already enforced)
4. Diversify across ORB labels (max N per ORB)
5. Diversify across entry models (at least 2 of E1/E2/E3)
6. Prefer lower correlation (< 0.5 pairwise daily R)
7. Prefer higher sample size (more statistical confidence)

### Position Sizing
From CANONICAL_LOGIC.txt Section 3:
```
PositionSize = (AccountEquity * Risk%) / (|Entry - Stop| * PointValue)
```

For MGC at $10/point:
- $25K account, 2% risk, 10pt ORB = $25K * 0.02 / ($10 * 10) = 5 contracts
- $25K account, 2% risk, 3pt ORB = $25K * 0.02 / ($10 * 3) = 16 contracts

### Prop Firm Mode (from CANONICAL_LOGIC.txt Section 3)
```
True_Risk% = RiskAmount / MaxDrawdown   (NOT / AccountBalance)
```

### Output
- `portfolio.json` — serialized Portfolio with all strategies and sizing
- Console summary: strategy count, total weight, estimated daily trades, capital needs

### Tests (~20)
- Selection filters work correctly
- Diversification enforced
- Position sizing matches formula
- Prop firm mode calculates True Risk%
- Correlation matrix is symmetric, diagonal = 1.0
- Edge cases: 0 validated strategies, 1 strategy, all same ORB

### Gate
- Portfolio builds from validated_setups without error
- At least 5 strategies selected (diversified)
- Position sizes are reasonable (not 0, not absurdly large)

---

## Phase 6b: Execution Engine

### Purpose
Detect ORB formations, confirm bar signals, and generate orders in real-time
from a stream of 1-minute bars.

### Module: `trading_app/execution_engine.py`

### State Machine

```
IDLE → WATCHING → ARMED → CONFIRMING → ENTERED → EXITED
```

| State | Trigger | Action |
|-------|---------|--------|
| IDLE | New trading day starts (09:00 Brisbane) | Reset all state, load portfolio |
| WATCHING | During ORB formation window | Accumulate bars, build ORB high/low |
| ARMED | ORB window closes, break detected | Start monitoring for confirm bars |
| CONFIRMING | Consecutive closes outside ORB | Count confirm bars per strategy CB |
| ENTERED | Confirm bars met, entry resolved | Place entry order (model-specific) |
| EXITED | Target hit / stop hit / session end | Log outcome, update daily PnL |

### Key Classes

```python
@dataclass
class LiveORB:
    label: str              # "0900", "2300", etc.
    window_start_utc: datetime
    window_end_utc: datetime
    high: float | None
    low: float | None
    size: float | None
    break_dir: str | None   # "long" / "short" / None
    break_ts: datetime | None

@dataclass
class ActiveTrade:
    strategy_id: str
    entry_model: str
    entry_ts: datetime
    entry_price: float
    stop_price: float
    target_price: float
    direction: str          # "long" / "short"
    contracts: int
    state: str              # "CONFIRMING" / "ENTERED" / "EXITED"
    confirm_count: int
    confirm_needed: int
    # Outcome tracking
    exit_ts: datetime | None
    exit_price: float | None
    pnl_r: float | None
    mae_r: float | None
    mfe_r: float | None

class ExecutionEngine:
    def __init__(self, portfolio: Portfolio, cost_spec: CostSpec):
        ...

    def on_bar(self, bar: dict) -> list[TradeEvent]:
        """Process one 1m bar. Returns list of events (entry, exit, etc.)."""

    def on_trading_day_start(self, trading_day: date):
        """Reset state for new trading day."""

    def on_trading_day_end(self):
        """Close all open positions as scratch. Log daily summary."""

    def get_active_trades(self) -> list[ActiveTrade]:
        """Return all currently active trades."""

    def get_daily_summary(self) -> dict:
        """Return PnL, trade count, win rate for current day."""
```

### Bar Processing Logic (on_bar)
1. For each ORB label: if within ORB window, update high/low
2. For each ORB label: if past ORB window, check for break
3. For each break: for each matching strategy in portfolio:
   a. If ARMED: check confirm bar condition
   b. If confirm_count == confirm_needed: resolve entry (E1/E2/E3)
   c. If ENTERED: check target/stop/session-end
4. Apply ORB size filter (matches_row from config)
5. Check risk limits before entering

### Order Generation (NOT broker-specific)
The engine generates abstract `TradeEvent` objects:
```python
@dataclass
class TradeEvent:
    event_type: str         # "ENTRY", "EXIT", "SCRATCH"
    strategy_id: str
    timestamp: datetime
    price: float
    direction: str
    contracts: int
    reason: str             # "confirm_bars_met", "target_hit", "stop_hit", "session_end"
```

These are broker-agnostic. A separate adapter layer (Phase 6d+) converts them
to actual broker orders (IB, NinjaTrader, etc.).

### Reuse of Existing Code
- `detect_confirm()` from `entry_rules.py` — used directly
- `resolve_entry()` from `entry_rules.py` — used directly
- `CostSpec` from `cost_model.py` — used for R-multiple calculations
- ORB window logic from `build_daily_features.py` — extracted/shared

### Tests (~30)
- State machine transitions (all valid paths)
- ORB detection matches build_daily_features output on historical data
- Confirm bar counting matches entry_rules.py
- Entry model resolution (E1/E2/E3) matches outcome_builder
- Target/stop detection matches outcome_builder
- Session-end scratch logic
- Multiple concurrent trades across different ORBs
- Filter application (ORB size filters)
- Edge: no bars in window, zero-size ORB, break in both directions

### Gate
- Replay 10 historical trading days through ExecutionEngine
- Compare every entry/exit to orb_outcomes table
- 100% match required (same entry_price, same outcome, same pnl_r)
- Any mismatch = bug in engine or builder (must resolve before continuing)

---

## Phase 6c: Risk Management

### Purpose
Prevent catastrophic losses. Enforce position limits. Circuit-break on bad days.

### Module: `trading_app/risk_manager.py`

### Key Functions

```python
@dataclass(frozen=True)
class RiskLimits:
    max_daily_loss_r: float         # e.g., -5.0R total across all trades
    max_concurrent_positions: int   # e.g., 3
    max_per_orb_positions: int      # e.g., 1 (one trade per ORB at a time)
    max_daily_trades: int           # e.g., 15
    drawdown_warning_r: float       # e.g., -3.0R (log warning)
    drawdown_halt_r: float          # e.g., -5.0R (stop trading)

class RiskManager:
    def __init__(self, limits: RiskLimits):
        ...

    def can_enter(self, trade: ActiveTrade, engine: ExecutionEngine) -> tuple[bool, str]:
        """Check all risk limits before allowing entry. Returns (allowed, reason)."""

    def on_trade_exit(self, trade: ActiveTrade):
        """Update daily PnL tracking after trade exits."""

    def is_halted(self) -> bool:
        """True if daily loss limit hit (circuit breaker)."""

    def daily_reset(self):
        """Reset daily counters for new trading day."""
```

### Risk Checks (ordered, fail-fast)
1. **Circuit breaker**: If `daily_pnl_r <= drawdown_halt_r`, reject ALL entries
2. **Max concurrent**: If `len(active_trades) >= max_concurrent_positions`, reject
3. **Max per ORB**: If already have a position from this ORB label, reject
4. **Max daily trades**: If `daily_trade_count >= max_daily_trades`, reject
5. **Drawdown warning**: If `daily_pnl_r <= drawdown_warning_r`, log warning but allow

### Tests (~15)
- Circuit breaker triggers at threshold
- Max concurrent enforced
- Per-ORB limit enforced
- Daily reset clears all counters
- Edge: exactly at limit, one below/above

### Gate
- Risk manager prevents entry when limits are hit
- Cannot bypass risk checks (no backdoor)

---

## Phase 6d: Paper Trading / Simulation

### Purpose
Prove the system works on forward data before risking money.
Two modes: historical replay and live paper trading.

### Module: `trading_app/paper_trader.py`

### Mode 1: Historical Replay
```python
def replay_historical(
    db_path: Path,
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
    risk_limits: RiskLimits,
) -> ReplayResult:
    """
    Feed historical bars_1m through ExecutionEngine + RiskManager.
    Compare results to orb_outcomes for validation.
    """
```

This is the CRITICAL validation step. It feeds real historical bars through
the execution engine and compares every trade to the pre-computed outcomes.
Any mismatch means the execution engine has a bug.

### Mode 2: Walk-Forward Validation
```python
def walk_forward(
    db_path: Path,
    instrument: str,
    train_years: int = 3,
    test_years: int = 1,
) -> list[WalkForwardResult]:
    """
    Train (discover+validate) on N years, test on year N+1.
    Roll forward. Report OOS degradation.
    """
```

From CANONICAL_LOGIC.txt Section 4: "Material degradation OOS = non-robust strategy."

### Mode 3: Live Paper Trading (Future)
Connect to real-time data feed (Databento live or broker API).
Run ExecutionEngine on live bars. Log all events. No real orders.
This is deferred until Modes 1-2 pass.

### Output: Trade Journal
Every trade logged to `trade_journal` table:
```sql
CREATE TABLE IF NOT EXISTS trade_journal (
    journal_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL,             -- 'replay', 'walk_forward', 'paper', 'live'
    trading_day DATE NOT NULL,
    strategy_id TEXT NOT NULL,
    entry_model TEXT NOT NULL,
    entry_ts TIMESTAMPTZ,
    entry_price DOUBLE,
    stop_price DOUBLE,
    target_price DOUBLE,
    direction TEXT,
    contracts INTEGER,
    exit_ts TIMESTAMPTZ,
    exit_price DOUBLE,
    outcome TEXT,
    pnl_r DOUBLE,
    pnl_dollars DOUBLE,
    mae_r DOUBLE,
    mfe_r DOUBLE,
    risk_check_passed BOOLEAN,
    risk_check_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Tests (~20)
- Replay matches orb_outcomes exactly
- Walk-forward splits data correctly
- Trade journal records all fields
- OOS degradation flagged correctly
- Risk limits applied during replay

### Gate
- Historical replay on FULL dataset matches orb_outcomes within tolerance
- Walk-forward shows no material degradation (OOS ExpR > 50% of IS ExpR)
- Trade journal is complete and queryable

---

## Phase 6e: Monitoring & Alerting

### Purpose
Detect when live performance diverges from backtest expectations.
Alert on regime changes, drawdowns, and anomalies.

### Module: `trading_app/monitor.py`

### Alerts

| Alert | Trigger | Severity |
|-------|---------|----------|
| Drawdown | Daily PnL < -3R | WARNING |
| Circuit Break | Daily PnL < -5R | CRITICAL |
| Win Rate Drift | Rolling 50-trade WR < backtest WR - 10pp | WARNING |
| ExpR Drift | Rolling 50-trade ExpR < 50% of backtest | CRITICAL |
| ORB Size Regime | 30-day median ORB size > 2x backtest median | INFO |
| Missing Data | Expected bar count < 80% of normal | WARNING |
| Strategy Stale | No trade in 30+ calendar days | INFO |

### Dashboard
Build on existing `pipeline/dashboard.py` pattern (self-contained HTML).
Add panels:
- Active portfolio strategies
- Daily PnL curve (cumulative R)
- Per-strategy performance vs backtest
- Risk utilization (% of limits used)
- ORB size regime tracker

### Tests (~10)
- Alert triggers at correct thresholds
- Dashboard generates valid HTML
- Regime detection works on synthetic data

### Gate
- All alerts fire on test data
- Dashboard displays correctly with real data

---

## Implementation Order

```
Week 1:  Phase 6a (Portfolio Construction)
         - portfolio.py + tests
         - Build initial portfolio from post-5b validated_setups

Week 2:  Phase 6b (Execution Engine)
         - execution_engine.py + tests
         - State machine + bar processing
         - Historical validation gate (replay must match)

Week 3:  Phase 6c (Risk Management)
         - risk_manager.py + tests
         - Wire into execution engine

Week 4:  Phase 6d Part 1 (Historical Replay)
         - paper_trader.py replay mode
         - Full dataset replay validation
         - Walk-forward validation

Week 5:  Phase 6d Part 2 + 6e (Paper Trading + Monitoring)
         - Trade journal
         - monitor.py + dashboard
         - Live paper trading setup (data feed TBD)
```

---

## Open Questions (Need User Decision)

1. **Account size and risk %**: What account equity and risk per trade to design for?
   - Options: $25K/2%, $50K/1%, $100K/0.5%, prop firm mode

2. **Max portfolio size**: How many strategies to run concurrently?
   - Options: 5 (conservative), 10 (balanced), 20 (aggressive)

3. **Live data source**: Where will real-time 1m bars come from?
   - Options: Databento live feed, Interactive Brokers API, NinjaTrader, manual CSV

4. **Broker integration**: Which broker for order execution?
   - Options: Interactive Brokers (TWS API), NinjaTrader, defer (paper only first)

5. **Prop firm mode**: Is this for a prop firm or personal account?
   - Affects position sizing formula (max drawdown vs equity based)

---

## File Structure (New Files)

```
trading_app/
    portfolio.py            # 6a: Portfolio construction + sizing
    execution_engine.py     # 6b: State machine + bar processing
    risk_manager.py         # 6c: Risk limits + circuit breaker
    paper_trader.py         # 6d: Replay + walk-forward + paper
    monitor.py              # 6e: Alerts + drift detection
    trade_journal.py        # Shared: trade logging to DB

tests/test_trading_app/
    test_portfolio.py
    test_execution_engine.py
    test_risk_manager.py
    test_paper_trader.py
    test_monitor.py
```

---

## Dependencies (New)

None required beyond what exists. All modules use:
- DuckDB (already installed)
- pandas/numpy (already installed)
- pipeline.cost_model (exists)
- trading_app.entry_rules (exists)
- trading_app.config (exists)

Live paper trading (6d Mode 3) will need a data feed library (TBD).
Broker integration will need broker API library (TBD).
These are deferred — not needed until Modes 1-2 pass.
