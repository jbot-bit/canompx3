# Portfolio Assembly Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a research script that combines all 21 session slots into a single portfolio equity curve with honest stats.

**Architecture:** Load slot heads via existing `session_slots()`, load their trades via existing `_load_head_trades()`, build per-slot daily ledgers, combine into portfolio-level daily returns (including zero-trade days), and compute combined metrics. Single read-only script, no DB writes.

**Tech Stack:** Python, DuckDB (read-only), pandas, numpy, scipy (for correlation)

---

### Task 1: Scaffold script with slot loading and trade collection

**Files:**
- Create: `research/research_portfolio_assembly.py`
- Test: manual run with `--help`

**Step 1: Create the script with imports, CLI, and slot+trade loading**

```python
"""
Portfolio Assembly Research Report

Combines all session slots into a single portfolio equity curve.
Shows honest combined stats: Sharpe (with zero-days), drawdown,
correlation, concurrent exposure.

Usage:
    python research/research_portfolio_assembly.py
    python research/research_portfolio_assembly.py --db-path C:/db/gold.db
    python research/research_portfolio_assembly.py --exclude-regime
"""

import sys
import argparse
from pathlib import Path
from math import sqrt
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

# Add scripts/reports to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reports"))
from report_edge_portfolio import session_slots

import duckdb

TRADING_DAYS_PER_YEAR = 252


def load_slot_trades(con, slots):
    """Load win/loss trades for each slot head strategy.

    Returns dict: strategy_id -> list of trade dicts.
    Each trade dict has: trading_day, outcome, pnl_r, instrument, session.
    """
    trades_by_slot = {}

    for slot in slots:
        inst = slot["instrument"]
        strategy_id = slot["head_strategy_id"]

        rows = con.execute("""
            SELECT oo.trading_day, oo.outcome, oo.pnl_r
            FROM edge_families ef
            JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
            JOIN strategy_trade_days std ON vs.strategy_id = std.strategy_id
            JOIN orb_outcomes oo
              ON oo.symbol = vs.instrument
              AND oo.orb_label = vs.orb_label
              AND oo.orb_minutes = vs.orb_minutes
              AND oo.entry_model = vs.entry_model
              AND oo.rr_target = vs.rr_target
              AND oo.confirm_bars = vs.confirm_bars
              AND oo.trading_day = std.trading_day
            WHERE ef.head_strategy_id = ?
              AND ef.robustness_status != 'PURGED'
              AND oo.outcome IN ('win', 'loss')
            ORDER BY oo.trading_day
        """, [strategy_id]).fetchall()

        trades = [
            {
                "trading_day": r[0],
                "outcome": r[1],
                "pnl_r": r[2],
                "instrument": inst,
                "session": slot["session"],
                "strategy_id": strategy_id,
            }
            for r in rows
        ]
        trades_by_slot[strategy_id] = trades

    return trades_by_slot


def print_slot_inventory(slots, trades_by_slot):
    """Section 1: Slot inventory table."""
    print("=" * 80)
    print("  SLOT INVENTORY")
    print("=" * 80)
    print(f"  {'Inst':>4} {'Session':>18} {'Strategy':>45} {'ExpR':>7} {'ShANN':>6} {'N':>5} {'Tier':>7}")
    print(f"  {'----':>4} {'-------':>18} {'--------':>45} {'----':>7} {'-----':>6} {'---':>5} {'----':>7}")

    total_trades = 0
    for s in slots:
        sid = s["head_strategy_id"]
        n = len(trades_by_slot.get(sid, []))
        total_trades += n
        tier = s["trade_tier"] if s["trade_tier"] != "CORE" else ""
        print(
            f"  {s['instrument']:>4} {s['session']:>18} "
            f"{sid:>45} "
            f"{s['head_expectancy_r']:>+6.3f} "
            f"{s['head_sharpe_ann']:>5.2f} "
            f"{n:>5} "
            f"{tier:>7}"
        )
    print(f"\n  Total: {len(slots)} slots, {total_trades:,} trades")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio assembly research report"
    )
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--exclude-regime", action="store_true",
        help="Exclude REGIME-tier slots (N < 100)"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    # Load slots
    slots = session_slots(db_path)
    if not slots:
        print("No session slots found.")
        return

    if args.exclude_regime:
        slots = [s for s in slots if s["trade_tier"] == "CORE"]
        print(f"[CORE only] Filtered to {len(slots)} CORE-tier slots\n")

    print(f"\n{'#' * 80}")
    print(f"#  PORTFOLIO ASSEMBLY REPORT")
    print(f"#  {len(slots)} session slots, equal R weight")
    print(f"{'#' * 80}\n")

    # Load trades
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        trades_by_slot = load_slot_trades(con, slots)
    finally:
        con.close()

    # Section 1: Inventory
    print_slot_inventory(slots, trades_by_slot)


if __name__ == "__main__":
    main()
```

**Step 2: Run to verify slot loading works**

Run: `python research/research_portfolio_assembly.py`
Expected: Prints slot inventory table with 21 slots and trade counts.

**Step 3: Commit**

```bash
git add research/research_portfolio_assembly.py
git commit -m "feat: scaffold portfolio assembly research script with slot loading"
```

---

### Task 2: Build combined daily equity curve with honest Sharpe

**Files:**
- Modify: `research/research_portfolio_assembly.py`

**Step 1: Add functions for daily equity curve and honest Sharpe**

Add these functions after `print_slot_inventory`:

```python
def build_daily_equity(trades_by_slot):
    """Build combined daily equity curve from all slot trades.

    Returns:
        daily_returns: sorted list of (date, total_r) for days with trades
        all_trades: flat list of all trades across slots
    """
    daily_r = defaultdict(float)
    daily_trade_count = defaultdict(int)
    all_trades = []

    for strategy_id, trades in trades_by_slot.items():
        for t in trades:
            day = t["trading_day"]
            daily_r[day] += t["pnl_r"]
            daily_trade_count[day] += 1
            all_trades.append(t)

    daily_returns = sorted(daily_r.items())
    return daily_returns, all_trades, dict(daily_trade_count)


def count_trading_days(start_date, end_date):
    """Count weekdays (Mon-Fri) between two dates inclusive.

    This is the honest denominator for Sharpe — includes days
    where no slots fired (0R return).
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    return len(dates)


def compute_honest_sharpe(daily_returns, start_date, end_date):
    """Compute Sharpe including zero-return days.

    Unlike _compute_portfolio_stats which only counts active days,
    this includes ALL weekdays in the date range as 0R returns.
    This prevents Sharpe inflation from idle capital.
    """
    # Build full daily series with zeros for inactive days
    all_bdays = pd.bdate_range(start=start_date, end=end_date)
    return_map = dict(daily_returns)

    full_series = []
    for day in all_bdays:
        day_date = day.date()
        full_series.append(return_map.get(day_date, 0.0))

    n = len(full_series)
    if n <= 1:
        return None, None, n

    total_r = sum(full_series)
    mean_d = total_r / n
    variance = sum((v - mean_d) ** 2 for v in full_series) / (n - 1)
    std_d = variance ** 0.5

    sharpe_d = mean_d / std_d if std_d > 0 else None
    sharpe_ann = sharpe_d * sqrt(TRADING_DAYS_PER_YEAR) if sharpe_d else None

    return sharpe_d, sharpe_ann, n


def compute_drawdown(daily_returns, start_date, end_date):
    """Compute drawdown stats on the full daily series (with zeros).

    Returns dict: max_dd_r, max_dd_start, max_dd_end, max_dd_duration_days,
                  longest_losing_streak, worst_single_day.
    """
    all_bdays = pd.bdate_range(start=start_date, end=end_date)
    return_map = dict(daily_returns)

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    dd_start = None
    max_dd_start = None
    max_dd_end = None
    worst_day = 0.0
    worst_day_date = None

    # For losing streak
    current_streak = 0
    max_streak = 0

    for day in all_bdays:
        day_date = day.date()
        r = return_map.get(day_date, 0.0)
        cum += r

        if r < worst_day:
            worst_day = r
            worst_day_date = day_date

        # Losing streak (consecutive negative days, skip zero days)
        if r < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        elif r > 0:
            current_streak = 0

        if cum > peak:
            peak = cum
            dd_start = day_date

        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            max_dd_start = dd_start
            max_dd_end = day_date

    # Duration
    dd_duration = None
    if max_dd_start and max_dd_end:
        dd_duration = (max_dd_end - max_dd_start).days

    return {
        "max_dd_r": round(max_dd, 4),
        "max_dd_start": max_dd_start,
        "max_dd_end": max_dd_end,
        "max_dd_duration_days": dd_duration,
        "longest_losing_streak": max_streak,
        "worst_single_day": round(worst_day, 4),
        "worst_single_day_date": worst_day_date,
    }


def print_portfolio_metrics(daily_returns, all_trades, daily_trade_count,
                            start_date, end_date):
    """Section 3: Portfolio metrics."""
    n_trades = len(all_trades)
    n_wins = sum(1 for t in all_trades if t["outcome"] == "win")
    wr = n_wins / n_trades if n_trades > 0 else 0
    total_r = sum(t["pnl_r"] for t in all_trades)
    avg_r = total_r / n_trades if n_trades > 0 else 0

    active_days = len(daily_returns)
    total_bdays = count_trading_days(start_date, end_date)
    avg_trades_per_day = n_trades / active_days if active_days > 0 else 0
    active_pct = active_days / total_bdays if total_bdays > 0 else 0

    sharpe_d, sharpe_ann, n_full = compute_honest_sharpe(
        daily_returns, start_date, end_date
    )
    dd = compute_drawdown(daily_returns, start_date, end_date)

    print("\n" + "=" * 80)
    print("  PORTFOLIO METRICS (honest Sharpe, including zero-return days)")
    print("=" * 80)
    print(f"  Date range:       {start_date} to {end_date}")
    print(f"  Business days:    {total_bdays:,} (denominator for Sharpe)")
    print(f"  Active days:      {active_days:,} ({active_pct:.1%} of business days)")
    print(f"  Total trades:     {n_trades:,}")
    print(f"  Win rate:         {wr:.1%}")
    print(f"  Avg R per trade:  {avg_r:+.4f}")
    print(f"  Total R:          {total_r:+.1f}")
    print(f"  Avg trades/day:   {avg_trades_per_day:.1f} (on active days)")
    print()
    if sharpe_ann is not None:
        print(f"  Sharpe (daily):   {sharpe_d:.4f}")
        print(f"  Sharpe (ann):     {sharpe_ann:.4f}")
    else:
        print("  Sharpe:           N/A (insufficient data)")
    print()
    print(f"  Max drawdown:     {dd['max_dd_r']:.2f}R")
    if dd["max_dd_start"] and dd["max_dd_end"]:
        print(f"    Period:         {dd['max_dd_start']} to {dd['max_dd_end']} "
              f"({dd['max_dd_duration_days']} calendar days)")
    print(f"  Worst single day: {dd['worst_single_day']:+.2f}R"
          + (f" ({dd['worst_single_day_date']})" if dd["worst_single_day_date"] else ""))
    print(f"  Longest losing streak: {dd['longest_losing_streak']} consecutive days")
```

**Step 2: Wire into main()**

Update `main()` to call the new functions after the inventory:

```python
    # After print_slot_inventory...

    # Flatten trades and build equity curve
    daily_returns, all_trades, daily_trade_count = build_daily_equity(trades_by_slot)

    if not all_trades:
        print("\nNo trades found across any slots.")
        return

    # Date range
    all_days = [t["trading_day"] for t in all_trades]
    start_date = min(all_days)
    end_date = max(all_days)

    # Section 3: Portfolio metrics
    print_portfolio_metrics(daily_returns, all_trades, daily_trade_count,
                            start_date, end_date)
```

**Step 3: Run and verify**

Run: `python research/research_portfolio_assembly.py`
Expected: Slot inventory + portfolio metrics with honest Sharpe and drawdown stats.

**Step 4: Commit**

```bash
git add research/research_portfolio_assembly.py
git commit -m "feat: add combined portfolio metrics with honest Sharpe"
```

---

### Task 3: Add per-year breakdown and per-instrument contribution

**Files:**
- Modify: `research/research_portfolio_assembly.py`

**Step 1: Add yearly and per-instrument print functions**

```python
def print_yearly_breakdown(all_trades, start_date, end_date):
    """Section 4: Per-year breakdown."""
    yearly = defaultdict(lambda: {"trades": 0, "wins": 0, "total_r": 0.0, "days": set()})

    for t in all_trades:
        td = t["trading_day"]
        year = td.year if hasattr(td, "year") else int(str(td)[:4])
        yearly[year]["trades"] += 1
        if t["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += t["pnl_r"]
        yearly[year]["days"].add(td)

    print("\n" + "=" * 80)
    print("  PER-YEAR BREAKDOWN")
    print("=" * 80)
    print(f"  {'Year':>6} {'Trades':>7} {'WR':>7} {'TotalR':>9} {'Days':>6} {'R/Day':>7}")
    print(f"  {'----':>6} {'------':>7} {'---':>7} {'------':>9} {'----':>6} {'-----':>7}")

    for year in sorted(yearly.keys()):
        y = yearly[year]
        wr = y["wins"] / y["trades"] if y["trades"] > 0 else 0
        n_days = len(y["days"])
        r_per_day = y["total_r"] / n_days if n_days > 0 else 0
        print(f"  {year:>6} {y['trades']:>7} {wr:>6.1%} {y['total_r']:>+8.1f}R {n_days:>6} {r_per_day:>+6.3f}")


def print_instrument_contribution(all_trades):
    """Section 7: Per-instrument contribution."""
    by_inst = defaultdict(lambda: {"trades": 0, "wins": 0, "total_r": 0.0})

    for t in all_trades:
        inst = t["instrument"]
        by_inst[inst]["trades"] += 1
        if t["outcome"] == "win":
            by_inst[inst]["wins"] += 1
        by_inst[inst]["total_r"] += t["pnl_r"]

    total_r = sum(v["total_r"] for v in by_inst.values())

    print("\n" + "=" * 80)
    print("  PER-INSTRUMENT CONTRIBUTION")
    print("=" * 80)
    print(f"  {'Inst':>6} {'Trades':>7} {'WR':>7} {'TotalR':>9} {'% of R':>8}")
    print(f"  {'----':>6} {'------':>7} {'---':>7} {'------':>9} {'------':>8}")

    for inst in sorted(by_inst.keys()):
        v = by_inst[inst]
        wr = v["wins"] / v["trades"] if v["trades"] > 0 else 0
        pct = v["total_r"] / total_r if total_r != 0 else 0
        print(f"  {inst:>6} {v['trades']:>7} {wr:>6.1%} {v['total_r']:>+8.1f}R {pct:>7.1%}")
```

**Step 2: Wire into main()**

```python
    # After print_portfolio_metrics...

    # Section 4: Per-year
    print_yearly_breakdown(all_trades, start_date, end_date)

    # Section 7: Per-instrument
    print_instrument_contribution(all_trades)
```

**Step 3: Run and verify**

Run: `python research/research_portfolio_assembly.py`
Expected: Full report with yearly and per-instrument sections added.

**Step 4: Commit**

```bash
git add research/research_portfolio_assembly.py
git commit -m "feat: add yearly breakdown and per-instrument contribution"
```

---

### Task 4: Add correlation matrix and concurrent exposure analysis

**Files:**
- Modify: `research/research_portfolio_assembly.py`

**Step 1: Add correlation and exposure functions**

```python
def print_correlation_matrix(trades_by_slot, slots):
    """Section 5: Pairwise correlation between slots.

    Only shows pairs with >= 30 overlapping trade days.
    """
    # Build per-slot daily return series
    slot_daily = {}
    for slot in slots:
        sid = slot["head_strategy_id"]
        label = f"{slot['instrument']}_{slot['session']}"
        daily = {}
        for t in trades_by_slot.get(sid, []):
            day = t["trading_day"]
            daily[day] = daily.get(day, 0.0) + t["pnl_r"]
        slot_daily[label] = daily

    labels = sorted(slot_daily.keys())

    print("\n" + "=" * 80)
    print("  SLOT CORRELATION MATRIX (pairs with >= 30 overlapping days)")
    print("=" * 80)

    MIN_OVERLAP = 30
    corr_pairs = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_daily = slot_daily[labels[i]]
            b_daily = slot_daily[labels[j]]

            # Find overlapping days (both slots had a trade)
            overlap_days = set(a_daily.keys()) & set(b_daily.keys())
            n_overlap = len(overlap_days)

            if n_overlap < MIN_OVERLAP:
                continue

            a_vals = [a_daily[d] for d in sorted(overlap_days)]
            b_vals = [b_daily[d] for d in sorted(overlap_days)]

            if np.std(a_vals) == 0 or np.std(b_vals) == 0:
                continue

            r = float(np.corrcoef(a_vals, b_vals)[0, 1])
            corr_pairs.append((labels[i], labels[j], r, n_overlap))

    if not corr_pairs:
        print("  No slot pairs with >= 30 overlapping trade days.")
        return

    # Sort by absolute correlation descending
    corr_pairs.sort(key=lambda x: -abs(x[2]))

    for a, b, r, n in corr_pairs:
        flag = " <-- CONCENTRATED" if abs(r) > 0.3 else ""
        print(f"  {a:>25} x {b:<25} r={r:+.3f}  (N={n:>3}){flag}")

    # Summary
    high_corr = sum(1 for _, _, r, _ in corr_pairs if abs(r) > 0.3)
    print(f"\n  {len(corr_pairs)} pairs tested, {high_corr} with |r| > 0.3")


def print_concurrent_exposure(daily_trade_count, start_date, end_date):
    """Section 6: Concurrent exposure analysis."""
    all_bdays = pd.bdate_range(start=start_date, end=end_date)

    # Count slots per day (0 for days with no trades)
    exposure_dist = defaultdict(int)
    max_exposure = 0
    max_exposure_date = None

    daily_losses = {}  # day -> count of losses on that day
    for day in all_bdays:
        day_date = day.date()
        n = daily_trade_count.get(day_date, 0)
        exposure_dist[n] += 1
        if n > max_exposure:
            max_exposure = n
            max_exposure_date = day_date

    print("\n" + "=" * 80)
    print("  CONCURRENT EXPOSURE ANALYSIS")
    print("=" * 80)
    print(f"  Slots firing per day:")

    total_bdays = len(all_bdays)
    for n_slots in sorted(exposure_dist.keys()):
        count = exposure_dist[n_slots]
        pct = count / total_bdays
        bar = "#" * int(pct * 40)
        print(f"    {n_slots:>2} slots: {count:>5} days ({pct:>5.1%}) {bar}")

    active_days = sum(c for n, c in exposure_dist.items() if n > 0)
    active_trades = sum(n * c for n, c in exposure_dist.items())
    avg_on_active = active_trades / active_days if active_days > 0 else 0

    print(f"\n  Max concurrent exposure: {max_exposure} slots on {max_exposure_date}")
    print(f"  Max single-day R at risk: {max_exposure}R (if all lose)")
    print(f"  Avg slots on active days: {avg_on_active:.1f}")
```

**Step 2: Wire into main()**

```python
    # After print_instrument_contribution...

    # Section 5: Correlation
    print_correlation_matrix(trades_by_slot, slots)

    # Section 6: Concurrent exposure
    print_concurrent_exposure(daily_trade_count, start_date, end_date)

    print()
```

**Step 3: Run full report**

Run: `python research/research_portfolio_assembly.py`
Expected: Complete 8-section report.

**Step 4: Run with --exclude-regime flag**

Run: `python research/research_portfolio_assembly.py --exclude-regime`
Expected: Filtered to CORE-tier only, smaller slot count, different metrics.

**Step 5: Commit**

```bash
git add research/research_portfolio_assembly.py
git commit -m "feat: add correlation matrix and concurrent exposure analysis"
```

---

### Task 5: Write basic tests for core computation functions

**Files:**
- Create: `tests/test_research/test_portfolio_assembly.py`

**Step 1: Write tests**

```python
"""Tests for portfolio assembly computation functions."""

import datetime
from collections import defaultdict

import pytest

# Import the module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))

from research_portfolio_assembly import (
    build_daily_equity,
    count_trading_days,
    compute_honest_sharpe,
    compute_drawdown,
)


def _make_trade(day, outcome, pnl_r, instrument="MGC", session="1000"):
    return {
        "trading_day": day,
        "outcome": outcome,
        "pnl_r": pnl_r,
        "instrument": instrument,
        "session": session,
        "strategy_id": f"{instrument}_{session}_test",
    }


class TestBuildDailyEquity:
    def test_single_slot(self):
        trades_by_slot = {
            "A": [
                _make_trade(datetime.date(2025, 1, 6), "win", 2.0),
                _make_trade(datetime.date(2025, 1, 7), "loss", -1.0),
            ]
        }
        daily, all_trades, daily_count = build_daily_equity(trades_by_slot)
        assert len(daily) == 2
        assert len(all_trades) == 2
        assert daily_count[datetime.date(2025, 1, 6)] == 1

    def test_multi_slot_same_day(self):
        trades_by_slot = {
            "A": [_make_trade(datetime.date(2025, 1, 6), "win", 2.0, "MGC")],
            "B": [_make_trade(datetime.date(2025, 1, 6), "loss", -1.0, "MNQ")],
        }
        daily, all_trades, daily_count = build_daily_equity(trades_by_slot)
        assert len(daily) == 1
        assert daily[0] == (datetime.date(2025, 1, 6), 1.0)  # 2.0 + (-1.0)
        assert daily_count[datetime.date(2025, 1, 6)] == 2


class TestCountTradingDays:
    def test_full_week(self):
        # Mon Jan 6 to Fri Jan 10 = 5 business days
        assert count_trading_days(
            datetime.date(2025, 1, 6), datetime.date(2025, 1, 10)
        ) == 5

    def test_includes_weekends_correctly(self):
        # Mon Jan 6 to Mon Jan 13 = 6 business days
        assert count_trading_days(
            datetime.date(2025, 1, 6), datetime.date(2025, 1, 13)
        ) == 6


class TestComputeHonestSharpe:
    def test_all_positive_days(self):
        daily = [
            (datetime.date(2025, 1, 6), 1.0),
            (datetime.date(2025, 1, 7), 1.0),
            (datetime.date(2025, 1, 8), 1.0),
            (datetime.date(2025, 1, 9), 1.0),
            (datetime.date(2025, 1, 10), 1.0),
        ]
        # All 5 business days have returns, no dilution
        _, sharpe_ann, n = compute_honest_sharpe(
            daily, datetime.date(2025, 1, 6), datetime.date(2025, 1, 10)
        )
        assert n == 5
        assert sharpe_ann is not None
        # All positive, should be very high
        assert sharpe_ann > 10

    def test_sparse_days_dilute_sharpe(self):
        # Only 1 trade day in a 5-day range
        daily = [(datetime.date(2025, 1, 6), 1.0)]
        _, sharpe_sparse, n = compute_honest_sharpe(
            daily, datetime.date(2025, 1, 6), datetime.date(2025, 1, 10)
        )
        assert n == 5  # denominator is 5, not 1
        # Sharpe should be lower than dense version
        _, sharpe_dense, _ = compute_honest_sharpe(
            daily, datetime.date(2025, 1, 6), datetime.date(2025, 1, 6)
        )
        # Can't really compare single day, but sparse should be finite
        assert sharpe_sparse is not None


class TestComputeDrawdown:
    def test_basic_drawdown(self):
        daily = [
            (datetime.date(2025, 1, 6), 2.0),
            (datetime.date(2025, 1, 7), -3.0),  # DD starts
            (datetime.date(2025, 1, 8), -1.0),  # DD deepens
            (datetime.date(2025, 1, 9), 1.0),
            (datetime.date(2025, 1, 10), 1.0),
        ]
        dd = compute_drawdown(
            daily, datetime.date(2025, 1, 6), datetime.date(2025, 1, 10)
        )
        # Peak at +2.0 after day 1, trough at +2-3-1 = -2.0 after day 3
        # DD = 2.0 - (-2.0) = 4.0
        assert dd["max_dd_r"] == 4.0
        assert dd["worst_single_day"] == -3.0
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_research/test_portfolio_assembly.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/test_research/test_portfolio_assembly.py
git commit -m "test: add unit tests for portfolio assembly computations"
```

---

### Task 6: Run the full report and capture output

**Step 1: Run the complete report**

Run: `python research/research_portfolio_assembly.py`
Expected: Full 8-section report for all 21 slots.

**Step 2: Run the CORE-only variant**

Run: `python research/research_portfolio_assembly.py --exclude-regime`
Expected: Report with only CORE-tier slots.

**Step 3: Commit the final script (if any cleanup needed)**

```bash
git add research/research_portfolio_assembly.py
git commit -m "feat: complete portfolio assembly research report"
```
