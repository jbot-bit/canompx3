"""Portfolio Assembly Research Report — Combines all session slots into a single portfolio equity curve with honest stats."""

import sys
import argparse
from pathlib import Path
from math import sqrt
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, VolumeFilter
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_daily_features,
)

# Add scripts/reports to sys.path for session_slots import
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reports"))
from report_edge_portfolio import session_slots

sys.stdout.reconfigure(line_buffering=True)

TRADING_DAYS_PER_YEAR = 252


def _get_strategy_params(con, strategy_id):
    """Look up strategy parameters from validated_setups."""
    row = con.execute("""
        SELECT instrument, orb_label, orb_minutes, entry_model,
               rr_target, confirm_bars, filter_type
        FROM validated_setups
        WHERE strategy_id = ?
    """, [strategy_id]).fetchone()
    if not row:
        return None
    cols = ["instrument", "orb_label", "orb_minutes", "entry_model",
            "rr_target", "confirm_bars", "filter_type"]
    return dict(zip(cols, row))


def load_slot_trades(con, slots):
    """Load trades for each slot's head strategy using canonical filter logic.

    Bypasses strategy_trade_days (incomplete for E0 strategies).
    Instead applies filters via daily_features using the same logic
    as strategy_discovery.py.

    Returns dict: strategy_id -> list of trade dicts.
    """
    trades_by_slot = {}

    # Group slots by instrument to batch filter computation
    by_instrument = defaultdict(list)
    for slot in slots:
        by_instrument[slot["instrument"]].append(slot)

    for instrument, inst_slots in by_instrument.items():
        # Collect filter types and sessions needed for this instrument
        slot_params = {}
        filter_types = set()
        orb_labels = set()
        for slot in inst_slots:
            params = _get_strategy_params(con, slot["head_strategy_id"])
            if params is None:
                print(f"  WARNING: {slot['head_strategy_id']} not in validated_setups")
                continue
            slot_params[slot["head_strategy_id"]] = params
            filter_types.add(params["filter_type"])
            orb_labels.add(params["orb_label"])

        if not slot_params:
            continue

        orb_labels = sorted(orb_labels)
        needed_filters = {k: v for k, v in ALL_FILTERS.items() if k in filter_types}

        # Handle filters not in ALL_FILTERS (e.g. DIR_LONG, ORB_G4_L12)
        missing = filter_types - set(needed_filters.keys())
        if missing:
            from trading_app.config import DirectionFilter, OrbSizeFilter, CompositeFilter, BreakSpeedFilter
            for ft in missing:
                if ft == "DIR_LONG":
                    needed_filters[ft] = DirectionFilter(
                        filter_type="DIR_LONG",
                        description="Long breaks only",
                        direction="long",
                    )
                elif ft == "DIR_SHORT":
                    needed_filters[ft] = DirectionFilter(
                        filter_type="DIR_SHORT",
                        description="Short breaks only",
                        direction="short",
                    )
                elif ft.startswith("ORB_G") and "_L" in ft:
                    # e.g. ORB_G4_L12 = G4 base + max break delay 12
                    parts = ft.split("_")
                    g_val = float(parts[1][1:])
                    l_val = float(parts[2][1:])
                    needed_filters[ft] = CompositeFilter(
                        filter_type=ft,
                        description=f"ORB >= {g_val} + delay <= {l_val}",
                        base=OrbSizeFilter(
                            filter_type=f"ORB_G{int(g_val)}",
                            description=f"ORB size >= {g_val}",
                            min_size=g_val,
                        ),
                        overlay=BreakSpeedFilter(
                            filter_type=f"BRK_FAST{int(l_val)}",
                            description=f"Break delay <= {l_val} min",
                            max_delay_minutes=l_val,
                        ),
                    )
                else:
                    print(f"  WARNING: Unknown filter type '{ft}', using NO_FILTER fallback")
                    needed_filters[ft] = ALL_FILTERS["NO_FILTER"]

        # Load daily features (canonical: orb_minutes=5)
        features = _load_daily_features(con, instrument, 5, None, None)

        # Compute relative volumes if needed
        has_vol = any(isinstance(f, VolumeFilter) for f in needed_filters.values())
        if has_vol:
            _compute_relative_volumes(con, features, instrument, orb_labels, needed_filters)

        # Build filter day sets
        filter_days = _build_filter_day_sets(features, orb_labels, needed_filters)

        # Load trades for each slot
        for slot in inst_slots:
            sid = slot["head_strategy_id"]
            params = slot_params.get(sid)
            if params is None:
                trades_by_slot[sid] = []
                continue

            # Get eligible days from filter
            eligible = filter_days.get(
                (params["filter_type"], params["orb_label"]), set()
            )

            # Query outcomes
            rows = con.execute("""
                SELECT trading_day, outcome, pnl_r
                FROM orb_outcomes
                WHERE symbol = ?
                  AND orb_label = ?
                  AND orb_minutes = ?
                  AND entry_model = ?
                  AND rr_target = ?
                  AND confirm_bars = ?
                  AND outcome IN ('win', 'loss')
                ORDER BY trading_day
            """, [
                params["instrument"], params["orb_label"], params["orb_minutes"],
                params["entry_model"], params["rr_target"], params["confirm_bars"],
            ]).fetchall()

            # Filter to eligible days
            trades_by_slot[sid] = [
                {
                    "trading_day": r[0],
                    "outcome": r[1],
                    "pnl_r": r[2],
                    "instrument": instrument,
                    "session": slot["session"],
                    "strategy_id": sid,
                }
                for r in rows
                if r[0] in eligible
            ]

    return trades_by_slot


def print_slot_inventory(slots, trades_by_slot):
    """Print formatted inventory table of all session slots and their trade counts."""
    print()
    print("=" * 110)
    print("SLOT INVENTORY")
    print("=" * 110)
    print(
        f"{'Instrument':<12} {'Session':<20} {'Strategy ID':<47} {'ExpR':>6} {'ShANN':>7} {'N':>6} {'Tier':<10}"
    )
    print("-" * 110)

    total_n = 0
    for slot in slots:
        sid = slot["head_strategy_id"]
        sid_display = sid[:45] if len(sid) > 45 else sid
        n = len(trades_by_slot.get(sid, []))
        total_n += n

        print(
            f"{slot['instrument']:<12} "
            f"{slot['session']:<20} "
            f"{sid_display:<47} "
            f"{slot['head_expectancy_r']:>+6.3f} "
            f"{slot['head_sharpe_ann']:>7.2f} "
            f"{n:>6d} "
            f"{slot['trade_tier']:<10}"
        )

    print("-" * 110)
    print(f"{'TOTAL':<12} {'':<20} {len(slots):>47d} slots {'':<6} {'':<7} {total_n:>6d}")
    print("=" * 110)
    print()


def build_daily_equity(trades_by_slot):
    """Build combined daily equity curve from all slot trades.

    Returns:
        daily_returns: sorted list of (date, total_r) for days with trades
        all_trades: flat list of all trades across slots
        daily_trade_count: dict day -> number of trades that day
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
    """Count weekdays (Mon-Fri) between two dates inclusive."""
    dates = pd.bdate_range(start=start_date, end=end_date)
    return len(dates)


def compute_honest_sharpe(daily_returns, start_date, end_date):
    """Compute Sharpe including zero-return days.

    Unlike per-trade Sharpe, this includes ALL weekdays as 0R returns.
    Prevents Sharpe inflation from idle capital.

    Returns (sharpe_daily, sharpe_ann, n_total_days).
    """
    all_bdays = pd.bdate_range(start=start_date, end=end_date)
    return_map = dict(daily_returns)

    full_series = [return_map.get(day.date(), 0.0) for day in all_bdays]

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
    """Compute drawdown stats on the full daily series (with zeros)."""
    all_bdays = pd.bdate_range(start=start_date, end=end_date)
    return_map = dict(daily_returns)

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    dd_start = all_bdays[0].date() if len(all_bdays) > 0 else None
    max_dd_start = None
    max_dd_end = None
    max_dd_trough = None
    worst_day = 0.0
    worst_day_date = None
    current_streak = 0
    max_streak = 0

    for day in all_bdays:
        day_date = day.date()
        r = return_map.get(day_date, 0.0)
        cum += r

        if r < worst_day:
            worst_day = r
            worst_day_date = day_date

        # Losing streak: only count actual negative days (zero-return days break streak)
        if r < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

        if cum > peak:
            peak = cum
            dd_start = day_date

        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            max_dd_start = dd_start
            max_dd_end = day_date

    # Compute recovery time: scan forward from max DD trough to find when equity returns to peak
    recovery_days = None
    if max_dd_end is not None and max_dd > 0:
        trough_idx = None
        for i, day in enumerate(all_bdays):
            if day.date() == max_dd_end:
                trough_idx = i
                break
        if trough_idx is not None:
            peak_at_trough = 0.0
            cum_scan = 0.0
            for day in all_bdays[:trough_idx + 1]:
                cum_scan += return_map.get(day.date(), 0.0)
            trough_cum = cum_scan
            target = trough_cum + max_dd  # The peak level before drawdown
            for day in all_bdays[trough_idx + 1:]:
                cum_scan += return_map.get(day.date(), 0.0)
                if cum_scan >= target:
                    recovery_days = (day.date() - max_dd_end).days
                    break

    dd_duration = None
    if max_dd_start and max_dd_end:
        dd_duration = (max_dd_end - max_dd_start).days

    return {
        "max_dd_r": round(max_dd, 4),
        "max_dd_start": max_dd_start,
        "max_dd_end": max_dd_end,
        "max_dd_duration_days": dd_duration,
        "recovery_days": recovery_days,
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

    active_days = len(daily_returns)
    total_bdays = count_trading_days(start_date, end_date)
    avg_trades_per_day = n_trades / active_days if active_days > 0 else 0
    active_pct = active_days / total_bdays if total_bdays > 0 else 0

    sharpe_d, sharpe_ann, n_full = compute_honest_sharpe(
        daily_returns, start_date, end_date
    )
    dd = compute_drawdown(daily_returns, start_date, end_date)

    print("=" * 80)
    print("  PORTFOLIO METRICS (honest Sharpe — includes zero-return days)")
    print("=" * 80)
    print(f"  Date range:        {start_date} to {end_date}")
    print(f"  Business days:     {total_bdays:,} (Sharpe denominator)")
    print(f"  Active days:       {active_days:,} ({active_pct:.1%} of business days)")
    print(f"  Total trades:      {n_trades:,}")
    print(f"  Win rate:          {wr:.1%}")
    print(f"  Total R:           {total_r:+.1f}")
    print(f"  Avg trades/day:    {avg_trades_per_day:.1f} (on active days)")
    print()
    if sharpe_ann is not None:
        print(f"  Sharpe (daily):    {sharpe_d:.4f}")
        print(f"  Sharpe (ann):      {sharpe_ann:.2f}")
    else:
        print("  Sharpe:            N/A")
    print()
    print(f"  Max drawdown:      {dd['max_dd_r']:.2f}R")
    if dd["max_dd_start"] and dd["max_dd_end"]:
        print(f"    Period:          {dd['max_dd_start']} to {dd['max_dd_end']} "
              f"({dd['max_dd_duration_days']} cal days)")
        if dd["recovery_days"] is not None:
            print(f"    Recovery:        {dd['recovery_days']} cal days from trough")
        else:
            print(f"    Recovery:        not yet recovered")
    print(f"  Worst single day:  {dd['worst_single_day']:+.2f}R"
          + (f" ({dd['worst_single_day_date']})" if dd["worst_single_day_date"] else ""))
    print(f"  Longest losing streak: {dd['longest_losing_streak']} consecutive negative days")


def print_yearly_breakdown(all_trades):
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

    print()
    print("=" * 80)
    print("  PER-YEAR BREAKDOWN")
    print("=" * 80)
    print(f"  {'Year':>6} {'Trades':>7} {'WR':>7} {'TotalR':>9} {'Days':>6} {'R/Day':>7}")
    print(f"  {'----':>6} {'------':>7} {'---':>7} {'------':>9} {'----':>6} {'-----':>7}")

    for year in sorted(yearly.keys()):
        y = yearly[year]
        wr = y["wins"] / y["trades"] if y["trades"] > 0 else 0
        n_days = len(y["days"])
        r_per_day = y["total_r"] / n_days if n_days > 0 else 0
        print(f"  {year:>6} {y['trades']:>7} {wr:>6.1%} "
              f"{y['total_r']:>+8.1f}R {n_days:>6} {r_per_day:>+6.3f}")


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

    print()
    print("=" * 80)
    print("  PER-INSTRUMENT CONTRIBUTION")
    print("=" * 80)
    print(f"  {'Inst':>6} {'Trades':>7} {'WR':>7} {'TotalR':>9} {'% of R':>8}")
    print(f"  {'----':>6} {'------':>7} {'---':>7} {'------':>9} {'------':>8}")

    for inst in sorted(by_inst.keys()):
        v = by_inst[inst]
        wr = v["wins"] / v["trades"] if v["trades"] > 0 else 0
        pct = v["total_r"] / total_r if total_r != 0 else 0
        print(f"  {inst:>6} {v['trades']:>7} {wr:>6.1%} "
              f"{v['total_r']:>+8.1f}R {pct:>7.1%}")


def print_correlation_matrix(trades_by_slot, slots):
    """Section 5: Pairwise correlation between slots."""
    MIN_OVERLAP = 30

    slot_daily = {}
    for slot in slots:
        sid = slot["head_strategy_id"]
        label = f"{slot['instrument']}_{slot['session']}"
        daily = {}
        for t in trades_by_slot.get(sid, []):
            day = t["trading_day"]
            daily[day] = daily.get(day, 0.0) + t["pnl_r"]
        if daily:
            slot_daily[label] = daily

    labels = sorted(slot_daily.keys())

    print()
    print("=" * 80)
    print(f"  SLOT CORRELATION (pairs with >= {MIN_OVERLAP} overlapping days)")
    print("=" * 80)

    corr_pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_daily = slot_daily[labels[i]]
            b_daily = slot_daily[labels[j]]
            overlap = sorted(set(a_daily) & set(b_daily))
            if len(overlap) < MIN_OVERLAP:
                continue
            a_vals = [a_daily[d] for d in overlap]
            b_vals = [b_daily[d] for d in overlap]
            if np.std(a_vals) == 0 or np.std(b_vals) == 0:
                continue
            r = float(np.corrcoef(a_vals, b_vals)[0, 1])
            corr_pairs.append((labels[i], labels[j], r, len(overlap)))

    if not corr_pairs:
        print("  No pairs with sufficient overlap.")
        return

    corr_pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, r, n in corr_pairs:
        flag = " <-- HIGH" if abs(r) > 0.3 else ""
        print(f"  {a:>22} x {b:<22} r={r:+.3f} (N={n:>3}){flag}")

    high = sum(1 for _, _, r, _ in corr_pairs if abs(r) > 0.3)
    print(f"\n  {len(corr_pairs)} pairs, {high} with |r| > 0.3")


def print_concurrent_exposure(daily_trade_count, start_date, end_date):
    """Section 6: Concurrent exposure analysis."""
    all_bdays = pd.bdate_range(start=start_date, end=end_date)

    exposure_dist = defaultdict(int)
    max_exposure = 0
    max_exposure_date = None

    for day in all_bdays:
        day_date = day.date()
        n = daily_trade_count.get(day_date, 0)
        exposure_dist[n] += 1
        if n > max_exposure:
            max_exposure = n
            max_exposure_date = day_date

    total_bdays = len(all_bdays)

    print()
    print("=" * 80)
    print("  CONCURRENT EXPOSURE")
    print("=" * 80)

    for n_slots in sorted(exposure_dist.keys()):
        count = exposure_dist[n_slots]
        pct = count / total_bdays
        bar = "#" * int(pct * 40)
        print(f"  {n_slots:>2} slots: {count:>5} days ({pct:>5.1%}) {bar}")

    active_days = sum(c for n, c in exposure_dist.items() if n > 0)
    active_trades = sum(n * c for n, c in exposure_dist.items())
    avg_on_active = active_trades / active_days if active_days > 0 else 0

    print(f"\n  Max exposure:      {max_exposure} slots on {max_exposure_date}")
    print(f"  Max single-day R:  {max_exposure}R at risk (if all lose)")
    print(f"  Avg slots/active day: {avg_on_active:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Assembly Research Report"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(GOLD_DB_PATH),
        help="Path to DuckDB database (default: project gold.db)",
    )
    parser.add_argument(
        "--exclude-regime",
        action="store_true",
        help="Exclude REGIME-tier slots (keep only CORE)",
    )
    args = parser.parse_args()

    db_path = args.db_path

    print(f"Database: {db_path}")
    print(f"Exclude REGIME: {args.exclude_regime}")

    # Load session slots
    slots = session_slots(db_path)
    print(f"Loaded {len(slots)} session slots")

    if args.exclude_regime:
        before = len(slots)
        slots = [s for s in slots if s["trade_tier"] != "REGIME"]
        print(f"Filtered to {len(slots)} slots (excluded {before - len(slots)} REGIME)")

    if not slots:
        print("No slots found. Exiting.")
        return

    # Open DB read-only and load trades
    con = duckdb.connect(db_path, read_only=True)
    try:
        trades_by_slot = load_slot_trades(con, slots)
        print_slot_inventory(slots, trades_by_slot)

        # Build combined equity curve
        daily_returns, all_trades, daily_trade_count = build_daily_equity(trades_by_slot)
        if not all_trades:
            print("\nNo trades found across any slots.")
            return

        all_days = [t["trading_day"] for t in all_trades]
        start_date = min(all_days)
        end_date = max(all_days)

        # ── Full History View ─────────────────────────────────────
        print()
        print("#" * 80)
        print("  VIEW 1: FULL HISTORY (each instrument from its earliest data)")
        print("#" * 80)

        print_portfolio_metrics(daily_returns, all_trades, daily_trade_count, start_date, end_date)
        print_yearly_breakdown(all_trades)
        print_instrument_contribution(all_trades)
        print_correlation_matrix(trades_by_slot, slots)
        print_concurrent_exposure(daily_trade_count, start_date, end_date)

        # ── Common Period View ────────────────────────────────────
        # Find date range where ALL instruments have at least one trade
        instruments_in_portfolio = set(t["instrument"] for t in all_trades)
        if len(instruments_in_portfolio) > 1:
            inst_first_day = {}
            inst_last_day = {}
            for t in all_trades:
                inst = t["instrument"]
                day = t["trading_day"]
                if inst not in inst_first_day or day < inst_first_day[inst]:
                    inst_first_day[inst] = day
                if inst not in inst_last_day or day > inst_last_day[inst]:
                    inst_last_day[inst] = day

            common_start = max(inst_first_day.values())
            common_end = min(inst_last_day.values())

            if common_start < common_end:
                print()
                print("#" * 80)
                print("  VIEW 2: COMMON PERIOD (all instruments present)")
                print(f"  (Latest instrument start: {common_start}, "
                      f"earliest instrument end: {common_end})")
                print("#" * 80)

                # Filter trades to common period
                common_trades_by_slot = {}
                for sid, trades in trades_by_slot.items():
                    filtered = [t for t in trades
                                if common_start <= t["trading_day"] <= common_end]
                    if filtered:
                        common_trades_by_slot[sid] = filtered

                c_daily, c_all, c_counts = build_daily_equity(common_trades_by_slot)
                if c_all:
                    print_portfolio_metrics(c_daily, c_all, c_counts,
                                            common_start, common_end)
                    print_yearly_breakdown(c_all)
                    print_instrument_contribution(c_all)
            else:
                print("\n  (Common period: no overlapping date range for all instruments)")
        else:
            print("\n  (Common period: only one instrument in portfolio, skipping)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
