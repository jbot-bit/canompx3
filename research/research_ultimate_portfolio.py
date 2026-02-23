"""Ultimate Portfolio — stacks all confirmed signals + correlation-aware position capping.

Combines:
1. All overlay signals (all_narrow AVOID, ATR contraction AVOID, Friday high-vol AVOID)
2. T80 time-stop (ts_pnl_r where available)
3. Correlation-aware daily position cap to crush max drawdown

Sweeps cap levels to show the Sharpe vs MaxDD tradeoff.

Usage:
    python research/research_ultimate_portfolio.py
    python research/research_ultimate_portfolio.py --db-path C:/db/gold.db
    python research/research_ultimate_portfolio.py --cap 8
"""

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
from trading_app.config import ALL_FILTERS, VolumeFilter, EARLY_EXIT_MINUTES
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_daily_features,
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reports"))
from report_edge_portfolio import session_slots

sys.stdout.reconfigure(line_buffering=True)

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# TRADE LOADING (from signal_stack.py)
# =============================================================================

def _get_strategy_params(con, strategy_id):
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


_ts_check_cache = {}

def _check_ts_columns(con):
    if "checked" in _ts_check_cache:
        return _ts_check_cache["checked"]
    try:
        con.execute("SELECT ts_pnl_r FROM orb_outcomes LIMIT 1")
        _ts_check_cache["checked"] = True
    except Exception:
        _ts_check_cache["checked"] = False
    return _ts_check_cache["checked"]


def load_trades_with_features(con, slots):
    """Load trades for each slot with daily_features + ts columns."""
    all_trades = []

    by_instrument = defaultdict(list)
    for slot in slots:
        by_instrument[slot["instrument"]].append(slot)

    for instrument, inst_slots in by_instrument.items():
        slot_params = {}
        filter_types = set()
        orb_labels = set()
        for slot in inst_slots:
            params = _get_strategy_params(con, slot["head_strategy_id"])
            if params is None:
                continue
            slot_params[slot["head_strategy_id"]] = params
            filter_types.add(params["filter_type"])
            orb_labels.add(params["orb_label"])

        if not slot_params:
            continue

        needed_filters = {k: v for k, v in ALL_FILTERS.items() if k in filter_types}
        features = _load_daily_features(con, instrument, 5, None, None)

        has_vol = any(isinstance(f, VolumeFilter) for f in needed_filters.values())
        if has_vol:
            _compute_relative_volumes(con, features, instrument, sorted(orb_labels), needed_filters)

        filter_days = _build_filter_day_sets(features, sorted(orb_labels), needed_filters)

        feat_lookup = {}
        for row in features:
            feat_lookup[row["trading_day"]] = row

        has_ts = _check_ts_columns(con)

        for slot in inst_slots:
            sid = slot["head_strategy_id"]
            params = slot_params.get(sid)
            if params is None:
                continue

            eligible = filter_days.get(
                (params["filter_type"], params["orb_label"]), set()
            )

            ts_cols = ", oo.ts_pnl_r, oo.ts_outcome" if has_ts else ""
            rows = con.execute(f"""
                SELECT oo.trading_day, oo.outcome, oo.pnl_r{ts_cols}
                FROM orb_outcomes oo
                WHERE oo.symbol = ?
                  AND oo.orb_label = ?
                  AND oo.orb_minutes = ?
                  AND oo.entry_model = ?
                  AND oo.rr_target = ?
                  AND oo.confirm_bars = ?
                  AND oo.outcome IN ('win', 'loss')
                ORDER BY oo.trading_day
            """, [
                params["instrument"], params["orb_label"], params["orb_minutes"],
                params["entry_model"], params["rr_target"], params["confirm_bars"],
            ]).fetchall()

            for r in rows:
                td = r[0]
                if td not in eligible:
                    continue

                feat = feat_lookup.get(td)
                sess_label = params["orb_label"]

                trade = {
                    "trading_day": td,
                    "outcome": r[1],
                    "pnl_r": r[2],
                    "ts_pnl_r": r[3] if has_ts and len(r) > 3 else None,
                    "ts_outcome": r[4] if has_ts and len(r) > 4 else None,
                    "instrument": instrument,
                    "session": sess_label,
                    "strategy_id": sid,
                    "slot_label": f"{instrument}_{sess_label}",
                    "atr_vel_ratio": feat["atr_vel_ratio"] if feat else None,
                    "compression_tier": (
                        feat.get(f"orb_{sess_label}_compression_tier")
                        if feat else None
                    ),
                    "atr_20": feat["atr_20"] if feat else None,
                    "day_of_week": td.weekday() if hasattr(td, "weekday") else None,
                    "is_friday": td.weekday() == 4 if hasattr(td, "weekday") else False,
                    # Slot priority for capping (from validated_setups metadata)
                    "slot_sharpe": slot["head_sharpe_ann"],
                    "slot_expr": slot["head_expectancy_r"],
                }
                all_trades.append(trade)

    return all_trades


# =============================================================================
# OVERLAY SIGNALS
# =============================================================================

def compute_all_narrow_days(con):
    rows = con.execute("""
        SELECT trading_day, symbol, orb_1000_size
        FROM daily_features
        WHERE orb_minutes = 5
          AND symbol IN ('MGC', 'MES', 'MNQ')
          AND orb_1000_size IS NOT NULL
        ORDER BY trading_day
    """).fetchall()
    if not rows:
        return set()

    df = pd.DataFrame(rows, columns=["trading_day", "symbol", "orb_1000_size"])
    avoid_days = set()
    medians = {}
    for sym in ["MGC", "MES", "MNQ"]:
        sym_df = df[df["symbol"] == sym].sort_values("trading_day").copy()
        sym_df["expanding_median"] = (
            sym_df["orb_1000_size"].expanding(min_periods=20).median().shift(1)
        )
        sym_df["below_median"] = sym_df["orb_1000_size"] < sym_df["expanding_median"]
        medians[sym] = dict(zip(sym_df["trading_day"], sym_df["below_median"]))

    all_days = set()
    for sym_days in medians.values():
        all_days.update(sym_days.keys())

    for day in all_days:
        checks = []
        for sym in ["MGC", "MES", "MNQ"]:
            val = medians[sym].get(day)
            if val is None:
                break
            checks.append(val)
        if len(checks) == 3 and all(checks):
            avoid_days.add(day)

    return avoid_days


def is_atr_contraction_avoid(trade):
    ratio = trade.get("atr_vel_ratio")
    tier = trade.get("compression_tier")
    if ratio is None or tier is None:
        return False
    return ratio < 0.95 and tier in ("Neutral", "Compressed")


def compute_friday_highvol_threshold(con):
    rows = con.execute("""
        SELECT atr_20 FROM daily_features
        WHERE orb_minutes = 5 AND symbol = 'MGC' AND atr_20 IS NOT NULL
        ORDER BY trading_day
    """).fetchall()
    if not rows:
        return None
    return float(np.percentile([r[0] for r in rows], 75))


def is_friday_highvol_avoid(trade, atr_threshold):
    if trade["instrument"] != "MGC" or trade["session"] != "1000":
        return False
    if not trade.get("is_friday", False):
        return False
    atr = trade.get("atr_20")
    if atr is None or atr_threshold is None:
        return False
    return atr > atr_threshold


def apply_signals(trades, all_narrow_days, atr_threshold):
    """Apply all overlay signals. Returns filtered trades with effective_pnl_r set."""
    result = []
    for t in trades:
        # all_narrow AVOID
        if t["trading_day"] in all_narrow_days:
            continue
        # ATR contraction AVOID
        if is_atr_contraction_avoid(t):
            continue
        # Friday high-vol AVOID
        if is_friday_highvol_avoid(t, atr_threshold):
            continue

        t_copy = dict(t)
        # T80 time-stop: use ts_pnl_r where available
        ts_pnl = t.get("ts_pnl_r")
        ts_out = t.get("ts_outcome")
        if ts_pnl is not None and ts_out is not None:
            t_copy["effective_pnl_r"] = ts_pnl
            if ts_out == "time_stop":
                t_copy["outcome"] = "loss"
        else:
            t_copy["effective_pnl_r"] = t["pnl_r"]

        result.append(t_copy)
    return result


# =============================================================================
# CORRELATION-AWARE POSITION CAPPING
# =============================================================================

def compute_slot_correlations(trades):
    """Compute pairwise daily-return correlations between slot_labels.

    Returns dict: (slotA, slotB) -> correlation coefficient.
    """
    MIN_OVERLAP = 30

    # Build daily returns per slot
    slot_daily = defaultdict(lambda: defaultdict(float))
    for t in trades:
        slot_daily[t["slot_label"]][t["trading_day"]] += t.get("effective_pnl_r", t["pnl_r"])

    labels = sorted(slot_daily.keys())
    corr = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = slot_daily[labels[i]]
            b = slot_daily[labels[j]]
            overlap = sorted(set(a) & set(b))
            if len(overlap) < MIN_OVERLAP:
                continue
            a_vals = [a[d] for d in overlap]
            b_vals = [b[d] for d in overlap]
            if np.std(a_vals) == 0 or np.std(b_vals) == 0:
                continue
            r = float(np.corrcoef(a_vals, b_vals)[0, 1])
            corr[(labels[i], labels[j])] = r
            corr[(labels[j], labels[i])] = r
    return corr


def apply_position_cap(trades, max_positions, slot_correlations, corr_penalty=0.3):
    """Cap concurrent positions per day with correlation awareness.

    Algorithm per day:
    1. List all slots that want to trade
    2. Sort by slot_sharpe descending (best slots first)
    3. Add slots greedily, tracking "effective position count"
       - Each new slot costs 1.0 base
       - Plus corr_penalty for each already-selected slot with |r| > 0.3
    4. Stop when effective count would exceed max_positions

    Returns filtered trade list.
    """
    if max_positions is None:
        return trades  # No cap

    # Group trades by day
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["trading_day"]].append(t)

    result = []
    cap_stats = {"capped_days": 0, "trades_removed": 0}

    for day in sorted(by_day.keys()):
        day_trades = by_day[day]

        # Group by slot_label (one trade per slot per day typically)
        slots_today = defaultdict(list)
        for t in day_trades:
            slots_today[t["slot_label"]].append(t)

        slot_labels = list(slots_today.keys())

        if len(slot_labels) <= max_positions:
            # Under cap — take everything
            result.extend(day_trades)
            continue

        # Sort slots by priority (Sharpe * ExpR for robust ranking)
        slot_priority = []
        for label in slot_labels:
            trades_for_slot = slots_today[label]
            sharpe = trades_for_slot[0].get("slot_sharpe", 0) or 0
            expr = trades_for_slot[0].get("slot_expr", 0) or 0
            priority = sharpe  # Primary sort by Sharpe
            slot_priority.append((label, priority, trades_for_slot))

        slot_priority.sort(key=lambda x: -x[1])

        # Greedy selection with correlation penalty
        selected = []
        effective_count = 0.0

        for label, priority, slot_trades in slot_priority:
            # Compute correlation cost with already-selected slots
            corr_cost = 0.0
            for sel_label in selected:
                r = slot_correlations.get((label, sel_label), 0.0)
                if abs(r) > 0.3:
                    corr_cost += corr_penalty

            new_effective = effective_count + 1.0 + corr_cost

            if new_effective <= max_positions + 0.5:  # Small tolerance
                selected.append(label)
                effective_count = new_effective
                result.extend(slot_trades)
            else:
                cap_stats["trades_removed"] += len(slot_trades)

        if len(selected) < len(slot_labels):
            cap_stats["capped_days"] += 1

    return result, cap_stats


# =============================================================================
# ADAPTIVE POSITION SIZING
# =============================================================================

def apply_adaptive_sizing(trades, dd_threshold=15.0, reduced_scale=0.5):
    """Scale down position size when in drawdown.

    When cumulative drawdown from peak exceeds dd_threshold,
    multiply effective_pnl_r by reduced_scale until equity recovers.

    This is the REAL lever for max DD reduction. Works because:
    - First N*R of losses hit at full size (unavoidable)
    - Remaining losses during extended DD hit at half size
    - Recovery trades also at half size, so recovery is slower BUT
    - Max dollar-at-risk DD drops significantly

    Returns new trade list with adjusted effective_pnl_r.
    """
    # Must process in day order
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["trading_day"]].append(t)

    result = []
    cum = 0.0
    peak = 0.0
    scale_days = 0
    total_days = 0

    for day in sorted(by_day.keys()):
        total_days += 1
        day_trades = by_day[day]

        # Check drawdown BEFORE today's trades
        dd = peak - cum
        current_scale = reduced_scale if dd >= dd_threshold else 1.0
        if current_scale < 1.0:
            scale_days += 1

        # Apply scale to today's trades
        day_r = 0.0
        for t in day_trades:
            t_copy = dict(t)
            t_copy["effective_pnl_r"] = t["effective_pnl_r"] * current_scale
            t_copy["position_scale"] = current_scale
            result.append(t_copy)
            day_r += t_copy["effective_pnl_r"]

        cum += day_r
        if cum > peak:
            peak = cum

    return result, {"scale_days": scale_days, "total_days": total_days}


# =============================================================================
# METRICS
# =============================================================================

def compute_full_metrics(trades, start_date, end_date):
    """Compute comprehensive portfolio metrics."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "total_r": 0, "exp_r": 0, "wr": 0,
                "sharpe_ann": None, "max_dd": 0, "max_dd_start": None,
                "max_dd_end": None, "dd_duration": None, "recovery_days": None,
                "worst_day": 0, "worst_day_date": None, "max_concurrent": 0,
                "avg_concurrent": 0}

    n_wins = sum(1 for t in trades if t["outcome"] == "win")
    total_r = sum(t["effective_pnl_r"] for t in trades)

    # Daily returns
    daily_r = defaultdict(float)
    daily_count = defaultdict(int)
    for t in trades:
        daily_r[t["trading_day"]] += t["effective_pnl_r"]
        daily_count[t["trading_day"]] += 1

    # Honest Sharpe (with zero-return days)
    all_bdays = pd.bdate_range(start=start_date, end=end_date)
    return_map = dict(daily_r)
    full_series = [return_map.get(day.date(), 0.0) for day in all_bdays]

    n_days = len(full_series)
    sharpe_ann = None
    if n_days > 1:
        mean_d = sum(full_series) / n_days
        var = sum((v - mean_d) ** 2 for v in full_series) / (n_days - 1)
        std_d = var ** 0.5
        if std_d > 0:
            sharpe_ann = (mean_d / std_d) * sqrt(TRADING_DAYS_PER_YEAR)

    # Drawdown with duration tracking
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    dd_start = all_bdays[0].date() if len(all_bdays) > 0 else None
    max_dd_start = None
    max_dd_end = None
    worst_day = 0.0
    worst_day_date = None

    for day in all_bdays:
        day_date = day.date()
        r = return_map.get(day_date, 0.0)
        cum += r

        if r < worst_day:
            worst_day = r
            worst_day_date = day_date

        if cum > peak:
            peak = cum
            dd_start = day_date

        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            max_dd_start = dd_start
            max_dd_end = day_date

    # Recovery
    recovery_days = None
    if max_dd_end is not None and max_dd > 0:
        trough_idx = None
        for i, day in enumerate(all_bdays):
            if day.date() == max_dd_end:
                trough_idx = i
                break
        if trough_idx is not None:
            cum_scan = 0.0
            for day in all_bdays[:trough_idx + 1]:
                cum_scan += return_map.get(day.date(), 0.0)
            target = cum_scan + max_dd
            for day in all_bdays[trough_idx + 1:]:
                cum_scan += return_map.get(day.date(), 0.0)
                if cum_scan >= target:
                    recovery_days = (day.date() - max_dd_end).days
                    break

    dd_duration = None
    if max_dd_start and max_dd_end:
        dd_duration = (max_dd_end - max_dd_start).days

    # Concurrent exposure
    max_concurrent = max(daily_count.values()) if daily_count else 0
    active_days = [c for c in daily_count.values() if c > 0]
    avg_concurrent = sum(active_days) / len(active_days) if active_days else 0

    return {
        "n": n,
        "total_r": round(total_r, 1),
        "exp_r": round(total_r / n, 4) if n > 0 else 0,
        "wr": round(n_wins / n, 3) if n > 0 else 0,
        "sharpe_ann": round(sharpe_ann, 2) if sharpe_ann else None,
        "max_dd": round(max_dd, 1),
        "max_dd_start": max_dd_start,
        "max_dd_end": max_dd_end,
        "dd_duration": dd_duration,
        "recovery_days": recovery_days,
        "worst_day": round(worst_day, 2),
        "worst_day_date": worst_day_date,
        "max_concurrent": max_concurrent,
        "avg_concurrent": round(avg_concurrent, 1),
    }


def yearly_breakdown(trades):
    yearly = defaultdict(lambda: {"n": 0, "wins": 0, "total_r": 0.0})
    for t in trades:
        td = t["trading_day"]
        year = td.year if hasattr(td, "year") else int(str(td)[:4])
        yearly[year]["n"] += 1
        if t["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += t["effective_pnl_r"]

    result = {}
    for year, y in yearly.items():
        result[year] = {
            "n": y["n"],
            "total_r": round(y["total_r"], 1),
            "exp_r": round(y["total_r"] / y["n"], 4) if y["n"] > 0 else 0,
            "wr": round(y["wins"] / y["n"], 3) if y["n"] > 0 else 0,
        }
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultimate Portfolio — signals + position capping")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--cap", type=int, default=None,
                        help="Fixed position cap (default: sweep 6-14)")
    parser.add_argument("--corr-penalty", type=float, default=0.3,
                        help="Effective position penalty per correlated slot (default: 0.3)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        slots = session_slots(db_path)
        if not slots:
            print("No session slots found.")
            return

        print(f"\n{'#' * 90}")
        print(f"#  ULTIMATE PORTFOLIO — All signals stacked + correlation-aware position cap")
        print(f"{'#' * 90}\n")
        print(f"Database: {db_path}")
        print(f"Slots: {len(slots)}")

        # Load trades
        print("\nLoading trades with features...")
        raw_trades = load_trades_with_features(con, slots)
        print(f"  Raw trades: {len(raw_trades)}")

        # Pre-compute signals
        print("Pre-computing signal data...")
        all_narrow_days = compute_all_narrow_days(con)
        atr_threshold = compute_friday_highvol_threshold(con)
        print(f"  all_narrow AVOID days: {len(all_narrow_days)}")
        print(f"  ATR P75 threshold: {atr_threshold:.1f}" if atr_threshold else "  No ATR data")

        # Apply all overlay signals
        print("\nApplying overlay signals (all_narrow + ATR contraction + Friday HV + T80)...")
        signaled_trades = apply_signals(raw_trades, all_narrow_days, atr_threshold)
        print(f"  After signals: {len(signaled_trades)} trades ({len(raw_trades) - len(signaled_trades)} removed)")

        ts_used = sum(1 for t in signaled_trades
                      if t.get("ts_pnl_r") is not None and t.get("ts_outcome") is not None)
        print(f"  T80 time-stop applied: {ts_used}/{len(signaled_trades)} trades")

        if not signaled_trades:
            print("No trades after signal filtering.")
            return

        all_days = [t["trading_day"] for t in signaled_trades]
        start_date = min(all_days)
        end_date = max(all_days)

        # Compute slot correlations for capping
        print("\nComputing slot correlations...")
        slot_corr = compute_slot_correlations(signaled_trades)
        high_corr = [(a, b, r) for (a, b), r in slot_corr.items()
                     if a < b and abs(r) > 0.3]
        high_corr.sort(key=lambda x: -abs(x[2]))
        print(f"  {len(high_corr)} pairs with |r| > 0.3:")
        for a, b, r in high_corr[:10]:
            print(f"    {a:>20} x {b:<20} r={r:+.3f}")

        # =====================================================================
        # SWEEP CAP LEVELS
        # =====================================================================
        if args.cap:
            cap_levels = [args.cap]
        else:
            cap_levels = [None, 14, 12, 10, 8, 6]

        print(f"\n{'=' * 100}")
        print(f"POSITION CAP SWEEP (corr_penalty={args.corr_penalty})")
        print(f"{'=' * 100}")
        print(f"{'Cap':>5} {'N':>6} {'TotalR':>9} {'ExpR':>8} {'WR':>6} "
              f"{'ShANN':>7} {'MaxDD':>7} {'DD days':>8} {'Recov':>7} "
              f"{'MaxConc':>8} {'AvgConc':>8} {'Worst':>7}")
        print("-" * 100)

        best_result = None
        best_ratio = -999

        for cap in cap_levels:
            cap_label = f"{cap}" if cap else "none"

            if cap is None:
                capped_trades = signaled_trades
                cap_info = {"capped_days": 0, "trades_removed": 0}
            else:
                capped_trades, cap_info = apply_position_cap(
                    signaled_trades, cap, slot_corr, args.corr_penalty
                )

            m = compute_full_metrics(capped_trades, start_date, end_date)

            sh = f"{m['sharpe_ann']:.2f}" if m['sharpe_ann'] else "N/A"
            dd_dur = f"{m['dd_duration']}d" if m['dd_duration'] else "N/A"
            recov = f"{m['recovery_days']}d" if m['recovery_days'] else "never"

            print(f"{cap_label:>5} {m['n']:>6} {m['total_r']:>+8.1f}R "
                  f"{m['exp_r']:>+7.4f} {m['wr']:>5.1%} "
                  f"{sh:>7} {m['max_dd']:>6.1f}R {dd_dur:>8} {recov:>7} "
                  f"{m['max_concurrent']:>8} {m['avg_concurrent']:>8.1f} "
                  f"{m['worst_day']:>+6.2f}R")

            # Track best Sharpe/DD ratio
            if m['sharpe_ann'] and m['max_dd'] > 0:
                ratio = m['sharpe_ann'] / m['max_dd']
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = (cap, m, capped_trades)

        # =====================================================================
        # DETAILED VIEW OF BEST CAP LEVEL
        # =====================================================================
        if best_result:
            best_cap, best_m, best_trades = best_result
            cap_label = f"{best_cap}" if best_cap else "uncapped"

            print(f"\n{'#' * 90}")
            print(f"#  BEST PORTFOLIO: cap={cap_label} (Sharpe/DD ratio = {best_ratio:.4f})")
            print(f"{'#' * 90}")

            print(f"\n  Trades:          {best_m['n']:,}")
            print(f"  Total R:         {best_m['total_r']:+.1f}")
            print(f"  ExpR:            {best_m['exp_r']:+.4f}")
            print(f"  Win rate:        {best_m['wr']:.1%}")
            if best_m['sharpe_ann']:
                print(f"  Sharpe (ann):    {best_m['sharpe_ann']:.2f}")
            print(f"  Max drawdown:    {best_m['max_dd']:.1f}R")
            if best_m['max_dd_start'] and best_m['max_dd_end']:
                print(f"    Period:        {best_m['max_dd_start']} to {best_m['max_dd_end']} "
                      f"({best_m['dd_duration']} cal days)")
                if best_m['recovery_days']:
                    print(f"    Recovery:      {best_m['recovery_days']} cal days")
                else:
                    print(f"    Recovery:      NOT YET RECOVERED")
            print(f"  Worst day:       {best_m['worst_day']:+.2f}R ({best_m['worst_day_date']})")
            print(f"  Max concurrent:  {best_m['max_concurrent']}")
            print(f"  Avg concurrent:  {best_m['avg_concurrent']:.1f}")

            # Per-year
            yb = yearly_breakdown(best_trades)
            print(f"\n  {'Year':>6} {'N':>7} {'TotalR':>9} {'ExpR':>10} {'WR':>7}")
            print(f"  {'----':>6} {'-----':>7} {'------':>9} {'------':>10} {'---':>7}")
            all_positive = True
            for year in sorted(yb.keys()):
                y = yb[year]
                print(f"  {year:>6} {y['n']:>7} {y['total_r']:>+8.1f}R "
                      f"{y['exp_r']:>+9.4f} {y['wr']:>6.1%}")
                if y['total_r'] < 0:
                    all_positive = False
            print(f"\n  Every year positive: {'YES' if all_positive else 'NO'}")

            # Per-instrument
            by_inst = defaultdict(lambda: {"n": 0, "total_r": 0.0})
            for t in best_trades:
                by_inst[t["instrument"]]["n"] += 1
                by_inst[t["instrument"]]["total_r"] += t["effective_pnl_r"]

            print(f"\n  {'Instrument':>12} {'N':>7} {'TotalR':>9} {'% of R':>8}")
            for inst in sorted(by_inst.keys()):
                v = by_inst[inst]
                pct = v["total_r"] / best_m["total_r"] if best_m["total_r"] != 0 else 0
                print(f"  {inst:>12} {v['n']:>7} {v['total_r']:>+8.1f}R {pct:>7.1%}")

            # Exposure distribution
            daily_count = defaultdict(int)
            for t in best_trades:
                daily_count[t["trading_day"]] += 1

            all_bdays = pd.bdate_range(start=start_date, end=end_date)
            exposure_dist = defaultdict(int)
            for day in all_bdays:
                n = daily_count.get(day.date(), 0)
                exposure_dist[n] += 1

            print(f"\n  Concurrent exposure distribution:")
            for n_slots in sorted(exposure_dist.keys()):
                count = exposure_dist[n_slots]
                pct = count / len(all_bdays)
                bar = "#" * int(pct * 40)
                print(f"    {n_slots:>2} slots: {count:>5} days ({pct:>5.1%}) {bar}")

        # =====================================================================
        # ADAPTIVE SIZING SWEEP (on top of best cap)
        # =====================================================================
        if best_result:
            best_cap, _, best_trades = best_result

            print(f"\n{'=' * 110}")
            print(f"ADAPTIVE SIZING SWEEP (cap={best_cap}, scale down to X when DD exceeds threshold)")
            print(f"{'=' * 110}")
            print(f"{'Threshold':>10} {'Scale':>6} {'N':>6} {'TotalR':>9} {'ExpR':>8} "
                  f"{'ShANN':>7} {'MaxDD':>7} {'DD days':>8} {'Recov':>7} "
                  f"{'Worst':>7} {'ScaleDays':>10}")
            print("-" * 110)

            # Baseline (no adaptive sizing)
            base_m = compute_full_metrics(best_trades, start_date, end_date)
            sh = f"{base_m['sharpe_ann']:.2f}" if base_m['sharpe_ann'] else "N/A"
            dd_dur = f"{base_m['dd_duration']}d" if base_m['dd_duration'] else "N/A"
            recov = f"{base_m['recovery_days']}d" if base_m['recovery_days'] else "never"
            print(f"{'none':>10} {'1.0':>6} {base_m['n']:>6} {base_m['total_r']:>+8.1f}R "
                  f"{base_m['exp_r']:>+7.4f} {sh:>7} {base_m['max_dd']:>6.1f}R "
                  f"{dd_dur:>8} {recov:>7} {base_m['worst_day']:>+6.2f}R {'0':>10}")

            # Sweep thresholds and scales
            configs = [
                (20.0, 0.5),   # Half size after -20R DD
                (15.0, 0.5),   # Half size after -15R DD
                (10.0, 0.5),   # Half size after -10R DD
                (20.0, 0.25),  # Quarter size after -20R DD
                (15.0, 0.25),  # Quarter size after -15R DD
                (10.0, 0.25),  # Quarter size after -10R DD
            ]

            ultimate_best = None
            ultimate_best_ratio = base_m['sharpe_ann'] / base_m['max_dd'] if base_m['sharpe_ann'] and base_m['max_dd'] > 0 else -999

            for threshold, scale in configs:
                adapted, stats = apply_adaptive_sizing(best_trades, threshold, scale)
                m = compute_full_metrics(adapted, start_date, end_date)

                sh = f"{m['sharpe_ann']:.2f}" if m['sharpe_ann'] else "N/A"
                dd_dur = f"{m['dd_duration']}d" if m['dd_duration'] else "N/A"
                recov = f"{m['recovery_days']}d" if m['recovery_days'] else "never"

                print(f"{threshold:>9.0f}R {scale:>5.2f} {m['n']:>6} {m['total_r']:>+8.1f}R "
                      f"{m['exp_r']:>+7.4f} {sh:>7} {m['max_dd']:>6.1f}R "
                      f"{dd_dur:>8} {recov:>7} {m['worst_day']:>+6.2f}R "
                      f"{stats['scale_days']:>10}")

                if m['sharpe_ann'] and m['max_dd'] > 0:
                    ratio = m['sharpe_ann'] / m['max_dd']
                    if ratio > ultimate_best_ratio:
                        ultimate_best_ratio = ratio
                        ultimate_best = (threshold, scale, m, adapted, stats)

            # =====================================================================
            # THE ULTIMATE PORTFOLIO
            # =====================================================================
            print(f"\n{'#' * 90}")
            if ultimate_best:
                threshold, scale, um, u_trades, u_stats = ultimate_best
                print(f"#  THE ULTIMATE PORTFOLIO")
                print(f"#  Signals: all_narrow + ATR contraction + Friday HV + T80")
                print(f"#  Position cap: {best_cap} (corr-aware)")
                print(f"#  Adaptive sizing: {scale}x when DD > {threshold:.0f}R")
                print(f"{'#' * 90}")
            else:
                um = base_m
                u_trades = best_trades
                print(f"#  THE ULTIMATE PORTFOLIO (no adaptive sizing improved ratio)")
                print(f"#  Signals: all_narrow + ATR contraction + Friday HV + T80")
                print(f"#  Position cap: {best_cap} (corr-aware)")
                print(f"{'#' * 90}")

            print(f"\n  Trades:          {um['n']:,}")
            print(f"  Total R:         {um['total_r']:+.1f}")
            print(f"  ExpR:            {um['exp_r']:+.4f}")
            print(f"  Win rate:        {um['wr']:.1%}")
            if um['sharpe_ann']:
                print(f"  Sharpe (ann):    {um['sharpe_ann']:.2f}")
            print(f"  Max drawdown:    {um['max_dd']:.1f}R")
            if um['max_dd_start'] and um['max_dd_end']:
                print(f"    Period:        {um['max_dd_start']} to {um['max_dd_end']} "
                      f"({um['dd_duration']} cal days)")
                if um['recovery_days']:
                    print(f"    Recovery:      {um['recovery_days']} cal days")
                else:
                    print(f"    Recovery:      NOT YET RECOVERED")
            print(f"  Worst day:       {um['worst_day']:+.02f}R ({um['worst_day_date']})")
            print(f"  Max concurrent:  {um['max_concurrent']}")
            print(f"  Avg concurrent:  {um['avg_concurrent']:.1f}")
            if ultimate_best:
                print(f"  Scaled-down days: {u_stats['scale_days']}/{u_stats['total_days']} "
                      f"({u_stats['scale_days']/u_stats['total_days']:.1%})")

            # Per-year
            yb = yearly_breakdown(u_trades)
            print(f"\n  {'Year':>6} {'N':>7} {'TotalR':>9} {'ExpR':>10} {'WR':>7}")
            print(f"  {'----':>6} {'-----':>7} {'------':>9} {'------':>10} {'---':>7}")
            all_positive = True
            for year in sorted(yb.keys()):
                y = yb[year]
                print(f"  {year:>6} {y['n']:>7} {y['total_r']:>+8.1f}R "
                      f"{y['exp_r']:>+9.4f} {y['wr']:>6.1%}")
                if y['total_r'] < 0:
                    all_positive = False
            print(f"\n  Every year positive: {'YES' if all_positive else 'NO'}")

            # =====================================================================
            # INCOME PROJECTION
            # =====================================================================
            # Use 2024-2025 as representative (all instruments, full coverage)
            recent_r = sum(yb[y]["total_r"] for y in yb if y in (2024, 2025))
            recent_years = sum(1 for y in yb if y in (2024, 2025))
            annual_r = recent_r / recent_years if recent_years > 0 else um["total_r"] / 5

            print(f"\n{'=' * 90}")
            print("INCOME PROJECTION (based on 2024-2025 avg annual R)")
            print(f"{'=' * 90}")
            print(f"  Avg annual R (2024-25): {annual_r:+.1f}R")
            print(f"  Max DD:                 {um['max_dd']:.1f}R")
            print(f"  Return/DD ratio:        {annual_r/um['max_dd']:.1f}x" if um['max_dd'] > 0 else "")
            print()
            print(f"  {'$ per R':>10} {'Annual $':>12} {'MaxDD $':>12} {'Min Account':>14} {'Contracts':>12}")
            print(f"  {'------':>10} {'--------':>12} {'------':>12} {'-----------':>14} {'---------':>12}")

            for r_dollar in [100, 200, 300, 400, 500]:
                annual_dollar = annual_r * r_dollar
                dd_dollar = um['max_dd'] * r_dollar
                # Account needs: MaxDD + 50% buffer for margin
                min_account = dd_dollar * 1.5
                # Rough contracts: MGC ~$200/R, MES ~$500/R, MNQ ~$160/R
                avg_r_per_contract = 250  # rough average across instruments
                contracts = r_dollar / avg_r_per_contract
                print(f"  ${r_dollar:>8} ${annual_dollar:>10,.0f} ${dd_dollar:>10,.0f} "
                      f"${min_account:>12,.0f} ~{contracts:>10.1f}")

            print()
            print("  NOTE: 'Min Account' = 1.5x MaxDD (conservative buffer).")
            print("  Actual R per contract varies by instrument and ORB size.")
            print("  Start with $100-200/R, scale up ONLY after 3+ months live.")

        # =====================================================================
        # FINAL COMPARISON
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("FINAL: Raw baseline vs Ultimate portfolio")
        print(f"{'=' * 90}")

        raw_base = compute_full_metrics(signaled_trades, start_date, end_date)
        final_m = um if best_result else raw_base

        print(f"  {'Metric':<25} {'Signals Only':>15} {'Ultimate':>15} {'Delta':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
        print(f"  {'Trades':<25} {raw_base['n']:>15,} {final_m['n']:>15,} "
              f"{final_m['n'] - raw_base['n']:>+12,}")
        print(f"  {'Total R':<25} {raw_base['total_r']:>+14.1f}R {final_m['total_r']:>+14.1f}R "
              f"{final_m['total_r'] - raw_base['total_r']:>+11.1f}R")
        if raw_base['sharpe_ann'] and final_m['sharpe_ann']:
            print(f"  {'Sharpe (ann)':<25} {raw_base['sharpe_ann']:>15.2f} "
                  f"{final_m['sharpe_ann']:>15.2f} "
                  f"{final_m['sharpe_ann'] - raw_base['sharpe_ann']:>+12.2f}")
        print(f"  {'Max DD':<25} {raw_base['max_dd']:>14.1f}R {final_m['max_dd']:>14.1f}R "
              f"{final_m['max_dd'] - raw_base['max_dd']:>+11.1f}R")
        raw_dur = f"{raw_base['dd_duration']}d" if raw_base['dd_duration'] else "N/A"
        fin_dur = f"{final_m['dd_duration']}d" if final_m['dd_duration'] else "N/A"
        print(f"  {'DD Duration':<25} {raw_dur:>15} {fin_dur:>15}")
        raw_rec = f"{raw_base['recovery_days']}d" if raw_base['recovery_days'] else "never"
        fin_rec = f"{final_m['recovery_days']}d" if final_m['recovery_days'] else "never"
        print(f"  {'Recovery':<25} {raw_rec:>15} {fin_rec:>15}")
        print(f"  {'Worst Day':<25} {raw_base['worst_day']:>+14.2f}R {final_m['worst_day']:>+14.2f}R")

    finally:
        con.close()


if __name__ == "__main__":
    main()
