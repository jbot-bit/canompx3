#!/usr/bin/env python3
"""
ORB reversal scalp research: when a breakout trade is losing at N minutes,
flip direction and target the opposite ORB edge (not a breakout — just a touch).

Idea: if 88% of 1000 losers at 30 min hit their stop (= opposite ORB edge),
then reversing and targeting that edge should have ~88% hit rate.

Tests across all sessions/entry models. Measures actual distance, reward, risk.

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_orb_reversal_scalp.py --db-path C:/db/gold.db
    python scripts/analyze_orb_reversal_scalp.py --sessions 1000 --check-minutes 30
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SESSIONS = ["0900", "1000", "1800", "2300"]
DEFAULT_CHECK_MINUTES = [15, 30]
DEFAULT_MIN_ORB_SIZE = 4.0

COST_SPEC = get_cost_spec("MGC")
FRICTION_POINTS = COST_SPEC.friction_in_points  # 0.84 points

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_outcomes(db_path: Path, sessions: list[str], min_orb_size: float,
                  start: date, end: date) -> pd.DataFrame:
    """Load orb_outcomes with ORB levels for reversal analysis."""
    session_placeholders = ", ".join(["?"] * len(sessions))

    # Get ORB high/low/size and break_dir from daily_features
    size_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_size" for s in sessions
    )
    high_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_high" for s in sessions
    )
    low_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_low" for s in sessions
    )
    dir_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_break_dir" for s in sessions
    )

    query = f"""
        SELECT
            o.trading_day, o.orb_label, o.rr_target, o.confirm_bars,
            o.entry_model, o.entry_ts, o.entry_price, o.stop_price,
            o.target_price, o.outcome, o.pnl_r,
            CASE {size_cases} ELSE NULL END AS orb_size,
            CASE {high_cases} ELSE NULL END AS orb_high,
            CASE {low_cases} ELSE NULL END AS orb_low,
            CASE {dir_cases} ELSE NULL END AS break_dir
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol
            AND o.trading_day = d.trading_day
            AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC'
            AND o.orb_minutes = 5
            AND o.orb_label IN ({session_placeholders})
            AND o.entry_model IN ('E1', 'E3')
            AND o.rr_target IN (1.5, 2.0, 2.5)
            AND o.entry_ts IS NOT NULL
            AND o.outcome IS NOT NULL
            AND o.pnl_r IS NOT NULL
            AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day, o.orb_label
    """
    params = sessions + [start, end]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    # Apply ORB size filter
    df = df[df["orb_size"] >= min_orb_size].copy()

    # Deduplicate E3 CB>1
    df = df[~((df["entry_model"] == "E3") & (df["confirm_bars"] > 1))].copy()

    # Keep only one RR per (day, session, entry_model) — the reversal scalp
    # doesn't depend on the original RR target, so deduplicate
    df = df.drop_duplicates(
        subset=["trading_day", "orb_label", "entry_model", "confirm_bars"],
        keep="first"
    ).copy()

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    print(f"Loaded {len(df)} unique trade entries ({df['trading_day'].nunique()} days)")
    return df

def load_bars_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 1-minute bars for one trading day."""
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, [start_utc, end_utc]).fetchdf()
    finally:
        con.close()
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df

# ---------------------------------------------------------------------------
# Reversal scalp simulation
# ---------------------------------------------------------------------------

def simulate_reversal_scalp(
    bars_df: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    stop_price: float,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    check_minutes: list[int],
) -> list[dict]:
    """For each check minute, if trade is losing, simulate a reversal scalp.

    Reversal: flip direction, target = opposite ORB edge (the original stop),
    stop = the ORB edge we originally broke through.

    Returns list of result dicts, one per check_minute.
    """
    is_long = break_dir == "long"
    orb_size = orb_high - orb_low

    post_entry = bars_df[bars_df["ts_utc"] > pd.Timestamp(entry_ts)].sort_values("ts_utc")
    if post_entry.empty:
        return []

    results = []

    for check_min in check_minutes:
        if check_min > len(post_entry):
            continue

        # Bar at check_min (0-indexed: bar 0 = 1 min after entry)
        check_bar = post_entry.iloc[check_min - 1]
        check_close = check_bar["close"]

        # Is the original trade losing?
        if is_long:
            original_losing = check_close < entry_price
        else:
            original_losing = check_close > entry_price

        if not original_losing:
            results.append({
                "check_min": check_min,
                "triggered": False,
            })
            continue

        # Is price inside the ORB?
        inside_orb = orb_low <= check_close <= orb_high
        if not inside_orb:
            # Price already past the ORB on the other side — don't reverse
            results.append({
                "check_min": check_min,
                "triggered": False,
                "reason": "already_past_orb",
            })
            continue

        # --- REVERSAL SCALP ---
        # Original was long, we go short. Target = orb_low (touch). Stop = orb_high.
        # Original was short, we go long. Target = orb_high (touch). Stop = orb_low.
        rev_entry = check_close
        if is_long:
            # Reverse to short
            rev_target = orb_low    # opposite edge (the original stop level)
            rev_stop = orb_high     # the edge we broke through
            rev_dir = "short"
        else:
            # Reverse to long
            rev_target = orb_high
            rev_stop = orb_low
            rev_dir = "long"

        rev_is_long = rev_dir == "long"

        # Distance to target and stop in points
        if rev_is_long:
            dist_to_target = rev_target - rev_entry
            dist_to_stop = rev_entry - rev_stop
        else:
            dist_to_target = rev_entry - rev_target
            dist_to_stop = rev_stop - rev_entry

        # Scan forward from check bar
        remaining_bars = post_entry.iloc[check_min:]
        if remaining_bars.empty:
            results.append({
                "check_min": check_min,
                "triggered": True,
                "rev_outcome": "no_bars",
                "rev_entry": rev_entry,
                "dist_to_target_pts": dist_to_target,
                "dist_to_stop_pts": dist_to_stop,
                "orb_size": orb_size,
            })
            continue

        rev_outcome = None
        rev_exit_price = None
        rev_bars_held = 0

        for i, (_, bar) in enumerate(remaining_bars.iterrows()):
            rev_bars_held = i + 1
            if rev_is_long:
                hit_target = bar["high"] >= rev_target
                hit_stop = bar["low"] <= rev_stop
            else:
                hit_target = bar["low"] <= rev_target
                hit_stop = bar["high"] >= rev_stop

            if hit_target and hit_stop:
                # Ambiguous — conservative loss
                rev_outcome = "loss"
                rev_exit_price = rev_stop
                break
            elif hit_target:
                rev_outcome = "win"
                rev_exit_price = rev_target
                break
            elif hit_stop:
                rev_outcome = "loss"
                rev_exit_price = rev_stop
                break

        if rev_outcome is None:
            # Never hit target or stop — scratch at last bar close
            rev_outcome = "scratch"
            rev_exit_price = remaining_bars.iloc[-1]["close"]
            rev_bars_held = len(remaining_bars)

        # Compute PnL
        if rev_is_long:
            pnl_points = rev_exit_price - rev_entry
        else:
            pnl_points = rev_entry - rev_exit_price

        pnl_dollars = pnl_points * COST_SPEC.point_value - COST_SPEC.total_friction

        results.append({
            "check_min": check_min,
            "triggered": True,
            "rev_dir": rev_dir,
            "rev_entry": rev_entry,
            "rev_target": rev_target,
            "rev_stop": rev_stop,
            "rev_outcome": rev_outcome,
            "rev_exit_price": rev_exit_price,
            "pnl_points": pnl_points,
            "pnl_dollars": pnl_dollars,
            "dist_to_target_pts": dist_to_target,
            "dist_to_stop_pts": dist_to_stop,
            "orb_size": orb_size,
            "rev_bars_held": rev_bars_held,
        })

    return results

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_results: list[dict], check_minutes: list[int],
                 start: date, end: date, min_orb_size: float):
    """Print structured report."""
    print("=" * 90)
    print("ORB REVERSAL SCALP ANALYSIS")
    print("=" * 90)
    print(f"Period: {start} to {end} | Filter: G{int(min_orb_size)}+")
    print("Logic: when losing at N min and still inside ORB, flip to target opposite ORB edge")
    print("=" * 90)

    # Group by (session, entry_model, check_min)
    grouped = defaultdict(list)
    for r in all_results:
        for scalp in r["scalps"]:
            key = (r["orb_label"], r["entry_model"], scalp["check_min"])
            grouped[key].append(scalp)

    # Also group original trades for context
    orig_grouped = defaultdict(list)
    for r in all_results:
        key = (r["orb_label"], r["entry_model"])
        orig_grouped[key].append(r)

    for session in sorted(set(r["orb_label"] for r in all_results)):
        for em in ["E1", "E3"]:
            orig_key = (session, em)
            if orig_key not in orig_grouped:
                continue

            orig_trades = orig_grouped[orig_key]
            print(f"\n--- {session} {em} ({len(orig_trades)} original entries) ---")

            for check_min in check_minutes:
                key = (session, em, check_min)
                scalps = grouped.get(key, [])
                triggered = [s for s in scalps if s.get("triggered")]
                not_triggered = len(scalps) - len(triggered)

                if not triggered:
                    print(f"\n  {check_min}-min check: 0 reversals triggered")
                    continue

                # Filter to actual scalps (not no_bars)
                actual = [s for s in triggered if s.get("rev_outcome") in ("win", "loss", "scratch")]
                if not actual:
                    print(f"\n  {check_min}-min check: {len(triggered)} triggered but no bars to trade")
                    continue

                wins = [s for s in actual if s["rev_outcome"] == "win"]
                losses = [s for s in actual if s["rev_outcome"] == "loss"]
                scratches = [s for s in actual if s["rev_outcome"] == "scratch"]

                n = len(actual)
                wr = len(wins) / n * 100 if n > 0 else 0

                pnl_arr = np.array([s["pnl_dollars"] for s in actual])
                total_pnl = pnl_arr.sum()
                avg_pnl = pnl_arr.mean()
                avg_win = np.mean([s["pnl_dollars"] for s in wins]) if wins else 0
                avg_loss = np.mean([s["pnl_dollars"] for s in losses]) if losses else 0

                avg_target_dist = np.mean([s["dist_to_target_pts"] for s in actual])
                avg_stop_dist = np.mean([s["dist_to_stop_pts"] for s in actual])
                avg_orb = np.mean([s["orb_size"] for s in actual])
                avg_bars = np.mean([s["rev_bars_held"] for s in actual])

                # Yearly breakdown
                yearly = defaultdict(list)
                for r in all_results:
                    for scalp in r["scalps"]:
                        if (scalp.get("check_min") == check_min
                                and scalp.get("triggered")
                                and scalp.get("rev_outcome") in ("win", "loss", "scratch")
                                and r["orb_label"] == session
                                and r["entry_model"] == em):
                            td = r["trading_day"]
                            if hasattr(td, 'year'):
                                yr = td.year
                            else:
                                yr = pd.Timestamp(td).year
                            yearly[yr].append(scalp["pnl_dollars"])

                print(f"\n  {check_min}-min check: {n} reversals "
                      f"({len(orig_trades) - not_triggered - len(triggered) + len(triggered)} checked, "
                      f"{n} inside ORB)")
                print(f"    Win rate:     {wr:.1f}%  ({len(wins)}W / {len(losses)}L / {len(scratches)}S)")
                print(f"    Avg PnL:      ${avg_pnl:+.2f}/trade")
                print(f"    Total PnL:    ${total_pnl:+,.0f}  ({n} trades)")
                print(f"    Avg win:      ${avg_win:+.2f}  |  Avg loss: ${avg_loss:+.2f}")
                print(f"    Avg target:   {avg_target_dist:.1f} pts  |  Avg stop: {avg_stop_dist:.1f} pts")
                print(f"    Avg ORB size: {avg_orb:.1f} pts  |  Avg bars held: {avg_bars:.0f} min")

                # MaxDD in dollars
                cumul = np.cumsum(pnl_arr)
                peak = np.maximum.accumulate(cumul)
                maxdd = float((cumul - peak).min())
                print(f"    MaxDD:        ${maxdd:+,.0f}")

                # Per year
                if yearly:
                    print(f"    By year:")
                    for yr in sorted(yearly.keys()):
                        yr_pnls = yearly[yr]
                        yr_n = len(yr_pnls)
                        yr_total = sum(yr_pnls)
                        yr_wr = sum(1 for p in yr_pnls if p > 0) / yr_n * 100
                        print(f"      {yr}: {yr_n:3d} trades, "
                              f"WR={yr_wr:5.1f}%, "
                              f"Total=${yr_total:+,.0f}, "
                              f"Avg=${yr_total/yr_n:+.0f}")

    # Summary: best combos
    print("\n" + "=" * 90)
    print("SUMMARY: Best reversal scalp combos (sorted by avg $/trade)")
    print("=" * 90)

    summary = []
    for (session, em, check_min), scalps in grouped.items():
        actual = [s for s in scalps
                  if s.get("triggered") and s.get("rev_outcome") in ("win", "loss", "scratch")]
        if len(actual) < 10:
            continue
        pnls = [s["pnl_dollars"] for s in actual]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg = np.mean(pnls)
        total = sum(pnls)
        summary.append({
            "label": f"{session} {em} @{check_min}min",
            "n": len(actual),
            "wr": wr,
            "avg": avg,
            "total": total,
        })

    summary.sort(key=lambda x: x["avg"], reverse=True)
    for s in summary:
        print(f"  {s['label']:20s}  N={s['n']:<4d}  WR={s['wr']:5.1f}%  "
              f"Avg=${s['avg']:+7.1f}  Total=${s['total']:+,.0f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ORB reversal scalp analysis"
    )
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--sessions", type=str, default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--check-minutes", type=str,
                        default=",".join(str(m) for m in DEFAULT_CHECK_MINUTES))
    parser.add_argument("--min-orb-size", type=float, default=DEFAULT_MIN_ORB_SIZE)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    parser.add_argument("--confirm-bars", type=int, default=2)
    args = parser.parse_args()

    sessions = args.sessions.split(",")
    check_minutes = [int(m) for m in args.check_minutes.split(",")]

    print(f"Loading outcomes from {args.db_path}...")
    outcomes_df = load_outcomes(
        db_path=args.db_path,
        sessions=sessions,
        min_orb_size=args.min_orb_size,
        start=args.start,
        end=args.end,
    )

    if outcomes_df.empty:
        print("No trades found. Exiting.")
        return

    # Filter E1 to requested CB
    mask = (
        (outcomes_df["entry_model"] == "E3")
        | (outcomes_df["confirm_bars"] == args.confirm_bars)
    )
    outcomes_df = outcomes_df[mask].copy()
    print(f"After CB filter: {len(outcomes_df)} entries")

    # Group by trading_day
    day_groups = outcomes_df.groupby("trading_day")
    unique_days = sorted(outcomes_df["trading_day"].unique())
    total_days = len(unique_days)

    print(f"Simulating reversals across {total_days} days...")
    t0 = time.monotonic()

    all_results = []

    for day_idx, trading_day in enumerate(unique_days):
        if hasattr(trading_day, 'date'):
            td = trading_day.date()
        elif isinstance(trading_day, np.datetime64):
            td = pd.Timestamp(trading_day).date()
        else:
            td = trading_day

        bars_df = load_bars_for_day(args.db_path, td)
        if bars_df.empty:
            continue

        day_trades = day_groups.get_group(trading_day)

        for _, trade in day_trades.iterrows():
            scalps = simulate_reversal_scalp(
                bars_df=bars_df,
                entry_ts=trade["entry_ts"],
                entry_price=float(trade["entry_price"]),
                stop_price=float(trade["stop_price"]),
                orb_high=float(trade["orb_high"]),
                orb_low=float(trade["orb_low"]),
                break_dir=trade["break_dir"],
                check_minutes=check_minutes,
            )

            all_results.append({
                "trading_day": td,
                "orb_label": trade["orb_label"],
                "entry_model": trade["entry_model"],
                "orb_size": float(trade["orb_size"]),
                "scalps": scalps,
            })

        if (day_idx + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (day_idx + 1) / elapsed
            remaining = (total_days - day_idx - 1) / rate if rate > 0 else 0
            print(f"  {day_idx + 1}/{total_days} days ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.monotonic() - t0
    print(f"Done: {len(all_results)} entries processed in {elapsed:.1f}s")

    print_report(all_results, check_minutes, args.start, args.end, args.min_orb_size)

if __name__ == "__main__":
    main()
