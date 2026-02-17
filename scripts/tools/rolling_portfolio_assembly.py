#!/usr/bin/env python3
"""
Rolling Portfolio Assembly -- final strategy allocation.

Logic is LOCKED:
  0900 = Fixed Target (2.0R)
  1000 = Target Unlock (IB-aligned hold 7h, opposed kill)
  1100 = Dead
  Pyramiding = OFF

This script:
  1. Simulates both strategies across the full date range
  2. Slices into rolling 12m windows (monthly step)
  3. Computes per-window Sharpe for each session
  4. Classifies STABLE / TRANSITIONING / DEGRADED
  5. Outputs TRADING_PLAN.md with position sizing rules

Read-only. No DB writes.

Usage:
    python scripts/rolling_portfolio_assembly.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple
from research._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_ORB_SIZE = 4.0
RR_TARGET = 2.0
CONFIRM_BARS = 2
HOLD_HOURS = 7

SESSION_UTC = {"0900": 23, "1000": 0}
MARKET_OPEN_UTC_HOUR = 23

# IB config per session (locked from research)
SESSION_IB = {
    "0900": ("session", 120),
    "1000": ("mktopen", 120),
}

SPEC = get_cost_spec("MGC")

# Rolling window
WINDOW_MONTHS = 12
STEP_MONTHS = 1

# Classification thresholds
STABLE_SHARPE = 0.10
DEGRADED_SHARPE = 0.0


# ---------------------------------------------------------------------------
# IB functions (from analyze_trend_holding.py)
# ---------------------------------------------------------------------------

def compute_ib(ts, highs, lows, anchor_utc_hour, duration_minutes):
    hours = np.array([t.hour for t in ts])
    minutes = np.array([t.minute for t in ts])
    anchor_idx = np.flatnonzero((hours == anchor_utc_hour) & (minutes == 0))
    if len(anchor_idx) == 0:
        return None
    ib_start = ts[anchor_idx[0]]
    ib_end = ib_start + timedelta(minutes=duration_minutes)
    ib_mask = (ts >= ib_start) & (ts < ib_end)
    if ib_mask.sum() < max(10, duration_minutes // 12):
        return None
    return {
        "ib_high": float(highs[ib_mask].max()),
        "ib_low": float(lows[ib_mask].min()),
        "ib_end": ib_end,
    }


def find_ib_break(ts, highs, lows, ib):
    post_idx = np.flatnonzero(ts >= ib["ib_end"])
    for i in post_idx:
        bh = highs[i] > ib["ib_high"]
        bl = lows[i] < ib["ib_low"]
        if bh and bl:
            return None, ts[i], i
        if bh:
            return "long", ts[i], i
        if bl:
            return "short", ts[i], i
    return None, None, None


# ---------------------------------------------------------------------------
# Target unlock simulation (1000 exploit)
# ---------------------------------------------------------------------------

def sim_exploit(ts, highs, lows, closes, entry_idx, entry_price,
                stop_price, target_price, is_long, cutoff_ts,
                ib_break_dir, ib_break_idx, orb_dir):
    """Honest bar-by-bar: limbo target -> aligned hold / opposed kill."""
    alignment_known = False
    alignment = None

    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]

        # Stop always active
        if is_long and lo <= stop_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 stop_price - entry_price), "stop"
        if not is_long and h >= stop_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 entry_price - stop_price), "stop"

        if not alignment_known:
            # Limbo: fixed target active
            if is_long and h >= target_price:
                return to_r_multiple(SPEC, entry_price, stop_price,
                                     target_price - entry_price), "limbo_target"
            if not is_long and lo <= target_price:
                return to_r_multiple(SPEC, entry_price, stop_price,
                                     entry_price - target_price), "limbo_target"

            if ib_break_idx is not None and i >= ib_break_idx:
                alignment_known = True
                if ib_break_dir is None:
                    alignment = "no_break"
                elif ib_break_dir == orb_dir:
                    alignment = "aligned"
                else:
                    alignment = "opposed"
                    pnl = (c - entry_price) if is_long else (entry_price - c)
                    return to_r_multiple(SPEC, entry_price, stop_price, pnl), "opposed_kill"
                continue

            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return to_r_multiple(SPEC, entry_price, stop_price, pnl), "limbo_time"
            continue

        # Post-alignment
        if alignment == "aligned":
            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return to_r_multiple(SPEC, entry_price, stop_price, pnl), "time_7h"
            continue

        # no_break: keep fixed target
        if is_long and h >= target_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 target_price - entry_price), "nobreak_target"
        if not is_long and lo <= target_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 entry_price - target_price), "nobreak_target"
        if ts[i] >= cutoff_ts:
            pnl = (c - entry_price) if is_long else (entry_price - c)
            return to_r_multiple(SPEC, entry_price, stop_price, pnl), "nobreak_time"

    c = closes[-1]
    pnl = (c - entry_price) if is_long else (entry_price - c)
    return to_r_multiple(SPEC, entry_price, stop_price, pnl), "eod"


# ---------------------------------------------------------------------------
# Load and simulate
# ---------------------------------------------------------------------------

def load_and_simulate(db_path, session_label, start, end):
    """Run both fixed-target and exploit for all trades. Returns DataFrame."""
    session_utc_hour = SESSION_UTC[session_label]
    anchor, duration = SESSION_IB[session_label]
    anchor_hour = session_utc_hour if anchor == "session" else MARKET_OPEN_UTC_HOUR

    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(f"""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.pnl_r,
               d.orb_{session_label}_break_dir
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = ? AND o.entry_model = 'E1'
          AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_{session_label}_size >= ?
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [session_label, RR_TARGET, CONFIRM_BARS, MIN_ORB_SIZE, start, end]).fetchdf()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    # Bulk-load bars
    unique_days = sorted(df["trading_day"].unique())
    bars_cache = {}
    for td in unique_days:
        s, e = compute_trading_day_utc_range(td)
        b = con.execute(
            "SELECT ts_utc, high, low, close FROM bars_1m "
            "WHERE symbol='MGC' AND ts_utc>=? AND ts_utc<? ORDER BY ts_utc",
            [s, e],
        ).fetchdf()
        if not b.empty:
            b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
            ts_raw = b["ts_utc"].values.astype("datetime64[ms]")
            ts_py = np.array([pd.Timestamp(t).to_pydatetime().replace(tzinfo=None)
                              for t in ts_raw])
            bars_cache[td] = (ts_py, ts_raw,
                              b["high"].values.astype(np.float64),
                              b["low"].values.astype(np.float64),
                              b["close"].values.astype(np.float64))
    con.close()

    rows = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        if td not in bars_cache:
            continue

        ts_py, ts_raw, h_arr, l_arr, c_arr = bars_cache[td]
        entry_ts_aware = row["entry_ts"].to_pydatetime()
        entry_ts = entry_ts_aware.replace(tzinfo=None)
        entry_p = float(row["entry_price"])
        stop_p = float(row["stop_price"])
        target_p = float(row["target_price"])
        orb_dir = row[f"orb_{session_label}_break_dir"]
        is_long = orb_dir == "long"

        entry_idx = int(np.searchsorted(ts_raw, np.datetime64(entry_ts_aware, "ms")))
        if entry_idx >= len(ts_py):
            continue

        cutoff = entry_ts + timedelta(hours=HOLD_HOURS)

        # Fixed target: use stored pnl_r
        fixed_pnl = float(row["pnl_r"])

        # Exploit: simulate if we have IB
        exploit_pnl = fixed_pnl  # default: same as fixed if no IB
        ib = compute_ib(ts_py, h_arr, l_arr, anchor_hour, duration)
        if ib is not None:
            ib_dir, _, ib_break_idx = find_ib_break(ts_py, h_arr, l_arr, ib)
            exploit_pnl, _ = sim_exploit(
                ts_py, h_arr, l_arr, c_arr, entry_idx, entry_p, stop_p,
                target_p, is_long, cutoff, ib_dir, ib_break_idx, orb_dir)

        rows.append({
            "trading_day": td,
            "year": td.year if hasattr(td, "year") else int(str(td)[:4]),
            "month": td.month if hasattr(td, "month") else int(str(td)[5:7]),
            "fixed_pnl": fixed_pnl,
            "exploit_pnl": exploit_pnl,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling window analysis
# ---------------------------------------------------------------------------

def rolling_windows(trade_df, strategy_col, data_start, data_end):
    """Compute rolling 12m window metrics. Returns list of dicts."""
    windows = []
    # First test window starts WINDOW_MONTHS after data start
    first_test = date(data_start.year, data_start.month, 1) + relativedelta(months=WINDOW_MONTHS)
    current = first_test

    while current <= data_end:
        w_start = current - relativedelta(months=WINDOW_MONTHS)
        w_end = current - relativedelta(days=1)
        w_label = f"{w_start.isoformat()} to {w_end.isoformat()}"

        mask = (trade_df["trading_day"] >= pd.Timestamp(w_start)) & (trade_df["trading_day"] <= pd.Timestamp(w_end))
        window_trades = trade_df[mask]

        if len(window_trades) >= 5:
            pnls = window_trades[strategy_col].values
            m = compute_strategy_metrics(pnls)
            windows.append({
                "start": w_start,
                "end": w_end,
                "label": w_label,
                "n": m["n"],
                "wr": m["wr"],
                "expr": m["expr"],
                "sharpe": m["sharpe"],
                "total": m["total"],
            })

        current += relativedelta(months=STEP_MONTHS)

    return windows


def classify_windows(windows, n_recent=6):
    """Classify strategy based on recent window performance."""
    if len(windows) == 0:
        return "DEGRADED", 0, 0

    recent = windows[-n_recent:] if len(windows) >= n_recent else windows
    n_stable = sum(1 for w in recent if w["sharpe"] >= STABLE_SHARPE)
    n_positive = sum(1 for w in recent if w["sharpe"] > DEGRADED_SHARPE)
    n_total = len(recent)

    if n_stable >= n_total * 0.6:
        return "STABLE", n_stable, n_total
    elif n_positive >= n_total * 0.5:
        return "TRANSITIONING", n_positive, n_total
    else:
        return "DEGRADED", n_positive, n_total


# ---------------------------------------------------------------------------
# Report + TRADING_PLAN.md generation
# ---------------------------------------------------------------------------

def run(db_path, start, end):
    print("Rolling Portfolio Assembly")
    print(f"Date range: {start} to {end}")
    print(f"Window: {WINDOW_MONTHS}m rolling, {STEP_MONTHS}m step")
    print(f"Classification: STABLE >= {STABLE_SHARPE} Sharpe in 60%+ of last 6 windows")
    print()

    session_results = {}

    for session, strategy_col, strategy_name in [
        ("0900", "fixed_pnl", "Fixed Target"),
        ("1000", "exploit_pnl", "Target Unlock"),
    ]:
        print(f"Processing {session} ({strategy_name})...")
        t0 = time.time()
        tdf = load_and_simulate(db_path, session, start, end)
        elapsed = time.time() - t0
        print(f"  {len(tdf)} trades in {elapsed:.1f}s")

        if len(tdf) == 0:
            session_results[session] = {
                "classification": "DEGRADED",
                "strategy": strategy_name,
                "windows": [],
                "tdf": tdf,
            }
            continue

        td_min = tdf["trading_day"].min()
        td_max = tdf["trading_day"].max()
        # Convert pandas Timestamp to date if needed
        if hasattr(td_min, "date"):
            td_min = td_min.date() if callable(td_min.date) else td_min
        if hasattr(td_max, "date"):
            td_max = td_max.date() if callable(td_max.date) else td_max
        windows = rolling_windows(tdf, strategy_col, td_min, td_max)
        classification, n_pass, n_total = classify_windows(windows)

        session_results[session] = {
            "classification": classification,
            "strategy": strategy_name,
            "windows": windows,
            "tdf": tdf,
            "n_pass": n_pass,
            "n_total": n_total,
            "strategy_col": strategy_col,
        }

        # Print window details
        print(f"\n  {'Window':28s} {'N':>4s} {'WR':>6s} {'ExpR':>8s} "
              f"{'Sharpe':>8s} {'Total':>7s} {'Status':>12s}")
        print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*12}")

        for w in windows:
            status = "STABLE" if w["sharpe"] >= STABLE_SHARPE else (
                "positive" if w["sharpe"] > 0 else "negative")
            print(f"  {w['label']:28s} {w['n']:>4d} {w['wr']:>5.1%} "
                  f"{w['expr']:>+8.4f} {w['sharpe']:>8.4f} {w['total']:>+7.1f} "
                  f"{status:>12s}")

        # Summary
        full_m = compute_strategy_metrics(tdf[strategy_col].values)
        print(f"\n  FULL PERIOD: N={full_m['n']} WR={full_m['wr']:.1%} "
              f"ExpR={full_m['expr']:+.4f} Sharpe={full_m['sharpe']:.4f} "
              f"Total={full_m['total']:+.1f}")
        print(f"  CLASSIFICATION: {classification} "
              f"({n_pass}/{n_total} recent windows >= {STABLE_SHARPE} Sharpe)")

    # --- Generate TRADING_PLAN.md ---
    print(f"\n{'=' * 90}")
    print("TRADING PLAN")
    print(f"{'=' * 90}")

    plan_lines = [
        "# TRADING PLAN",
        f"",
        f"Generated: 2026-02-13",
        f"Data: {start} to {end} | Cost model: MGC ($10/pt, $8.40 RT)",
        f"Rolling window: {WINDOW_MONTHS}m | Classification threshold: Sharpe >= {STABLE_SHARPE}",
        f"",
        f"## Session Logic (LOCKED)",
        f"",
        f"| Session | Logic | Status | Rationale |",
        f"|---------|-------|--------|-----------|",
    ]

    for session in ["0900", "1000", "1100"]:
        if session == "1100":
            plan_lines.append(
                f"| {session} | OFF | DEAD | 74% double-break, IB/ORB tautology |")
            continue

        sr = session_results[session]
        plan_lines.append(
            f"| {session} | {sr['strategy']} | {sr['classification']} | "
            f"{sr.get('n_pass', 0)}/{sr.get('n_total', 0)} recent windows pass |")

    plan_lines.extend([
        f"",
        f"## Position Sizing Rules",
        f"",
    ])

    for session in ["0900", "1000"]:
        sr = session_results[session]
        cls = sr["classification"]

        if cls == "STABLE":
            sizing = "Normal (1.0x risk)"
            if session == "1000":
                sizing = "Half (0.5x risk) -- thinner edge, higher variance"
        elif cls == "TRANSITIONING":
            sizing = "Half (0.5x risk)" if session == "0900" else "Quarter (0.25x risk)"
        else:
            sizing = "OFF (0x) -- regime filter active"

        plan_lines.append(f"- **{session} ({sr['strategy']})**: {sizing}")
        plan_lines.append(f"  - Classification: {cls}")

        if sr["windows"]:
            recent = sr["windows"][-3:]
            recent_sharpes = [f"{w['sharpe']:.3f}" for w in recent]
            plan_lines.append(f"  - Last 3 window Sharpes: {', '.join(recent_sharpes)}")

        full_m = compute_strategy_metrics(sr["tdf"][sr["strategy_col"]].values) if len(sr["tdf"]) > 0 else None
        if full_m:
            plan_lines.append(
                f"  - Full period: N={full_m['n']} WR={full_m['wr']:.1%} "
                f"ExpR={full_m['expr']:+.3f} Sharpe={full_m['sharpe']:.3f}")

    plan_lines.extend([
        f"",
        f"## Pyramiding",
        f"",
        f"**OFF** -- destroyed value at both sessions. Intraday mean-reversion snap-back",
        f"kills the second unit. Do not revisit.",
        f"",
        f"## 1000 Target Unlock Rules",
        f"",
        f"1. Entry: E1 CB2 G4+ (standard ORB break)",
        f"2. Limbo phase: Fixed target ({RR_TARGET}R) + stop active",
        f"3. IB breaks ALIGNED: cancel target, hold {HOLD_HOURS}h with stop only",
        f"4. IB breaks OPPOSED: exit at market immediately",
        f"5. No IB break within {HOLD_HOURS}h: fixed target stays active",
        f"6. IB definition: market open (0900 Brisbane) + 120min",
        f"",
        f"## Rolling Re-evaluation",
        f"",
        f"- Run monthly: `python scripts/rolling_portfolio_assembly.py --db-path C:/db/gold.db`",
        f"- STABLE -> TRANSITIONING: reduce size by 50%",
        f"- TRANSITIONING -> DEGRADED: turn OFF",
        f"- DEGRADED -> STABLE: requires 3 consecutive passing windows before re-entry",
    ])

    plan_text = "\n".join(plan_lines) + "\n"

    # Print to console
    for line in plan_lines:
        print(f"  {line}")

    # Write TRADING_PLAN.md
    plan_path = PROJECT_ROOT / "TRADING_PLAN.md"
    plan_path.write_text(plan_text)
    print(f"\n  Written to {plan_path}")


def main():
    parser = argparse.ArgumentParser(description="Rolling Portfolio Assembly")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()
    run(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()
