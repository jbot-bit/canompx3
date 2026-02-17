#!/usr/bin/env python3
"""
Early exit rules research: replay trades bar-by-bar and test 4 independent
early-exit rules across all sessions and entry models.

Rules tested:
  1. N-Minute Loser Check (N=10,15,20,30): if losing at N min, exit at close
  2. Retrace Dwell Time (M=5,10,15): after +0.3R MFE, if price dwells M
     consecutive minutes within 0.15R of entry, exit at close
  3. Breakeven Trail (T=0.5R,0.75R,1.0R): once MFE >= T, move stop to entry
  4. First N-Bar Momentum (K=3,5): if first K bars ALL close in wrong
     direction, exit at close of bar K

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_early_exits.py --db-path C:/db/gold.db
    python scripts/analyze_early_exits.py --sessions 0900,1800 --min-orb-size 4
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SESSIONS = ["0900", "1000", "1800", "2300"]
DEFAULT_ENTRY_MODELS = ["E1", "E3"]
DEFAULT_RR_TARGETS = [1.5, 2.0, 2.5]
DEFAULT_MIN_ORB_SIZE = 4.0

# Rule parameters
RULE1_MINUTES = [10, 15, 20, 30]
RULE2_DWELL_MINS = [5, 10, 15]
RULE2_MFE_THRESHOLD_R = 0.3
RULE2_ENTRY_ZONE_R = 0.15
RULE3_MFE_TRIGGERS = [0.5, 0.75, 1.0]
RULE4_BAR_COUNTS = [3, 5]

ALL_RULE_KEYS = (
    [f"rule1_N{n}" for n in RULE1_MINUTES]
    + [f"rule2_M{m}" for m in RULE2_DWELL_MINS]
    + [f"rule3_T{str(t).replace('.', '')}" for t in RULE3_MFE_TRIGGERS]
    + [f"rule4_K{k}" for k in RULE4_BAR_COUNTS]
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_outcomes(db_path: Path, sessions: list[str], entry_models: list[str],
                  rr_targets: list[float], min_orb_size: float,
                  start: date, end: date) -> pd.DataFrame:
    """Load orb_outcomes with entry_ts NOT NULL, joined with ORB size filter."""
    session_placeholders = ", ".join(["?"] * len(sessions))
    em_placeholders = ", ".join(["?"] * len(entry_models))
    rr_placeholders = ", ".join(["?"] * len(rr_targets))

    # Build ORB size CASE expression for filtering
    size_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_size" for s in sessions
    )
    size_expr = f"CASE {size_cases} ELSE NULL END"

    # Also get break_dir from daily_features
    dir_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_break_dir" for s in sessions
    )
    dir_expr = f"CASE {dir_cases} ELSE NULL END"

    query = f"""
        SELECT
            o.trading_day, o.orb_label, o.rr_target, o.confirm_bars,
            o.entry_model, o.entry_ts, o.entry_price, o.stop_price,
            o.target_price, o.outcome, o.pnl_r,
            {size_expr} AS orb_size,
            {dir_expr} AS break_dir
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol
            AND o.trading_day = d.trading_day
            AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC'
            AND o.orb_minutes = 5
            AND o.orb_label IN ({session_placeholders})
            AND o.entry_model IN ({em_placeholders})
            AND o.rr_target IN ({rr_placeholders})
            AND o.entry_ts IS NOT NULL
            AND o.outcome IS NOT NULL
            AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day, o.orb_label
    """
    params = sessions + entry_models + rr_targets + [start, end]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    # Apply ORB size filter
    df = df[df["orb_size"] >= min_orb_size].copy()

    # Drop scratches (pnl_r is NULL) — no bar-by-bar replay possible
    before = len(df)
    df = df[df["pnl_r"].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} scratches (NULL pnl_r)")

    # Deduplicate: for E3, CB1-CB5 are identical — keep CB1 only
    df = df[~((df["entry_model"] == "E3") & (df["confirm_bars"] > 1))].copy()

    # Convert timestamps to UTC
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    print(f"Loaded {len(df)} trades ({df['trading_day'].nunique()} days)")
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
# Per-trade replay
# ---------------------------------------------------------------------------

def replay_trade_with_exits(
    bars_df: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    stop_price: float,
    target_price: float,
    break_dir: str,
    cost_spec,
    original_outcome: str,
    original_pnl_r: float,
) -> dict:
    """Replay one trade bar-by-bar, applying all 4 exit rules independently.

    Returns dict keyed by rule variant with {pnl_r, triggered} for each.
    """
    is_long = break_dir == "long"
    risk_points = abs(entry_price - stop_price)

    # Initialize results — default to original outcome (rule did not trigger)
    results = {"original": {"pnl_r": original_pnl_r, "outcome": original_outcome}}
    for key in ALL_RULE_KEYS:
        results[key] = {"pnl_r": original_pnl_r, "triggered": False}

    if risk_points <= 0:
        return results

    # Get post-entry bars (strictly after entry)
    post_entry = bars_df[bars_df["ts_utc"] > pd.Timestamp(entry_ts)].sort_values("ts_utc")
    if post_entry.empty:
        return results

    # State tracking
    max_favorable_r = 0.0
    has_gone_green = False       # MFE >= RULE2_MFE_THRESHOLD_R
    consecutive_dwell = 0        # consecutive minutes in entry zone after going green
    wrong_dir_streak = 0         # consecutive bars closing in wrong direction
    breakeven_triggers = {t: False for t in RULE3_MFE_TRIGGERS}  # per threshold

    # Track which rules have already fired (first trigger only)
    rule_fired = {key: False for key in ALL_RULE_KEYS}

    # Pre-compute entry zone boundary in points
    entry_zone_pts = RULE2_ENTRY_ZONE_R * risk_points

    for bar_idx, (_, bar) in enumerate(post_entry.iterrows()):
        minutes_since = bar_idx + 1  # bar 0 = 1 min after entry

        bar_close = bar["close"]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_open = bar["open"]

        # Check if original stop/target hit on this bar (for MFE tracking)
        if is_long:
            hit_stop = bar_low <= stop_price
            hit_target = bar_high >= target_price
            mark_to_market_pts = bar_close - entry_price
            favorable_pts = bar_high - entry_price
        else:
            hit_stop = bar_high >= stop_price
            hit_target = bar_low <= target_price
            mark_to_market_pts = entry_price - bar_close
            favorable_pts = entry_price - bar_low

        # Update running MFE
        favorable_r = favorable_pts / risk_points if risk_points > 0 else 0.0
        if favorable_r > max_favorable_r:
            max_favorable_r = favorable_r

        # Has gone green check (for Rule 2)
        if max_favorable_r >= RULE2_MFE_THRESHOLD_R:
            has_gone_green = True

        # Dwell tracking (Rule 2): check if close is within entry zone
        if has_gone_green:
            close_dist = abs(bar_close - entry_price)
            if close_dist <= entry_zone_pts:
                consecutive_dwell += 1
            else:
                consecutive_dwell = 0
        else:
            consecutive_dwell = 0

        # Wrong direction streak (Rule 4): bar close vs bar open
        if is_long:
            wrong_close = bar_close < bar_open  # bearish bar for long
        else:
            wrong_close = bar_close > bar_open  # bullish bar for short

        if wrong_close:
            wrong_dir_streak += 1
        else:
            wrong_dir_streak = 0

        # Breakeven tracking (Rule 3): mark active per threshold
        for t in RULE3_MFE_TRIGGERS:
            if max_favorable_r >= t:
                breakeven_triggers[t] = True

        # Compute early-exit pnl_r at bar close
        def _exit_pnl_r():
            pnl_pts = mark_to_market_pts
            return round(to_r_multiple(cost_spec, entry_price, stop_price, pnl_pts), 4)

        # --- Rule 1: N-Minute Loser Check ---
        for n in RULE1_MINUTES:
            key = f"rule1_N{n}"
            if not rule_fired[key] and minutes_since == n:
                if mark_to_market_pts < 0:  # in loss
                    rule_fired[key] = True
                    results[key] = {"pnl_r": _exit_pnl_r(), "triggered": True}
                # If not in loss at minute N, rule doesn't trigger — keep original

        # --- Rule 2: Retrace Dwell Time ---
        for m in RULE2_DWELL_MINS:
            key = f"rule2_M{m}"
            if not rule_fired[key] and consecutive_dwell >= m:
                rule_fired[key] = True
                results[key] = {"pnl_r": _exit_pnl_r(), "triggered": True}

        # --- Rule 3: Breakeven Trail ---
        for t in RULE3_MFE_TRIGGERS:
            key = f"rule3_T{str(t).replace('.', '')}"
            if not rule_fired[key] and breakeven_triggers[t]:
                # Breakeven stop active: check if price retraced to entry
                if is_long:
                    breached = bar_low <= entry_price
                else:
                    breached = bar_high >= entry_price
                if breached:
                    rule_fired[key] = True
                    # Exit at entry_price (breakeven), compute actual R
                    be_pnl_pts = 0.0  # breakeven = 0 points PnL
                    be_pnl_r = round(
                        to_r_multiple(cost_spec, entry_price, stop_price, be_pnl_pts), 4
                    )
                    results[key] = {"pnl_r": be_pnl_r, "triggered": True}

        # --- Rule 4: First N-Bar Momentum ---
        for k in RULE4_BAR_COUNTS:
            key = f"rule4_K{k}"
            if not rule_fired[key] and minutes_since == k:
                if wrong_dir_streak >= k:
                    rule_fired[key] = True
                    results[key] = {"pnl_r": _exit_pnl_r(), "triggered": True}

        # If original trade would have exited on this bar (stop or target),
        # stop scanning — no rule can trigger after original exit
        if hit_stop or hit_target:
            break

    return results

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_metrics(pnls: np.ndarray) -> dict:
    """Compute trading stats from array of R-multiples."""
    n = len(pnls)
    if n == 0:
        return {"n": 0, "wr": 0.0, "expr": 0.0, "sharpe": 0.0,
                "maxdd": 0.0, "total": 0.0}
    wr = float((pnls > 0).sum() / n)
    expr = float(pnls.mean())
    std = float(pnls.std())
    sharpe = expr / std if std > 0 else 0.0
    cumul = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumul)
    maxdd = float((cumul - peak).min())
    total = float(pnls.sum())
    return {"n": n, "wr": wr, "expr": expr, "sharpe": sharpe,
            "maxdd": maxdd, "total": total}

def aggregate_results(all_results: list[dict]) -> dict:
    """Group results by (session, entry_model) and compute metrics per rule."""
    # Group by (orb_label, entry_model)
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["orb_label"], r["entry_model"])
        grouped[key].append(r)

    report = {}
    for (session, em), trades in sorted(grouped.items()):
        group_key = f"{session}_{em}"
        original_pnls = np.array([t["replay"]["original"]["pnl_r"] for t in trades])
        report[group_key] = {
            "original": compute_metrics(original_pnls),
            "rules": {},
        }

        for rule_key in ALL_RULE_KEYS:
            modified_pnls = np.array([t["replay"][rule_key]["pnl_r"] for t in trades])
            triggered_trades = [
                t for t in trades if t["replay"][rule_key]["triggered"]
            ]
            triggered_count = len(triggered_trades)
            metrics = compute_metrics(modified_pnls)
            metrics["triggered"] = triggered_count
            metrics["triggered_pct"] = (
                triggered_count / len(trades) * 100 if trades else 0.0
            )

            # Validation: original WR of the triggered subset
            # (e.g., Rule1 N30 triggered subset should show ~24% original WR for 1800 E3)
            if triggered_trades:
                orig_pnls_triggered = np.array([
                    t["replay"]["original"]["pnl_r"] for t in triggered_trades
                ])
                metrics["triggered_orig_wr"] = float(
                    (orig_pnls_triggered > 0).sum() / len(orig_pnls_triggered)
                )
                metrics["triggered_orig_expr"] = float(orig_pnls_triggered.mean())
            else:
                metrics["triggered_orig_wr"] = None
                metrics["triggered_orig_expr"] = None
            report[group_key]["rules"][rule_key] = metrics

    return report

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_metrics(m: dict, triggered: bool = False) -> str:
    """Format metrics dict as a compact string."""
    parts = [
        f"N={m['n']:<4d}",
        f"WR={m['wr']*100:5.1f}%",
        f"ExpR={m['expr']:+.3f}",
        f"Sharpe={m['sharpe']:.3f}",
        f"MaxDD={m['maxdd']:.1f}R",
    ]
    if triggered and "triggered_pct" in m:
        trig_str = f"[{m['triggered_pct']:.0f}% triggered"
        if m.get("triggered_orig_wr") is not None:
            trig_str += f", orig WR={m['triggered_orig_wr']*100:.0f}%"
        trig_str += "]"
        parts.append(trig_str)
    return "  ".join(parts)

def print_report(report: dict, start: date, end: date, min_orb_size: float):
    """Print the structured report."""
    rule_names = {
        "rule1": "Rule 1 (N-min loser check)",
        "rule2": "Rule 2 (Retrace dwell)",
        "rule3": "Rule 3 (Breakeven trail)",
        "rule4": "Rule 4 (First N-bar momentum)",
    }
    rule_param_labels = {}
    for n in RULE1_MINUTES:
        rule_param_labels[f"rule1_N{n}"] = f"  {n}-min check:"
    for m in RULE2_DWELL_MINS:
        rule_param_labels[f"rule2_M{m}"] = f"  {m}-min dwell:"
    for t in RULE3_MFE_TRIGGERS:
        rule_param_labels[f"rule3_T{str(t).replace('.', '')}"] = f"  +{t}R trigger:"
    for k in RULE4_BAR_COUNTS:
        rule_param_labels[f"rule4_K{k}"] = f"  {k}-bar check:"

    print("=" * 80)
    print("EARLY EXIT ANALYSIS REPORT")
    print("=" * 80)
    print(f"Period: {start} to {end} | Filter: G{int(min_orb_size)}+ | "
          f"Entries: E1, E3")
    print("=" * 80)

    improvements = []

    for group_key, data in sorted(report.items()):
        orig = data["original"]
        print(f"\n--- {group_key} (N={orig['n']} original trades) ---")
        print(f"Original:            {fmt_metrics(orig)}")

        # Group rules by prefix
        current_rule = None
        for rule_key in ALL_RULE_KEYS:
            rule_prefix = rule_key.split("_")[0] + "_" + rule_key.split("_")[1][0]
            # Normalize: rule1, rule2, rule3, rule4
            rule_id = rule_key[:5]  # "rule1", "rule2", etc.

            if rule_id != current_rule:
                current_rule = rule_id
                print(f"\n{rule_names[rule_id]}:")

            m = data["rules"][rule_key]
            label = rule_param_labels[rule_key]
            print(f"{label:22s} {fmt_metrics(m, triggered=True)}")

            # Track improvements
            if orig["sharpe"] != 0:
                delta_sharpe = m["sharpe"] - orig["sharpe"]
                delta_expr = m["expr"] - orig["expr"]
                improvements.append({
                    "group": group_key,
                    "rule": rule_key,
                    "orig_sharpe": orig["sharpe"],
                    "new_sharpe": m["sharpe"],
                    "delta_sharpe": delta_sharpe,
                    "delta_expr": delta_expr,
                    "triggered_pct": m.get("triggered_pct", 0),
                })

    # Best improvements
    improvements.sort(key=lambda x: x["delta_sharpe"], reverse=True)
    print("\n" + "=" * 80)
    print("BEST IMPROVEMENTS (sorted by delta Sharpe, top 15)")
    print("=" * 80)
    for i, imp in enumerate(improvements[:15]):
        pct = ((imp["new_sharpe"] / imp["orig_sharpe"]) - 1) * 100 if imp["orig_sharpe"] != 0 else 0
        print(
            f"  {i+1:2d}. {imp['group']:12s} + {imp['rule']:12s}: "
            f"Sharpe {imp['orig_sharpe']:.3f} -> {imp['new_sharpe']:.3f} "
            f"({imp['delta_sharpe']:+.3f}), "
            f"ExpR {imp['delta_expr']:+.3f}, "
            f"{imp['triggered_pct']:.0f}% triggered"
        )

    # Worst degradations
    print("\n" + "=" * 80)
    print("WORST DEGRADATIONS (sorted by delta Sharpe, bottom 15)")
    print("=" * 80)
    for i, imp in enumerate(improvements[-15:][::-1]):
        print(
            f"  {i+1:2d}. {imp['group']:12s} + {imp['rule']:12s}: "
            f"Sharpe {imp['orig_sharpe']:.3f} -> {imp['new_sharpe']:.3f} "
            f"({imp['delta_sharpe']:+.3f}), "
            f"ExpR {imp['delta_expr']:+.3f}, "
            f"{imp['triggered_pct']:.0f}% triggered"
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze early exit rules across sessions and entry models"
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path("C:/db/gold.db"),
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--sessions", type=str, default=",".join(DEFAULT_SESSIONS),
        help="Comma-separated ORB sessions (default: 0900,1000,1800,2300)"
    )
    parser.add_argument(
        "--entry-models", type=str, default=",".join(DEFAULT_ENTRY_MODELS),
        help="Comma-separated entry models (default: E1,E3)"
    )
    parser.add_argument(
        "--rr-targets", type=str, default=",".join(str(r) for r in DEFAULT_RR_TARGETS),
        help="Comma-separated RR targets (default: 1.5,2.0,2.5)"
    )
    parser.add_argument(
        "--min-orb-size", type=float, default=DEFAULT_MIN_ORB_SIZE,
        help="Minimum ORB size in points (default: 4.0)"
    )
    parser.add_argument(
        "--start", type=date.fromisoformat, default=date(2021, 1, 1),
        help="Start date (default: 2021-01-01)"
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, default=date(2026, 2, 4),
        help="End date (default: 2026-02-04)"
    )
    parser.add_argument(
        "--confirm-bars", type=int, default=2,
        help="Confirm bars for E1 (default: 2, E3 always uses 1)"
    )
    args = parser.parse_args()

    sessions = args.sessions.split(",")
    entry_models = args.entry_models.split(",")
    rr_targets = [float(r) for r in args.rr_targets.split(",")]

    print(f"Loading outcomes from {args.db_path}...")
    outcomes_df = load_outcomes(
        db_path=args.db_path,
        sessions=sessions,
        entry_models=entry_models,
        rr_targets=rr_targets,
        min_orb_size=args.min_orb_size,
        start=args.start,
        end=args.end,
    )

    if outcomes_df.empty:
        print("No trades found matching criteria. Exiting.")
        return

    # Further filter: keep only the requested CB for E1
    mask = (
        (outcomes_df["entry_model"] == "E3")
        | (outcomes_df["confirm_bars"] == args.confirm_bars)
    )
    outcomes_df = outcomes_df[mask].copy()
    print(f"After CB filter (E1 CB{args.confirm_bars}): {len(outcomes_df)} trades")

    cost_spec = get_cost_spec("MGC")

    # Group by trading_day for efficient bar loading
    day_groups = outcomes_df.groupby("trading_day")
    unique_days = sorted(outcomes_df["trading_day"].unique())
    total_days = len(unique_days)

    print(f"Replaying {len(outcomes_df)} trades across {total_days} trading days...")
    t0 = time.monotonic()

    all_results = []

    for day_idx, trading_day in enumerate(unique_days):
        # Convert numpy date to python date
        if hasattr(trading_day, 'date'):
            td = trading_day.date() if hasattr(trading_day, 'date') else trading_day
        elif isinstance(trading_day, np.datetime64):
            td = pd.Timestamp(trading_day).date()
        else:
            td = trading_day

        bars_df = load_bars_for_day(args.db_path, td)
        if bars_df.empty:
            continue

        day_trades = day_groups.get_group(trading_day)

        for _, trade in day_trades.iterrows():
            replay = replay_trade_with_exits(
                bars_df=bars_df,
                entry_ts=trade["entry_ts"],
                entry_price=float(trade["entry_price"]),
                stop_price=float(trade["stop_price"]),
                target_price=float(trade["target_price"]),
                break_dir=trade["break_dir"],
                cost_spec=cost_spec,
                original_outcome=trade["outcome"],
                original_pnl_r=float(trade["pnl_r"]),
            )

            all_results.append({
                "trading_day": td,
                "orb_label": trade["orb_label"],
                "entry_model": trade["entry_model"],
                "rr_target": trade["rr_target"],
                "replay": replay,
            })

        if (day_idx + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (day_idx + 1) / elapsed
            remaining = (total_days - day_idx - 1) / rate if rate > 0 else 0
            print(
                f"  {day_idx + 1}/{total_days} days "
                f"({len(all_results)} trades replayed, "
                f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    elapsed = time.monotonic() - t0
    print(f"Replay complete: {len(all_results)} trades in {elapsed:.1f}s")

    # Aggregate and report
    report = aggregate_results(all_results)
    print_report(report, args.start, args.end, args.min_orb_size)

if __name__ == "__main__":
    main()
