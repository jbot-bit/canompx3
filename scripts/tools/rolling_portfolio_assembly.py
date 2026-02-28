#!/usr/bin/env python3
"""
Rolling Portfolio Assembly -- multi-instrument strategy allocation.

For each instrument, identifies the best edge family per session (session slots),
runs rolling 12-month window analysis, and classifies each slot as
STABLE / TRANSITIONING / DEGRADED.

Position sizing follows classification:
  STABLE = full size, TRANSITIONING = half, DEGRADED = off.

Outputs TRADING_PLAN.md with per-instrument, per-session sizing rules.

Read-only. No DB writes.

Usage:
    python scripts/tools/rolling_portfolio_assembly.py                    # MGC only (default)
    python scripts/tools/rolling_portfolio_assembly.py --instrument MNQ   # single instrument
    python scripts/tools/rolling_portfolio_assembly.py --all              # all 4 instruments
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import duckdb
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_MONTHS = 12
STEP_MONTHS = 1
MIN_WINDOW_TRADES = 5

# Classification thresholds
STABLE_SHARPE = 0.10
DEGRADED_SHARPE = 0.0

SIZING = {
    "STABLE": "Full (1.0x)",
    "TRANSITIONING": "Half (0.5x)",
    "DEGRADED": "OFF (0x)",
}

# ---------------------------------------------------------------------------
# Data loading — edge family session slots
# ---------------------------------------------------------------------------

def load_slot_data(db_path, instrument):
    """Load session slot heads and their trade outcomes.

    A "session slot" is the best edge family per (instrument, session),
    ranked by annualized Sharpe then trade count.

    Returns:
        slots: list of dicts with slot metadata
        session_trades: dict[session -> DataFrame(trading_day, pnl_r)]
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Step 1: Find slot heads (best family per session)
        slot_rows = con.execute("""
            WITH ranked AS (
                SELECT ef.instrument,
                       vs.orb_label AS session,
                       ef.head_strategy_id,
                       ef.head_expectancy_r,
                       ef.head_sharpe_ann,
                       ef.trade_day_count,
                       ef.trade_tier,
                       ef.member_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY ef.instrument, vs.orb_label
                           ORDER BY ef.head_sharpe_ann DESC,
                                    ef.trade_day_count DESC
                       ) AS rn
                FROM edge_families ef
                JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
                WHERE ef.robustness_status IN ('ROBUST', 'WHITELISTED', 'SINGLETON')
                  AND ef.instrument = ?
            )
            SELECT instrument, session, head_strategy_id,
                   head_expectancy_r, head_sharpe_ann,
                   trade_day_count, trade_tier, member_count
            FROM ranked WHERE rn = 1
            ORDER BY head_sharpe_ann DESC
        """, [instrument]).fetchall()

        slot_cols = [
            "instrument", "session", "head_strategy_id",
            "head_expectancy_r", "head_sharpe_ann",
            "trade_day_count", "trade_tier", "member_count",
        ]
        slots = [dict(zip(slot_cols, r)) for r in slot_rows]

        if not slots:
            return [], {}

        # Step 2: Load trades for all slot heads
        strategy_ids = [s["head_strategy_id"] for s in slots]
        placeholders = ", ".join(["?"] * len(strategy_ids))

        trade_df = con.execute(f"""
            SELECT vs.orb_label AS session,
                   oo.trading_day,
                   oo.pnl_r
            FROM validated_setups vs
            JOIN strategy_trade_days std ON vs.strategy_id = std.strategy_id
            JOIN orb_outcomes oo
              ON oo.symbol = vs.instrument
              AND oo.orb_label = vs.orb_label
              AND oo.orb_minutes = vs.orb_minutes
              AND oo.entry_model = vs.entry_model
              AND oo.rr_target = vs.rr_target
              AND oo.confirm_bars = vs.confirm_bars
              AND oo.trading_day = std.trading_day
            WHERE vs.strategy_id IN ({placeholders})
              AND oo.outcome IN ('win', 'loss')
              AND oo.pnl_r IS NOT NULL
            ORDER BY vs.orb_label, oo.trading_day
        """, strategy_ids).fetchdf()

        # Group by session
        session_trades = {}
        if not trade_df.empty:
            for session in trade_df["session"].unique():
                mask = trade_df["session"] == session
                sdf = trade_df[mask][["trading_day", "pnl_r"]].copy()
                sdf = sdf.sort_values("trading_day").reset_index(drop=True)
                session_trades[session] = sdf

        return slots, session_trades
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Rolling window analysis
# ---------------------------------------------------------------------------

def rolling_windows(trade_df, data_start, data_end):
    """Compute rolling 12m window metrics."""
    windows = []
    first_test = date(data_start.year, data_start.month, 1) + relativedelta(
        months=WINDOW_MONTHS
    )
    current = first_test

    while current <= data_end:
        w_start = current - relativedelta(months=WINDOW_MONTHS)
        w_end = current - relativedelta(days=1)
        mask = (trade_df["trading_day"] >= pd.Timestamp(w_start)) & (
            trade_df["trading_day"] <= pd.Timestamp(w_end)
        )
        window_trades = trade_df[mask]

        if len(window_trades) >= MIN_WINDOW_TRADES:
            m = compute_strategy_metrics(window_trades["pnl_r"].values)
            if m is not None:
                windows.append({
                    "start": w_start,
                    "end": w_end,
                    "label": f"{w_start} to {w_end}",
                    **m,
                })

        current += relativedelta(months=STEP_MONTHS)

    return windows


def classify_windows(windows, n_recent=6):
    """Classify: STABLE / TRANSITIONING / DEGRADED."""
    if not windows:
        return "DEGRADED", 0, 0

    recent = windows[-n_recent:]
    n_stable = sum(1 for w in recent if w["sharpe"] >= STABLE_SHARPE)
    n_positive = sum(1 for w in recent if w["sharpe"] > DEGRADED_SHARPE)
    n_total = len(recent)

    if n_stable >= n_total * 0.6:
        return "STABLE", n_stable, n_total
    if n_positive >= n_total * 0.5:
        return "TRANSITIONING", n_positive, n_total
    return "DEGRADED", n_positive, n_total


# ---------------------------------------------------------------------------
# Per-instrument processing
# ---------------------------------------------------------------------------

def process_instrument(db_path, instrument):
    """Run rolling analysis on all session slots for an instrument."""
    spec = get_cost_spec(instrument)

    print(f"\n{'='*80}")
    print(f"  {instrument}  (${spec.point_value}/pt, ${spec.total_friction:.2f} RT)")
    print(f"{'='*80}")

    slots, session_trades = load_slot_data(db_path, instrument)

    if not slots:
        print("  No active session slots.")
        return {}

    results = {}

    for slot in slots:
        session = slot["session"]
        head_id = slot["head_strategy_id"]
        tier = slot["trade_tier"]
        tdf = session_trades.get(session, pd.DataFrame())

        print(f"\n  {session} [{head_id}]")
        print(f"  {len(tdf)} trades | tier={tier} | families={slot['member_count']}")

        if tdf.empty:
            results[session] = {
                "classification": "DEGRADED",
                "head_id": head_id,
                "tier": tier,
                "windows": [],
                "n_pass": 0,
                "n_total": 0,
                "full_metrics": None,
                "slot": slot,
            }
            print("  >> DEGRADED (no trades)")
            continue

        # Date range
        td_min = tdf["trading_day"].min()
        td_max = tdf["trading_day"].max()
        if hasattr(td_min, "date") and callable(td_min.date):
            td_min = td_min.date()
        if hasattr(td_max, "date") and callable(td_max.date):
            td_max = td_max.date()

        windows = rolling_windows(tdf, td_min, td_max)
        classification, n_pass, n_total = classify_windows(windows)
        full_m = compute_strategy_metrics(tdf["pnl_r"].values)

        results[session] = {
            "classification": classification,
            "head_id": head_id,
            "tier": tier,
            "windows": windows,
            "n_pass": n_pass,
            "n_total": n_total,
            "full_metrics": full_m,
            "slot": slot,
        }

        # Print window table
        if windows:
            print(
                f"\n  {'Window':28s} {'N':>4s} {'WR':>6s} {'ExpR':>8s} "
                f"{'Sharpe':>8s} {'Total':>7s} {'Status':>12s}"
            )
            print(
                f"  {'-'*28} {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*12}"
            )

            for w in windows:
                status = (
                    "STABLE"
                    if w["sharpe"] >= STABLE_SHARPE
                    else "positive" if w["sharpe"] > 0 else "negative"
                )
                print(
                    f"  {w['label']:28s} {w['n']:>4d} {w['wr']:>5.1%} "
                    f"{w['expr']:>+8.4f} {w['sharpe']:>8.4f} {w['total']:>+7.1f} "
                    f"{status:>12s}"
                )

        if full_m:
            print(
                f"\n  FULL: N={full_m['n']} WR={full_m['wr']:.1%} "
                f"ExpR={full_m['expr']:+.4f} Sharpe={full_m['sharpe']:.4f} "
                f"Total={full_m['total']:+.1f}"
            )
        print(f"  >> {classification} ({n_pass}/{n_total} recent windows)")

    return results


# ---------------------------------------------------------------------------
# TRADING_PLAN.md
# ---------------------------------------------------------------------------

def generate_trading_plan(all_results):
    """Write multi-instrument TRADING_PLAN.md."""
    today = date.today().isoformat()

    lines = [
        "# TRADING PLAN",
        "",
        f"Generated: {today}",
        (
            f"Rolling window: {WINDOW_MONTHS}m | Classification: "
            f"STABLE >= {STABLE_SHARPE} Sharpe in 60%+ of last 6 windows"
        ),
        "",
        "## Portfolio Overview",
        "",
        (
            "| Instrument | Session | Head Strategy | Status | Size "
            "| N | ExpR | Sharpe(ann) |"
        ),
        (
            "|-----------|---------|---------------|--------|------"
            "|---|------|-------------|"
        ),
    ]

    total_slots = 0
    active_slots = 0

    for instrument in sorted(all_results.keys()):
        for session in sorted(all_results[instrument].keys()):
            sr = all_results[instrument][session]
            cls = sr["classification"]
            sizing = SIZING[cls]
            head_id = sr["head_id"]
            slot = sr.get("slot", {})
            fm = sr.get("full_metrics") or {}
            n = fm.get("n", 0)
            expr = slot.get("head_expectancy_r", 0) or 0
            sha = slot.get("head_sharpe_ann", 0) or 0
            tier = sr.get("tier", "")
            tier_tag = f" [{tier}]" if tier and tier != "CORE" else ""

            lines.append(
                f"| {instrument} | {session} | `{head_id}`{tier_tag} | "
                f"{cls} | {sizing} | {n} | {expr:+.3f} | {sha:.2f} |"
            )
            total_slots += 1
            if cls != "DEGRADED":
                active_slots += 1

    lines.extend(["", f"**Active: {active_slots}/{total_slots} session slots**", ""])

    # Per-instrument detail
    for instrument in sorted(all_results.keys()):
        inst_results = all_results[instrument]
        if not inst_results:
            continue

        spec = get_cost_spec(instrument)
        lines.extend([
            f"## {instrument} (${spec.point_value}/pt, ${spec.total_friction:.2f} RT)",
            "",
        ])

        for session in sorted(inst_results.keys()):
            sr = inst_results[session]
            cls = sr["classification"]
            sizing = SIZING[cls]
            lines.append(f"- **{session}**: {cls} -> {sizing}")
            lines.append(f"  - Head: `{sr['head_id']}`")

            fm = sr.get("full_metrics")
            if fm:
                lines.append(
                    f"  - Full period: N={fm['n']} WR={fm['wr']:.1%} "
                    f"ExpR={fm['expr']:+.4f} Sharpe={fm['sharpe']:.4f}"
                )

            if sr["windows"]:
                recent = sr["windows"][-3:]
                sharpes = ", ".join(f"{w['sharpe']:.3f}" for w in recent)
                lines.append(f"  - Last 3 window Sharpes: {sharpes}")
            lines.append("")

    lines.extend([
        "## Position Sizing Rules",
        "",
        "- **STABLE**: Full size (1.0x risk per trade)",
        "- **TRANSITIONING**: Half size (0.5x risk per trade)",
        "- **DEGRADED**: OFF — do not trade until 3 consecutive passing windows",
        "",
        "## Rolling Re-evaluation",
        "",
        "- Run monthly: `python scripts/tools/rolling_portfolio_assembly.py --all`",
        "- STABLE -> TRANSITIONING: reduce size by 50%",
        "- TRANSITIONING -> DEGRADED: turn OFF",
        "- DEGRADED -> STABLE: requires 3 consecutive passing windows",
        "",
    ])

    plan_text = "\n".join(lines) + "\n"

    print(f"\n{'='*80}")
    print("TRADING PLAN")
    print(f"{'='*80}")
    for line in lines:
        print(f"  {line}")

    plan_path = PROJECT_ROOT / "TRADING_PLAN.md"
    plan_path.write_text(plan_text)
    print(f"\n  Written to {plan_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path, instruments):
    print("Rolling Portfolio Assembly")
    print(f"Window: {WINDOW_MONTHS}m rolling, {STEP_MONTHS}m step")
    print(f"Classification: STABLE >= {STABLE_SHARPE} Sharpe in 60%+ of last 6 windows")
    print(f"Instruments: {', '.join(instruments)}")

    all_results = {}
    t0_all = time.time()

    for instrument in instruments:
        t0 = time.time()
        all_results[instrument] = process_instrument(db_path, instrument)
        print(f"\n  {instrument} complete in {time.time() - t0:.1f}s")

    generate_trading_plan(all_results)
    print(f"\nTotal: {time.time() - t0_all:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Rolling Portfolio Assembly")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument(
        "--instrument",
        type=str,
        help="Single instrument (MGC/MNQ/MES/M2K)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all active instruments",
    )
    args = parser.parse_args()

    if args.instrument:
        instruments = [args.instrument.upper()]
    elif args.all:
        instruments = sorted(ACTIVE_ORB_INSTRUMENTS)
    else:
        instruments = ["MGC"]

    run(args.db_path, instruments)


if __name__ == "__main__":
    main()
