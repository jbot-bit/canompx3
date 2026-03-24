#!/usr/bin/env python3
"""Paper-trade forward monitor for MNQ unfiltered baseline.

Reads orb_outcomes directly (canonical layer) for the 6 paper-trade
candidate sessions. Produces:
  1. Trade journal (CSV or stdout)
  2. Per-session summary with rolling stats
  3. Portfolio-level summary (CORE-only and CORE+REGIME)
  4. Kill-rule flags

No DB writes. No schema changes. No live execution paths.
Source of truth: PAPER_TRADE_CANDIDATES in trading_app/live_config.py.

Usage:
    python -m scripts.reports.monitor_paper_forward
    python -m scripts.reports.monitor_paper_forward --start 2026-01-01
    python -m scripts.reports.monitor_paper_forward --output forward_journal.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import duckdb

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.live_config import PAPER_TRADE_CANDIDATES

# ── Constants ───────────────────────────────────────────────────────────────
# Entry model and filter type validated against PAPER_TRADE_CANDIDATES at import.

INSTRUMENT = "MNQ"
ENTRY_MODEL = "E2"
ORB_MINUTES = 5
RR_TARGET = 1.0
CONFIRM_BARS = 1

# Validate entry_model, filter_type, and rr_target match PAPER_TRADE_CANDIDATES
for _s in PAPER_TRADE_CANDIDATES:
    assert _s.entry_model == ENTRY_MODEL, f"{_s.family_id}: entry_model mismatch"
    assert _s.filter_type == "NO_FILTER", f"{_s.family_id}: expected NO_FILTER"
    assert _s.rr_target is None or _s.rr_target == RR_TARGET, f"{_s.family_id}: rr_target mismatch"

# Kill rules
KILL_CONSECUTIVE_NEGATIVE_MONTHS = 3
KILL_CUMULATIVE_R = -10.0

# Rolling window
ROLLING_WINDOW = 20


def _t_test_p(values: list[float]) -> float | None:
    """Two-tailed t-test p-value, H0: mean=0.

    Uses normal approximation (erfc). Only valid for N >= 30.
    Returns None for N < 30 to avoid overstating significance at small samples.
    """
    n = len(values)
    if n < 30:
        return None
    mean_v = sum(values) / n
    var_v = sum((x - mean_v) ** 2 for x in values) / (n - 1)
    if var_v <= 0:
        return 1.0
    se = math.sqrt(var_v / n)
    t = mean_v / se
    return min(math.erfc(abs(t) / math.sqrt(2)), 1.0)


def _compute_session_stats(trades: list[dict]) -> dict:
    """Compute stats for a list of trade dicts with pnl_r and outcome."""
    pnls = [t["pnl_r"] for t in trades if t["pnl_r"] is not None]
    n = len(pnls)
    if n == 0:
        return {"n": 0, "wr": 0, "expr": 0, "p": None, "cum_r": 0, "max_dd": 0}
    wins = sum(1 for t in trades if t["outcome"] == "win")
    wr = wins / n
    expr = sum(pnls) / n
    p = _t_test_p(pnls)

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in pnls:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {"n": n, "wr": wr, "expr": expr, "p": p, "cum_r": cum, "max_dd": max_dd}


def _rolling_stats(pnls: list[float], window: int = ROLLING_WINDOW) -> dict:
    """Compute rolling ExpR and WR over last N trades."""
    if len(pnls) < window:
        recent = pnls
    else:
        recent = pnls[-window:]
    if not recent:
        return {"rolling_expr": None, "rolling_wr": None, "rolling_n": 0}
    mean_r = sum(recent) / len(recent)
    wins = sum(1 for r in recent if r > 0)
    return {
        "rolling_expr": round(mean_r, 4),
        "rolling_wr": round(wins / len(recent), 3),
        "rolling_n": len(recent),
    }


def _monthly_pnl(trades: list[dict]) -> dict[str, float]:
    """Sum pnl_r by YYYY-MM."""
    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        key = str(t["trading_day"])[:7]
        if t["pnl_r"] is not None:
            monthly[key] += t["pnl_r"]
    return dict(sorted(monthly.items()))


def _check_kill_rules(monthly: dict[str, float], cum_r: float) -> list[str]:
    """Check kill-rule conditions. Return list of triggered flags."""
    flags = []
    # Consecutive negative months
    months = list(monthly.values())
    if len(months) >= KILL_CONSECUTIVE_NEGATIVE_MONTHS:
        tail = months[-KILL_CONSECUTIVE_NEGATIVE_MONTHS:]
        if all(m <= 0 for m in tail):
            flags.append(
                f"KILL: {KILL_CONSECUTIVE_NEGATIVE_MONTHS} consecutive negative months "
                f"({', '.join(f'{m:+.2f}R' for m in tail)})"
            )
    # Cumulative threshold
    if cum_r <= KILL_CUMULATIVE_R:
        flags.append(f"KILL: cumulative R = {cum_r:+.1f} <= {KILL_CUMULATIVE_R}")
    return flags


def run_monitor(
    start_date: date | None = None,
    end_date: date | None = None,
    output_csv: str | None = None,
    db_path: str | None = None,
) -> None:
    """Run the paper-trade forward monitor."""
    db = Path(db_path) if db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db), read_only=True)
    cost_spec = get_cost_spec(INSTRUMENT)

    # Resolve sessions from PAPER_TRADE_CANDIDATES
    core_sessions = [
        s.orb_label for s in PAPER_TRADE_CANDIDATES if s.tier == "core"
    ]
    regime_sessions = [
        s.orb_label for s in PAPER_TRADE_CANDIDATES if s.tier == "regime"
    ]
    all_sessions = core_sessions + regime_sessions

    # Date range
    if start_date is None:
        start_date = date(2026, 1, 1)  # Default: 2026 forward
    if end_date is None:
        max_td = con.execute(
            "SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = ?",
            [INSTRUMENT],
        ).fetchone()[0]
        end_date = max_td if max_td else date.today()

    print(f"MNQ UNFILTERED PAPER-TRADE FORWARD MONITOR")
    print(f"=" * 90)
    print(f"Instrument: {INSTRUMENT} | Entry: {ENTRY_MODEL} | ORB: O{ORB_MINUTES} | RR: {RR_TARGET} | CB: {CONFIRM_BARS}")
    print(f"Period: {start_date} to {end_date}")
    print(f"CORE sessions ({len(core_sessions)}): {', '.join(core_sessions)}")
    print(f"REGIME sessions ({len(regime_sessions)}): {', '.join(regime_sessions)}")
    print(f"Friction: ${cost_spec.total_friction:.2f} RT ({cost_spec.friction_in_points:.3f} pts)")
    print(f"Kill rules: {KILL_CONSECUTIVE_NEGATIVE_MONTHS} consecutive neg months OR cumR <= {KILL_CUMULATIVE_R}")
    print()

    # ── Load trades ─────────────────────────────────────────────────────────
    all_trades: dict[str, list[dict]] = {}
    for session in all_sessions:
        rows = con.execute(
            """
            SELECT trading_day, orb_label, pnl_r, outcome,
                   entry_price, stop_price, target_price, exit_price,
                   entry_ts, exit_ts, mae_r, mfe_r
            FROM orb_outcomes
            WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
              AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
              AND outcome IN ('win', 'loss')
              AND trading_day >= ? AND trading_day <= ?
            ORDER BY trading_day
            """,
            [INSTRUMENT, session, ORB_MINUTES, ENTRY_MODEL, RR_TARGET, CONFIRM_BARS,
             start_date, end_date],
        ).fetchall()
        cols = ["trading_day", "orb_label", "pnl_r", "outcome",
                "entry_price", "stop_price", "target_price", "exit_price",
                "entry_ts", "exit_ts", "mae_r", "mfe_r"]
        all_trades[session] = [dict(zip(cols, r)) for r in rows]

    # ── Per-session summary ─────────────────────────────────────────────────
    print("PER-SESSION SUMMARY")
    print("-" * 90)
    print(f"{'Session':16s} {'Tier':>6s} {'N':>5s} {'WR':>6s} {'ExpR':>7s} {'CumR':>7s} {'MaxDD':>6s} "
          f"{'Roll20':>7s} {'p':>10s}")
    print("-" * 90)

    for session in all_sessions:
        trades = all_trades[session]
        tier = "CORE" if session in core_sessions else "REGIME"
        stats = _compute_session_stats(trades)
        pnls = [t["pnl_r"] for t in trades if t["pnl_r"] is not None]
        roll = _rolling_stats(pnls)
        roll_str = f"{roll['rolling_expr']:+.3f}" if roll["rolling_expr"] is not None else "  N/A"
        p_str = f"{stats['p']:.6f}" if stats["p"] is not None else "N/A"
        print(f"{session:16s} {tier:>6s} {stats['n']:5d} {stats['wr']:6.1%} {stats['expr']:+7.4f} "
              f"{stats['cum_r']:+7.1f} {stats['max_dd']:6.1f} {roll_str:>7s} {p_str:>10s}")

    # ── Portfolio summaries ─────────────────────────────────────────────────
    for label, sessions in [("CORE ONLY", core_sessions), ("CORE + REGIME", all_sessions)]:
        # Combine trades by day, equal-weight average
        daily: dict[date, list[float]] = defaultdict(list)
        combined_trades: list[dict] = []
        for s in sessions:
            for t in all_trades[s]:
                if t["pnl_r"] is not None:
                    daily[t["trading_day"]].append(t["pnl_r"])
                    combined_trades.append(t)

        daily_avg = {d: sum(rs) / len(rs) for d, rs in sorted(daily.items())}
        port_pnls = list(daily_avg.values())

        if not port_pnls:
            print(f"\n{label} PORTFOLIO: no trades in period")
            continue

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in port_pnls:
            cum += r
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        mean_r = sum(port_pnls) / len(port_pnls)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in port_pnls) / max(len(port_pnls) - 1, 1))
        sharpe = mean_r / std_r if std_r > 0 else 0

        monthly = _monthly_pnl(combined_trades)
        kill_flags = _check_kill_rules(monthly, cum)

        n_sessions = len(sessions)
        total_trades = sum(len(all_trades[s]) for s in sessions)

        print(f"\n{label} PORTFOLIO ({n_sessions} sessions, {total_trades} trades)")
        print("-" * 90)
        print(f"  Days: {len(port_pnls)} | Mean daily R: {mean_r:+.4f} | Std: {std_r:.4f}")
        print(f"  Daily Sharpe: {sharpe:.4f} | Cumulative R: {cum:+.1f} | Max DD: {max_dd:.1f}R")

        # Monthly breakdown
        print(f"  Monthly R:")
        for month, mr in monthly.items():
            flag = " <<<" if mr <= 0 else ""
            print(f"    {month}: {mr:+6.1f}R{flag}")

        # Kill-rule check
        if kill_flags:
            for f in kill_flags:
                print(f"  *** {f} ***")
        else:
            print(f"  Kill rules: CLEAR")

    # ── Trade journal (CSV) ─────────────────────────────────────────────────
    if output_csv:
        journal_rows = []
        for session in all_sessions:
            tier = "CORE" if session in core_sessions else "REGIME"
            pnls_so_far: list[float] = []
            cum_r = 0.0
            peak = 0.0
            max_dd_r = 0.0
            for t in all_trades[session]:
                if t["pnl_r"] is not None:
                    cum_r += t["pnl_r"]
                    peak = max(peak, cum_r)
                    max_dd_r = max(max_dd_r, peak - cum_r)
                    pnls_so_far.append(t["pnl_r"])
                roll = _rolling_stats(pnls_so_far)
                journal_rows.append({
                    "trading_day": t["trading_day"],
                    "session": session,
                    "tier": tier,
                    "side": "long" if t.get("entry_price", 0) > t.get("stop_price", 0) else "short",
                    "entry_price": t.get("entry_price"),
                    "stop_price": t.get("stop_price"),
                    "target_price": t.get("target_price"),
                    "exit_price": t.get("exit_price"),
                    "outcome": t["outcome"],
                    "pnl_r": t["pnl_r"],
                    "cum_r": round(cum_r, 2),
                    "rolling_20_expr": roll["rolling_expr"],
                    "rolling_20_wr": roll["rolling_wr"],
                    "max_dd_r": round(max_dd_r, 2),
                })

        journal_rows.sort(key=lambda r: (r["trading_day"], r["session"]))
        fieldnames = [
            "trading_day", "session", "tier", "side", "entry_price", "stop_price",
            "target_price", "exit_price", "outcome", "pnl_r", "cum_r",
            "rolling_20_expr", "rolling_20_wr", "max_dd_r",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(journal_rows)
        print(f"\nJournal exported: {output_csv} ({len(journal_rows)} rows)")

    # ── Operator note ───────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("OPERATOR NOTE")
    print("-" * 90)
    print("This is PAPER TRADE monitoring only. No live capital at risk.")
    print("2026 remains sacred for discovery — do not use forward results to select new sessions.")
    print("No rule changes allowed during the forward test period.")
    print("TOKYO_OPEN is monitored separately as REGIME (borderline at 5yr, dead at 10yr unfiltered).")
    print(f"Kill: {KILL_CONSECUTIVE_NEGATIVE_MONTHS} consecutive negative months OR cumulative R <= {KILL_CUMULATIVE_R}.")
    print("Promotion to LIVE_PORTFOLIO requires separate approval after paper-trade period.")
    print(f"{'=' * 90}")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="MNQ unfiltered paper-trade forward monitor")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Start date (default: 2026-01-01)")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="End date (default: latest orb_outcomes)")
    parser.add_argument("--output", type=str, default=None,
                        help="Export trade journal to CSV")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Override DB path")
    parser.add_argument("--all-time", action="store_true",
                        help="Show all-time stats (start from earliest data)")
    args = parser.parse_args()

    start = args.start
    if args.all_time:
        start = date(2021, 1, 1)

    run_monitor(
        start_date=start,
        end_date=args.end,
        output_csv=args.output,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
