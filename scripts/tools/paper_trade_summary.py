#!/usr/bin/env python3
"""
Daily paper trade summary for Apex MNQ lanes.

Usage:
    python scripts/tools/paper_trade_summary.py           # Full summary
    python scripts/tools/paper_trade_summary.py --today    # Today's trades only
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH


def summary(*, today_only: bool = False) -> None:
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)

        # Check table exists
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        if "paper_trades" not in tables:
            print("paper_trades table does not exist. Run: python -m trading_app.paper_trade_logger")
            return

        total = con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        if total == 0:
            print("No paper trades. Run: python -m trading_app.paper_trade_logger")
            return

        # Per-lane summary (WR excludes scratches — only wins+losses in denominator)
        lanes = con.execute("""
            SELECT
                lane_name,
                COUNT(*) AS n,
                ROUND(SUM(pnl_r), 2) AS cum_r,
                ROUND(
                    AVG(CASE WHEN exit_reason IN ('win', 'loss')
                        THEN CASE WHEN exit_reason = 'win' THEN 1.0 ELSE 0.0 END
                    END) * 100, 1
                ) AS wr_pct,
                MIN(trading_day) AS first_day,
                MAX(trading_day) AS last_day
            FROM paper_trades
            GROUP BY lane_name
            ORDER BY cum_r DESC
        """).fetchall()

        # Today's trades
        today = datetime.date.today()
        today_trades = con.execute(
            "SELECT lane_name, direction, exit_reason, ROUND(pnl_r, 2) "
            "FROM paper_trades WHERE trading_day = ? ORDER BY lane_name",
            [today],
        ).fetchall()

        # Stale lane check (no trade in 5+ calendar days)
        stale = con.execute("""
            SELECT lane_name, MAX(trading_day) AS last_trade,
                   DATE_DIFF('day', MAX(trading_day), CURRENT_DATE) AS days_ago
            FROM paper_trades
            GROUP BY lane_name
            HAVING DATE_DIFF('day', MAX(trading_day), CURRENT_DATE) >= 5
        """).fetchall()

        # Portfolio totals (WR excludes scratches)
        portfolio = con.execute("""
            SELECT COUNT(*), ROUND(SUM(pnl_r), 2),
                   ROUND(
                       AVG(CASE WHEN exit_reason IN ('win', 'loss')
                           THEN CASE WHEN exit_reason = 'win' THEN 1.0 ELSE 0.0 END
                       END) * 100, 1
                   )
            FROM paper_trades
        """).fetchone()

    # ── Print ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("PAPER TRADE SUMMARY — Apex MNQ Lanes (2026 OOS)")
    print("=" * 65)

    if today_only:
        if today_trades:
            print(f"\nToday ({today}):")
            for t in today_trades:
                pnl_str = f"{t[3]:+.2f}R" if t[3] is not None else " 0.00R"
                print(f"  {t[0]:<20s}  {t[1]:<6s}  {t[2]:<6s}  {pnl_str}")
        else:
            print(f"\nNo trades today ({today}).")
        print()
        return

    print(f"\n{'Lane':<20s}  {'N':>4s}  {'CumR':>7s}  {'WR%':>5s}  {'First':>12s}  {'Last':>12s}")
    print("-" * 65)
    for lane in lanes:
        wr_str = f"{lane[3]:5.1f}" if lane[3] is not None else "  N/A"
        print(
            f"  {lane[0]:<20s}  {lane[1]:>3d}  {lane[2]:>+7.2f}  {wr_str}  "
            f"{str(lane[4]):>12s}  {str(lane[5]):>12s}"
        )
    print("-" * 65)
    p_wr = f"{portfolio[2]:5.1f}" if portfolio[2] is not None else "  N/A"
    print(f"  {'PORTFOLIO':<20s}  {portfolio[0]:>3d}  {portfolio[1]:>+7.2f}  {p_wr}")
    print()

    if today_trades:
        print(f"Today ({today}):")
        for t in today_trades:
            print(f"  {t[0]:<20s}  {t[1]:<6s}  {t[2]:<6s}  {t[3]:+.2f}R")
    else:
        print(f"No trades today ({today}).")

    if stale:
        print("\nSTALE LANES (no trade in 5+ days):")
        for s in stale:
            print(f"  {s[0]:<20s}  last trade: {s[1]}  ({s[2]} days ago)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Paper trade daily summary")
    parser.add_argument("--today", action="store_true", help="Show only today's trades")
    args = parser.parse_args()
    summary(today_only=args.today)


if __name__ == "__main__":
    main()
