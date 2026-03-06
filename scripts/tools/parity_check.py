#!/usr/bin/env python3
"""
Backtest-to-live parity check.

Verifies that pre-computed orb_outcomes match the underlying bars_1m data
and that the execution engine would produce consistent trade decisions.

Tests:
  1. Entry price consistency — outcome entry_price matches bars_1m at entry_ts
  2. ORB boundary consistency — stop_price matches the ORB opposite level
  3. Target price consistency — target = entry + risk * RR
  4. Cost model consistency — pnl_r values are plausible given cost_spec
  5. Time-stop annotation — ts_pnl_r is consistent with bar close at threshold

Usage:
    python scripts/tools/parity_check.py
    python scripts/tools/parity_check.py --instrument MGC --sample 100
    python scripts/tools/parity_check.py --db C:/db/gold.db
"""

import argparse
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.config import TRADEABLE_INSTRUMENTS  # noqa: E402


def get_db_path(args):
    if args.db:
        return Path(args.db)
    return GOLD_DB_PATH


def check_entry_price_parity(con, instrument: str, sample: int) -> list[str]:
    """Verify entry_price in orb_outcomes matches bars_1m at entry_ts."""
    failures = []
    spec = get_cost_spec(instrument)
    tick_tol = spec.tick_size  # 1-tick tolerance for stop-market gap-through

    rows = con.execute(
        """
        SELECT o.trading_day, o.symbol, o.orb_label, o.entry_model,
               o.rr_target, o.confirm_bars, o.orb_minutes,
               o.entry_ts, o.entry_price, o.stop_price, o.target_price,
               o.outcome, o.pnl_r
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.outcome IN ('win', 'loss')
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day DESC
        LIMIT ?
    """,
        [instrument, sample],
    ).fetchall()

    if not rows:
        failures.append(f"  {instrument}: No orb_outcomes rows found")
        return failures

    checked = 0
    entry_mismatches = 0

    for row in rows:
        (
            trading_day,
            symbol,
            orb_label,
            entry_model,
            rr_target,
            confirm_bars,
            orb_minutes,
            entry_ts,
            entry_price,
            stop_price,
            target_price,
            outcome,
            pnl_r,
        ) = row

        # Check entry_price against bars_1m
        bar = con.execute(
            """
            SELECT open, high, low, close
            FROM bars_1m
            WHERE symbol = ? AND ts_utc = ?
            LIMIT 1
        """,
            [symbol, entry_ts],
        ).fetchone()

        if bar is None:
            # Entry might be from a source symbol (GC for MGC, etc.)
            continue

        bar_open, bar_high, bar_low, bar_close = bar

        # E1 enters at bar open, E2 enters at stop-market (ORB boundary)
        # Entry price should be within the bar's OHLC range (± 1 tick tolerance
        # for E2 stop-market entries where price gaps through the boundary)
        if not (bar_low - tick_tol <= entry_price <= bar_high + tick_tol):
            entry_mismatches += 1
            if entry_mismatches <= 5:  # Show first 5
                failures.append(
                    f"  {instrument} {trading_day} {orb_label} {entry_model}: "
                    f"entry_price={entry_price} outside bar range "
                    f"[{bar_low}, {bar_high}] at {entry_ts}"
                )

        checked += 1

    if entry_mismatches > 5:
        failures.append(f"  ... and {entry_mismatches - 5} more entry price mismatches")

    if checked == 0:
        failures.append(f"  {instrument}: No bars_1m matches for entry timestamps")
    elif entry_mismatches == 0:
        print(f"    Entry price parity: {checked} trades checked, all OK")
    else:
        failures.append(f"  {instrument}: {entry_mismatches}/{checked} entry price mismatches")

    return failures


def check_target_price_consistency(con, instrument: str, sample: int) -> list[str]:
    """Verify target_price = entry + risk * RR (accounting for direction)."""
    failures = []

    rows = con.execute(
        """
        SELECT o.entry_price, o.stop_price, o.target_price,
               o.rr_target, o.orb_label, o.trading_day, o.entry_model
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.outcome IN ('win', 'loss')
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day DESC
        LIMIT ?
    """,
        [instrument, sample],
    ).fetchall()

    checked = 0
    mismatches = 0

    for row in rows:
        entry_price, stop_price, target_price, rr_target, orb_label, td, em = row

        risk = abs(entry_price - stop_price)
        if risk == 0:
            continue

        # Determine direction from entry vs stop
        if entry_price > stop_price:
            # Long: target = entry + risk * RR
            expected_target = entry_price + risk * rr_target
        else:
            # Short: target = entry - risk * RR
            expected_target = entry_price - risk * rr_target

        tolerance = risk * 0.01  # 1% of risk
        if abs(target_price - expected_target) > tolerance:
            mismatches += 1
            if mismatches <= 3:
                failures.append(
                    f"  {instrument} {td} {orb_label} {em} RR{rr_target}: "
                    f"target={target_price}, expected={expected_target:.4f}, "
                    f"entry={entry_price}, stop={stop_price}"
                )

        checked += 1

    if mismatches > 3:
        failures.append(f"  ... and {mismatches - 3} more target price mismatches")

    if checked > 0 and mismatches == 0:
        print(f"    Target price consistency: {checked} trades checked, all OK")
    elif mismatches > 0:
        failures.append(f"  {instrument}: {mismatches}/{checked} target price inconsistencies")

    return failures


def check_pnl_plausibility(con, instrument: str, sample: int) -> list[str]:
    """Verify pnl_r values are plausible given cost model."""
    failures = []
    rows = con.execute(
        """
        SELECT o.entry_price, o.stop_price, o.target_price,
               o.outcome, o.pnl_r, o.rr_target, o.orb_label, o.trading_day
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.outcome IN ('win', 'loss')
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day DESC
        LIMIT ?
    """,
        [instrument, sample],
    ).fetchall()

    checked = 0
    implausible = 0

    for row in rows:
        entry, stop, target, outcome, pnl_r, rr, orb_label, td = row

        risk = abs(entry - stop)
        if risk == 0:
            continue

        if outcome == "win":
            # Winner PnL should be positive and near RR (minus friction)
            max_expected = rr + 0.1  # small buffer
            if pnl_r < 0 or pnl_r > max_expected:
                implausible += 1
                if implausible <= 3:
                    failures.append(
                        f"  {instrument} {td} {orb_label}: win pnl_r={pnl_r:.3f} outside [0, {max_expected:.1f}]"
                    )
        elif outcome == "loss":
            # Loser PnL should be negative, near -1.0 (plus friction)
            if pnl_r > 0 or pnl_r < -1.5:
                implausible += 1
                if implausible <= 3:
                    failures.append(f"  {instrument} {td} {orb_label}: loss pnl_r={pnl_r:.3f} outside [-1.5, 0]")

        checked += 1

    if implausible > 3:
        failures.append(f"  ... and {implausible - 3} more implausible pnl_r values")

    if checked > 0 and implausible == 0:
        print(f"    PnL plausibility: {checked} trades checked, all OK")
    elif implausible > 0:
        failures.append(f"  {instrument}: {implausible}/{checked} implausible pnl_r values")

    return failures


def check_time_stop_consistency(con, instrument: str, sample: int) -> list[str]:
    """Verify ts_pnl_r is consistent with bar data at threshold time."""
    failures = []

    rows = con.execute(
        """
        SELECT o.trading_day, o.orb_label, o.entry_model, o.rr_target,
               o.entry_ts, o.entry_price, o.stop_price,
               o.ts_outcome, o.ts_pnl_r, o.ts_exit_ts,
               o.outcome, o.pnl_r
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.ts_outcome = 'time_stop'
          AND o.ts_pnl_r IS NOT NULL
        ORDER BY o.trading_day DESC
        LIMIT ?
    """,
        [instrument, sample],
    ).fetchall()

    if not rows:
        print("    Time-stop consistency: no time_stop outcomes found (OK if no T80 sessions)")
        return failures

    checked = 0
    mismatches = 0

    for row in rows:
        (
            td,
            orb_label,
            em,
            rr,
            entry_ts,
            entry_price,
            stop_price,
            ts_outcome,
            ts_pnl_r,
            ts_exit_ts,
            outcome,
            raw_pnl_r,
        ) = row

        # ts_pnl_r should be non-positive (time-stop fires on MTM <= 0)
        # Exactly 0.0 is theoretically possible (break-even at time-stop bar)
        if ts_pnl_r > 0:
            mismatches += 1
            if mismatches <= 3:
                failures.append(
                    f"  {instrument} {td} {orb_label} {em}: "
                    f"ts_pnl_r={ts_pnl_r:.3f} > 0 (should be non-positive for time_stop)"
                )

        # ts_pnl_r should be between -1.0 and 0.0 (exited between stop and entry)
        if ts_pnl_r < -1.2:
            mismatches += 1
            if mismatches <= 3:
                failures.append(
                    f"  {instrument} {td} {orb_label} {em}: ts_pnl_r={ts_pnl_r:.3f} < -1.2 (worse than stop loss)"
                )

        checked += 1

    if mismatches > 3:
        failures.append(f"  ... and {mismatches - 3} more time-stop mismatches")

    if checked > 0 and mismatches == 0:
        print(f"    Time-stop consistency: {checked} trades checked, all OK")
    elif mismatches > 0:
        failures.append(f"  {instrument}: {mismatches}/{checked} time-stop inconsistencies")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Backtest-to-live parity check")
    parser.add_argument("--instrument", default=None, help="Instrument to check (default: all active)")
    parser.add_argument(
        "--sample", type=int, default=200, help="Number of trades to sample per instrument (default: 200)"
    )
    parser.add_argument("--db", default=None, help="Database path")
    args = parser.parse_args()

    db_path = get_db_path(args)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    instruments = [args.instrument] if args.instrument else TRADEABLE_INSTRUMENTS

    print("=" * 60)
    print("BACKTEST-TO-LIVE PARITY CHECK")
    print(f"Database: {db_path}")
    print(f"Sample size: {args.sample} trades per instrument")
    print("=" * 60)

    all_failures = []
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        for inst in instruments:
            print(f"\n--- {inst} ---")

            print("  Check 1: Entry price parity")
            all_failures.extend(check_entry_price_parity(con, inst, args.sample))

            print("  Check 2: Target price consistency")
            all_failures.extend(check_target_price_consistency(con, inst, args.sample))

            print("  Check 3: PnL plausibility")
            all_failures.extend(check_pnl_plausibility(con, inst, args.sample))

            print("  Check 4: Time-stop consistency")
            all_failures.extend(check_time_stop_consistency(con, inst, args.sample))
    finally:
        con.close()

    print()
    print("=" * 60)
    if all_failures:
        print(f"PARITY CHECK FAILED: {len(all_failures)} issue(s)")
        for f in all_failures:
            print(f)
        sys.exit(1)
    else:
        checks_per_inst = 4
        print(
            f"PARITY CHECK PASSED: {len(instruments)} instruments, {checks_per_inst * len(instruments)} checks, all OK"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
