"""
Pre-compute outcomes for all RR targets x confirm_bars combinations.

Populates the orb_outcomes table from daily_features + bars_1m data.
For each (trading_day, orb_label) with a break, computes outcomes at
multiple RR targets (1.0-4.0) and confirm_bars (1-5).

Usage:
    python trading_app/outcome_builder.py --instrument MGC --start 2024-01-01 --end 2024-12-31
    python trading_app/outcome_builder.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --dry-run
"""

import sys
import time
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, pnl_points_to_r
from pipeline.init_db import ORB_LABELS
from pipeline.build_daily_features import compute_trading_day_utc_range
from trading_app.entry_rules import detect_entry_with_confirm_bars
from trading_app.db_manager import init_trading_app_schema

# Grid parameters
RR_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
CONFIRM_BARS_OPTIONS = [1, 2, 3, 4, 5]


def compute_single_outcome(
    bars_df: pd.DataFrame,
    break_ts: datetime,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    rr_target: float,
    confirm_bars: int,
    trading_day_end: datetime,
    cost_spec,
) -> dict:
    """
    Compute outcome for a single (rr_target, confirm_bars) combination.

    Returns dict with keys matching orb_outcomes columns.
    """
    result = {
        "entry_ts": None,
        "entry_price": None,
        "stop_price": None,
        "target_price": None,
        "outcome": None,
        "exit_ts": None,
        "exit_price": None,
        "pnl_r": None,
        "mae_r": None,
        "mfe_r": None,
    }

    # Detect entry with confirm bars
    signal = detect_entry_with_confirm_bars(
        bars_df=bars_df,
        orb_break_ts=break_ts,
        orb_high=orb_high,
        orb_low=orb_low,
        break_dir=break_dir,
        confirm_bars=confirm_bars,
        detection_window_end=trading_day_end,
    )

    if not signal.triggered:
        return result

    entry_price = signal.entry_price
    stop_price = signal.stop_price
    entry_ts = signal.entry_ts
    risk_points = abs(entry_price - stop_price)

    if risk_points <= 0:
        return result

    # Compute target price
    if break_dir == "long":
        target_price = entry_price + risk_points * rr_target
    else:
        target_price = entry_price - risk_points * rr_target

    result["entry_ts"] = entry_ts
    result["entry_price"] = entry_price
    result["stop_price"] = stop_price
    result["target_price"] = target_price

    # Scan bars forward from entry to determine outcome
    post_entry = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(entry_ts))
        & (bars_df["ts_utc"] < pd.Timestamp(trading_day_end))
    ].sort_values("ts_utc")

    if post_entry.empty:
        result["outcome"] = "scratch"
        result["mae_r"] = round(
            pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4
        )
        result["mfe_r"] = round(
            pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4
        )
        return result

    highs = post_entry["high"].values
    lows = post_entry["low"].values

    if break_dir == "long":
        hit_target = highs >= target_price
        hit_stop = lows <= stop_price
        favorable = highs - entry_price
        adverse = entry_price - lows
    else:
        hit_target = lows <= target_price
        hit_stop = highs >= stop_price
        favorable = entry_price - lows
        adverse = highs - entry_price

    any_hit = hit_target | hit_stop

    if not any_hit.any():
        # No target or stop hit — scratch
        result["outcome"] = "scratch"
        max_favorable_points = float(np.max(favorable))
        max_adverse_points = float(np.max(adverse))
    else:
        first_hit_idx = int(np.argmax(any_hit))
        # Use .iloc to preserve tz-aware timestamp (not .values which strips tz)
        exit_ts_val = post_entry.iloc[first_hit_idx]["ts_utc"].to_pydatetime()

        if hit_target[first_hit_idx] and hit_stop[first_hit_idx]:
            # Ambiguous bar — conservative: assume loss
            result["outcome"] = "loss"
            result["exit_ts"] = exit_ts_val
            result["exit_price"] = stop_price
            result["pnl_r"] = -1.0
        elif hit_target[first_hit_idx]:
            result["outcome"] = "win"
            result["exit_ts"] = exit_ts_val
            result["exit_price"] = target_price
            result["pnl_r"] = round(
                pnl_points_to_r(cost_spec, entry_price, stop_price,
                                risk_points * rr_target),
                4,
            )
        else:
            result["outcome"] = "loss"
            result["exit_ts"] = exit_ts_val
            result["exit_price"] = stop_price
            result["pnl_r"] = -1.0

        # MAE/MFE up to and including the exit bar
        max_favorable_points = float(np.max(favorable[: first_hit_idx + 1]))
        max_adverse_points = float(np.max(adverse[: first_hit_idx + 1]))

    # MAE/MFE in R
    result["mae_r"] = round(
        pnl_points_to_r(cost_spec, entry_price, stop_price, max_adverse_points), 4
    )
    result["mfe_r"] = round(
        pnl_points_to_r(cost_spec, entry_price, stop_price, max_favorable_points), 4
    )

    return result


def build_outcomes(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes: int = 5,
    dry_run: bool = False,
) -> int:
    """
    Build orb_outcomes for all RR targets x confirm_bars.

    Returns count of rows written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path))
    try:
        # Ensure trading_app tables exist
        if not dry_run:
            init_trading_app_schema(db_path=db_path)

        # Build date filter
        date_clauses = []
        params = [instrument, orb_minutes]
        if start_date:
            date_clauses.append("AND trading_day >= ?")
            params.append(start_date)
        if end_date:
            date_clauses.append("AND trading_day <= ?")
            params.append(end_date)
        date_filter = " ".join(date_clauses)

        # Fetch all daily_features rows
        query = f"""
            SELECT trading_day, symbol, orb_minutes,
                   {', '.join(
                       f'orb_{lbl}_high, orb_{lbl}_low, orb_{lbl}_break_dir, orb_{lbl}_break_ts'
                       for lbl in ORB_LABELS
                   )}
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = ?
            {date_filter}
            ORDER BY trading_day
        """
        rows = con.execute(query, params).fetchall()
        col_names = [desc[0] for desc in con.description]

        total_written = 0
        total_days = len(rows)
        t0 = time.monotonic()

        for day_idx, row in enumerate(rows):
            row_dict = dict(zip(col_names, row))
            trading_day = row_dict["trading_day"]
            symbol = row_dict["symbol"]

            # Get bars_1m for this trading day
            td_start, td_end = compute_trading_day_utc_range(trading_day)
            bars_query = """
                SELECT ts_utc, open, high, low, close, volume
                FROM bars_1m
                WHERE symbol = ?
                AND ts_utc >= ?::TIMESTAMPTZ
                AND ts_utc < ?::TIMESTAMPTZ
                ORDER BY ts_utc ASC
            """
            bars_df = con.execute(
                bars_query, [symbol, td_start.isoformat(), td_end.isoformat()]
            ).fetchdf()

            if bars_df.empty:
                continue

            day_batch = []

            for orb_label in ORB_LABELS:
                break_dir = row_dict.get(f"orb_{orb_label}_break_dir")
                break_ts = row_dict.get(f"orb_{orb_label}_break_ts")
                orb_high = row_dict.get(f"orb_{orb_label}_high")
                orb_low = row_dict.get(f"orb_{orb_label}_low")

                if break_dir is None or break_ts is None:
                    continue
                if orb_high is None or orb_low is None:
                    continue

                for rr_target in RR_TARGETS:
                    for cb in CONFIRM_BARS_OPTIONS:
                        outcome = compute_single_outcome(
                            bars_df=bars_df,
                            break_ts=break_ts,
                            orb_high=orb_high,
                            orb_low=orb_low,
                            break_dir=break_dir,
                            rr_target=rr_target,
                            confirm_bars=cb,
                            trading_day_end=td_end,
                            cost_spec=cost_spec,
                        )

                        day_batch.append([
                            trading_day, symbol, orb_label, orb_minutes,
                            rr_target, cb,
                            outcome["entry_ts"], outcome["entry_price"],
                            outcome["stop_price"], outcome["target_price"],
                            outcome["outcome"], outcome["exit_ts"],
                            outcome["exit_price"], outcome["pnl_r"],
                            outcome["mae_r"], outcome["mfe_r"],
                        ])

                        total_written += 1

            # Batch insert all outcomes for this trading day
            if day_batch and not dry_run:
                con.executemany(
                    """
                    INSERT OR REPLACE INTO orb_outcomes
                    (trading_day, symbol, orb_label, orb_minutes,
                     rr_target, confirm_bars,
                     entry_ts, entry_price, stop_price, target_price,
                     outcome, exit_ts, exit_price, pnl_r,
                     mae_r, mfe_r)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    day_batch,
                )

            if (day_idx + 1) % 50 == 0:
                if not dry_run:
                    con.commit()
                elapsed = time.monotonic() - t0
                rate = (day_idx + 1) / elapsed
                remaining = (total_days - day_idx - 1) / rate if rate > 0 else 0
                print(
                    f"  {day_idx + 1}/{total_days} days "
                    f"({total_written} outcomes, "
                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )

        if not dry_run:
            con.commit()

        elapsed = time.monotonic() - t0
        print(f"Done: {total_written} outcomes for {total_days} trading days in {elapsed:.1f}s")
        if dry_run:
            print("  (DRY RUN — no data written)")

        return total_written

    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compute ORB outcomes for all RR targets x confirm_bars"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB duration in minutes")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    build_outcomes(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        orb_minutes=args.orb_minutes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
