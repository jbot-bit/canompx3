"""
Build nested ORB outcomes: wider ORB range (15/30 min) with 5-minute entry bars.

Resamples post-ORB 1m bars to 5m, then runs the same outcome computation
(compute_single_outcome) with 5m bars. For E3 entries, performs a sub-bar
fill verification against the underlying 1m data.

Usage:
    python -m trading_app.nested.builder --instrument MGC --orb-minutes 15 30
    python -m trading_app.nested.builder --instrument MGC --orb-minutes 15 --dry-run
"""

import sys
import time
from pathlib import Path
from datetime import date, datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from pipeline.init_db import ORB_LABELS
from pipeline.build_daily_features import compute_trading_day_utc_range
from trading_app.outcome_builder import compute_single_outcome, RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.config import ENTRY_MODELS
from trading_app.nested.schema import init_nested_schema

# Entry resolution for nested ORB: always 5-minute bars
ENTRY_RESOLUTION = 5


def resample_to_5m(bars_1m_df: pd.DataFrame, after_ts: datetime) -> pd.DataFrame:
    """Resample post-ORB 1m bars to 5m OHLCV.

    Takes 1m bars with ts_utc > after_ts, groups into 5-minute buckets
    (floor to nearest 5-minute boundary), and aggregates OHLCV.

    Args:
        bars_1m_df: DataFrame with columns [ts_utc, open, high, low, close, volume]
        after_ts: Only include bars strictly after this timestamp

    Returns:
        DataFrame with same columns, timestamps floored to 5m boundaries.
        Empty DataFrame if no bars after after_ts.
    """
    post_orb = bars_1m_df[
        bars_1m_df["ts_utc"] > pd.Timestamp(after_ts)
    ].copy()

    if post_orb.empty:
        return pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])

    # Floor to 5-minute boundaries (robust for any datetime64 resolution)
    post_orb["bucket"] = post_orb["ts_utc"].dt.floor("5min")

    # Aggregate per bucket
    grouped = post_orb.groupby("bucket", sort=True)
    bars_5m = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    bars_5m = bars_5m.rename(columns={"bucket": "ts_utc"})
    return bars_5m


def _verify_e3_sub_bar_fill(
    bars_1m_df: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    break_dir: str,
) -> bool:
    """Verify that 1m bars within the 5m entry candle actually touched the limit price.

    For E3 (limit-at-ORB), the 5m candle may show the price touched the ORB level,
    but the underlying 1m data might not actually reach it. This post-processing
    check ensures fill accuracy.

    Args:
        bars_1m_df: Original 1m bars
        entry_ts: The 5m bar timestamp where E3 fill was detected
        entry_price: The limit order price (ORB level)
        break_dir: "long" or "short"

    Returns:
        True if 1m data confirms the fill, False otherwise.
    """
    # Find 1m bars within the 5m candle: [entry_ts, entry_ts + 5min)
    bucket_start = pd.Timestamp(entry_ts)
    bucket_end = bucket_start + pd.Timedelta(minutes=5)

    bars_in_candle = bars_1m_df[
        (bars_1m_df["ts_utc"] >= bucket_start)
        & (bars_1m_df["ts_utc"] < bucket_end)
    ]

    if bars_in_candle.empty:
        return False

    if break_dir == "long":
        # Long E3: limit buy at orb_high. Need low <= entry_price
        return bool((bars_in_candle["low"].values <= entry_price).any())
    else:
        # Short E3: limit sell at orb_low. Need high >= entry_price
        return bool((bars_in_candle["high"].values >= entry_price).any())


def build_nested_outcomes(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes_list: list[int] | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> int:
    """Build nested_outcomes for all RR targets x confirm_bars using 5m entry bars.

    For each orb_minutes in orb_minutes_list:
      1. Query daily_features for ORB levels and break info
      2. Query bars_1m for each trading day
      3. Resample post-ORB bars to 5m
      4. For each (ORB x RR x CB x EM): compute outcome with 5m bars
      5. For E3: verify sub-bar fill against 1m data
      6. INSERT into nested_outcomes

    With --resume, skips days already completed (queries max trading_day per orb_minutes).

    Returns count of rows written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if orb_minutes_list is None:
        orb_minutes_list = [15, 30]

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path))
    try:
        if not dry_run:
            init_nested_schema(con=con)

        total_written = 0
        t0 = time.monotonic()

        for orb_minutes in orb_minutes_list:
            print(f"\n--- Building nested outcomes for orb_minutes={orb_minutes} ---")

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

            # Fetch all daily_features rows for this orb_minutes
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

            total_days = len(rows)
            if total_days == 0:
                print(f"  No daily_features rows for orb_minutes={orb_minutes}. "
                      "Run: python pipeline/build_daily_features.py --orb-minutes {orb_minutes}")
                continue

            # Resume: skip days already completed
            skip_before = None
            if resume and not dry_run:
                result = con.execute(
                    """SELECT MAX(trading_day) FROM nested_outcomes
                       WHERE symbol = ? AND orb_minutes = ?""",
                    [instrument, orb_minutes],
                ).fetchone()
                if result and result[0] is not None:
                    skip_before = result[0]
                    original_count = total_days
                    rows = [r for r in rows if dict(zip(col_names, r))["trading_day"] >= skip_before]
                    total_days = len(rows)
                    print(f"  {original_count} total trading days, resuming from {skip_before} (re-processes last day)")
                    print(f"  {total_days} days remaining")
                    if total_days == 0:
                        print(f"  All days already completed for orb_minutes={orb_minutes}")
                        continue
                else:
                    print(f"  {total_days} trading days loaded (fresh start)")
            else:
                print(f"  {total_days} trading days loaded")

            for day_idx, row in enumerate(rows):
                row_dict = dict(zip(col_names, row))
                trading_day = row_dict["trading_day"]
                symbol = row_dict["symbol"]

                # Get bars_1m for this trading day
                td_start, td_end = compute_trading_day_utc_range(trading_day)
                bars_1m_df = con.execute(
                    """SELECT ts_utc, open, high, low, close, volume
                       FROM bars_1m
                       WHERE symbol = ?
                       AND ts_utc >= ?::TIMESTAMPTZ
                       AND ts_utc < ?::TIMESTAMPTZ
                       ORDER BY ts_utc ASC""",
                    [symbol, td_start.isoformat(), td_end.isoformat()],
                ).fetchdf()
                if not bars_1m_df.empty:
                    bars_1m_df["ts_utc"] = pd.to_datetime(bars_1m_df["ts_utc"], utc=True)

                if bars_1m_df.empty:
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

                    # Resample post-ORB bars to 5m
                    bars_5m_df = resample_to_5m(bars_1m_df, break_ts)

                    if bars_5m_df.empty:
                        continue

                    for rr_target in RR_TARGETS:
                        for cb in CONFIRM_BARS_OPTIONS:
                            for em in ENTRY_MODELS:
                                outcome = compute_single_outcome(
                                    bars_df=bars_5m_df,
                                    break_ts=break_ts,
                                    orb_high=orb_high,
                                    orb_low=orb_low,
                                    break_dir=break_dir,
                                    rr_target=rr_target,
                                    confirm_bars=cb,
                                    trading_day_end=td_end,
                                    cost_spec=cost_spec,
                                    entry_model=em,
                                )

                                # E3 sub-bar fill verification
                                if (em == "E3"
                                        and outcome.get("entry_ts") is not None
                                        and outcome.get("outcome") is not None):
                                    fill_ok = _verify_e3_sub_bar_fill(
                                        bars_1m_df,
                                        outcome["entry_ts"],
                                        outcome["entry_price"],
                                        break_dir,
                                    )
                                    if not fill_ok:
                                        # Override to no-fill
                                        outcome = {
                                            "entry_ts": None, "entry_price": None,
                                            "stop_price": None, "target_price": None,
                                            "outcome": None, "exit_ts": None,
                                            "exit_price": None, "pnl_r": None,
                                            "mae_r": None, "mfe_r": None,
                                        }

                                day_batch.append([
                                    trading_day, symbol, orb_label, orb_minutes,
                                    ENTRY_RESOLUTION,
                                    rr_target, cb, em,
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
                        """INSERT OR REPLACE INTO nested_outcomes
                           (trading_day, symbol, orb_label, orb_minutes,
                            entry_resolution,
                            rr_target, confirm_bars, entry_model,
                            entry_ts, entry_price, stop_price, target_price,
                            outcome, exit_ts, exit_price, pnl_r,
                            mae_r, mfe_r)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        print(f"\nDone: {total_written} nested outcomes in {elapsed:.1f}s")
        if dry_run:
            print("  (DRY RUN -- no data written)")

        return total_written

    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build nested ORB outcomes (wider range + 5m entry bars)"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--orb-minutes", type=int, nargs="+", default=[15, 30],
        help="ORB duration(s) to build (default: 15 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed day (skip already-built days)")
    parser.add_argument("--audit", action="store_true",
                        help="Run spot-check audit after build completes")
    parser.add_argument("--audit-days", type=int, default=10,
                        help="Number of random days to audit (default: 10)")
    args = parser.parse_args()

    total = build_nested_outcomes(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        orb_minutes_list=args.orb_minutes,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    if args.audit and not args.dry_run and total > 0:
        print("\n--- Running post-build audit ---")
        from trading_app.nested.audit_outcomes import audit_nested_outcomes
        results = audit_nested_outcomes(
            instrument=args.instrument,
            n_days=args.audit_days,
            seed=42,
            orb_minutes_list=args.orb_minutes,
        )
        if results["n_mismatch"] > 0:
            print(f"\nWARNING: {results['n_mismatch']} audit mismatches found!")
            sys.exit(1)


if __name__ == "__main__":
    main()
