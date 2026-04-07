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
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb
import numpy as np
import pandas as pd

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, pnl_points_to_r, to_r_multiple
from pipeline.init_db import ORB_LABELS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ENTRY_MODELS
from trading_app.entry_rules import (
    _verify_e3_sub_bar_fill,
    detect_confirm,
    resample_to_5m,
    resolve_entry,
)
from trading_app.nested.schema import init_nested_schema
from trading_app.outcome_builder import CONFIRM_BARS_OPTIONS, RR_TARGETS

# Entry resolution for nested ORB: always 5-minute bars
ENTRY_RESOLUTION = 5

# resample_to_5m and _verify_e3_sub_bar_fill were extracted to
# trading_app/entry_rules.py during the E2 canonical-window refactor
# (2026-04-07, Stage 4). They are imported above so the rest of this
# module (which calls them in build_nested_outcomes) still works
# unchanged. This file is slated for deletion in Stage 7.


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
      2. Bulk-load bars_1m for the full date range (single query)
      3. Partition bars by trading day boundaries in Python
      4. Resample post-ORB bars to 5m
      5. For each (ORB x RR x CB x EM): compute outcome with 5m bars
      6. For E3: verify sub-bar fill against 1m data
      7. INSERT into nested_outcomes

    With --resume, skips days already in nested_outcomes (set-based gap detection).

    Returns count of rows written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if orb_minutes_list is None:
        orb_minutes_list = [15, 30]

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path))
    try:
        from pipeline.db_config import configure_connection

        configure_connection(con, writing=True)

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
                       {
                ", ".join(
                    f"orb_{lbl}_high, orb_{lbl}_low, orb_{lbl}_break_dir, orb_{lbl}_break_ts" for lbl in ORB_LABELS
                )
            }
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = ?
                {date_filter}
                ORDER BY trading_day
            """
            rows = con.execute(query, params).fetchall()
            col_names = [desc[0] for desc in con.description]

            total_days = len(rows)
            if total_days == 0:
                print(
                    f"  No daily_features rows for orb_minutes={orb_minutes}. "
                    "Run: python pipeline/build_daily_features.py --orb-minutes {orb_minutes}"
                )
                continue

            # Resume: skip days already completed (set-based gap detection)
            if resume and not dry_run:
                completed_days = {
                    r[0]
                    for r in con.execute(
                        """SELECT DISTINCT trading_day FROM nested_outcomes
                           WHERE symbol = ? AND orb_minutes = ?""",
                        [instrument, orb_minutes],
                    ).fetchall()
                }
                if completed_days:
                    original_count = total_days
                    rows = [
                        r for r in rows if dict(zip(col_names, r, strict=False))["trading_day"] not in completed_days
                    ]
                    total_days = len(rows)
                    print(
                        f"  {original_count} total trading days, "
                        f"{len(completed_days)} already completed, "
                        f"{total_days} remaining"
                    )
                    if total_days == 0:
                        print(f"  All days already completed for orb_minutes={orb_minutes}")
                        continue
                else:
                    print(f"  {total_days} trading days loaded (fresh start)")
            else:
                print(f"  {total_days} trading days loaded")

            # --- Bulk-load bars_1m for the full date range ---
            trading_days = [dict(zip(col_names, r, strict=False))["trading_day"] for r in rows]
            first_td_start, _ = compute_trading_day_utc_range(trading_days[0])
            _, last_td_end = compute_trading_day_utc_range(trading_days[-1])

            print(f"  Bulk-loading bars_1m ({first_td_start.date()} to {last_td_end.date()})...")
            bulk_t0 = time.monotonic()
            all_bars_1m = con.execute(
                """SELECT ts_utc, open, high, low, close, volume
                   FROM bars_1m
                   WHERE symbol = ?
                   AND ts_utc >= ?::TIMESTAMPTZ
                   AND ts_utc < ?::TIMESTAMPTZ
                   ORDER BY ts_utc ASC""",
                [instrument, first_td_start.isoformat(), last_td_end.isoformat()],
            ).fetchdf()
            if not all_bars_1m.empty:
                all_bars_1m["ts_utc"] = pd.to_datetime(all_bars_1m["ts_utc"], utc=True)
            print(f"  Loaded {len(all_bars_1m)} bars in {time.monotonic() - bulk_t0:.1f}s")

            # Pre-compute trading day boundaries using searchsorted for O(log n) partitioning
            ts_values = all_bars_1m["ts_utc"].values  # numpy datetime64 array (sorted)

            for day_idx, row in enumerate(rows):
                row_dict = dict(zip(col_names, row, strict=False))
                trading_day = row_dict["trading_day"]
                symbol = row_dict["symbol"]

                # Partition bars for this trading day using searchsorted
                td_start, td_end = compute_trading_day_utc_range(trading_day)
                td_start_np = pd.Timestamp(td_start).to_numpy()
                td_end_np = pd.Timestamp(td_end).to_numpy()
                i_start = ts_values.searchsorted(td_start_np, side="left")
                i_end = ts_values.searchsorted(td_end_np, side="left")
                bars_1m_df = all_bars_1m.iloc[i_start:i_end]

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

                    # OPTIMIZATION: cache detect_confirm per (ORB, CB)
                    # detect_confirm does NOT depend on RR or EM
                    confirm_cache = {}
                    for cb in CONFIRM_BARS_OPTIONS:
                        confirm_cache[cb] = detect_confirm(
                            bars_5m_df,
                            break_ts,
                            orb_high,
                            orb_low,
                            break_dir,
                            cb,
                            td_end,
                        )

                    # OPTIMIZATION: cache resolve_entry per (ORB, CB, EM)
                    # resolve_entry does NOT depend on RR
                    entry_cache = {}
                    for cb in CONFIRM_BARS_OPTIONS:
                        for em in ENTRY_MODELS:
                            if em == "E2":
                                # E2: stop-market uses break-touch, not confirm
                                if cb > 1:
                                    continue  # E2 always CB1
                                from trading_app.config import E2_SLIPPAGE_TICKS
                                from trading_app.entry_rules import _resolve_e2, detect_break_touch

                                touch = detect_break_touch(
                                    bars_5m_df,
                                    orb_high=orb_high,
                                    orb_low=orb_low,
                                    break_dir=break_dir,
                                    detection_window_start=break_ts,
                                    detection_window_end=td_end,
                                )
                                signal = _resolve_e2(
                                    touch, slippage_ticks=E2_SLIPPAGE_TICKS, tick_size=cost_spec.tick_size
                                )
                                entry_cache[(cb, em)] = signal
                                continue

                            if em == "E3" and cb > 1:
                                continue
                            signal = resolve_entry(
                                bars_5m_df,
                                confirm_cache[cb],
                                em,
                                td_end,
                            )

                            # E3 sub-bar fill verification
                            if em == "E3" and signal.triggered and signal.entry_ts is not None:
                                fill_ok = _verify_e3_sub_bar_fill(
                                    bars_1m_df,
                                    signal.entry_ts,
                                    signal.entry_price,
                                    break_dir,
                                )
                                if not fill_ok:
                                    signal = None  # Mark as invalid

                            entry_cache[(cb, em)] = signal

                    # OPTIMIZATION: for each (CB, EM) entry, compute all RR targets
                    # sharing the same post-entry bars
                    for cb in CONFIRM_BARS_OPTIONS:
                        for em in ENTRY_MODELS:
                            if em in ("E2", "E3") and cb > 1:
                                continue
                            signal = entry_cache[(cb, em)]

                            # No entry (not triggered or E3 fill failed)
                            if signal is None or not signal.triggered:
                                for rr_target in RR_TARGETS:
                                    day_batch.append(
                                        [
                                            trading_day,
                                            symbol,
                                            orb_label,
                                            orb_minutes,
                                            ENTRY_RESOLUTION,
                                            rr_target,
                                            cb,
                                            em,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                        ]
                                    )
                                    total_written += 1
                                continue

                            entry_price = signal.entry_price
                            stop_price = signal.stop_price
                            entry_ts = signal.entry_ts
                            risk_points = abs(entry_price - stop_price)

                            if risk_points <= 0:
                                for rr_target in RR_TARGETS:
                                    day_batch.append(
                                        [
                                            trading_day,
                                            symbol,
                                            orb_label,
                                            orb_minutes,
                                            ENTRY_RESOLUTION,
                                            rr_target,
                                            cb,
                                            em,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                        ]
                                    )
                                    total_written += 1
                                continue

                            # Get post-entry bars ONCE for all RR targets
                            post_entry = bars_5m_df[
                                (bars_5m_df["ts_utc"] > pd.Timestamp(entry_ts))
                                & (bars_5m_df["ts_utc"] < pd.Timestamp(td_end))
                            ].sort_values("ts_utc")

                            if post_entry.empty:
                                # Scratch for all RR targets
                                mae_r = round(pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4)
                                mfe_r = mae_r
                                for rr_target in RR_TARGETS:
                                    target_price = (
                                        entry_price + risk_points * rr_target
                                        if break_dir == "long"
                                        else entry_price - risk_points * rr_target
                                    )
                                    day_batch.append(
                                        [
                                            trading_day,
                                            symbol,
                                            orb_label,
                                            orb_minutes,
                                            ENTRY_RESOLUTION,
                                            rr_target,
                                            cb,
                                            em,
                                            entry_ts,
                                            entry_price,
                                            stop_price,
                                            target_price,
                                            "scratch",
                                            None,
                                            None,
                                            None,
                                            mae_r,
                                            mfe_r,
                                        ]
                                    )
                                    total_written += 1
                                continue

                            # Pre-extract arrays ONCE
                            highs = post_entry["high"].values
                            lows = post_entry["low"].values
                            if break_dir == "long":
                                hit_stop = lows <= stop_price
                                favorable = highs - entry_price
                                adverse = entry_price - lows
                            else:
                                hit_stop = highs >= stop_price
                                favorable = entry_price - lows
                                adverse = highs - entry_price

                            # Compute each RR target using shared arrays
                            for rr_target in RR_TARGETS:
                                if break_dir == "long":
                                    target_price = entry_price + risk_points * rr_target
                                    hit_target = highs >= target_price
                                else:
                                    target_price = entry_price - risk_points * rr_target
                                    hit_target = lows <= target_price

                                any_hit = hit_target | hit_stop

                                if not any_hit.any():
                                    outcome_str = "scratch"
                                    max_fav = float(np.max(favorable))
                                    max_adv = float(np.max(adverse))
                                    mae_r = round(
                                        pnl_points_to_r(cost_spec, entry_price, stop_price, max(max_adv, 0.0)), 4
                                    )
                                    mfe_r = round(
                                        pnl_points_to_r(cost_spec, entry_price, stop_price, max(max_fav, 0.0)), 4
                                    )
                                    day_batch.append(
                                        [
                                            trading_day,
                                            symbol,
                                            orb_label,
                                            orb_minutes,
                                            ENTRY_RESOLUTION,
                                            rr_target,
                                            cb,
                                            em,
                                            entry_ts,
                                            entry_price,
                                            stop_price,
                                            target_price,
                                            outcome_str,
                                            None,
                                            None,
                                            None,
                                            mae_r,
                                            mfe_r,
                                        ]
                                    )
                                else:
                                    first_hit_idx = int(np.argmax(any_hit))
                                    exit_ts_val = post_entry.iloc[first_hit_idx]["ts_utc"].to_pydatetime()
                                    max_fav = float(np.max(favorable[: first_hit_idx + 1]))
                                    max_adv = float(np.max(adverse[: first_hit_idx + 1]))
                                    mae_r = round(
                                        pnl_points_to_r(cost_spec, entry_price, stop_price, max(max_adv, 0.0)), 4
                                    )
                                    mfe_r = round(
                                        pnl_points_to_r(cost_spec, entry_price, stop_price, max(max_fav, 0.0)), 4
                                    )

                                    if hit_target[first_hit_idx] and hit_stop[first_hit_idx]:
                                        outcome_str = "loss"
                                        exit_price = stop_price
                                        pnl_r = -1.0
                                    elif hit_target[first_hit_idx]:
                                        outcome_str = "win"
                                        exit_price = target_price
                                        pnl_r = round(
                                            to_r_multiple(cost_spec, entry_price, stop_price, risk_points * rr_target),
                                            4,
                                        )
                                    else:
                                        outcome_str = "loss"
                                        exit_price = stop_price
                                        pnl_r = -1.0

                                    day_batch.append(
                                        [
                                            trading_day,
                                            symbol,
                                            orb_label,
                                            orb_minutes,
                                            ENTRY_RESOLUTION,
                                            rr_target,
                                            cb,
                                            em,
                                            entry_ts,
                                            entry_price,
                                            stop_price,
                                            target_price,
                                            outcome_str,
                                            exit_ts_val,
                                            exit_price,
                                            pnl_r,
                                            mae_r,
                                            mfe_r,
                                        ]
                                    )

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

                if (day_idx + 1) % 100 == 0:
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

    parser = argparse.ArgumentParser(description="Build nested ORB outcomes (wider range + 5m entry bars)")
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--orb-minutes",
        type=int,
        nargs="+",
        default=[15, 30],
        help="ORB duration(s) to build (default: 15 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last completed day (skip already-built days)"
    )
    parser.add_argument("--audit", action="store_true", help="Run spot-check audit after build completes")
    parser.add_argument("--audit-days", type=int, default=10, help="Number of random days to audit (default: 10)")
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
