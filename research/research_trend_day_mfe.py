"""research/research_trend_day_mfe.py

Phase 1 Research: Unicorn/Trend-Day MFE Discovery

Computes TRUE session MFE from bars_1m (uncapped by target/stop),
quantifies the gap vs stored mfe_r, and tests predictors of outsized moves.

@research-source: docs/plans/2026-03-06-trend-day-mfe-impl.md
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS, CostSpec, pnl_points_to_r
from pipeline.paths import GOLD_DB_PATH
from pipeline.dst import SESSION_CATALOG
from pipeline.build_daily_features import compute_trading_day_utc_range
from research.lib.stats import bh_fdr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIVE_INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)
SESSIONS = sorted(SESSION_CATALOG.keys())
ORB_MINUTES = [5, 15, 30]
RR_TARGETS = [1.5, 2.0, 2.5, 3.0]
DB_PATH = GOLD_DB_PATH
MIN_TRADES = 30


# ---------------------------------------------------------------------------
# TRUE MFE computation
# ---------------------------------------------------------------------------
def compute_true_session_mfe(
    bars_1m: pd.DataFrame,  # pre-filtered to entry_ts < ts_utc <= trading_day_end
    entry_price: float,
    stop_price: float,
    break_dir: int,  # 1=long, -1=short
    cost_spec: CostSpec,
) -> dict:
    """Compute uncapped MFE/MAE from entry to session end using 1m bars."""
    if bars_1m.empty:
        return {
            "true_mfe_r": None,
            "true_mae_r": None,
            "session_close_r": None,
            "time_to_mfe_min": None,
            "bars_after_entry": 0,
        }

    highs = bars_1m["high"].values
    lows = bars_1m["low"].values

    if break_dir == 1:  # long
        favorable_pts = highs - entry_price
        adverse_pts = entry_price - lows
    else:  # short
        favorable_pts = entry_price - lows
        adverse_pts = highs - entry_price

    max_fav = max(float(np.max(favorable_pts)), 0.0)
    max_adv = max(float(np.max(adverse_pts)), 0.0)

    true_mfe_r = pnl_points_to_r(cost_spec, entry_price, stop_price, max_fav)
    true_mae_r = pnl_points_to_r(cost_spec, entry_price, stop_price, max_adv)

    # Session close R (signed)
    last_close = float(bars_1m["close"].iloc[-1])
    if break_dir == 1:
        close_pts = last_close - entry_price
    else:
        close_pts = entry_price - last_close
    session_close_r = pnl_points_to_r(cost_spec, entry_price, stop_price, close_pts)

    mfe_bar_idx = int(np.argmax(favorable_pts))
    time_to_mfe_min = mfe_bar_idx  # bars are 1-minute
    bars_after_entry = len(bars_1m)

    return {
        "true_mfe_r": round(true_mfe_r, 4),
        "true_mae_r": round(true_mae_r, 4),
        "session_close_r": round(session_close_r, 4),
        "time_to_mfe_min": time_to_mfe_min,
        "bars_after_entry": bars_after_entry,
    }


def load_bars_1m_for_day(
    con: duckdb.DuckDBPyConnection, instrument: str, trading_day
) -> pd.DataFrame:
    """Load all 1m bars for a trading day's UTC range."""
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)
    sql = """
        SELECT ts_utc, high, low, close
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
    """
    return con.execute(sql, [instrument, start_utc, end_utc]).fetchdf()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_outcomes(con: duckdb.DuckDBPyConnection, instrument: str,
                  limit: int | None = None) -> pd.DataFrame:
    """Load E1/E2 outcomes for a single instrument.

    Returns DataFrame with entry details needed for TRUE MFE computation.
    """
    sql = """
        SELECT o.trading_day, o.symbol, o.orb_label, o.orb_minutes,
               o.entry_model, o.rr_target, o.confirm_bars,
               o.entry_ts, o.entry_price, o.stop_price, o.target_price,
               o.pnl_r, o.mfe_r AS capped_mfe_r, o.outcome,
               o.exit_ts,
               CASE WHEN o.entry_price > o.stop_price THEN 1 ELSE -1 END AS break_dir
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.entry_model IN ('E1', 'E2')
          AND o.outcome IS NOT NULL
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day, o.orb_label
    """
    if limit is not None:
        sql += f"\n        LIMIT {int(limit)}"

    return con.execute(sql, [instrument]).fetchdf()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend-Day MFE Discovery — TRUE session MFE from bars_1m"
    )
    parser.add_argument(
        "--instrument", type=str, default=None,
        help="Single instrument (default: all active)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print row counts per session and exit",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows per instrument (for testing)",
    )
    parser.add_argument(
        "--predictors", action="store_true",
        help="Enable predictor analysis (Task 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="research/output/",
        help="Output directory for results",
    )
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ACTIVE_INSTRUMENTS

    print(f"=== Trend-Day MFE Research ===")
    print(f"DB: {DB_PATH}")
    print(f"Instruments: {instruments}")
    print()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        for instrument in instruments:
            t0 = time.time()
            df = load_outcomes(con, instrument, limit=args.limit)
            elapsed = time.time() - t0
            print(f"--- {instrument}: {len(df):,} outcomes loaded ({elapsed:.1f}s) ---")

            # Row counts per session
            if len(df) > 0:
                session_counts = (
                    df.groupby("orb_label")
                    .size()
                    .sort_values(ascending=False)
                )
                for session, count in session_counts.items():
                    print(f"  {session:25s} {count:>7,}")
            print()

            if args.dry_run:
                continue

            # ----- Task 2: TRUE MFE computation from bars_1m -----
            if len(df) == 0:
                print(f"  No outcomes for {instrument}, skipping.\n")
                continue

            cost_spec = COST_SPECS[instrument]

            # Group by trading_day for efficient bar loading
            grouped = df.groupby("trading_day")
            n_days = len(grouped)
            print(f"  Computing TRUE MFE: {len(df):,} outcomes across {n_days:,} trading days...")

            t_mfe = time.time()
            true_mfe_results = []
            rows_processed = 0

            for day_idx, (td, day_df) in enumerate(grouped):
                # Load bars once per trading day
                day_bars = load_bars_1m_for_day(con, instrument, td)
                _, td_end = compute_trading_day_utc_range(td)
                td_end_ts = pd.Timestamp(td_end)

                # Process all outcomes for this day
                for row_idx, row in day_df.iterrows():
                    entry_ts = pd.Timestamp(row["entry_ts"])

                    # Filter to post-entry bars up to session end
                    post_entry = day_bars[
                        (day_bars["ts_utc"] > entry_ts) & (day_bars["ts_utc"] <= td_end_ts)
                    ]

                    result = compute_true_session_mfe(
                        bars_1m=post_entry,
                        entry_price=row["entry_price"],
                        stop_price=row["stop_price"],
                        break_dir=row["break_dir"],
                        cost_spec=cost_spec,
                    )
                    result["idx"] = row_idx
                    true_mfe_results.append(result)

                rows_processed += len(day_df)
                if (day_idx + 1) % 200 == 0 or (day_idx + 1) == n_days:
                    elapsed_mfe = time.time() - t_mfe
                    pct = rows_processed / len(df) * 100
                    print(
                        f"    [{day_idx + 1:,}/{n_days:,} days] "
                        f"{rows_processed:,}/{len(df):,} rows ({pct:.0f}%) "
                        f"— {elapsed_mfe:.1f}s"
                    )

            # Merge results back
            mfe_df = pd.DataFrame(true_mfe_results).set_index("idx")
            for col in mfe_df.columns:
                df[col] = mfe_df[col]

            elapsed_mfe = time.time() - t_mfe
            print(f"  TRUE MFE complete: {elapsed_mfe:.1f}s total\n")

            # Verification: compare capped vs true MFE for first 10 rows
            valid = df.dropna(subset=["true_mfe_r", "capped_mfe_r"]).head(10)
            if len(valid) > 0:
                print("  Verification (first 10 rows): capped_mfe_r vs true_mfe_r")
                print(f"  {'capped':>10s}  {'true':>10s}  {'gap':>10s}  {'ok':>4s}")
                for _, r in valid.iterrows():
                    gap = r["true_mfe_r"] - r["capped_mfe_r"]
                    ok = "YES" if r["true_mfe_r"] >= r["capped_mfe_r"] - 0.001 else "NO"
                    print(
                        f"  {r['capped_mfe_r']:10.4f}  {r['true_mfe_r']:10.4f}  "
                        f"{gap:10.4f}  {ok:>4s}"
                    )

                # Aggregate check
                check_df = df.dropna(subset=["true_mfe_r", "capped_mfe_r"])
                n_valid = len(check_df)
                n_ok = int(
                    (check_df["true_mfe_r"] >= check_df["capped_mfe_r"] - 0.001).sum()
                )
                print(f"\n  Aggregate: {n_ok:,}/{n_valid:,} rows have true_mfe_r >= capped_mfe_r")
                print()

    finally:
        con.close()

    if args.dry_run:
        print("Dry run complete.")
        return

    print("Done.")


if __name__ == "__main__":
    main()
