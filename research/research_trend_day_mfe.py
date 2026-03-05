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
            # TODO: For each outcome row, fetch bars_1m from entry_ts to
            #       session end, compute uncapped MFE in R-multiples,
            #       compare to capped_mfe_r.
            pass

    finally:
        con.close()

    if args.dry_run:
        print("Dry run complete.")
        return

    print("Done.")


if __name__ == "__main__":
    main()
