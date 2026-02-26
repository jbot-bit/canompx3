#!/usr/bin/env python3
"""
Prospective tracker for prior-day outcome signals.

Tracks qualifying days for frozen hypotheses, accumulating prospective
evidence toward confirmation thresholds (N=100 re-evaluation, N=150 deployment).

Usage:
    python scripts/tools/prospective_tracker.py
    python scripts/tools/prospective_tracker.py --freeze-date 2026-02-22
"""
import argparse
import datetime
import logging

import duckdb
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------
DEFAULT_FREEZE_DATE = datetime.date(2026, 2, 22)

SIGNALS = {
    "MGC_CME_REOPEN_PREV_LOSS": {
        "symbol": "MGC",
        "session": "CME_REOPEN",
        "orb_label": "CME_REOPEN",
        "prev_outcome_filter": "loss",
        "entry_model": "E2",
        "confirm_bars": 1,
        "rr_target": 2.0,
        "min_orb_pts": 4.0,
    },
}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def fetch_qualifying_days(con, sig: dict) -> list[tuple]:
    """
    Return qualifying days for a prior-day outcome signal.

    Uses LAG() on daily_features (where orb_{session}_outcome lives),
    then joins orb_outcomes for the realized pnl_r.
    Mirrors research/research_prev_day_signal.py query pattern.
    """
    label = sig["orb_label"]
    outcome_col = f"orb_{label}_outcome"
    size_col = f"orb_{label}_size"

    sql = f"""
        WITH lag_feats AS (
            SELECT
                trading_day,
                symbol,
                {outcome_col}  AS curr_outcome,
                LAG({outcome_col}) OVER (
                    PARTITION BY symbol ORDER BY trading_day
                )              AS prev_outcome,
                {size_col}     AS orb_size_pts
            FROM daily_features
            WHERE orb_minutes = 5
              AND symbol = ?
        )
        SELECT
            l.trading_day,
            l.prev_outcome,
            l.orb_size_pts,
            o.outcome,
            o.pnl_r
        FROM lag_feats l
        JOIN orb_outcomes o
          ON  l.trading_day  = o.trading_day
          AND l.symbol       = o.symbol
          AND o.orb_label    = ?
          AND o.orb_minutes  = 5
          AND o.entry_model  = ?
          AND o.rr_target    = ?
          AND o.confirm_bars = ?
          AND o.outcome IS NOT NULL
          AND o.pnl_r   IS NOT NULL
        WHERE l.prev_outcome = ?
          AND l.orb_size_pts IS NOT NULL
          AND l.orb_size_pts >= ?
        ORDER BY l.trading_day
    """
    return con.execute(sql, [
        sig["symbol"],
        sig["orb_label"],
        sig["entry_model"],
        sig["rr_target"],
        sig["confirm_bars"],
        sig["prev_outcome_filter"],
        sig["min_orb_pts"],
    ]).fetchall()


# ---------------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------------
def populate_signal(con, signal_id: str, sig: dict, freeze_date: datetime.date):
    """DELETE+INSERT all qualifying days for this signal."""
    rows = fetch_qualifying_days(con, sig)
    logger.info(f"{signal_id}: {len(rows)} qualifying days found")

    con.execute(
        "DELETE FROM prospective_signals WHERE signal_id = ?",
        [signal_id],
    )

    for trading_day, prev_outcome, orb_size, outcome, pnl_r in rows:
        # trading_day from DuckDB may be datetime.date or datetime.datetime
        if hasattr(trading_day, 'date'):
            td = trading_day.date()
        else:
            td = trading_day
        is_prospective = td >= freeze_date
        con.execute("""
            INSERT INTO prospective_signals
                (signal_id, trading_day, symbol, session,
                 prev_day_outcome, orb_size, entry_model,
                 confirm_bars, rr_target, outcome, pnl_r,
                 is_prospective, freeze_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            signal_id,
            trading_day,
            sig["symbol"],
            sig["session"],
            prev_outcome,
            orb_size,
            sig["entry_model"],
            sig["confirm_bars"],
            sig["rr_target"],
            outcome,
            pnl_r,
            is_prospective,
            freeze_date,
        ])

    con.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def compute_stats(pnl_values: list[float]) -> dict:
    """Compute N, avgR, WR, t-stat, p-value from a list of pnl_r values."""
    n = len(pnl_values)
    if n == 0:
        return {"N": 0, "avgR": 0.0, "WR": 0.0, "t": 0.0, "p": 1.0}

    avg_r = sum(pnl_values) / n
    wr = sum(1 for x in pnl_values if x > 0) / n * 100

    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(pnl_values, 0.0)
    else:
        t_stat, p_val = 0.0, 1.0

    return {"N": n, "avgR": avg_r, "WR": wr, "t": t_stat, "p": p_val}


def compute_yearly_stats(rows: list[tuple]) -> dict:
    """Group by year and compute stats per year."""
    yearly = {}
    for trading_day, pnl_r in rows:
        yr = trading_day.year if hasattr(trading_day, 'year') else int(str(trading_day)[:4])
        yearly.setdefault(yr, []).append(pnl_r)
    return {yr: compute_stats(vals) for yr, vals in sorted(yearly.items())}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(con, signal_id: str, sig: dict, freeze_date: datetime.date):
    """Print console report for a signal."""
    retro_rows = con.execute("""
        SELECT trading_day, pnl_r FROM prospective_signals
        WHERE signal_id = ? AND is_prospective = FALSE
        ORDER BY trading_day
    """, [signal_id]).fetchall()

    prosp_rows = con.execute("""
        SELECT trading_day, pnl_r FROM prospective_signals
        WHERE signal_id = ? AND is_prospective = TRUE
        ORDER BY trading_day
    """, [signal_id]).fetchall()

    retro_pnl = [r[1] for r in retro_rows]
    prosp_pnl = [r[1] for r in prosp_rows]
    combined_pnl = retro_pnl + prosp_pnl

    retro_s = compute_stats(retro_pnl)
    prosp_s = compute_stats(prosp_pnl)
    combined_s = compute_stats(combined_pnl)

    label = sig["orb_label"]
    em = sig["entry_model"]
    cb = sig["confirm_bars"]
    rr = sig["rr_target"]
    prev = sig["prev_outcome_filter"]
    g = sig["min_orb_pts"]

    print()
    print(f"=== Prospective Signal Tracker: {signal_id} ===")
    print(f"Signal: MGC {label} {em} CB{cb} RR{rr} G{int(g)}+ | prev_day = {prev}")
    print(f"Freeze date: {freeze_date}")
    print()

    def fmt_stats(s):
        return f"  N={s['N']}  avgR={s['avgR']:+.3f}  WR={s['WR']:.1f}%  t={s['t']:+.2f}  p={s['p']:.4f}"

    print("--- RETROSPECTIVE (before freeze) ---")
    print(fmt_stats(retro_s))
    print()
    print("--- PROSPECTIVE (after freeze) ---")
    if prosp_s["N"] == 0:
        print("  No prospective data yet (freeze date is today or in the future)")
    else:
        print(fmt_stats(prosp_s))
    print()
    print("--- COMBINED ---")
    print(fmt_stats(combined_s))
    print()

    # Progress bar
    prosp_n = prosp_s["N"]
    target = 100
    pct = min(prosp_n / target * 100, 100)
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "=" * filled + "." * (bar_len - filled)
    print("--- PROGRESS ---")
    print(f"  Prospective N: {prosp_n:3d} / {target}  [{bar}] {pct:.0f}%")
    print(f"  Next milestone:  N=100 -> formal re-evaluation")
    print(f"  Final milestone: N=150 -> full validation pipeline")

    if prosp_n >= 150:
        print()
        print("  *** THRESHOLD REACHED: N=150 prospective ***")
        print("  *** ACTION: Run full validation pipeline  ***")
        print("  *** Consider 1.5x position-size overlay   ***")
    elif prosp_n >= 100:
        print()
        print("  *** THRESHOLD REACHED: N=100 prospective ***")
        print("  *** ACTION: Formal re-evaluation required ***")

    # Year-by-year (prospective only)
    if prosp_rows:
        print()
        print("--- YEAR-BY-YEAR (prospective only) ---")
        yearly = compute_yearly_stats(prosp_rows)
        for yr, s in yearly.items():
            print(f"  {yr}:  N={s['N']}  avgR={s['avgR']:+.3f}  WR={s['WR']:.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prospective tracker for prior-day outcome signals"
    )
    parser.add_argument(
        "--freeze-date",
        type=lambda s: datetime.date.fromisoformat(s),
        default=DEFAULT_FREEZE_DATE,
        help="Date from which tracking is prospective (default: 2026-02-22)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(GOLD_DB_PATH),
        help="Path to DuckDB database",
    )
    args = parser.parse_args()

    con = duckdb.connect(args.db_path)

    for signal_id, sig in SIGNALS.items():
        populate_signal(con, signal_id, sig, args.freeze_date)
        print_report(con, signal_id, sig, args.freeze_date)

    con.close()


if __name__ == "__main__":
    main()
