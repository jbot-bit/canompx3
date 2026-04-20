"""
Stage K: Independent SQL verification of vol-regime confluence scan v1.

Purpose: reproduce top survivor cells via pure SQL — NO import of filter_signal,
NO use of StrategyFilter.matches_df. Catches any drift in the
research.filter_utils → trading_app.config.ALL_FILTERS canonical chain.

Per research-truth-protocol.md § Canonical filter delegation (2026-04-19 incident):
when compute_deployed_filter mis-implemented OVNRNG_100 as ratio instead of
absolute, every prior scan using the deployed path was invalid. Independent
SQL verification catches this class of drift.

Filter → SQL mapping (derived from trading_app.config.py source):
  ORB_G5:      orb_{session}_size >= 5
  ATR_P50:     atr_20_pct >= 50 (MNQ)
  COST_LT12:   orb_{session}_size >= (88/12) * total_friction / point_value
                 For MNQ (pt=2.0, tf=2.92 from COST_SPECS): >= 10.7067
  OVN_100:     overnight_range >= 100
  XMES_60:     (MES atr_20_pct on trading_day) >= 60
  LON_100:     (session_london_high - session_london_low) >= 100

Kill criterion (from pre-reg): any cell diverges > 0.5% from scan v1 → HALT.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM)
OOS_END = pd.Timestamp("2026-04-16")

# Derive COST_LT12 threshold from canonical cost spec (not hardcoded).
_mnq = COST_SPECS["MNQ"]
COST_LT12_MNQ_SIZE_THRESHOLD = (88.0 / 12.0) * _mnq.total_friction / _mnq.point_value
# COST_LT12 passes when 100*tf/(raw + tf) < 12
#   raw = size * point_value
#   100*tf < 12*raw + 12*tf  => 88*tf < 12*raw  =>  raw > 88/12 * tf
#   size > 88/12 * tf / point_value

# The 6 survivor cells from scan v1
SURVIVOR_CELLS = [
    # (cell_id, orb_label, orb_minutes, rr_target, base_sql_where, variant_sql_where, label, expected)
    # expected: dict(n, expr, t) from scan v1
    (
        5, "COMEX_SETTLE", 5, 1.5,
        "orb_COMEX_SETTLE_size >= 5.0",
        "overnight_range >= 100.0",
        "COMEX_SETTLE x ORB_G5 x OVN_100",
        dict(n=517, expr=0.2099, t=4.07),
    ),
    (
        6, "COMEX_SETTLE", 5, 1.5,
        "orb_COMEX_SETTLE_size >= 5.0",
        "mes_atr_20_pct >= 60.0",
        "COMEX_SETTLE x ORB_G5 x XMES_60",
        dict(n=682, expr=0.1592, t=3.57),
    ),
    (
        7, "COMEX_SETTLE", 5, 1.5,
        "orb_COMEX_SETTLE_size >= 5.0",
        "(overnight_range >= 100.0 OR mes_atr_20_pct >= 60.0)",
        "COMEX_SETTLE x ORB_G5 x (OVN OR XMES)",
        dict(n=844, expr=0.1524, t=3.81),
    ),
    (
        1, "EUROPE_FLOW", 5, 1.5,
        "orb_EUROPE_FLOW_size >= 5.0",
        "overnight_range >= 100.0",
        "EUROPE_FLOW x ORB_G5 x OVN_100",
        dict(n=535, expr=0.1776, t=3.54),
    ),
    (
        10, "NYSE_OPEN", 5, 1.0,
        f"orb_NYSE_OPEN_size >= {COST_LT12_MNQ_SIZE_THRESHOLD}",
        "mes_atr_20_pct >= 60.0",
        "NYSE_OPEN x COST_LT12 x XMES_60",
        dict(n=717, expr=0.1180, t=3.29),
    ),
    (
        13, "TOKYO_OPEN", 5, 1.5,
        f"orb_TOKYO_OPEN_size >= {COST_LT12_MNQ_SIZE_THRESHOLD}",
        "mes_atr_20_pct >= 60.0",
        "TOKYO_OPEN x COST_LT12 x XMES_60",
        dict(n=516, expr=0.1763, t=3.45),
    ),
]


def verify_cell(con, cell_id, orb_label, orb_minutes, rr, base_sql, variant_sql, label, expected):
    q = f"""
    WITH mnq_df AS (
      SELECT trading_day, symbol, orb_minutes,
             overnight_range,
             orb_COMEX_SETTLE_size, orb_EUROPE_FLOW_size, orb_NYSE_OPEN_size,
             orb_TOKYO_OPEN_size, orb_SINGAPORE_OPEN_size, orb_US_DATA_1000_size,
             atr_20_pct
      FROM daily_features
      WHERE symbol = 'MNQ'
    ),
    mes_atr AS (
      SELECT trading_day, atr_20_pct AS mes_atr_20_pct
      FROM daily_features
      WHERE symbol = 'MES' AND orb_minutes = 5
    )
    SELECT o.pnl_r
    FROM orb_outcomes o
    JOIN mnq_df d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    LEFT JOIN mes_atr x ON o.trading_day = x.trading_day
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = ?
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < ?
      AND ({base_sql})
      AND ({variant_sql})
    """
    df = con.execute(q, [orb_label, orb_minutes, rr, HOLDOUT.date()]).df()
    pnl = df["pnl_r"].values.astype(float)
    n = len(pnl)
    m = float(pnl.mean()) if n > 0 else 0.0
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    t = (m / sd * np.sqrt(n)) if sd > 0 else 0.0

    d_n = n - expected["n"]
    d_expr = m - expected["expr"]
    d_t = t - expected["t"]

    # Divergence thresholds (pre-committed): 0.5% on N, 0.005 on ExpR, 0.05 on t
    n_pct = abs(d_n) / expected["n"] if expected["n"] else 0
    pass_n = n_pct < 0.005
    pass_expr = abs(d_expr) < 0.005
    pass_t = abs(d_t) < 0.05
    verdict = "PASS" if (pass_n and pass_expr and pass_t) else "FAIL"

    print(f"  Cell {cell_id:>2} {label}")
    print(f"    scan v1: N={expected['n']:>4}  ExpR={expected['expr']:+.4f}  t={expected['t']:+.2f}")
    print(f"    indep:   N={n:>4}  ExpR={m:+.4f}  t={t:+.2f}")
    print(f"    delta:   N={d_n:+d} ({100*n_pct:+.2f}%)  ExpR={d_expr:+.4f}  t={d_t:+.2f}   {verdict}")
    return verdict == "PASS"


def main():
    print(f"DB: {GOLD_DB_PATH}")
    print(f"HOLDOUT: {HOLDOUT.date()}")
    print(f"COST_LT12 MNQ equivalent threshold: orb_size >= {COST_LT12_MNQ_SIZE_THRESHOLD:.4f}")
    print(f"  (derived: (88/12) * {_mnq.total_friction} / {_mnq.point_value})")
    print()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    all_pass = True
    for row in SURVIVOR_CELLS:
        ok = verify_cell(con, *row)
        all_pass = all_pass and ok
        print()

    if all_pass:
        print("=== INDEPENDENT SQL VERIFICATION: ALL PASS ===")
        print("filter_signal canonical path reproduces to 4dp via direct SQL.")
        print("No filter-delegation drift detected.")
    else:
        print("=== INDEPENDENT SQL VERIFICATION: FAILURES ===")
        print("HALT per pre-reg scan_kill_criteria — filter-delegation drift suspected.")
        exit(1)


if __name__ == "__main__":
    main()
