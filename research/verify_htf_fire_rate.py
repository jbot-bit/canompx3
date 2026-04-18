#!/usr/bin/env python3
"""Fire-rate smoke test for HTF v1-index hypothesis family.

Purpose:
    After HTF fields are built (via build_daily_features or the one-shot
    backfill), verify that every cell in the intended HTF v1-index
    pre-registration family shows a fire_rate strictly within [5%, 95%]
    on the pre-holdout horizon. Any cell outside that band is dead
    (too rare to test) or constant (no predictive signal), and must be
    dropped from the pre-reg before lock per
    .claude/rules/backtesting-methodology.md RULE 8.1.

This is NOT a discovery or validation run. It only verifies structural
fire-rate integrity so the next pass (pre-reg lock + validation) can
proceed with honest sample sizes.

Family (v1-index, pre-holdout only; trading_day < 2026-01-01):
    instruments: MNQ, MES
    sessions:    TOKYO_OPEN, EUROPE_FLOW, NYSE_OPEN
    aperture:    O15
    directions / cells:
        pwh_first_touch_long  = break_dir='long'  AND orb_high > prev_week_high
                                AND week_took_pwh_to_date = FALSE
        pwl_first_touch_short = break_dir='short' AND orb_low  < prev_week_low
                                AND week_took_pwl_to_date = FALSE

The ``week_took_pwh_to_date`` flag is derived at query time from the raw
HTF fields + daily_high/daily_low (no new canonical column needed). See
"DERIVATION" comment in the SQL below.

Usage:
    DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/verify_htf_fire_rate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH


INSTRUMENTS = ("MNQ", "MES")
SESSIONS = ("TOKYO_OPEN", "EUROPE_FLOW", "NYSE_OPEN")
ORB_MINUTES = 15
HOLDOUT_BOUNDARY = "2026-01-01"

# DERIVATION — week_took_pwh_to_date / week_took_pwl_to_date are derived at
# scan time as a window over daily_high / daily_low keyed by
# DATE_TRUNC('week', trading_day). No canonical column is added.
FIRE_RATE_SQL = """
WITH df AS (
    SELECT trading_day, symbol,
           daily_high, daily_low,
           prev_week_high, prev_week_low,
           orb_{sess}_break_dir AS break_dir,
           orb_{sess}_high      AS o_high,
           orb_{sess}_low       AS o_low,
           DATE_TRUNC('week', trading_day)::DATE AS week_key
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = ?
      AND trading_day < '{holdout}'
),
flags AS (
    SELECT df.*,
           -- week_took_pwh_to_date: any earlier same-week row had daily_high > prev_week_high
           COALESCE(
               MAX(CASE WHEN df.prev_week_high IS NOT NULL
                         AND df.daily_high > df.prev_week_high THEN 1 ELSE 0 END)
                 OVER (PARTITION BY df.symbol, df.week_key
                       ORDER BY df.trading_day
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
               0
           ) AS took_pwh_prior,
           COALESCE(
               MAX(CASE WHEN df.prev_week_low IS NOT NULL
                         AND df.daily_low < df.prev_week_low THEN 1 ELSE 0 END)
                 OVER (PARTITION BY df.symbol, df.week_key
                       ORDER BY df.trading_day
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
               0
           ) AS took_pwl_prior
    FROM df
)
SELECT
    SUM(CASE WHEN prev_week_high IS NOT NULL AND break_dir IS NOT NULL THEN 1 ELSE 0 END) AS base_rows,
    SUM(CASE WHEN break_dir = 'long' AND prev_week_high IS NOT NULL
              AND o_high > prev_week_high AND took_pwh_prior = 0 THEN 1 ELSE 0 END) AS pwh_first_touch_long_fires,
    SUM(CASE WHEN break_dir = 'short' AND prev_week_low IS NOT NULL
              AND o_low  < prev_week_low  AND took_pwl_prior = 0 THEN 1 ELSE 0 END) AS pwl_first_touch_short_fires
FROM flags
"""


def _fire_rate(fires: int | None, base: int | None) -> float | None:
    if fires is None or base is None or base == 0:
        return None
    return round(100.0 * fires / base, 2)


def _band_flag(rate: float | None) -> str:
    if rate is None:
        return "NO_BASE"
    if rate < 5.0:
        return "DEAD (<5%)"
    if rate > 95.0:
        return "CONSTANT (>95%)"
    return "OK"


def main() -> int:
    db = GOLD_DB_PATH
    print(f"Canonical DB: {db}")
    if not db.exists():
        print(f"FATAL: DB not found at {db}", file=sys.stderr)
        return 2

    print()
    print(f"{'symbol':<6} {'session':<14} base  pwh_long  rate%   band         pwl_short rate%   band")
    print("-" * 110)

    any_bad = False
    with duckdb.connect(str(db), read_only=True) as con:
        for sym in INSTRUMENTS:
            for sess in SESSIONS:
                sql = FIRE_RATE_SQL.format(sess=sess, holdout=HOLDOUT_BOUNDARY)
                try:
                    row = con.execute(sql, [sym, ORB_MINUTES]).fetchone()
                except duckdb.Error as e:
                    print(f"{sym:<6} {sess:<14} ERROR: {e}")
                    any_bad = True
                    continue
                if row is None:
                    print(f"{sym:<6} {sess:<14} no rows")
                    any_bad = True
                    continue
                base, pwh_long, pwl_short = row
                r_long = _fire_rate(pwh_long, base)
                r_short = _fire_rate(pwl_short, base)
                band_long = _band_flag(r_long)
                band_short = _band_flag(r_short)
                print(
                    f"{sym:<6} {sess:<14} {base or 0:>5} "
                    f"{pwh_long or 0:>8} {str(r_long):>7} {band_long:<13} "
                    f"{pwl_short or 0:>8} {str(r_short):>7} {band_short}"
                )
                if band_long != "OK" or band_short != "OK":
                    any_bad = True

    print()
    if any_bad:
        print("Some cells outside [5%, 95%] band — they must be dropped from the v1-index pre-reg.")
    else:
        print("All cells within band. Family is pre-reg-ready for validation pass.")
    return 0 if not any_bad else 1


if __name__ == "__main__":
    sys.exit(main())
