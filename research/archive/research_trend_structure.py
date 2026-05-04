"""
Trend Structure & Trendline Research

Q1: Does trading the ORB in the direction of prevailing day structure work?
    HH_HL (uptrend) → long ORB should win more than short
    LH_LL (downtrend) → short ORB should win more than long

Q2: Multi-day trend — 2+ consecutive same-structure days = stronger signal?

Q3: Inside day breakout — inside days break with high conviction?
    (Price coiled inside prior range, then ORB breaks = explosive?)

Q4: Day-type from existing feature (TREND_UP/TREND_DOWN vs BALANCED/NON_TREND)
    Does TREND_UP day predict long ORB win?

All year-by-year consistency checked.
"""

import sys, io
from pathlib import Path
from collections import defaultdict
import numpy as np
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
SYMBOLS = ["MES", "MNQ", "MGC", "M2K"]
MIN_N = 40


def wr_stats(rows, win_col="outcome"):
    if not rows:
        return 0, 0, {}
    wins = sum(1 for r in rows if r[win_col] == "win")
    wr = wins / len(rows)
    by_yr = defaultdict(list)
    for r in rows:
        by_yr[r["yr"]].append(1 if r[win_col] == "win" else 0)
    return len(rows), wr, by_yr


def yr_consist(by_yr, thresh=0.55):
    rates = {y: np.mean(v) for y, v in by_yr.items() if len(v) >= 8}
    pos = sum(1 for v in rates.values() if v > thresh)
    return pos, len(rates)


# Load data — use computed structure (no null dependency)
sessions = ["1000", "US_EQUITY_OPEN", "0900"]
for session in sessions:
    q = f"""
    SELECT
        symbol,
        trading_day,
        YEAR(trading_day) as yr,
        -- Day structure computed from actual OHLC vs prior day
        CASE
            WHEN daily_high > prev_day_high AND daily_low > prev_day_low THEN 'HH_HL'
            WHEN daily_high < prev_day_high AND daily_low < prev_day_low THEN 'LH_LL'
            WHEN daily_high > prev_day_high AND daily_low < prev_day_low THEN 'EXPAND'
            WHEN daily_high < prev_day_high AND daily_low > prev_day_low THEN 'INSIDE'
        END as struct,
        day_type,
        prev_day_direction,
        orb_{session}_break_dir  as break_dir,
        orb_{session}_outcome    as outcome,
        orb_{session}_mfe_r      as mfe_r,
        orb_{session}_mae_r      as mae_r
    FROM daily_features
    WHERE symbol IN ('MES','MNQ','MGC','M2K')
      AND prev_day_high IS NOT NULL
      AND daily_high IS NOT NULL
      AND orb_{session}_break_dir IS NOT NULL
      AND orb_{session}_outcome   IS NOT NULL
    """
    rows = [
        dict(zip(["sym", "day", "yr", "struct", "day_type", "prev_dir", "break_dir", "outcome", "mfe", "mae"], r))
        for r in con.execute(q).fetchall()
    ]

    if not rows:
        continue

    print("=" * 72)
    print(f"SESSION: {session}")
    print("=" * 72)

    for sym in SYMBOLS:
        sym_rows = [r for r in rows if r["sym"] == sym]
        if len(sym_rows) < MIN_N:
            continue

        baseline_wr = sum(1 for r in sym_rows if r["outcome"] == "win") / len(sym_rows)
        print(f"\n  {sym} — baseline WR={baseline_wr:.1%} (n={len(sym_rows)})")

        # Q1: Structure-aligned vs counter
        print("  Q1: Trade direction vs day structure")
        for struct_label, trend_dir, counter_dir in [
            ("HH_HL (uptrend)", "long", "short"),
            ("LH_LL (downtrend)", "short", "long"),
            ("INSIDE_DAY", None, None),
            ("EXPAND", None, None),
        ]:
            struct_key = struct_label.split()[0]
            struct_rows = [r for r in sym_rows if r["struct"] == struct_key]
            if not struct_rows:
                continue

            if trend_dir:
                aligned = [r for r in struct_rows if r["break_dir"] == trend_dir]
                counter = [r for r in struct_rows if r["break_dir"] == counter_dir]

                n_a, wr_a, by_yr_a = wr_stats(aligned)
                n_c, wr_c, by_yr_c = wr_stats(counter)
                p_a, t_a = yr_consist(by_yr_a)
                uplift = wr_a - baseline_wr

                flag = " << SIGNAL" if uplift > 0.04 and p_a >= max(t_a * 0.6, 2) else ""
                if n_a >= MIN_N:
                    print(
                        f"    {struct_label:<22} aligned:  n={n_a:3d}  WR={wr_a:.1%}  "
                        f"uplift={uplift:+.1%}  yrs={p_a}/{t_a}{flag}"
                    )
                if n_c >= MIN_N:
                    up_c = wr_c - baseline_wr
                    print(f"    {struct_label:<22} counter:  n={n_c:3d}  WR={wr_c:.1%}  uplift={up_c:+.1%}")
            else:
                n_s, wr_s, by_yr_s = wr_stats(struct_rows)
                p_s, t_s = yr_consist(by_yr_s)
                uplift = wr_s - baseline_wr
                flag = " << SIGNAL" if abs(uplift) > 0.04 and p_s >= max(t_s * 0.6, 2) else ""
                if n_s >= MIN_N:
                    print(
                        f"    {struct_label:<22} all:      n={n_s:3d}  WR={wr_s:.1%}  "
                        f"uplift={uplift:+.1%}  yrs={p_s}/{t_s}{flag}"
                    )

        # Q4: day_type column (where available)
        dt_rows = [r for r in sym_rows if r["day_type"] is not None]
        if len(dt_rows) > MIN_N:
            print("  Q4: day_type feature vs ORB direction alignment")
            for dt, aligned_dir in [
                ("TREND_UP", "long"),
                ("TREND_DOWN", "short"),
                ("BALANCED", None),
                ("NON_TREND", None),
            ]:
                dt_sub = [r for r in dt_rows if r["day_type"] == dt]
                if not dt_sub or len(dt_sub) < MIN_N:
                    continue
                if aligned_dir:
                    aligned = [r for r in dt_sub if r["break_dir"] == aligned_dir]
                    n_a, wr_a, by_yr_a = wr_stats(aligned)
                    p_a, t_a = yr_consist(by_yr_a)
                    uplift = wr_a - baseline_wr
                    flag = " << SIGNAL" if uplift > 0.04 and p_a >= max(t_a * 0.6, 2) else ""
                    if n_a >= MIN_N:
                        print(
                            f"    {dt:<15} aligned: n={n_a:3d}  WR={wr_a:.1%}  "
                            f"uplift={uplift:+.1%}  yrs={p_a}/{t_a}{flag}"
                        )
                else:
                    n_s, wr_s, by_yr_s = wr_stats(dt_sub)
                    p_s, t_s = yr_consist(by_yr_s)
                    uplift = wr_s - baseline_wr
                    if n_s >= MIN_N:
                        print(f"    {dt:<15} all:     n={n_s:3d}  WR={wr_s:.1%}  uplift={uplift:+.1%}  yrs={p_s}/{t_s}")

print()
print("=" * 72)
print("HONEST NOTE ON TRENDLINE TRADING (ALGORITHMIC):")
print("=" * 72)
print("""
True trendline detection (swing point -> swing point lines) requires:
  1. bars_5m data + pivot detection algorithm
  2. Min 2 touches to confirm the line
  3. A definition of 'at the line' vs 'broke through'
  4. Many parameter choices = HIGH overfit risk

The tests above capture the CORE hypothesis: does daily structural
trend direction predict ORB break success?

If yes: can add as a filter/overlay without needing swing-point trendlines.
If no: traditional trendline algo will likely overfit to a non-existent edge.
""")

con.close()
