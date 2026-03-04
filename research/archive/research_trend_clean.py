"""
Trend direction using only PRE-TRADE data (no look-ahead bias).

For US_EQUITY_OPEN (9:30am ET):
  Known before trade: overnight direction, London session direction,
  whether overnight is above/below prior day close.

For 1000 session (10am ET):
  Known before trade: pre_1000_high/low (data from open to 9:59am),
  overnight direction.

Filter: does pre-trade trend direction predict ORB break outcome?
"""

import sys, io
from pathlib import Path
from collections import defaultdict
import numpy as np
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
SYMBOLS = ['MES', 'MNQ', 'MGC', 'M2K']
MIN_N = 40

def yr_consist(by_yr, thresh=0.55):
    rates = {y: np.mean(v) for y, v in by_yr.items() if len(v) >= 8}
    pos = sum(1 for v in rates.values() if v > thresh)
    return pos, len(rates)

# ============================================================
# US_EQUITY_OPEN — pre-trade context from overnight + London
# ============================================================
print("=" * 72)
print("US_EQUITY_OPEN — pre-trade trend (overnight + London, no look-ahead)")
print("=" * 72)

q = """
SELECT
    symbol, YEAR(trading_day) as yr,

    -- Overnight direction vs prior day close (known at 9:30am)
    CASE
        WHEN daily_open > prev_day_close AND daily_open > prev_day_high THEN 'gap_up_above_pdh'
        WHEN daily_open > prev_day_close                                 THEN 'gap_up'
        WHEN daily_open < prev_day_close AND daily_open < prev_day_low  THEN 'gap_down_below_pdl'
        WHEN daily_open < prev_day_close                                 THEN 'gap_down'
        ELSE 'flat'
    END as open_context,

    -- London session direction vs Asia (known at 9:30am)
    CASE
        WHEN session_london_high > session_asia_high
         AND session_london_low  > session_asia_low  THEN 'london_hh_hl'
        WHEN session_london_high < session_asia_high
         AND session_london_low  < session_asia_low  THEN 'london_lh_ll'
        ELSE 'london_mixed'
    END as london_ctx,

    -- Overnight range position at open (above/below midpoint)
    CASE
        WHEN daily_open > (overnight_high + overnight_low) / 2 THEN 'open_top_half'
        ELSE 'open_bottom_half'
    END as overnight_pos,

    orb_US_EQUITY_OPEN_break_dir as break_dir,
    orb_US_EQUITY_OPEN_outcome   as outcome,
    orb_US_EQUITY_OPEN_mfe_r     as mfe_r

FROM daily_features
WHERE symbol IN ('MES','MNQ','MGC','M2K')
  AND prev_day_close IS NOT NULL
  AND overnight_high IS NOT NULL
  AND session_london_high IS NOT NULL
  AND orb_US_EQUITY_OPEN_break_dir IS NOT NULL
  AND orb_US_EQUITY_OPEN_outcome   IS NOT NULL
"""
rows = [dict(zip(['sym','yr','open_ctx','london_ctx','on_pos','break_dir','outcome','mfe'], r))
        for r in con.execute(q).fetchall()]

for sym in SYMBOLS:
    sr = [r for r in rows if r['sym'] == sym]
    if not sr: continue
    base_wr = sum(1 for r in sr if r['outcome']=='win') / len(sr)
    print(f"\n{sym} — baseline WR={base_wr:.1%} (n={len(sr)})")

    # Gap context
    print("  Open context (gap direction vs prior day):")
    for ctx in ['gap_up_above_pdh','gap_up','gap_down','gap_down_below_pdl','flat']:
        grp = [r for r in sr if r['open_ctx']==ctx]
        if len(grp) < MIN_N: continue
        # aligned = gap_up → long, gap_down → short
        if 'up' in ctx:
            aligned = [r for r in grp if r['break_dir']=='long']
            counter = [r for r in grp if r['break_dir']=='short']
        else:
            aligned = [r for r in grp if r['break_dir']=='short']
            counter = [r for r in grp if r['break_dir']=='long']
        if len(aligned) < MIN_N: continue
        wr_a = sum(1 for r in aligned if r['outcome']=='win') / len(aligned)
        by_yr = defaultdict(list)
        for r in aligned: by_yr[r['yr']].append(1 if r['outcome']=='win' else 0)
        p, t = yr_consist(by_yr)
        uplift = wr_a - base_wr
        flag = " << SIGNAL" if abs(uplift) > 0.05 and p >= max(t*0.6, 2) else ""
        print(f"    {ctx:<30} aligned n={len(aligned):3d}  WR={wr_a:.1%}  "
              f"uplift={uplift:+.1%}  yrs={p}/{t}{flag}")

    # London context
    print("  London direction:")
    for ctx, aligned_dir in [('london_hh_hl','long'),('london_lh_ll','short')]:
        grp = [r for r in sr if r['london_ctx']==ctx]
        aligned = [r for r in grp if r['break_dir']==aligned_dir]
        if len(aligned) < MIN_N: continue
        wr_a = sum(1 for r in aligned if r['outcome']=='win') / len(aligned)
        by_yr = defaultdict(list)
        for r in aligned: by_yr[r['yr']].append(1 if r['outcome']=='win' else 0)
        p, t = yr_consist(by_yr)
        uplift = wr_a - base_wr
        flag = " << SIGNAL" if abs(uplift) > 0.05 and p >= max(t*0.6, 2) else ""
        print(f"    {ctx:<30} aligned n={len(aligned):3d}  WR={wr_a:.1%}  "
              f"uplift={uplift:+.1%}  yrs={p}/{t}{flag}")

    # Overnight position
    print("  Overnight range position at open:")
    for pos, aligned_dir in [('open_top_half','long'),('open_bottom_half','short')]:
        grp = [r for r in sr if r['on_pos']==pos]
        aligned = [r for r in grp if r['break_dir']==aligned_dir]
        if len(aligned) < MIN_N: continue
        wr_a = sum(1 for r in aligned if r['outcome']=='win') / len(aligned)
        by_yr = defaultdict(list)
        for r in aligned: by_yr[r['yr']].append(1 if r['outcome']=='win' else 0)
        p, t = yr_consist(by_yr)
        uplift = wr_a - base_wr
        flag = " << SIGNAL" if abs(uplift) > 0.05 and p >= max(t*0.6, 2) else ""
        print(f"    {pos:<30} aligned n={len(aligned):3d}  WR={wr_a:.1%}  "
              f"uplift={uplift:+.1%}  yrs={p}/{t}{flag}")


# ============================================================
# 1000 SESSION — pre-trade context from pre_1000 data
# ============================================================
print()
print("=" * 72)
print("1000 SESSION — pre-trade trend (pre_1000 data, no look-ahead)")
print("=" * 72)

q2 = """
SELECT
    symbol, YEAR(trading_day) as yr,

    -- Pre-10am structure vs prior day (all known before 10am ORB)
    CASE
        WHEN pre_1000_high > prev_day_high AND pre_1000_low > prev_day_low THEN 'pre_HH_HL'
        WHEN pre_1000_high < prev_day_high AND pre_1000_low < prev_day_low THEN 'pre_LH_LL'
        WHEN pre_1000_high > prev_day_high AND pre_1000_low < prev_day_low THEN 'pre_EXPAND'
        WHEN pre_1000_high < prev_day_high AND pre_1000_low > prev_day_low THEN 'pre_INSIDE'
        ELSE 'other'
    END as pre_struct,

    -- Did pre-market test PDH or PDL? (known before 10am)
    took_pdh_before_1000,
    took_pdl_before_1000,

    orb_1000_break_dir as break_dir,
    orb_1000_outcome   as outcome,
    orb_1000_mfe_r     as mfe_r

FROM daily_features
WHERE symbol IN ('MES','MNQ','MGC','M2K')
  AND pre_1000_high IS NOT NULL
  AND prev_day_high IS NOT NULL
  AND orb_1000_break_dir IS NOT NULL
  AND orb_1000_outcome IS NOT NULL
"""
rows2 = [dict(zip(['sym','yr','struct','pdh_hit','pdl_hit','break_dir','outcome','mfe'], r))
         for r in con.execute(q2).fetchall()]

for sym in SYMBOLS:
    sr = [r for r in rows2 if r['sym'] == sym]
    if not sr: continue
    base_wr = sum(1 for r in sr if r['outcome']=='win') / len(sr)
    print(f"\n{sym} — baseline WR={base_wr:.1%} (n={len(sr)})")

    print("  Pre-10am structure:")
    for struct, aligned_dir, counter_dir in [
        ('pre_HH_HL', 'long',  'short'),
        ('pre_LH_LL', 'short', 'long'),
        ('pre_INSIDE', None,   None),
        ('pre_EXPAND', None,   None),
    ]:
        grp = [r for r in sr if r['struct']==struct]
        if not grp: continue
        if aligned_dir:
            aligned = [r for r in grp if r['break_dir']==aligned_dir]
            if len(aligned) < MIN_N: continue
            wr_a = sum(1 for r in aligned if r['outcome']=='win') / len(aligned)
            by_yr = defaultdict(list)
            for r in aligned: by_yr[r['yr']].append(1 if r['outcome']=='win' else 0)
            p, t = yr_consist(by_yr)
            uplift = wr_a - base_wr
            flag = " << SIGNAL" if abs(uplift) > 0.04 and p >= max(t*0.6, 2) else ""
            print(f"    {struct:<15} aligned n={len(aligned):3d}  WR={wr_a:.1%}  "
                  f"uplift={uplift:+.1%}  yrs={p}/{t}{flag}")
        else:
            if len(grp) < MIN_N: continue
            wr_g = sum(1 for r in grp if r['outcome']=='win') / len(grp)
            by_yr = defaultdict(list)
            for r in grp: by_yr[r['yr']].append(1 if r['outcome']=='win' else 0)
            p, t = yr_consist(by_yr)
            uplift = wr_g - base_wr
            flag = " << SIGNAL" if abs(uplift) > 0.04 and p >= max(t*0.6, 2) else ""
            print(f"    {struct:<15} all    n={len(grp):3d}  WR={wr_g:.1%}  "
                  f"uplift={uplift:+.1%}  yrs={p}/{t}{flag}")

    # PDH/PDL hit before 10am
    print("  PDH/PDL test before 10am:")
    pdh_short = [r for r in sr if r['pdh_hit'] and r['break_dir']=='short']
    pdl_long  = [r for r in sr if r['pdl_hit'] and r['break_dir']=='long']
    for label, grp in [('pdh_hit→short', pdh_short), ('pdl_hit→long', pdl_long)]:
        if len(grp) < MIN_N: continue
        wr_g = sum(1 for r in grp if r['outcome']=='win') / len(grp)
        by_yr = defaultdict(list)
        for r in grp: by_yr[r['yr']].append(1 if r['outcome']=='win' else 0)
        p, t = yr_consist(by_yr)
        uplift = wr_g - base_wr
        flag = " << SIGNAL" if abs(uplift) > 0.04 and p >= max(t*0.6, 2) else ""
        print(f"    {label:<20} n={len(grp):3d}  WR={wr_g:.1%}  "
              f"uplift={uplift:+.1%}  yrs={p}/{t}{flag}")

con.close()
print("\nDone. All data is pre-trade (no look-ahead).")
