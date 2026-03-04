"""
New hypotheses — nothing to do with mean reversion or VWAP.

H1: COMPRESSION → EXPLOSION
    When the ORB is in the tightest compression tier, does the break
    have better win rate / larger moves than an uncompressed ORB?

H2: MULTI-DAY MOMENTUM
    When prev_day went LONG and today's ORB breaks LONG (same direction),
    is the win rate higher than a random ORB break?

H3: VOLUME IMPULSE AT BREAK
    When the break bar has high relative volume (top 40%),
    does the trade win more often?

All anti-overfit: year-by-year consistency required.
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
MIN_N = 50

def year_consistency(by_year, threshold):
    rates = {y: np.mean(v) for y, v in by_year.items() if len(v) >= 10}
    pos = sum(1 for v in rates.values() if v > threshold)
    return pos, len(rates), rates

# ============================================================
print("=" * 70)
print("H1: COMPRESSION TIER -> ORB OUTCOME")
print("Does a compressed ORB break harder than a wide one?")
print("=" * 70)

sessions_comp = ['0900', '1000', '1800']
for session in sessions_comp:
    q = f"""
        SELECT symbol, YEAR(trading_day),
               orb_{session}_compression_tier,
               orb_{session}_outcome,
               orb_{session}_mfe_r
        FROM daily_features
        WHERE symbol IN ('MES','MNQ','MGC','M2K')
          AND orb_{session}_compression_tier IS NOT NULL
          AND orb_{session}_outcome IS NOT NULL
          AND orb_{session}_mfe_r IS NOT NULL
    """
    rows = con.execute(q).fetchall()
    if not rows: continue

    tiers = {}
    for sym, yr, tier, outcome, mfe in rows:
        key = (sym, tier)
        if key not in tiers:
            tiers[key] = {'wins': 0, 'n': 0, 'mfe': [], 'by_yr': defaultdict(list)}
        tiers[key]['n'] += 1
        tiers[key]['mfe'].append(mfe)
        tiers[key]['by_yr'][yr].append(1 if outcome == 'win' else 0)
        if outcome == 'win':
            tiers[key]['wins'] += 1

    print(f"\nSession {session}:")
    for sym in SYMBOLS:
        row_data = []
        for tier in ['compressed', 'normal', 'wide']:
            k = (sym, tier)
            if k not in tiers or tiers[k]['n'] < MIN_N: continue
            d = tiers[k]
            wr = d['wins'] / d['n']
            avg_mfe = np.mean(d['mfe'])
            pos, tot, _ = year_consistency(d['by_yr'], 0.55)
            row_data.append((tier, d['n'], wr, avg_mfe, pos, tot))
        if not row_data: continue
        print(f"  {sym}:")
        for tier, n, wr, mfe, pos, tot in row_data:
            flag = " << " if tier == 'compressed' and wr > 0.60 else ""
            print(f"    {tier:12s}: n={n:4d}  WR={wr:.1%}  avgMFE={mfe:.2f}R  yrs={pos}/{tot}{flag}")


# ============================================================
print("\n")
print("=" * 70)
print("H2: MULTI-DAY MOMENTUM")
print("Prev day direction SAME as today's ORB break = higher win rate?")
print("=" * 70)

sessions_mom = ['1000', 'US_EQUITY_OPEN']
for session in sessions_mom:
    q = f"""
        SELECT symbol, YEAR(trading_day),
               prev_day_direction,
               orb_{session}_break_dir,
               orb_{session}_outcome,
               orb_{session}_mfe_r
        FROM daily_features
        WHERE symbol IN ('MES','MNQ','MGC','M2K')
          AND prev_day_direction IS NOT NULL
          AND orb_{session}_break_dir IS NOT NULL
          AND orb_{session}_outcome IS NOT NULL
    """
    rows = con.execute(q).fetchall()
    if not rows: continue

    print(f"\nSession {session}:")
    for sym in SYMBOLS:
        sym_rows = [r for r in rows if r[0] == sym]
        if not sym_rows: continue

        # With momentum (prev day same direction as break)
        with_mom = [r for r in sym_rows
                    if (r[2] == 'up' and r[3] == 'long') or
                       (r[2] == 'down' and r[3] == 'short')]
        against_mom = [r for r in sym_rows
                       if (r[2] == 'up' and r[3] == 'short') or
                          (r[2] == 'down' and r[3] == 'long')]
        baseline = sym_rows

        def stats(rows_):
            if not rows_: return 0, 0, 0, {}
            wr = sum(1 for r in rows_ if r[4] == 'win') / len(rows_)
            mfe = np.mean([r[5] for r in rows_ if r[5] is not None])
            by_yr = defaultdict(list)
            for r in rows_:
                by_yr[r[1]].append(1 if r[4] == 'win' else 0)
            return len(rows_), wr, mfe, by_yr

        n_m, wr_m, mfe_m, by_yr_m = stats(with_mom)
        n_a, wr_a, mfe_a, by_yr_a = stats(against_mom)
        n_b, wr_b, mfe_b, _ = stats(baseline)

        if n_m < MIN_N: continue

        pos_m, tot_m, _ = year_consistency(by_yr_m, 0.55)
        uplift = wr_m - wr_b

        flag = " << SIGNAL" if uplift > 0.04 and pos_m >= tot_m * 0.6 else ""
        print(f"  {sym}:")
        print(f"    baseline:   n={n_b:4d}  WR={wr_b:.1%}  avgMFE={mfe_b:.2f}R")
        print(f"    with_mom:   n={n_m:4d}  WR={wr_m:.1%}  avgMFE={mfe_m:.2f}R  "
              f"uplift={uplift:+.1%}  yrs={pos_m}/{tot_m}{flag}")
        print(f"    against_mom:n={n_a:4d}  WR={wr_a:.1%}  avgMFE={mfe_a:.2f}R")


# ============================================================
print("\n")
print("=" * 70)
print("H3: VOLUME IMPULSE AT BREAK BAR")
print("High relative volume at break = stronger follow-through?")
print("=" * 70)

sessions_vol = ['1000', 'US_EQUITY_OPEN', '0900']
for session in sessions_vol:
    rel_col = f"rel_vol_{session}"
    q = f"""
        SELECT symbol, YEAR(trading_day),
               {rel_col},
               orb_{session}_outcome,
               orb_{session}_mfe_r
        FROM daily_features
        WHERE symbol IN ('MES','MNQ','MGC','M2K')
          AND {rel_col} IS NOT NULL
          AND orb_{session}_outcome IS NOT NULL
    """
    try:
        rows = con.execute(q).fetchall()
    except: continue
    if not rows: continue

    print(f"\nSession {session}:")
    for sym in SYMBOLS:
        sym_rows = [r for r in rows if r[0] == sym]
        if len(sym_rows) < MIN_N: continue

        # Volume percentiles
        vols = sorted([r[2] for r in sym_rows if r[2] is not None])
        p40 = np.percentile(vols, 40)
        p60 = np.percentile(vols, 60)
        p80 = np.percentile(vols, 80)

        groups = {
            'low_vol (<40%)':  [r for r in sym_rows if r[2] and r[2] < p40],
            'mid_vol (40-80%)': [r for r in sym_rows if r[2] and p40 <= r[2] < p80],
            'high_vol (>80%)': [r for r in sym_rows if r[2] and r[2] >= p80],
        }

        baseline_wr = sum(1 for r in sym_rows if r[3] == 'win') / len(sym_rows)
        print(f"  {sym} (baseline WR={baseline_wr:.1%}):")

        for label, g in groups.items():
            if len(g) < 20: continue
            wr = sum(1 for r in g if r[3] == 'win') / len(g)
            mfe = np.mean([r[4] for r in g if r[4] is not None])
            by_yr = defaultdict(list)
            for r in g:
                by_yr[r[1]].append(1 if r[3] == 'win' else 0)
            pos, tot, _ = year_consistency(by_yr, 0.55)
            uplift = wr - baseline_wr
            flag = " << SIGNAL" if abs(uplift) > 0.04 and pos >= tot * 0.6 else ""
            print(f"    {label}: n={len(g):4d}  WR={wr:.1%}  "
                  f"avgMFE={mfe:.2f}R  uplift={uplift:+.1%}  yrs={pos}/{tot}{flag}")

con.close()
print("\nDone.")
