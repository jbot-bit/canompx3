"""
Level-Tag Reversion Research

Hypothesis: After an ORB break, price extends to tag an external reference level
(overnight high/low, prior day high/low, prior session extreme), then REVERTS
back — often through the ORB entirely.

Setup (long break example):
  1. 10am ORB breaks LONG
  2. External level exists ABOVE ORB (overnight_high, prev_day_high, etc.)
  3. Daily high REACHES that external level (confirmed tag)
  4. Daily close ends UP BELOW the ORB high (reverted back through ORB top)

Measure:
  - How often does the tag → reversion pattern occur?
  - How far does it revert? (to ORB mid? to ORB low? below?)
  - Is it consistent across years? (anti-overfit)
  - Which external levels work best?
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
MIN_N = 30

# Load base data
q = """
SELECT
    symbol,
    trading_day,
    YEAR(trading_day) as yr,

    -- ORB boundaries (using 10am session — most liquid)
    orb_1000_high,
    orb_1000_low,
    orb_1000_size,
    orb_1000_break_dir,
    orb_1000_outcome,

    -- External reference levels
    overnight_high,
    overnight_low,
    prev_day_high,
    prev_day_low,
    prev_day_close,
    session_london_high,
    session_london_low,
    session_asia_high,
    session_asia_low,

    -- What actually happened
    daily_high,
    daily_low,
    daily_close,
    daily_open

FROM daily_features
WHERE symbol IN ('MES','MNQ','MGC','M2K')
  AND orb_1000_break_dir IS NOT NULL
  AND orb_1000_high IS NOT NULL
  AND daily_high IS NOT NULL
  AND daily_close IS NOT NULL
  AND overnight_high IS NOT NULL
  AND prev_day_high IS NOT NULL
"""
rows = con.execute(q).fetchall()
cols = ['symbol','trading_day','yr','orb_h','orb_l','orb_sz','break_dir','outcome',
        'on_h','on_l','pdh','pdl','pdc','lon_h','lon_l','asia_h','asia_l',
        'dh','dl','dc','do_']

data = [dict(zip(cols, r)) for r in rows]

def orb_mid(r):
    if r['orb_h'] and r['orb_l']:
        return (r['orb_h'] + r['orb_l']) / 2
    return None

def to_r(points, orb_sz):
    if orb_sz and orb_sz > 0:
        return points / orb_sz
    return None

print("=" * 70)
print("LEVEL-TAG REVERSION ANALYSIS")
print("=" * 70)

# External levels to test (label, level_for_long_break, level_for_short_break)
ext_levels = [
    ("Overnight H/L",   'on_h',   'on_l'),
    ("Prior Day H/L",   'pdh',    'pdl'),
    ("London H/L",      'lon_h',  'lon_l'),
    ("Asia H/L",        'asia_h', 'asia_l'),
]

for level_name, long_level_col, short_level_col in ext_levels:
    print(f"\n--- External level: {level_name} ---")

    for sym in SYMBOLS:
        sym_data = [r for r in data if r['symbol'] == sym]

        results = {'long': [], 'short': []}

        for r in sym_data:
            dir_ = r['break_dir']
            orb_h = r['orb_h']
            orb_l = r['orb_l']
            orb_sz = r['orb_sz']
            mid = orb_mid(r)
            if not mid or not orb_sz:
                continue

            if dir_ == 'long':
                ext_col = long_level_col
                ext_val = r[ext_col]
                if not ext_val or ext_val <= orb_h:
                    continue  # level not above ORB — no tag possible from above

                # Did price tag the external level?
                tagged = r['dh'] >= ext_val

                # After tagging: did price revert back below ORB top?
                reverted_through_orb = r['dc'] < orb_h

                # How far did it revert? In R units
                revert_depth_r = to_r(orb_h - r['dc'], orb_sz)  # + = below ORB top

                results['long'].append({
                    'yr': r['yr'],
                    'tagged': tagged,
                    'reverted': tagged and reverted_through_orb,
                    'revert_r': revert_depth_r if tagged else None,
                    'ext_dist_r': to_r(ext_val - orb_h, orb_sz),  # how far level was above ORB
                })

            else:  # short
                ext_col = short_level_col
                ext_val = r[ext_col]
                if not ext_val or ext_val >= orb_l:
                    continue  # level not below ORB

                tagged = r['dl'] <= ext_val
                reverted_through_orb = r['dc'] > orb_l
                revert_depth_r = to_r(r['dc'] - orb_l, orb_sz)

                results['short'].append({
                    'yr': r['yr'],
                    'tagged': tagged,
                    'reverted': tagged and reverted_through_orb,
                    'revert_r': revert_depth_r if tagged else None,
                    'ext_dist_r': to_r(orb_l - ext_val, orb_sz),
                })

        for dir_, res in results.items():
            tagged_res = [r for r in res if r['tagged']]
            if len(tagged_res) < MIN_N:
                continue

            total_with_level = len(res)
            n_tagged = len(tagged_res)
            tag_rate = n_tagged / total_with_level

            n_reverted = sum(1 for r in tagged_res if r['reverted'])
            rev_rate = n_reverted / n_tagged

            revert_rs = [r['revert_r'] for r in tagged_res if r['reverted'] and r['revert_r'] is not None]
            avg_rev_r = np.mean(revert_rs) if revert_rs else 0

            # Year consistency of reversion
            by_yr = defaultdict(list)
            for r in tagged_res:
                by_yr[r['yr']].append(r['reverted'])
            yr_rates = {y: np.mean(v) for y, v in by_yr.items() if len(v) >= 5}
            yrs_positive = sum(1 for v in yr_rates.values() if v > 0.45)
            yrs_total = len(yr_rates)

            # Non-tagged: what's baseline reversion rate?
            non_tagged = [r for r in res if not r['tagged']]
            base_rev_rs = [r['revert_r'] for r in non_tagged if r['revert_r'] is not None]
            # base rev rate = close below ORB top (for longs)
            base_rev = [r for r in res if not r['tagged'] and to_r(
                (r.get('orb_h',0) or 0) - 0, 1) is not None]
            # simpler: just check baseline outcome
            base_close_below = sum(1 for r in non_tagged
                                   if r.get('revert_r') is not None and r['revert_r'] and r['revert_r'] > 0)
            base_rev_rate = base_close_below / len(non_tagged) if non_tagged else 0

            uplift = rev_rate - base_rev_rate

            signal = " << SIGNAL" if uplift > 0.08 and yrs_positive >= yrs_total * 0.7 else ""

            print(f"  {sym} {dir_:5s}: "
                  f"level exists {total_with_level:4d}d | "
                  f"tagged {n_tagged:4d} ({tag_rate:.0%}) | "
                  f"then reverted {rev_rate:.0%} | "
                  f"base {base_rev_rate:.0%} | "
                  f"uplift {uplift:+.0%} | "
                  f"avg rev depth {avg_rev_r:.2f}R | "
                  f"yrs consistent {yrs_positive}/{yrs_total}{signal}")

print()
print("=" * 70)
print("SUMMARY: Reversion after level tag (close ends below ORB top for longs)")
print("Uplift = tagged-days reversion rate MINUS non-tagged reversion rate")
print("Rev depth = how many R below ORB top does close end up (avg, on reverted days)")
print("=" * 70)

con.close()
