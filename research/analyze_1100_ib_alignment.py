"""
Research: Does 0900 IB (Initial Balance) direction alignment improve 1100?

The 0900 IB = high/low of first 120 min after 09:00 Brisbane (23:00-01:00 UTC).
IB completes at 11:00 Brisbane = 01:00 UTC = exactly when 1100 ORB starts.
IB break = first 1m close above IB high or below IB low AFTER 01:00 UTC.

For 1100 trades:
  - If IB breaks same direction as 1100 ORB = ALIGNED
  - If IB breaks opposite direction = OPPOSED
  - If IB hasn't broken by 1100 entry time = UNBROKEN

Zero lookahead: IB range is known at 01:00 UTC. IB break direction
may or may not be known by 1100 entry time (~01:10+ UTC for CB4).
We check if IB broke BEFORE the 1100 entry timestamp.

Also test: is the 1100 ORB break itself just the IB break? (correlation check)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec


def run_analysis(db_path=None):
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path), read_only=True)

    # Get 1100 outcomes with entry details
    query = """
    SELECT
        o.trading_day,
        o.rr_target,
        o.confirm_bars,
        o.outcome,
        o.pnl_r,
        o.entry_price,
        o.entry_ts,
        o.stop_price,
        df.orb_1100_break_dir AS break_dir_1100,
        df.orb_1100_size AS orb_size_1100,
        df.orb_0900_break_dir AS break_dir_0900,
        df.orb_0900_size AS orb_size_0900
    FROM orb_outcomes o
    JOIN daily_features df
      ON o.trading_day = df.trading_day
      AND df.symbol = 'MGC' AND df.orb_minutes = 5
    WHERE o.symbol = 'MGC'
      AND o.orb_label = '1100'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E1'
      AND o.entry_price IS NOT NULL
      AND o.outcome IS NOT NULL
      AND df.orb_1100_size >= 4.0
    ORDER BY o.trading_day, o.rr_target, o.confirm_bars
    """
    outcomes = con.execute(query).fetchdf()
    outcomes['entry_ts'] = pd.to_datetime(outcomes['entry_ts'], utc=True)

    print(f"Total 1100 E1 G4+ trades: {len(outcomes)}")
    print(f"Unique days: {outcomes['trading_day'].nunique()}")
    print()

    # For each trading day, compute IB range (23:00-01:00 UTC) and IB break direction
    unique_days = sorted(outcomes['trading_day'].unique())
    print(f"Computing IB for {len(unique_days)} trading days...")

    ib_data = {}
    batch_size = 50
    for i in range(0, len(unique_days), batch_size):
        batch = unique_days[i:i+batch_size]

        for td in batch:
            td_str = str(td)[:10]
            # IB range: 23:00 UTC prev day to 01:00 UTC this day
            # Trading day in Brisbane starts at 09:00 = 23:00 UTC prev day
            prev_day = pd.Timestamp(td) - pd.Timedelta(days=1)
            prev_str = prev_day.strftime('%Y-%m-%d')

            # Get IB bars (23:00 prev day to 01:00 this day UTC)
            ib_query = """
            SELECT ts_utc, open, high, low, close
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ?::TIMESTAMPTZ
              AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc
            """
            ib_start = f"{prev_str} 23:00:00+00:00"
            ib_end = f"{td_str} 01:00:00+00:00"
            ib_bars = con.execute(ib_query, [ib_start, ib_end]).fetchdf()

            if ib_bars.empty:
                continue

            ib_high = float(ib_bars['high'].max())
            ib_low = float(ib_bars['low'].min())

            # Now find IB break: first 1m close above IB high or below IB low
            # AFTER 01:00 UTC (after IB completes)
            post_ib_query = """
            SELECT ts_utc, close, high, low
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ?::TIMESTAMPTZ
              AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc
            """
            post_ib_start = f"{td_str} 01:00:00+00:00"
            post_ib_end = f"{td_str} 04:00:00+00:00"
            post_bars = con.execute(post_ib_query, [post_ib_start, post_ib_end]).fetchdf()

            if post_bars.empty:
                ib_data[td_str] = {
                    'ib_high': ib_high, 'ib_low': ib_low,
                    'ib_break_dir': None, 'ib_break_ts': None,
                }
                continue

            post_bars['ts_utc'] = pd.to_datetime(post_bars['ts_utc'], utc=True)

            # Find first close above IB high or below IB low
            ib_break_dir = None
            ib_break_ts = None
            for _, bar in post_bars.iterrows():
                if bar['close'] > ib_high:
                    ib_break_dir = 'long'
                    ib_break_ts = bar['ts_utc']
                    break
                elif bar['close'] < ib_low:
                    ib_break_dir = 'short'
                    ib_break_ts = bar['ts_utc']
                    break

            ib_data[td_str] = {
                'ib_high': ib_high, 'ib_low': ib_low,
                'ib_break_dir': ib_break_dir, 'ib_break_ts': ib_break_ts,
            }

    con.close()
    print(f"Computed IB for {len(ib_data)} days")
    print()

    # Merge IB data into outcomes
    outcomes['td_str'] = outcomes['trading_day'].apply(lambda x: str(x)[:10])
    outcomes['ib_break_dir'] = outcomes['td_str'].map(lambda x: ib_data.get(x, {}).get('ib_break_dir'))
    outcomes['ib_break_ts'] = outcomes['td_str'].map(lambda x: ib_data.get(x, {}).get('ib_break_ts'))

    # Classification:
    # 1. IB broke BEFORE 1100 entry -> we know direction at entry time (zero lookahead)
    # 2. IB broke AFTER 1100 entry -> can't use as entry filter
    # 3. IB never broke -> no signal
    outcomes['ib_known_at_entry'] = False
    outcomes['ib_aligned'] = None

    for idx, row in outcomes.iterrows():
        ib_dir = row['ib_break_dir']
        ib_ts = row['ib_break_ts']
        entry_ts = row['entry_ts']

        if ib_dir is None or pd.isna(entry_ts):
            continue

        if ib_ts is not None and ib_ts <= entry_ts:
            outcomes.at[idx, 'ib_known_at_entry'] = True
            outcomes.at[idx, 'ib_aligned'] = (ib_dir == row['break_dir_1100'])

    # ================================================================
    # Stats
    # ================================================================
    has_ib = outcomes[outcomes['ib_break_dir'].notna()]
    known = outcomes[outcomes['ib_known_at_entry'] == True]
    unknown = outcomes[(outcomes['ib_break_dir'].notna()) & (outcomes['ib_known_at_entry'] == False)]
    no_break = outcomes[outcomes['ib_break_dir'].isna()]

    print("=" * 70)
    print("IB BREAK TIMING RELATIVE TO 1100 ENTRY")
    print("=" * 70)
    print(f"  IB broke before 1100 entry (usable):  {len(known)} trades ({len(known)/len(outcomes):.0%})")
    print(f"  IB broke after 1100 entry (too late):  {len(unknown)} trades ({len(unknown)/len(outcomes):.0%})")
    print(f"  IB never broke (no signal):            {len(no_break)} trades ({len(no_break)/len(outcomes):.0%})")

    # ================================================================
    # Alignment analysis (only trades where IB was known at entry)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("IB ALIGNMENT (zero-lookahead: IB broke BEFORE 1100 entry)")
    print("=" * 70)

    aligned = known[known['ib_aligned'] == True]
    opposed = known[known['ib_aligned'] == False]

    print(f"\n  Aligned: {len(aligned)} trades ({len(aligned)/len(known):.0%})")
    print(f"  Opposed: {len(opposed)} trades ({len(opposed)/len(known):.0%})")

    for rr in [1.5, 2.0, 2.5, 3.0]:
        for cb in [3, 4, 5]:
            subset = known[(known['rr_target'] == rr) & (known['confirm_bars'] == cb)]
            if len(subset) < 20:
                continue
            a = subset[subset['ib_aligned'] == True]
            o = subset[subset['ib_aligned'] == False]

            print(f"\n  E1 CB{cb} RR{rr} G4+ (IB known at entry):")
            _print_stats("  Aligned", a)
            _print_stats("  Opposed", o)
            _print_stats("  All", subset)

    # G6+ and G8+ on best variants
    for min_size, flabel in [(6.0, "G6+"), (8.0, "G8+")]:
        for rr in [2.0, 2.5]:
            for cb in [4]:
                subset = known[(known['rr_target'] == rr) & (known['confirm_bars'] == cb)
                               & (known['orb_size_1100'] >= min_size)]
                if len(subset) < 15:
                    continue
                a = subset[subset['ib_aligned'] == True]
                o = subset[subset['ib_aligned'] == False]
                print(f"\n  E1 CB{cb} RR{rr} {flabel} (IB known at entry):")
                _print_stats("  Aligned", a)
                _print_stats("  Opposed", o)
                _print_stats("  All", subset)

    # ================================================================
    # Correlation: does 1100 break = IB break?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CORRELATION: Does 1100 ORB break direction = IB break direction?")
    print("=" * 70)

    has_both = has_ib.drop_duplicates('trading_day')
    same_dir = (has_both['ib_break_dir'] == has_both['break_dir_1100']).sum()
    total = len(has_both)
    print(f"  Days with both IB and 1100 break: {total}")
    print(f"  Same direction: {same_dir} ({same_dir/total:.0%})")
    print(f"  Opposite direction: {total - same_dir} ({(total-same_dir)/total:.0%})")

    # ================================================================
    # Yearly breakdown for best setup
    # ================================================================
    print(f"\n{'=' * 70}")
    print("YEARLY: E1 CB4 RR2.5 G4+, IB known at entry")
    print("=" * 70)

    yearly_sub = known[(known['rr_target'] == 2.5) & (known['confirm_bars'] == 4)]
    yearly_sub = yearly_sub.copy()
    yearly_sub['year'] = yearly_sub['trading_day'].apply(lambda x: int(str(x)[:4]))

    print(f"  {'Year':<6} {'N_aln':>6} {'WR_aln':>7} {'ExpR_aln':>9} {'N_opp':>6} {'WR_opp':>7} {'ExpR_opp':>9} {'Delta':>7}")
    print("  " + "-" * 65)
    for year in sorted(yearly_sub['year'].unique()):
        yr = yearly_sub[yearly_sub['year'] == year]
        a = yr[yr['ib_aligned'] == True]
        o = yr[yr['ib_aligned'] == False]
        a_pnl = _clean(a['pnl_r'])
        o_pnl = _clean(o['pnl_r'])
        a_wr = (a_pnl > 0).mean() if len(a_pnl) > 0 else 0
        o_wr = (o_pnl > 0).mean() if len(o_pnl) > 0 else 0
        a_expr = a_pnl.mean() if len(a_pnl) > 0 else 0
        o_expr = o_pnl.mean() if len(o_pnl) > 0 else 0
        delta = a_expr - o_expr if len(a_pnl) > 0 and len(o_pnl) > 0 else 0
        print(f"  {year:<6} {len(a_pnl):>6} {a_wr:>6.1%} {a_expr:>+9.3f} "
              f"{len(o_pnl):>6} {o_wr:>6.1%} {o_expr:>+9.3f} {delta:>+7.3f}")

    # Also do RR2.0
    print(f"\n  YEARLY: E1 CB4 RR2.0 G4+, IB known at entry")
    yearly_sub2 = known[(known['rr_target'] == 2.0) & (known['confirm_bars'] == 4)].copy()
    yearly_sub2['year'] = yearly_sub2['trading_day'].apply(lambda x: int(str(x)[:4]))

    print(f"  {'Year':<6} {'N_aln':>6} {'WR_aln':>7} {'ExpR_aln':>9} {'N_opp':>6} {'WR_opp':>7} {'ExpR_opp':>9} {'Delta':>7}")
    print("  " + "-" * 65)
    for year in sorted(yearly_sub2['year'].unique()):
        yr = yearly_sub2[yearly_sub2['year'] == year]
        a = yr[yr['ib_aligned'] == True]
        o = yr[yr['ib_aligned'] == False]
        a_pnl = _clean(a['pnl_r'])
        o_pnl = _clean(o['pnl_r'])
        a_wr = (a_pnl > 0).mean() if len(a_pnl) > 0 else 0
        o_wr = (o_pnl > 0).mean() if len(o_pnl) > 0 else 0
        a_expr = a_pnl.mean() if len(a_pnl) > 0 else 0
        o_expr = o_pnl.mean() if len(o_pnl) > 0 else 0
        delta = a_expr - o_expr if len(a_pnl) > 0 and len(o_pnl) > 0 else 0
        print(f"  {year:<6} {len(a_pnl):>6} {a_wr:>6.1%} {a_expr:>+9.3f} "
              f"{len(o_pnl):>6} {o_wr:>6.1%} {o_expr:>+9.3f} {delta:>+7.3f}")


def _clean(series):
    return np.array([float(x) for x in series if x is not None and not np.isnan(float(x))])


def _print_stats(label, group):
    pnl = _clean(group['pnl_r'])
    n = len(pnl)
    if n == 0:
        print(f"    {label:<20} N={n:>4}")
        return
    wins = (pnl > 0).sum()
    wr = wins / n
    expr = pnl.mean()
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    total_r = pnl.sum()
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak).min()
    print(f"    {label:<20} N={n:>4}  WR={wr:>5.1%}  ExpR={expr:>+.3f}  "
          f"Sharpe={sharpe:>+.3f}  TotalR={total_r:>+.1f}  MaxDD={dd:>+.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    run_analysis(args.db_path)
