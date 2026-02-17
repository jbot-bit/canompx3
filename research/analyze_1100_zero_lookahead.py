"""
Research: Zero-lookahead signals for 1100 session.

At 1100 entry time, what do we ALREADY know that predicts outcome?

Signals tested (all known before 1100 ORB forms):
  1. 0900 break direction alignment (0900 broke same way as 1100)
  2. Gap direction (overnight gap up/down matches 1100 break)
  3. Prior day trend (prev close > prev open = up day)
  4. ATR regime (high vol vs low vol, using ATR(20) median split)
  5. 0900 ORB size regime (was the 0900 ORB big or small?)

Methodology:
  - Use G4+ filter on 1100 (where edge lives)
  - E1 CB4 RR2.5 as reference strat (top validated)
  - Also test E1 CB4 RR2.0 for robustness
  - Split each signal into buckets, report N/WR/ExpR/Sharpe
  - Family-level: average across RR variants
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb
import numpy as np
from pipeline.paths import GOLD_DB_PATH

def run_analysis(db_path=None):
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path), read_only=True)

    # Get 1100 outcomes joined with daily features for context signals
    # 0900 break direction is zero-lookahead for 1100 (happens 2 hours earlier)
    query = """
    WITH day_context AS (
        SELECT
            df.trading_day,
            df.orb_0900_break_dir AS dir_0900,
            df.orb_1100_break_dir AS dir_1100,
            df.orb_0900_size AS size_0900,
            df.orb_1100_size AS size_1100,
            df.gap_open_points,
            df.atr_20,
            df.daily_open,
            df.daily_close
        FROM daily_features df
        WHERE df.symbol = 'MGC'
          AND df.orb_minutes = 5
          AND df.orb_1100_break_dir IS NOT NULL
    ),
    prev_day AS (
        SELECT
            d1.trading_day,
            d1.dir_0900,
            d1.dir_1100,
            d1.size_0900,
            d1.size_1100,
            d1.gap_open_points,
            d1.atr_20,
            -- Prior day trend: use LAG
            LAG(d1.daily_close) OVER (ORDER BY d1.trading_day) AS prev_close,
            LAG(d1.daily_open) OVER (ORDER BY d1.trading_day) AS prev_open
        FROM day_context d1
    )
    SELECT
        o.trading_day,
        o.orb_label,
        o.rr_target,
        o.confirm_bars,
        o.entry_model,
        o.outcome,
        o.pnl_r,
        o.entry_price,
        o.stop_price,
        p.dir_0900,
        p.dir_1100,
        p.size_0900,
        p.size_1100,
        p.gap_open_points,
        p.atr_20,
        p.prev_close,
        p.prev_open
    FROM orb_outcomes o
    JOIN prev_day p ON o.trading_day = p.trading_day
    WHERE o.symbol = 'MGC'
      AND o.orb_label = '1100'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E1'
      AND o.outcome IS NOT NULL
      AND p.size_1100 >= 4.0
    ORDER BY o.trading_day
    """

    df = con.execute(query).fetchdf()
    con.close()

    if df.empty:
        print("No data found.")
        return

    print(f"Total 1100 E1 G4+ outcomes: {len(df)}")
    print(f"Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")
    print(f"Unique days: {df['trading_day'].nunique()}")
    print()

    # ================================================================
    # Signal 1: 0900 direction alignment
    # ================================================================
    print("=" * 70)
    print("SIGNAL 1: 0900 Break Direction Alignment")
    print("  (Does 0900 break direction match 1100 break direction?)")
    print("=" * 70)

    # Classify alignment
    has_0900 = df[df['dir_0900'].notna()].copy()
    has_0900['aligned'] = has_0900['dir_0900'] == has_0900['dir_1100']

    for rr in [2.0, 2.5]:
        for cb in [4]:
            subset = has_0900[(has_0900['rr_target'] == rr) & (has_0900['confirm_bars'] == cb)]
            if subset.empty:
                continue
            print(f"\n  E1 CB{cb} RR{rr} G4+:")
            for label, group in [("Aligned", subset[subset['aligned']]),
                                  ("Opposed", subset[~subset['aligned']]),
                                  ("All", subset)]:
                _print_stats(label, group)

    # Also test G6+ and G8+
    for min_size, label in [(6.0, "G6+"), (8.0, "G8+")]:
        sized = has_0900[has_0900['size_1100'] >= min_size]
        for rr in [2.5]:
            for cb in [4]:
                subset = sized[(sized['rr_target'] == rr) & (sized['confirm_bars'] == cb)]
                if subset.empty:
                    continue
                print(f"\n  E1 CB{cb} RR{rr} {label}:")
                for slabel, group in [("Aligned", subset[subset['aligned']]),
                                       ("Opposed", subset[~subset['aligned']]),
                                       ("All", subset)]:
                    _print_stats(slabel, group)

    # ================================================================
    # Signal 2: Gap direction alignment
    # ================================================================
    print("\n" + "=" * 70)
    print("SIGNAL 2: Overnight Gap Direction Alignment")
    print("  (Does gap direction match 1100 break direction?)")
    print("=" * 70)

    has_gap = df[df['gap_open_points'].notna()].copy()
    has_gap['gap_dir'] = np.where(has_gap['gap_open_points'] > 0, 'long', 'short')
    has_gap['gap_aligned'] = has_gap['gap_dir'] == has_gap['dir_1100']

    for rr in [2.0, 2.5]:
        subset = has_gap[(has_gap['rr_target'] == rr) & (has_gap['confirm_bars'] == 4)]
        if subset.empty:
            continue
        print(f"\n  E1 CB4 RR{rr} G4+:")
        for label, group in [("Gap Aligned", subset[subset['gap_aligned']]),
                              ("Gap Opposed", subset[~subset['gap_aligned']]),
                              ("All", subset)]:
            _print_stats(label, group)

    # ================================================================
    # Signal 3: Prior day trend
    # ================================================================
    print("\n" + "=" * 70)
    print("SIGNAL 3: Prior Day Trend")
    print("  (Did yesterday close > open? Does that match 1100 break?)")
    print("=" * 70)

    has_prev = df[df['prev_close'].notna() & df['prev_open'].notna()].copy()
    has_prev['prev_trend'] = np.where(has_prev['prev_close'] > has_prev['prev_open'], 'long', 'short')
    has_prev['trend_aligned'] = has_prev['prev_trend'] == has_prev['dir_1100']

    for rr in [2.0, 2.5]:
        subset = has_prev[(has_prev['rr_target'] == rr) & (has_prev['confirm_bars'] == 4)]
        if subset.empty:
            continue
        print(f"\n  E1 CB4 RR{rr} G4+:")
        for label, group in [("Trend Aligned", subset[subset['trend_aligned']]),
                              ("Trend Opposed", subset[~subset['trend_aligned']]),
                              ("All", subset)]:
            _print_stats(label, group)

    # ================================================================
    # Signal 4: ATR regime (median split)
    # ================================================================
    print("\n" + "=" * 70)
    print("SIGNAL 4: ATR(20) Regime (Median Split)")
    print("  (High vol vs low vol day)")
    print("=" * 70)

    has_atr = df[df['atr_20'].notna()].copy()
    atr_median = has_atr['atr_20'].median()
    has_atr['high_vol'] = has_atr['atr_20'] >= atr_median
    print(f"  ATR(20) median: {atr_median:.2f}")

    for rr in [2.0, 2.5]:
        subset = has_atr[(has_atr['rr_target'] == rr) & (has_atr['confirm_bars'] == 4)]
        if subset.empty:
            continue
        print(f"\n  E1 CB4 RR{rr} G4+:")
        for label, group in [("High Vol", subset[subset['high_vol']]),
                              ("Low Vol", subset[~subset['high_vol']]),
                              ("All", subset)]:
            _print_stats(label, group)

    # ================================================================
    # Signal 5: 0900 ORB size (big 0900 = trending day?)
    # ================================================================
    print("\n" + "=" * 70)
    print("SIGNAL 5: 0900 ORB Size (Big 0900 = Trend Day?)")
    print("=" * 70)

    has_0900_size = df[df['size_0900'].notna()].copy()
    size_0900_median = has_0900_size['size_0900'].median()
    has_0900_size['big_0900'] = has_0900_size['size_0900'] >= size_0900_median
    print(f"  0900 ORB size median: {size_0900_median:.2f}")

    for rr in [2.0, 2.5]:
        subset = has_0900_size[(has_0900_size['rr_target'] == rr) & (has_0900_size['confirm_bars'] == 4)]
        if subset.empty:
            continue
        print(f"\n  E1 CB4 RR{rr} G4+:")
        for label, group in [("Big 0900", subset[subset['big_0900']]),
                              ("Small 0900", subset[~subset['big_0900']]),
                              ("All", subset)]:
            _print_stats(label, group)

    # ================================================================
    # Combined: 0900 aligned + high vol
    # ================================================================
    print("\n" + "=" * 70)
    print("COMBINED: 0900 Aligned + High Vol")
    print("=" * 70)

    combo = has_0900[has_0900['atr_20'].notna()].copy()
    combo['high_vol'] = combo['atr_20'] >= atr_median
    combo['both'] = combo['aligned'] & combo['high_vol']
    combo['neither'] = ~combo['aligned'] & ~combo['high_vol']

    for rr in [2.0, 2.5]:
        subset = combo[(combo['rr_target'] == rr) & (combo['confirm_bars'] == 4)]
        if subset.empty:
            continue
        print(f"\n  E1 CB4 RR{rr} G4+:")
        for label, group in [
            ("Aligned+HiVol", subset[subset['both']]),
            ("Aligned+LoVol", subset[subset['aligned'] & ~subset['high_vol']]),
            ("Opposed+HiVol", subset[~subset['aligned'] & subset['high_vol']]),
            ("Opposed+LoVol", subset[subset['neither']]),
            ("All", subset),
        ]:
            _print_stats(label, group)

    # ================================================================
    # Yearly breakdown for best signal
    # ================================================================
    print("\n" + "=" * 70)
    print("YEARLY BREAKDOWN: 0900 Alignment (E1 CB4 RR2.5 G4+)")
    print("=" * 70)

    yearly_data = has_0900[(has_0900['rr_target'] == 2.5) & (has_0900['confirm_bars'] == 4)].copy()
    yearly_data['year'] = yearly_data['trading_day'].apply(lambda x: x.year)

    print(f"\n  {'Year':<6} {'N_aln':>6} {'WR_aln':>7} {'ExpR_aln':>9} {'N_opp':>6} {'WR_opp':>7} {'ExpR_opp':>9} {'Delta':>7}")
    print("  " + "-" * 62)
    for year in sorted(yearly_data['year'].unique()):
        yr = yearly_data[yearly_data['year'] == year]
        aln = yr[yr['aligned']]
        opp = yr[~yr['aligned']]
        aln_n = len(aln)
        opp_n = len(opp)
        aln_wr = aln['pnl_r'].apply(lambda x: x > 0).mean() if aln_n > 0 else 0
        opp_wr = opp['pnl_r'].apply(lambda x: x > 0).mean() if opp_n > 0 else 0
        aln_expr = aln['pnl_r'].mean() if aln_n > 0 else 0
        opp_expr = opp['pnl_r'].mean() if opp_n > 0 else 0
        delta = aln_expr - opp_expr if aln_n > 0 and opp_n > 0 else 0
        print(f"  {year:<6} {aln_n:>6} {aln_wr:>6.1%} {aln_expr:>+9.3f} {opp_n:>6} {opp_wr:>6.1%} {opp_expr:>+9.3f} {delta:>+7.3f}")

def _print_stats(label, group):
    n = len(group)
    if n == 0:
        print(f"    {label:<20} N={n:>4}")
        return
    pnl = group['pnl_r']
    wins = (pnl > 0).sum()
    wr = wins / n
    expr = pnl.mean()
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    total_r = pnl.sum()
    # Max drawdown (cumulative)
    cum = pnl.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    print(f"    {label:<20} N={n:>4}  WR={wr:>5.1%}  ExpR={expr:>+.3f}  "
          f"Sharpe={sharpe:>+.3f}  TotalR={total_r:>+.1f}  MaxDD={dd:>+.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    run_analysis(args.db_path)
