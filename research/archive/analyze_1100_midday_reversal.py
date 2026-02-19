"""
Research: Does price reverse toward Asia open around 11:30-12:30 Brisbane?

Hypothesis: After the 1100 ORB break, price tends to reverse back toward
the Asia session open (09:00 Brisbane open price) around 11:30-12:30.

Method:
  1. Get 1100 break direction and entry price
  2. Track MTM at 11:30, 12:00, 12:30, 13:00 (30-min intervals)
  3. Check if MTM deteriorates (reversal) vs continues (trend)
  4. Check if price crosses back through Asia open

All timestamps Brisbane local = UTC+10.
  11:30 Bris = 01:30 UTC
  12:00 Bris = 02:00 UTC
  12:30 Bris = 02:30 UTC
  13:00 Bris = 03:00 UTC
  Asia open (09:00 Bris) = 23:00 UTC prev day
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb
import numpy as np
import pandas as pd
from pipeline.paths import GOLD_DB_PATH

def run_analysis(db_path=None):
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path), read_only=True)

    # Get 1100 outcomes with entry price, plus bars for MTM tracking
    # 1100 Brisbane = 01:00 UTC. ORB window = 01:00-01:05 UTC.
    # Break + confirm + entry happens after 01:05.
    # We want to track price at 01:30, 02:00, 02:30, 03:00 UTC
    query = """
    SELECT
        o.trading_day,
        o.rr_target,
        o.confirm_bars,
        o.entry_model,
        o.outcome,
        o.pnl_r,
        o.entry_price,
        o.entry_ts,
        o.stop_price,
        df.orb_1100_break_dir AS break_dir,
        df.orb_1100_size AS orb_size,
        df.daily_open AS asia_open,
        df.session_asia_high,
        df.session_asia_low
    FROM orb_outcomes o
    JOIN daily_features df
      ON o.trading_day = df.trading_day
      AND df.symbol = 'MGC' AND df.orb_minutes = 5
    WHERE o.symbol = 'MGC'
      AND o.orb_label = '1100'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E1'
      AND o.confirm_bars = 4
      AND o.rr_target = 2.5
      AND o.entry_price IS NOT NULL
      AND df.orb_1100_size >= 4.0
    ORDER BY o.trading_day
    """
    outcomes = con.execute(query).fetchdf()

    if outcomes.empty:
        print("No 1100 outcomes found.")
        con.close()
        return

    print(f"1100 E1 CB4 RR2.5 G4+ trades with entries: {len(outcomes)}")
    print(f"Date range: {outcomes['trading_day'].min()} to {outcomes['trading_day'].max()}")
    print()

    # For each trade day, get bar closes at checkpoint times
    checkpoints_utc_offsets = {
        "11:15": (1, 15),   # 15 min after ORB (quick check)
        "11:30": (1, 30),
        "12:00": (2, 0),
        "12:30": (2, 30),
        "13:00": (3, 0),
    }

    results = []
    for _, row in outcomes.iterrows():
        td = row['trading_day']
        entry_price = row['entry_price']
        break_dir = row['break_dir']
        asia_open = row['asia_open']

        # Get bars for this day around the checkpoint times
        bar_query = """
        SELECT ts_utc, close
        FROM bars_1m
        WHERE symbol = 'MGC'
          AND ts_utc >= ?::TIMESTAMPTZ
          AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc
        """
        # 01:00 UTC to 04:00 UTC covers our window
        td_str = str(td)[:10]
        start_ts = f"{td_str} 01:00:00+00:00"
        end_ts = f"{td_str} 04:00:00+00:00"
        bars = con.execute(bar_query, [start_ts, end_ts]).fetchdf()

        if bars.empty:
            continue

        bars['ts_utc'] = pd.to_datetime(bars['ts_utc'], utc=True)

        rec = {
            'trading_day': td,
            'entry_price': entry_price,
            'break_dir': break_dir,
            'asia_open': asia_open,
            'outcome': row['outcome'],
            'pnl_r': row['pnl_r'],
            'orb_size': row['orb_size'],
        }

        # Get close at each checkpoint (nearest bar at or before checkpoint)
        for label, (hour, minute) in checkpoints_utc_offsets.items():
            checkpoint_ts = pd.Timestamp(f"{td}T{hour:02d}:{minute:02d}:00+00:00")
            bars_before = bars[bars['ts_utc'] <= checkpoint_ts]
            if not bars_before.empty:
                close_at = float(bars_before.iloc[-1]['close'])
                if break_dir == 'long':
                    mtm = close_at - entry_price
                else:
                    mtm = entry_price - close_at
                rec[f'mtm_{label}'] = mtm
                rec[f'close_{label}'] = close_at
            else:
                rec[f'mtm_{label}'] = None
                rec[f'close_{label}'] = None

        results.append(rec)

    con.close()

    df = pd.DataFrame(results)
    print(f"Trades with bar data: {len(df)}")
    print()

    # ================================================================
    # MTM trajectory: does it deteriorate after entry?
    # ================================================================
    print("=" * 70)
    print("MTM TRAJECTORY (points from entry, in break direction)")
    print("  Positive = still in profit, Negative = underwater")
    print("=" * 70)

    for label in ["11:15", "11:30", "12:00", "12:30", "13:00"]:
        col = f'mtm_{label}'
        valid = df[df[col].notna()]
        if valid.empty:
            continue
        mtm = valid[col]
        pct_positive = (mtm > 0).mean()
        pct_negative = (mtm < 0).mean()
        print(f"  {label} Bris:  N={len(valid):>4}  "
              f"Mean MTM={mtm.mean():>+.2f}  Median={mtm.median():>+.2f}  "
              f"+ve={pct_positive:.0%}  -ve={pct_negative:.0%}")

    # ================================================================
    # Reversal detection: does MTM go from positive to negative?
    # ================================================================
    print("\n" + "=" * 70)
    print("REVERSAL DETECTION")
    print("  How often does a trade that's winning at 11:15 become losing by 12:30?")
    print("=" * 70)

    both = df[df['mtm_11:15'].notna() & df['mtm_12:30'].notna()].copy()
    winning_early = both[both['mtm_11:15'] > 0]
    losing_early = both[both['mtm_11:15'] <= 0]

    if len(winning_early) > 0:
        reversed_by_1230 = (winning_early['mtm_12:30'] <= 0).sum()
        still_winning = (winning_early['mtm_12:30'] > 0).sum()
        print(f"\n  Winning at 11:15 (N={len(winning_early)}):")
        print(f"    Still winning at 12:30: {still_winning} ({still_winning/len(winning_early):.0%})")
        print(f"    Reversed by 12:30:      {reversed_by_1230} ({reversed_by_1230/len(winning_early):.0%})")

    if len(losing_early) > 0:
        recovered_by_1230 = (losing_early['mtm_12:30'] > 0).sum()
        still_losing = (losing_early['mtm_12:30'] <= 0).sum()
        print(f"\n  Losing at 11:15 (N={len(losing_early)}):")
        print(f"    Recovered by 12:30: {recovered_by_1230} ({recovered_by_1230/len(losing_early):.0%})")
        print(f"    Still losing:       {still_losing} ({still_losing/len(losing_early):.0%})")

    # ================================================================
    # Asia open reversion: does price cross back through asia open?
    # ================================================================
    print("\n" + "=" * 70)
    print("ASIA OPEN REVERSION")
    print("  Does price cross back through Asia open (09:00 Brisbane) by 12:30?")
    print("=" * 70)

    has_asia = df[df['asia_open'].notna() & df['close_12:30'].notna()].copy()
    if len(has_asia) > 0:
        for direction in ['long', 'short']:
            sub = has_asia[has_asia['break_dir'] == direction]
            if sub.empty:
                continue
            if direction == 'long':
                # Long break: entry above asia open. Reversal = price drops below asia open
                started_above = sub[sub['entry_price'] > sub['asia_open']]
                if len(started_above) > 0:
                    crossed_back = (started_above['close_12:30'] < started_above['asia_open']).sum()
                    print(f"\n  {direction.upper()} breaks (entry above Asia open): N={len(started_above)}")
                    print(f"    Price below Asia open by 12:30: {crossed_back} ({crossed_back/len(started_above):.0%})")
            else:
                started_below = sub[sub['entry_price'] < sub['asia_open']]
                if len(started_below) > 0:
                    crossed_back = (started_below['close_12:30'] > started_below['asia_open']).sum()
                    print(f"\n  {direction.upper()} breaks (entry below Asia open): N={len(started_below)}")
                    print(f"    Price above Asia open by 12:30: {crossed_back} ({crossed_back/len(started_below):.0%})")

    # ================================================================
    # Mean MTM by outcome: winners vs losers trajectory
    # ================================================================
    print("\n" + "=" * 70)
    print("MTM TRAJECTORY BY FINAL OUTCOME")
    print("=" * 70)

    for outcome_type in ['win', 'loss', 'scratch']:
        sub = df[df['outcome'] == outcome_type]
        if sub.empty:
            continue
        print(f"\n  {outcome_type.upper()} trades (N={len(sub)}):")
        for label in ["11:15", "11:30", "12:00", "12:30", "13:00"]:
            col = f'mtm_{label}'
            valid = sub[sub[col].notna()]
            if valid.empty:
                continue
            mtm = valid[col]
            print(f"    {label}: Mean={mtm.mean():>+.2f}  Median={mtm.median():>+.2f}  "
                  f"+ve={( mtm > 0).mean():.0%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    run_analysis(args.db_path)
