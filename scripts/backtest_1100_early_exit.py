"""
Backtest: 1100 timed early exit at multiple thresholds.

For each 1100 trade with an entry:
  1. Fetch bars post-entry
  2. At threshold minutes, check bar close vs entry
  3. If MTM < 0, override outcome to early_exit with actual pnl_r
  4. If stop/target hit BEFORE threshold, keep original outcome
  5. If MTM >= 0 at threshold, keep original outcome (let winner run)

Compare baseline (no early exit) vs early exit at 15/30/45 min.
Report: N, WR, ExpR, Sharpe, MaxDD, TotalR -- per strategy variant and family.
Include yearly breakdown for the best threshold.

Uses cost model for honest R-multiple calculation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, to_r_multiple


def run_backtest(db_path=None):
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost = get_cost_spec("MGC")
    con = duckdb.connect(str(db_path), read_only=True)

    # Get all 1100 E1 outcomes with entries, G4+ filter
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
        o.target_price,
        o.exit_ts,
        o.mae_r,
        o.mfe_r,
        df.orb_1100_break_dir AS break_dir,
        df.orb_1100_size AS orb_size
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
    if 'exit_ts' in outcomes.columns:
        outcomes['exit_ts'] = pd.to_datetime(outcomes['exit_ts'], utc=True)

    print(f"Total 1100 E1 G4+ trades with entries: {len(outcomes)}")
    print(f"Unique days: {outcomes['trading_day'].nunique()}")
    print(f"Date range: {outcomes['trading_day'].min()} to {outcomes['trading_day'].max()}")
    print()

    # Pre-load all relevant bars (01:00 - 04:00 UTC for all trading days)
    # This is much faster than querying per-trade
    unique_days = outcomes['trading_day'].unique()
    print(f"Loading bars for {len(unique_days)} trading days...")

    all_bars = {}
    batch_size = 50
    for i in range(0, len(unique_days), batch_size):
        batch = unique_days[i:i+batch_size]
        day_list = ", ".join(f"'{str(d)[:10]}'" for d in batch)
        bar_query = f"""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = 'MGC'
          AND ts_utc::DATE IN ({day_list})
          AND EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'UTC') BETWEEN 1 AND 5
        ORDER BY ts_utc
        """
        batch_bars = con.execute(bar_query).fetchdf()
        if not batch_bars.empty:
            batch_bars['ts_utc'] = pd.to_datetime(batch_bars['ts_utc'], utc=True)
            batch_bars['day_key'] = batch_bars['ts_utc'].dt.strftime('%Y-%m-%d')
            for day_key, group in batch_bars.groupby('day_key'):
                all_bars[day_key] = group.sort_values('ts_utc').reset_index(drop=True)

    con.close()
    print(f"Loaded bars for {len(all_bars)} days")
    print()

    # Test thresholds
    thresholds = [15, 20, 30, 45, 60]

    # For each trade, simulate early exit at each threshold
    results = {t: [] for t in thresholds}
    results['baseline'] = []

    for _, trade in outcomes.iterrows():
        td_str = str(trade['trading_day'])[:10]
        bars = all_bars.get(td_str)

        baseline_rec = {
            'trading_day': trade['trading_day'],
            'rr_target': trade['rr_target'],
            'confirm_bars': trade['confirm_bars'],
            'outcome': trade['outcome'],
            'pnl_r': trade['pnl_r'],
            'break_dir': trade['break_dir'],
            'orb_size': trade['orb_size'],
        }
        results['baseline'].append(baseline_rec)

        entry_ts = trade['entry_ts']
        entry_price = trade['entry_price']
        stop_price = trade['stop_price']
        target_price = trade['target_price']
        break_dir = trade['break_dir']

        for threshold in thresholds:
            rec = baseline_rec.copy()

            if bars is None or pd.isna(entry_ts):
                results[threshold].append(rec)
                continue

            # Get bars strictly after entry
            post_entry = bars[bars['ts_utc'] > entry_ts].sort_values('ts_utc')
            if post_entry.empty:
                results[threshold].append(rec)
                continue

            # Find the threshold bar
            elapsed = (post_entry['ts_utc'] - entry_ts).dt.total_seconds() / 60.0
            threshold_mask = elapsed.values >= threshold

            if not threshold_mask.any():
                # Not enough bars to reach threshold -- keep original
                results[threshold].append(rec)
                continue

            threshold_idx = int(np.argmax(threshold_mask))

            # Check if stop or target hit BEFORE threshold bar
            pre_bars = post_entry.iloc[:threshold_idx]
            if not pre_bars.empty:
                highs = pre_bars['high'].values
                lows = pre_bars['low'].values
                if break_dir == 'long':
                    prior_hit = (highs >= target_price).any() or (lows <= stop_price).any()
                else:
                    prior_hit = (lows <= target_price).any() or (highs >= stop_price).any()

                if prior_hit:
                    # Stop/target hit before threshold -- keep original
                    results[threshold].append(rec)
                    continue

            # Also check the threshold bar itself for stop/target
            t_bar = post_entry.iloc[threshold_idx]
            if break_dir == 'long':
                target_hit_on_tbar = t_bar['high'] >= target_price
                stop_hit_on_tbar = t_bar['low'] <= stop_price
            else:
                target_hit_on_tbar = t_bar['low'] <= target_price
                stop_hit_on_tbar = t_bar['high'] >= stop_price

            if target_hit_on_tbar or stop_hit_on_tbar:
                # Target or stop hit on the threshold bar itself
                # Ambiguous -- but the early exit check uses bar CLOSE, not H/L
                # If close is negative, exit at close regardless of H/L
                # This matches the production logic in outcome_builder
                pass  # Fall through to MTM check below

            # Check MTM at threshold bar close
            close_at_threshold = float(t_bar['close'])
            if break_dir == 'long':
                mtm_points = close_at_threshold - entry_price
            else:
                mtm_points = entry_price - close_at_threshold

            if mtm_points < 0:
                # Early exit: loser at threshold
                pnl_r = round(
                    to_r_multiple(cost, entry_price, stop_price, mtm_points), 4
                )
                rec['outcome'] = 'early_exit'
                rec['pnl_r'] = pnl_r

            # MTM >= 0: winner, keep original outcome

            results[threshold].append(rec)

    # ================================================================
    # Report results
    # ================================================================

    print("=" * 80)
    print("FAMILY-LEVEL COMPARISON: Baseline vs Early Exit Thresholds")
    print("  Family = (RR, CB) averaged. 1100 E1 G4+.")
    print("=" * 80)

    # First: overall (all RR/CB pooled)
    print("\n--- ALL VARIANTS POOLED ---")
    _compare_header()
    _print_comparison("Baseline", results['baseline'])
    for t in thresholds:
        _print_comparison(f"Exit {t}m", results[t])

    # Per RR/CB family
    for rr in [1.5, 2.0, 2.5, 3.0]:
        for cb in [3, 4, 5]:
            baseline_sub = [r for r in results['baseline']
                           if r['rr_target'] == rr and r['confirm_bars'] == cb]
            if len(baseline_sub) < 30:
                continue

            print(f"\n--- E1 CB{cb} RR{rr} G4+ ---")
            _compare_header()
            _print_comparison("Baseline", baseline_sub)
            for t in thresholds:
                sub = [r for r in results[t]
                       if r['rr_target'] == rr and r['confirm_bars'] == cb]
                _print_comparison(f"Exit {t}m", sub)

    # ================================================================
    # Best threshold: yearly breakdown
    # ================================================================
    # Find best threshold by Sharpe improvement
    baseline_sharpe = _sharpe([r['pnl_r'] for r in results['baseline']])
    best_t = max(thresholds, key=lambda t: _sharpe([r['pnl_r'] for r in results[t]]) - baseline_sharpe)

    print(f"\n{'=' * 80}")
    print(f"YEARLY BREAKDOWN: Best threshold = {best_t}m")
    print(f"  E1 CB4 RR2.5 G4+ (reference strat)")
    print(f"{'=' * 80}")

    baseline_cb4_rr25 = [r for r in results['baseline']
                          if r['rr_target'] == 2.5 and r['confirm_bars'] == 4]
    exit_cb4_rr25 = [r for r in results[best_t]
                      if r['rr_target'] == 2.5 and r['confirm_bars'] == 4]

    _yearly_comparison(baseline_cb4_rr25, exit_cb4_rr25, best_t)

    # Also show CB4 RR2.0
    print(f"\n  E1 CB4 RR2.0 G4+:")
    baseline_cb4_rr20 = [r for r in results['baseline']
                          if r['rr_target'] == 2.0 and r['confirm_bars'] == 4]
    exit_cb4_rr20 = [r for r in results[best_t]
                      if r['rr_target'] == 2.0 and r['confirm_bars'] == 4]
    _yearly_comparison(baseline_cb4_rr20, exit_cb4_rr20, best_t)

    # ================================================================
    # Early exit impact: what % of trades get cut, avg loss saved
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"EARLY EXIT IMPACT ANALYSIS (threshold={best_t}m)")
    print(f"{'=' * 80}")

    for rr in [2.0, 2.5]:
        baseline_sub = [r for r in results['baseline']
                       if r['rr_target'] == rr and r['confirm_bars'] == 4]
        exit_sub = [r for r in results[best_t]
                   if r['rr_target'] == rr and r['confirm_bars'] == 4]

        n_total = len(baseline_sub)
        n_early = sum(1 for r in exit_sub if r['outcome'] == 'early_exit')
        n_were_loss = sum(1 for b, e in zip(baseline_sub, exit_sub)
                         if e['outcome'] == 'early_exit' and b['outcome'] == 'loss')
        n_were_win = sum(1 for b, e in zip(baseline_sub, exit_sub)
                        if e['outcome'] == 'early_exit' and b['outcome'] == 'win')
        n_were_scratch = sum(1 for b, e in zip(baseline_sub, exit_sub)
                            if e['outcome'] == 'early_exit' and b['outcome'] == 'scratch')

        # Average pnl_r of early-exited trades: baseline vs after
        early_baseline_pnl = _clean_pnl([b['pnl_r'] for b, e in zip(baseline_sub, exit_sub)
                              if e['outcome'] == 'early_exit'])
        early_exit_pnl = _clean_pnl([e['pnl_r'] for e in exit_sub if e['outcome'] == 'early_exit'])

        print(f"\n  E1 CB4 RR{rr} G4+:")
        print(f"    Total trades: {n_total}")
        print(f"    Early exits:  {n_early} ({n_early/n_total:.0%})")
        print(f"      Were losses:   {n_were_loss}")
        print(f"      Were wins:     {n_were_win}  (WINNERS KILLED)")
        print(f"      Were scratch:  {n_were_scratch}")
        if len(early_baseline_pnl) > 0:
            print(f"    Avg baseline PnL of cut trades: {early_baseline_pnl.mean():+.3f}R")
            print(f"    Avg early exit PnL:             {early_exit_pnl.mean():+.3f}R")
            print(f"    Avg R saved per cut trade:      {early_exit_pnl.mean() - early_baseline_pnl.mean():+.3f}R")

    # ================================================================
    # G6+ and G8+ subsets
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"FILTER-LEVEL COMPARISON (threshold={best_t}m, CB4 RR2.5)")
    print(f"{'=' * 80}")

    for min_size, flabel in [(4.0, "G4+"), (6.0, "G6+"), (8.0, "G8+")]:
        baseline_sub = [r for r in results['baseline']
                       if r['rr_target'] == 2.5 and r['confirm_bars'] == 4
                       and r['orb_size'] >= min_size]
        exit_sub = [r for r in results[best_t]
                   if r['rr_target'] == 2.5 and r['confirm_bars'] == 4
                   and r['orb_size'] >= min_size]
        if len(baseline_sub) < 20:
            continue
        print(f"\n  {flabel}:")
        _compare_header()
        _print_comparison("Baseline", baseline_sub)
        _print_comparison(f"Exit {best_t}m", exit_sub)


def _clean_pnl(pnl_list):
    return np.array([float(p) for p in pnl_list if p is not None and not np.isnan(float(p))])


def _sharpe(pnl_list):
    arr = _clean_pnl(pnl_list)
    if len(arr) == 0 or arr.std() == 0:
        return 0
    return arr.mean() / arr.std()


def _compare_header():
    print(f"  {'Label':<14} {'N':>5} {'EarlyX':>6} {'WR':>6} {'ExpR':>7} "
          f"{'Sharpe':>7} {'TotalR':>8} {'MaxDD':>7}")
    print("  " + "-" * 62)


def _print_comparison(label, records):
    n = len(records)
    if n == 0:
        print(f"  {label:<14} {'N/A':>5}")
        return
    pnl = _clean_pnl([r['pnl_r'] for r in records])
    if len(pnl) == 0:
        return
    n_early = sum(1 for r in records if r['outcome'] == 'early_exit')
    wins = (pnl > 0).sum()
    wr = wins / len(pnl)
    expr = pnl.mean()
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    total_r = pnl.sum()
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak).min()
    print(f"  {label:<14} {len(pnl):>5} {n_early:>6} {wr:>5.1%} {expr:>+7.3f} "
          f"{sharpe:>+7.3f} {total_r:>+8.1f} {dd:>+7.1f}")


def _yearly_comparison(baseline, exit_data, threshold):
    if not baseline:
        return

    years = sorted(set(str(r['trading_day'])[:4] for r in baseline))
    print(f"  {'Year':<6} {'N':>4} {'EarlyX':>6} "
          f"{'WR_base':>7} {'WR_exit':>7} "
          f"{'ExpR_base':>9} {'ExpR_exit':>9} {'Delta':>7}")
    print("  " + "-" * 62)

    for year in years:
        b = [r for r in baseline if str(r['trading_day'])[:4] == year]
        e = [r for r in exit_data if str(r['trading_day'])[:4] == year]
        if not b:
            continue
        b_pnl = _clean_pnl([r['pnl_r'] for r in b])
        e_pnl = _clean_pnl([r['pnl_r'] for r in e])
        n_early = sum(1 for r in e if r['outcome'] == 'early_exit')

        if len(b_pnl) == 0:
            continue

        b_wr = (b_pnl > 0).mean()
        e_wr = (e_pnl > 0).mean() if len(e_pnl) > 0 else 0
        b_expr = b_pnl.mean()
        e_expr = e_pnl.mean() if len(e_pnl) > 0 else 0
        delta = e_expr - b_expr

        print(f"  {year:<6} {len(b_pnl):>4} {n_early:>6} "
              f"{b_wr:>6.1%} {e_wr:>6.1%} "
              f"{b_expr:>+9.3f} {e_expr:>+9.3f} {delta:>+7.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    run_backtest(args.db_path)
