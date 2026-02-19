"""
Multi-instrument edge hunter + regime scan.
Runs edge_hunter logic and regime analysis for MES, MNQ, MGC.
ZERO look-ahead. Correct joins. Audited.

Usage:
    python research/multi_instrument_scan.py             # all instruments
    python research/multi_instrument_scan.py --instrument MES
    python research/multi_instrument_scan.py --instrument MNQ
"""
import duckdb
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH

SAFE_JOIN = """
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
"""

REGIME_START = '2025-01-01'


def get_available_sessions(con, instrument, min_n=50):
    """Auto-detect sessions with enough data for an instrument."""
    rows = con.execute(f"""
        SELECT o.orb_label, COUNT(*) as n
        {SAFE_JOIN}
        WHERE o.symbol = '{instrument}'
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) >= {min_n}
        ORDER BY 1
    """).fetchall()
    return [r[0] for r in rows]


def audit_join(con, instrument, session, em, rr, cb, size_min):
    """Verify join doesn't inflate rows."""
    size_col = f"orb_{session}_size"

    n_outcomes = con.execute(f"""
        SELECT COUNT(*) FROM orb_outcomes
        WHERE symbol = '{instrument}' AND orb_label = '{session}'
          AND entry_model = '{em}' AND rr_target = {rr} AND confirm_bars = {cb}
          AND outcome IN ('win','loss','early_exit') AND pnl_r IS NOT NULL
    """).fetchone()[0]

    n_joined = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """).fetchone()[0]

    n_filtered = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND d.{size_col} >= {size_min}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """).fetchone()[0]

    ok = n_joined <= n_outcomes and n_filtered <= n_joined
    if not ok:
        print(f"  AUDIT FAIL: outcomes={n_outcomes} joined={n_joined} filtered={n_filtered}")
        sys.exit(1)
    return n_outcomes, n_joined, n_filtered


def get_pnl(con, instrument, session, em, rr, cb, size_min, extra=""):
    """Get pnl_r array with safe join."""
    size_col = f"orb_{session}_size"
    return con.execute(f"""
        SELECT o.pnl_r {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND d.{size_col} >= {size_min}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          {extra}
    """).fetchnumpy()['pnl_r']


# Session ordering for "earlier session" predictor tests
# Only fixed sessions have a clear temporal ordering
FIXED_SESSION_ORDER = ['0900', '1000', '1100', '1800', '2300', '0030']

def get_earlier_sessions(session):
    """Return sessions that are KNOWN to happen before the target session."""
    if session not in FIXED_SESSION_ORDER:
        return []  # dynamic sessions — can't guarantee ordering
    idx = FIXED_SESSION_ORDER.index(session)
    return FIXED_SESSION_ORDER[:idx]


def run_edge_hunter(con, instrument, sessions, size_min=4.0):
    """Run edge hunter for an instrument across its sessions."""
    print(f"\n{'='*80}")
    print(f"EDGE HUNTER — {instrument} — Zero Look-Ahead, Audited Joins")
    print(f"{'='*80}")

    for sess in sessions:
        for em in ['E1', 'E3']:
            for rr in [2.0, 2.5, 3.0]:
                cb = 2

                n_raw, n_join, n_filt = audit_join(con, instrument, sess, em, rr, cb, size_min)
                if n_filt < 50:
                    continue

                baseline = get_pnl(con, instrument, sess, em, rr, cb, size_min)
                base_expr = np.mean(baseline) if len(baseline) > 0 else 0

                print(f"\n--- {instrument} {sess} {em} RR{rr} CB{cb} G{int(size_min)}+ | N={n_filt} ExpR={base_expr:+.4f} ---")

                results = []

                # 1. Earlier-session double-breaks
                earlier = get_earlier_sessions(sess)
                for earlier_sess in earlier:
                    col = f"d.orb_{earlier_sess}_double_break"
                    # Check column exists and has data
                    try:
                        true_arr = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                           f"AND ({col}) = true")
                        false_arr = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                            f"AND ({col}) = false")
                    except Exception:
                        continue

                    if len(true_arr) >= 20 and len(false_arr) >= 20:
                        t, p = stats.ttest_ind(true_arr, false_arr, equal_var=False)
                        results.append({
                            'name': f'{earlier_sess}_dbl_break',
                            'n_true': len(true_arr), 'mean_true': np.mean(true_arr),
                            'n_false': len(false_arr), 'mean_false': np.mean(false_arr),
                            'delta': np.mean(false_arr) - np.mean(true_arr),
                            't': t, 'p': p,
                        })

                # 2. Day of week
                for dow_name, dow_val in [('Mon', 1), ('Tue', 2), ('Wed', 3), ('Thu', 4), ('Fri', 5)]:
                    on_day = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                     f"AND EXTRACT(isodow FROM o.trading_day) = {dow_val}")
                    off_day = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                      f"AND EXTRACT(isodow FROM o.trading_day) != {dow_val}")
                    if len(on_day) >= 15 and len(off_day) >= 30:
                        t, p = stats.ttest_ind(on_day, off_day, equal_var=False)
                        results.append({
                            'name': f'DOW_{dow_name}',
                            'n_true': len(on_day), 'mean_true': np.mean(on_day),
                            'n_false': len(off_day), 'mean_false': np.mean(off_day),
                            'delta': np.mean(on_day) - np.mean(off_day),
                            't': t, 'p': p,
                        })

                # 3. DST regime (for sessions that have DST columns)
                dst_sessions_us = ['0900', '0030', '2300']
                dst_sessions_uk = ['1800']
                if sess in dst_sessions_us:
                    dst_col = 'us_dst'
                elif sess in dst_sessions_uk:
                    dst_col = 'uk_dst'
                else:
                    dst_col = None

                if dst_col:
                    try:
                        dst_on = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                         f"AND d.{dst_col} = true")
                        dst_off = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                          f"AND d.{dst_col} = false")
                        if len(dst_on) >= 20 and len(dst_off) >= 20:
                            t, p = stats.ttest_ind(dst_on, dst_off, equal_var=False)
                            results.append({
                                'name': f'DST_{dst_col}',
                                'n_true': len(dst_on), 'mean_true': np.mean(dst_on),
                                'n_false': len(dst_off), 'mean_false': np.mean(dst_off),
                                'delta': np.mean(dst_on) - np.mean(dst_off),
                                't': t, 'p': p,
                            })
                    except Exception:
                        pass

                # 4. ORB size of earlier session
                for earlier_sess in earlier[:2]:
                    ecol = f"orb_{earlier_sess}_size"
                    try:
                        small = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                        f"AND d.{ecol} < 4 AND d.{ecol} IS NOT NULL")
                        big = get_pnl(con, instrument, sess, em, rr, cb, size_min,
                                      f"AND d.{ecol} >= 4 AND d.{ecol} IS NOT NULL")
                    except Exception:
                        continue

                    if len(small) >= 20 and len(big) >= 20:
                        t, p = stats.ttest_ind(big, small, equal_var=False)
                        results.append({
                            'name': f'{earlier_sess}_size_big_vs_small',
                            'n_true': len(big), 'mean_true': np.mean(big),
                            'n_false': len(small), 'mean_false': np.mean(small),
                            'delta': np.mean(big) - np.mean(small),
                            't': t, 'p': p,
                        })

                # Report
                if not results:
                    print("  No testable predictors")
                    continue

                results.sort(key=lambda x: x['p'])
                n_tests = len(results)
                for i, r in enumerate(results):
                    r['rank'] = i + 1
                    r['bh_threshold'] = 0.05 * r['rank'] / n_tests
                    r['bh_significant'] = r['p'] <= r['bh_threshold']

                for r in results:
                    sig = "BH-SIG" if r['bh_significant'] else ""
                    raw = "***" if r['p'] < 0.005 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
                    print(f"  {r['name']:<30} delta={r['delta']:>+.4f}  p={r['p']:.4f} {raw:<4} N={r['n_true']}+{r['n_false']}  {sig}")


def run_regime_scan(con, instrument, sessions, size_min=4.0):
    """Run regime scan for an instrument (2025+ focus)."""
    print(f"\n{'='*80}")
    print(f"REGIME SCAN — {instrument} — 2025+ Performance")
    print(f"{'='*80}")

    # Best and worst configs
    results = []
    for sess in sessions:
        size_col = f"orb_{sess}_size"
        for em in ['E1', 'E3']:
            for rr in [2.0, 2.5, 3.0]:
                for cb in [2, 3]:
                    row = con.execute(f"""
                        SELECT COUNT(*) as n,
                               AVG(o.pnl_r) as expr,
                               SUM(o.pnl_r) as totr,
                               SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
                        {SAFE_JOIN}
                        WHERE o.symbol = '{instrument}' AND o.orb_label = '{sess}'
                          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
                          AND d.{size_col} >= {size_min}
                          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
                          AND o.trading_day >= '{REGIME_START}'
                    """).fetchone()
                    if row[0] >= 20:
                        results.append({
                            'config': f'{sess} {em} RR{rr} CB{cb} G{int(size_min)}+',
                            'session': sess, 'em': em, 'rr': rr, 'cb': cb,
                            'n': row[0], 'expr': row[1], 'totr': row[2], 'wr': row[3]
                        })

    if not results:
        print("  No configs with N>=20 in 2025+ regime")
        return

    # Worst
    results.sort(key=lambda x: x['expr'])
    print(f"\nTOP 10 WORST configs (N>=20, 2025+):")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>2}. {r['config']:<35} N={r['n']:<4} ExpR={r['expr']:>+.4f}  TotalR={r['totr']:>+.1f}  WR={r['wr']:.0f}%")

    # Best
    results.sort(key=lambda x: x['expr'], reverse=True)
    print(f"\nTOP 10 BEST configs (N>=20, 2025+):")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>2}. {r['config']:<35} N={r['n']:<4} ExpR={r['expr']:>+.4f}  TotalR={r['totr']:>+.1f}  WR={r['wr']:.0f}%")

    # Also show full-period (all time) for comparison
    print(f"\n--- Full Period Comparison (top 5 best 2025+ configs) ---")
    results.sort(key=lambda x: x['expr'], reverse=True)
    for r in results[:5]:
        sess, em, rr, cb = r['session'], r['em'], r['rr'], r['cb']
        size_col = f"orb_{sess}_size"
        full = con.execute(f"""
            SELECT COUNT(*) as n, AVG(o.pnl_r) as expr, SUM(o.pnl_r) as totr
            {SAFE_JOIN}
            WHERE o.symbol = '{instrument}' AND o.orb_label = '{sess}'
              AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
              AND d.{size_col} >= {size_min}
              AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
        """).fetchone()
        print(f"  {r['config']:<35} 2025+: N={r['n']:<4} ExpR={r['expr']:>+.4f} | ALL: N={full[0]:<5} ExpR={full[1]:>+.4f}")

    # DOW breakdown for best session
    if results:
        best = results[0]
        sess, em, rr, cb = best['session'], best['em'], best['rr'], best['cb']
        size_col = f"orb_{sess}_size"
        print(f"\n--- DOW Breakdown: {best['config']} (2025+) ---")
        rows = con.execute(f"""
            SELECT
                CASE EXTRACT(isodow FROM o.trading_day)
                    WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' WHEN 3 THEN 'Wed'
                    WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri'
                END as dow,
                COUNT(*) as n, AVG(o.pnl_r) as expr, SUM(o.pnl_r) as totr,
                SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
            {SAFE_JOIN}
            WHERE o.symbol = '{instrument}' AND o.orb_label = '{sess}'
              AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
              AND d.{size_col} >= {size_min}
              AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
              AND o.trading_day >= '{REGIME_START}'
            GROUP BY 1 ORDER BY 1
        """).fetchall()
        for r in rows:
            tag = ' <<<' if r[1] >= 5 and r[2] > 0.5 else (' AVOID' if r[1] >= 5 and r[2] < -0.5 else '')
            print(f"  {r[0]:<5} N={r[1]:<4} ExpR={r[2]:>+.4f}  TotalR={r[3]:>+.1f}  WR={r[4]:.0f}%{tag}")

    # Monthly for worst session (death certificate)
    if results:
        results.sort(key=lambda x: x['expr'])
        worst = results[0]
        sess, em, rr, cb = worst['session'], worst['em'], worst['rr'], worst['cb']
        size_col = f"orb_{sess}_size"
        print(f"\n--- Monthly PnL: {worst['config']} (2024+) ---")
        rows = con.execute(f"""
            SELECT
                EXTRACT(year FROM o.trading_day) || '-' ||
                LPAD(CAST(EXTRACT(month FROM o.trading_day) AS VARCHAR), 2, '0') as month,
                COUNT(*) as n, AVG(o.pnl_r) as expr, SUM(o.pnl_r) as totr
            {SAFE_JOIN}
            WHERE o.symbol = '{instrument}' AND o.orb_label = '{sess}'
              AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
              AND d.{size_col} >= {size_min}
              AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
              AND o.trading_day >= '2024-01-01'
            GROUP BY 1 ORDER BY 1
        """).fetchall()
        neg = 0
        total = 0
        for r in rows:
            mark = ' NEG' if r[3] < 0 else ''
            print(f"  {r[0]} N={r[1]:<3} ExpR={r[2]:>+.4f} TotalR={r[3]:>+.1f}{mark}")
            total += 1
            if r[3] < 0:
                neg += 1
        print(f"  --- {neg}/{total} months negative ---")


def main():
    parser = argparse.ArgumentParser(description="Multi-instrument edge hunter + regime scan")
    parser.add_argument('--instrument', type=str, default=None,
                        help='Single instrument (MES, MNQ, MGC). Default: all three.')
    parser.add_argument('--size-min', type=float, default=4.0,
                        help='Minimum ORB size filter (default: 4.0)')
    parser.add_argument('--edge-only', action='store_true',
                        help='Only run edge hunter (skip regime scan)')
    parser.add_argument('--regime-only', action='store_true',
                        help='Only run regime scan (skip edge hunter)')
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ['MGC', 'MES', 'MNQ']

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    for inst in instruments:
        sessions = get_available_sessions(con, inst, min_n=50)
        if not sessions:
            print(f"\n{inst}: No sessions with N>=50. Skipping.")
            continue

        print(f"\n{'#'*80}")
        print(f"# {inst} — Sessions: {', '.join(sessions)}")
        print(f"{'#'*80}")

        if not args.regime_only:
            run_edge_hunter(con, inst, sessions, args.size_min)

        if not args.edge_only:
            run_regime_scan(con, inst, sessions, args.size_min)

    con.close()
    print(f"\n\nDone. All results use correct joins and zero look-ahead.")


if __name__ == '__main__':
    main()
