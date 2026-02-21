"""
Generalized edge discovery scanner.

Runs a structured battery of pre-entry predictor tests on any (instrument, session)
combination. All tests use correct joins (orb_minutes match), zero look-ahead,
and Benjamini-Hochberg FDR correction.

Usage:
    python research/discover.py --instrument MGC --session 1000
    python research/discover.py --instrument MES --session 0030 --entry-model E0
    python research/discover.py --instrument MGC --all-sessions
    python research/discover.py --instrument MGC --session 1000 --json
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH
from pipeline.dst import DST_AFFECTED_SESSIONS


# === CORRECT JOIN TEMPLATE ===
SAFE_JOIN = """
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
"""

# Sessions ordered by time-of-day (Brisbane time)
ALL_SESSIONS = ['0900', '1000', '1100', '1130', '1800', '2300', '0030']

# Which sessions happen before each session (for cross-session predictors)
EARLIER_SESSIONS = {
    '0900': [],
    '1000': ['0900'],
    '1100': ['0900', '1000'],
    '1130': ['0900', '1000', '1100'],
    '1800': ['0900', '1000', '1100', '1130'],
    '2300': ['0900', '1000', '1100', '1130', '1800'],
    '0030': ['0900', '1000', '1100', '1130', '1800', '2300'],
}


def get_connection():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def get_pnl_array(con, instrument, session, em, rr, cb, size_min, extra_where=""):
    """Get pnl_r array with safe join."""
    size_col = f"orb_{session}_size"
    result = con.execute(f"""
        SELECT o.pnl_r {SAFE_JOIN}
        WHERE o.symbol = ? AND o.orb_label = ?
          AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
          AND d.{size_col} >= ?
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          {extra_where}
    """, [instrument, session, em, rr, cb, size_min]).fetchnumpy()
    return result.get('pnl_r', np.array([]))


def audit_join(con, instrument, session, em, rr, cb, size_min):
    """Verify join doesn't inflate rows. Returns (n_raw, n_joined, n_filtered)."""
    size_col = f"orb_{session}_size"

    n_raw = con.execute("""
        SELECT COUNT(*) FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ?
          AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
          AND outcome IN ('win','loss','early_exit') AND pnl_r IS NOT NULL
    """, [instrument, session, em, rr, cb]).fetchone()[0]

    n_joined = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = ? AND o.orb_label = ?
          AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """, [instrument, session, em, rr, cb]).fetchone()[0]

    n_filtered = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = ? AND o.orb_label = ?
          AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
          AND d.{size_col} >= ?
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """, [instrument, session, em, rr, cb, size_min]).fetchone()[0]

    if n_joined > n_raw:
        raise ValueError(f"JOIN INFLATION: raw={n_raw} joined={n_joined}")
    if n_filtered > n_joined:
        raise ValueError(f"FILTER INFLATION: joined={n_joined} filtered={n_filtered}")

    return n_raw, n_joined, n_filtered


def test_binary_split(con, name, instrument, session, em, rr, cb, size_min,
                      col_expr, true_label="true", false_label="false"):
    """Test a binary predictor. Returns result dict or None."""
    true_arr = get_pnl_array(con, instrument, session, em, rr, cb, size_min,
                             f"AND ({col_expr}) = true")
    false_arr = get_pnl_array(con, instrument, session, em, rr, cb, size_min,
                              f"AND ({col_expr}) = false")

    n_t, n_f = len(true_arr), len(false_arr)
    if n_t < 20 or n_f < 20:
        return None

    m_t, m_f = float(np.mean(true_arr)), float(np.mean(false_arr))
    t_stat, p = stats.ttest_ind(true_arr, false_arr, equal_var=False)

    return {
        'name': name, 'type': 'binary',
        'n_true': int(n_t), 'mean_true': round(m_t, 4),
        'n_false': int(n_f), 'mean_false': round(m_f, 4),
        'delta': round(m_f - m_t, 4),
        't': round(float(t_stat), 4), 'p': round(float(p), 6),
        'true_label': true_label, 'false_label': false_label,
    }


def apply_bh_fdr(results, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction to results list (in-place)."""
    results.sort(key=lambda x: x['p'])
    n_tests = len(results)
    for i, r in enumerate(results):
        rank = i + 1
        r['bh_rank'] = rank
        r['bh_threshold'] = round(alpha * rank / n_tests, 6)
        r['bh_significant'] = r['p'] <= r['bh_threshold']
    return results


def scan_session(con, instrument, session, entry_model='E1', rr=2.0, cb=2,
                 size_min=4.0):
    """Run full predictor battery on a single (instrument, session) combo.

    Returns dict with metadata and results list.
    """
    # Audit
    try:
        n_raw, n_joined, n_filtered = audit_join(
            con, instrument, session, entry_model, rr, cb, size_min)
    except ValueError as e:
        return {'error': str(e), 'instrument': instrument, 'session': session}

    if n_filtered < 30:
        return {
            'instrument': instrument, 'session': session,
            'entry_model': entry_model, 'rr': rr, 'cb': cb,
            'n_filtered': n_filtered, 'skipped': True,
            'reason': f'N={n_filtered} < 30 minimum',
            'results': [],
        }

    baseline = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min)
    base_expr = float(np.mean(baseline)) if len(baseline) > 0 else 0.0
    base_wr = float(np.mean(baseline > 0)) if len(baseline) > 0 else 0.0

    results = []

    # -- 1. Day of week --
    for dow_name, dow_val in [('Mon', 1), ('Tue', 2), ('Wed', 3), ('Thu', 4), ('Fri', 5)]:
        on_day = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                               f"AND EXTRACT(dow FROM o.trading_day) = {dow_val}")
        off_day = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                f"AND EXTRACT(dow FROM o.trading_day) != {dow_val}")
        if len(on_day) >= 15 and len(off_day) >= 30:
            t_stat, p = stats.ttest_ind(on_day, off_day, equal_var=False)
            results.append({
                'name': f'DOW_{dow_name}', 'type': 'dow',
                'n_true': int(len(on_day)), 'mean_true': round(float(np.mean(on_day)), 4),
                'n_false': int(len(off_day)), 'mean_false': round(float(np.mean(off_day)), 4),
                'delta': round(float(np.mean(on_day) - np.mean(off_day)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # -- 2. Prior-day direction (prev_day_direction column: 'up' or 'down') --
    for direction in ['up', 'down']:
        arr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                            f"AND d.prev_day_direction = '{direction}'")
        other = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                              f"AND d.prev_day_direction IS NOT NULL AND d.prev_day_direction != '{direction}'")
        if len(arr) >= 20 and len(other) >= 20:
            t_stat, p = stats.ttest_ind(arr, other, equal_var=False)
            results.append({
                'name': f'prev_day_{direction}', 'type': 'prior_day',
                'n_true': int(len(arr)), 'mean_true': round(float(np.mean(arr)), 4),
                'n_false': int(len(other)), 'mean_false': round(float(np.mean(other)), 4),
                'delta': round(float(np.mean(arr) - np.mean(other)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # -- 3. Earlier-session outcomes (cross-session context) --
    for earlier in EARLIER_SESSIONS.get(session, [])[:3]:
        win_arr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                f"AND d.orb_{earlier}_outcome = 'win'")
        loss_arr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                 f"AND d.orb_{earlier}_outcome = 'loss'")
        if len(win_arr) >= 20 and len(loss_arr) >= 20:
            t_stat, p = stats.ttest_ind(win_arr, loss_arr, equal_var=False)
            results.append({
                'name': f'{earlier}_outcome_win_vs_loss', 'type': 'cross_session',
                'n_true': int(len(win_arr)), 'mean_true': round(float(np.mean(win_arr)), 4),
                'n_false': int(len(loss_arr)), 'mean_false': round(float(np.mean(loss_arr)), 4),
                'delta': round(float(np.mean(win_arr) - np.mean(loss_arr)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # -- 4. ATR regime (compressed spring — atr_vel_regime) --
    contracting = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                "AND d.atr_vel_regime = 'Contracting'")
    not_contracting = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                    "AND d.atr_vel_regime IS NOT NULL AND d.atr_vel_regime != 'Contracting'")
    if len(contracting) >= 20 and len(not_contracting) >= 20:
        t_stat, p = stats.ttest_ind(contracting, not_contracting, equal_var=False)
        results.append({
            'name': 'ATR_contracting', 'type': 'regime',
            'n_true': int(len(contracting)), 'mean_true': round(float(np.mean(contracting)), 4),
            'n_false': int(len(not_contracting)), 'mean_false': round(float(np.mean(not_contracting)), 4),
            'delta': round(float(np.mean(contracting) - np.mean(not_contracting)), 4),
            't': round(float(t_stat), 4), 'p': round(float(p), 6),
        })

    # -- 5. RSI regime --
    for rsi_col in ['rsi_14_at_0900']:
        extreme = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                f"AND (d.{rsi_col} < 30 OR d.{rsi_col} >= 70) AND d.{rsi_col} IS NOT NULL")
        middle = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                               f"AND d.{rsi_col} >= 30 AND d.{rsi_col} < 70 AND d.{rsi_col} IS NOT NULL")
        if len(extreme) >= 20 and len(middle) >= 20:
            t_stat, p = stats.ttest_ind(middle, extreme, equal_var=False)
            results.append({
                'name': 'RSI_middle_vs_extreme', 'type': 'indicator',
                'n_true': int(len(extreme)), 'mean_true': round(float(np.mean(extreme)), 4),
                'n_false': int(len(middle)), 'mean_false': round(float(np.mean(middle)), 4),
                'delta': round(float(np.mean(middle) - np.mean(extreme)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # -- 6. DST regime (for affected sessions) --
    if session in DST_AFFECTED_SESSIONS:
        dst_col = 'us_dst' if session != '1800' else 'uk_dst'
        r = test_binary_split(con, f'DST_{dst_col}', instrument, session,
                              entry_model, rr, cb, size_min,
                              f"d.{dst_col}", "DST_on", "DST_off")
        if r:
            r['type'] = 'dst'
            results.append(r)

    # -- 7. Earlier-session ORB size --
    for earlier in EARLIER_SESSIONS.get(session, [])[:2]:
        ecol = f"orb_{earlier}_size"
        small = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                              f"AND d.{ecol} < 4 AND d.{ecol} IS NOT NULL")
        big = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                            f"AND d.{ecol} >= 4 AND d.{ecol} IS NOT NULL")
        if len(small) >= 20 and len(big) >= 20:
            t_stat, p = stats.ttest_ind(big, small, equal_var=False)
            results.append({
                'name': f'{earlier}_size_big_vs_small', 'type': 'cross_session',
                'n_true': int(len(big)), 'mean_true': round(float(np.mean(big)), 4),
                'n_false': int(len(small)), 'mean_false': round(float(np.mean(small)), 4),
                'delta': round(float(np.mean(big) - np.mean(small)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # -- 8. Monthly seasonality --
    for month, month_name in [(1, 'Jan'), (6, 'Jun'), (12, 'Dec')]:
        in_month = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                 f"AND EXTRACT(month FROM o.trading_day) = {month}")
        out_month = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                  f"AND EXTRACT(month FROM o.trading_day) != {month}")
        if len(in_month) >= 10 and len(out_month) >= 30:
            t_stat, p = stats.ttest_ind(in_month, out_month, equal_var=False)
            results.append({
                'name': f'month_{month_name}', 'type': 'seasonality',
                'n_true': int(len(in_month)), 'mean_true': round(float(np.mean(in_month)), 4),
                'n_false': int(len(out_month)), 'mean_false': round(float(np.mean(out_month)), 4),
                'delta': round(float(np.mean(in_month) - np.mean(out_month)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # -- Apply BH FDR --
    if results:
        apply_bh_fdr(results)

    return {
        'instrument': instrument,
        'session': session,
        'entry_model': entry_model,
        'rr': rr, 'cb': cb, 'size_min': size_min,
        'n_raw': n_raw, 'n_filtered': n_filtered,
        'baseline_expr': round(base_expr, 4),
        'baseline_wr': round(base_wr, 4),
        'n_tests': len(results),
        'n_bh_significant': sum(1 for r in results if r.get('bh_significant')),
        'results': results,
        'skipped': False,
    }


def run_discovery(instrument, sessions=None, entry_model='E1', rr=2.0, cb=2,
                  size_min=4.0, output_json=False):
    """Run discovery across one or more sessions."""
    if sessions is None:
        sessions = ALL_SESSIONS

    con = get_connection()
    all_scans = []

    for session in sessions:
        scan = scan_session(con, instrument, session, entry_model, rr, cb, size_min)
        all_scans.append(scan)

        if not output_json:
            if scan.get('skipped'):
                print(f"\n--- {instrument} {session} {entry_model} RR{rr} CB{cb} | SKIPPED: {scan['reason']} ---")
                continue
            if scan.get('error'):
                print(f"\n--- {instrument} {session} | ERROR: {scan['error']} ---")
                continue

            print(f"\n{'='*70}")
            print(f"{instrument} {session} {entry_model} RR{rr} CB{cb} G{int(size_min)}+ | "
                  f"N={scan['n_filtered']} ExpR={scan['baseline_expr']:+.4f} WR={scan['baseline_wr']:.1%}")
            print(f"{'='*70}")

            if not scan['results']:
                print("  No testable predictors (all splits below N threshold)")
                continue

            print(f"  {scan['n_tests']} tests, {scan['n_bh_significant']} BH-significant\n")
            for r in scan['results']:
                sig = " BH-SIG" if r.get('bh_significant') else ""
                raw = "***" if r['p'] < 0.005 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
                print(f"  {r['name']:<35} delta={r['delta']:>+.4f}  "
                      f"p={r['p']:.4f} {raw:<4} "
                      f"N={r['n_true']}+{r['n_false']}{sig}")

    con.close()

    if output_json:
        print(json.dumps(all_scans, indent=2, default=str))

    return all_scans


def main():
    parser = argparse.ArgumentParser(description="Edge discovery scanner")
    parser.add_argument("--instrument", required=True, help="MGC, MNQ, MES, M2K")
    parser.add_argument("--session", default=None, help="Session label (e.g. 1000, 0900). Omit for all.")
    parser.add_argument("--all-sessions", action="store_true", help="Scan all sessions")
    parser.add_argument("--entry-model", default="E1", help="Entry model (default: E1)")
    parser.add_argument("--rr", type=float, default=2.0, help="RR target (default: 2.0)")
    parser.add_argument("--cb", type=int, default=2, help="Confirm bars (default: 2)")
    parser.add_argument("--size-min", type=float, default=4.0, help="Min ORB size filter (default: 4.0)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    sessions = None
    if args.session:
        sessions = [args.session]
    elif args.all_sessions:
        sessions = ALL_SESSIONS

    run_discovery(
        instrument=args.instrument,
        sessions=sessions,
        entry_model=args.entry_model,
        rr=args.rr, cb=args.cb,
        size_min=args.size_min,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
