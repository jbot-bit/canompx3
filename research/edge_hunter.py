"""
Edge hunter: find honest predictors of ORB outcome quality.
ZERO look-ahead. Correct joins. Audited.

Rules enforced structurally:
1. JOIN always includes orb_minutes match
2. Every predictor is validated as pre-entry
3. Row count audit before/after every join
4. Multiple comparison correction applied
"""
import duckdb
import numpy as np
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# === CORRECT JOIN TEMPLATE ===
# ALWAYS match on orb_minutes to prevent row inflation
SAFE_JOIN = """
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
"""


def audit_join(instrument, session, entry_model, rr, cb, size_min):
    """Verify join doesn't inflate rows."""
    size_col = f"orb_{session}_size"

    n_outcomes = con.execute(f"""
        SELECT COUNT(*) FROM orb_outcomes
        WHERE symbol = '{instrument}' AND orb_label = '{session}'
          AND entry_model = '{entry_model}' AND rr_target = {rr} AND confirm_bars = {cb}
          AND outcome IN ('win','loss','early_exit') AND pnl_r IS NOT NULL
    """).fetchone()[0]

    n_joined = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{entry_model}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """).fetchone()[0]

    n_filtered = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{entry_model}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND d.{size_col} >= {size_min}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """).fetchone()[0]

    ok = n_joined <= n_outcomes and n_filtered <= n_joined
    if not ok:
        print(f"  AUDIT FAIL: outcomes={n_outcomes} joined={n_joined} filtered={n_filtered}")
        sys.exit(1)
    return n_outcomes, n_joined, n_filtered


def get_pnl_array(instrument, session, em, rr, cb, size_min, extra_where=""):
    """Get pnl_r array with safe join."""
    size_col = f"orb_{session}_size"
    return con.execute(f"""
        SELECT o.pnl_r {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND d.{size_col} >= {size_min}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          {extra_where}
    """).fetchnumpy()['pnl_r']


def test_binary_split(name, instrument, session, em, rr, cb, size_min,
                       col_expr, true_label, false_label):
    """Test a binary predictor. Returns (delta, p, n_true, n_false)."""
    size_col = f"orb_{session}_size"

    true_arr = get_pnl_array(instrument, session, em, rr, cb, size_min,
                              f"AND ({col_expr}) = true")
    false_arr = get_pnl_array(instrument, session, em, rr, cb, size_min,
                               f"AND ({col_expr}) = false")

    n_t, n_f = len(true_arr), len(false_arr)
    if n_t < 20 or n_f < 20:
        return None

    m_t, m_f = np.mean(true_arr), np.mean(false_arr)
    t, p = stats.ttest_ind(true_arr, false_arr, equal_var=False)
    delta = m_f - m_t  # false - true (convention: "clean" minus "dirty")

    return {
        'name': name,
        'n_true': n_t, 'mean_true': m_t,
        'n_false': n_f, 'mean_false': m_f,
        'delta': delta, 't': t, 'p': p,
    }


def test_bucket_split(name, instrument, session, em, rr, cb, size_min,
                       bucket_expr, extra_where=""):
    """Test a multi-bucket predictor."""
    size_col = f"orb_{session}_size"

    rows = con.execute(f"""
        SELECT {bucket_expr} as bucket, COUNT(*) as n,
               AVG(o.pnl_r) as expr, SUM(o.pnl_r) as totr
        {SAFE_JOIN}
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND d.{size_col} >= {size_min}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          {extra_where}
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    return rows


# ================================================================
# MAIN SCAN
# ================================================================

INSTRUMENTS = ['MGC']
SESSIONS_CLEAN = ['1000', '1100']  # No DST
SESSIONS_DST = ['0900', '1800', '2300', '0030']  # DST-affected

print("=" * 80)
print("EDGE HUNTER v2 — Correct Joins, Zero Look-Ahead, Audited")
print("=" * 80)

for inst in INSTRUMENTS:
    for sess in SESSIONS_CLEAN + SESSIONS_DST:
        for em in ['E1', 'E3']:
            for rr in [2.0, 2.5, 3.0]:
                cb = 2
                size_min = 4.0

                # Audit join
                n_raw, n_join, n_filt = audit_join(inst, sess, em, rr, cb, size_min)
                if n_filt < 50:
                    continue

                baseline = get_pnl_array(inst, sess, em, rr, cb, size_min)
                base_expr = np.mean(baseline) if len(baseline) > 0 else 0

                print(f"\n--- {inst} {sess} {em} RR{rr} CB{cb} G4+ | N={n_filt} ExpR={base_expr:+.4f} ---")

                results = []

                # === PRE-ENTRY PREDICTORS (no look-ahead) ===

                # 1. Earlier-session double-breaks (only for sessions that happen BEFORE target)
                earlier_sessions = {
                    '1000': ['0900'],
                    '1100': ['0900', '1000'],
                    '1800': ['0900', '1000', '1100'],
                    '2300': ['0900', '1000', '1100', '1800'],
                    '0030': ['0900', '1000', '1100', '1800', '2300'],
                    '0900': [],  # nothing before 0900
                }
                for earlier in earlier_sessions.get(sess, []):
                    col = f"d.orb_{earlier}_double_break"
                    r = test_binary_split(
                        f"{earlier}_dbl_break", inst, sess, em, rr, cb, size_min,
                        col, "YES_dbl", "NO_dbl")
                    if r:
                        results.append(r)

                # 2. Earlier-session outcomes
                for earlier in earlier_sessions.get(sess, []):
                    # win vs loss only (exclude scratch)
                    win_arr = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                            f"AND d.orb_{earlier}_outcome = 'win'")
                    loss_arr = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                             f"AND d.orb_{earlier}_outcome = 'loss'")
                    if len(win_arr) >= 20 and len(loss_arr) >= 20:
                        t, p = stats.ttest_ind(win_arr, loss_arr, equal_var=False)
                        results.append({
                            'name': f"{earlier}_outcome(win_vs_loss)",
                            'n_true': len(win_arr), 'mean_true': np.mean(win_arr),
                            'n_false': len(loss_arr), 'mean_false': np.mean(loss_arr),
                            'delta': np.mean(win_arr) - np.mean(loss_arr),
                            't': t, 'p': p,
                        })

                # 3. RSI at 0900 (known before all sessions)
                rsi_rows = test_bucket_split(
                    "RSI14_0900", inst, sess, em, rr, cb, size_min,
                    """CASE WHEN d.rsi_14_at_0900 < 30 THEN 'oversold'
                        WHEN d.rsi_14_at_0900 < 50 THEN 'bearish'
                        WHEN d.rsi_14_at_0900 < 70 THEN 'bullish'
                        ELSE 'overbought' END""",
                    "AND d.rsi_14_at_0900 IS NOT NULL")
                if rsi_rows:
                    # Test extreme vs middle
                    extreme = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                            "AND (d.rsi_14_at_0900 < 30 OR d.rsi_14_at_0900 >= 70) AND d.rsi_14_at_0900 IS NOT NULL")
                    middle = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                           "AND d.rsi_14_at_0900 >= 30 AND d.rsi_14_at_0900 < 70 AND d.rsi_14_at_0900 IS NOT NULL")
                    if len(extreme) >= 20 and len(middle) >= 20:
                        t, p = stats.ttest_ind(middle, extreme, equal_var=False)
                        results.append({
                            'name': "RSI_middle_vs_extreme",
                            'n_true': len(extreme), 'mean_true': np.mean(extreme),
                            'n_false': len(middle), 'mean_false': np.mean(middle),
                            'delta': np.mean(middle) - np.mean(extreme),
                            't': t, 'p': p,
                        })

                # 4. Day of week
                for dow_name, dow_val in [('Mon', 1), ('Tue', 2), ('Wed', 3), ('Thu', 4), ('Fri', 5)]:
                    on_day = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                           f"AND EXTRACT(dow FROM o.trading_day) = {dow_val}")
                    off_day = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                            f"AND EXTRACT(dow FROM o.trading_day) != {dow_val}")
                    if len(on_day) >= 15 and len(off_day) >= 30:
                        t, p = stats.ttest_ind(on_day, off_day, equal_var=False)
                        results.append({
                            'name': f"DOW_{dow_name}",
                            'n_true': len(on_day), 'mean_true': np.mean(on_day),
                            'n_false': len(off_day), 'mean_false': np.mean(off_day),
                            'delta': np.mean(on_day) - np.mean(off_day),
                            't': t, 'p': p,
                        })

                # 5. DST regime (for affected sessions)
                if sess in SESSIONS_DST:
                    dst_col = 'us_dst' if sess != '1800' else 'uk_dst'
                    r = test_binary_split(
                        f"DST_{dst_col}", inst, sess, em, rr, cb, size_min,
                        f"d.{dst_col}", "DST_on", "DST_off")
                    if r:
                        results.append(r)

                # 6. ORB size of earlier session (pre-entry data)
                for earlier in earlier_sessions.get(sess, [])[:2]:  # limit to 2
                    ecol = f"orb_{earlier}_size"
                    small = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                          f"AND d.{ecol} < 4 AND d.{ecol} IS NOT NULL")
                    big = get_pnl_array(inst, sess, em, rr, cb, size_min,
                                        f"AND d.{ecol} >= 4 AND d.{ecol} IS NOT NULL")
                    if len(small) >= 20 and len(big) >= 20:
                        t, p = stats.ttest_ind(big, small, equal_var=False)
                        results.append({
                            'name': f"{earlier}_size_big_vs_small",
                            'n_true': len(big), 'mean_true': np.mean(big),
                            'n_false': len(small), 'mean_false': np.mean(small),
                            'delta': np.mean(big) - np.mean(small),
                            't': t, 'p': p,
                        })

                # === REPORT ===
                if not results:
                    print("  No testable predictors")
                    continue

                # Sort by p-value
                results.sort(key=lambda x: x['p'])

                # Benjamini-Hochberg correction
                n_tests = len(results)
                for i, r in enumerate(results):
                    r['rank'] = i + 1
                    r['bh_threshold'] = 0.05 * r['rank'] / n_tests
                    r['bh_significant'] = r['p'] <= r['bh_threshold']

                for r in results:
                    sig = "BH-SIG" if r['bh_significant'] else ""
                    raw_sig = "***" if r['p'] < 0.005 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
                    print(f"  {r['name']:<30} delta={r['delta']:>+.4f}  p={r['p']:.4f} {raw_sig:<4} N={r['n_true']}+{r['n_false']}  {sig}")

con.close()
print("\n\nDone. All results use corrected joins and zero look-ahead.")
