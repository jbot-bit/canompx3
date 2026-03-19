"""Overnight job: BH FDR correction on all quintile feature tests.
Tests cross-session features on all 3 instruments with proper multiple testing correction.
Also runs per-session breakdown for features that survive FDR."""
import sys; sys.path.insert(0, r"C:\Users\joshd\canompx3")
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

LEGIT_PATTERNS = [
    "break_bar_volume", "break_delay_min", "rel_vol_",
    "atr_20", "atr_vel", "gap_open", "prev_day_range",
    "orb_size", "_size", "compression_z", "volume",
    "prior_sessions", "levels_within", "nearest_level",
    "orb_nested", "prior_orb_size",
]
BLACKLIST_PATTERNS = [
    "mfe_r", "mae_r", "overnight_range", "overnight_high", "overnight_low",
    "break_ts", "prev_day_high", "prev_day_low", "pre_1000_high", "pre_1000_low",
    "session_asia", "took_pdh", "took_pdl", "garch", "day_type",
]

def is_legit(col):
    for b in BLACKLIST_PATTERNS:
        if b in col.lower():
            return False
    for l in LEGIT_PATTERNS:
        if l in col.lower():
            return True
    return False

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

all_tests = []  # (instrument, feature, spread, q1_avg, q5_avg, p_val, direction, n_q1, n_q5, session)

for instrument, rr in [("MNQ", None), ("MGC", 2.5), ("MES", 2.5)]:
    rr_clause = f"AND o.rr_target = {rr}" if rr else ""
    
    if instrument == "MNQ":
        configs = con.execute("""
            SELECT DISTINCT orb_label, entry_model, rr_target, confirm_bars, orb_minutes
            FROM validated_setups WHERE instrument = 'MNQ' AND status = 'active'
        """).fetchall()
        if not configs:
            continue
        config_conditions = " OR ".join([
            f"(o.orb_label='{c[0]}' AND o.entry_model='{c[1]}' AND o.rr_target={c[2]} AND o.confirm_bars={c[3]} AND o.orb_minutes={c[4]})"
            for c in configs
        ])
        where = f"o.symbol = '{instrument}' AND o.pnl_r IS NOT NULL AND ({config_conditions})"
    else:
        where = f"o.symbol = '{instrument}' AND o.pnl_r IS NOT NULL AND o.entry_model = 'E2' AND o.confirm_bars = 1 {rr_clause}"
    
    df = con.execute(f"""
        SELECT o.pnl_r, o.orb_label, o.orb_minutes, o.trading_day, d.*
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day 
            AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE {where}
        ORDER BY o.trading_day
    """).fetchdf()
    
    if df.empty:
        continue
    
    # Use last 20% as test
    n = len(df)
    test_df = df.iloc[int(n * 0.8):]
    pnl = test_df["pnl_r"].values
    
    print(f"\n{instrument}: {len(test_df)} test trades, baseline={pnl.mean():+.4f}", flush=True)
    
    # POOLED test (all sessions combined)
    for col in test_df.columns:
        if not is_legit(col):
            continue
        vals = pd.to_numeric(test_df[col], errors="coerce")
        valid = vals.notna() & np.isfinite(vals)
        if valid.sum() < 50:
            continue
        v = vals[valid].values
        p = pnl[valid.values]
        try:
            quintiles = pd.qcut(v, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        if len(set(quintiles)) < 3:
            continue
        q1 = p[quintiles == 0]
        q5 = p[quintiles == max(quintiles)]
        if len(q1) < 10 or len(q5) < 10:
            continue
        spread = q5.mean() - q1.mean()
        t_stat, p_val = stats.ttest_ind(q5, q1)
        q_means = [p[quintiles == q].mean() for q in sorted(set(quintiles))]
        mono = all(q_means[i] <= q_means[i+1] for i in range(len(q_means)-1))
        anti = all(q_means[i] >= q_means[i+1] for i in range(len(q_means)-1))
        direction = "MONO+" if mono else ("MONO-" if anti else "mixed")
        all_tests.append((instrument, col, spread, q1.mean(), q5.mean(), p_val, direction, len(q1), len(q5), "POOLED"))
    
    # PER-SESSION test for top features
    for session in test_df["orb_label"].unique():
        sess_mask = test_df["orb_label"] == session
        sess_df = test_df[sess_mask]
        sess_pnl = pnl[sess_mask.values]
        if len(sess_df) < 30:
            continue
        for col in sess_df.columns:
            if not is_legit(col):
                continue
            vals = pd.to_numeric(sess_df[col], errors="coerce")
            valid = vals.notna() & np.isfinite(vals)
            if valid.sum() < 30:
                continue
            v = vals[valid].values
            p_arr = sess_pnl[valid.values]
            try:
                quintiles = pd.qcut(v, 3, labels=False, duplicates="drop")  # terciles for smaller N
            except ValueError:
                continue
            if len(set(quintiles)) < 2:
                continue
            q1 = p_arr[quintiles == 0]
            q3 = p_arr[quintiles == max(quintiles)]
            if len(q1) < 5 or len(q3) < 5:
                continue
            spread = q3.mean() - q1.mean()
            t_stat, p_val = stats.ttest_ind(q3, q1)
            all_tests.append((instrument, col, spread, q1.mean(), q3.mean(), p_val, "tercile", len(q1), len(q3), session))

con.close()

# BH FDR correction
print(f"\n{'='*90}")
print(f"BH FDR CORRECTION — {len(all_tests)} total tests")
print(f"{'='*90}")

# Sort by p-value
all_tests.sort(key=lambda x: x[5])
n_tests = len(all_tests)

# BH procedure at q=0.10
survivors = []
for rank, test in enumerate(all_tests, 1):
    bh_threshold = (rank / n_tests) * 0.10
    if test[5] <= bh_threshold:
        survivors.append(test)
    else:
        break  # BH is step-up: stop at first failure

print(f"\nTotal tests: {n_tests}")
print(f"BH FDR survivors (q=0.10): {len(survivors)}")

if survivors:
    print(f"\n{'Inst':<5} {'Feature':<40} {'Spread':>7} {'Q1':>7} {'Q5':>7} {'p_raw':>8} {'Dir':>7} {'Scope':>12}")
    print("-" * 95)
    for inst, col, spread, q1, q5, pv, d, n1, n5, scope in survivors:
        print(f"{inst:<5} {col:<40} {spread:>+7.3f} {q1:>+7.3f} {q5:>+7.3f} {pv:>8.5f} {d:>7} {scope:>12}")
    
    # Group by feature across instruments
    print(f"\n{'='*90}")
    print("FEATURES THAT SURVIVE BH FDR ACROSS INSTRUMENTS:")
    print(f"{'='*90}")
    from collections import defaultdict
    feat_instruments = defaultdict(list)
    for inst, col, spread, q1, q5, pv, d, n1, n5, scope in survivors:
        if scope == "POOLED":
            feat_instruments[col].append((inst, spread, pv))
    
    for feat, entries in sorted(feat_instruments.items(), key=lambda x: -len(x[1])):
        if len(entries) >= 2:
            insts = ", ".join(f"{e[0]}({e[1]:+.3f})" for e in entries)
            print(f"  {feat:<40} — {len(entries)} instruments: {insts}")
else:
    print("\nZERO survivors. Cross-session signals do not survive honest multiple testing correction.")

print(f"\nDone. {len(all_tests)} tests, {len(survivors)} BH FDR survivors at q=0.10.")
