"""
Cross-Session Prediction Scan v2 - 7 Hypotheses
MGC, E2 CB1 RR2.0, 5m aperture, 2016-2026

break_dir is 'long'/'short'/None (string), not numeric.
"""
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

con = duckdb.connect('gold.db', read_only=True)

SYMBOL = 'MGC'
ORB_MIN = 5
ENTRY_MODEL = 'E2'
RR_TARGET = 2.0
CONFIRM_BARS = 1

SESSIONS_ORDERED = [
    'CME_REOPEN', 'TOKYO_OPEN', 'SINGAPORE_OPEN',
    'LONDON_METALS', 'EUROPE_FLOW',
    'NYSE_OPEN', 'US_DATA_830', 'US_DATA_1000',
    'COMEX_SETTLE',
]
US_SESSIONS = ['NYSE_OPEN', 'US_DATA_830', 'US_DATA_1000']
ASIA_SESSIONS = ['TOKYO_OPEN', 'SINGAPORE_OPEN']
PRE_US = ['TOKYO_OPEN', 'SINGAPORE_OPEN', 'LONDON_METALS']

# ===== LOAD =====
print("Loading data...")
df_rows = con.execute(f"""
    SELECT * FROM daily_features
    WHERE symbol='{SYMBOL}' AND orb_minutes={ORB_MIN}
    ORDER BY trading_day
""").fetchdf()
print(f"daily_features: {len(df_rows)} rows")

outcomes = con.execute(f"""
    SELECT trading_day, orb_label, pnl_r
    FROM orb_outcomes
    WHERE symbol='{SYMBOL}' AND orb_minutes={ORB_MIN}
      AND entry_model='{ENTRY_MODEL}' AND rr_target={RR_TARGET}
      AND confirm_bars={CONFIRM_BARS}
""").fetchdf()

outcome_pivot = outcomes.pivot_table(
    index='trading_day', columns='orb_label', values='pnl_r', aggfunc='first'
)

df = df_rows.merge(outcome_pivot, left_on='trading_day', right_index=True,
                   how='inner', suffixes=('', '_out'))
df['year'] = pd.to_datetime(df['trading_day']).dt.year
print(f"Merged: {len(df)} rows, years: {df['year'].min()}-{df['year'].max()}")

# Load multi-aperture data
df15 = con.execute(f"""
    SELECT trading_day,
           orb_TOKYO_OPEN_break_dir as tokyo_dir_15,
           orb_SINGAPORE_OPEN_break_dir as sg_dir_15,
           orb_LONDON_METALS_break_dir as lm_dir_15
    FROM daily_features WHERE symbol='{SYMBOL}' AND orb_minutes=15
""").fetchdf()
df30 = con.execute(f"""
    SELECT trading_day,
           orb_TOKYO_OPEN_break_dir as tokyo_dir_30,
           orb_SINGAPORE_OPEN_break_dir as sg_dir_30,
           orb_LONDON_METALS_break_dir as lm_dir_30
    FROM daily_features WHERE symbol='{SYMBOL}' AND orb_minutes=30
""").fetchdf()
df_multi = df.merge(df15, on='trading_day', how='inner').merge(df30, on='trading_day', how='inner')
print(f"Multi-aperture merged: {len(df_multi)} rows")

con.close()

all_results = []


def quintile_test(pred_arr, out_arr, label, years_arr=None, min_per_q=30):
    """Quintile split on predictor, measure outcome ExpR per quintile."""
    mask = np.isfinite(pred_arr) & np.isfinite(out_arr)
    pred = pred_arr[mask]
    out = out_arr[mask]
    yrs = years_arr[mask] if years_arr is not None else None

    if len(pred) < min_per_q * 5:
        return None
    try:
        quintiles = pd.qcut(pred, 5, labels=False, duplicates='drop')
    except ValueError:
        return None
    if len(set(quintiles)) < 3:
        return None

    q_keys = sorted(set(quintiles))
    q_min, q_max = q_keys[0], q_keys[-1]
    q_out = {q: out[quintiles == q] for q in q_keys}
    spread = np.mean(q_out[q_max]) - np.mean(q_out[q_min])
    _, p_val = stats.ttest_ind(q_out[q_max], q_out[q_min])

    # Yearly consistency for extreme quintile
    yearly_detail = {}
    if yrs is not None:
        best_q = q_max if spread > 0 else q_min
        best_mask = quintiles == best_q
        best_yrs = yrs[best_mask]
        best_out = out[best_mask]
        for yr in sorted(set(best_yrs)):
            yr_vals = best_out[best_yrs == yr]
            yearly_detail[int(yr)] = round(float(np.mean(yr_vals)), 4)

    return {
        'label': label,
        'type': 'quintile',
        'n_total': len(pred),
        'n_per_q': {int(q): int((quintiles == q).sum()) for q in q_keys},
        'expr_per_q': {int(q): round(float(np.mean(q_out[q])), 4) for q in q_keys},
        'spread': round(float(spread), 4),
        'p_value': float(p_val),
        'yearly': yearly_detail,
    }


def binary_test(cond_arr, out_arr, label, years_arr=None):
    """Binary split, measure outcome ExpR for True vs False."""
    # cond_arr should be boolean-like, out_arr float
    try:
        cond = np.array(cond_arr, dtype=bool)
    except (ValueError, TypeError):
        return None

    mask = np.isfinite(out_arr) & ~pd.isna(cond_arr)
    c = cond[mask]
    o = out_arr[mask]
    yrs = years_arr[mask] if years_arr is not None else None

    grp_t = o[c]
    grp_f = o[~c]

    if len(grp_t) < 30 or len(grp_f) < 30:
        return None

    spread = float(np.mean(grp_t) - np.mean(grp_f))
    _, p_val = stats.ttest_ind(grp_t, grp_f)

    # Yearly consistency for the TRUE group spread over FALSE
    yearly_detail = {}
    if yrs is not None:
        for yr in sorted(set(yrs)):
            yr_mask = yrs == yr
            yr_t = o[yr_mask & c]
            yr_f = o[yr_mask & ~c]
            if len(yr_t) > 0 and len(yr_f) > 0:
                yearly_detail[int(yr)] = round(float(np.mean(yr_t) - np.mean(yr_f)), 4)

    return {
        'label': label,
        'type': 'binary',
        'n_total': len(o),
        'n_true': int(len(grp_t)),
        'n_false': int(len(grp_f)),
        'expr_true': round(float(np.mean(grp_t)), 4),
        'expr_false': round(float(np.mean(grp_f)), 4),
        'spread': round(float(spread), 4),
        'p_value': float(p_val),
        'yearly': yearly_detail,
    }


# =====================================================================
# H1: SESSION LEVEL SWEEPS
# =====================================================================
print("\n" + "=" * 70)
print("H1: SESSION LEVEL SWEEPS (Asia high > PDH / Asia low < PDL -> US)")
print("=" * 70)

for target in US_SESSIONS:
    if target not in df.columns:
        continue
    out = df[target].values.astype(float)
    yrs = df['year'].values

    sweep_h = (df['session_asia_high'] > df['prev_day_high']).values
    r = binary_test(sweep_h, out, f"H1: Asia>PDH -> {target}", yrs)
    if r:
        all_results.append(r)
        print(f"  {target} PDH: T({r['n_true']},{r['expr_true']:.4f}) F({r['n_false']},{r['expr_false']:.4f}) sp={r['spread']:.4f} p={r['p_value']:.4f}")

    sweep_l = (df['session_asia_low'] < df['prev_day_low']).values
    r2 = binary_test(sweep_l, out, f"H1: Asia<PDL -> {target}", yrs)
    if r2:
        all_results.append(r2)
        print(f"  {target} PDL: T({r2['n_true']},{r2['expr_true']:.4f}) F({r2['n_false']},{r2['expr_false']:.4f}) sp={r2['spread']:.4f} p={r2['p_value']:.4f}")

# =====================================================================
# H2: DOUBLE BREAK AT PRIOR SESSION
# =====================================================================
print("\n" + "=" * 70)
print("H2: DOUBLE BREAK (prior session) -> US")
print("=" * 70)

for pred_s in PRE_US:
    dbl_col = f'orb_{pred_s}_double_break'
    if dbl_col not in df.columns:
        continue
    for target in US_SESSIONS:
        if target not in df.columns:
            continue
        # Handle NaN in boolean column
        valid_mask = df[dbl_col].notna()
        dbl_vals = df.loc[valid_mask, dbl_col].astype(bool).values
        out_vals = df.loc[valid_mask, target].values.astype(float)
        yrs_vals = df.loc[valid_mask, 'year'].values

        r = binary_test(dbl_vals, out_vals, f"H2: {pred_s}_dbl -> {target}", yrs_vals)
        if r:
            all_results.append(r)
            print(f"  {pred_s} -> {target}: Dbl({r['n_true']},{r['expr_true']:.4f}) No({r['n_false']},{r['expr_false']:.4f}) sp={r['spread']:.4f} p={r['p_value']:.4f}")

# =====================================================================
# H3: BREAK DELAY AT PRIOR SESSION
# =====================================================================
print("\n" + "=" * 70)
print("H3: BREAK DELAY (prior session) -> US")
print("=" * 70)

for pred_s in PRE_US:
    delay_col = f'orb_{pred_s}_break_delay_min'
    if delay_col not in df.columns:
        continue
    for target in US_SESSIONS:
        if target not in df.columns:
            continue
        pred = df[delay_col].values.astype(float)
        out = df[target].values.astype(float)
        yrs = df['year'].values
        r = quintile_test(pred, out, f"H3: {pred_s}_delay -> {target}", yrs)
        if r:
            all_results.append(r)
            qs = r['expr_per_q']
            qk = sorted(qs.keys())
            print(f"  {pred_s} -> {target}: Q1={qs[qk[0]]} Q5={qs[qk[-1]]} sp={r['spread']:.4f} p={r['p_value']:.4f}")

# =====================================================================
# H4: ATR ACCELERATION
# =====================================================================
print("\n" + "=" * 70)
print("H4: ATR VELOCITY RATIO -> session outcomes")
print("=" * 70)

for target in SESSIONS_ORDERED:
    if target not in df.columns:
        continue
    pred = df['atr_vel_ratio'].values.astype(float)
    out = df[target].values.astype(float)
    yrs = df['year'].values
    r = quintile_test(pred, out, f"H4: ATR_vel -> {target}", yrs)
    if r:
        all_results.append(r)
        qs = r['expr_per_q']
        qk = sorted(qs.keys())
        print(f"  {target}: Q1={qs[qk[0]]} Q5={qs[qk[-1]]} sp={r['spread']:.4f} p={r['p_value']:.4f}")

# =====================================================================
# H5: GAP x DIRECTION INTERACTION
# =====================================================================
print("\n" + "=" * 70)
print("H5: GAP x DIRECTION (aligned vs opposed)")
print("=" * 70)

# break_dir is 'long'/'short' string. Convert to numeric sign.
def dir_to_sign(d):
    if d == 'long':
        return 1
    elif d == 'short':
        return -1
    return 0

for target in SESSIONS_ORDERED:
    if target not in df.columns:
        continue
    dir_col = f'orb_{target}_break_dir'
    if dir_col not in df.columns:
        continue

    gap = df['gap_open_points'].values.astype(float)
    brk_dir = np.array([dir_to_sign(d) for d in df[dir_col].values])
    out = df[target].values.astype(float)
    yrs = df['year'].values

    valid = np.isfinite(gap) & np.isfinite(out) & (brk_dir != 0) & (gap != 0)
    if valid.sum() < 100:
        continue

    aligned = valid & (np.sign(gap) == brk_dir)
    opposed = valid & (np.sign(gap) != brk_dir) & (brk_dir != 0)

    if aligned.sum() < 30 or opposed.sum() < 30:
        continue

    # Create binary: aligned=True, opposed=False, rest excluded
    cond = np.full(len(df), np.nan)
    cond[aligned] = 1.0
    cond[opposed] = 0.0
    sub_mask = np.isfinite(cond)

    r = binary_test(cond[sub_mask].astype(bool), out[sub_mask],
                    f"H5: Gap*Dir -> {target}", yrs[sub_mask])
    if r:
        all_results.append(r)
        print(f"  {target}: Aligned({r['n_true']},{r['expr_true']:.4f}) Opposed({r['n_false']},{r['expr_false']:.4f}) sp={r['spread']:.4f} p={r['p_value']:.4f}")

# Gap magnitude quintiles
print("\n  --- Gap magnitude quintiles ---")
for target in US_SESSIONS:
    if target not in df.columns:
        continue
    pred = np.abs(df['gap_open_points'].values.astype(float))
    out = df[target].values.astype(float)
    yrs = df['year'].values
    r = quintile_test(pred, out, f"H5b: |Gap| -> {target}", yrs)
    if r:
        all_results.append(r)
        qs = r['expr_per_q']
        qk = sorted(qs.keys())
        print(f"  {target}: Q1={qs[qk[0]]} Q5={qs[qk[-1]]} sp={r['spread']:.4f} p={r['p_value']:.4f}")

# =====================================================================
# H6: MULTI-APERTURE ALIGNMENT
# =====================================================================
print("\n" + "=" * 70)
print("H6: MULTI-APERTURE ALIGNMENT (5/15/30m same dir) -> US")
print("=" * 70)

apt_map = {
    'TOKYO_OPEN': ('orb_TOKYO_OPEN_break_dir', 'tokyo_dir_15', 'tokyo_dir_30'),
    'SINGAPORE_OPEN': ('orb_SINGAPORE_OPEN_break_dir', 'sg_dir_15', 'sg_dir_30'),
    'LONDON_METALS': ('orb_LONDON_METALS_break_dir', 'lm_dir_15', 'lm_dir_30'),
}

for pred_s, (c5, c15, c30) in apt_map.items():
    # String comparison: all three same non-null direction
    d5 = df_multi[c5].values
    d15 = df_multi[c15].values
    d30 = df_multi[c30].values

    aligned = np.array([
        (a is not None and a == b and b == c)
        for a, b, c in zip(d5, d15, d30)
    ])

    for target in US_SESSIONS:
        if target not in df_multi.columns:
            continue
        out = df_multi[target].values.astype(float)
        yrs = df_multi['year'].values
        r = binary_test(aligned, out, f"H6: {pred_s}_3apt -> {target}", yrs)
        if r:
            all_results.append(r)
            print(f"  {pred_s} -> {target}: Aligned({r['n_true']},{r['expr_true']:.4f}) Not({r['n_false']},{r['expr_false']:.4f}) sp={r['spread']:.4f} p={r['p_value']:.4f}")

# =====================================================================
# H7: PRIOR SESSION MFE/MAE QUALITY
# =====================================================================
print("\n" + "=" * 70)
print("H7: PRIOR SESSION MFE -> later session outcomes")
print("=" * 70)

for pred_s in PRE_US:
    mfe_col = f'orb_{pred_s}_mfe_r'
    mae_col = f'orb_{pred_s}_mae_r'

    for target in US_SESSIONS:
        if target not in df.columns:
            continue
        out = df[target].values.astype(float)
        yrs = df['year'].values

        # MFE quintiles
        r = quintile_test(df[mfe_col].values.astype(float), out,
                         f"H7: {pred_s}_MFE -> {target}", yrs)
        if r:
            all_results.append(r)
            qs = r['expr_per_q']
            qk = sorted(qs.keys())
            print(f"  {pred_s} MFE -> {target}: Q1={qs[qk[0]]} Q5={qs[qk[-1]]} sp={r['spread']:.4f} p={r['p_value']:.4f}")

        # MAE quintiles
        r2 = quintile_test(df[mae_col].values.astype(float), out,
                          f"H7: {pred_s}_MAE -> {target}", yrs)
        if r2:
            all_results.append(r2)
            qs2 = r2['expr_per_q']
            qk2 = sorted(qs2.keys())
            print(f"  {pred_s} MAE -> {target}: Q1={qs2[qk2[0]]} Q5={qs2[qk2[-1]]} sp={r2['spread']:.4f} p={r2['p_value']:.4f}")

    # MFE/MAE ratio
    mfe = df[mfe_col].values.astype(float)
    mae = np.abs(df[mae_col].values.astype(float))
    mae_safe = np.where(mae < 0.01, 0.01, mae)
    quality = mfe / mae_safe

    for target in US_SESSIONS:
        if target not in df.columns:
            continue
        out = df[target].values.astype(float)
        yrs = df['year'].values
        r = quintile_test(quality, out, f"H7b: {pred_s}_Q -> {target}", yrs)
        if r:
            all_results.append(r)
            qs = r['expr_per_q']
            qk = sorted(qs.keys())
            print(f"  {pred_s} MFE/MAE -> {target}: Q1={qs[qk[0]]} Q5={qs[qk[-1]]} sp={r['spread']:.4f} p={r['p_value']:.4f}")


# =====================================================================
# BH FDR CORRECTION
# =====================================================================
print("\n" + "=" * 70)
print("BH FDR CORRECTION ACROSS ALL TESTS")
print("=" * 70)

n_tests = len(all_results)
print(f"Total tests: {n_tests}")

pvals = np.array([r['p_value'] for r in all_results])
sorted_idx = np.argsort(pvals)

# BH step-up adjusted p-values
adj_pvals = np.ones(n_tests)
for i in range(n_tests - 1, -1, -1):
    idx = sorted_idx[i]
    rank = i + 1
    raw_adj = pvals[idx] * n_tests / rank
    if i == n_tests - 1:
        adj_pvals[idx] = min(raw_adj, 1.0)
    else:
        next_idx = sorted_idx[i + 1]
        adj_pvals[idx] = min(raw_adj, adj_pvals[next_idx], 1.0)

survivors = [i for i in range(n_tests) if adj_pvals[i] <= 0.05]
print(f"BH FDR survivors (alpha=0.05): {len(survivors)} / {n_tests}")

# Also check at alpha=0.10
survivors_10 = [i for i in range(n_tests) if adj_pvals[i] <= 0.10]
print(f"BH FDR survivors (alpha=0.10): {len(survivors_10)} / {n_tests}")

# =====================================================================
# SUMMARY TABLE
# =====================================================================
print("\n" + "=" * 80)
print("FULL SUMMARY TABLE (sorted by raw p-value)")
print("=" * 80)
hdr = f"{'#':>2} {'Hypothesis':<52} {'Spread':>7} {'p_raw':>7} {'p_BH':>7} {'FDR':>4} {'N':>5}"
print(hdr)
print("-" * len(hdr))

for rank, idx in enumerate(sorted_idx):
    r = all_results[idx]
    fdr = "Y" if idx in survivors else ("~" if idx in survivors_10 else "")
    n = r.get('n_total', r.get('n_true', 0) + r.get('n_false', 0))
    print(f"{rank+1:>2} {r['label']:<52} {r['spread']:>7.4f} {r['p_value']:>7.4f} {adj_pvals[idx]:>7.4f} {fdr:>4} {n:>5}")

# =====================================================================
# DETAILED TOP 15 WITH YEARLY CONSISTENCY
# =====================================================================
print("\n" + "=" * 80)
print("TOP 15 DETAILED WITH YEARLY CONSISTENCY")
print("=" * 80)

for rank, idx in enumerate(sorted_idx[:15]):
    r = all_results[idx]
    fdr_note = "[FDR-0.05]" if idx in survivors else ("[FDR-0.10]" if idx in survivors_10 else "[NO FDR]")
    print(f"\n{'='*60}")
    print(f"{rank+1}. {r['label']}  {fdr_note}")
    print(f"   spread={r['spread']:.4f}  p_raw={r['p_value']:.4f}  p_BH={adj_pvals[idx]:.4f}")

    if r['type'] == 'binary':
        print(f"   TRUE: N={r['n_true']}, ExpR={r['expr_true']:.4f}")
        print(f"   FALSE: N={r['n_false']}, ExpR={r['expr_false']:.4f}")
    else:
        qs = r['expr_per_q']
        ns = r['n_per_q']
        print(f"   Quintile ExpR: {qs}")
        print(f"   Quintile N:    {ns}")

    yearly = r.get('yearly', {})
    if yearly:
        pos_yrs = sum(1 for v in yearly.values() if v > 0)
        total_yrs = len(yearly)
        print(f"   Yearly consistency (best quintile/TRUE group spread): {pos_yrs}/{total_yrs} positive years")
        for yr in sorted(yearly.keys()):
            marker = "+" if yearly[yr] > 0 else "-"
            print(f"     {yr}: {yearly[yr]:>8.4f} {marker}")
    else:
        print("   (No yearly data available)")


# =====================================================================
# HYPOTHESIS-LEVEL SUMMARY
# =====================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS-LEVEL VERDICT")
print("=" * 80)

hypothesis_groups = {
    'H1': 'Session Level Sweeps (Asia > PDH/PDL)',
    'H2': 'Double Break at Prior Session',
    'H3': 'Break Delay at Prior Session',
    'H4': 'ATR Velocity Ratio',
    'H5': 'Gap x Direction / Gap Magnitude',
    'H6': 'Multi-Aperture Alignment',
    'H7': 'Prior Session MFE/MAE Quality',
}

for h_prefix, h_name in hypothesis_groups.items():
    h_tests = [(i, all_results[i]) for i in range(n_tests)
               if all_results[i]['label'].startswith(h_prefix)]
    if not h_tests:
        print(f"\n{h_prefix}: {h_name}")
        print(f"  No valid tests produced")
        continue

    best_idx, best_r = min(h_tests, key=lambda x: x[1]['p_value'])
    fdr_any = any(i in survivors for i, _ in h_tests)
    fdr10_any = any(i in survivors_10 for i, _ in h_tests)

    print(f"\n{h_prefix}: {h_name}")
    print(f"  Tests: {len(h_tests)}")
    print(f"  Best: {best_r['label']}  spread={best_r['spread']:.4f}  p={best_r['p_value']:.4f}  p_BH={adj_pvals[best_idx]:.4f}")
    if fdr_any:
        print(f"  VERDICT: FDR SURVIVOR (alpha=0.05)")
    elif fdr10_any:
        print(f"  VERDICT: MARGINAL (FDR alpha=0.10 but not 0.05)")
    elif best_r['p_value'] < 0.05:
        print(f"  VERDICT: NOMINALLY SIGNIFICANT but DEAD after FDR")
    else:
        print(f"  VERDICT: DEAD. No signal.")

print("\n\nDONE.")
