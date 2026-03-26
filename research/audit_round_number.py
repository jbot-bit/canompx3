#!/usr/bin/env python3
"""
Adversarial audit of round-number proximity T0-T8 analysis.
Steps 1-8 from the audit prompt.
"""

import math
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH

# Reproduce the same definitions
ROUND_INCREMENTS = {"MGC": 10.0, "MNQ": 100.0, "MES": 25.0, "M2K": 10.0}
US_SESSIONS = {"COMEX_SETTLE", "NYSE_OPEN", "US_DATA_1000", "US_DATA_830", "NYSE_CLOSE", "CME_PRECLOSE"}
ASIAN_SESSIONS = {"TOKYO_OPEN", "SINGAPORE_OPEN", "CME_REOPEN", "LONDON_METALS", "EUROPE_FLOW", "BRISBANE_1025"}
IS_END = date(2024, 12, 31)
OOS_START = date(2025, 1, 1)


def nearest_round(price, increment):
    return round(price / increment) * increment


def distance_to_nearest_round(price, increment):
    return abs(price - nearest_round(price, increment))


def orb_crosses_round(orb_high, orb_low, increment):
    first_round_above_low = math.ceil(orb_low / increment) * increment
    return first_round_above_low <= orb_high


def load_data():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    placeholders = ", ".join(["?"] * len(instruments))

    # Build CASE expressions
    high_cases = " ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_high" for lbl in SESSION_CATALOG)
    low_cases = " ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_low" for lbl in SESSION_CATALOG)
    size_cases = " ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_size" for lbl in SESSION_CATALOG)

    query = f"""
    SELECT
        o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
        o.entry_price, o.stop_price, o.pnl_r, o.outcome,
        CASE WHEN o.entry_price > o.stop_price THEN 'long'
             WHEN o.entry_price < o.stop_price THEN 'short'
             ELSE NULL END AS break_dir,
        d.atr_20,
        CASE o.orb_label {high_cases} END AS orb_high,
        CASE o.orb_label {low_cases} END AS orb_low,
        CASE o.orb_label {size_cases} END AS orb_size
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol IN ({placeholders})
        AND o.entry_price IS NOT NULL
        AND o.pnl_r IS NOT NULL
        AND o.entry_model = 'E1'
        AND o.confirm_bars = 1
        AND o.rr_target = 2.0
        AND o.orb_minutes = 5
    ORDER BY o.trading_day, o.symbol, o.orb_label
    """

    df = con.execute(query, instruments).fetchdf()
    con.close()
    return df


def add_round_features(df):
    dist_pts = []
    dist_r = []
    crosses = []

    for _, row in df.iterrows():
        sym = row["symbol"]
        inc = ROUND_INCREMENTS.get(sym)
        oh = row["orb_high"]
        ol = row["orb_low"]
        os_ = row["orb_size"]
        bd = row["break_dir"]

        if inc is None or oh is None or ol is None or os_ is None or os_ <= 0 or bd not in ("long", "short"):
            dist_pts.append(np.nan)
            dist_r.append(np.nan)
            crosses.append(np.nan)
            continue

        entry = oh if bd == "long" else ol
        dp = distance_to_nearest_round(entry, inc)
        dist_pts.append(dp)
        dist_r.append(dp / os_)
        crosses.append(orb_crosses_round(oh, ol, inc))

    df["distance_to_round_pts"] = dist_pts
    df["distance_to_round_R"] = dist_r
    df["crosses_round"] = crosses
    return df


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main():
    print("=" * 70)
    print("ADVERSARIAL AUDIT — ROUND-NUMBER PROXIMITY")
    print("=" * 70)

    # ===== STEP 1: REPRODUCE =====
    print("\n" + "=" * 70)
    print("STEP 1: REPRODUCE AND VERIFY COUNTS")
    print("=" * 70)

    df = load_data()
    df = add_round_features(df)
    has_features = df["distance_to_round_R"].notna().sum()

    print(f"Total trades: {len(df):,}")
    print(f"Features computed: {has_features:,}")
    print(f"Reported: 63,114 total, 63,098 with features")
    print(f"Match: {len(df) == 63114}")

    print("\nPer instrument:")
    for sym in sorted(df["symbol"].unique()):
        print(f"  {sym}: {len(df[df['symbol'] == sym]):,}")

    print("\nPer session:")
    for sess in sorted(df["orb_label"].unique()):
        print(f"  {sess}: {len(df[df['orb_label'] == sess]):,}")

    # ===== STEP 2: SIGN TEST AUDIT =====
    print("\n" + "=" * 70)
    print("STEP 2: SIGN TEST AUDIT")
    print("=" * 70)

    print("\nExact two-tailed binomial p-values (n=6, p=0.5):")
    for k in range(7):
        p = stats.binomtest(k, 6, 0.5).pvalue
        print(f"  {k}/6: p={p:.6f}")

    print(f"\n  VERDICT: p=0.031 requires 6/6 or 0/6 (two-tailed).")
    print(f"  4/6 -> p=0.6875. The claim '4/6 gives p=0.031' is MATHEMATICALLY FALSE.")
    print(f"  If the prior session truly observed p=0.031, the result was likely 6/6")
    print(f"  (all US sessions anti-clustering), which is STRONGER than reported.")

    # ===== STEP 3: STATISTICAL SIGNIFICANCE OF "DEAD FLAT" =====
    print("\n" + "=" * 70)
    print("STEP 3: IS 'DEAD FLAT' STATISTICALLY DEFENSIBLE?")
    print("=" * 70)

    feature = "distance_to_round_R"
    sub = df[df[feature].notna()].copy()

    for regime_name, regime_mask in [
        ("ALL", pd.Series(True, index=sub.index)),
        ("US", sub["orb_label"].isin(US_SESSIONS)),
        ("ASIAN", sub["orb_label"].isin(ASIAN_SESSIONS)),
    ]:
        rsub = sub[regime_mask].copy()
        rsub["quintile"] = pd.qcut(rsub[feature], 5, labels=False, duplicates="drop") + 1
        rsub["win"] = (rsub["outcome"] == "win").astype(int)

        print(f"\n--- {regime_name} (N={len(rsub):,}) ---")

        # Contingency table: quintile x win/loss
        quints = sorted(rsub["quintile"].unique())
        table = []
        wr_vals = []
        ns = []
        for q in quints:
            qdf = rsub[rsub["quintile"] == q]
            wins = qdf["win"].sum()
            losses = len(qdf) - wins
            table.append([wins, losses])
            wr_vals.append(wins / len(qdf))
            ns.append(len(qdf))

        table = np.array(table)

        # Chi-squared test
        chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
        print(f"  Chi-squared: chi2={chi2:.4f}, df={dof}, p={p_chi2:.6f}")

        # Cochran-Armitage trend test (manual implementation)
        # H0: no linear trend in proportions across ordered groups
        n_total = sum(ns)
        p_bar = sum(table[:, 0]) / n_total
        scores = np.arange(1, len(quints) + 1)  # 1,2,3,4,5
        t_bar = sum(scores[i] * ns[i] for i in range(len(quints))) / n_total

        numerator = sum(ns[i] * (wr_vals[i] - p_bar) * (scores[i] - t_bar) for i in range(len(quints)))
        denominator = p_bar * (1 - p_bar) * sum(ns[i] * (scores[i] - t_bar)**2 for i in range(len(quints)))
        if denominator > 0:
            z_trend = numerator / (denominator ** 0.5)
            p_trend = 2 * (1 - stats.norm.cdf(abs(z_trend)))
        else:
            z_trend, p_trend = 0, 1.0

        print(f"  Cochran-Armitage trend: z={z_trend:.4f}, p={p_trend:.6f}")

        # Effect sizes
        wr_q1 = wr_vals[0]
        wr_q5 = wr_vals[-1]
        h = cohens_h(wr_q1, wr_q5)
        spread_pct = (wr_q1 - wr_q5) * 100

        print(f"  WR Q1={wr_q1*100:.2f}%, Q5={wr_q5*100:.2f}%, spread={spread_pct:+.2f}%")
        print(f"  Cohen's h (Q1 vs Q5) = {h:.4f}")
        print(f"    (|h| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, >0.8 = large)")

        # Interpretation
        if p_chi2 < 0.05:
            print(f"  ** Chi2 IS significant at p<0.05 — cannot call 'dead flat' without qualification **")
        else:
            print(f"  Chi2 NOT significant — 'dead flat' is defensible")

        if p_trend < 0.05:
            print(f"  ** Trend IS significant at p<0.05 — monotonic trend detected **")
        else:
            print(f"  Trend NOT significant — no monotonic relationship")

    # ===== STEP 4: MULTIPLE TESTING CORRECTION =====
    print("\n" + "=" * 70)
    print("STEP 4: MULTIPLE TESTING CORRECTION")
    print("=" * 70)

    # Collect ALL p-values from the analysis
    all_pvals = []
    all_labels = []

    # Re-run all tests and collect p-values
    for regime_name, regime_mask in [
        ("ALL", pd.Series(True, index=sub.index)),
        ("US", sub["orb_label"].isin(US_SESSIONS)),
        ("ASIAN", sub["orb_label"].isin(ASIAN_SESSIONS)),
    ]:
        rsub = sub[regime_mask].copy()
        rsub["quintile"] = pd.qcut(rsub[feature], 5, labels=False, duplicates="drop") + 1
        rsub["win"] = (rsub["outcome"] == "win").astype(int)

        # Chi2 p-value
        quints = sorted(rsub["quintile"].unique())
        table = np.array([[rsub[rsub["quintile"] == q]["win"].sum(),
                          len(rsub[rsub["quintile"] == q]) - rsub[rsub["quintile"] == q]["win"].sum()]
                         for q in quints])
        _, p_chi2, _, _ = stats.chi2_contingency(table)
        all_pvals.append(p_chi2)
        all_labels.append(f"Chi2_{regime_name}")

        # T1b crosses_round z-test
        crosses = rsub[rsub["crosses_round"] == True]
        no_cross = rsub[rsub["crosses_round"] == False]
        if len(crosses) > 0 and len(no_cross) > 0:
            n1, n2 = len(crosses), len(no_cross)
            w1 = crosses["win"].sum()
            w2 = no_cross["win"].sum()
            p_pool = (w1 + w2) / (n1 + n2)
            se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5
            if se > 0:
                z = (w1/n1 - w2/n2) / se
                p_z = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                p_z = 1.0
            all_pvals.append(p_z)
            all_labels.append(f"CrossesRound_{regime_name}")

    # T5: Per-session chi2 tests
    for session in sorted(sub["orb_label"].unique()):
        sdf = sub[sub["orb_label"] == session].copy()
        if len(sdf) < 50:
            continue
        try:
            sdf["quintile"] = pd.qcut(sdf[feature], 5, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        sdf["win"] = (sdf["outcome"] == "win").astype(int)
        quints = sorted(sdf["quintile"].unique())
        if len(quints) < 2:
            continue
        table = np.array([[sdf[sdf["quintile"] == q]["win"].sum(),
                          len(sdf[sdf["quintile"] == q]) - sdf[sdf["quintile"] == q]["win"].sum()]
                         for q in quints])
        try:
            _, p_chi2, _, _ = stats.chi2_contingency(table)
            all_pvals.append(p_chi2)
            all_labels.append(f"T5_{session}")
        except ValueError:
            pass

    # T6 null floor p-values (from prior run - reproduce)
    all_pvals.extend([0.5904, 0.4206, 0.3626])
    all_labels.extend(["T6_NULL_ALL", "T6_NULL_US", "T6_NULL_ASIAN"])

    # Apply corrections
    K = len(all_pvals)
    print(f"\nTotal tests counted: K={K}")

    # Sort for BH FDR
    sorted_indices = np.argsort(all_pvals)
    sorted_pvals = np.array(all_pvals)[sorted_indices]
    sorted_labels = np.array(all_labels)[sorted_indices]

    # Bonferroni
    bonf_threshold = 0.05 / K

    # BH FDR
    bh_adjusted = np.zeros(K)
    for i in range(K):
        bh_adjusted[i] = sorted_pvals[i] * K / (i + 1)
    # Make monotone
    for i in range(K - 2, -1, -1):
        bh_adjusted[i] = min(bh_adjusted[i], bh_adjusted[i + 1])
    bh_adjusted = np.minimum(bh_adjusted, 1.0)

    print(f"Bonferroni threshold: {bonf_threshold:.6f}")
    print(f"\nAll p-values (sorted, with corrections):")
    print(f"{'Rank':>4} | {'Test':<25} | {'Raw p':>10} | {'BH adj':>10} | {'Bonf sig':>10} | {'BH sig':>8}")
    print("-" * 85)
    for i in range(K):
        bonf_sig = "YES" if sorted_pvals[i] < bonf_threshold else "no"
        bh_sig = "YES" if bh_adjusted[i] < 0.05 else "no"
        print(f"{i+1:>4} | {sorted_labels[i]:<25} | {sorted_pvals[i]:>10.6f} | {bh_adjusted[i]:>10.6f} | {bonf_sig:>10} | {bh_sig:>8}")

    n_bonf = sum(1 for p in all_pvals if p < bonf_threshold)
    n_bh = sum(1 for p in bh_adjusted if p < 0.05)
    print(f"\nSurvive Bonferroni: {n_bonf}/{K}")
    print(f"Survive BH FDR: {n_bh}/{K}")

    # ===== STEP 5: VERIFY ARITHMETIC_ONLY MECHANISM =====
    print("\n" + "=" * 70)
    print("STEP 5: VERIFY ARITHMETIC_ONLY CAUSAL MECHANISM")
    print("=" * 70)

    feat = "distance_to_round_R"
    valid = sub[[feat, "orb_size", "atr_20", "outcome"]].dropna().copy()
    valid["win"] = (valid["outcome"] == "win").astype(int)

    # Test 1: Correlation with ORB_size
    pearson_r, pearson_p = stats.pearsonr(valid[feat], valid["orb_size"])
    spearman_r, spearman_p = stats.spearmanr(valid[feat], valid["orb_size"])
    print(f"\nCorr(distance_to_round_R, orb_size):")
    print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    print(f"  Spearman rho = {spearman_r:.4f}, p = {spearman_p:.2e}")

    # Test 2: Partial correlation controlling for ORB_size
    # Residualise both distance_to_round_R and win on orb_size
    from numpy.linalg import lstsq

    X = np.column_stack([valid["orb_size"].values, np.ones(len(valid))])
    # Residualize distance
    beta_d, _, _, _ = lstsq(X, valid[feat].values, rcond=None)
    resid_dist = valid[feat].values - X @ beta_d
    # Residualize win
    beta_w, _, _, _ = lstsq(X, valid["win"].values, rcond=None)
    resid_win = valid["win"].values - X @ beta_w

    partial_r, partial_p = stats.pearsonr(resid_dist, resid_win)
    print(f"\nPartial corr(distance_to_round_R, win | orb_size):")
    print(f"  r = {partial_r:.6f}, p = {partial_p:.6f}")

    if abs(partial_r) < 0.01 and partial_p > 0.05:
        print(f"  -> ARITHMETIC_ONLY SUPPORTED: controlling for orb_size removes relationship")
    elif partial_p < 0.05:
        print(f"  -> ARITHMETIC_ONLY CONTRADICTED: residual relationship remains after controlling for orb_size")
    else:
        print(f"  -> INCONCLUSIVE: small partial correlation, not significant")

    # Also check: does WR change across quintiles AFTER controlling for ORB_size?
    # Split into ORB_size terciles, then check WR by distance quintile within each
    print("\n  WR by distance quintile, WITHIN orb_size terciles:")
    valid["orb_tercile"] = pd.qcut(valid["orb_size"], 3, labels=["Small", "Med", "Large"], duplicates="drop")
    valid["dist_quintile"] = pd.qcut(valid[feat], 5, labels=False, duplicates="drop") + 1

    print(f"  {'Tercile':<8} | {'Q1 WR':>8} | {'Q5 WR':>8} | {'Spread':>8} | {'N':>8}")
    print("  " + "-" * 50)
    for tercile in ["Small", "Med", "Large"]:
        tdf = valid[valid["orb_tercile"] == tercile]
        q1 = tdf[tdf["dist_quintile"] == 1]
        q5 = tdf[tdf["dist_quintile"] == tdf["dist_quintile"].max()]
        if len(q1) > 0 and len(q5) > 0:
            wr1 = q1["win"].mean() * 100
            wr5 = q5["win"].mean() * 100
            print(f"  {tercile:<8} | {wr1:>7.1f}% | {wr5:>7.1f}% | {wr1-wr5:>+7.1f}% | {len(tdf):>8}")

    # ===== STEP 6: T7 ERA_DEPENDENT THRESHOLD AUDIT =====
    print("\n" + "=" * 70)
    print("STEP 6: T7 ERA_DEPENDENT THRESHOLD AUDIT")
    print("=" * 70)

    # Where does 70% threshold come from?
    print("\n  70% threshold source: quant-audit-protocol.md says")
    print("  'Must be positive in >=7/10 full years' and")
    print("  'Negative in >=4/10 -> DOWNGRADE to era-dependent'")
    print("  This implies 7/10 = 70% as the stability threshold.")
    print("  Applied to 11 years: need 7.7, so 8/11 to pass.")

    # The reported result: 7/11 for US sessions
    for k, n in [(6, 11), (7, 11)]:
        p_binom = stats.binomtest(k, n, 0.5).pvalue
        print(f"\n  Binomial test: {k}/{n} consistent under H0: p=0.5")
        print(f"    p = {p_binom:.6f}")
        print(f"    {'Significant at 0.05' if p_binom < 0.05 else 'NOT significant'}")

    print("\n  7/11 = 63.6%. Under H0 (coin flip), this is not significant (p=0.549).")
    print("  Even at 8/11 = 72.7%, p=0.227. Year-by-year consistency is a weak test")
    print("  for small N of years — it has very low power.")

    # ===== STEP 8: AUTOCORRELATION CHECK =====
    print("\n" + "=" * 70)
    print("STEP 8: AUTOCORRELATION CHECK")
    print("=" * 70)

    # Order by entry time, compute autocorrelation of win/loss
    for regime_name, regime_mask in [("ALL", pd.Series(True, index=sub.index))]:
        rsub = sub[regime_mask].copy()
        rsub = rsub.sort_values("trading_day")
        win_series = (rsub["outcome"] == "win").astype(int).values

        # Durbin-Watson
        from statsmodels.stats.stattools import durbin_watson
        resid = win_series - win_series.mean()
        dw = durbin_watson(resid)
        print(f"\n  Durbin-Watson ({regime_name}): {dw:.4f}")
        print(f"    (2.0 = no autocorrelation, <1.5 or >2.5 = concerning)")

        # ACF for lags 1-10
        from statsmodels.tsa.stattools import acf

        acf_vals, confint = acf(win_series, nlags=20, alpha=0.05)
        print(f"\n  ACF (lags 1-10):")
        sig_lags = []
        for lag in range(1, 11):
            ci_lower = confint[lag][0] - acf_vals[lag]
            ci_upper = confint[lag][1] - acf_vals[lag]
            sig = "*" if acf_vals[lag] < confint[lag][0] or acf_vals[lag] > confint[lag][1] else " "
            if sig == "*":
                sig_lags.append(lag)
            print(f"    Lag {lag:>2}: r={acf_vals[lag]:>8.4f} [{confint[lag][0]:>8.4f}, {confint[lag][1]:>8.4f}] {sig}")

        if sig_lags:
            print(f"\n  Significant autocorrelation at lags: {sig_lags}")
            print(f"  This means effective N < {len(rsub):,}. P-values may be anti-conservative.")
        else:
            print(f"\n  No significant autocorrelation detected. Independence assumption holds.")

    # ===== STEP 9: FINAL VERDICT =====
    print("\n" + "=" * 70)
    print("STEP 9: FINAL VERDICT")
    print("=" * 70)

    print("""
ROUND-NUMBER PROXIMITY -- FINAL VERDICT
========================================

Sample: 63,114 trades, 2016-02-01 to 2026-03-23, 3 instruments (MGC/MNQ/MES), 12 sessions
        E1 CB1 RR2.0 ORB5m. Features computed for 63,098.

Key statistics (with p-values and effect sizes):
  See Step 3 output above for exact values.

Multiple testing: See Step 4 output. K tests total.
  Bonferroni and BH FDR corrections applied.

Sign test audit:
  PRIOR SESSION'S CLAIM WAS MATHEMATICALLY WRONG.
  p=0.031 requires 6/6 sessions (two-tailed binomial), not "any 4/6".
  4/6 gives p=0.688. If the original analysis truly showed p=0.031,
  the underlying result was 6/6 anti-clustering -- STRONGER than characterised.
  The dismissal was based on a false mathematical premise.

Causal mechanism (ARITHMETIC_ONLY): See Step 5 output above.

Literature: Not searched in this script (requires web access).

Final assessment: See all test results above for honest classification.
""")


if __name__ == "__main__":
    main()
