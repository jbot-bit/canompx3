#!/usr/bin/env python3
"""
Contraction/Expansion research — testing Crabel's principle at session ORB level.

Core question: Does the ORB *break* predict follow-through? Or is the break
theater and the real signal is volatility expansion from contraction?

Three tests:
  1. EXPANSION RATIO: today's ORB / median of last N same-session ORBs.
     Does high expansion predict better follow-through (avgR, MFE)?
  2. NR-STYLE CONTRACTION: was yesterday's ORB the smallest of last N?
     Does contraction predict NEXT session's follow-through?
  3. HONEST COMPARISON: after controlling for absolute ORB size,
     does expansion ratio add ANY information?

Follow-through is measured by MFE distribution (how far price actually goes),
NOT just win/loss at a specific RR target. The break is trivial (100% on MNQ).
What matters is distance traveled after break.

Date: Feb 2026. Read-only — does NOT write to the database.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "local_db" / "gold.db"

# All instruments × their active sessions
INSTRUMENTS = {
    "MGC": ["0900", "1000", "1800"],
    "MNQ": ["0900", "1000", "1130"],
    "MES": ["0900", "1000", "0030"],
}

# Lookback windows for contraction/expansion measurement
LOOKBACKS = [5, 7, 10]

# Expansion ratio thresholds to test
EXPANSION_THRESHOLDS = [1.0, 1.25, 1.5, 2.0, 3.0]

# Use RR=4.0 CB=1 E1 to get the widest MFE window — we want to see
# how far price ACTUALLY goes, not whether it hit a specific target
REFERENCE_RR = 4.0
REFERENCE_CB = 1
REFERENCE_EM = "E1"


def load_data(instrument: str, session: str) -> pd.DataFrame:
    """Load ORB outcomes joined with daily features for one instrument/session.

    Returns DataFrame with: trading_day, orb_size, pnl_r, mfe_r, mae_r,
    outcome, break_dir, double_break, atr_20, us_dst, uk_dst.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT
                o.trading_day,
                d.orb_{session}_size AS orb_size,
                d.orb_{session}_break_dir AS break_dir,
                d.orb_{session}_double_break AS double_break,
                d.atr_20,
                d.us_dst,
                d.uk_dst,
                o.pnl_r,
                o.mfe_r,
                o.mae_r,
                o.outcome,
                o.entry_ts,
                o.entry_price,
                o.stop_price
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.symbol = d.symbol
                AND o.trading_day = d.trading_day
                AND d.orb_minutes = 5
            WHERE o.symbol = $1
                AND o.orb_label = $2
                AND o.orb_minutes = 5
                AND o.rr_target = $3
                AND o.confirm_bars = $4
                AND o.entry_model = $5
                AND o.entry_ts IS NOT NULL
                AND o.outcome IS NOT NULL
                AND o.pnl_r IS NOT NULL
            ORDER BY o.trading_day
        """.replace("{session}", session),
            [instrument, session, REFERENCE_RR, REFERENCE_CB, REFERENCE_EM]
        ).fetchdf()
    finally:
        con.close()
    return df


def compute_expansion_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Add contraction/expansion features to the dataframe.

    For each row:
    - rolling_median: median ORB size over last `lookback` sessions
    - rolling_min: min ORB size over last `lookback` sessions
    - rolling_max: max ORB size over last `lookback` sessions
    - expansion_ratio: today's ORB / rolling_median (>1 = expanding)
    - is_nr: today's ORB is the smallest of last `lookback` sessions
    - post_nr: yesterday was NR (contraction just ended)
    - rank_in_window: where today's ORB ranks (1=smallest, lookback=largest)
    """
    col = "orb_size"
    df = df.copy()

    # Rolling stats — shift(1) so we compare to PRIOR sessions, not including today
    # Actually: we want today's ORB relative to recent context.
    # The ORB forms at session start. We KNOW today's ORB when we decide to trade.
    # But we compare to PRIOR sessions' ORBs (no lookahead).
    prior_sizes = df[col].shift(1)

    df[f"rolling_median_{lookback}"] = prior_sizes.rolling(lookback, min_periods=lookback).median()
    df[f"rolling_min_{lookback}"] = prior_sizes.rolling(lookback, min_periods=lookback).min()
    df[f"rolling_max_{lookback}"] = prior_sizes.rolling(lookback, min_periods=lookback).max()
    df[f"rolling_std_{lookback}"] = prior_sizes.rolling(lookback, min_periods=lookback).std()

    # Expansion ratio: today vs recent median
    df[f"expansion_ratio_{lookback}"] = df[col] / df[f"rolling_median_{lookback}"]

    # NR detection: was the PREVIOUS session the narrowest of its lookback window?
    # If so, today is "post-contraction" — Crabel's expansion signal
    def is_nr(series, lb):
        """Is each value the minimum of the preceding lb values (inclusive)?"""
        result = pd.Series(False, index=series.index)
        for i in range(lb, len(series)):
            window = series.iloc[i - lb + 1:i + 1]
            if len(window) == lb and series.iloc[i] == window.min():
                result.iloc[i] = True
        return result

    nr_flags = is_nr(df[col], lookback)
    df[f"is_nr_{lookback}"] = nr_flags
    df[f"post_nr_{lookback}"] = nr_flags.shift(1).fillna(False)

    # Z-score: how unusual is today's ORB vs recent distribution
    df[f"orb_zscore_{lookback}"] = (
        (df[col] - df[f"rolling_median_{lookback}"]) / df[f"rolling_std_{lookback}"]
    )

    return df


def follow_through_stats(pnl_r: np.ndarray, mfe_r: np.ndarray,
                          label: str = "") -> dict:
    """Compute follow-through statistics from R-multiples.

    This is the HONEST measure — not just WR at one target,
    but the full distribution of how far price goes.
    """
    n = len(pnl_r)
    if n == 0:
        return {"label": label, "n": 0}

    # Basic stats
    avg_r = float(np.mean(pnl_r))
    std_r = float(np.std(pnl_r))
    wr = float((pnl_r > 0).sum() / n)
    total_r = float(np.sum(pnl_r))
    sharpe = avg_r / std_r if std_r > 0 else 0.0

    # MFE distribution — how far does price actually go FOR you?
    # This is the follow-through curve
    mfe_valid = mfe_r[~np.isnan(mfe_r)]
    mfe_n = len(mfe_valid)

    ft_stats = {}
    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pct = float((mfe_valid >= threshold).sum() / mfe_n) if mfe_n > 0 else 0.0
        ft_stats[f"mfe_gte_{threshold}R"] = pct

    # Median MFE — typical best price during trade
    median_mfe = float(np.median(mfe_valid)) if mfe_n > 0 else 0.0

    return {
        "label": label,
        "n": n,
        "avg_r": avg_r,
        "wr": wr,
        "total_r": total_r,
        "sharpe": sharpe,
        "median_mfe": median_mfe,
        **ft_stats,
    }


def test_expansion_ratio(df: pd.DataFrame, lookback: int,
                          instrument: str, session: str) -> list[dict]:
    """TEST 1: Does high expansion ratio predict better follow-through?"""
    col_ratio = f"expansion_ratio_{lookback}"
    valid = df.dropna(subset=[col_ratio, "pnl_r", "mfe_r"])

    if len(valid) < 30:
        return []

    results = []

    # Baseline: ALL trades (no expansion filter)
    baseline = follow_through_stats(
        valid["pnl_r"].values, valid["mfe_r"].values,
        label=f"ALL (N={len(valid)})"
    )
    baseline.update({"instrument": instrument, "session": session,
                     "lookback": lookback, "test": "expansion_ratio",
                     "threshold": "ALL"})
    results.append(baseline)

    # Split by expansion ratio thresholds
    for thresh in EXPANSION_THRESHOLDS:
        expanding = valid[valid[col_ratio] >= thresh]
        contracting = valid[valid[col_ratio] < thresh]

        if len(expanding) >= 10:
            e_stats = follow_through_stats(
                expanding["pnl_r"].values, expanding["mfe_r"].values,
                label=f"EXPAND>={thresh}x (N={len(expanding)})"
            )
            e_stats.update({"instrument": instrument, "session": session,
                           "lookback": lookback, "test": "expansion_ratio",
                           "threshold": f">={thresh}"})
            results.append(e_stats)

        if len(contracting) >= 10:
            c_stats = follow_through_stats(
                contracting["pnl_r"].values, contracting["mfe_r"].values,
                label=f"CONTRACT<{thresh}x (N={len(contracting)})"
            )
            c_stats.update({"instrument": instrument, "session": session,
                           "lookback": lookback, "test": "expansion_ratio",
                           "threshold": f"<{thresh}"})
            results.append(c_stats)

    # Correlation: expansion_ratio vs pnl_r and mfe_r
    r_pnl, p_pnl = sp_stats.pearsonr(valid[col_ratio], valid["pnl_r"])
    r_mfe, p_mfe = sp_stats.pearsonr(valid[col_ratio], valid["mfe_r"])

    results.append({
        "instrument": instrument, "session": session,
        "lookback": lookback, "test": "expansion_ratio_correlation",
        "threshold": "continuous",
        "label": f"r(expansion,pnl)={r_pnl:.3f} p={p_pnl:.4f} | r(expansion,mfe)={r_mfe:.3f} p={p_mfe:.4f}",
        "n": len(valid),
        "avg_r": r_pnl,  # store correlation in avg_r field for CSV
        "wr": p_pnl,     # store p-value in wr field for CSV
        "median_mfe": r_mfe,
    })

    return results


def test_nr_contraction(df: pd.DataFrame, lookback: int,
                         instrument: str, session: str) -> list[dict]:
    """TEST 2: Does prior contraction (NR-style) predict follow-through?

    Crabel's principle: contraction → expansion. So if yesterday's ORB
    was the narrowest of last N sessions, today should expand WITH follow-through.
    """
    col_post_nr = f"post_nr_{lookback}"
    valid = df.dropna(subset=[col_post_nr, "pnl_r", "mfe_r"])

    if len(valid) < 30:
        return []

    results = []

    post_nr = valid[valid[col_post_nr] == True]
    not_nr = valid[valid[col_post_nr] == False]

    if len(post_nr) >= 5:
        nr_stats = follow_through_stats(
            post_nr["pnl_r"].values, post_nr["mfe_r"].values,
            label=f"POST-NR{lookback} (N={len(post_nr)})"
        )
        nr_stats.update({"instrument": instrument, "session": session,
                        "lookback": lookback, "test": "post_nr",
                        "threshold": "post_nr=True"})
        results.append(nr_stats)

    if len(not_nr) >= 10:
        notnr_stats = follow_through_stats(
            not_nr["pnl_r"].values, not_nr["mfe_r"].values,
            label=f"NOT-NR{lookback} (N={len(not_nr)})"
        )
        notnr_stats.update({"instrument": instrument, "session": session,
                           "lookback": lookback, "test": "post_nr",
                           "threshold": "post_nr=False"})
        results.append(notnr_stats)

    # If enough data, test statistical significance
    if len(post_nr) >= 10 and len(not_nr) >= 10:
        t_stat, p_val = sp_stats.ttest_ind(
            post_nr["pnl_r"].values, not_nr["pnl_r"].values,
            equal_var=False
        )
        mfe_t, mfe_p = sp_stats.ttest_ind(
            post_nr["mfe_r"].values, not_nr["mfe_r"].values,
            equal_var=False
        )
        results.append({
            "instrument": instrument, "session": session,
            "lookback": lookback, "test": "post_nr_significance",
            "threshold": "t-test",
            "label": f"pnl t={t_stat:.2f} p={p_val:.4f} | mfe t={mfe_t:.2f} p={mfe_p:.4f}",
            "n": len(post_nr),
            "avg_r": float(post_nr["pnl_r"].mean() - not_nr["pnl_r"].mean()),
        })

    return results


def test_expansion_vs_absolute(df: pd.DataFrame, lookback: int,
                                instrument: str, session: str) -> list[dict]:
    """TEST 3: After controlling for absolute size, does expansion ratio add info?

    This is the HONEST test. If expansion ratio is just correlated with
    absolute ORB size (big ORBs tend to be expanding), then it adds nothing
    beyond the existing G4+/G5+ filter. We need to show it's INDEPENDENT.
    """
    col_ratio = f"expansion_ratio_{lookback}"
    valid = df.dropna(subset=[col_ratio, "pnl_r", "mfe_r", "orb_size"])

    if len(valid) < 50:
        return []

    results = []

    # First: how correlated is expansion ratio with absolute size?
    r_size, p_size = sp_stats.pearsonr(valid[col_ratio], valid["orb_size"])
    results.append({
        "instrument": instrument, "session": session,
        "lookback": lookback, "test": "confound_check",
        "threshold": "ratio_vs_abs_size",
        "label": f"r(expansion_ratio, abs_size)={r_size:.3f} p={p_size:.4f}",
        "n": len(valid), "avg_r": r_size,
    })

    # Second: within SIZE BANDS, does expansion ratio still predict?
    # This is the key test — control for absolute size, then check expansion
    size_bands = [
        ("small", 0, valid["orb_size"].quantile(0.33)),
        ("medium", valid["orb_size"].quantile(0.33), valid["orb_size"].quantile(0.67)),
        ("large", valid["orb_size"].quantile(0.67), valid["orb_size"].max() + 1),
    ]

    for band_name, lo, hi in size_bands:
        band = valid[(valid["orb_size"] >= lo) & (valid["orb_size"] < hi)]
        if len(band) < 20:
            continue

        # Within this size band, split by expansion ratio (above/below median)
        median_ratio = band[col_ratio].median()
        expanding = band[band[col_ratio] >= median_ratio]
        contracting = band[band[col_ratio] < median_ratio]

        if len(expanding) >= 10 and len(contracting) >= 10:
            e_avg = expanding["pnl_r"].mean()
            c_avg = contracting["pnl_r"].mean()
            e_mfe = expanding["mfe_r"].mean()
            c_mfe = contracting["mfe_r"].mean()
            diff = e_avg - c_avg

            t_stat, p_val = sp_stats.ttest_ind(
                expanding["pnl_r"].values, contracting["pnl_r"].values,
                equal_var=False
            )

            results.append({
                "instrument": instrument, "session": session,
                "lookback": lookback, "test": "expansion_within_size_band",
                "threshold": band_name,
                "label": (f"SIZE={band_name} ({lo:.1f}-{hi:.1f}pt): "
                         f"expanding={e_avg:.3f}R(N={len(expanding)}) "
                         f"contracting={c_avg:.3f}R(N={len(contracting)}) "
                         f"diff={diff:+.3f}R p={p_val:.4f}"),
                "n": len(band), "avg_r": diff,
                "wr": p_val,
                "median_mfe": e_mfe - c_mfe,
            })

    return results


def test_follow_through_curve(df: pd.DataFrame, instrument: str,
                               session: str) -> list[dict]:
    """BONUS: The raw follow-through curve — how far does price go?

    This answers the user's fundamental question: "broke the same means
    nothing if it doesn't go the distance." How often does it go the distance?
    """
    valid = df.dropna(subset=["mfe_r", "orb_size"])

    if len(valid) < 30:
        return []

    results = []

    # Follow-through curve by absolute ORB size bands
    pcts = [0, 25, 50, 75, 90, 100]
    size_quartiles = np.percentile(valid["orb_size"].values,
                                    [0, 25, 50, 75, 100])

    for i in range(len(size_quartiles) - 1):
        lo, hi = size_quartiles[i], size_quartiles[i + 1]
        band = valid[(valid["orb_size"] >= lo) &
                     (valid["orb_size"] <= (hi if i == len(size_quartiles) - 2 else hi - 0.001))]

        if len(band) < 10:
            continue

        mfe = band["mfe_r"].values
        pnl = band["pnl_r"].values

        # What % of trades reach each distance?
        ft_data = {f"reach_{t}R": float((mfe >= t).sum() / len(mfe))
                   for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]}

        results.append({
            "instrument": instrument, "session": session,
            "test": "follow_through_curve",
            "threshold": f"Q{i+1}({lo:.1f}-{hi:.1f}pt)",
            "label": (f"SIZE Q{i+1} ({lo:.1f}-{hi:.1f}pt, N={len(band)}): "
                     f"avgR={np.mean(pnl):.3f} medianMFE={np.median(mfe):.2f}R "
                     f"reach1R={ft_data['reach_1.0R']:.0%} "
                     f"reach2R={ft_data['reach_2.0R']:.0%} "
                     f"reach3R={ft_data['reach_3.0R']:.0%}"),
            "n": len(band),
            "avg_r": float(np.mean(pnl)),
            "median_mfe": float(np.median(mfe)),
            **ft_data,
        })

    return results


def main():
    print("=" * 80)
    print("CONTRACTION / EXPANSION RESEARCH")
    print("Crabel's principle applied to session-level ORBs")
    print("=" * 80)
    print(f"\nDB: {DB_PATH}")
    print(f"Reference: RR={REFERENCE_RR} CB={REFERENCE_CB} EM={REFERENCE_EM}")
    print(f"  (Using highest RR to capture full MFE distribution)")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Instruments: {list(INSTRUMENTS.keys())}")

    all_results = []
    all_ft_curves = []

    for instrument, sessions in INSTRUMENTS.items():
        for session in sessions:
            print(f"\n{'─' * 60}")
            print(f"  {instrument} / {session}")
            print(f"{'─' * 60}")

            df = load_data(instrument, session)
            print(f"  Loaded {len(df):,} trades")

            if len(df) < 30:
                print(f"  SKIP: insufficient data")
                continue

            # Follow-through curve (no lookback needed)
            ft_results = test_follow_through_curve(df, instrument, session)
            all_ft_curves.extend(ft_results)
            print(f"\n  FOLLOW-THROUGH CURVE:")
            for r in ft_results:
                print(f"    {r['label']}")

            for lb in LOOKBACKS:
                df_exp = compute_expansion_features(df, lb)
                valid_count = df_exp.dropna(
                    subset=[f"expansion_ratio_{lb}"]).shape[0]
                print(f"\n  LOOKBACK={lb} ({valid_count} valid rows)")

                # Test 1: Expansion ratio
                exp_results = test_expansion_ratio(df_exp, lb, instrument, session)
                all_results.extend(exp_results)
                for r in exp_results:
                    if "correlation" in r.get("test", ""):
                        print(f"    CORR: {r['label']}")
                    elif r.get("threshold") == "ALL":
                        print(f"    BASELINE: avgR={r['avg_r']:.3f} "
                              f"WR={r['wr']:.0%} medMFE={r['median_mfe']:.2f}R")

                # Show expansion splits
                for r in exp_results:
                    thresh = r.get("threshold", "")
                    if thresh.startswith(">=") and r.get("n", 0) >= 10:
                        print(f"    {r['label']}: avgR={r['avg_r']:.3f} "
                              f"WR={r.get('wr', 0):.0%} medMFE={r.get('median_mfe', 0):.2f}R")

                # Test 2: NR contraction
                nr_results = test_nr_contraction(df_exp, lb, instrument, session)
                all_results.extend(nr_results)
                for r in nr_results:
                    if r.get("test") == "post_nr":
                        n = r.get("n", 0)
                        ar = r.get("avg_r", 0)
                        print(f"    {r['label']}: avgR={ar:.3f}")
                    elif r.get("test") == "post_nr_significance":
                        print(f"    NR SIGNIFICANCE: {r['label']}")

                # Test 3: Expansion vs absolute size
                vs_results = test_expansion_vs_absolute(df_exp, lb, instrument, session)
                all_results.extend(vs_results)
                for r in vs_results:
                    if r.get("test") == "confound_check":
                        print(f"    CONFOUND: {r['label']}")
                    elif r.get("test") == "expansion_within_size_band":
                        print(f"    {r['label']}")

    # =========================================================================
    # HONEST SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("HONEST SUMMARY")
    print("=" * 80)

    # Check: does expansion ratio predict AT ALL?
    corr_results = [r for r in all_results if r.get("test") == "expansion_ratio_correlation"]
    print("\n1. EXPANSION RATIO → PNL CORRELATION (all instruments/sessions):")
    for r in corr_results:
        sig = "***" if abs(r.get("wr", 1)) < 0.01 else "**" if abs(r.get("wr", 1)) < 0.05 else ""
        print(f"   {r['instrument']} {r['session']} lb={r['lookback']}: {r['label']} {sig}")

    # Check: confound with absolute size
    confound_results = [r for r in all_results if r.get("test") == "confound_check"]
    print("\n2. CONFOUND: EXPANSION RATIO vs ABSOLUTE SIZE:")
    for r in confound_results:
        print(f"   {r['instrument']} {r['session']} lb={r['lookback']}: {r['label']}")

    # Check: within-band results
    band_results = [r for r in all_results if r.get("test") == "expansion_within_size_band"]
    print("\n3. EXPANSION WITHIN SIZE BANDS (controlled for absolute size):")
    sig_count = 0
    for r in band_results:
        p = r.get("wr", 1)
        sig_marker = " ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.10 else ""
        print(f"   {r['instrument']} {r['session']} lb={r['lookback']}: {r['label']}{sig_marker}")
        if p < 0.05:
            sig_count += 1
    total_band = len(band_results)
    expected_false = total_band * 0.05
    print(f"\n   Significant at p<0.05: {sig_count}/{total_band} "
          f"(expected by chance: {expected_false:.1f})")

    # NR results
    nr_sig_results = [r for r in all_results if r.get("test") == "post_nr_significance"]
    print("\n4. NR CONTRACTION → NEXT SESSION FOLLOW-THROUGH:")
    for r in nr_sig_results:
        print(f"   {r['instrument']} {r['session']} lb={r['lookback']}: {r['label']}")

    # Follow-through curves
    print("\n5. RAW FOLLOW-THROUGH CURVES (the actual question — does it go the distance?):")
    for r in all_ft_curves:
        print(f"   {r['instrument']} {r['session']}: {r['label']}")

    # Final verdict
    print("\n" + "=" * 80)
    print("SURVIVED SCRUTINY: [to be filled after review]")
    print("DID NOT SURVIVE: [to be filled after review]")
    print("CAVEATS:")
    print("  - All results are IN-SAMPLE on available data")
    print("  - MGC has ~5yr data, MNQ/MES have ~2yr (579 overlapping days)")
    print("  - Multiple comparisons: tested ~100+ combinations")
    print("  - Benjamini-Hochberg correction needed for any claimed discoveries")
    print("  - Expansion ratio may simply be a proxy for absolute size")
    print("NEXT STEPS:")
    print("  - If signal found: walk-forward validation")
    print("  - If expansion ratio is just size proxy: the absolute filter IS the edge")
    print("  - If NR predicts: build into daily_features as a new column")
    print("=" * 80)

    # Save detailed results to CSV
    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(exist_ok=True)

    if all_results:
        pd.DataFrame(all_results).to_csv(
            out_dir / "contraction_expansion_results.csv", index=False)
        print(f"\nDetailed results: {out_dir / 'contraction_expansion_results.csv'}")

    if all_ft_curves:
        pd.DataFrame(all_ft_curves).to_csv(
            out_dir / "follow_through_curves.csv", index=False)
        print(f"Follow-through curves: {out_dir / 'follow_through_curves.csv'}")


if __name__ == "__main__":
    main()
