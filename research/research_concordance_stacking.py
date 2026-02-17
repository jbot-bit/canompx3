#!/usr/bin/env python3
"""
Concordance Filter Stacking Analysis.

Tests whether 3-instrument concordance (MGC, MES, MNQ all breaking same
direction) adds edge on top of existing G4+/G5+ ORB strategies.

Analysis 1: Concordance as overlay filter
  For each shared session (0900, 1000, 1100, 1800), split orb_outcomes
  into three buckets:
    - 3/3 Concordant: all 3 instruments broke same direction
    - 2/3 Majority: 2 of 3 broke same, 1 opposite or no-break
    - Remaining: no clear majority (split, 0-1 instruments broke)
  Compute WR, ExpR, Sharpe per bucket, compare to unfiltered baseline.
  If edge degrades gradually across tiers, the mechanism is real.

Analysis 2: Concordance vs G4+ overlap (independence sanity check)
  What % of concordant-3 days are also G4+ eligible?
  If overlap is >90%, concordance is just a noisier ORB-size proxy.
  If overlap is 60-70%, concordance is genuinely independent signal.

Analysis 3: MGC 2300 ORB size as volatility gate for MES/MNQ 0030
  Split MES/MNQ 0030 outcomes by MGC 2300 ORB size (large vs small
  relative to ATR), using actual orb_outcomes pnl_r for real strategy
  metrics instead of daily_features RR=1.0 outcome.

Usage:
    python research/research_concordance_stacking.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

# =========================================================================
# Configuration
# =========================================================================

INSTRUMENTS = ["MGC", "MES", "MNQ"]
SHARED_SESSIONS = ["0900", "1000", "1100", "1800"]
MIN_SAMPLE = 30

# Representative strategy parameters per session (from validated_setups)
# These are the best-performing combos -- we test concordance on each.
STRATEGY_COMBOS = [
    # 0900 momentum
    {"orb_label": "0900", "entry_model": "E1", "confirm_bars": 2, "rr_target": 2.5},
    {"orb_label": "0900", "entry_model": "E1", "confirm_bars": 3, "rr_target": 2.0},
    # 1000 various
    {"orb_label": "1000", "entry_model": "E1", "confirm_bars": 5, "rr_target": 1.0},
    {"orb_label": "1000", "entry_model": "E1", "confirm_bars": 5, "rr_target": 3.0},
    {"orb_label": "1000", "entry_model": "E3", "confirm_bars": 1, "rr_target": 1.0},
    # 1100
    {"orb_label": "1100", "entry_model": "E1", "confirm_bars": 3, "rr_target": 4.0},
    {"orb_label": "1100", "entry_model": "E1", "confirm_bars": 4, "rr_target": 2.5},
    # 1800 retrace
    {"orb_label": "1800", "entry_model": "E3", "confirm_bars": 1, "rr_target": 2.0},
    {"orb_label": "1800", "entry_model": "E1", "confirm_bars": 1, "rr_target": 2.0},
]

# ORB size filters to test
SIZE_FILTERS = {"G4": 4.0, "G5": 5.0, "G6": 6.0}


# =========================================================================
# Data Loading
# =========================================================================


def load_data(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily_features and orb_outcomes for cross-instrument analysis."""
    features = con.execute("""
        SELECT trading_day, symbol, atr_20,
            orb_0900_break_dir, orb_0900_size,
            orb_1000_break_dir, orb_1000_size,
            orb_1100_break_dir, orb_1100_size,
            orb_1800_break_dir, orb_1800_size,
            orb_2300_break_dir, orb_2300_size,
            orb_0030_break_dir, orb_0030_size
        FROM daily_features
        WHERE symbol IN ('MGC', 'MES', 'MNQ')
          AND orb_minutes = 5
        ORDER BY trading_day, symbol
    """).fetchdf()

    outcomes = con.execute("""
        SELECT trading_day, symbol, orb_label, rr_target, confirm_bars,
               entry_model, outcome, pnl_r
        FROM orb_outcomes
        WHERE symbol IN ('MGC', 'MES', 'MNQ')
          AND orb_label IN ('0900', '1000', '1100', '1800', '2300', '0030')
          AND outcome IS NOT NULL
        ORDER BY trading_day, symbol, orb_label
    """).fetchdf()

    return features, outcomes


def build_concordance_map(features: pd.DataFrame) -> pd.DataFrame:
    """Build per-day concordance classification for each shared session.

    Returns DataFrame with columns:
        trading_day, session, concordance_tier, majority_dir,
        n_active, n_long, n_short,
        MGC_break_dir, MES_break_dir, MNQ_break_dir,
        MGC_orb_size, MES_orb_size, MNQ_orb_size
    """
    # Pivot to wide format
    dfs = {}
    for inst in INSTRUMENTS:
        sub = features[features["symbol"] == inst].copy()
        rename = {"atr_20": f"{inst}_atr_20"}
        for sess in SHARED_SESSIONS + ["2300", "0030"]:
            rename[f"orb_{sess}_break_dir"] = f"{inst}_{sess}_break_dir"
            rename[f"orb_{sess}_size"] = f"{inst}_{sess}_size"
        sub = sub.rename(columns=rename)
        keep = ["trading_day"] + list(rename.values())
        dfs[inst] = sub[[c for c in keep if c in sub.columns]]

    wide = dfs["MGC"]
    for inst in ["MES", "MNQ"]:
        wide = wide.merge(dfs[inst], on="trading_day", how="inner")

    rows = []
    for sess in SHARED_SESSIONS:
        dirs = {i: wide[f"{i}_{sess}_break_dir"] for i in INSTRUMENTS}
        sizes = {i: wide[f"{i}_{sess}_size"] for i in INSTRUMENTS}

        is_long = pd.DataFrame({i: dirs[i] == "long" for i in INSTRUMENTS})
        is_short = pd.DataFrame({i: dirs[i] == "short" for i in INSTRUMENTS})
        has_break = is_long | is_short

        n_long = is_long.sum(axis=1)
        n_short = is_short.sum(axis=1)
        n_active = has_break.sum(axis=1)

        conc3 = (n_active == 3) & ((n_long == 3) | (n_short == 3))
        maj2 = ~conc3 & (n_active >= 2) & ((n_long >= 2) | (n_short >= 2))

        tier = pd.Series("remaining", index=wide.index)
        tier[conc3] = "concordant_3"
        tier[maj2] = "majority_2"

        maj_dir = np.where(n_long >= n_short, "long", "short")
        maj_dir = np.where(n_active < 2, "none", maj_dir)

        sess_df = pd.DataFrame({
            "trading_day": wide["trading_day"],
            "session": sess,
            "concordance_tier": tier,
            "majority_dir": maj_dir,
            "n_active": n_active,
            "n_long": n_long,
            "n_short": n_short,
        })
        for i in INSTRUMENTS:
            sess_df[f"{i}_break_dir"] = dirs[i].values
            sess_df[f"{i}_orb_size"] = sizes[i].values

        rows.append(sess_df)

    return pd.concat(rows, ignore_index=True)


def build_mgc_2300_context(features: pd.DataFrame) -> pd.DataFrame:
    """Build per-day MGC 2300 context for MES/MNQ 0030 gate analysis."""
    dfs = {}
    for inst in INSTRUMENTS:
        sub = features[features["symbol"] == inst].copy()
        rename = {"atr_20": f"{inst}_atr_20"}
        for sess in ["2300", "0030"]:
            rename[f"orb_{sess}_break_dir"] = f"{inst}_{sess}_break_dir"
            rename[f"orb_{sess}_size"] = f"{inst}_{sess}_size"
        sub = sub.rename(columns=rename)
        keep = ["trading_day"] + list(rename.values())
        dfs[inst] = sub[[c for c in keep if c in sub.columns]]

    wide = dfs["MGC"]
    for inst in ["MES", "MNQ"]:
        wide = wide.merge(dfs[inst], on="trading_day", how="inner")

    # Compute MGC 2300 ORB size / ATR ratio
    has_data = (wide["MGC_2300_break_dir"].notna() &
                wide["MGC_2300_size"].notna() &
                wide["MGC_atr_20"].notna() &
                (wide["MGC_atr_20"] > 0))
    wide["mgc_2300_size_atr"] = np.where(
        has_data,
        wide["MGC_2300_size"] / wide["MGC_atr_20"],
        np.nan,
    )
    median_ratio = wide.loc[has_data, "mgc_2300_size_atr"].median()
    wide["mgc_2300_size_bucket"] = np.where(
        ~has_data, "no_data",
        np.where(wide["mgc_2300_size_atr"] >= median_ratio, "large", "small"),
    )
    wide["mgc_2300_median_ratio"] = median_ratio

    return wide


# =========================================================================
# Metrics
# =========================================================================


def compute_metrics(pnl_r: pd.Series) -> dict:
    """Compute WR, ExpR, Sharpe from pnl_r series."""
    valid = pnl_r.dropna()
    n = len(valid)
    if n == 0:
        return {"n": 0, "wr": None, "expr": None, "sharpe": None}

    wins = (valid > 0).sum()
    wr = wins / n
    expr = valid.mean()
    std = valid.std(ddof=1)
    sharpe = (expr / std * np.sqrt(252)) if std > 0 and n >= 2 else None

    return {"n": n, "wr": wr, "expr": expr, "sharpe": sharpe}


def sig_flag(n: int) -> str:
    return "SIG" if n >= MIN_SAMPLE else f"INSUFFICIENT (N={n})"


def fmt_wr(wr):
    return f"{wr:.1%}" if wr is not None else "N/A"


def fmt_f(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


# =========================================================================
# Analysis 1: Concordance as overlay filter
# =========================================================================


def analysis_1_concordance_overlay(conc_map: pd.DataFrame,
                                   outcomes: pd.DataFrame):
    """Test concordance tiers on top of G4+/G5+ strategies."""
    print("\n--- ANALYSIS 1: CONCORDANCE AS OVERLAY FILTER ---")
    print("Three tiers: 3/3 concordant > 2/3 majority > remaining")
    print("Testing whether edge degrades gradually across tiers.")
    print()

    tiers = ["concordant_3", "majority_2", "remaining"]
    tier_labels = {"concordant_3": "3/3 Conc", "majority_2": "2/3 Maj", "remaining": "Rest"}

    for combo in STRATEGY_COMBOS:
        sess = combo["orb_label"]
        em = combo["entry_model"]
        cb = combo["confirm_bars"]
        rr = combo["rr_target"]
        combo_label = f"{sess} {em}/CB{cb}/RR{rr}"

        # Get outcomes matching this combo for all instruments at this session
        mask = ((outcomes["orb_label"] == sess) &
                (outcomes["entry_model"] == em) &
                (outcomes["confirm_bars"] == cb) &
                (outcomes["rr_target"] == rr))
        combo_outcomes = outcomes[mask].copy()

        if len(combo_outcomes) == 0:
            continue

        # Get concordance map for this session
        sess_conc = conc_map[conc_map["session"] == sess][
            ["trading_day", "concordance_tier", "majority_dir"]
        ].copy()

        for size_label, min_size in SIZE_FILTERS.items():
            print(f"  {combo_label} / {size_label}+:")

            # Join outcomes with concordance and apply size filter
            merged = combo_outcomes.merge(sess_conc, on="trading_day", how="inner")

            # Apply ORB size filter per instrument
            # Need orb_size from concordance map
            conc_with_size = conc_map[conc_map["session"] == sess].copy()
            for inst in INSTRUMENTS:
                inst_merged = merged[merged["symbol"] == inst].copy()
                inst_sizes = conc_with_size[["trading_day", f"{inst}_orb_size"]].rename(
                    columns={f"{inst}_orb_size": "_orb_size"})
                inst_merged = inst_merged.merge(inst_sizes, on="trading_day", how="left")
                inst_merged = inst_merged[inst_merged["_orb_size"] >= min_size]
                merged = pd.concat([
                    merged[merged["symbol"] != inst],
                    inst_merged.drop(columns=["_orb_size"]),
                ], ignore_index=True)

            if len(merged) == 0:
                print(f"    (no data after {size_label} filter)")
                print()
                continue

            # Baseline (all tiers combined)
            bl = compute_metrics(merged["pnl_r"])
            print(f"    Baseline: WR={fmt_wr(bl['wr'])} ExpR={fmt_f(bl['expr'])} "
                  f"Sharpe={fmt_f(bl['sharpe'])} N={bl['n']}")

            # Per tier
            monotonic = True
            prev_wr = None
            for tier in tiers:
                tier_data = merged[merged["concordance_tier"] == tier]
                m = compute_metrics(tier_data["pnl_r"])
                flag = sig_flag(m["n"])
                print(f"    {tier_labels[tier]:>8}: WR={fmt_wr(m['wr'])} "
                      f"ExpR={fmt_f(m['expr'])} Sharpe={fmt_f(m['sharpe'])} "
                      f"N={m['n']:>4} {flag}")

                if m["wr"] is not None and prev_wr is not None:
                    if m["wr"] > prev_wr:
                        monotonic = False
                if m["wr"] is not None:
                    prev_wr = m["wr"]

            if monotonic and prev_wr is not None:
                print(f"    >> Monotonic degradation: YES (concordance -> edge)")
            else:
                print(f"    >> Monotonic degradation: NO")
            print()


# =========================================================================
# Analysis 2: Concordance vs G4+ overlap
# =========================================================================


def analysis_2_overlap(conc_map: pd.DataFrame):
    """Check independence of concordance-3 from ORB size filters."""
    print("\n--- ANALYSIS 2: CONCORDANCE vs ORB SIZE OVERLAP ---")
    print("If concordant-3 days are >90% G4+ days, concordance is redundant.")
    print("If overlap is 60-70%, concordance is genuinely independent signal.")
    print()

    for sess in SHARED_SESSIONS:
        sess_data = conc_map[conc_map["session"] == sess]
        conc3_days = sess_data[sess_data["concordance_tier"] == "concordant_3"]
        all_days = sess_data

        print(f"  Session {sess} ({len(all_days)} total days, "
              f"{len(conc3_days)} concordant-3 days):")

        for inst in INSTRUMENTS:
            size_col = f"{inst}_orb_size"
            if size_col not in sess_data.columns:
                continue

            for filt_name, min_size in SIZE_FILTERS.items():
                # G4+ eligible days (this instrument)
                g_eligible = all_days[all_days[size_col] >= min_size]
                n_g = len(g_eligible)

                # Concordant-3 AND G4+ eligible
                conc3_g = conc3_days[conc3_days[size_col] >= min_size]
                n_both = len(conc3_g)

                # Concordant-3 total
                n_conc3 = len(conc3_days)

                if n_conc3 > 0:
                    overlap_pct = n_both / n_conc3
                    g_pct = n_g / len(all_days) if len(all_days) > 0 else 0
                    print(f"    {inst} {filt_name}+: "
                          f"conc3 & {filt_name}+={n_both}/{n_conc3} ({overlap_pct:.0%}) "
                          f"| {filt_name}+ alone={n_g}/{len(all_days)} ({g_pct:.0%})")

        # Summary: is concordance independent?
        # Use MGC G5 as representative
        mgc_size = f"MGC_orb_size"
        if mgc_size in sess_data.columns:
            g5_all = (all_days[mgc_size] >= 5.0).sum()
            g5_conc = (conc3_days[mgc_size] >= 5.0).sum()
            overlap = g5_conc / len(conc3_days) if len(conc3_days) > 0 else 0
            g5_rate = g5_all / len(all_days) if len(all_days) > 0 else 0

            if overlap > 0.90:
                verdict = "REDUNDANT -- concordance ~= size filter"
            elif overlap > 0.75:
                verdict = "PARTIAL OVERLAP -- some independent information"
            else:
                verdict = "INDEPENDENT -- concordance adds new information"
            print(f"    >> MGC G5 overlap: {overlap:.0%} (G5 base rate: {g5_rate:.0%}) -> {verdict}")
        print()


# =========================================================================
# Analysis 3: MGC 2300 ORB size gate for MES/MNQ 0030
# =========================================================================


def analysis_3_mgc_2300_gate(mgc_context: pd.DataFrame,
                              outcomes: pd.DataFrame):
    """Test MGC 2300 ORB size as a gate for MES/MNQ 0030 strategies."""
    print("\n--- ANALYSIS 3: MGC 2300 ORB SIZE GATE FOR MES/MNQ 0030 ---")
    print("Does a large MGC 2300 ORB (relative to ATR) predict better")
    print("MES/MNQ 0030 outcomes?")
    print()

    median_ratio = mgc_context["mgc_2300_median_ratio"].iloc[0]
    print(f"  Median MGC 2300 ORB/ATR ratio: {median_ratio:.4f}")
    print()

    for follower in ["MES", "MNQ"]:
        # 0030 outcomes for this follower
        f_outcomes = outcomes[
            (outcomes["symbol"] == follower) &
            (outcomes["orb_label"] == "0030") &
            (outcomes["outcome"].notna())
        ].copy()

        if len(f_outcomes) == 0:
            print(f"  {follower} 0030: no outcomes available")
            continue

        # Test a few parameter combos
        test_combos = [
            {"entry_model": "E1", "confirm_bars": 2, "rr_target": 2.5},
            {"entry_model": "E1", "confirm_bars": 1, "rr_target": 1.5},
            {"entry_model": "E1", "confirm_bars": 3, "rr_target": 2.0},
            {"entry_model": "E3", "confirm_bars": 1, "rr_target": 1.5},
        ]

        for tc in test_combos:
            combo_mask = ((f_outcomes["entry_model"] == tc["entry_model"]) &
                          (f_outcomes["confirm_bars"] == tc["confirm_bars"]) &
                          (f_outcomes["rr_target"] == tc["rr_target"]))
            combo_out = f_outcomes[combo_mask].copy()

            if len(combo_out) == 0:
                continue

            label = f"{follower} 0030 {tc['entry_model']}/CB{tc['confirm_bars']}/RR{tc['rr_target']}"

            # Join with MGC 2300 context
            merged = combo_out.merge(
                mgc_context[["trading_day", "mgc_2300_size_bucket",
                             "MGC_2300_break_dir", "MGC_2300_size"]],
                on="trading_day", how="inner",
            )

            if len(merged) == 0:
                continue

            # Baseline
            bl = compute_metrics(merged["pnl_r"])
            print(f"  {label}:")
            print(f"    Baseline: WR={fmt_wr(bl['wr'])} ExpR={fmt_f(bl['expr'])} "
                  f"Sharpe={fmt_f(bl['sharpe'])} N={bl['n']}")

            # By MGC 2300 ORB size bucket
            for bucket in ["large", "small", "no_data"]:
                bk = merged[merged["mgc_2300_size_bucket"] == bucket]
                if len(bk) == 0:
                    continue
                m = compute_metrics(bk["pnl_r"])
                flag = sig_flag(m["n"])
                print(f"    MGC 2300 {bucket:>7}: WR={fmt_wr(m['wr'])} "
                      f"ExpR={fmt_f(m['expr'])} Sharpe={fmt_f(m['sharpe'])} "
                      f"N={m['n']:>4} {flag}")

            # By MGC 2300 break direction
            for mgc_dir in ["long", "short"]:
                dir_mask = merged["MGC_2300_break_dir"] == mgc_dir
                bk = merged[dir_mask]
                if len(bk) == 0:
                    continue
                m = compute_metrics(bk["pnl_r"])
                flag = sig_flag(m["n"])
                print(f"    MGC 2300 {mgc_dir:>7}: WR={fmt_wr(m['wr'])} "
                      f"ExpR={fmt_f(m['expr'])} Sharpe={fmt_f(m['sharpe'])} "
                      f"N={m['n']:>4} {flag}")

            print()


# =========================================================================
# Main
# =========================================================================


def main():
    print("=" * 60)
    print("CONCORDANCE STACKING & MGC 2300 GATE ANALYSIS")
    print("=" * 60)

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        features, outcomes = load_data(con)

    # Overlapping days
    day_counts = features.groupby("trading_day")["symbol"].nunique()
    overlap_days = day_counts[day_counts == 3].index
    features = features[features["trading_day"].isin(overlap_days)]
    outcomes = outcomes[outcomes["trading_day"].isin(overlap_days)]

    print(f"Instruments: {sorted(features['symbol'].unique())}")
    print(f"Overlapping days: {len(overlap_days)}")
    print(f"Period: {features['trading_day'].min()} to {features['trading_day'].max()}")
    print(f"Outcomes rows: {len(outcomes):,}")

    # Build concordance map
    conc_map = build_concordance_map(features)
    mgc_context = build_mgc_2300_context(features)

    # Print concordance tier distribution
    print()
    print("Concordance tier distribution:")
    for sess in SHARED_SESSIONS:
        sess_data = conc_map[conc_map["session"] == sess]
        vc = sess_data["concordance_tier"].value_counts()
        total = len(sess_data)
        parts = []
        for tier in ["concordant_3", "majority_2", "remaining"]:
            n = vc.get(tier, 0)
            parts.append(f"{tier}={n} ({n/total:.0%})")
        print(f"  {sess}: {', '.join(parts)}")

    # Run analyses
    analysis_1_concordance_overlay(conc_map, outcomes)
    analysis_2_overlap(conc_map)
    analysis_3_mgc_2300_gate(mgc_context, outcomes)

    # Honest summary
    print("=" * 60)
    print("HONEST SUMMARY")
    print("=" * 60)
    print()
    print("CAVEATS:")
    print(f"  1. {len(overlap_days)} overlapping days -- 2-year window only")
    print("  2. Concordance reduces trade count -- fewer trades = wider CIs")
    print("  3. If concordance just selects 'easy days' (low vol, trending),")
    print("     it may not add edge beyond what regime filters already capture")
    print("  4. MGC 2300 gate is cross-asset (different market dynamics),")
    print("     so structural argument is weaker than within-asset filters")
    print("  5. No multiple comparison correction applied")


if __name__ == "__main__":
    main()
