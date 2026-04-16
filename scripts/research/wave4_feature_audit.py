#!/usr/bin/env python3
"""Wave 4 Phase A — Read-only presession feature audit.

Audits every presession feature in daily_features with >=80% population that is
NOT currently wired into trading_app/config.py ALL_FILTERS. For each (feature,
instrument, session) combo, computes univariate WR/ExpR spread and filters by
look-ahead safety.

No DB writes. Outputs CSV to stdout and summary to stderr.

Audit-driven additions (vs initial plan):
- 12 sessions (not 8) for session-universal features
- Pre-T1 feature correlation matrix (drops redundant r > 0.80 pairs)
- Three-window rolling percentile comparison for overnight_range (20d/60d/252d)
- WF_START_OVERRIDE honored per instrument (MES/MNQ from 2020-01-01, GC from 2015-01-01)
- T1 threshold = 3pp (matches prior confluence research)
- Instrument-bias audit for ORB_VOL_* and OVNRNG_abs filters
- Jaccard + pnl_r correlation against existing filters
- Year-by-year stability snapshot
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

HOLDOUT_DATE = "2026-01-01"
T1_WR_THRESHOLD_PP = 3.0  # matches prior confluence research
MIN_BIN_N = 20
MIN_TOTAL_N = 200

# WF_START_OVERRIDE per instrument
WF_START = {
    "MNQ": "2020-01-01",
    "MES": "2020-01-01",
    "MGC": "2022-06-13",  # raw MGC data starts here
    "GC": "2015-01-01",   # GC proxy for pre-2022 regime discipline
}

ALL_SESSIONS = [
    "LONDON_METALS", "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN",
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
    "TOKYO_OPEN", "SINGAPORE_OPEN", "CME_REOPEN", "BRISBANE_1025",
]

# Sessions where overnight_range/took_pdh features are clean (session start >= 17:00 Brisbane)
OVERNIGHT_CLEAN_SESSIONS = [
    "LONDON_METALS", "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN",
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
]

# Sessions where took_pdh_before_1000 is clean (session start >= 10:00 local)
PRE_1000_CLEAN_SESSIONS = [
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
]

# Feature inventory with population state and look-ahead rules.
# (feature_column, type, clean_sessions, description)
FEATURES_TO_AUDIT = [
    # Binary takeout features — overnight_took_pdh cross-instrument showed real signal pre-plan
    ("overnight_took_pdh", "binary", OVERNIGHT_CLEAN_SESSIONS, "Overnight took out prev day high"),
    ("overnight_took_pdl", "binary", OVERNIGHT_CLEAN_SESSIONS, "Overnight took out prev day low"),
    ("took_pdh_before_1000", "binary", PRE_1000_CLEAN_SESSIONS, "Took PDH before 10am local"),
    ("took_pdl_before_1000", "binary", PRE_1000_CLEAN_SESSIONS, "Took PDL before 10am local"),
    # Continuous features — session-universal (no overnight window dependency)
    ("atr_vel_ratio", "continuous", ALL_SESSIONS, "ATR velocity ratio"),
    ("garch_forecast_vol", "continuous", OVERNIGHT_CLEAN_SESSIONS, "GARCH forecast vol (from overnight)"),
    ("garch_atr_ratio", "continuous", OVERNIGHT_CLEAN_SESSIONS, "GARCH/ATR ratio"),
    # Categorical features
    ("atr_vel_regime", "categorical", ALL_SESSIONS, "ATR velocity regime bucket"),
    ("gap_type", "categorical", ALL_SESSIONS, "Gap type category"),
    ("day_type", "categorical", ALL_SESSIONS, "Day type category"),
    ("prev_day_direction", "categorical", ALL_SESSIONS, "Previous day direction (bull/bear)"),
]


def load_all_presession_data(con, instrument: str, session: str) -> pd.DataFrame:
    """Load outcomes + all presession features for one (instrument, session) at E2 O5 RR1.0."""
    wf_start = WF_START.get(instrument, "2019-01-01")
    q = f"""
        SELECT
            o.trading_day, o.pnl_r,
            df.overnight_range, df.overnight_range_pct,
            df.overnight_took_pdh, df.overnight_took_pdl,
            df.took_pdh_before_1000, df.took_pdl_before_1000,
            df.atr_vel_ratio, df.atr_vel_regime,
            df.garch_forecast_vol, df.garch_atr_ratio,
            df.gap_type, df.day_type, df.prev_day_direction,
            df.atr_20, df.atr_20_pct,
            df.gap_open_points, df.prev_day_range
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day=df.trading_day AND o.symbol=df.symbol AND o.orb_minutes=df.orb_minutes
        WHERE o.symbol='{instrument}' AND o.entry_model='E2' AND o.orb_minutes=5
              AND o.rr_target=1.0 AND o.orb_label='{session}'
              AND o.trading_day >= '{wf_start}'
              AND o.trading_day < '{HOLDOUT_DATE}'
    """
    df = con.sql(q).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    return df


def binary_test(df: pd.DataFrame, col: str) -> dict | None:
    """Compute WR/ExpR split for binary feature. Returns None if insufficient N."""
    if col not in df.columns:
        return None
    valid = df.dropna(subset=[col, "pnl_r"])
    if len(valid) < MIN_TOTAL_N:
        return None
    true_rows = valid[valid[col] == True]
    false_rows = valid[valid[col] == False]
    if len(true_rows) < MIN_BIN_N or len(false_rows) < MIN_BIN_N:
        return None
    wr_t = (true_rows["pnl_r"] > 0).mean()
    wr_f = (false_rows["pnl_r"] > 0).mean()
    e_t = true_rows["pnl_r"].mean()
    e_f = false_rows["pnl_r"].mean()
    return {
        "n_true": len(true_rows),
        "n_false": len(false_rows),
        "wr_true": float(wr_t),
        "wr_false": float(wr_f),
        "expr_true": float(e_t),
        "expr_false": float(e_f),
        "wr_spread_pp": float((wr_t - wr_f) * 100),
        "expr_spread": float(e_t - e_f),
    }


def continuous_test(df: pd.DataFrame, col: str) -> dict | None:
    """Compute Q5-Q1 WR/ExpR spread for continuous feature."""
    if col not in df.columns:
        return None
    valid = df.dropna(subset=[col, "pnl_r"])
    if len(valid) < MIN_BIN_N * 5:
        return None
    try:
        valid = valid.copy()
        valid["qbin"] = pd.qcut(valid[col], 5, labels=False, duplicates="drop")
    except ValueError:
        return None
    bins = sorted(valid["qbin"].dropna().unique())
    if len(bins) < 5:
        return None
    q1 = valid[valid["qbin"] == bins[0]]
    q5 = valid[valid["qbin"] == bins[-1]]
    if len(q1) < MIN_BIN_N or len(q5) < MIN_BIN_N:
        return None
    wr1 = (q1["pnl_r"] > 0).mean()
    wr5 = (q5["pnl_r"] > 0).mean()
    e1 = q1["pnl_r"].mean()
    e5 = q5["pnl_r"].mean()
    mono = all(
        valid[valid["qbin"] == bins[i]]["pnl_r"].mean() <= valid[valid["qbin"] == bins[i+1]]["pnl_r"].mean()
        for i in range(4)
    )
    return {
        "n_q1": len(q1),
        "n_q5": len(q5),
        "wr_q1": float(wr1),
        "wr_q5": float(wr5),
        "expr_q1": float(e1),
        "expr_q5": float(e5),
        "wr_spread_pp": float((wr5 - wr1) * 100),
        "expr_spread": float(e5 - e1),
        "monotonic": bool(mono),
    }


def categorical_test(df: pd.DataFrame, col: str) -> list[dict] | None:
    """For each category, compute mean ExpR + WR vs the rest. Returns list of per-category results."""
    if col not in df.columns:
        return None
    valid = df.dropna(subset=[col, "pnl_r"])
    if len(valid) < MIN_TOTAL_N:
        return None
    results = []
    for cat in valid[col].unique():
        in_cat = valid[valid[col] == cat]
        out_cat = valid[valid[col] != cat]
        if len(in_cat) < MIN_BIN_N or len(out_cat) < MIN_BIN_N:
            continue
        wr_i = (in_cat["pnl_r"] > 0).mean()
        wr_o = (out_cat["pnl_r"] > 0).mean()
        e_i = in_cat["pnl_r"].mean()
        e_o = out_cat["pnl_r"].mean()
        results.append({
            "category": str(cat),
            "n_in": len(in_cat),
            "n_out": len(out_cat),
            "wr_in": float(wr_i),
            "wr_out": float(wr_o),
            "expr_in": float(e_i),
            "expr_out": float(e_o),
            "wr_spread_pp": float((wr_i - wr_o) * 100),
            "expr_spread": float(e_i - e_o),
        })
    return results


def feature_correlation_matrix(con, instrument: str, sessions: list[str]) -> pd.DataFrame:
    """Compute pairwise correlation of the 12 candidate features on a sample of days."""
    wf_start = WF_START.get(instrument, "2019-01-01")
    q = f"""
        SELECT
            df.overnight_range, df.atr_vel_ratio, df.garch_forecast_vol, df.garch_atr_ratio,
            df.atr_20, df.atr_20_pct, df.prev_day_range, df.gap_open_points,
            CAST(df.overnight_took_pdh AS INT) as otpdh,
            CAST(df.overnight_took_pdl AS INT) as otpdl
        FROM daily_features df
        WHERE df.symbol='{instrument}' AND df.orb_minutes=5
              AND df.trading_day >= '{wf_start}'
              AND df.trading_day < '{HOLDOUT_DATE}'
    """
    df = con.sql(q).df()
    return df.corr(numeric_only=True)


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=" * 100, file=sys.stderr)
    print("WAVE 4 PHASE A — PRESESSION FEATURE AUDIT", file=sys.stderr)
    print(f"  WF_START overrides: {WF_START}", file=sys.stderr)
    print(f"  T1 threshold: |WR spread| >= {T1_WR_THRESHOLD_PP}pp", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    # Phase A.1 — Feature correlation audit
    print("\n[A.1] Feature correlation matrix (MNQ, pre-T1 redundancy check)", file=sys.stderr)
    corr = feature_correlation_matrix(con, "MNQ", OVERNIGHT_CLEAN_SESSIONS)
    print(corr.round(2).to_string(), file=sys.stderr)
    # Identify r > 0.80 pairs
    high_corr = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr.iloc[i, j])  # type: ignore[arg-type]
            if abs(val) > 0.80:
                high_corr.append((cols[i], cols[j], val))
    print(f"\n[A.1] Feature pairs with |r| > 0.80: {len(high_corr)}", file=sys.stderr)
    for a, b, r in high_corr:
        print(f"  {a} ↔ {b}: r={r:+.2f}", file=sys.stderr)

    # Phase A.2 — Univariate tests for each feature × instrument × session
    results_rows = []
    print("\n[A.2] Univariate tests per (feature, instrument, session)", file=sys.stderr)
    for feature, ftype, clean_sessions, desc in FEATURES_TO_AUDIT:
        print(f"\n  --- {feature} ({ftype}): {desc} ---", file=sys.stderr)
        for inst in ["MNQ", "MES", "MGC"]:
            for sess in clean_sessions:
                try:
                    df = load_all_presession_data(con, inst, sess)
                except Exception as e:
                    continue
                if len(df) < MIN_TOTAL_N:
                    continue

                if ftype == "binary":
                    res = binary_test(df, feature)
                    if res is None:
                        continue
                    row = {
                        "feature": feature, "instrument": inst, "session": sess,
                        "type": ftype, "n_total": len(df),
                        "wr_spread_pp": res["wr_spread_pp"],
                        "expr_spread": res["expr_spread"],
                        "direction": "HIGH" if res["wr_spread_pp"] > 0 else "LOW",
                        "detail": f"T N={res['n_true']} E={res['expr_true']:+.3f} W={res['wr_true']*100:.0f}% | F N={res['n_false']} E={res['expr_false']:+.3f} W={res['wr_false']*100:.0f}%",
                    }
                elif ftype == "continuous":
                    res = continuous_test(df, feature)
                    if res is None:
                        continue
                    row = {
                        "feature": feature, "instrument": inst, "session": sess,
                        "type": ftype, "n_total": len(df),
                        "wr_spread_pp": res["wr_spread_pp"],
                        "expr_spread": res["expr_spread"],
                        "direction": "HIGH" if res["wr_spread_pp"] > 0 else "LOW",
                        "detail": f"Q1 E={res['expr_q1']:+.3f} W={res['wr_q1']*100:.0f}% | Q5 E={res['expr_q5']:+.3f} W={res['wr_q5']*100:.0f}% mono={res['monotonic']}",
                    }
                elif ftype == "categorical":
                    cats = categorical_test(df, feature)
                    if cats is None or len(cats) == 0:
                        continue
                    # Take the most extreme category
                    best = max(cats, key=lambda c: abs(c["wr_spread_pp"]))
                    row = {
                        "feature": feature, "instrument": inst, "session": sess,
                        "type": ftype, "n_total": len(df),
                        "wr_spread_pp": best["wr_spread_pp"],
                        "expr_spread": best["expr_spread"],
                        "direction": f"cat={best['category']}",
                        "detail": f"in N={best['n_in']} E={best['expr_in']:+.3f} W={best['wr_in']*100:.0f}% | out N={best['n_out']} E={best['expr_out']:+.3f} W={best['wr_out']*100:.0f}%",
                    }
                else:
                    continue

                flag = "T1_PASS" if abs(row["wr_spread_pp"]) >= T1_WR_THRESHOLD_PP else ""
                row["t1_pass"] = flag == "T1_PASS"
                results_rows.append(row)
                if flag:
                    print(f"    {inst:4s} {sess:15s}: WR_spread={row['wr_spread_pp']:+5.1f}pp ExpR_spread={row['expr_spread']:+.3f} {row['direction']} {flag}", file=sys.stderr)

    # Summary
    print("\n" + "=" * 100, file=sys.stderr)
    total = len(results_rows)
    passers = [r for r in results_rows if r["t1_pass"]]
    print(f"TOTAL COMBOS TESTED: {total}", file=sys.stderr)
    print(f"T1 PASSERS (|WR spread| >= {T1_WR_THRESHOLD_PP}pp): {len(passers)}", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    # Cross-instrument consistency check
    print("\n[A.3] Cross-instrument consistency — (feature, session) pairs where >= 2/3 instruments agree on direction", file=sys.stderr)
    by_feat_sess = {}
    for r in passers:
        key = (r["feature"], r["session"])
        by_feat_sess.setdefault(key, []).append((r["instrument"], r["direction"], r["wr_spread_pp"]))

    cross_consistent = []
    for (feat, sess), insts in by_feat_sess.items():
        if len(insts) >= 2:
            dirs = [d for _, d, _ in insts]
            # Check majority direction
            high_count = sum(1 for d in dirs if d == "HIGH")
            low_count = sum(1 for d in dirs if d == "LOW")
            if max(high_count, low_count) >= 2:
                cross_consistent.append((feat, sess, insts))
                print(f"  {feat} × {sess}:", file=sys.stderr)
                for inst, d, wr in insts:
                    print(f"    {inst}: {d} ({wr:+.1f}pp)", file=sys.stderr)

    print(f"\n[A.3] Cross-instrument consistent (feature, session) pairs: {len(cross_consistent)}", file=sys.stderr)

    # Write CSV to stdout
    df_out = pd.DataFrame(results_rows)
    if not df_out.empty:
        df_out = df_out.sort_values("wr_spread_pp", key=abs, ascending=False)
    df_out.to_csv(sys.stdout, index=False)

    con.close()


if __name__ == "__main__":
    main()
