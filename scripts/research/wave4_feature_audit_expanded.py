#!/usr/bin/env python3
"""Wave 4 Phase A-2 — EXPANDED presession feature audit.

Audit found that the initial Phase A script missed 3 feature groups that are
populated in daily_features and verified no-lookahead:
1. orb_{session}_pre_velocity — slope of 5 closes before session (clean)
2. orb_{session}_vwap — pre-session VWAP (clean, use as distance ratio)
3. orb_{session}_compression_z/tier — rolling 20d z-score of orb_size/atr (clean)

Tests these at all 12 sessions for pre_velocity/vwap, and 3 sessions for
compression_z (only CME_REOPEN/TOKYO_OPEN/LONDON_METALS have the column).

Uses same methodology as wave4_feature_audit.py:
- T1 threshold 3pp WR spread
- Cross-instrument consistency check
- WF_START_OVERRIDE honored
- No DB writes

Read-only research. Output: CSV + stderr summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

HOLDOUT_DATE = "2026-01-01"
T1_WR_THRESHOLD_PP = 3.0
MIN_BIN_N = 20
MIN_TOTAL_N = 200

WF_START = {
    "MNQ": "2020-01-01",
    "MES": "2020-01-01",
    "MGC": "2022-06-13",
    "GC": "2015-01-01",
}

# All 12 sessions for pre_velocity and vwap
ALL_SESSIONS_12 = [
    "LONDON_METALS", "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN",
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
    "TOKYO_OPEN", "SINGAPORE_OPEN", "CME_REOPEN", "BRISBANE_1025",
]

# Only 3 sessions have compression_z populated (per build_daily_features.py COMPRESSION_SESSIONS)
COMPRESSION_SESSIONS = ["CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"]


def load_session_data(con, instrument: str, session: str) -> pd.DataFrame:
    """Load outcomes + per-session features for one (instrument, session) at E2 O5 RR1.0."""
    wf_start = WF_START.get(instrument, "2019-01-01")
    pre_vel_col = f"orb_{session}_pre_velocity"
    vwap_col = f"orb_{session}_vwap"
    comp_z_col = f"orb_{session}_compression_z"
    comp_tier_col = f"orb_{session}_compression_tier"

    # Check which columns exist for this session
    cols_exist = con.sql(f"DESCRIBE daily_features").fetchall()
    col_names = {c[0] for c in cols_exist}

    select_cols = ["o.trading_day", "o.pnl_r", "df.daily_close", "df.atr_20"]
    if pre_vel_col in col_names:
        select_cols.append(f'df."{pre_vel_col}" as pre_velocity')
    if vwap_col in col_names:
        select_cols.append(f'df."{vwap_col}" as vwap')
    if comp_z_col in col_names:
        select_cols.append(f'df."{comp_z_col}" as compression_z')
    if comp_tier_col in col_names:
        select_cols.append(f'df."{comp_tier_col}" as compression_tier')

    q = f"""
        SELECT {", ".join(select_cols)}
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

    # Derived: vwap_distance — how far daily_close is from pre-session vwap, normalized by atr
    if "vwap" in df.columns and "daily_close" in df.columns and "atr_20" in df.columns:
        df["vwap_distance_norm"] = (df["daily_close"] - df["vwap"]) / df["atr_20"]

    return df


def continuous_test(df: pd.DataFrame, col: str) -> dict | None:
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
        "n_q1": len(q1), "n_q5": len(q5),
        "wr_q1": float(wr1), "wr_q5": float(wr5),
        "expr_q1": float(e1), "expr_q5": float(e5),
        "wr_spread_pp": float((wr5 - wr1) * 100),
        "expr_spread": float(e5 - e1),
        "monotonic": bool(mono),
    }


def categorical_test(df: pd.DataFrame, col: str) -> list[dict] | None:
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
            "n_in": len(in_cat), "n_out": len(out_cat),
            "wr_in": float(wr_i), "wr_out": float(wr_o),
            "expr_in": float(e_i), "expr_out": float(e_o),
            "wr_spread_pp": float((wr_i - wr_o) * 100),
            "expr_spread": float(e_i - e_o),
        })
    return results


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=" * 100, file=sys.stderr)
    print("WAVE 4 PHASE A-2 — EXPANDED FEATURE AUDIT (pre_velocity, vwap_distance, compression)", file=sys.stderr)
    print(f"  T1 threshold: |WR spread| >= {T1_WR_THRESHOLD_PP}pp", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    results_rows = []

    # Pre-velocity test: all 12 sessions × 3 instruments
    print("\n--- pre_velocity (all 12 sessions) ---", file=sys.stderr)
    for inst in ["MNQ", "MES", "MGC"]:
        for sess in ALL_SESSIONS_12:
            df = load_session_data(con, inst, sess)
            if len(df) < MIN_TOTAL_N:
                continue
            res = continuous_test(df, "pre_velocity")
            if res is None:
                continue
            flag = "T1_PASS" if abs(res["wr_spread_pp"]) >= T1_WR_THRESHOLD_PP else ""
            if flag:
                print(f"  {inst} {sess:15s}: WR_spread={res['wr_spread_pp']:+5.1f}pp ExpR={res['expr_spread']:+.3f} mono={res['monotonic']} {flag}", file=sys.stderr)
            results_rows.append({
                "feature": "pre_velocity", "instrument": inst, "session": sess,
                "type": "continuous", "n_total": len(df),
                "wr_spread_pp": res["wr_spread_pp"],
                "expr_spread": res["expr_spread"],
                "monotonic": res["monotonic"],
                "direction": "HIGH" if res["wr_spread_pp"] > 0 else "LOW",
                "q1_expr": res["expr_q1"], "q5_expr": res["expr_q5"],
                "t1_pass": flag == "T1_PASS",
            })

    # VWAP distance test
    print("\n--- vwap_distance_norm (all 12 sessions) ---", file=sys.stderr)
    for inst in ["MNQ", "MES", "MGC"]:
        for sess in ALL_SESSIONS_12:
            df = load_session_data(con, inst, sess)
            if len(df) < MIN_TOTAL_N:
                continue
            if "vwap_distance_norm" not in df.columns:
                continue
            res = continuous_test(df, "vwap_distance_norm")
            if res is None:
                continue
            flag = "T1_PASS" if abs(res["wr_spread_pp"]) >= T1_WR_THRESHOLD_PP else ""
            if flag:
                print(f"  {inst} {sess:15s}: WR_spread={res['wr_spread_pp']:+5.1f}pp ExpR={res['expr_spread']:+.3f} mono={res['monotonic']} {flag}", file=sys.stderr)
            results_rows.append({
                "feature": "vwap_distance_norm", "instrument": inst, "session": sess,
                "type": "continuous", "n_total": len(df),
                "wr_spread_pp": res["wr_spread_pp"],
                "expr_spread": res["expr_spread"],
                "monotonic": res["monotonic"],
                "direction": "HIGH" if res["wr_spread_pp"] > 0 else "LOW",
                "q1_expr": res["expr_q1"], "q5_expr": res["expr_q5"],
                "t1_pass": flag == "T1_PASS",
            })

    # Compression_z — only 3 sessions
    print("\n--- compression_z (CME_REOPEN, TOKYO_OPEN, LONDON_METALS only) ---", file=sys.stderr)
    for inst in ["MNQ", "MES", "MGC"]:
        for sess in COMPRESSION_SESSIONS:
            df = load_session_data(con, inst, sess)
            if len(df) < MIN_TOTAL_N:
                continue
            if "compression_z" not in df.columns:
                continue
            res = continuous_test(df, "compression_z")
            if res is None:
                continue
            flag = "T1_PASS" if abs(res["wr_spread_pp"]) >= T1_WR_THRESHOLD_PP else ""
            if flag:
                print(f"  {inst} {sess:15s}: WR_spread={res['wr_spread_pp']:+5.1f}pp ExpR={res['expr_spread']:+.3f} mono={res['monotonic']} {flag}", file=sys.stderr)
            results_rows.append({
                "feature": "compression_z", "instrument": inst, "session": sess,
                "type": "continuous", "n_total": len(df),
                "wr_spread_pp": res["wr_spread_pp"],
                "expr_spread": res["expr_spread"],
                "monotonic": res["monotonic"],
                "direction": "HIGH" if res["wr_spread_pp"] > 0 else "LOW",
                "q1_expr": res["expr_q1"], "q5_expr": res["expr_q5"],
                "t1_pass": flag == "T1_PASS",
            })

    # Compression tier (categorical) — 3 sessions
    print("\n--- compression_tier (categorical, 3 sessions) ---", file=sys.stderr)
    for inst in ["MNQ", "MES", "MGC"]:
        for sess in COMPRESSION_SESSIONS:
            df = load_session_data(con, inst, sess)
            if len(df) < MIN_TOTAL_N:
                continue
            if "compression_tier" not in df.columns:
                continue
            cats = categorical_test(df, "compression_tier")
            if cats is None or len(cats) == 0:
                continue
            best = max(cats, key=lambda c: abs(c["wr_spread_pp"]))
            flag = "T1_PASS" if abs(best["wr_spread_pp"]) >= T1_WR_THRESHOLD_PP else ""
            if flag:
                print(f"  {inst} {sess:15s}: cat={best['category']} WR_spread={best['wr_spread_pp']:+5.1f}pp ExpR={best['expr_spread']:+.3f} {flag}", file=sys.stderr)
            results_rows.append({
                "feature": "compression_tier", "instrument": inst, "session": sess,
                "type": "categorical", "n_total": len(df),
                "wr_spread_pp": best["wr_spread_pp"],
                "expr_spread": best["expr_spread"],
                "monotonic": None,
                "direction": f"cat={best['category']}",
                "q1_expr": best["expr_out"], "q5_expr": best["expr_in"],
                "t1_pass": flag == "T1_PASS",
            })

    # Summary
    print("\n" + "=" * 100, file=sys.stderr)
    passers = [r for r in results_rows if r["t1_pass"]]
    print(f"TOTAL COMBOS: {len(results_rows)}", file=sys.stderr)
    print(f"T1 PASSERS: {len(passers)}", file=sys.stderr)

    # Cross-instrument consistency
    print("\n=== CROSS-INSTRUMENT CONSISTENCY (>= 2/3 same direction) ===", file=sys.stderr)
    by_feat_sess = {}
    for r in passers:
        key = (r["feature"], r["session"])
        by_feat_sess.setdefault(key, []).append((r["instrument"], r["direction"], r["wr_spread_pp"]))
    cross_count = 0
    for (feat, sess), insts in sorted(by_feat_sess.items()):
        if len(insts) >= 2:
            dirs = [d for _, d, _ in insts]
            high_count = sum(1 for d in dirs if d == "HIGH")
            low_count = sum(1 for d in dirs if d == "LOW")
            if max(high_count, low_count) >= 2:
                cross_count += 1
                print(f"  {feat} x {sess}:", file=sys.stderr)
                for inst, d, wr in insts:
                    print(f"    {inst}: {d} ({wr:+.1f}pp)", file=sys.stderr)
    print(f"\nCross-instrument consistent pairs: {cross_count}", file=sys.stderr)

    df_out = pd.DataFrame(results_rows)
    if not df_out.empty:
        df_out = df_out.sort_values("wr_spread_pp", key=abs, ascending=False)
    df_out.to_csv(sys.stdout, index=False)

    con.close()


if __name__ == "__main__":
    main()
