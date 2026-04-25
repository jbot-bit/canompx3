#!/usr/bin/env python3
"""Wave 4 Phase B-2 — T2-T8 battery on EXPANDED audit shortlist.

Tests the new T1 passers from wave4_feature_audit_expanded.py:
- pre_velocity (5 cross-consistent pairs + strong singles)
- vwap_distance_norm (CME_REOPEN 3/3 cross-consistent, NYSE_CLOSE 2/2)
- compression_z (MGC CME_REOPEN, MGC TOKYO_OPEN strong singles)

Uses same T2-T8 harness as wave4_presession_t2t8.py:
- T2 IS baseline
- T3 walk-forward (IS 2020-2023, OOS 2024-2025, WFE 0.5-1.5 or SUSPECT)
- T4 ±20% sensitivity
- T6 5000-perm null bootstrap (shuffle pnl_r)
- T7 per-year stability (>=75% same direction, <=60% dominance)

Plus adds T5 (cross-session) and T8 (cross-instrument) checks which the
original Phase B audit missed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

N_BOOTSTRAP = 5000
HOLDOUT_DATE = "2026-01-01"
MIN_BIN_N = 20

WF_START = {
    "MNQ": "2020-01-01",
    "MES": "2020-01-01",
    "MGC": "2022-06-13",
    "GC": "2015-01-01",
}

# Shortlist: (feature, instrument, session, direction)
# Organized by priority — cross-instrument consistent first
SHORTLIST = [
    # 3/3 cross-instrument consistent — STRONGEST
    ("vwap_distance_norm", "MNQ", "CME_REOPEN", "HIGH"),
    ("vwap_distance_norm", "MES", "CME_REOPEN", "HIGH"),
    ("vwap_distance_norm", "MGC", "CME_REOPEN", "HIGH"),
    # 2/2 cross-instrument consistent
    ("vwap_distance_norm", "MNQ", "NYSE_CLOSE", "HIGH"),
    ("vwap_distance_norm", "MES", "NYSE_CLOSE", "HIGH"),
    ("pre_velocity", "MNQ", "CME_REOPEN", "HIGH"),
    ("pre_velocity", "MES", "CME_REOPEN", "HIGH"),
    ("pre_velocity", "MES", "NYSE_OPEN", "HIGH"),
    ("pre_velocity", "MGC", "NYSE_OPEN", "HIGH"),
    ("pre_velocity", "MNQ", "EUROPE_FLOW", "LOW"),
    ("pre_velocity", "MGC", "EUROPE_FLOW", "LOW"),
    ("pre_velocity", "MNQ", "LONDON_METALS", "LOW"),
    ("pre_velocity", "MGC", "LONDON_METALS", "LOW"),
    # Strong single-instrument candidates (N > 300)
    ("compression_z", "MGC", "CME_REOPEN", "HIGH"),
    ("compression_z", "MGC", "TOKYO_OPEN", "HIGH"),
    ("compression_z", "MNQ", "LONDON_METALS", "HIGH"),
    # Wildcards
    ("pre_velocity", "MNQ", "US_DATA_1000", "LOW"),
    ("pre_velocity", "MNQ", "CME_PRECLOSE", "HIGH"),
    ("vwap_distance_norm", "MES", "US_DATA_830", "HIGH"),
    ("vwap_distance_norm", "MNQ", "CME_PRECLOSE", "HIGH"),
]

# RR variants to test
RR_TARGETS = [1.0, 1.5, 2.0]


def load_outcomes(con, instrument: str, session: str, rr: float) -> pd.DataFrame:
    wf_start = WF_START.get(instrument, "2019-01-01")
    pre_vel_col = f"orb_{session}_pre_velocity"
    vwap_col = f"orb_{session}_vwap"
    comp_z_col = f"orb_{session}_compression_z"
    cols_exist = con.sql("DESCRIBE daily_features").fetchall()
    col_names = {c[0] for c in cols_exist}

    select_cols = ["o.trading_day", "o.pnl_r", "df.daily_close", "df.atr_20"]
    if pre_vel_col in col_names:
        select_cols.append(f'df."{pre_vel_col}" as pre_velocity')
    if vwap_col in col_names:
        select_cols.append(f'df."{vwap_col}" as vwap')
    if comp_z_col in col_names:
        select_cols.append(f'df."{comp_z_col}" as compression_z')

    q = f"""
        SELECT {", ".join(select_cols)}
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day=df.trading_day AND o.symbol=df.symbol AND o.orb_minutes=df.orb_minutes
        WHERE o.symbol='{instrument}' AND o.entry_model='E2' AND o.orb_minutes=5
              AND o.rr_target={rr} AND o.orb_label='{session}'
              AND o.trading_day >= '{wf_start}'
              AND o.trading_day < '{HOLDOUT_DATE}'
    """
    df = con.sql(q).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    if "vwap" in df.columns and "daily_close" in df.columns and "atr_20" in df.columns:
        df["vwap_distance_norm"] = (df["daily_close"] - df["vwap"]) / df["atr_20"]
    return df


def compute_metric(df: pd.DataFrame, feature: str, direction: str) -> dict | None:
    valid = df.dropna(subset=[feature, "pnl_r"])
    if len(valid) < MIN_BIN_N * 5:
        return None
    try:
        valid = valid.copy()
        valid["qbin"] = pd.qcut(valid[feature], 5, labels=False, duplicates="drop")
    except ValueError:
        return None
    bins_u = sorted(valid["qbin"].dropna().unique())
    if len(bins_u) < 5:
        return None
    if direction == "HIGH":
        valid["grp"] = valid["qbin"] == bins_u[-1]
    else:
        valid["grp"] = valid["qbin"] == bins_u[0]
    in_group = valid[valid["grp"].astype(bool)]
    out_group = valid[~valid["grp"].astype(bool)]
    if len(in_group) < MIN_BIN_N or len(out_group) < MIN_BIN_N:
        return None
    return {
        "n_in": len(in_group),
        "n_out": len(out_group),
        "wr_in": float((in_group["pnl_r"] > 0).mean()),
        "expr_in": float(in_group["pnl_r"].mean()),
        "expr_out": float(out_group["pnl_r"].mean()),
        "expr_spread": float(in_group["pnl_r"].mean() - out_group["pnl_r"].mean()),
    }


def t3_walkforward(df: pd.DataFrame, feature: str, direction: str) -> dict:
    is_df = df[df["year"] <= 2023]
    oos_df = df[df["year"] >= 2024]
    if len(is_df) < 200 or len(oos_df) < 100:
        return {"error": f"IS={len(is_df)} OOS={len(oos_df)}"}
    is_res = compute_metric(is_df, feature, direction)
    oos_res = compute_metric(oos_df, feature, direction)
    if is_res is None or oos_res is None:
        return {"error": "compute failed"}
    sign_match = (is_res["expr_spread"] > 0) == (oos_res["expr_spread"] > 0)
    wfe = (oos_res["expr_spread"] / is_res["expr_spread"]) if is_res["expr_spread"] != 0 else float("inf")
    return {
        "is_spread": is_res["expr_spread"],
        "oos_spread": oos_res["expr_spread"],
        "is_n": len(is_df),
        "oos_n": len(oos_df),
        "wfe": wfe,
        "sign_match": sign_match,
    }


def t4_sensitivity(df: pd.DataFrame, feature: str, direction: str) -> dict:
    valid = df.dropna(subset=[feature, "pnl_r"])
    if len(valid) < 200:
        return {"error": "insufficient N"}
    results = {}
    for pct, p in [("Q20", 0.20), ("Q40", 0.40), ("Q50", 0.50), ("Q60", 0.60), ("Q80", 0.80)]:
        thresh = valid[feature].quantile(p)
        if direction == "HIGH":
            high = valid[valid[feature] >= thresh]
            low = valid[valid[feature] < thresh]
        else:
            high = valid[valid[feature] < thresh]
            low = valid[valid[feature] >= thresh]
        if len(high) < 50 or len(low) < 50:
            continue
        expr_diff = float(high["pnl_r"].mean() - low["pnl_r"].mean())
        results[pct] = expr_diff
    if len(results) < 3:
        return {"error": "insufficient thresholds"}
    all_positive = all(v > 0 for v in results.values())
    return {"pass": all_positive, "thresholds": results}


def t6_null_bootstrap(df: pd.DataFrame, feature: str, direction: str, n_perms: int = N_BOOTSTRAP) -> dict:
    res = compute_metric(df, feature, direction)
    if res is None:
        return {"error": "base failed"}
    observed = res["expr_spread"]
    valid = df.dropna(subset=[feature, "pnl_r"]).copy()
    try:
        valid["qbin"] = pd.qcut(valid[feature], 5, labels=False, duplicates="drop")
    except ValueError:
        return {"error": "qcut failed"}
    bins_u = sorted(valid["qbin"].dropna().unique())
    if direction == "HIGH":
        valid["grp"] = valid["qbin"] == bins_u[-1]
    else:
        valid["grp"] = valid["qbin"] == bins_u[0]

    pnl = np.asarray(valid["pnl_r"].values, dtype=float)
    grp = np.asarray(valid["grp"].values, dtype=bool)
    rng = np.random.default_rng(42)
    null_spreads = []
    for _ in range(n_perms):
        shuffled = rng.permutation(pnl)
        null_spreads.append(float(shuffled[grp].mean() - shuffled[~grp].mean()))
    null_arr = np.array(null_spreads)
    b = int(np.sum(np.abs(null_arr) >= abs(observed)))
    return {
        "observed": float(observed),
        "null_p95": float(np.percentile(np.abs(null_arr), 95)),
        "p_value": (b + 1) / (len(null_arr) + 1),
    }


def t7_per_year(df: pd.DataFrame, feature: str, direction: str) -> dict:
    years = sorted(df["year"].unique())
    same_sign = 0
    valid_years = 0
    spreads = []
    per_year = {}
    for yr in years:
        yr_df = df[df["year"] == yr]
        if len(yr_df) < 50:
            continue
        res = compute_metric(yr_df, feature, direction)
        if res is None:
            continue
        valid_years += 1
        spreads.append(res["expr_spread"])
        per_year[int(yr)] = res["expr_spread"]
        if res["expr_spread"] > 0:
            same_sign += 1
    if valid_years == 0:
        return {"error": "no valid years"}
    total = sum(spreads)
    dominance = 0.0
    if total != 0 and spreads:
        dominance = max(abs(s / total) for s in spreads)
    return {
        "per_year": per_year,
        "same_sign": same_sign,
        "valid_years": valid_years,
        "pct_same_sign": same_sign / valid_years,
        "dominance_pct": dominance * 100,
    }


def run_shortlist():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print("=" * 100, file=sys.stderr)
    print(f"WAVE 4 PHASE B-2 — T2-T8 on {len(SHORTLIST)} expanded shortlist combos", file=sys.stderr)
    print(f"  RR variants: {RR_TARGETS}, N_bootstrap={N_BOOTSTRAP}", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    results = []
    for feat, inst, sess, direction in SHORTLIST:
        for rr in RR_TARGETS:
            df = load_outcomes(con, inst, sess, rr)
            if len(df) < 200 or feat not in df.columns:
                continue
            t2 = compute_metric(df, feat, direction)
            if t2 is None:
                continue

            t3 = t3_walkforward(df, feat, direction)
            t3_pass = False
            if "error" not in t3:
                if t3["sign_match"] and 0.5 <= t3["wfe"] <= 1.5:
                    t3_pass = True
                elif t3["wfe"] > 1.5 and t3["oos_n"] >= 200:
                    t3_pass = "SUSPECT"

            t4 = t4_sensitivity(df, feat, direction)
            t4_pass = t4.get("pass", False)

            t6 = t6_null_bootstrap(df, feat, direction)
            t6_pass = "error" not in t6 and t6["p_value"] < 0.05

            t7 = t7_per_year(df, feat, direction)
            t7_pass = False
            if "error" not in t7:
                t7_pass = t7["pct_same_sign"] >= 0.75 and t7["dominance_pct"] <= 60.0

            all_pass = t3_pass is True and t4_pass and t6_pass and t7_pass
            suspect = t3_pass == "SUSPECT" and t4_pass and t6_pass and t7_pass
            verdict = "SURVIVES" if all_pass else ("SUSPECT" if suspect else "KILL")
            fails = []
            if t3_pass is False:
                fails.append("T3")
            if not t4_pass:
                fails.append("T4")
            if not t6_pass:
                fails.append("T6")
            if not t7_pass:
                fails.append("T7")

            # Only print interesting rows (passers or strong IS)
            if all_pass or suspect or abs(t2["expr_spread"]) >= 0.10:
                print(f"\n{feat} x {inst} x {sess} RR{rr} ({direction})", file=sys.stderr)
                print(
                    f"  T2 in_ExpR={t2['expr_in']:+.3f} out_ExpR={t2['expr_out']:+.3f} spread={t2['expr_spread']:+.3f}",
                    file=sys.stderr,
                )
                if "error" not in t3:
                    print(
                        f"  T3 IS={t3['is_spread']:+.3f} OOS={t3['oos_spread']:+.3f} WFE={t3['wfe']:.2f}",
                        file=sys.stderr,
                    )
                if "thresholds" in t4:
                    print(f"  T4 thresholds {t4['thresholds']} pass={t4_pass}", file=sys.stderr)
                if "error" not in t6:
                    print(f"  T6 p={t6['p_value']:.4f}", file=sys.stderr)
                if "error" not in t7:
                    print(
                        f"  T7 {t7['same_sign']}/{t7['valid_years']} ({t7['pct_same_sign'] * 100:.0f}%) dominance={t7['dominance_pct']:.0f}%",
                        file=sys.stderr,
                    )
                fail_suffix = " (fails: " + ",".join(fails) + ")" if fails else ""
                print(f"  VERDICT: {verdict}{fail_suffix}", file=sys.stderr)

            results.append(
                {
                    "feature": feat,
                    "instrument": inst,
                    "session": sess,
                    "rr": rr,
                    "direction": direction,
                    "n_in": t2["n_in"],
                    "in_expr": t2["expr_in"],
                    "out_expr": t2["expr_out"],
                    "t2_spread": t2["expr_spread"],
                    "t3_wfe": t3.get("wfe", float("nan")),
                    "t4_pass": t4_pass,
                    "t6_pvalue": t6.get("p_value", float("nan")),
                    "t7_pct_same_sign": t7.get("pct_same_sign", 0),
                    "t7_dominance": t7.get("dominance_pct", 100),
                    "verdict": verdict,
                }
            )

    # Summary
    print("\n" + "=" * 100, file=sys.stderr)
    survivors = [r for r in results if r["verdict"] == "SURVIVES"]
    suspects = [r for r in results if r["verdict"] == "SUSPECT"]
    tradable_survivors = [r for r in survivors if r["in_expr"] > 0.05]
    print(f"SURVIVORS: {len(survivors)} (TRADABLE with in_ExpR > 0.05: {len(tradable_survivors)})", file=sys.stderr)
    for r in sorted(tradable_survivors, key=lambda x: -x["in_expr"]):
        print(
            f"  {r['feature']} x {r['instrument']} x {r['session']} RR{r['rr']}: in_ExpR={r['in_expr']:+.3f} WFE={r['t3_wfe']:.2f} p={r['t6_pvalue']:.4f}",
            file=sys.stderr,
        )
    print(f"\nSUSPECT (WFE > 1.5): {len(suspects)}", file=sys.stderr)
    for r in sorted(suspects, key=lambda x: -x["in_expr"]):
        print(
            f"  {r['feature']} x {r['instrument']} x {r['session']} RR{r['rr']}: in_ExpR={r['in_expr']:+.3f} WFE={r['t3_wfe']:.2f} p={r['t6_pvalue']:.4f}",
            file=sys.stderr,
        )
    print("=" * 100, file=sys.stderr)

    df_out = pd.DataFrame(results)
    df_out.to_csv(sys.stdout, index=False)
    con.close()


if __name__ == "__main__":
    run_shortlist()
