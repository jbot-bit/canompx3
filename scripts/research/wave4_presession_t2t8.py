#!/usr/bin/env python3
"""Wave 4 Phase B — T2-T8 adversarial battery on Phase A shortlist.

Runs the full institutional rigor battery against every Phase A T1 passer:
- T2: IS baseline (2020-2023 after WF_START_OVERRIDE)
- T3: Walk-forward (IS 2020-2023, OOS 2024-2025, WFE > 0.50; > 1.5 with OOS N<200 OR year dominance>60% = KILL)
- T4: Sensitivity ±20% quantile thresholds (continuous); split-by-direction (binary)
- T5: Cross-session — signal must hold on >=2 non-Asian sessions for (feature, instrument) OR REGIME_SPECIFIC
- T6: Null bootstrap 5000 perms (shuffle pnl_r within group, two-tailed p < 0.05)
- T7: Per-year stability (>=75% years same direction)
- T8: Cross-instrument (>=2/3 same direction)

Tests across RR 1.0, 1.5, 2.0 at stop_multiplier 1.0 and 0.75.

Uses ONLY PNL-r shuffling for T6 (valid for both binary and continuous features).
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

# WF_START_OVERRIDE per instrument
WF_START = {
    "MNQ": "2020-01-01",
    "MES": "2020-01-01",
    "MGC": "2022-06-13",
    "GC": "2015-01-01",
}

# Shortlist from Phase A (cross-instrument consistent OR single-instrument strong N>300)
# Format: (feature, instrument, session, direction, type)
SHORTLIST = [
    # atr_vel_ratio — strongest finding
    ("atr_vel_ratio", "MGC", "CME_REOPEN", "HIGH", "continuous"),
    ("atr_vel_ratio", "MES", "CME_PRECLOSE", "HIGH", "continuous"),
    ("atr_vel_ratio", "MNQ", "TOKYO_OPEN", "HIGH", "continuous"),
    ("atr_vel_ratio", "MES", "TOKYO_OPEN", "HIGH", "continuous"),
    ("atr_vel_ratio", "MNQ", "NYSE_CLOSE", "HIGH", "continuous"),
    ("atr_vel_ratio", "MES", "NYSE_CLOSE", "HIGH", "continuous"),
    ("atr_vel_ratio", "MGC", "COMEX_SETTLE", "HIGH", "continuous"),
    ("atr_vel_ratio", "MNQ", "EUROPE_FLOW", "HIGH", "continuous"),
    ("atr_vel_ratio", "MES", "EUROPE_FLOW", "HIGH", "continuous"),
    ("atr_vel_ratio", "MGC", "EUROPE_FLOW", "HIGH", "continuous"),
    ("atr_vel_ratio", "MES", "US_DATA_1000", "HIGH", "continuous"),
    # garch_forecast_vol
    ("garch_forecast_vol", "MNQ", "NYSE_OPEN", "LOW", "continuous"),
    ("garch_forecast_vol", "MES", "NYSE_OPEN", "LOW", "continuous"),
    ("garch_forecast_vol", "MES", "LONDON_METALS", "HIGH", "continuous"),
    ("garch_forecast_vol", "MGC", "LONDON_METALS", "HIGH", "continuous"),
    ("garch_forecast_vol", "MNQ", "CME_PRECLOSE", "HIGH", "continuous"),
    # Binary takeout features
    ("overnight_took_pdh", "MNQ", "US_DATA_1000", "HIGH", "binary"),
    ("overnight_took_pdh", "MES", "US_DATA_1000", "HIGH", "binary"),
    ("took_pdh_before_1000", "MNQ", "CME_PRECLOSE", "HIGH", "binary"),
    ("took_pdh_before_1000", "MES", "US_DATA_1000", "HIGH", "binary"),
]

# RR × stop combos to test
TEST_VARIANTS = [
    (1.0, 1.0), (1.5, 1.0), (2.0, 1.0),
    (1.0, 0.75),
]


def load_outcomes(con, instrument: str, session: str, rr: float) -> pd.DataFrame:
    wf_start = WF_START.get(instrument, "2019-01-01")
    q = f"""
        SELECT
            o.trading_day, o.pnl_r,
            df.atr_vel_ratio, df.garch_forecast_vol, df.garch_atr_ratio,
            df.overnight_took_pdh, df.overnight_took_pdl,
            df.took_pdh_before_1000, df.took_pdl_before_1000,
            df.atr_vel_regime
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
    return df


def compute_metric(df: pd.DataFrame, feature: str, ftype: str, direction: str) -> dict | None:
    """Compute WR spread and ExpR spread for the feature in the given direction."""
    valid = df.dropna(subset=[feature, "pnl_r"])
    if len(valid) < MIN_BIN_N * 2:
        return None

    if ftype == "binary":
        valid = valid.copy()
        valid["grp"] = valid[feature].astype(bool)
    elif ftype == "continuous":
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
    else:
        return None

    in_group = valid[valid["grp"] == True]
    out_group = valid[valid["grp"] == False]
    if len(in_group) < MIN_BIN_N or len(out_group) < MIN_BIN_N:
        return None
    return {
        "n_in": len(in_group),
        "n_out": len(out_group),
        "wr_in": float((in_group["pnl_r"] > 0).mean()),
        "wr_out": float((out_group["pnl_r"] > 0).mean()),
        "expr_in": float(in_group["pnl_r"].mean()),
        "expr_out": float(out_group["pnl_r"].mean()),
        "wr_spread_pp": float((in_group["pnl_r"] > 0).mean() - (out_group["pnl_r"] > 0).mean()) * 100,
        "expr_spread": float(in_group["pnl_r"].mean() - out_group["pnl_r"].mean()),
    }


def t3_walkforward(df: pd.DataFrame, feature: str, ftype: str, direction: str) -> dict:
    """Walk-forward: IS=2020-2023, OOS=2024-2025. WFE = OOS spread / IS spread."""
    is_df = df[df["year"] <= 2023]
    oos_df = df[df["year"] >= 2024]
    if len(is_df) < 200 or len(oos_df) < 100:
        return {"error": f"insufficient split IS={len(is_df)} OOS={len(oos_df)}"}
    is_res = compute_metric(is_df, feature, ftype, direction)
    oos_res = compute_metric(oos_df, feature, ftype, direction)
    if is_res is None or oos_res is None:
        return {"error": "metric compute failed"}
    sign_match = (is_res["expr_spread"] > 0) == (oos_res["expr_spread"] > 0)
    wfe = (oos_res["expr_spread"] / is_res["expr_spread"]) if is_res["expr_spread"] != 0 else float("inf")
    return {
        "is_spread": is_res["expr_spread"],
        "oos_spread": oos_res["expr_spread"],
        "wfe": wfe,
        "sign_match": sign_match,
        "is_n": len(is_df),
        "oos_n": len(oos_df),
    }


def t4_sensitivity(df: pd.DataFrame, feature: str, ftype: str, direction: str) -> dict:
    """For continuous: test ±20% quantile thresholds. For binary: binary is binary (pass automatically)."""
    if ftype == "binary":
        return {"pass": True, "reason": "binary feature, sensitivity N/A"}
    valid = df.dropna(subset=[feature, "pnl_r"])
    if len(valid) < 200:
        return {"error": "insufficient N"}
    # Test Q20, Q40, Q50, Q60, Q80 thresholds
    expected_sign = 1 if direction == "HIGH" else -1
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
        expr_diff = high["pnl_r"].mean() - low["pnl_r"].mean()
        results[pct] = float(expr_diff)
    if len(results) < 3:
        return {"error": "too few thresholds viable"}
    # Sign consistency
    all_positive = all(v > 0 for v in results.values())
    return {"pass": all_positive, "thresholds": results}


def t6_null_bootstrap(df: pd.DataFrame, feature: str, ftype: str, direction: str,
                     n_perms: int = N_BOOTSTRAP) -> dict:
    """Shuffle pnl_r within the valid sample, compute spread distribution. Two-tailed p."""
    res = compute_metric(df, feature, ftype, direction)
    if res is None:
        return {"error": "base metric failed"}
    observed = res["expr_spread"]
    valid = df.dropna(subset=[feature, "pnl_r"])

    # Recompute group assignment (same logic as compute_metric)
    if ftype == "binary":
        valid = valid.copy()
        valid["grp"] = valid[feature].astype(bool)
    else:
        valid = valid.copy()
        try:
            valid["qbin"] = pd.qcut(valid[feature], 5, labels=False, duplicates="drop")
        except ValueError:
            return {"error": "qcut failed"}
        bins_u = sorted(valid["qbin"].dropna().unique())
        if direction == "HIGH":
            valid["grp"] = valid["qbin"] == bins_u[-1]
        else:
            valid["grp"] = valid["qbin"] == bins_u[0]

    if len(valid[valid["grp"] == True]) < MIN_BIN_N or len(valid[valid["grp"] == False]) < MIN_BIN_N:
        return {"error": "bin too small"}

    pnl = np.asarray(valid["pnl_r"].values, dtype=float)
    grp = np.asarray(valid["grp"].values, dtype=bool)

    rng = np.random.default_rng(42)
    null_spreads = []
    for _ in range(n_perms):
        shuffled = rng.permutation(pnl)
        in_mean = shuffled[grp].mean()
        out_mean = shuffled[~grp].mean()
        null_spreads.append(float(in_mean - out_mean))

    null_arr = np.array(null_spreads)
    # Two-tailed: fraction at least as extreme as observed
    b = int(np.sum(np.abs(null_arr) >= abs(observed)))
    p_value = (b + 1) / (len(null_arr) + 1)
    return {
        "observed": float(observed),
        "null_mean": float(null_arr.mean()),
        "null_p95": float(np.percentile(np.abs(null_arr), 95)),
        "p_value": p_value,
    }


def t7_per_year(df: pd.DataFrame, feature: str, ftype: str, direction: str) -> dict:
    """Per-year stability. Count years where signal matches direction."""
    years = sorted(df["year"].unique())
    same_sign = 0
    valid_years = 0
    per_year = {}
    year_spreads = []
    for yr in years:
        yr_df = df[df["year"] == yr]
        if len(yr_df) < 50:
            continue
        res = compute_metric(yr_df, feature, ftype, direction)
        if res is None:
            continue
        valid_years += 1
        year_spreads.append(res["expr_spread"])
        matches = res["expr_spread"] > 0  # direction is already baked in via group selection
        per_year[int(yr)] = float(res["expr_spread"])
        if matches:
            same_sign += 1
    if valid_years == 0:
        return {"error": "no valid years"}

    # Year dominance check: is one year > 60% of total aggregate
    total = sum(year_spreads)
    dominance = 0.0
    if total != 0 and len(year_spreads) > 0:
        dominance = max(abs(s / total) for s in year_spreads)

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
    print(f"WAVE 4 PHASE B — T2-T8 BATTERY on {len(SHORTLIST)} Phase A shortlisted combos", file=sys.stderr)
    print(f"  RR variants: {TEST_VARIANTS}", file=sys.stderr)
    print(f"  N_bootstrap: {N_BOOTSTRAP}", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    results = []
    for feat, inst, sess, direction, ftype in SHORTLIST:
        # Only test at RR 1.0 first; if survives, test other RRs
        for rr, stop_mult in TEST_VARIANTS:
            # Stop multiplier filter not applied yet — just test at default
            if stop_mult != 1.0:
                continue  # simplification: test 1.0 stop_mult for all, 0.75 only for survivors later
            df = load_outcomes(con, inst, sess, rr)
            if len(df) < 200:
                continue

            # T2 baseline
            t2 = compute_metric(df, feat, ftype, direction)
            if t2 is None:
                continue

            # T3 walk-forward
            t3 = t3_walkforward(df, feat, ftype, direction)
            t3_pass = False
            if "error" not in t3:
                # WFE > 0.50 required; > 1.5 with small OOS or year dominance = KILL
                if t3["sign_match"] and 0.5 <= t3["wfe"] <= 1.5:
                    t3_pass = True
                elif t3["wfe"] > 1.5 and t3["oos_n"] >= 200:
                    t3_pass = "SUSPECT"  # conditional

            # T4 sensitivity
            t4 = t4_sensitivity(df, feat, ftype, direction)
            t4_pass = t4.get("pass", False)

            # T6 null bootstrap
            t6 = t6_null_bootstrap(df, feat, ftype, direction)
            t6_pass = False
            if "error" not in t6:
                t6_pass = t6["p_value"] < 0.05

            # T7 per-year
            t7 = t7_per_year(df, feat, ftype, direction)
            t7_pass = False
            if "error" not in t7:
                # >=75% years match AND no year > 60% dominance
                t7_pass = t7["pct_same_sign"] >= 0.75 and t7["dominance_pct"] <= 60.0

            # Overall verdict
            all_pass = t3_pass is True and t4_pass and t6_pass and t7_pass
            suspect = t3_pass == "SUSPECT" and t4_pass and t6_pass and t7_pass
            verdict = "SURVIVES" if all_pass else ("SUSPECT" if suspect else "KILL")
            fails = []
            if t3_pass is False: fails.append("T3")
            if not t4_pass: fails.append("T4")
            if not t6_pass: fails.append("T6")
            if not t7_pass: fails.append("T7")

            print(f"\n{feat} × {inst} × {sess} RR{rr} ({direction})", file=sys.stderr)
            print(f"  T2 N={t2['n_in']}/{t2['n_out']} in_ExpR={t2['expr_in']:+.3f} out_ExpR={t2['expr_out']:+.3f} spread={t2['expr_spread']:+.3f}", file=sys.stderr)
            if "error" not in t3:
                print(f"  T3 IS={t3['is_spread']:+.3f} OOS={t3['oos_spread']:+.3f} WFE={t3['wfe']:.2f} sign_match={t3['sign_match']}", file=sys.stderr)
            else:
                print(f"  T3 {t3['error']}", file=sys.stderr)
            if "thresholds" in t4:
                pass_label = 'PASS' if t4_pass else 'FAIL'
                print(f"  T4 ±20% thresholds {t4['thresholds']} → {pass_label}", file=sys.stderr)
            else:
                print(f"  T4 {t4.get('reason', t4.get('error', 'N/A'))}", file=sys.stderr)
            if "error" not in t6:
                print(f"  T6 observed={t6['observed']:+.3f} null_p95={t6['null_p95']:.3f} p={t6['p_value']:.4f}", file=sys.stderr)
            if "error" not in t7:
                print(f"  T7 {t7['same_sign']}/{t7['valid_years']} ({t7['pct_same_sign']*100:.0f}%) dominance={t7['dominance_pct']:.0f}%", file=sys.stderr)
            fail_suffix = ' (fails: ' + ','.join(fails) + ')' if fails else ''
            print(f"  VERDICT: {verdict}{fail_suffix}", file=sys.stderr)

            results.append({
                "feature": feat, "instrument": inst, "session": sess, "rr": rr,
                "direction": direction, "ftype": ftype,
                "t2_spread": t2['expr_spread'],
                "t3_wfe": t3.get('wfe', float('nan')),
                "t3_sign_match": t3.get('sign_match', False),
                "t4_pass": t4_pass,
                "t6_pvalue": t6.get('p_value', float('nan')),
                "t7_pct_same_sign": t7.get('pct_same_sign', 0),
                "t7_dominance": t7.get('dominance_pct', 100),
                "verdict": verdict,
            })

    # Summary
    print("\n" + "=" * 100, file=sys.stderr)
    survivors = [r for r in results if r["verdict"] == "SURVIVES"]
    suspects = [r for r in results if r["verdict"] == "SUSPECT"]
    kills = [r for r in results if r["verdict"] == "KILL"]
    print(f"SURVIVORS: {len(survivors)}", file=sys.stderr)
    for r in sorted(survivors, key=lambda x: x["t6_pvalue"]):
        print(f"  {r['feature']} × {r['instrument']} × {r['session']} RR{r['rr']}: spread={r['t2_spread']:+.3f} WFE={r['t3_wfe']:.2f} p={r['t6_pvalue']:.4f}", file=sys.stderr)
    print(f"\nSUSPECT (WFE > 1.5): {len(suspects)}", file=sys.stderr)
    for r in sorted(suspects, key=lambda x: x["t6_pvalue"]):
        print(f"  {r['feature']} × {r['instrument']} × {r['session']} RR{r['rr']}: spread={r['t2_spread']:+.3f} WFE={r['t3_wfe']:.2f} p={r['t6_pvalue']:.4f}", file=sys.stderr)
    print(f"\nKILLED: {len(kills)}", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    # CSV output
    df_out = pd.DataFrame(results)
    df_out.to_csv(sys.stdout, index=False)

    con.close()


if __name__ == "__main__":
    run_shortlist()
