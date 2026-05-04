"""OVNRNG router — rolling CV + binning-variable ablation.

PR #62 single-fold walk-forward showed ΔSR_ann=+0.87. This runs 4
rolling annual folds and compares the router's signal across 3
binning variables (ovn/atr, atr_20_pct, garch_forecast_vol_pct) to
establish:

1. Fold-to-fold stability of router SR advantage
2. Stability of the train-derived best-session-per-bin map
3. Whether ovn/atr is special vs other vol-regime features

Theory grounding:
- Chan 2008 Ch 7 (regime switching — canonical extract at
  docs/institutional/literature/chan_2008_ch7_regime_switching.md)
- Chordia et al 2018 (factor-segmented testing, t ≥ 3.79)
- Carver 2015 Ch 10-11 (forecast combination — bin-conditional
  switching is a discrete case)

Canonical: orb_outcomes and daily_features (triple-joined).
Read-only. No production code touched.  2026 OOS sacred and untouched.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")

SESSIONS = [
    "LONDON_METALS", "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN",
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
]

# Rolling folds: 3-year train → 1-year test
FOLDS = [
    {"train": [2019, 2020, 2021], "test": 2022},
    {"train": [2020, 2021, 2022], "test": 2023},
    {"train": [2021, 2022, 2023], "test": 2024},
    {"train": [2022, 2023, 2024], "test": 2025},
]


def load_all() -> pd.DataFrame:
    q = """
    SELECT o.trading_day, o.orb_label, o.pnl_r,
           d.overnight_range, d.atr_20, d.atr_20_pct,
           d.garch_forecast_vol_pct
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label IN ('LONDON_METALS','EUROPE_FLOW','US_DATA_830',
                          'NYSE_OPEN','US_DATA_1000','COMEX_SETTLE',
                          'CME_PRECLOSE','NYSE_CLOSE')
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.rr_target = 1.5
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < ?
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
    ORDER BY o.trading_day, o.orb_label
    """
    df = DB.execute(q, [str(HOLDOUT.date())]).fetchdf()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    df["ovn_atr"] = df["overnight_range"] / df["atr_20"]
    return df


def _sr(pnl: pd.Series) -> tuple[float, float]:
    n = len(pnl)
    if n < 2:
        return float("nan"), float("nan")
    m = float(pnl.mean())
    s = float(pnl.std(ddof=1))
    sr = m / s if s > 0 else float("nan")
    return m, float(sr * np.sqrt(252)) if not np.isnan(sr) else float("nan")


def bin_by(df: pd.DataFrame, col: str, bounds: np.ndarray) -> pd.Series:
    """Assign Q1..Q5 based on pre-computed quantile bounds (train-only)."""
    def _assign(x: float) -> str:
        if pd.isna(x): return "NA"
        if x <= bounds[0]: return "Q1"
        if x <= bounds[1]: return "Q2"
        if x <= bounds[2]: return "Q3"
        if x <= bounds[3]: return "Q4"
        return "Q5"
    return df[col].apply(_assign)


def run_fold(df: pd.DataFrame, binning_col: str, fold: dict) -> dict:
    train = df[df["year"].isin(fold["train"])].copy()
    test = df[df["year"] == fold["test"]].copy()
    if len(train) < 100 or len(test) < 30:
        return {"skipped": "insufficient_n", "train_n": len(train), "test_n": len(test)}

    # Skip rows where binning_col is NULL (e.g., garch where unpopulated)
    train_valid = train.dropna(subset=[binning_col])
    test_valid = test.dropna(subset=[binning_col])
    if len(train_valid) < 100 or len(test_valid) < 30:
        return {"skipped": "insufficient_valid_n",
                "train_n": len(train_valid), "test_n": len(test_valid)}

    # Train-only daily quantile bounds (one row per trading_day — collapse)
    train_daily = train_valid.groupby("trading_day").agg(
        feat=(binning_col, "first")).reset_index()
    bounds = train_daily["feat"].quantile([0.2, 0.4, 0.6, 0.8]).values

    train_valid = train_valid.copy()
    test_valid = test_valid.copy()
    train_valid["bin"] = bin_by(train_valid, binning_col, bounds)
    test_valid["bin"] = bin_by(test_valid, binning_col, bounds)

    # Train map: best session per bin
    tr_pivot = train_valid.pivot_table(index="orb_label", columns="bin",
                                        values="pnl_r", aggfunc="mean", observed=True)
    # Only keep Q-bins (drop NA)
    tr_pivot = tr_pivot[[c for c in ["Q1","Q2","Q3","Q4","Q5"] if c in tr_pivot.columns]]
    best_per_bin = {b: tr_pivot[b].idxmax() for b in tr_pivot.columns}
    tr_bin_agnostic = train_valid.groupby("orb_label", observed=True)["pnl_r"].mean().sort_values(ascending=False)
    top1_agnostic = tr_bin_agnostic.index[0]

    # Apply to test
    router_mask = pd.Series(False, index=test_valid.index)
    for b, s in best_per_bin.items():
        router_mask |= (test_valid["bin"] == b) & (test_valid["orb_label"] == s)
    control_mask = test_valid["orb_label"] == top1_agnostic

    router_pnl = test_valid.loc[router_mask, "pnl_r"]
    control_pnl = test_valid.loc[control_mask, "pnl_r"]
    uniform_pnl = test_valid["pnl_r"]

    r_m, r_sr = _sr(router_pnl)
    c_m, c_sr = _sr(control_pnl)
    u_m, u_sr = _sr(uniform_pnl)

    return {
        "fold": f"train {fold['train'][0]}-{fold['train'][-1]} test {fold['test']}",
        "best_per_bin": best_per_bin,
        "top1_agnostic": top1_agnostic,
        "router_n": len(router_pnl), "router_expr": r_m, "router_sr_ann": r_sr,
        "control_n": len(control_pnl), "control_expr": c_m, "control_sr_ann": c_sr,
        "uniform_n": len(uniform_pnl), "uniform_expr": u_m, "uniform_sr_ann": u_sr,
        "delta_sr": r_sr - c_sr,
    }


def main() -> None:
    print("=" * 80)
    print("OVNRNG ROUTER — ROLLING CV + BINNING-VARIABLE ABLATION")
    print(f"ran {pd.Timestamp.now('UTC')}")
    print("MNQ E2 RR=1.5 CB=1 orb_minutes=5, IS only (pre-2026-01-01)")
    print("2026 OOS SACRED — NOT TOUCHED")
    print("=" * 80)

    df = load_all()
    print(f"\nUniverse n={len(df)} across {df['trading_day'].nunique()} trading days")

    # Run for each binning variable
    binnings = [
        ("ovn_atr", "overnight_range / atr_20 (PR #62 variable)"),
        ("atr_20_pct", "atr_20_pct (MNQ-wide 252d rolling percentile)"),
        ("garch_forecast_vol_pct", "garch_forecast_vol_pct (where populated)"),
    ]

    for col, label in binnings:
        print("\n" + "=" * 80)
        print(f"BINNING VARIABLE: {col}")
        print(f"  {label}")
        print("=" * 80)

        fold_results = []
        for f in FOLDS:
            r = run_fold(df, col, f)
            if "skipped" in r:
                print(f"\n  Fold train {f['train'][0]}-{f['train'][-1]} test {f['test']}: SKIPPED ({r['skipped']})")
                continue
            print(f"\n  Fold: {r['fold']}")
            bpb_str = ", ".join(f"{b}:{s}" for b, s in r['best_per_bin'].items())
            print(f"    Train best-per-bin: {bpb_str}")
            print(f"    Train bin-agnostic top-1: {r['top1_agnostic']}")
            print(f"    TEST ROUTER:   n={r['router_n']:>4d}  ExpR={r['router_expr']:+.4f}  SR_ann={r['router_sr_ann']:+.3f}")
            print(f"    TEST CONTROL:  n={r['control_n']:>4d}  ExpR={r['control_expr']:+.4f}  SR_ann={r['control_sr_ann']:+.3f}")
            print(f"    TEST UNIFORM:  n={r['uniform_n']:>4d}  ExpR={r['uniform_expr']:+.4f}  SR_ann={r['uniform_sr_ann']:+.3f}")
            print(f"    Δ (router − control) SR_ann = {r['delta_sr']:+.3f}")
            fold_results.append(r)

        # Fold summary
        if fold_results:
            print(f"\n  ROLLING-CV SUMMARY ({col}):")
            deltas = [r["delta_sr"] for r in fold_results if not np.isnan(r["delta_sr"])]
            wins = sum(1 for d in deltas if d > 0)
            print(f"    Folds: {len(fold_results)}  Router beat Control: {wins}")
            if deltas:
                print(f"    ΔSR_ann per-fold: " + " ".join(f"{d:+.2f}" for d in deltas))
                print(f"    ΔSR_ann mean: {np.mean(deltas):+.3f}, median: {np.median(deltas):+.3f}")

            # Map stability (bin → set of sessions chosen across folds)
            print(f"\n  MAP STABILITY ({col}) — session chosen per bin per fold:")
            all_bins = sorted({b for r in fold_results for b in r["best_per_bin"]})
            for b in all_bins:
                choices = [r["best_per_bin"].get(b, "—") for r in fold_results]
                unique = len(set(choices) - {"—"})
                print(f"    {b}: {choices}  ({unique} unique)")

    # ================================================================
    # Comparison across binning variables
    # ================================================================
    print("\n" + "=" * 80)
    print("ABLATION — IS OVN/ATR SPECIAL?")
    print("=" * 80)
    print("  Per-fold ΔSR_ann (router − control) by binning variable:\n")
    print(f"  {'Fold':>12s}  {'ovn_atr':>10s}  {'atr_20_pct':>12s}  {'garch_pct':>11s}")
    summary = {col: [] for col, _ in binnings}
    for f_idx, f in enumerate(FOLDS):
        row = f"  test {f['test']}"
        values = []
        for col, _ in binnings:
            r = run_fold(df, col, f)
            if "skipped" in r:
                values.append("   —")
                summary[col].append(float("nan"))
            else:
                values.append(f"{r['delta_sr']:+.3f}")
                summary[col].append(r["delta_sr"])
        print(f"  {row:>12s}  " + "  ".join(f"{v:>10s}" for v in values))

    print()
    for col, arr in summary.items():
        arr_clean = [v for v in arr if not np.isnan(v)]
        wins = sum(1 for v in arr_clean if v > 0)
        if arr_clean:
            print(f"  {col:25s}: mean ΔSR={np.mean(arr_clean):+.3f}, "
                  f"wins {wins}/{len(arr_clean)}")

    # ================================================================
    # Final verdict
    # ================================================================
    print("\n" + "=" * 80)
    print("FINAL VERDICT (ovn_atr, primary variable)")
    print("=" * 80)
    ovn_arr = [v for v in summary["ovn_atr"] if not np.isnan(v)]
    wins = sum(1 for v in ovn_arr if v > 0)
    mean_d = np.mean(ovn_arr) if ovn_arr else float("nan")
    if wins >= 3 and mean_d >= 0.30:
        v = "ROUTER_DEPLOY_READY"
    elif wins >= 2 and mean_d >= 0.10:
        v = "ROUTER_MARGINAL"
    elif wins <= 1:
        v = "ROUTER_BRITTLE"
    else:
        v = "ROUTER_AMBIGUOUS"
    print(f"  wins {wins}/{len(ovn_arr)}  mean ΔSR {mean_d:+.3f}")
    print(f"  VERDICT: {v}")


if __name__ == "__main__":
    main()
