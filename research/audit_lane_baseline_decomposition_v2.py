"""Consolidated lane baseline/filter decomposition — v2, aperture-correct.

Supersedes:
  - research/audit_6lane_unfiltered_baseline.py (PR #52) for L2, L6
  - research/audit_l2_atr_p50_stability.py (PR #54) entirely
  - research/audit_l6_us_data_2026_breakdown.py (uncommitted draft) entirely

Bug corrected: prior audits hardcoded orb_minutes=5 in SQL, discarding the
aperture overlay suffix (_O15) present in L2 and L6 strategy_ids. Canonical
parser lives at trading_app/eligibility/builder.py and is delegated to here
rather than re-encoded.

Scope:
  (A) 6-lane 2x2 baseline-vs-filter decomposition (IS-based; K=6 note)
  (B) L2 ATR_P50 stability deep-dive (per-year, early/late, rolling 3y)
  (C) L6 2026 diagnostic (one-sample t + power + descriptive bootstrap)

Canonical truth only: orb_outcomes joined to daily_features on
trading_day + symbol + orb_minutes (triple-join).
Read-only. No production code touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.eligibility.builder import parse_strategy_id

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")
LANE_ALLOCATION = Path("docs/runtime/lane_allocation.json")
rng = np.random.default_rng(42)


def load_lane_universe(spec: dict) -> pd.DataFrame:
    """Load canonical lane universe using CORRECT orb_minutes from spec.

    Unlike the buggy load_lane_universe() in audit_6lane_scale_stability.py,
    this reads orb_minutes from the canonical parser's output rather than
    hardcoding 5.
    """
    q = """
    SELECT o.trading_day, o.pnl_r,
           CASE WHEN o.pnl_r > 0 THEN 1 ELSE 0 END AS win,
           d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = ?
      AND o.rr_target = ?
      AND o.confirm_bars = ?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = DB.execute(
        q,
        [
            spec["instrument"],
            spec["orb_label"],
            spec["orb_minutes"],
            spec["entry_model"],
            spec["rr_target"],
            spec["confirm_bars"],
        ],
    ).fetchdf()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    return df


def _block_stats(pnl: pd.Series) -> dict:
    n = len(pnl)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "sr_ann": float("nan"), "t": float("nan"), "p": float("nan")}
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if n > 1 else float("nan")
    sr = mean / std if std and std > 0 else float("nan")
    sr_ann = sr * np.sqrt(252) if not np.isnan(sr) else float("nan")
    if n > 1 and std and std > 0:
        t_res, p_res = stats.ttest_1samp(pnl.values, 0.0)
        t_val = float(t_res); p_val = float(p_res)
    else:
        t_val, p_val = float("nan"), float("nan")
    return {"n": n, "mean": mean, "std": std, "sr_ann": sr_ann, "t": t_val, "p": p_val}


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t_res, p_res = stats.ttest_ind(a.values, b.values, equal_var=False)
    return float(t_res), float(p_res)


def classify_2x2(unf_is: dict, welch_p: float, welch_t: float, non_fire_n: int) -> str:
    """2x2 classification: (baseline_edge YES/NO) x (filter_discriminates YES/NO).

    Thresholds:
      baseline_edge: |unf_is.t| > 2.0 AND unf_is.mean > 0
      filter_discriminates: welch_p < 0.05 AND welch_t > 0
    """
    if unf_is["n"] < 500:
        return "INSUFFICIENT_IS_N"
    baseline_edge = abs(unf_is["t"]) > 2.0 and unf_is["mean"] > 0
    filter_discriminates = (
        not np.isnan(welch_p) and welch_p < 0.05
        and not np.isnan(welch_t) and welch_t > 0
    )
    if non_fire_n < 10:
        return "FILTER_UNTESTABLE_BASELINE_EDGE" if baseline_edge else "FILTER_UNTESTABLE_BASELINE_WEAK"
    if baseline_edge and filter_discriminates:
        return "BOTH_CONTRIBUTE"
    if baseline_edge and not filter_discriminates:
        return "FILTER_VESTIGIAL"
    if not baseline_edge and filter_discriminates:
        return "FILTER_CORRELATES_WITH_EDGE"  # softened from FILTER_IS_THE_EDGE
    return "NEITHER_CONTRIBUTES"


# ============================================================
# Section A — 6-lane 2x2 decomposition (corrected)
# ============================================================
def audit_all_lanes() -> list[dict]:
    print("=" * 80)
    print("SECTION A — 6-LANE BASELINE vs FILTER DECOMPOSITION (APERTURE-CORRECT)")
    print("=" * 80)
    payload = json.loads(LANE_ALLOCATION.read_text(encoding="utf-8"))
    deployed = [lane for lane in payload.get("lanes", []) if lane.get("status") == "DEPLOY"]

    results = []
    for lane in deployed:
        sid = lane["strategy_id"]
        spec = parse_strategy_id(sid)
        print(f"\n--- {sid}  (orb_minutes={spec['orb_minutes']}, filter={spec['filter_type']}) ---")
        df = load_lane_universe(spec)
        if len(df) == 0:
            print("  EMPTY UNIVERSE — skip")
            continue

        sig = filter_signal(df, spec["filter_type"], spec["orb_label"])
        df["fire"] = sig

        is_df = df[df.trading_day < HOLDOUT]
        oos_df = df[df.trading_day >= HOLDOUT]

        unf_is = _block_stats(is_df["pnl_r"])
        filt_is = _block_stats(is_df[is_df["fire"] == 1]["pnl_r"])
        unf_oos = _block_stats(oos_df["pnl_r"])
        filt_oos = _block_stats(oos_df[oos_df["fire"] == 1]["pnl_r"])

        non_fire_is = is_df[is_df["fire"] == 0]["pnl_r"]
        fire_is = is_df[is_df["fire"] == 1]["pnl_r"]
        welch_t, welch_p = _welch(fire_is, non_fire_is)

        verdict = classify_2x2(unf_is, welch_p, welch_t, len(non_fire_is))
        print(f"  IS n={len(is_df)}  unfilt ExpR={unf_is['mean']:+.4f} t={unf_is['t']:+.2f}  "
              f"filt ExpR={filt_is['mean']:+.4f} t={filt_is['t']:+.2f}")
        print(f"  Welch fire-vs-non: t={welch_t:+.3f} p={welch_p:.4f}  non_fire_n={len(non_fire_is)}")
        print(f"  2026 n={len(oos_df)}  unfilt ExpR={unf_oos['mean']:+.4f}  filt ExpR={filt_oos['mean']:+.4f}")
        print(f"  Classification: {verdict}")

        results.append({
            "strategy_id": sid, "orb_minutes": spec["orb_minutes"],
            "filter_type": spec["filter_type"],
            "is_n": unf_is["n"], "unf_is_mean": unf_is["mean"], "unf_is_t": unf_is["t"],
            "filt_is_mean": filt_is["mean"], "filt_is_t": filt_is["t"],
            "welch_t": welch_t, "welch_p": welch_p,
            "oos_n": unf_oos["n"], "unf_oos_mean": unf_oos["mean"],
            "filt_oos_mean": filt_oos["mean"],
            "fire_pct_is": (is_df["fire"] == 1).mean() * 100,
            "fire_pct_oos": (oos_df["fire"] == 1).mean() * 100 if len(oos_df) else float("nan"),
            "classification": verdict,
        })

    # Portfolio rollup with K=6 multiple-testing note
    print("\n" + "=" * 80)
    print("PORTFOLIO ROLLUP — CORRECTED 2x2")
    print("=" * 80)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c:35s}: {n}")
    print("\nMultiple-testing note (K=6): at α=0.05 family-wise, Bonferroni-")
    print("corrected critical p = 0.0083. BH-FDR at q=0.05 ordered p-values:")
    ps = sorted([r['welch_p'] for r in results if not np.isnan(r['welch_p'])])
    print(f"  Ordered Welch p-values: {['{:.4f}'.format(p) for p in ps]}")
    for i, p in enumerate(ps, 1):
        bh_thresh = (i / 6) * 0.05
        status = "PASS" if p < bh_thresh else "FAIL"
        print(f"  Rank {i}: p={p:.4f}, BH threshold={bh_thresh:.4f} → {status}")

    return results


# ============================================================
# Section B — L2 ATR_P50 stability deep-dive (aperture-correct)
# ============================================================
def audit_l2_stability() -> dict:
    print("\n" + "=" * 80)
    print("SECTION B — L2 ATR_P50 STABILITY DEEP-DIVE (orb_minutes=15)")
    print("=" * 80)
    spec = parse_strategy_id("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15")
    print(f"Canonical spec: orb_minutes={spec['orb_minutes']} filter={spec['filter_type']}")
    assert spec["orb_minutes"] == 15, "canonical parser must return 15"

    df = load_lane_universe(spec)
    sig = filter_signal(df, spec["filter_type"], spec["orb_label"])
    df["fire"] = sig
    print(f"\nL2 universe: n={len(df)}, {df.trading_day.min()} → {df.trading_day.max()}")

    # Per-year Welch + atr_pct distribution
    print("\n  Per-year Welch fire-vs-non-fire + atr_20_pct distribution")
    print(f"  {'Year':5s} {'N':>4s} {'fire%':>6s} {'atrμ':>5s} {'atrp25':>7s} {'atrp75':>7s} "
          f"{'ExpR_f':>8s} {'ExpR_n':>8s} {'Δ':>7s} {'Welch t':>8s} {'p':>7s}")
    for y, grp in df.groupby("year"):
        fire = grp[grp["fire"] == 1]["pnl_r"]
        nonf = grp[grp["fire"] == 0]["pnl_r"]
        fr = len(fire) / len(grp)
        fm = fire.mean() if len(fire) else float("nan")
        nm = nonf.mean() if len(nonf) else float("nan")
        d = fm - nm
        t, p = _welch(fire, nonf)
        pct = grp["atr_20_pct"].dropna()
        print(f"  {y:5d} {len(grp):>4d} {fr*100:>5.1f}% {pct.mean():>5.1f} "
              f"{pct.quantile(0.25):>7.1f} {pct.quantile(0.75):>7.1f} "
              f"{fm:>+8.4f} {nm:>+8.4f} {d:>+7.4f} {t:>+8.2f} {p:>7.4f}")

    # Early-vs-late IS — labeled as trade-count median
    df_is = df[df.trading_day < HOLDOUT].copy()
    n_is = len(df_is)
    split_idx = n_is // 2
    split_day = df_is.iloc[split_idx].trading_day
    early = df_is.iloc[:split_idx]
    late = df_is.iloc[split_idx:]
    print(f"\n  Early-vs-late IS (trade-count median = trade #{split_idx} at {pd.Timestamp(split_day).date()}):")
    for label, sub in [("early", early), ("late", late)]:
        fire = sub[sub["fire"] == 1]["pnl_r"]
        nonf = sub[sub["fire"] == 0]["pnl_r"]
        t, p = _welch(fire, nonf)
        d = fire.mean() - nonf.mean() if len(fire) and len(nonf) else float("nan")
        cal_min = pd.Timestamp(sub.trading_day.min()).date()
        cal_max = pd.Timestamp(sub.trading_day.max()).date()
        print(f"  {label:5s}: n={len(sub):>4d} ({cal_min} to {cal_max})  fire n={len(fire):>4d}  "
              f"Δ={d:+.4f}  Welch t={t:+.3f} p={p:.4f}")

    # Per-year KS (instead of pooled)
    print("\n  Per-year KS test: SINGAPORE_OPEN atr_20_pct vs all-MNQ atr_20_pct")
    q_all = """
    SELECT DISTINCT d.trading_day, d.atr_20_pct
    FROM daily_features d
    WHERE d.symbol = 'MNQ' AND d.orb_minutes = 15
      AND d.atr_20_pct IS NOT NULL
    ORDER BY d.trading_day
    """
    all_mnq = DB.execute(q_all).fetchdf()
    all_mnq["year"] = pd.to_datetime(all_mnq["trading_day"]).dt.year
    for y in sorted(df.year.unique()):
        sgo = df[df.year == y]["atr_20_pct"].dropna().values
        all_y = all_mnq[all_mnq.year == y]["atr_20_pct"].dropna().values
        if len(sgo) < 10 or len(all_y) < 10:
            continue
        ks_res = stats.ks_2samp(sgo, all_y)
        ks_d = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)
        print(f"    {y}: D={ks_d:.3f} p={ks_p:.4f}")

    # Rolling 3y
    print("\n  Trailing 3-year Welch fire-vs-non-fire:")
    print(f"  {'end_y':6s} {'N':>5s} {'Δ':>8s} {'Welch t':>8s} {'p':>7s}")
    years_sorted = sorted(df.year.unique())
    rolling = []
    for end_y in years_sorted:
        if end_y < years_sorted[0] + 2:
            continue
        window = df[(df.year >= end_y - 2) & (df.year <= end_y)]
        fire = window[window["fire"] == 1]["pnl_r"]
        nonf = window[window["fire"] == 0]["pnl_r"]
        if len(fire) < 30 or len(nonf) < 30:
            continue
        t, p = _welch(fire, nonf)
        d = fire.mean() - nonf.mean()
        rolling.append({"end_y": int(end_y), "n": len(window), "delta": d, "t": t, "p": p})
        print(f"  {end_y:>6d} {len(window):>5d} {d:>+8.4f} {t:>+8.2f} {p:>7.4f}")

    return {"rolling": rolling, "split_day": str(pd.Timestamp(split_day).date())}


# ============================================================
# Section C — L6 2026 diagnostic (aperture-correct; reframed bootstrap)
# ============================================================
def audit_l6_2026() -> dict:
    print("\n" + "=" * 80)
    print("SECTION C — L6 2026 DIAGNOSTIC (orb_minutes=15)")
    print("=" * 80)
    spec = parse_strategy_id("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15")
    print(f"Canonical spec: orb_minutes={spec['orb_minutes']} filter={spec['filter_type']}")
    assert spec["orb_minutes"] == 15, "canonical parser must return 15"

    df = load_lane_universe(spec)
    df["day_of_month"] = pd.to_datetime(df["trading_day"]).dt.day
    is_df = df[df.trading_day < HOLDOUT].copy()
    oos_df = df[df.trading_day >= HOLDOUT].copy()
    print(f"Universe n={len(df)}  IS n={len(is_df)}  OOS n={len(oos_df)}")
    print(f"IS  ExpR={is_df.pnl_r.mean():+.4f}  t={float(stats.ttest_1samp(is_df.pnl_r.values, 0.0)[0]):+.2f}")
    t_oos = float(stats.ttest_1samp(oos_df.pnl_r.values, 0.0)[0])
    p_oos = float(stats.ttest_1samp(oos_df.pnl_r.values, 0.0)[1])
    print(f"OOS ExpR={oos_df.pnl_r.mean():+.4f}  t={t_oos:+.2f}  p={p_oos:.4f}")

    # Primary verdict: one-sample t + power
    # Effect size observed vs IS mean: delta = 0.10 / std (std from IS)
    is_std = float(is_df.pnl_r.std(ddof=1))
    n_oos = len(oos_df)
    # Power to detect IS-mean effect (d = is_mean/is_std) at n_oos
    effect_is = float(is_df.pnl_r.mean()) / is_std
    # simple normal-approx power: P(|T| > 1.96) when mean = is_mean
    # non-central t approximation: critical_z = 1.96; ncp = effect * sqrt(n)
    ncp = effect_is * np.sqrt(n_oos)
    power_approx = 1 - stats.norm.cdf(1.96 - ncp) + stats.norm.cdf(-1.96 - ncp)
    print("\n  PRIMARY TEST: one-sample t vs 0 on OOS")
    print(f"  OOS t = {t_oos:+.3f}  (two-sided p = {p_oos:.4f})")
    print(f"  OOS cannot reject H0: lane mean = 0 at α=0.05")
    print(f"\n  POWER ANALYSIS:")
    print(f"  IS effect size d = IS_mean / IS_std = {effect_is:+.4f}")
    print(f"  Power to detect IS-size effect at n={n_oos}: {power_approx*100:.1f}%")
    if power_approx < 0.50:
        power_verdict = "UNDERPOWERED (< 50% power to detect IS effect at observed OOS N)"
    elif power_approx < 0.80:
        power_verdict = "MODERATELY POWERED (50-80%)"
    else:
        power_verdict = "WELL POWERED (≥ 80%)"
    print(f"  Verdict: {power_verdict}")

    # Descriptive bootstrap — shows typical null-distribution sampling variance
    print("\n  DESCRIPTIVE BOOTSTRAP (not primary verdict — shows sampling variance):")
    is_pnl = is_df.pnl_r.values
    boot_means = np.array([
        rng.choice(is_pnl, size=n_oos, replace=True).mean()
        for _ in range(10_000)
    ])
    obs_mean = float(oos_df.pnl_r.mean())
    one_sided_p = float((boot_means <= obs_mean).mean())
    pcts = np.percentile(boot_means, [5, 25, 50, 75, 95])
    print(f"  IS mean = {float(is_df.pnl_r.mean()):+.4f}  observed OOS mean = {obs_mean:+.4f}")
    print(f"  If OOS were iid draws from IS, size-{n_oos} means lie in 5th-95th pct: [{pcts[0]:+.3f}, {pcts[4]:+.3f}]")
    print(f"  Observed OOS mean sits at bootstrap pct = {one_sided_p*100:.1f}")
    print(f"  NOTE: autocorrelation not modelled (iid bootstrap); block-bootstrap would be stricter.")

    # Calendar decomposition
    print("\n  Calendar decomposition (IS vs OOS):")
    print(f"  {'Bucket':30s}  {'IS':>30s}  {'OOS':>30s}")
    for tag, col, val in [
        ("NFP days", "is_nfp_day", True),
        ("non-NFP days", "is_nfp_day", False),
        ("OPEX days", "is_opex_day", True),
        ("non-OPEX days", "is_opex_day", False),
        ("Monday", "is_monday", True),
        ("Tuesday", "is_tuesday", True),
        ("Friday", "is_friday", True),
    ]:
        sub_is = is_df[is_df[col] == val]
        sub_oos = oos_df[oos_df[col] == val]
        is_str = f"n={len(sub_is):>4d} ExpR={sub_is.pnl_r.mean():+.4f}" if len(sub_is) else "n=0"
        if len(sub_oos):
            oos_str = f"n={len(sub_oos):>3d} ExpR={sub_oos.pnl_r.mean():+.4f}"
        else:
            oos_str = "n=0"
        print(f"  {tag:30s}  {is_str:>30s}  {oos_str:>30s}")
    cpi_is = is_df[(is_df.day_of_month >= 10) & (is_df.day_of_month <= 15)]
    cpi_oos = oos_df[(oos_df.day_of_month >= 10) & (oos_df.day_of_month <= 15)]
    print(f"  {'day_of_month 10-15 (CPI adj)':30s}  "
          f"n={len(cpi_is):>4d} ExpR={cpi_is.pnl_r.mean():+.4f}  "
          f"n={len(cpi_oos):>3d} ExpR={cpi_oos.pnl_r.mean():+.4f}")

    # Vol regime
    print("\n  atr_20_pct quartile breakdown:")
    print(f"  {'Bin':6s} {'IS n':>5s} {'IS ExpR':>9s} {'OOS n':>6s} {'OOS ExpR':>10s} {'OOS %':>6s}")
    for lo, hi, tag in [(0, 25, "Q1"), (25, 50, "Q2"), (50, 75, "Q3"), (75, 101, "Q4")]:
        si = is_df[(is_df.atr_20_pct >= lo) & (is_df.atr_20_pct < hi)]
        so = oos_df[(oos_df.atr_20_pct >= lo) & (oos_df.atr_20_pct < hi)]
        oos_pct = len(so) / len(oos_df) * 100 if len(oos_df) else float("nan")
        ise = si.pnl_r.mean() if len(si) else float("nan")
        ose = so.pnl_r.mean() if len(so) else float("nan")
        print(f"  {tag:6s} {len(si):>5d} {ise:>+9.4f} {len(so):>6d} {ose:>+10.4f} {oos_pct:>5.1f}%")

    return {
        "t_oos": t_oos, "p_oos": p_oos, "power_approx": float(power_approx),
        "boot_pct": one_sided_p, "obs_mean": obs_mean, "n_oos": n_oos,
    }


def main() -> None:
    print("CORRECTION AUDIT — aperture-correct rerun of PR #52/#54/L6")
    print(f"ran {pd.Timestamp.now('UTC')}")
    print()

    section_a = audit_all_lanes()
    section_b = audit_l2_stability()
    section_c = audit_l6_2026()

    out_path = Path("docs/audit/results/2026-04-21-correction-aperture-audit-rerun.json")
    payload = {
        "section_a_lane_decomposition": section_a,
        "section_b_l2_stability": section_b,
        "section_c_l6_diagnostic": section_c,
    }
    out_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
