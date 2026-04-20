"""6-lane unfiltered baseline stress-test.

For each of the 6 currently-DEPLOYED lanes (per `docs/runtime/lane_allocation.json`),
decompose the lane's edge into UNFILTERED-baseline contribution and FILTER contribution.

Motivation: PR #47 6-lane audit found that 5 of 6 deployed filters fire ≥75%
in 2026 — largely vestigial. This script answers the follow-up:
  - Does the unfiltered baseline (session + RR + direction + entry + CB geometry
    alone) already carry the edge?
  - Or is the filter — even at high fire rate — still adding meaningful lift?

Reuses parse_strategy_id() and load_lane_universe() from the scale-stability
audit so the two audits are directly comparable (same orb_minutes=5, same
aperture-overlay handling).

Canonical truth only: orb_outcomes JOIN daily_features + lane_allocation.json.
No production code touched. Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.audit_6lane_scale_stability import (
    load_lane_universe,
    parse_strategy_id,
)
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LANE_ALLOCATION = Path("docs/runtime/lane_allocation.json")
HOLDOUT = pd.Timestamp("2026-01-01")


def _block_stats(pnl: pd.Series) -> dict:
    """Return N, mean, std, sharpe (non-ann), sharpe_ann, t-stat vs 0."""
    n = len(pnl)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "sr": float("nan"), "sr_ann": float("nan"),
                "t": float("nan"), "p": float("nan"), "wr": float("nan")}
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if n > 1 else float("nan")
    sr = mean / std if std and std > 0 else float("nan")
    sr_ann = sr * np.sqrt(252) if not np.isnan(sr) else float("nan")
    # one-sample t-test vs 0
    if n > 1 and std and std > 0:
        t, p = stats.ttest_1samp(pnl.values, 0.0)
    else:
        t, p = float("nan"), float("nan")
    wr = float((pnl > 0).mean())
    return {"n": n, "mean": mean, "std": std, "sr": float(sr) if not np.isnan(sr) else float("nan"),
            "sr_ann": float(sr_ann) if not np.isnan(sr_ann) else float("nan"),
            "t": float(t) if not np.isnan(t) else float("nan"),
            "p": float(p) if not np.isnan(p) else float("nan"),
            "wr": wr}


def audit_lane(lane: dict) -> dict:
    sid = lane["strategy_id"]
    spec = parse_strategy_id(sid)
    print("\n" + "=" * 80)
    print(f"LANE: {sid}")
    print(f"  session={spec['session']} entry={spec['entry']} rr={spec['rr']} "
          f"cb={spec['cb']} filter={spec['filter']} aperture={spec['aperture_overlay']}")
    print("=" * 80)

    df = load_lane_universe(spec["instrument"], spec["session"], spec["entry"],
                            spec["rr"], spec["cb"])
    if len(df) == 0:
        print("  EMPTY UNIVERSE — skipping")
        return {"strategy_id": sid, "spec": spec, "classification": "ERROR_EMPTY_UNIVERSE"}

    print(f"Universe n={len(df)}  {df.trading_day.min()} → {df.trading_day.max()}")

    sig = filter_signal(df, spec["filter"], spec["session"])
    df = df.copy()
    df["fire"] = sig

    # Per-year table
    print(f"\n  {'Year':6s} {'N_unf':>5s} {'ExpR_unf':>9s} {'SR_unf':>7s} "
          f"{'N_filt':>6s} {'ExpR_filt':>10s} {'SR_filt':>8s} {'ΔExpR':>7s} {'fire%':>6s}")
    year_rows = []
    for y, grp in df.groupby("year"):
        unf = _block_stats(grp["pnl_r"])
        filt = _block_stats(grp[grp["fire"] == 1]["pnl_r"])
        delta = filt["mean"] - unf["mean"] if not np.isnan(filt["mean"]) else float("nan")
        fire_pct = (grp["fire"] == 1).mean() * 100
        year_rows.append({
            "year": int(y), "n_unf": unf["n"], "expr_unf": unf["mean"], "sr_unf": unf["sr"],
            "n_filt": filt["n"], "expr_filt": filt["mean"], "sr_filt": filt["sr"],
            "delta": delta, "fire_pct": fire_pct,
        })
        print(f"  {y:6d} {unf['n']:>5d} {unf['mean']:>+9.4f} {unf['sr']:>+7.3f} "
              f"{filt['n']:>6d} {filt['mean']:>+10.4f} {filt['sr']:>+8.3f} "
              f"{delta:>+7.4f} {fire_pct:>5.1f}%")

    # Full sample + 2026 OOS
    is_df = df[df.trading_day < HOLDOUT]
    oos_df = df[df.trading_day >= HOLDOUT]

    unf_full = _block_stats(df["pnl_r"])
    filt_full = _block_stats(df[df["fire"] == 1]["pnl_r"])
    unf_is = _block_stats(is_df["pnl_r"])
    filt_is = _block_stats(is_df[is_df["fire"] == 1]["pnl_r"])
    unf_oos = _block_stats(oos_df["pnl_r"])
    filt_oos = _block_stats(oos_df[oos_df["fire"] == 1]["pnl_r"])

    print(f"\n  Block totals:")
    print(f"    FULL  unf n={unf_full['n']:4d} ExpR={unf_full['mean']:+.4f} SR_ann={unf_full['sr_ann']:+.3f} t={unf_full['t']:+.2f} p={unf_full['p']:.4f}")
    print(f"    FULL  flt n={filt_full['n']:4d} ExpR={filt_full['mean']:+.4f} SR_ann={filt_full['sr_ann']:+.3f} t={filt_full['t']:+.2f} p={filt_full['p']:.4f}")
    print(f"    IS    unf n={unf_is['n']:4d} ExpR={unf_is['mean']:+.4f} SR_ann={unf_is['sr_ann']:+.3f} t={unf_is['t']:+.2f} p={unf_is['p']:.4f}")
    print(f"    IS    flt n={filt_is['n']:4d} ExpR={filt_is['mean']:+.4f} SR_ann={filt_is['sr_ann']:+.3f} t={filt_is['t']:+.2f} p={filt_is['p']:.4f}")
    print(f"    2026  unf n={unf_oos['n']:4d} ExpR={unf_oos['mean']:+.4f} SR_ann={unf_oos['sr_ann']:+.3f} t={unf_oos['t']:+.2f} p={unf_oos['p']:.4f}")
    print(f"    2026  flt n={filt_oos['n']:4d} ExpR={filt_oos['mean']:+.4f} SR_ann={filt_oos['sr_ann']:+.3f} t={filt_oos['t']:+.2f} p={filt_oos['p']:.4f}")

    # Classification operates on IS, not 2026 OOS. 2026 is ~3.5 months,
    # n=54-72 per lane — cannot distinguish +0.0R from +0.15R filters at
    # 95% CI ±0.22R per-trade std. IS has n=900-1800, real power.
    filt_is_series = is_df[is_df["fire"] == 1]["pnl_r"]
    non_fire_is = is_df[is_df["fire"] == 0]["pnl_r"]
    welch_t, welch_p = _welch_diff(filt_is_series, non_fire_is) if len(non_fire_is) >= 10 else (float("nan"), float("nan"))
    classification = _classify_is(unf_is, welch_p, welch_t, len(non_fire_is))

    # Also report descriptive 2026 label (not used for classification)
    oos_n = min(unf_oos["n"], filt_oos["n"])
    oos_delta = filt_oos["mean"] - unf_oos["mean"] if not np.isnan(filt_oos["mean"]) else float("nan")
    is_delta = filt_is["mean"] - unf_is["mean"]
    print(f"\n  Classification (IS-based): {classification}  (IS n_unf={unf_is['n']})")
    print(f"  IS fire-vs-non-fire Welch: t={welch_t:+.3f} p={welch_p:.4f}  "
          f"ΔExpR(filt-unf)={is_delta:+.4f}  "
          f"unfilt t={unf_is['t']:+.2f}  filt t={filt_is['t']:+.2f}")
    print(f"  2026 descriptive only: min-N={oos_n} unfilt={unf_oos['mean']:+.4f} "
          f"filt={filt_oos['mean']:+.4f} Δ={oos_delta:+.4f} "
          f"(CI ±{1.96 * 0.9 / np.sqrt(oos_n):+.3f}R per-trade std-based)")

    return {
        "strategy_id": sid,
        "spec": spec,
        "year_rows": year_rows,
        "unf_full": unf_full, "filt_full": filt_full,
        "unf_is": unf_is, "filt_is": filt_is,
        "unf_oos": unf_oos, "filt_oos": filt_oos,
        "classification": classification,
        "is_welch_fire_vs_nonfire_t": welch_t,
        "is_welch_fire_vs_nonfire_p": welch_p,
        "is_delta_filt_minus_unf": is_delta,
        "fire_pct_2026": (oos_df["fire"] == 1).mean() * 100 if len(oos_df) else float("nan"),
    }


def _welch_diff(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Welch two-sample t on mean(a) - mean(b). Returns (t, p-two-sided)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a.values, b.values, equal_var=False)
    return float(t), float(p)


def _classify_is(unf_is: dict, welch_p: float, welch_t: float,
                 non_fire_n: int) -> str:
    """Classify lane by IS 2×2 decomposition:
      axis A — does the unfiltered baseline carry edge alone?  (|unf_is t| > 2.0)
      axis B — does the filter discriminate fire-vs-non-fire?  (Welch p < 0.05 and t > 0)

    Why IS and not 2026 OOS? The 2026 slice is ~3.5 months, n=54-72 per lane.
    At per-trade pnl_r std ≈ 0.9, the 95% CI on a 65-trade mean is ±0.22R.
    We cannot distinguish a +0.0R filter from a +0.15R filter at 2026 N.
    The IS period (2019-2025, n>1500 per lane) has real power.

    | baseline_edge | filter_discriminates | verdict               |
    |---------------|---------------------|-----------------------|
    | YES           | YES                 | BOTH_CONTRIBUTE       |
    | YES           | NO                  | FILTER_VESTIGIAL      |
    | NO            | YES                 | FILTER_IS_THE_EDGE    |
    | NO            | NO                  | DEAD_LANE             |

    Special case: if filter fires on every trade in IS (no non-fire subset,
    n<10), the filter-vs-non-fire test is untestable — report as
    FILTER_UNTESTABLE. Apply axis A separately: if baseline carries edge,
    the 100%-fire filter is operationally equivalent to no filter for
    every post-2021 trade but historically discriminated (pre-100%-fire era).
    """
    if unf_is["n"] < 500:
        return "INSUFFICIENT_IS_N"

    baseline_edge = abs(unf_is["t"]) > 2.0 and unf_is["mean"] > 0
    filter_discriminates = (not np.isnan(welch_p) and welch_p < 0.05
                            and not np.isnan(welch_t) and welch_t > 0)

    if non_fire_n < 10:
        # Filter fires ~100% — untestable axis B; report axis A only.
        return "FILTER_UNTESTABLE_BASELINE_EDGE" if baseline_edge else "FILTER_UNTESTABLE_BASELINE_WEAK"

    if baseline_edge and filter_discriminates:
        return "BOTH_CONTRIBUTE"
    if baseline_edge and not filter_discriminates:
        return "FILTER_VESTIGIAL"
    if not baseline_edge and filter_discriminates:
        return "FILTER_IS_THE_EDGE"
    return "DEAD_LANE"


def main() -> None:
    print("=" * 80)
    print(f"6-LANE UNFILTERED BASELINE STRESS-TEST  (ran {pd.Timestamp.now('UTC')})")
    print("=" * 80)

    payload = json.loads(LANE_ALLOCATION.read_text(encoding="utf-8"))
    deployed = [lane for lane in payload.get("lanes", []) if lane.get("status") == "DEPLOY"]
    print(f"DEPLOY lanes from {LANE_ALLOCATION}: {len(deployed)}")

    results = [audit_lane(lane) for lane in deployed]

    # Portfolio rollup
    print("\n" + "=" * 80)
    print("PORTFOLIO ROLLUP")
    print("=" * 80)
    counts: dict[str, int] = {}
    for r in results:
        c = r.get("classification", "ERROR")
        counts[c] = counts.get(c, 0) + 1
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:30s}: {n} lane(s)")

    print("\n  Per-lane verdict:")
    print(f"  {'Lane':50s} {'2026 unf ExpR':>14s} {'2026 flt ExpR':>14s} {'ΔExpR':>8s} {'fire%':>6s} {'class':20s}")
    for r in results:
        if "unf_oos" not in r:
            continue
        u = r["unf_oos"]["mean"]
        f = r["filt_oos"]["mean"]
        d = f - u if not np.isnan(f) else float("nan")
        fp = r.get("fire_pct_2026", float("nan"))
        sid = r["strategy_id"]
        print(f"  {sid:50s} {u:>+14.4f} {f:>+14.4f} {d:>+8.4f} {fp:>5.1f}% {r['classification']:20s}")

    # Write machine-readable JSON for the results MD
    out_path = Path("docs/audit/results/2026-04-20-6lane-unfiltered-baseline-stress.json")
    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "spec"}
        entry["spec"] = r.get("spec", {})
        serializable.append(entry)
    out_path.write_text(json.dumps(serializable, default=str, indent=2), encoding="utf-8")
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
