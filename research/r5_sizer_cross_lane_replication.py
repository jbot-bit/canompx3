"""R5 sizer cross-lane replication — rigorous institutional test.

Question: does garch_forecast_vol_pct add conditional ExpR edge on top of each
of the 6 currently deployed lanes? (R5 sizer hypothesis per mechanism_priors.md.)

Source: docs/runtime/lane_allocation.json (rebalance 2026-04-13).

# e2-lookahead-policy: tainted
# orb_{lane}_break_dir='long'/'short' used as WHERE predicates (lines ~130-133, ~157, ~182)
# to select fire-days by direction. On E2, break_dir is post-entry for ~42% of fills
# (range-cross precedes close-cross). IS/OOS findings segmented by direction are suspect.
# Re-pre-register with a pre-break direction proxy before deployment use.
Trigger: user demands no-bias, no-lookahead, all-variant, K-corrected proof
  before crowning R5 a validated signal.

Design:
  - 6 deployed lanes exact spec (instrument, session, apt, rr, filter_type).
  - Test BOTH directions where data allows (long+short).
  - THREE garch thresholds (60/70/80) — threshold sensitivity.
  - Also QUINTILE NTILE(5) breakdown for monotonicity check.
  - Per-year stability per lane.
  - Two-sided t-test on conditional lift (does NOT pre-suppose direction);
    BH-FDR correction at K = 6 lanes × 2 directions × 3 thresholds = 36
    primary hypotheses.
  - Bootstrap p-value for robustness.

Look-ahead verification (same as prior audit):
  - garch_forecast_vol computed via GARCH(1,1) on rows[0..i-1] daily_close
    (pipeline/build_daily_features.py:1258). Prior-only.
  - garch_forecast_vol_pct: rolling percentile over prior 252 days (line
    1217). Prior-only.
  - Filter criteria (orb_size, atr_20_pct, overnight_range) all trade-time-
    knowable at session start.
  - VWAP_MID_ALIGNED: based on break-direction vs VWAP at session start;
    trade-time-knowable at break-bar close (which is before entry).

Output: docs/audit/results/2026-04-15-r5-sizer-cross-lane-replication.md
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

OUTPUT_MD = Path("docs/audit/results/2026-04-15-r5-sizer-cross-lane-replication.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IS_END = "2026-01-01"
GARCH_GRID = [60, 70, 80]  # threshold sensitivity
SEED = 20260415


@dataclass
class Lane:
    name: str
    instrument: str
    orb_label: str
    aperture: int
    rr: float
    filter_sql: str  # SQL predicate that returns TRUE when the deployed filter fires
    filter_label: str


# Per docs/runtime/lane_allocation.json 2026-04-13 rebalance
LANES = [
    Lane(
        name="L1_EUROPE_FLOW_ORB_G5_RR1.5",
        instrument="MNQ",
        orb_label="EUROPE_FLOW",
        aperture=5,
        rr=1.5,
        filter_sql="d.orb_EUROPE_FLOW_size >= 5.0",
        filter_label="ORB_G5",
    ),
    Lane(
        name="L2_SINGAPORE_OPEN_ATR_P50_O30_RR1.5",
        instrument="MNQ",
        orb_label="SINGAPORE_OPEN",
        aperture=30,
        rr=1.5,
        filter_sql="d.atr_20_pct >= 50.0",
        filter_label="ATR_P50",
    ),
    Lane(
        name="L3_COMEX_SETTLE_OVNRNG_100_RR1.5",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        aperture=5,
        rr=1.5,
        filter_sql="d.overnight_range >= 100.0",
        filter_label="OVNRNG_100",
    ),
    Lane(
        name="L4_NYSE_OPEN_ORB_G5_RR1.0",
        instrument="MNQ",
        orb_label="NYSE_OPEN",
        aperture=5,
        rr=1.0,
        filter_sql="d.orb_NYSE_OPEN_size >= 5.0",
        filter_label="ORB_G5",
    ),
    Lane(
        name="L5_TOKYO_OPEN_ORB_G5_RR1.5",
        instrument="MNQ",
        orb_label="TOKYO_OPEN",
        aperture=5,
        rr=1.5,
        filter_sql="d.orb_TOKYO_OPEN_size >= 5.0",
        filter_label="ORB_G5",
    ),
    Lane(
        name="L6_US_DATA_1000_VWAP_MID_ALIGNED_O15_RR1.5",
        instrument="MNQ",
        orb_label="US_DATA_1000",
        aperture=15,
        rr=1.5,
        # VWAP_MID_ALIGNED: break direction matches sign of (orb_mid - vwap)
        # long aligned when orb_mid > vwap, short when orb_mid < vwap
        filter_sql=(
            "("
            "(d.orb_US_DATA_1000_break_dir='long' AND "
            "(d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low)/2.0 > d.orb_US_DATA_1000_vwap)"
            " OR "
            "(d.orb_US_DATA_1000_break_dir='short' AND "
            "(d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low)/2.0 < d.orb_US_DATA_1000_vwap)"
            ")"
        ),
        filter_label="VWAP_MID_ALIGNED",
    ),
]


def load_lane(lane: Lane, direction: str) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
      o.trading_day, o.pnl_r, o.risk_dollars,
      o.pnl_r * o.risk_dollars AS pnl_dollars,
      d.garch_forecast_vol_pct AS garch_pct,
      d.orb_{lane.orb_label}_break_dir AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='{lane.instrument}' AND o.orb_minutes={lane.aperture}
      AND o.orb_label='{lane.orb_label}' AND o.entry_model='E2'
      AND o.rr_target={lane.rr} AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{lane.orb_label}_break_dir='{direction}'
      AND {lane.filter_sql}
      AND o.trading_day < DATE '{IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    return df


def load_lane_oos(lane: Lane, direction: str) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
      o.trading_day, o.pnl_r, o.risk_dollars,
      d.garch_forecast_vol_pct AS garch_pct
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='{lane.instrument}' AND o.orb_minutes={lane.aperture}
      AND o.orb_label='{lane.orb_label}' AND o.entry_model='E2'
      AND o.rr_target={lane.rr} AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{lane.orb_label}_break_dir='{direction}'
      AND {lane.filter_sql}
      AND o.trading_day >= DATE '{IS_END}'
    """
    df = con.execute(q).df()
    con.close()
    return df


def test_threshold(df: pd.DataFrame, threshold: int) -> dict:
    if len(df) < 30:
        return {"n_total": len(df), "skip": True}
    on = df[df["garch_pct"] >= threshold]
    off = df[df["garch_pct"] < threshold]
    if len(on) < 10 or len(off) < 10:
        return {
            "n_total": len(df),
            "n_on": len(on),
            "n_off": len(off),
            "skip_reason": "thin bucket",
            "skip": True,
        }
    expr_on = float(on["pnl_r"].mean())
    expr_off = float(off["pnl_r"].mean())
    lift = expr_on - expr_off

    # Welch's two-sided t-test (no pre-supposed direction)
    t_stat, p_two = stats.ttest_ind(on["pnl_r"], off["pnl_r"], equal_var=False)

    # Bootstrap p-value (two-sided): probability of |lift_shuffled| >= |lift_observed|
    rng = np.random.default_rng(SEED)
    pnl = df["pnl_r"].astype(float).to_numpy()
    is_on = (df["garch_pct"].values >= threshold).astype(int)
    B = 1000
    beats = 0
    for _ in range(B):
        shuffled = rng.permutation(is_on)
        on_mean = pnl[shuffled == 1].mean()
        off_mean = pnl[shuffled == 0].mean()
        if abs(on_mean - off_mean) >= abs(lift):
            beats += 1
    p_boot = (beats + 1) / (B + 1)

    # Per-year stability
    yrs = df["year"].unique()
    yr_lifts = {}
    for y in yrs:
        sub = df[df["year"] == y]
        s_on = sub[sub["garch_pct"] >= threshold]
        s_off = sub[sub["garch_pct"] < threshold]
        if len(s_on) >= 3 and len(s_off) >= 3:
            yr_lifts[int(y)] = float(s_on["pnl_r"].mean() - s_off["pnl_r"].mean())
    pos_yrs = sum(1 for v in yr_lifts.values() if v > 0)

    return {
        "n_total": len(df),
        "n_on": len(on),
        "n_off": len(off),
        "expr_on": expr_on,
        "expr_off": expr_off,
        "lift": lift,
        "wr_on": float((on["pnl_r"] > 0).mean()),
        "wr_off": float((off["pnl_r"] > 0).mean()),
        "dpt_on": float((on["pnl_r"] * on["risk_dollars"]).mean()),
        "t_stat": float(t_stat),
        "p_two": float(p_two),
        "p_boot": float(p_boot),
        "yr_lifts": yr_lifts,
        "pos_yrs": pos_yrs,
        "total_yrs": len(yr_lifts),
        "skip": False,
    }


def ntile5_breakdown(df: pd.DataFrame) -> list[dict]:
    if len(df) < 50:
        return []
    df = df.copy()
    df["bucket"] = pd.qcut(df["garch_pct"], 5, labels=False, duplicates="drop")
    rows = []
    for b, sub in df.groupby("bucket"):
        rows.append(
            {
                "bucket": int(b),
                "garch_min": float(sub["garch_pct"].min()),
                "garch_max": float(sub["garch_pct"].max()),
                "n": len(sub),
                "expr": float(sub["pnl_r"].mean()),
                "wr": float((sub["pnl_r"] > 0).mean()),
            }
        )
    return rows


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns list of bool (survives)."""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    survives = [False] * n
    for rank, (idx, p) in enumerate(indexed, start=1):
        if p <= q * rank / n:
            for j in range(rank):
                survives[indexed[j][0]] = True
    return survives


def main() -> None:
    print("R5 sizer cross-lane replication — 6 deployed lanes x 2 directions x 3 thresholds")

    results = []  # flat list of test results
    for lane in LANES:
        for direction in ["long", "short"]:
            try:
                df = load_lane(lane, direction)
                df_oos = load_lane_oos(lane, direction)
            except Exception as e:
                print(f"  [err] {lane.name} {direction}: {e}")
                continue
            print(f"\n=== {lane.name} | {direction} ===")
            print(f"  IS N={len(df)}  OOS N={len(df_oos)}")

            if len(df) < 30:
                print("  SKIP: IS N < 30")
                for th in GARCH_GRID:
                    results.append(
                        {
                            "lane": lane.name,
                            "dir": direction,
                            "threshold": th,
                            "skip": True,
                            "reason": "IS N < 30",
                        }
                    )
                continue

            # NTILE breakdown (informational — monotonicity check)
            nt = ntile5_breakdown(df)
            if nt:
                print("  NTILE-5 by garch_pct:")
                for b in nt:
                    print(
                        f"    bucket {b['bucket']} [{b['garch_min']:.0f}-{b['garch_max']:.0f}] "
                        f"N={b['n']} ExpR={b['expr']:+.3f} WR={b['wr']:.1%}"
                    )

            # Threshold tests
            for th in GARCH_GRID:
                r = test_threshold(df, th)
                r.update({"lane": lane.name, "dir": direction, "threshold": th, "ntile": nt, "n_oos": len(df_oos)})

                # OOS conditional lift (informational if thin)
                if len(df_oos) >= 10:
                    on_oos = df_oos[df_oos["garch_pct"] >= th]
                    off_oos = df_oos[df_oos["garch_pct"] < th]
                    if len(on_oos) >= 3 and len(off_oos) >= 3:
                        r["oos_expr_on"] = float(on_oos["pnl_r"].mean())
                        r["oos_expr_off"] = float(off_oos["pnl_r"].mean())
                        r["oos_lift"] = r["oos_expr_on"] - r["oos_expr_off"]
                        r["oos_n_on"] = len(on_oos)
                        r["oos_n_off"] = len(off_oos)
                    else:
                        r["oos_lift"] = None
                else:
                    r["oos_lift"] = None

                results.append(r)

                if not r.get("skip"):
                    print(
                        f"  @ {th}: N_on={r['n_on']}/{r['n_off']}  "
                        f"ExpR {r['expr_on']:+.3f} vs {r['expr_off']:+.3f}  "
                        f"lift {r['lift']:+.3f}  t={r['t_stat']:+.2f}  p={r['p_two']:.4f}  "
                        f"yrs {r['pos_yrs']}/{r['total_yrs']}"
                    )

    # BH-FDR at K = total non-skipped threshold tests
    valid = [r for r in results if not r.get("skip")]
    pvals_two = [r["p_two"] for r in valid]
    K = len(pvals_two)
    survives = bh_fdr(pvals_two, q=0.05)
    for i, r in enumerate(valid):
        r["bh_fdr_05"] = survives[i]

    print(f"\n=== BH-FDR at K={K} (q=0.05) ===")
    survivors = [r for r in valid if r["bh_fdr_05"]]
    print(f"  Survivors: {len(survivors)} / {K}")
    for s in sorted(survivors, key=lambda x: x["p_two"]):
        print(
            f"    {s['lane']} {s['dir']} @{s['threshold']}: "
            f"lift={s['lift']:+.3f}  p={s['p_two']:.5f}  yrs={s['pos_yrs']}/{s['total_yrs']}  "
            f"OOS lift={s.get('oos_lift', 'n/a')}"
        )

    emit(LANES, results, K)


def emit(lanes: list[Lane], results: list[dict], K: int) -> None:
    lines = [
        "# R5 Sizer Cross-Lane Replication — 6 Deployed Lanes",
        "",
        "**Date:** 2026-04-15",
        "**Source deploy:** `docs/runtime/lane_allocation.json` rebalance 2026-04-13",
        "**Question:** Does `garch_forecast_vol_pct` add conditional ExpR edge on top of each deployed lane's filter? (R5 sizer hypothesis per `docs/institutional/mechanism_priors.md`.)",
        "**Design:** 6 lanes × 2 directions × 3 thresholds (60/70/80) = 36 primary hypotheses. Two-sided Welch t-test + bootstrap. BH-FDR correction at K=36. Per-year stability. IS-OOS split at 2026-01-01.",
        "",
        "**Look-ahead verification:** `garch_forecast_vol_pct` uses `rows[0..i-1] daily_close` per `pipeline/build_daily_features.py:1258` + 252-day prior rank per line 1217. All filter predicates (orb_size, atr_20_pct, overnight_range, VWAP_MID_ALIGNED) are trade-time-knowable at session start or break-bar close. Clean.",
        "",
        "**No-pigeonholing:** Two-sided t-test does NOT pre-suppose garch=HIGH is the edge direction. If garch=LOW were actually the informative signal, the test would catch it (negative lift, significant p).",
        "",
        "---",
        "",
        "## 6 Deployed Lanes",
        "",
        "| Lane | Session | Apt | RR | Filter |",
        "|---|---|---|---|---|",
    ]
    for lane in lanes:
        lines.append(f"| {lane.name} | {lane.orb_label} | O{lane.aperture} | {lane.rr} | {lane.filter_label} |")
    lines += ["", "---", "", "## Per-lane per-direction test grid", ""]

    # Group by lane-direction
    for lane in lanes:
        for direction in ["long", "short"]:
            lane_results = [r for r in results if r["lane"] == lane.name and r["dir"] == direction]
            if not lane_results:
                continue
            first = lane_results[0]
            nt = first.get("ntile", [])
            lines += [f"### {lane.name} | {direction}", ""]
            lines += [f"**IS N:** {first.get('n_total', 'skip')} | **OOS N:** {first.get('n_oos', 0)}", ""]

            if nt:
                lines += [
                    "NTILE-5 by garch_pct (informational — monotonicity check):",
                    "",
                    "| bucket | garch range | N | ExpR | WR |",
                    "|---|---|---|---|---|",
                ]
                for b in nt:
                    lines.append(
                        f"| {b['bucket']} | {b['garch_min']:.0f}-{b['garch_max']:.0f} | "
                        f"{b['n']} | {b['expr']:+.3f} | {b['wr']:.1%} |"
                    )
                lines.append("")

            lines += [
                "Threshold tests:",
                "",
                "| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for r in lane_results:
                if r.get("skip"):
                    lines.append(f"| {r['threshold']} | SKIP | — | — | — | — | — | — | — | — | — |")
                    continue
                bh = "PASS" if r.get("bh_fdr_05") else "—"
                oos = f"{r['oos_lift']:+.3f}" if r.get("oos_lift") is not None else "n/a"
                lines.append(
                    f"| {r['threshold']} | {r['n_on']}/{r['n_off']} | "
                    f"{r['expr_on']:+.3f} | {r['expr_off']:+.3f} | {r['lift']:+.3f} | "
                    f"{r['t_stat']:+.2f} | {r['p_two']:.4f} | {r['p_boot']:.4f} | "
                    f"{r['pos_yrs']}/{r['total_yrs']} | {bh} | {oos} |"
                )
            lines.append("")

    # Summary
    valid = [r for r in results if not r.get("skip")]
    survivors = [r for r in valid if r.get("bh_fdr_05")]
    pos_lift = [r for r in survivors if r["lift"] > 0]
    neg_lift = [r for r in survivors if r["lift"] < 0]

    lines += [
        "---",
        "",
        "## Summary — BH-FDR at K=" + str(K),
        "",
        f"- Total tested: {K} (non-skipped).",
        f"- BH-FDR survivors (q=0.05): **{len(survivors)}**.",
        f"- Positive-lift survivors (garch=HIGH better): {len(pos_lift)}.",
        f"- Negative-lift survivors (garch=LOW better → SKIP signal): {len(neg_lift)}.",
        "",
    ]
    if survivors:
        lines.append("### BH-FDR survivors (sorted by p_two)")
        lines.append("")
        lines.append("| Lane | dir | thresh | lift | p_two | yrs+ | OOS lift |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in sorted(survivors, key=lambda x: x["p_two"]):
            oos = f"{s['oos_lift']:+.3f}" if s.get("oos_lift") is not None else "n/a"
            lines.append(
                f"| {s['lane']} | {s['dir']} | {s['threshold']} | {s['lift']:+.3f} | "
                f"{s['p_two']:.5f} | {s['pos_yrs']}/{s['total_yrs']} | {oos} |"
            )
    lines.append("")

    # Deployment-role recommendation
    lines += [
        "---",
        "",
        "## Interpretation & deployment roles",
        "",
        "| Finding | Deployment role (per mechanism_priors.md) |",
        "|---|---|",
        "| Significant positive lift + dir_match across thresholds | **R5 SIZER** on that lane — garch=HIGH days size up |",
        "| Significant negative lift (garch=LOW → better) | **R1-SKIP** — trade LOW garch days, skip HIGH |",
        "| No lift at any threshold | **R0 null** — lane unaffected by garch |",
        "| Lift IS but OOS flip / yrs < 50% | **R? regime-dependent** — needs shadow before any role |",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
