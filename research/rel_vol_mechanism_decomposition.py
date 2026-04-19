"""Mechanism decomposition — what is rel_vol_HIGH_Q3 actually catching?

Per 2026-04-15 user directive: no narrative injection, let the raw data speak.
Empirical test of whether rel_vol_HIGH is independent of vol / regime / calendar
proxies, or is actually a proxy for one of them.

Test methodology:
1. Per BH-global survivor lane, compute correlation matrix of:
   - rel_vol_HIGH_Q3 (target)
   - atr_vel_HIGH (vol acceleration, P67)
   - atr_20_pct_GT80 (high vol regime)
   - garch_vol_pct_GT70 (forward vol regime)
   - day_type ∈ {trend_up, trend_down, range}
   - gap_UP, gap_DOWN
   - is_nfp, is_opex, is_friday, is_monday
2. Partial-dependence test: Welch t of rel_vol_HIGH stratified by:
   - atr_vel_HIGH vs atr_vel_LOW
   - day_type ∈ {trend_up, trend_down, range}
   - calendar flags
3. Multi-variate regression: does rel_vol_HIGH add predictive power for pnl_r
   beyond atr_vel_HIGH + day_type + atr_20_pct_GT80?
4. Report raw numbers only. No pre-labeled mechanism.

Output:
  docs/audit/results/2026-04-15-rel-vol-mechanism-decomposition.md
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OOS_START = HOLDOUT_SACRED_FROM

# 5 BH-global volume survivor lanes from comprehensive scan
LANES = [
    ("MES", "COMEX_SETTLE", 5, 1.0, "short"),
    ("MGC", "LONDON_METALS", 5, 1.0, "short"),
    ("MES", "TOKYO_OPEN", 5, 1.5, "long"),
    ("MNQ", "SINGAPORE_OPEN", 5, 1.0, "short"),
    ("MES", "COMEX_SETTLE", 5, 1.5, "short"),
]

OUTPUT_MD = Path("docs/audit/results/2026-04-15-rel-vol-mechanism-decomposition.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_lane(instrument, session, apt, rr):
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
      o.trading_day, o.pnl_r,
      d.atr_20, d.atr_20_pct, d.atr_vel_ratio,
      d.garch_forecast_vol_pct,
      d.day_type, d.gap_type,
      d.is_nfp_day, d.is_opex_day, d.is_friday, d.is_monday, d.day_of_week,
      d.rel_vol_{session} AS rel_vol,
      d.orb_{session}_break_dir AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol
      AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{session}'
      AND o.symbol = '{instrument}'
      AND o.orb_minutes = {apt}
      AND o.entry_model = 'E2'
      AND o.rr_target = {rr}
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
      AND d.rel_vol_{session} IS NOT NULL
    """
    df = con.execute(q).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    return df


def build_indicators(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Build binary indicator columns for mechanism decomposition.

    LOOK-AHEAD AUDIT: only trade-time-knowable features allowed.
    - rel_vol: computed at ORB end ✓
    - atr_vel_ratio, atr_20_pct: 20-day rolling from prior close ✓
    - garch_forecast_vol_pct: forecast made at prior close ✓
    - gap_type: session open, before ORB ✓
    - Calendar (is_nfp/opex/friday/monday): ✓
    - day_type: EXCLUDED — pipeline/build_daily_features.py:510 says
      "LOOK-AHEAD relative to intraday entry. For research only."
    """
    sub = df[df["break_dir"] == direction].copy()
    if len(sub) == 0:
        return sub
    # Compute IS percentiles for thresholds (IS-only, no OOS leakage)
    is_sub = sub[sub["is_is"]]
    rel_vol_p67 = np.nanpercentile(is_sub["rel_vol"].astype(float), 67)
    atr_vel_p67 = (
        np.nanpercentile(is_sub["atr_vel_ratio"].astype(float), 67) if is_sub["atr_vel_ratio"].notna().any() else np.nan
    )
    sub["rel_vol_HIGH"] = (sub["rel_vol"].astype(float) > rel_vol_p67).fillna(False).astype(int)
    sub["atr_vel_HIGH"] = (
        (sub["atr_vel_ratio"].astype(float) > atr_vel_p67).fillna(False).astype(int) if not np.isnan(atr_vel_p67) else 0
    )
    sub["atr_20_pct_GT80"] = (sub["atr_20_pct"].astype(float) > 80).fillna(False).astype(int)
    sub["atr_20_pct_LT20"] = (sub["atr_20_pct"].astype(float) < 20).fillna(False).astype(int)
    sub["garch_vol_pct_GT70"] = (sub["garch_forecast_vol_pct"].astype(float) > 70).fillna(False).astype(int)
    sub["garch_vol_pct_LT30"] = (sub["garch_forecast_vol_pct"].astype(float) < 30).fillna(False).astype(int)
    # day_type EXCLUDED — look-ahead per pipeline/build_daily_features.py:510
    sub["gap_up"] = (sub["gap_type"] == "gap_up").astype(int)
    sub["gap_down"] = (sub["gap_type"] == "gap_down").astype(int)
    sub["is_nfp"] = sub["is_nfp_day"].fillna(False).astype(int)
    sub["is_opex"] = sub["is_opex_day"].fillna(False).astype(int)
    sub["is_fri"] = sub["is_friday"].fillna(False).astype(int)
    sub["is_mon"] = sub["is_monday"].fillna(False).astype(int)
    return sub


def correlation_table(df_is: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation of rel_vol_HIGH with TRADE-TIME-KNOWABLE proxies only.
    day_type excluded (look-ahead per pipeline source)."""
    indicators = [
        "atr_vel_HIGH",
        "atr_20_pct_GT80",
        "atr_20_pct_LT20",
        "garch_vol_pct_GT70",
        "garch_vol_pct_LT30",
        "gap_up",
        "gap_down",
        "is_nfp",
        "is_opex",
        "is_fri",
        "is_mon",
    ]
    rows = []
    for ind in indicators:
        if ind not in df_is.columns:
            continue
        try:
            c_result = np.corrcoef(df_is["rel_vol_HIGH"].astype(float), df_is[ind].astype(float))
            c = float(c_result[0, 1])
            if np.isnan(c):
                c = 0.0
        except Exception:
            c = 0.0
        rows.append({"indicator": ind, "corr_vs_rel_vol_HIGH": c})
    return pd.DataFrame(rows).sort_values("corr_vs_rel_vol_HIGH", key=abs, ascending=False)


def partial_dependence(df_is: pd.DataFrame, strat_col: str) -> dict:
    """Welch t of pnl_r by rel_vol_HIGH stratified on `strat_col`."""
    out = {}
    for stratum_val in [0, 1]:
        sub = df_is[df_is[strat_col] == stratum_val]
        on = sub[sub["rel_vol_HIGH"] == 1]["pnl_r"]
        off = sub[sub["rel_vol_HIGH"] == 0]["pnl_r"]
        if len(on) >= 30 and len(off) >= 30:
            t, p = stats.ttest_ind(on, off, equal_var=False)
            out[stratum_val] = {
                "n_on": len(on),
                "n_off": len(off),
                "expr_on": float(on.mean()),
                "expr_off": float(off.mean()),
                "delta": float(on.mean() - off.mean()),
                "t": float(t),
                "p": float(p),
            }
        else:
            out[stratum_val] = {
                "n_on": len(on),
                "n_off": len(off),
                "expr_on": float("nan"),
                "expr_off": float("nan"),
                "delta": float("nan"),
                "t": float("nan"),
                "p": float("nan"),
            }
    return out


def multivariate_regression(df_is: pd.DataFrame) -> dict:
    """Does rel_vol_HIGH add predictive power BEYOND trade-time-knowable vol/regime/calendar?
    Fit pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + garch_vol_pct_GT70 + gap_up + gap_down + is_nfp + is_opex.
    No day_type (look-ahead). Report t-stat of rel_vol_HIGH coefficient."""
    cols = [
        "rel_vol_HIGH",
        "atr_vel_HIGH",
        "atr_20_pct_GT80",
        "garch_vol_pct_GT70",
        "gap_up",
        "gap_down",
        "is_nfp",
        "is_opex",
    ]

    def fire_count(s):
        return int(np.asarray(s).sum())

    available = [c for c in cols if c in df_is.columns and fire_count(df_is[c]) >= 10]
    if "rel_vol_HIGH" not in available or len(available) < 2:
        return {"error": "insufficient indicators"}
    X_raw = df_is[available].astype(float)
    X = sm.add_constant(X_raw)
    y = df_is["pnl_r"].astype(float)
    valid = (~X.isna().any(axis=1)) & (~y.isna())
    X_v = X[valid]
    y_v = y[valid]
    if len(y_v) < 100:
        return {"error": f"insufficient N={len(y_v)}"}
    try:
        model = sm.OLS(y_v, X_v).fit()
        rv_t = float(model.tvalues.get("rel_vol_HIGH", float("nan")))
        rv_coef = float(model.params.get("rel_vol_HIGH", float("nan")))
        rv_p = float(model.pvalues.get("rel_vol_HIGH", float("nan")))
        return {
            "rel_vol_coef": rv_coef,
            "rel_vol_t": rv_t,
            "rel_vol_p": rv_p,
            "adj_r2": float(model.rsquared_adj),
            "n_used": int(len(y_v)),
            "covariates": [c for c in available if c != "rel_vol_HIGH"],
        }
    except Exception as e:
        return {"error": str(e)}


def emit(results: list[dict]) -> None:
    lines = [
        "# rel_vol_HIGH_Q3 Mechanism Decomposition",
        "",
        "**Date:** 2026-04-15",
        "**Source:** 5 BH-global survivor lanes from comprehensive scan",
        "**Purpose:** empirical test of whether rel_vol_HIGH is INDEPENDENT of vol-regime / day-type / calendar proxies, or is a re-labeled version of one of them. No narrative injection — raw numbers only.",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['instrument']} {r['session']} O{r['apt']} RR{r['rr']:.1f} {r['direction']}",
            f"**N_is:** {r['n_is']} | **N_on_is (rel_vol_HIGH fire):** {r['n_on']} | **Fire rate:** {r['fire_rate']:.1%}",
            "",
            "### Correlation of rel_vol_HIGH with candidate proxies (IS data)",
            "",
            "| Indicator | corr vs rel_vol_HIGH |",
            "|-----------|----------------------|",
        ]
        for _, row in r["corr_table"].iterrows():
            lines.append(f"| {row['indicator']} | {row['corr_vs_rel_vol_HIGH']:+.3f} |")

        lines += [
            "",
            "### Partial dependence — does rel_vol_HIGH predict pnl_r within subsets?",
            "",
        ]
        for strat_col, strat_results in r["partial"].items():
            lines += [
                f"**Stratified on {strat_col}:**",
                "",
                "| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |",
                "|---------|------|-------|---------|----------|---|---|---|",
            ]
            for v, res in strat_results.items():
                lines.append(
                    f"| {strat_col}={v} | {res['n_on']} | {res['n_off']} | "
                    f"{res['expr_on']:+.3f} | {res['expr_off']:+.3f} | {res['delta']:+.3f} | "
                    f"{res['t']:+.2f} | {res['p']:.4f} |"
                )
            lines.append("")

        lines += [
            "### Multivariate regression",
            "`pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + day_type indicators`",
            "",
        ]
        mv = r["multivar"]
        if "error" in mv:
            lines.append(f"ERROR: {mv['error']}")
        else:
            lines += [
                f"- **rel_vol_HIGH coefficient:** {mv['rel_vol_coef']:+.4f}",
                f"- **rel_vol_HIGH t-stat:** {mv['rel_vol_t']:+.2f}",
                f"- **rel_vol_HIGH p:** {mv['rel_vol_p']:.4f}",
                f"- **Adj R²:** {mv['adj_r2']:.4f}",
                f"- **N used:** {mv['n_used']}",
                f"- **Covariates:** {', '.join(mv['covariates'])}",
                "",
                "**Interpretation of rel_vol_HIGH coefficient:**",
                f"  - coefficient > 0 AND |t| >= 2.0: rel_vol_HIGH adds INDEPENDENT predictive power beyond the controls.",
                f"  - coefficient near 0 OR |t| < 2.0: rel_vol_HIGH effect is MEDIATED (absorbed) by the controls.",
            ]
        lines += ["", "---", ""]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main():
    results = []
    for instr, session, apt, rr, direction in LANES:
        print(f"\n=== {instr} {session} O{apt} RR{rr:.1f} {direction} ===")
        df = load_lane(instr, session, apt, rr)
        sub = build_indicators(df, direction)
        is_sub = sub[sub["is_is"]]
        if len(is_sub) < 100:
            print(f"  insufficient N={len(is_sub)}")
            continue

        n_on = int(is_sub["rel_vol_HIGH"].sum())
        fire_rate = n_on / len(is_sub)
        print(f"  IS N={len(is_sub)}, rel_vol_HIGH fire N={n_on} ({fire_rate:.1%})")

        corr_table = correlation_table(is_sub)
        print("  top correlations:")
        for _, row in corr_table.head(5).iterrows():
            print(f"    {row['indicator']}: {row['corr_vs_rel_vol_HIGH']:+.3f}")

        partial = {}
        # Stratify on trade-time-knowable variables only (no look-ahead)
        for strat_col in ["atr_vel_HIGH", "atr_20_pct_GT80", "garch_vol_pct_GT70", "gap_up", "is_nfp"]:
            if strat_col in is_sub.columns:
                partial[strat_col] = partial_dependence(is_sub, strat_col)

        mv = multivariate_regression(is_sub)
        if "rel_vol_t" in mv:
            print(
                f"  multivariate rel_vol_HIGH t={mv['rel_vol_t']:+.2f}, coef={mv['rel_vol_coef']:+.4f}, p={mv['rel_vol_p']:.4f}"
            )

        results.append(
            {
                "instrument": instr,
                "session": session,
                "apt": apt,
                "rr": rr,
                "direction": direction,
                "n_is": len(is_sub),
                "n_on": n_on,
                "fire_rate": fire_rate,
                "corr_table": corr_table,
                "partial": partial,
                "multivar": mv,
            }
        )

    emit(results)


if __name__ == "__main__":
    main()
