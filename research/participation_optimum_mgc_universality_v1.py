"""Participation-optimum universality test on MGC — asset-class generalisation check.

Instrument-swap of research/participation_optimum_mes_universality_v1.py.
Same pooled quadratic regression, K=1 primary + K=18 per-cell secondary.

Pre-reg: docs/audit/hypotheses/2026-04-20-participation-optimum-mgc-universality-v1.yaml
Parent MNQ verdict: CONFIRMED_UNIVERSAL (beta2=-0.00156, t=-5.189, 21/24 cells).
Parent MES verdict: MES_NO_REPLICATION (same sign, t=-1.834, below threshold).

No capital action permitted under any verdict.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-participation-optimum-mgc-universality-v1.yaml"
PREREG_SHA = "9ea0a5ef"
RESULT_DOC = Path("docs/audit/results/2026-04-20-participation-optimum-mgc-universality-v1.md")

INSTRUMENT = "MGC"
# Canonically discovered 2026-04-20 via:
#   SELECT DISTINCT orb_label FROM orb_outcomes
#   WHERE symbol='MGC' AND orb_minutes=5 AND pnl_r IS NOT NULL
SESSIONS = [
    "CME_REOPEN",
    "COMEX_SETTLE",
    "EUROPE_FLOW",
    "LONDON_METALS",
    "NYSE_OPEN",
    "SINGAPORE_OPEN",
    "TOKYO_OPEN",
    "US_DATA_830",
    "US_DATA_1000",
]
DIRECTIONS = ["long", "short"]

# Parent summaries for cross-instrument comparison table
MNQ_POOLED = {"beta2": -0.00156, "t": -5.189, "p": 0.0, "N": 17828, "agreement": 0.875}
MES_POOLED = {"beta2": -0.00061, "t": -1.834, "p": 0.0333, "N": None, "agreement": 0.955}


def _load_session(con: duckdb.DuckDBPyConnection, session: str) -> pd.DataFrame:
    rel_col = f"rel_vol_{session}"
    # RULE 9: triple-join orb_outcomes x daily_features on (trading_day, symbol, orb_minutes).
    # CTE reads daily_features with orb_minutes=5 only to prevent 3x N-inflation.
    sql = f"""
    WITH df AS (
      SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
      FROM daily_features d
      WHERE d.symbol = '{INSTRUMENT}' AND d.orb_minutes = 5
    )
    SELECT o.trading_day, o.pnl_r, o.entry_price, o.stop_price, df.rel_vol
    FROM orb_outcomes o
    JOIN df ON o.trading_day = df.trading_day AND o.symbol = df.symbol
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.5
      AND o.pnl_r IS NOT NULL
    """
    df = con.sql(sql).to_df()
    if len(df) == 0:
        return df
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["session"] = session
    df["lane"] = df["session"] + "_" + df["direction"]
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def _load_all(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frames = [sub for s in SESSIONS if len(sub := _load_session(con, s)) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass
class RegressionResult:
    n: int
    beta0: float
    beta1: float
    beta2: float
    beta2_se: float
    beta2_t: float
    beta2_p_one_tailed: float


def _regress_quadratic(df: pd.DataFrame, with_lane_fe: bool) -> RegressionResult:
    sub = df.dropna(subset=["rel_vol", "pnl_r"]).copy()
    if len(sub) < 50:
        return RegressionResult(len(sub), *([float("nan")] * 6))
    sub["rel_vol_sq"] = sub["rel_vol"].astype(float) ** 2
    X_cols = ["rel_vol", "rel_vol_sq"]
    X = sub[X_cols].astype(float).copy()
    if with_lane_fe and sub["lane"].nunique() > 1:
        lane_d = pd.get_dummies(sub["lane"], drop_first=True, dtype=float)
        X = pd.concat([X, lane_d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = sub["pnl_r"].astype(float)
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return RegressionResult(len(sub), *([float("nan")] * 6))
    b2 = float(model.params.get("rel_vol_sq", float("nan")))
    se2 = float(model.bse.get("rel_vol_sq", float("nan")))
    t2 = float(model.tvalues.get("rel_vol_sq", float("nan")))
    p_one = float(sm.stats.stattools.stats.t.cdf(t2, model.df_resid)) if not np.isnan(t2) else float("nan")
    return RegressionResult(
        n=len(sub),
        beta0=float(model.params.get("const", float("nan"))),
        beta1=float(model.params.get("rel_vol", float("nan"))),
        beta2=b2, beta2_se=se2, beta2_t=t2, beta2_p_one_tailed=p_one,
    )


def _is_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_mask = df["trading_day"] < holdout
    return df.loc[is_mask].reset_index(drop=True), df.loc[~is_mask].reset_index(drop=True)


def _render_percell(rows: list[tuple[str, str, RegressionResult]]) -> str:
    lines = ["| session | direction | N | beta2 | t(beta2) | one-tailed p | sign |",
             "|---|---|---:|---:|---:|---:|:---:|"]
    for session, direction, res in sorted(rows, key=lambda r: (r[0], r[1])):
        if np.isnan(res.beta2):
            lines.append(f"| {session} | {direction} | {res.n} | - | - | - | - |")
        else:
            lines.append(f"| {session} | {direction} | {res.n} | "
                         f"{res.beta2:+.4f} | {res.beta2_t:+.3f} | "
                         f"{res.beta2_p_one_tailed:.4f} | "
                         f"{'neg' if res.beta2 < 0 else 'pos'} |")
    return "\n".join(lines)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        data = _load_all(con)
    finally:
        con.close()
    if len(data) == 0:
        print("VERDICT: SCAN_ABORT - no MGC canonical data at 5m")
        return 1
    is_df, _ = _is_split(data)
    rv_non_null = is_df["rel_vol"].notna().sum() / max(len(is_df), 1)
    integrity_ok = rv_non_null >= 0.90

    pooled = _regress_quadratic(is_df, with_lane_fe=True)

    percell = []
    for s in SESSIONS:
        for d in DIRECTIONS:
            cell = is_df[(is_df["session"] == s) & (is_df["direction"] == d)]
            percell.append((s, d, _regress_quadratic(cell, with_lane_fe=False) if len(cell) >= 50
                                    else RegressionResult(len(cell), *([float("nan")] * 6))))

    valid = [r for _, _, r in percell if not np.isnan(r.beta2)]
    neg = [r for r in valid if r.beta2 < 0]
    agreement = len(neg) / max(len(valid), 1)

    if not integrity_ok or pooled.n < 1000:
        verdict, reason = "SCAN_ABORT", (
            f"integrity fail: rel_vol non-null {rv_non_null:.1%} "
            f"(need >= 90%) OR pooled N {pooled.n} < 1000"
        )
    elif np.isnan(pooled.beta2_t) or pooled.beta2 >= 0 or pooled.beta2_t > -3.0 or pooled.beta2_p_one_tailed >= 0.05:
        verdict = "MGC_NO_REPLICATION"
        reason = (f"Pooled beta2={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} p={pooled.beta2_p_one_tailed:.4f} "
                  f"does NOT clear t<=-3.0 or p<0.05. MNQ finding does NOT replicate on MGC — "
                  f"mechanism is index-specific or attenuated on gold.")
    elif agreement < 0.25:
        verdict = "MGC_SIMPSON_ARTEFACT"
        reason = f"Pooled significant but per-lane agreement {agreement:.1%} < 25% — Simpson's artefact."
    elif agreement < 0.50:
        verdict = "MGC_CONFIRMED_HETEROGENEOUS"
        reason = (f"Pooled beta2={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} significant; "
                  f"per-lane agreement {agreement:.1%} in 25-50% band — heterogeneous replication. "
                  f"Per-lane breakdown required per RULE 14 before any downstream use.")
    else:
        verdict = "MGC_CONFIRMED_UNIVERSAL"
        reason = (f"Pooled beta2={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} significant; "
                  f"per-lane agreement {agreement:.1%} >= 50%. Mechanism replicates on MGC.")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    parts.append(f"# Participation-optimum universality — MGC v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`")
    parts.append(f"**Script:** `research/participation_optimum_mgc_universality_v1.py`")
    parts.append(f"**Parent MNQ merged:** PR #41, commit `126ed6b8`")
    parts.append(f"**Parent MES merged:** PR #42, commit `75b0d0bc`")
    parts.append(f"**Scope:** MGC x 9 canonical sessions x both directions x 5m E2 CB1 RR1.5 unfiltered, IS only")
    parts.append("")
    parts.append(f"## Verdict: **{verdict}**")
    parts.append("")
    parts.append(f"> {reason}")
    parts.append("")
    parts.append("## Cross-instrument comparison")
    parts.append("")
    parts.append("| instrument | N (IS pooled) | beta2 | t | one-tailed p | per-lane agreement |")
    parts.append("|---|---:|---:|---:|---:|---:|")
    parts.append(f"| MNQ (parent) | {MNQ_POOLED['N']:,} | {MNQ_POOLED['beta2']:+.5f} | {MNQ_POOLED['t']:+.3f} | {MNQ_POOLED['p']:.4f} | {MNQ_POOLED['agreement']:.1%} |")
    mes_n_str = f"{MES_POOLED['N']:,}" if MES_POOLED["N"] else "n/a"
    parts.append(f"| MES (parent) | {mes_n_str} | {MES_POOLED['beta2']:+.5f} | {MES_POOLED['t']:+.3f} | {MES_POOLED['p']:.4f} | {MES_POOLED['agreement']:.1%} |")
    parts.append(f"| MGC (this)   | {pooled.n:,} | {pooled.beta2:+.5f} | {pooled.beta2_t:+.3f} | {pooled.beta2_p_one_tailed:.4f} | {agreement:.1%} |")
    parts.append("")
    parts.append("## Integrity")
    parts.append("")
    parts.append(f"- rel_vol non-null on IS 5m: {rv_non_null:.1%} (threshold >= 90%)")
    parts.append(f"- IS 5m N (raw): {len(is_df)}")
    parts.append(f"- IS 5m N (rel_vol available): {pooled.n}")
    parts.append(f"- Sessions tested: {len(SESSIONS)}")
    parts.append(f"- Cells loaded: {len(SESSIONS) * 2} (max)")
    parts.append("")
    parts.append("## Pooled regression (5m, IS, lane_FE, HC3)")
    parts.append("")
    parts.append("| param | value |")
    parts.append("|---|---:|")
    parts.append(f"| N | {pooled.n} |")
    parts.append(f"| beta0 | {pooled.beta0:+.5f} |")
    parts.append(f"| beta1 | {pooled.beta1:+.5f} |")
    parts.append(f"| **beta2 (rel_vol^2)** | **{pooled.beta2:+.5f}** |")
    parts.append(f"| SE(beta2) | {pooled.beta2_se:.5f} |")
    parts.append(f"| t(beta2) | {pooled.beta2_t:+.3f} |")
    parts.append(f"| one-tailed p | {pooled.beta2_p_one_tailed:.4f} |")
    parts.append("")
    parts.append("## Per-cell regression")
    parts.append("")
    parts.append(_render_percell(percell))
    parts.append("")
    parts.append(f"- Valid cells (N>=50): {len(valid)}/{len(SESSIONS) * 2}")
    parts.append(f"- Cells with beta2 < 0: {len(neg)}/{len(valid)} = **{agreement:.1%}**")
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No capital action.")
    parts.append("- Does NOT modify the Q4-band MNQ deployment-shape contract.")
    parts.append("- Does NOT test 15m/30m, E3/E4, other RRs, or any MGC filter.")
    parts.append("- If MGC_CONFIRMED_*, unblocks a multi-instrument deployment-shape pre-reg; does not deploy on its own.")
    parts.append("")
    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    ascii_reason = reason.encode("ascii", "replace").decode("ascii")
    print(f"VERDICT: {verdict}")
    print(f"REASON: {ascii_reason}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
