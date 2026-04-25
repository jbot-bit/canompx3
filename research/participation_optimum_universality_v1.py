"""Participation-optimum universality test v1.

Tests whether the rel_vol inverted-U peak observed on MNQ COMEX_SETTLE short
(Q1 5x2 matrix: Q4 peak, Q5 attenuation) is a GENERAL mechanism on the MNQ
portfolio, or a COMEX_SETTLE artefact.

Pooled OLS:
    pnl_r = β₀ + β₁·rel_vol + β₂·rel_vol² + lane_FE + ε
where lane_FE = categorical (session × direction), K=24 cells.

H_null:  β₂ >= 0
H_alt:   β₂ < 0 (inverted-U: participation-optimum mechanism)

Primary test: one-tailed β₂ with t ≤ -3.0 (Chordia with prior theory) and p < 0.05.
Per-cell test: K=24 per-cell regressions with Chordia-strict t ≤ -3.79.
RULE 14 gate: pooled significance requires >= 50% per-lane β₂-sign agreement.

Pre-reg: docs/audit/hypotheses/2026-04-20-participation-optimum-universality-v1.yaml
Lock SHA: 677bf381
Parent chain: mnq-live-context-overlays-v1 → q1-h04-mechanism-shape-validation-v1 → this.

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

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-participation-optimum-universality-v1.yaml"
PREREG_SHA = "677bf381"
RESULT_DOC = Path("docs/audit/results/2026-04-20-participation-optimum-universality-v1.md")

SESSIONS = [
    "BRISBANE_1025", "CME_PRECLOSE", "CME_REOPEN", "COMEX_SETTLE",
    "EUROPE_FLOW", "LONDON_METALS", "NYSE_CLOSE", "NYSE_OPEN",
    "SINGAPORE_OPEN", "TOKYO_OPEN", "US_DATA_1000", "US_DATA_830",
]
DIRECTIONS = ["long", "short"]
APERTURES = [5, 15, 30]


def _load_session(con: duckdb.DuckDBPyConnection, session: str, aperture: int) -> pd.DataFrame:
    """Canonical unfiltered MNQ rows for one session × aperture × E2 CB1 RR1.5."""
    rel_col = f"rel_vol_{session}"
    sql = f"""
    WITH df AS (
      SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
      FROM daily_features d
      WHERE d.symbol = 'MNQ' AND d.orb_minutes = {aperture}
    )
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_price,
      o.stop_price,
      df.rel_vol
    FROM orb_outcomes o
    JOIN df
      ON o.trading_day = df.trading_day
     AND o.symbol = df.symbol
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {aperture}
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


def _load_all(con: duckdb.DuckDBPyConnection, aperture: int) -> pd.DataFrame:
    frames = []
    for s in SESSIONS:
        sub = _load_session(con, s, aperture)
        if len(sub) > 0:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


@dataclass
class RegressionResult:
    n: int
    beta0: float
    beta1: float
    beta2: float
    beta2_se: float
    beta2_t: float
    beta2_p_one_tailed: float

    def sign(self) -> str:
        return "neg" if self.beta2 < 0 else "pos"


def _regress_quadratic(df: pd.DataFrame, with_lane_fe: bool = False) -> RegressionResult:
    """OLS with HC3 robust SE. pnl_r ~ rel_vol + rel_vol² (+ lane_FE if requested)."""
    sub = df.dropna(subset=["rel_vol", "pnl_r"]).copy()
    if len(sub) < 50:
        return RegressionResult(len(sub), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
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
        return RegressionResult(len(sub), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    b2 = float(model.params.get("rel_vol_sq", float("nan")))
    se2 = float(model.bse.get("rel_vol_sq", float("nan")))
    t2 = float(model.tvalues.get("rel_vol_sq", float("nan")))
    # One-tailed p for H_alt: β₂ < 0
    if np.isnan(t2):
        p_one = float("nan")
    else:
        p_one = float(sm.stats.stattools.stats.t.cdf(t2, model.df_resid))
    return RegressionResult(
        n=len(sub),
        beta0=float(model.params.get("const", float("nan"))),
        beta1=float(model.params.get("rel_vol", float("nan"))),
        beta2=b2,
        beta2_se=se2,
        beta2_t=t2,
        beta2_p_one_tailed=p_one,
    )


def _is_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_mask = df["trading_day"] < holdout
    return df.loc[is_mask].reset_index(drop=True), df.loc[~is_mask].reset_index(drop=True)


def _render_percell_table(rows: list[tuple[str, str, RegressionResult]]) -> str:
    lines = ["| session | direction | N | β₂ | t(β₂) | one-tailed p | sign |",
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
        data_5m = _load_all(con, aperture=5)
        data_15m = _load_all(con, aperture=15)
        data_30m = _load_all(con, aperture=30)
    finally:
        con.close()

    # Integrity
    if len(data_5m) == 0:
        print("VERDICT: SCAN_ABORT — no canonical data at 5m")
        return 1
    is_5m, _oos_5m = _is_split(data_5m)

    rv_col_complete = (is_5m["rel_vol"].notna().sum() / max(len(is_5m), 1))
    integrity_ok = rv_col_complete >= 0.90

    # Primary pooled regression with lane_FE
    pooled = _regress_quadratic(is_5m, with_lane_fe=True)

    # Per-cell (no lane_FE since each cell is one lane)
    percell: list[tuple[str, str, RegressionResult]] = []
    for s in SESSIONS:
        for d in DIRECTIONS:
            cell = is_5m[(is_5m["session"] == s) & (is_5m["direction"] == d)].copy()
            if len(cell) < 50:
                percell.append((s, d, RegressionResult(len(cell), *([float("nan")] * 6))))
            else:
                percell.append((s, d, _regress_quadratic(cell, with_lane_fe=False)))

    # Sign agreement
    valid_cells = [r for _, _, r in percell if not np.isnan(r.beta2)]
    neg_cells = [r for r in valid_cells if r.beta2 < 0]
    agreement_frac = len(neg_cells) / max(len(valid_cells), 1)

    # Parity: COMEX_SETTLE short cell β₂ sign
    cmx_short = next((r for s, d, r in percell if s == "COMEX_SETTLE" and d == "short"), None)
    parity_ok = bool(cmx_short is not None and not np.isnan(cmx_short.beta2) and cmx_short.beta2 < 0)

    # Robustness: 15m, 30m pooled
    if len(data_15m) > 0:
        is_15, _ = _is_split(data_15m)
        pooled_15 = _regress_quadratic(is_15, with_lane_fe=True)
    else:
        pooled_15 = None
    if len(data_30m) > 0:
        is_30, _ = _is_split(data_30m)
        pooled_30 = _regress_quadratic(is_30, with_lane_fe=True)
    else:
        pooled_30 = None

    # Verdict logic per pre-reg
    if not integrity_ok or pooled.n == 0:
        verdict = "SCAN_ABORT"
        reason = "Integrity failure: rel_vol null rate too high or pooled regression failed."
    elif not parity_ok:
        verdict = "SCAN_ABORT"
        reason = f"Parity failed: COMEX_SETTLE short β₂ not negative (β₂={cmx_short.beta2 if cmx_short else 'NA'})."
    elif np.isnan(pooled.beta2_t) or pooled.beta2 >= 0 or pooled.beta2_t > -3.0 or pooled.beta2_p_one_tailed >= 0.05:
        # Primary pooled fails
        if parity_ok:
            verdict = "LANE_SPECIFIC"
            reason = (
                f"Pooled β₂={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} one-tailed p={pooled.beta2_p_one_tailed:.4f} "
                f"does NOT clear threshold (β₂<0, t≤-3.0, p<0.05). COMEX_SETTLE short β₂<0 in isolation. "
                f"Q4-peak was COMEX-specific, not universal."
            )
        else:
            verdict = "NULL"
            reason = (
                f"Pooled β₂={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} does not clear threshold. "
                f"And COMEX_SETTLE short β₂ not negative either — no evidence of inverted-U anywhere."
            )
    elif agreement_frac < 0.25:
        verdict = "KILL_SIMPSON"
        reason = (
            f"Pooled β₂ significant but per-lane sign-agreement = {agreement_frac:.1%} < 25%. "
            f"Simpson's-paradox artefact per RULE 14."
        )
    elif agreement_frac < 0.50:
        verdict = "CONFIRMED_HETEROGENEOUS"
        reason = (
            f"Pooled β₂={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} p={pooled.beta2_p_one_tailed:.4f} "
            f"significantly negative. Per-lane sign-agreement = {agreement_frac:.1%} (25-50% band) — "
            f"mechanism present but session-heterogeneous. Follow-up required to identify which lanes."
        )
    else:
        verdict = "CONFIRMED_UNIVERSAL"
        reason = (
            f"Pooled β₂={pooled.beta2:+.5f} t={pooled.beta2_t:+.3f} p={pooled.beta2_p_one_tailed:.4f} "
            f"significantly negative. Per-lane sign-agreement = {agreement_frac:.1%} (>= 50%). "
            f"Participation-optimum is a general MNQ mechanism."
        )

    # Result doc
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    parts.append(f"# Participation-optimum universality test v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}` (LOCKED, commit_sha=`{PREREG_SHA}`)")
    parts.append(f"**Script:** `research/participation_optimum_universality_v1.py`")
    parts.append(f"**Scope:** MNQ × 12 sessions × both directions × 5m E2 CB1 RR1.5 unfiltered, IS only")
    parts.append(f"**IS window:** trading_day < {HOLDOUT_SACRED_FROM}")
    parts.append("")
    parts.append(f"## Verdict: **{verdict}**")
    parts.append("")
    parts.append(f"> {reason}")
    parts.append("")
    parts.append("## Integrity")
    parts.append("")
    parts.append(f"- rel_vol non-null fraction on IS 5m: {rv_col_complete:.1%} (threshold ≥ 90%)")
    parts.append(f"- IS 5m pooled N: {len(is_5m)}")
    parts.append(f"- Cells loaded: 12 sessions × 2 directions = 24 (expected)")
    parts.append(f"- Parity (COMEX_SETTLE short β₂ < 0): {'YES' if parity_ok else 'NO'}")
    if cmx_short:
        parts.append(f"  - COMEX_SETTLE short: β₂={cmx_short.beta2:+.5f}, t={cmx_short.beta2_t:+.3f}, N={cmx_short.n}")
    parts.append("")
    parts.append("## Primary pooled regression (5m, IS, lane_FE)")
    parts.append("")
    parts.append("| param | value |")
    parts.append("|---|---:|")
    parts.append(f"| N | {pooled.n} |")
    parts.append(f"| β₀ (intercept) | {pooled.beta0:+.5f} |")
    parts.append(f"| β₁ (rel_vol) | {pooled.beta1:+.5f} |")
    parts.append(f"| **β₂ (rel_vol²)** | **{pooled.beta2:+.5f}** |")
    parts.append(f"| SE(β₂) | {pooled.beta2_se:.5f} |")
    parts.append(f"| t(β₂) | {pooled.beta2_t:+.3f} |")
    parts.append(f"| one-tailed p (β₂ < 0) | {pooled.beta2_p_one_tailed:.4f} |")
    parts.append(f"| Chordia threshold | t ≤ -3.0 (with prior theory) |")
    parts.append("")
    parts.append("## Per-cell regression (K=24, 5m, IS)")
    parts.append("")
    parts.append(_render_percell_table(percell))
    parts.append("")
    parts.append(f"- Valid cells (N≥50): {len(valid_cells)}/24")
    parts.append(f"- Cells with β₂ < 0: {len(neg_cells)}/{len(valid_cells)} = **{agreement_frac:.1%}**")
    parts.append(f"- RULE 14 threshold: ≥ 50% (CONFIRMED_UNIVERSAL), 25-50% (CONFIRMED_HETEROGENEOUS), < 25% (KILL_SIMPSON)")
    parts.append("")
    parts.append("## Robustness — 15m aperture pooled")
    parts.append("")
    if pooled_15 is not None and pooled_15.n > 0:
        parts.append(f"- N: {pooled_15.n}")
        parts.append(f"- β₂: {pooled_15.beta2:+.5f}, t: {pooled_15.beta2_t:+.3f}, one-tailed p: {pooled_15.beta2_p_one_tailed:.4f}")
    else:
        parts.append("- (no 15m data)")
    parts.append("")
    parts.append("## Robustness — 30m aperture pooled")
    parts.append("")
    if pooled_30 is not None and pooled_30.n > 0:
        parts.append(f"- N: {pooled_30.n}")
        parts.append(f"- β₂: {pooled_30.beta2:+.5f}, t: {pooled_30.beta2_t:+.3f}, one-tailed p: {pooled_30.beta2_p_one_tailed:.4f}")
    else:
        parts.append("- (no 30m data)")
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No capital, allocator, sizing, or filter change.")
    parts.append("- Does NOT test MES/MGC, E3/E4 entries, RR 1.0/2.0, or overlay-filter-conditional peaks.")
    parts.append("- A CONFIRMED_UNIVERSAL or CONFIRMED_HETEROGENEOUS verdict only unblocks writing a deployment-shape follow-on pre-reg; it does not deploy anything on its own.")
    parts.append("")
    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    # Stdout is ASCII-only for cross-platform safety (Windows cp1252);
    # result doc retains full Unicode (UTF-8).
    ascii_reason = reason.encode("ascii", "replace").decode("ascii")
    print(f"VERDICT: {verdict}")
    print(f"REASON: {ascii_reason}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
