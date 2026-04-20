"""Cross-instrument participation-shape test — monotonic-up specification.

Pre-reg: docs/audit/hypotheses/2026-04-20-participation-shape-cross-instrument-v1.yaml
Theory: Fitschen 2013 Ch 3 + Chan 2013 Ch 7 (both verified via local extracts).

Primary H1 (Pathway B K=1 per instrument): pooled OLS pnl_r ~
rank_within_lane(rel_vol) + lane_FE, HC3 SE, β₁ > 0 one-tailed, t ≥ +3.0.

Secondary descriptive shape-map: 3x3x3 = 27 (instrument × aperture × RR)
cells — β₁ (rank) and β₂ (quadratic) per cell. No hypothesis testing on
secondary — descriptive only.

No capital action under any verdict.
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

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-participation-shape-cross-instrument-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-20-participation-shape-cross-instrument-v1.md")
INSTRUMENTS = ["MNQ", "MES", "MGC"]
PRIMARY_APERTURE = 5
PRIMARY_RR = 1.5
APERTURES = [5, 15, 30]
RRS = [1.0, 1.5, 2.0]


@dataclass
class RegResult:
    n: int
    beta: float  # slope coefficient of interest
    t: float
    p_one_tailed: float  # one-tailed, direction specified by caller

    @classmethod
    def nan(cls, n: int = 0) -> "RegResult":
        return cls(n=n, beta=float("nan"), t=float("nan"), p_one_tailed=float("nan"))


def _list_sessions(con: duckdb.DuckDBPyConnection, symbol: str, orb_minutes: int) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [symbol, orb_minutes],
    ).fetchall()
    return [r[0] for r in rows]


def _load_cell(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    session: str,
    orb_minutes: int,
    rr_target: float,
) -> pd.DataFrame:
    rel_col = f"rel_vol_{session}"
    # RULE 9: triple-join orb_outcomes × daily_features on (trading_day, symbol, orb_minutes).
    # CTE reads daily_features with orb_minutes filter to prevent 3x N-inflation.
    sql = f"""
    WITH df AS (
      SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
      FROM daily_features d
      WHERE d.symbol = '{symbol}' AND d.orb_minutes = {orb_minutes}
    )
    SELECT o.trading_day, o.pnl_r, o.entry_price, o.stop_price, df.rel_vol
    FROM orb_outcomes o
    JOIN df ON o.trading_day = df.trading_day AND o.symbol = df.symbol
    WHERE o.symbol = '{symbol}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {orb_minutes}
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = {rr_target}
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


def _load_all(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    orb_minutes: int,
    rr_target: float,
    sessions: list[str],
) -> pd.DataFrame:
    frames = [sub for s in sessions if len(sub := _load_cell(con, symbol, s, orb_minutes, rr_target)) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _is_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)


def _rank_within_lane(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized rank [1/n, n/n] of rel_vol within each lane (session, direction)."""
    df = df.copy()
    df["rank_rel_vol"] = (
        df.groupby("lane")["rel_vol"]
        .rank(method="average")
        .div(df.groupby("lane")["lane"].transform("count"))
    )
    return df


def _regress_rank(df: pd.DataFrame, with_lane_fe: bool) -> RegResult:
    """OLS pnl_r ~ rank_rel_vol (+ lane_FE if multi-lane), HC3 SE. Returns β₁ one-tailed p (β > 0)."""
    sub = df.dropna(subset=["rank_rel_vol", "pnl_r"]).copy()
    if len(sub) < 50:
        return RegResult.nan(len(sub))
    X = sub[["rank_rel_vol"]].astype(float).copy()
    if with_lane_fe and sub["lane"].nunique() > 1:
        lane_d = pd.get_dummies(sub["lane"], drop_first=True, dtype=float)
        X = pd.concat([X, lane_d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = sub["pnl_r"].astype(float)
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return RegResult.nan(len(sub))
    b = float(model.params.get("rank_rel_vol", float("nan")))
    t = float(model.tvalues.get("rank_rel_vol", float("nan")))
    # one-tailed p for β > 0: P(T > t) = 1 - CDF(t)
    if np.isnan(t):
        p = float("nan")
    else:
        p = float(1.0 - sm.stats.stattools.stats.t.cdf(t, model.df_resid))
    return RegResult(n=len(sub), beta=b, t=t, p_one_tailed=p)


def _regress_quadratic(df: pd.DataFrame, with_lane_fe: bool) -> RegResult:
    """OLS pnl_r ~ rel_vol + rel_vol^2 (+ lane_FE), HC3. Returns β₂ (quadratic) and one-tailed p (β₂ < 0)."""
    sub = df.dropna(subset=["rel_vol", "pnl_r"]).copy()
    if len(sub) < 50:
        return RegResult.nan(len(sub))
    sub["rel_vol_sq"] = sub["rel_vol"].astype(float) ** 2
    X = sub[["rel_vol", "rel_vol_sq"]].astype(float).copy()
    if with_lane_fe and sub["lane"].nunique() > 1:
        lane_d = pd.get_dummies(sub["lane"], drop_first=True, dtype=float)
        X = pd.concat([X, lane_d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = sub["pnl_r"].astype(float)
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return RegResult.nan(len(sub))
    b2 = float(model.params.get("rel_vol_sq", float("nan")))
    t2 = float(model.tvalues.get("rel_vol_sq", float("nan")))
    if np.isnan(t2):
        p = float("nan")
    else:
        # one-tailed for β₂ < 0: P(T < t2) = CDF(t2)
        p = float(sm.stats.stattools.stats.t.cdf(t2, model.df_resid))
    return RegResult(n=len(sub), beta=b2, t=t2, p_one_tailed=p)


def _classify_shape(rank_res: RegResult, quad_res: RegResult, t_descriptive: float = 2.0) -> str:
    """Descriptive classification for the secondary shape-map."""
    mono = (
        not np.isnan(rank_res.t)
        and rank_res.beta > 0
        and rank_res.t >= t_descriptive
    )
    inv_u = (
        not np.isnan(quad_res.t)
        and quad_res.beta < 0
        and quad_res.t <= -t_descriptive
    )
    if mono and inv_u:
        return "BOTH"
    if mono:
        return "MONOTONIC_UP"
    if inv_u:
        return "INVERTED_U"
    return "NULL"


@dataclass
class PrimaryResult:
    instrument: str
    n_is_raw: int
    rv_non_null: float
    pooled: RegResult
    percell_count_valid: int
    percell_count_beta_pos: int
    agreement: float
    verdict: str
    reason: str


def _primary_test(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
) -> PrimaryResult:
    sessions = _list_sessions(con, symbol, PRIMARY_APERTURE)
    raw = _load_all(con, symbol, PRIMARY_APERTURE, PRIMARY_RR, sessions)
    is_df = _is_split(raw)
    rv_non_null = float(is_df["rel_vol"].notna().sum() / max(len(is_df), 1)) if len(is_df) else 0.0
    integrity_ok = rv_non_null >= 0.90
    is_df = is_df.dropna(subset=["rel_vol"]).reset_index(drop=True)
    is_df = _rank_within_lane(is_df)
    pooled = _regress_rank(is_df, with_lane_fe=True)

    # Per-cell secondary agreement
    percell: list[RegResult] = []
    for (session, direction), grp in is_df.groupby(["session", "direction"]):
        cell = grp.copy()
        # within-lane rank already computed
        percell.append(_regress_rank(cell, with_lane_fe=False))
    valid = [r for r in percell if not np.isnan(r.beta)]
    pos = [r for r in valid if r.beta > 0]
    agreement = float(len(pos) / max(len(valid), 1))

    if not integrity_ok or pooled.n < 1000:
        verdict = f"{symbol}_SCAN_ABORT"
        reason = (
            f"integrity fail: rel_vol non-null {rv_non_null:.1%} "
            f"(need >= 90%) OR pooled N {pooled.n} < 1000"
        )
    elif np.isnan(pooled.t) or pooled.beta <= 0 or pooled.t < 3.0 or pooled.p_one_tailed >= 0.05:
        verdict = f"{symbol}_NO_MONOTONIC"
        reason = (
            f"Pooled β1={pooled.beta:+.5f} t={pooled.t:+.3f} "
            f"p={pooled.p_one_tailed:.4f} does NOT clear t>=+3.0 or p<0.05."
        )
    elif agreement < 0.25:
        verdict = f"{symbol}_SIMPSON_MONOTONIC"
        reason = (
            f"Pooled significant but per-lane agreement {agreement:.1%} "
            f"< 25% — Simpson artefact."
        )
    elif agreement < 0.50:
        verdict = f"{symbol}_MONOTONIC_HETEROGENEOUS"
        reason = (
            f"Pooled β1={pooled.beta:+.5f} t={pooled.t:+.3f} significant; "
            f"per-lane agreement {agreement:.1%} in 25-50% band — "
            f"heterogeneous replication."
        )
    else:
        verdict = f"{symbol}_MONOTONIC_CONFIRMED"
        reason = (
            f"Pooled β1={pooled.beta:+.5f} t={pooled.t:+.3f} significant; "
            f"per-lane agreement {agreement:.1%} >= 50%. "
            f"Monotonic-up replicates on {symbol}."
        )

    return PrimaryResult(
        instrument=symbol,
        n_is_raw=len(is_df),
        rv_non_null=rv_non_null,
        pooled=pooled,
        percell_count_valid=len(valid),
        percell_count_beta_pos=len(pos),
        agreement=agreement,
        verdict=verdict,
        reason=reason,
    )


def _secondary_shape_map(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Descriptive 27-cell map. No hypothesis testing, no FDR."""
    out = []
    for inst in INSTRUMENTS:
        for apt in APERTURES:
            sessions = _list_sessions(con, inst, apt)
            for rr in RRS:
                raw = _load_all(con, inst, apt, rr, sessions)
                is_df = _is_split(raw).dropna(subset=["rel_vol"]).reset_index(drop=True)
                if len(is_df) < 100:
                    out.append(
                        {
                            "instrument": inst,
                            "aperture": apt,
                            "rr": rr,
                            "n": len(is_df),
                            "rank_beta": float("nan"),
                            "rank_t": float("nan"),
                            "quad_beta": float("nan"),
                            "quad_t": float("nan"),
                            "shape": "INSUFFICIENT_N",
                        }
                    )
                    continue
                is_df = _rank_within_lane(is_df)
                rank_res = _regress_rank(is_df, with_lane_fe=True)
                quad_res = _regress_quadratic(is_df, with_lane_fe=True)
                out.append(
                    {
                        "instrument": inst,
                        "aperture": apt,
                        "rr": rr,
                        "n": int(rank_res.n),
                        "rank_beta": rank_res.beta,
                        "rank_t": rank_res.t,
                        "quad_beta": quad_res.beta,
                        "quad_t": quad_res.t,
                        "shape": _classify_shape(rank_res, quad_res),
                    }
                )
    return out


def _render_primary_table(results: list[PrimaryResult]) -> str:
    lines = [
        "| Instrument | N | β₁ (rank) | t | one-tailed p | per-cell agreement | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if np.isnan(r.pooled.beta):
            lines.append(
                f"| {r.instrument} | {r.pooled.n} | - | - | - | "
                f"{r.percell_count_beta_pos}/{r.percell_count_valid} | {r.verdict} |"
            )
        else:
            lines.append(
                f"| {r.instrument} | {r.pooled.n} | {r.pooled.beta:+.5f} | "
                f"{r.pooled.t:+.3f} | {r.pooled.p_one_tailed:.4f} | "
                f"{r.agreement:.1%} ({r.percell_count_beta_pos}/{r.percell_count_valid}) | "
                f"**{r.verdict}** |"
            )
    return "\n".join(lines)


def _render_shape_map(rows: list[dict]) -> str:
    lines = [
        "| Instrument | Aperture | RR | N | β₁ (rank) | t(rank) | β₂ (quad) | t(quad) | Shape |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["instrument"], x["aperture"], x["rr"])):
        if np.isnan(r["rank_beta"]):
            lines.append(
                f"| {r['instrument']} | {r['aperture']} | {r['rr']} | {r['n']} | "
                f"- | - | - | - | {r['shape']} |"
            )
        else:
            lines.append(
                f"| {r['instrument']} | {r['aperture']} | {r['rr']} | {r['n']} | "
                f"{r['rank_beta']:+.5f} | {r['rank_t']:+.3f} | "
                f"{r['quad_beta']:+.5f} | {r['quad_t']:+.3f} | {r['shape']} |"
            )
    return "\n".join(lines)


def _combined_interpretation(results: list[PrimaryResult]) -> str:
    confirmed = [r for r in results if r.verdict.endswith("_MONOTONIC_CONFIRMED")]
    inst_set = {r.instrument for r in confirmed}
    if len(confirmed) == 3:
        return (
            "3/3 CONFIRMED — monotonic-up is the universal ORB 5m E2 RR1.5 spec. "
            "PR #41 MNQ inverted-U likely a shape-artefact atop a monotonic-up base; "
            "PR #43 Q4-band contract may be under-scoped or mis-shaped."
        )
    if "MGC" in inst_set and len(confirmed) == 2:
        other = [i for i in inst_set if i != "MGC"][0]
        return (
            f"2/3 CONFIRMED (MGC + {other}). Monotonic-up is asset-class-conditional; "
            f"gold has it, {other} has it too. Opens a follow-on for a multi-instrument "
            f"size-filter + rel_vol-conditioner pre-reg on MGC and {other}."
        )
    if inst_set == {"MGC"}:
        return (
            "MGC-only CONFIRMED. Gold-specific monotonic-up mechanism. Future filter/"
            "conditioner pre-reg can test tradeability post-cost on MGC only."
        )
    if len(confirmed) == 0:
        return (
            "0/3 CONFIRMED. Monotonic-up rejected; PR #41 MNQ inverted-U finding is a "
            "real shape distinct from participation-direction. No reframe needed."
        )
    return f"Other combination: {sorted(inst_set)} CONFIRMED. See per-instrument verdicts."


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        primary: list[PrimaryResult] = []
        for inst in INSTRUMENTS:
            primary.append(_primary_test(con, inst))
        shape_map = _secondary_shape_map(con)
    finally:
        con.close()

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# Participation-shape cross-instrument — v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`\n")
    parts.append(f"**Script:** `research/participation_shape_cross_instrument_v1.py`\n")
    parts.append(
        f"**Scope:** MNQ + MES + MGC at 5m E2 CB1 RR1.5 unfiltered (primary); "
        f"3x3x3 shape-map secondary.\n"
    )
    parts.append("")
    parts.append("## Primary verdicts (per-instrument Pathway-B K=1)")
    parts.append("")
    parts.append(_render_primary_table(primary))
    parts.append("")
    for r in primary:
        parts.append(f"- **{r.instrument}:** {r.reason}")
    parts.append("")
    parts.append("## Combined cross-instrument interpretation")
    parts.append("")
    parts.append(_combined_interpretation(primary))
    parts.append("")
    parts.append("## Secondary shape-map (descriptive — not decision-bearing)")
    parts.append("")
    parts.append(
        "Per-cell descriptive classification: `MONOTONIC_UP` if rank β₁ > 0 and "
        "|t| ≥ 2.0; `INVERTED_U` if quad β₂ < 0 and |t| ≥ 2.0; `BOTH` if both; "
        "`NULL` otherwise. `INSUFFICIENT_N` if n < 100."
    )
    parts.append("")
    parts.append(_render_shape_map(shape_map))
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No capital action.")
    parts.append("- Does NOT modify Q4-band MNQ contract (PR #43).")
    parts.append("- Does NOT revise PRs #41/#42/#45 verdicts.")
    parts.append("- Does NOT test filtered universes / size filters / multi-variate specs.")
    parts.append("- Does NOT run OOS validation.")
    parts.append("")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    for r in primary:
        reason_ascii = r.reason.encode("ascii", "replace").decode("ascii")
        print(f"{r.instrument}: VERDICT={r.verdict}")
        print(f"    {reason_ascii}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
