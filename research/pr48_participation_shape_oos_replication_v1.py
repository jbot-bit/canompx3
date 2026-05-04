"""PR #48 participation-shape — OOS β₁ replication (Pathway B K=3).

PR #48's own result doc line 66 states "Does NOT run OOS validation." This
script closes that gap. Same axes, same regression, same data pipeline as
`research/participation_shape_cross_instrument_v1.py` — only difference is
IS_split → OOS_split (trading_day >= HOLDOUT_SACRED_FROM).

Confirmatory replication per Pathway B K=1 per instrument (K_family=3
independent instrument-level hypotheses). Pass criterion (pre-committed):
- sign(β₁_OOS) == sign(β₁_IS) AND t_OOS >= +2.0 per instrument.

No new pre-reg required (Phase 0 § research-truth-protocol.md § 10: confirmatory
audits on prior survivors are exempt). Result is read-only; no capital action.

Grounding:
- PR #48 IS result doc: `docs/audit/results/2026-04-20-participation-shape-cross-instrument-v1.md`
- PR #48 pre-reg: `docs/audit/hypotheses/2026-04-20-participation-shape-cross-instrument-v1.yaml`
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR not
  directly applicable to OLS β, but sign/power stability under OOS is the
  institutional out-of-sample complement.

No capital action.
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

RESULT_DOC = Path("docs/audit/results/2026-04-21-pr48-participation-shape-oos-replication-v1.md")
INSTRUMENTS = ["MNQ", "MES", "MGC"]
APERTURE = 5
RR = 1.5
IS_PASS_T = 2.0  # per pre-committed Pathway B K=1 criterion

# PR #48 reported IS β₁ per instrument (from result doc table)
IS_BETAS = {"MNQ": +0.27775, "MES": +0.33025, "MGC": +0.29975}


@dataclass
class RegResult:
    n: int
    beta: float
    t: float
    p_one_tailed: float


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
    """Replica of participation_shape_cross_instrument_v1._load_cell — exact axes."""
    rel_col = f"rel_vol_{session}"
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


def _oos_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df["trading_day"] >= pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)


def _rank_within_lane(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_rel_vol"] = (
        df.groupby("lane")["rel_vol"].rank(method="average").div(df.groupby("lane")["lane"].transform("count"))
    )
    return df


def _regress_rank(df: pd.DataFrame) -> RegResult:
    sub = df.dropna(subset=["rank_rel_vol", "pnl_r"]).copy()
    if len(sub) < 50:
        return RegResult(n=len(sub), beta=float("nan"), t=float("nan"), p_one_tailed=float("nan"))
    X = sub[["rank_rel_vol"]].astype(float).copy()
    if sub["lane"].nunique() > 1:
        lane_d = pd.get_dummies(sub["lane"], drop_first=True, dtype=float)
        X = pd.concat([X, lane_d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = sub["pnl_r"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    b = float(model.params.get("rank_rel_vol", float("nan")))
    t = float(model.tvalues.get("rank_rel_vol", float("nan")))
    if np.isnan(t):
        p = float("nan")
    else:
        p = float(1.0 - sm.stats.stattools.stats.t.cdf(t, model.df_resid))
    return RegResult(n=len(sub), beta=b, t=t, p_one_tailed=p)


def _per_year_stability(df: pd.DataFrame) -> list[tuple[int, int, float, float]]:
    """β₁ per-year on the OOS split (post-2026-01-01, so usually only 2026)."""
    rows: list[tuple[int, int, float, float]] = []
    df = df.copy()
    df["year"] = df["trading_day"].dt.year
    for year, g in df.groupby("year"):
        if len(g) < 50:
            continue
        r = _regress_rank(g)
        rows.append((int(year), r.n, r.beta, r.t))
    return rows


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: dict[str, dict] = {}
    try:
        for inst in INSTRUMENTS:
            sessions = _list_sessions(con, inst, APERTURE)
            raw = _load_all(con, inst, APERTURE, RR, sessions)
            oos = _oos_split(raw)
            oos_ranked = _rank_within_lane(oos)
            r = _regress_rank(oos_ranked)
            per_year = _per_year_stability(oos_ranked)
            results[inst] = {
                "n_oos": r.n,
                "beta_oos": r.beta,
                "t_oos": r.t,
                "p_oos": r.p_one_tailed,
                "per_year": per_year,
                "beta_is": IS_BETAS[inst],
            }
    finally:
        con.close()

    # Pass / fail per instrument
    per_inst_verdict: dict[str, str] = {}
    for inst, r in results.items():
        sign_match = (not np.isnan(r["beta_oos"])) and np.sign(r["beta_oos"]) == np.sign(r["beta_is"])
        t_ok = (not np.isnan(r["t_oos"])) and r["t_oos"] >= IS_PASS_T
        if sign_match and t_ok:
            per_inst_verdict[inst] = "OOS_CONFIRMED"
        elif sign_match:
            per_inst_verdict[inst] = "OOS_WEAK_BUT_RIGHT_SIGN"
        elif (not np.isnan(r["beta_oos"])) and np.sign(r["beta_oos"]) != np.sign(r["beta_is"]):
            per_inst_verdict[inst] = "OOS_SIGN_FLIP"
        else:
            per_inst_verdict[inst] = "OOS_NULL"

    # Render result
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# PR #48 participation-shape — OOS β₁ replication v1\n")
    parts.append(
        "**Replication of:** `docs/audit/results/2026-04-20-participation-shape-cross-instrument-v1.md` — same axes, OOS window only.\n"
    )
    parts.append("**Pre-commit:** Pathway B K=1 per instrument, K_family=3 independent hypotheses.\n")
    parts.append(f"**Pass criterion:** sign(β₁_OOS) == sign(β₁_IS) AND t_OOS >= +{IS_PASS_T}.\n")
    parts.append("**Confirmatory audit** (no new pre-reg per research-truth-protocol.md § 10).\n")
    parts.append("## Headline")
    parts.append("")
    parts.append(
        "| Instrument | N_OOS | β₁_OOS | t_OOS | one-tailed p | β₁_IS (PR #48) | Sign match? | t>=+2.0? | Verdict |"
    )
    parts.append("|---|---:|---:|---:|---:|---:|:---:|:---:|---|")
    for inst in INSTRUMENTS:
        r = results[inst]
        sign_match = (not np.isnan(r["beta_oos"])) and np.sign(r["beta_oos"]) == np.sign(r["beta_is"])
        t_ok = (not np.isnan(r["t_oos"])) and r["t_oos"] >= IS_PASS_T
        v = per_inst_verdict[inst]
        beta_str = f"{r['beta_oos']:+.5f}" if not np.isnan(r["beta_oos"]) else "-"
        t_str = f"{r['t_oos']:+.3f}" if not np.isnan(r["t_oos"]) else "-"
        p_str = f"{r['p_oos']:.4f}" if not np.isnan(r["p_oos"]) else "-"
        sign_str = "✔" if sign_match else "✘"
        t_str_ok = "✔" if t_ok else "✘"
        parts.append(
            f"| {inst} | {r['n_oos']} | {beta_str} | {t_str} | {p_str} | "
            f"{r['beta_is']:+.5f} | {sign_str} | {t_str_ok} | **{v}** |"
        )
    parts.append("")

    parts.append("## Per-year OOS (where N>=50)")
    parts.append("")
    parts.append("| Instrument | Year | N | β₁ | t |")
    parts.append("|---|---:|---:|---:|---:|")
    for inst in INSTRUMENTS:
        for year, n, beta, t in results[inst]["per_year"]:
            beta_s = f"{beta:+.5f}" if not np.isnan(beta) else "-"
            t_s = f"{t:+.3f}" if not np.isnan(t) else "-"
            parts.append(f"| {inst} | {year} | {n} | {beta_s} | {t_s} |")
    parts.append("")

    # Summary interpretation
    confirmed = [i for i, v in per_inst_verdict.items() if v == "OOS_CONFIRMED"]
    right_sign = [i for i, v in per_inst_verdict.items() if v in ("OOS_CONFIRMED", "OOS_WEAK_BUT_RIGHT_SIGN")]
    flipped = [i for i, v in per_inst_verdict.items() if v == "OOS_SIGN_FLIP"]
    parts.append("## Summary + interpretation")
    parts.append("")
    parts.append(
        f"- OOS_CONFIRMED (sign-match AND t>=+{IS_PASS_T}): **{len(confirmed)} of 3** — {', '.join(confirmed) if confirmed else 'none'}"
    )
    parts.append(
        f"- Right-sign-but-weak (sign-match, t<+{IS_PASS_T}): {', '.join([i for i in right_sign if i not in confirmed]) or 'none'}"
    )
    parts.append(f"- Sign-flipped: {', '.join(flipped) or 'none'}")
    parts.append("")
    if len(confirmed) == 3:
        parts.append(
            "**Verdict:** PR #48 participation-shape monotonic-up is **OOS-CONFIRMED UNIVERSALLY**. All three instruments replicate the IS finding on the 2026-01-01-onwards holdout at t>=+2.0. The platform is genuine, not a backtest artefact. Next step: derive a concrete sizer rule (e.g., quintile-based position multiplier) and forward-shadow."
        )
    elif len(confirmed) >= 1:
        parts.append(
            f"**Verdict:** PR #48 participation-shape is **OOS-CONFIRMED on {', '.join(confirmed)}**. "
            f"The {3 - len(confirmed)} non-confirming instrument(s) must be treated as UNVERIFIED "
            "(right-sign or weak) or DEAD (sign-flipped). Cross-instrument universality is weakened; "
            "deploy-as-sizer restricted to confirmed instruments only."
        )
    elif len(right_sign) >= 1:
        parts.append(
            "**Verdict:** No instrument clears the pre-committed sign+t gate. "
            f"{len(right_sign)} show right-sign but insufficient power. PR #48 IS finding survives "
            "as a descriptive pattern but NOT as a deploy candidate — OOS data are either too thin "
            "or the effect has faded. Re-run in 6-12 months with more OOS."
        )
    else:
        parts.append(
            "**Verdict:** PR #48 participation-shape is **OOS-FALSIFIED**. "
            "All three instruments fail the sign+t gate. The IS universality was an in-sample "
            "artefact; treat PR #48 as DEAD for deployment. Move top-priority to Path 2 "
            "(conditioner/confluence infrastructure) or Path 3 (Pathway B K=1 rewrite of PR #51 cells)."
        )
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No deployment or capital action.")
    parts.append("- No writes to validated_setups / lane_allocation / edge_families / live_config.")
    parts.append("- Does NOT derive a concrete sizer rule (next step if OOS confirms).")
    parts.append("- Does NOT re-test filtered universes (PR #48 is unfiltered-only).")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print("PR #48 OOS beta replication (K=3 Pathway B)")
    print(f"  IS beta: MNQ {IS_BETAS['MNQ']:+.5f}  MES {IS_BETAS['MES']:+.5f}  MGC {IS_BETAS['MGC']:+.5f}")
    print()
    for inst in INSTRUMENTS:
        r = results[inst]
        print(
            f"  {inst}: N_OOS={r['n_oos']}  beta_OOS={r['beta_oos']:+.5f}  "
            f"t_OOS={r['t_oos']:+.3f}  p={r['p_oos']:.4f}  -> {per_inst_verdict[inst]}"
        )
    print(f"\nRESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
