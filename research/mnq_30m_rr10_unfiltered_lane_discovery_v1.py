"""MNQ 30m E2 CB1 RR=1.0 unfiltered lane discovery — Pathway B K=1x4.

Pre-reg: docs/audit/hypotheses/2026-04-20-mnq-30m-rr10-unfiltered-lane-discovery-v1.yaml

For each of 4 pre-committed sessions (US_DATA_1000, NYSE_OPEN, COMEX_SETTLE,
NYSE_CLOSE), test the hypothesis that MNQ 30m E2 CB1 RR=1.0 unfiltered
has a COST-SURVIVING edge meeting Chordia t ≥ +3.0 and downstream Phase 0
discovery gates (C6 WFE, C8 OOS, C9 era stability).

No capital action, no validated_setups write.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# Per pipeline/cost_model.py line 464, canonical `pnl_r` is computed as
#   R = (pnl_points * point_value - total_friction) / risk_in_dollars
# i.e. pnl_r is ALREADY net-of-cost. Using it directly avoids double-deduction.

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-mnq-30m-rr10-unfiltered-lane-discovery-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-20-mnq-30m-rr10-unfiltered-lane-discovery-v1.md")

INSTRUMENT = "MNQ"
APERTURE = 30
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.0
SESSIONS = ["US_DATA_1000", "NYSE_OPEN", "COMEX_SETTLE", "NYSE_CLOSE"]


@dataclass
class CellResult:
    session: str
    n_is: int
    raw_expr_is: float
    net_expr_is: float
    t_is: float
    p_one_tailed_is: float
    wfe: float  # C6
    n_oos: int
    raw_expr_oos: float
    net_expr_oos: float
    ratio_oos_is: float  # C8: net_oos / net_is (IS effect = net_expr_is)
    era_min_expr: float  # worst-year ExpR on years with N>=50
    era_n_eligible_years: int
    era_n_fail_years: int  # years with net ExpR < -0.05 on N>=50
    c3_bh_q: float  # one-tailed p under BH-FDR at K=4
    h1_pass: bool
    c6_pass: bool
    c8_pass: bool
    c9_pass: bool
    verdict: str


def _load_cell(con: duckdb.DuckDBPyConnection, session: str) -> pd.DataFrame:
    sql = """
    SELECT o.trading_day, o.pnl_r, o.risk_dollars
    FROM orb_outcomes o
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = ?
      AND o.confirm_bars = ?
      AND o.rr_target = ?
      AND o.pnl_r IS NOT NULL
    """
    df = con.execute(
        sql,
        [INSTRUMENT, session, APERTURE, ENTRY_MODEL, CONFIRM_BARS, RR_TARGET],
    ).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    # pnl_r is canonical net-of-cost; use directly as net.
    df["net_pnl_r"] = df["pnl_r"].astype(float)
    return df


def _is_oos_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = df.loc[df["trading_day"] < holdout].reset_index(drop=True)
    oos_df = df.loc[df["trading_day"] >= holdout].reset_index(drop=True)
    return is_df, oos_df


def _t_test_on_mean(values: np.ndarray) -> tuple[float, float]:
    """One-sample t-test that mean > 0; returns (t, one-tailed-p)."""
    if len(values) < 2 or np.all(values == values[0]):
        return float("nan"), float("nan")
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    if se == 0 or np.isnan(se):
        return float("nan"), float("nan")
    t = mean / se
    p_one = float(1.0 - stats.t.cdf(t, len(values) - 1))
    return float(t), p_one


def _walk_forward_efficiency(is_df: pd.DataFrame, n_folds: int = 3) -> float:
    """C6: WFE = mean-of-OOS-folds ExpR / IS-mean ExpR.

    Expanding walk-forward: split IS chronologically into n_folds. Each fold
    uses prior folds as 'IS slice' and next fold as 'OOS slice'. Return ratio
    of avg OOS-slice ExpR to full-IS ExpR. WFE ≥ 0.50 is the pre-registered
    threshold.
    """
    if len(is_df) < n_folds * 50 or n_folds < 2:
        return float("nan")
    is_df = is_df.sort_values("trading_day").reset_index(drop=True)
    fold_edges = np.linspace(0, len(is_df), n_folds + 1, dtype=int)
    is_mean = is_df["net_pnl_r"].mean()
    if is_mean == 0 or np.isnan(is_mean):
        return float("nan")
    oos_slice_means: list[float] = []
    for i in range(1, n_folds):
        oos_slice = is_df.iloc[fold_edges[i] : fold_edges[i + 1]]
        if len(oos_slice) < 10:
            continue
        oos_slice_means.append(float(oos_slice["net_pnl_r"].mean()))
    if not oos_slice_means:
        return float("nan")
    return float(np.mean(oos_slice_means) / is_mean)


def _era_stability(is_df: pd.DataFrame) -> tuple[float, int, int]:
    """C9: per-calendar-year net ExpR on years with N>=50. Returns
    (worst-year-ExpR, n_eligible_years, n_fail_years) where fail means ExpR < -0.05."""
    if is_df.empty:
        return float("nan"), 0, 0
    is_df = is_df.copy()
    is_df["year"] = is_df["trading_day"].dt.year
    rows = is_df.groupby("year")["net_pnl_r"].agg(["count", "mean"]).reset_index()
    eligible = rows[rows["count"] >= 50]
    if len(eligible) == 0:
        return float("nan"), 0, 0
    worst = float(eligible["mean"].min())
    n_fail = int((eligible["mean"] < -0.05).sum())
    return worst, int(len(eligible)), n_fail


def _test_cell(con: duckdb.DuckDBPyConnection, session: str) -> CellResult:
    df = _load_cell(con, session)
    is_df, oos_df = _is_oos_split(df)
    n_is = len(is_df)
    if n_is < 100:
        return CellResult(
            session=session,
            n_is=n_is,
            raw_expr_is=float("nan"),
            net_expr_is=float("nan"),
            t_is=float("nan"),
            p_one_tailed_is=float("nan"),
            wfe=float("nan"),
            n_oos=0,
            raw_expr_oos=float("nan"),
            net_expr_oos=float("nan"),
            ratio_oos_is=float("nan"),
            era_min_expr=float("nan"),
            era_n_eligible_years=0,
            era_n_fail_years=0,
            c3_bh_q=float("nan"),
            h1_pass=False,
            c6_pass=False,
            c8_pass=False,
            c9_pass=False,
            verdict=f"{session}_SCAN_ABORT",
        )

    # IS stats
    net_is = is_df["net_pnl_r"].dropna().to_numpy(dtype=np.float64)
    raw_is = is_df["pnl_r"].to_numpy(dtype=np.float64)
    t_is, p_one_is = _t_test_on_mean(net_is)
    raw_expr = float(raw_is.mean())
    net_expr = float(net_is.mean()) if len(net_is) else float("nan")

    h1_pass = (
        net_expr > 0
        and not np.isnan(t_is)
        and t_is >= 3.0
        and not np.isnan(p_one_is)
        and p_one_is < 0.05
    )

    # C6 WFE
    wfe = _walk_forward_efficiency(is_df)
    c6_pass = not np.isnan(wfe) and wfe >= 0.50

    # C8 OOS
    n_oos = len(oos_df)
    if n_oos > 0:
        raw_expr_oos = float(oos_df["pnl_r"].mean())
        net_oos_arr = oos_df["net_pnl_r"].dropna().to_numpy(dtype=np.float64)
        net_expr_oos = float(net_oos_arr.mean()) if len(net_oos_arr) else float("nan")
        if net_expr != 0 and not np.isnan(net_expr) and not np.isnan(net_expr_oos):
            ratio = net_expr_oos / net_expr
        else:
            ratio = float("nan")
    else:
        raw_expr_oos = float("nan")
        net_expr_oos = float("nan")
        ratio = float("nan")
    c8_pass = not np.isnan(ratio) and ratio >= 0.40 and n_oos >= 50

    # C9 Era stability
    era_min, era_n_elig, era_n_fail = _era_stability(is_df)
    c9_pass = era_n_elig >= 2 and era_n_fail == 0

    # Verdict
    if h1_pass and c6_pass and c8_pass and c9_pass:
        verdict = f"{session}_CANDIDATE_READY"
    elif h1_pass:
        verdict = f"{session}_RESEARCH_SURVIVOR"
    else:
        verdict = f"{session}_KILL_IS"

    return CellResult(
        session=session,
        n_is=n_is,
        raw_expr_is=raw_expr,
        net_expr_is=net_expr,
        t_is=t_is,
        p_one_tailed_is=p_one_is,
        wfe=wfe,
        n_oos=n_oos,
        raw_expr_oos=raw_expr_oos,
        net_expr_oos=net_expr_oos,
        ratio_oos_is=ratio,
        era_min_expr=era_min,
        era_n_eligible_years=era_n_elig,
        era_n_fail_years=era_n_fail,
        c3_bh_q=float("nan"),  # filled below
        h1_pass=h1_pass,
        c6_pass=c6_pass,
        c8_pass=c8_pass,
        c9_pass=c9_pass,
        verdict=verdict,
    )


def _bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR. Returns q-values."""
    import numpy as np
    p_arr = np.array([p if not np.isnan(p) else 1.0 for p in p_values])
    n = len(p_arr)
    order = np.argsort(p_arr)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p_arr * n / ranks
    # Enforce monotonicity
    q_sorted = q[order]
    for i in range(len(q_sorted) - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_final = np.empty_like(q_sorted)
    q_final[order] = q_sorted
    return q_final.tolist()


def _render_table(results: list[CellResult]) -> str:
    lines = [
        "| Session | N_IS | Raw ExpR IS | Net ExpR IS | t | p(1t) | q(BH) | WFE | N_OOS | Net ExpR OOS | OOS/IS | Worst Era | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        def f(x: float, spec: str = "+.4f") -> str:
            return f"{x:{spec}}" if not np.isnan(x) else "-"
        lines.append(
            f"| {r.session} | {r.n_is} | {f(r.raw_expr_is)} | {f(r.net_expr_is)} | "
            f"{f(r.t_is, '+.3f')} | {f(r.p_one_tailed_is, '.4f')} | "
            f"{f(r.c3_bh_q, '.4f')} | {f(r.wfe, '.3f')} | {r.n_oos} | "
            f"{f(r.net_expr_oos)} | {f(r.ratio_oos_is, '+.2f')} | "
            f"{f(r.era_min_expr)} | **{r.verdict}** |"
        )
    return "\n".join(lines)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        results: list[CellResult] = []
        for session in SESSIONS:
            results.append(_test_cell(con, session))
    finally:
        con.close()

    # Apply BH-FDR on IS p-values across 4 cells
    p_list = [r.p_one_tailed_is for r in results]
    q_list = _bh_fdr(p_list)
    for r, q in zip(results, q_list):
        r.c3_bh_q = q

    # Produce results MD
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# MNQ 30m E2 CB1 RR=1.0 unfiltered lane discovery — v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`\n")
    parts.append(f"**Script:** `research/mnq_30m_rr10_unfiltered_lane_discovery_v1.py`\n")
    parts.append(f"**Cost model:** MNQ $2.74 RT (from `pipeline.cost_model.COST_SPECS`)\n")
    parts.append("## Per-cell verdicts")
    parts.append("")
    parts.append(_render_table(results))
    parts.append("")
    parts.append("## Gate summary")
    parts.append("")
    parts.append("H1 = Net ExpR > 0 AND HC3 t ≥ +3.0 AND p < 0.05 (Chordia with-theory).")
    parts.append("C6 (WFE ≥ 0.50), C8 (Net_OOS / Net_IS ≥ 0.40, N_OOS ≥ 50), C9 (no era N≥50 with ExpR < -0.05).")
    parts.append("BH-FDR q-value reported at K=4 (pre-reg K framing).")
    parts.append("")
    candidate = [r for r in results if r.verdict.endswith("_CANDIDATE_READY")]
    survivor = [r for r in results if r.verdict.endswith("_RESEARCH_SURVIVOR")]
    kill = [r for r in results if r.verdict.endswith("_KILL_IS")]
    abort = [r for r in results if r.verdict.endswith("_SCAN_ABORT")]
    parts.append(f"- CANDIDATE_READY ({len(candidate)}): {[r.session for r in candidate]}")
    parts.append(f"- RESEARCH_SURVIVOR ({len(survivor)}): {[r.session for r in survivor]}")
    parts.append(f"- KILL_IS ({len(kill)}): {[r.session for r in kill]}")
    parts.append(f"- SCAN_ABORT ({len(abort)}): {[r.session for r in abort]}")
    parts.append("")
    parts.append("## Follow-on actions (NOT taken by this pre-reg)")
    parts.append("")
    parts.append("- `CANDIDATE_READY` cells → Phase 0 validated_setups promotion flow (C5 DSR / C11 account-death MC / C12 SR-monitor). Requires its own pre-reg and runner.")
    parts.append("- `RESEARCH_SURVIVOR` cells → document deploy-gate failure mode for future research; do NOT deploy.")
    parts.append("- `KILL_IS` cells → closed under this pre-reg.")
    parts.append("- Monotonic-rank sizing overlay on top of any passing cell → SEPARATE pre-reg after candidate promotion.")
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No write to validated_setups / edge_families / lane_allocation / live_config.")
    parts.append("- Does NOT modify Q4-band MNQ contract (PR #43).")
    parts.append("- Does NOT test other RRs, apertures, or filtered overlays.")
    parts.append("- Does NOT apply monotonic-rank sizing (separate follow-on).")
    parts.append("")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    # Console output
    for r in results:
        reason_parts = [
            f"Net={r.net_expr_is:+.4f}" if not np.isnan(r.net_expr_is) else "Net=-",
            f"t={r.t_is:+.3f}" if not np.isnan(r.t_is) else "t=-",
            f"p={r.p_one_tailed_is:.4f}" if not np.isnan(r.p_one_tailed_is) else "p=-",
            f"WFE={r.wfe:.2f}" if not np.isnan(r.wfe) else "WFE=-",
            f"OOS/IS={r.ratio_oos_is:+.2f}" if not np.isnan(r.ratio_oos_is) else "OOS=-",
            f"worst_yr={r.era_min_expr:+.3f}" if not np.isnan(r.era_min_expr) else "yr=-",
        ]
        print(f"{r.verdict}: " + " | ".join(reason_parts))
    print(f"RESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
