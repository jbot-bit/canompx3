"""MNQ unfiltered-baseline cross-family scan — Pathway A with BH-FDR.

Pre-reg: docs/audit/hypotheses/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.yaml

Broader companion to PRs #49 (30m RR=1.0) and #50 (15m RR=1.0). Covers
MNQ × (5m, 15m, 30m) × (RR=1.0, 1.5, 2.0) × all canonical sessions on
unfiltered E2 CB1 baseline.

K_family ≤ 108. BH-FDR at q=0.05. Chordia t ≥ +3.00 with theory.
Downstream C6/C8/C9 mandatory per pre_registered_criteria.md.

No capital action.
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

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md")
INSTRUMENT = "MNQ"
APERTURES = [5, 15, 30]
RRS = [1.0, 1.5, 2.0]
MIN_N_IS = 100
MIN_N_OOS = 50
BH_FDR_ALPHA = 0.05


@dataclass
class CellResult:
    aperture: int
    rr: float
    session: str
    n_is: int
    net_expr_is: float
    t_is: float
    p_one_tailed_is: float
    q_bh: float
    wfe: float
    n_oos: int
    net_expr_oos: float
    ratio_oos_is: float
    era_min_expr: float
    era_n_fail: int
    h1_t_pass: bool
    h1_fdr_pass: bool
    c6_pass: bool
    c8_pass: bool
    c9_pass: bool
    verdict: str


def _list_sessions(con: duckdb.DuckDBPyConnection, orb_minutes: int) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND entry_model = 'E2'
          AND confirm_bars = 1 AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [INSTRUMENT, orb_minutes],
    ).fetchall()
    return [r[0] for r in rows]


def _load_cell(con: duckdb.DuckDBPyConnection, orb_minutes: int, rr: float, session: str) -> pd.DataFrame:
    sql = """
    SELECT trading_day, pnl_r
    FROM orb_outcomes
    WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
      AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = ?
      AND pnl_r IS NOT NULL
    """
    df = con.execute(sql, [INSTRUMENT, session, orb_minutes, rr]).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def _t_mean(vals: np.ndarray) -> tuple[float, float]:
    if len(vals) < 2 or np.all(vals == vals[0]):
        return float("nan"), float("nan")
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    if se == 0:
        return float("nan"), float("nan")
    t = m / se
    p = float(1.0 - stats.t.cdf(t, len(vals) - 1))
    return float(t), p


def _wfe(is_df: pd.DataFrame, n_folds: int = 3) -> float:
    if len(is_df) < n_folds * 50:
        return float("nan")
    df = is_df.sort_values("trading_day").reset_index(drop=True)
    edges = np.linspace(0, len(df), n_folds + 1, dtype=int)
    is_mean = df["pnl_r"].mean()
    if is_mean == 0 or np.isnan(is_mean):
        return float("nan")
    means = []
    for i in range(1, n_folds):
        slc = df.iloc[edges[i] : edges[i + 1]]
        if len(slc) < 10:
            continue
        means.append(float(slc["pnl_r"].mean()))
    return float(np.mean(means) / is_mean) if means else float("nan")


def _era_stability(is_df: pd.DataFrame) -> tuple[float, int]:
    if is_df.empty:
        return float("nan"), 0
    df = is_df.copy()
    df["year"] = df["trading_day"].dt.year
    agg = df.groupby("year")["pnl_r"].agg(["count", "mean"]).reset_index()
    elig = agg[agg["count"] >= 50]
    if len(elig) == 0:
        return float("nan"), 0
    return float(elig["mean"].min()), int((elig["mean"] < -0.05).sum())


def _bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values (monotonicity enforced)."""
    p_arr = np.array([p if not np.isnan(p) else 1.0 for p in p_values])
    n = len(p_arr)
    if n == 0:
        return []
    order = np.argsort(p_arr)
    ranks = np.arange(1, n + 1)
    q = p_arr[order] * n / ranks
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q_final = np.empty_like(q)
    q_final[order] = q
    return q_final.tolist()


def _test_cell(
    con: duckdb.DuckDBPyConnection,
    orb_minutes: int,
    rr: float,
    session: str,
) -> CellResult | None:
    df = _load_cell(con, orb_minutes, rr, session)
    is_df = df.loc[df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)
    oos_df = df.loc[df["trading_day"] >= pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)
    n_is = len(is_df)
    if n_is < MIN_N_IS:
        return None  # skip; INSUFFICIENT_N

    vals_is = is_df["pnl_r"].astype(float).to_numpy()
    net_is = float(vals_is.mean())
    t_is, p_is = _t_mean(vals_is)

    n_oos = len(oos_df)
    if n_oos > 0:
        net_oos = float(oos_df["pnl_r"].mean())
        ratio = net_oos / net_is if net_is != 0 else float("nan")
    else:
        net_oos, ratio = float("nan"), float("nan")

    wfe = _wfe(is_df)
    era_min, era_nfail = _era_stability(is_df)

    h1_t_pass = (
        net_is > 0
        and not np.isnan(t_is)
        and t_is >= 3.0
        and not np.isnan(p_is)
        and p_is < 0.05
    )
    c6_pass = not np.isnan(wfe) and wfe >= 0.50
    c8_pass = not np.isnan(ratio) and ratio >= 0.40 and n_oos >= MIN_N_OOS
    c9_pass = not np.isnan(era_min) and era_nfail == 0

    return CellResult(
        aperture=orb_minutes,
        rr=rr,
        session=session,
        n_is=n_is,
        net_expr_is=net_is,
        t_is=t_is,
        p_one_tailed_is=p_is,
        q_bh=float("nan"),  # filled after family collected
        wfe=wfe,
        n_oos=n_oos,
        net_expr_oos=net_oos,
        ratio_oos_is=ratio,
        era_min_expr=era_min,
        era_n_fail=era_nfail,
        h1_t_pass=h1_t_pass,
        h1_fdr_pass=False,  # filled after BH
        c6_pass=c6_pass,
        c8_pass=c8_pass,
        c9_pass=c9_pass,
        verdict="TBD",
    )


def _assign_verdicts(cells: list[CellResult]) -> None:
    p_values = [c.p_one_tailed_is for c in cells]
    qs = _bh_fdr(p_values)
    for c, q in zip(cells, qs):
        c.q_bh = q
        c.h1_fdr_pass = c.h1_t_pass and not np.isnan(q) and q < BH_FDR_ALPHA
        if c.n_is < MIN_N_IS:
            c.verdict = "SCAN_ABORT"
        elif not (c.h1_t_pass and c.h1_fdr_pass):
            c.verdict = "KILL_IS"
        elif c.c6_pass and c.c8_pass and c.c9_pass:
            c.verdict = "CANDIDATE_READY"
        else:
            c.verdict = "RESEARCH_SURVIVOR"


def _render_full_table(cells: list[CellResult]) -> str:
    lines = [
        "| Apt | RR | Session | N_IS | Net ExpR | t | q(BH) | WFE | N_OOS | Net OOS | OOS/IS | Worst Yr | Verdict |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def f(x: float, spec: str = "+.4f") -> str:
        return f"{x:{spec}}" if not np.isnan(x) else "-"
    for c in sorted(cells, key=lambda x: (x.aperture, x.rr, -x.net_expr_is if not np.isnan(x.net_expr_is) else 1e9)):
        lines.append(
            f"| {c.aperture} | {c.rr} | {c.session} | {c.n_is} | "
            f"{f(c.net_expr_is)} | {f(c.t_is, '+.3f')} | {f(c.q_bh, '.4f')} | "
            f"{f(c.wfe, '.3f')} | {c.n_oos} | {f(c.net_expr_oos)} | "
            f"{f(c.ratio_oos_is, '+.2f')} | {f(c.era_min_expr)} | **{c.verdict}** |"
        )
    return "\n".join(lines)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cells: list[CellResult] = []
    try:
        for apt in APERTURES:
            sessions = _list_sessions(con, apt)
            for rr in RRS:
                for s in sessions:
                    r = _test_cell(con, apt, rr, s)
                    if r is not None:
                        cells.append(r)
    finally:
        con.close()

    _assign_verdicts(cells)
    # Count by verdict
    cr = [c for c in cells if c.verdict == "CANDIDATE_READY"]
    rs = [c for c in cells if c.verdict == "RESEARCH_SURVIVOR"]
    kill = [c for c in cells if c.verdict == "KILL_IS"]

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# MNQ unfiltered-baseline cross-family — v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`\n")
    parts.append(f"**Script:** `research/mnq_unfiltered_baseline_cross_family_v1.py`\n")
    parts.append(f"**Family K:** {len(cells)} cells (N_IS ≥ {MIN_N_IS})\n")
    parts.append(f"**Gates:** H1 (t≥+3.0, BH-FDR q<0.05), C6 (WFE≥0.50), C8 (OOS/IS≥0.40, N_OOS≥{MIN_N_OOS}), C9 (era stability)\n")
    parts.append("## Summary counts")
    parts.append("")
    parts.append(f"- CANDIDATE_READY (all 4 gates pass): **{len(cr)}**")
    parts.append(f"- RESEARCH_SURVIVOR (H1 pass, ≥1 downstream fails): {len(rs)}")
    parts.append(f"- KILL_IS (H1 fail): {len(kill)}")
    parts.append("")
    parts.append("## CANDIDATE_READY cells")
    parts.append("")
    if cr:
        parts.append(_render_full_table(cr))
    else:
        parts.append("_None._")
    parts.append("")
    parts.append("## RESEARCH_SURVIVOR cells (for reference; not deployable)")
    parts.append("")
    if rs:
        parts.append(_render_full_table(rs))
    else:
        parts.append("_None._")
    parts.append("")
    parts.append("## Full result table (all cells passing N_IS ≥ 100)")
    parts.append("")
    parts.append(_render_full_table(cells))
    parts.append("")
    parts.append("## Comparison to prior PRs")
    parts.append("")
    parts.append("- PR #49 (30m RR=1.0) found 2 RESEARCH_SURVIVORs: US_DATA_1000, NYSE_OPEN.")
    parts.append("- PR #50 (15m RR=1.0) found 2 CANDIDATE_READY: NYSE_OPEN, US_DATA_1000.")
    parts.append("- This scan confirms/extends both and tests RR=1.5/2.0 + 5m overlap.")
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No writes to validated_setups / lane_allocation / edge_families.")
    parts.append("- Does NOT modify Q4-band MNQ contract (PR #43).")
    parts.append("- Does NOT apply monotonic-rank sizing (separate follow-on).")
    parts.append("- Does NOT test MES / MGC / filter overlays — future pre-regs.")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print(f"Family K: {len(cells)}")
    print(f"CANDIDATE_READY: {len(cr)}")
    print(f"RESEARCH_SURVIVOR: {len(rs)}")
    print(f"KILL_IS: {len(kill)}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    if cr:
        print("\nCANDIDATE_READY cells:")
        for c in sorted(cr, key=lambda x: -x.net_expr_is):
            print(
                f"  MNQ {c.aperture}m RR={c.rr} {c.session}: "
                f"N={c.n_is} NetExpR={c.net_expr_is:+.4f} t={c.t_is:+.3f} "
                f"q={c.q_bh:.4f} WFE={c.wfe:.2f} OOS/IS={c.ratio_oos_is:+.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
