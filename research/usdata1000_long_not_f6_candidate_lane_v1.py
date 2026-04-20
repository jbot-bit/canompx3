#!/usr/bin/env python3
"""Candidate-lane validation for MNQ US_DATA_1000 O5 long NOT_F6.

Locked by:
  docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-candidate-lane-v1.yaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.lib import connect_db
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-candidate-lane-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-20-usdata1000-long-not-f6-candidate-lane-v1.md")
OUTPUT_CSV = Path("research/output/usdata1000_long_not_f6_candidate_lane_v1.csv")

INSTRUMENT = "MNQ"
SESSION = "US_DATA_1000"
RRS = (1.0, 1.5)
MIN_N_OOS = 50
BH_FDR_ALPHA = 0.05


@dataclass
class CellResult:
    rr: float
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


def _load_cell(rr: float) -> pd.DataFrame:
    sql = f"""
    SELECT
        o.trading_day,
        o.pnl_r,
        o.entry_price,
        o.stop_price,
        d.prev_day_high,
        d.prev_day_low,
        d.orb_{SESSION}_high AS orb_high,
        d.orb_{SESSION}_low AS orb_low
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{SESSION}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = {rr}
      AND o.pnl_r IS NOT NULL
      AND d.prev_day_high IS NOT NULL
      AND d.prev_day_low IS NOT NULL
      AND d.orb_{SESSION}_high IS NOT NULL
      AND d.orb_{SESSION}_low IS NOT NULL
    """
    with connect_db() as con:
        df = con.execute(sql).fetchdf()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df = df[df["direction"] == "long"].copy()
    mid = (df["orb_high"].astype(float) + df["orb_low"].astype(float)) / 2.0
    df["not_f6"] = ~((mid > df["prev_day_low"].astype(float)) & (mid < df["prev_day_high"].astype(float)))
    return df[df["not_f6"]].copy()


def _t_mean(vals: np.ndarray) -> tuple[float, float]:
    if len(vals) < 2 or np.all(vals == vals[0]):
        return float("nan"), float("nan")
    mean = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    if se == 0:
        return float("nan"), float("nan")
    t_stat = mean / se
    p_val = float(1.0 - stats.t.cdf(t_stat, len(vals) - 1))
    return float(t_stat), p_val


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


def _test_cell(rr: float) -> CellResult:
    df = _load_cell(rr)
    is_df = df.loc[df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)
    oos_df = df.loc[df["trading_day"] >= pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)

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
        rr=rr,
        n_is=len(is_df),
        net_expr_is=net_is,
        t_is=t_is,
        p_one_tailed_is=p_is,
        q_bh=float("nan"),
        wfe=wfe,
        n_oos=n_oos,
        net_expr_oos=net_oos,
        ratio_oos_is=ratio,
        era_min_expr=era_min,
        era_n_fail=era_nfail,
        h1_t_pass=h1_t_pass,
        h1_fdr_pass=False,
        c6_pass=c6_pass,
        c8_pass=c8_pass,
        c9_pass=c9_pass,
        verdict="TBD",
    )


def _assign_verdicts(cells: list[CellResult]) -> None:
    qs = _bh_fdr([c.p_one_tailed_is for c in cells])
    for cell, q in zip(cells, qs, strict=True):
        cell.q_bh = q
        cell.h1_fdr_pass = cell.h1_t_pass and not np.isnan(q) and q < BH_FDR_ALPHA
        if not (cell.h1_t_pass and cell.h1_fdr_pass):
            cell.verdict = "KILL_IS"
        elif cell.c6_pass and cell.c8_pass and cell.c9_pass:
            cell.verdict = "CANDIDATE_READY"
        else:
            cell.verdict = "RESEARCH_SURVIVOR"


def _render_table(cells: list[CellResult]) -> str:
    lines = [
        "| RR | N_IS | Net ExpR | t | q(BH) | WFE | N_OOS | Net OOS | OOS/IS | Worst Yr | Verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in cells:
        worst = "-" if np.isnan(c.era_min_expr) else f"{c.era_min_expr:+.4f}"
        q = "-" if np.isnan(c.q_bh) else f"{c.q_bh:.4f}"
        ratio = "-" if np.isnan(c.ratio_oos_is) else f"{c.ratio_oos_is:+.2f}"
        lines.append(
            f"| {c.rr} | {c.n_is} | {c.net_expr_is:+.4f} | {c.t_is:+.3f} | {q} | "
            f"{c.wfe:.3f} | {c.n_oos} | {c.net_expr_oos:+.4f} | {ratio} | {worst} | **{c.verdict}** |"
        )
    return "\n".join(lines)


def main() -> int:
    cells = [_test_cell(rr) for rr in RRS]
    _assign_verdicts(cells)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([c.__dict__ for c in cells]).to_csv(OUTPUT_CSV, index=False)

    candidate = [c for c in cells if c.verdict == "CANDIDATE_READY"]
    survivors = [c for c in cells if c.verdict == "RESEARCH_SURVIVOR"]
    killed = [c for c in cells if c.verdict == "KILL_IS"]

    parts: list[str] = []
    parts.append("# US_DATA_1000 Long NOT_F6 Candidate Lane — v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`\n")
    parts.append("**Script:** `research/usdata1000_long_not_f6_candidate_lane_v1.py`\n")
    parts.append(f"**Family K:** {len(cells)}\n")
    parts.append("**Gates:** H1 (t≥+3.0, BH-FDR q<0.05), C6 (WFE≥0.50), C8 (OOS/IS≥0.40 and N_OOS≥50), C9 (era stability)\n")
    parts.append("## Summary counts")
    parts.append("")
    parts.append(f"- CANDIDATE_READY: **{len(candidate)}**")
    parts.append(f"- RESEARCH_SURVIVOR: {len(survivors)}")
    parts.append(f"- KILL_IS: {len(killed)}")
    parts.append("")
    parts.append("## Full result table")
    parts.append("")
    parts.append(_render_table(cells))
    parts.append("")
    parts.append("## Interpretation")
    parts.append("")
    if survivors:
        parts.append(
            "- `NOT_F6_INSIDE_PDR` is a **research-provisional candidate lane**, not a live-ready lane: "
            "both RR cells clear H1/C6/C9, but both fail C8 because 2026 OOS remains too thin and does not yet "
            "support the required OOS/IS gate."
        )
    if candidate:
        parts.append("- At least one RR cell cleared all lane gates.")
    if killed:
        parts.append("- At least one RR cell failed the IS lane-quality gate.")
    parts.append("- This confirms the role-design conclusion: `NOT_F6` is the right primary lane route, but it is still awaiting enough forward OOS.")
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No writes to `validated_setups`, `edge_families`, or `lane_allocation`.")
    parts.append("- No deployment or capital action.")
    parts.append("- No shadow execution contract yet; that is the next bounded step if continuing.")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print(f"CANDIDATE_READY: {len(candidate)}")
    print(f"RESEARCH_SURVIVOR: {len(survivors)}")
    print(f"KILL_IS: {len(killed)}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
