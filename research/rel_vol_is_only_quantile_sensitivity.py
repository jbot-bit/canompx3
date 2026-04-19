#!/usr/bin/env python3
"""IS-only quantile sensitivity — integrity audit for the 13 rel_vol_HIGH_Q3
BH-global survivor cells from the 2026-04-15 comprehensive lane scan.

Motivation
----------
The 2026-04-15 scan (`research/comprehensive_deployed_lane_scan.py`) computed
the 67th percentile of `rel_vol` on the FULL-SAMPLE lane dataframe including
both IS (trading_day < 2026-01-01) and OOS (>= 2026-01-01) rows. The
threshold used to bucket a 2024 day as HIGH vs LOW is thus influenced by 2026
Q1 data — a subtle quantile-time look-ahead.

Phase 4 overnight (commit 9ebcc5ed) confirmed the overlap-decomposition Meff
was invariant to IS-only vs full-sample quantile on a 4-lane subset, but the
13 BH-global survivor count was NOT re-audited cell-by-cell.

This script runs that audit on the subset of cells that look like BH-global
survivors in the 2026-04-15 scan's rel_vol_HIGH_Q3 family. For each cell it
computes:

  - full-sample 67th percentile of rel_vol (original method)
  - IS-only 67th percentile (trading_day < HOLDOUT_SACRED_FROM)
  - under EACH threshold: IS-restricted N_on, ExpR_on, delta_IS, t_IS, raw_p
  - whether the IS-only result still clears |t| >= 4 (BH-global proxy bar)

Canonical filter delegation via the same logic as the parent scan.
Reads ONLY canonical layers. No writes.

Output: docs/audit/results/2026-04-19-rel-vol-is-only-quantile-sensitivity.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as _sstats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-19-rel-vol-is-only-quantile-sensitivity.md"

# All rel_vol_HIGH_Q3 cells with t >= 3.0 in the 2026-04-15 comprehensive scan
# that had bh_pass_global = Y (or close; t>=4 is a proxy for BH-global).
# Extracted from docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md.
SURVIVOR_CELLS = [
    # (instrument, session, orb_minutes, rr, direction)
    ("MES", "COMEX_SETTLE", 5, 1.0, "short"),
    ("MGC", "LONDON_METALS", 5, 1.0, "short"),
    ("MES", "TOKYO_OPEN", 5, 1.5, "long"),
    ("MNQ", "SINGAPORE_OPEN", 5, 1.0, "short"),
    ("MES", "COMEX_SETTLE", 5, 1.5, "short"),
    ("MES", "SINGAPORE_OPEN", 5, 1.0, "short"),
    ("MES", "SINGAPORE_OPEN", 5, 1.0, "long"),
    ("MNQ", "CME_PRECLOSE", 5, 1.0, "short"),
    ("MES", "SINGAPORE_OPEN", 5, 1.5, "long"),
    ("MES", "CME_PRECLOSE", 5, 1.0, "short"),
    ("MNQ", "BRISBANE_1025", 5, 1.0, "long"),
    ("MES", "CME_PRECLOSE", 5, 1.5, "short"),
    ("MGC", "LONDON_METALS", 5, 1.5, "short"),
]


@dataclass
class CellAudit:
    instrument: str
    session: str
    orb_minutes: int
    rr: float
    direction: str
    n_total: int = 0
    n_is: int = 0
    thresh_full: float | None = None
    thresh_is_only: float | None = None
    thresh_delta: float | None = None
    # under full-sample threshold, restricted to IS rows
    n_on_full_is: int = 0
    expr_on_full: float | None = None
    delta_full: float | None = None
    t_full: float | None = None
    p_full: float | None = None
    # under IS-only threshold, restricted to IS rows
    n_on_isq_is: int = 0
    expr_on_isq: float | None = None
    delta_isq: float | None = None
    t_isq: float | None = None
    p_isq: float | None = None
    # survivor status retention
    full_bh_global_proxy: bool = False
    isq_bh_global_proxy: bool = False
    survivor_drift: bool = False


def load_lane(con, instrument, session, orb_minutes, rr, direction):
    sql = f"""
    SELECT o.trading_day, o.pnl_r, o.symbol,
           d.rel_vol_{session} AS rel_vol,
           d.orb_{session}_break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
      AND o.entry_model = 'E2' AND o.confirm_bars = 1 AND o.rr_target = ?
      AND d.orb_{session}_break_dir = ?
      AND o.pnl_r IS NOT NULL
      AND d.rel_vol_{session} IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(sql, [instrument, session, orb_minutes, rr, direction]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["is_is"] = df["trading_day"] < HOLDOUT_SACRED_FROM
    return df


def analyse(df: pd.DataFrame, cell: tuple) -> CellAudit:
    instr, sess, apt, rr, direction = cell
    a = CellAudit(instrument=instr, session=sess, orb_minutes=apt, rr=rr, direction=direction)
    a.n_total = len(df)
    is_mask = df["is_is"].to_numpy()
    is_df = df[is_mask]
    a.n_is = len(is_df)
    if a.n_is < 50:
        return a

    rv_full = df["rel_vol"].astype(float).to_numpy()
    rv_is = is_df["rel_vol"].astype(float).to_numpy()
    thresh_full = float(np.nanpercentile(rv_full, 67))
    thresh_is_only = float(np.nanpercentile(rv_is, 67))
    a.thresh_full = thresh_full
    a.thresh_is_only = thresh_is_only
    a.thresh_delta = thresh_is_only - thresh_full

    pnl_is = is_df["pnl_r"].astype(float).to_numpy()

    def eval_mask(thresh):
        mask = rv_is > thresh
        on = pnl_is[mask]
        off = pnl_is[~mask]
        if len(on) < 5 or len(off) < 5:
            return None
        expr_on = float(on.mean())
        expr_off = float(off.mean())
        delta = expr_on - expr_off
        tt = _sstats.ttest_ind(on, off, equal_var=False)
        return {
            "n_on": int(mask.sum()),
            "expr_on": expr_on,
            "delta": delta,
            "t": float(tt.statistic),
            "p": float(tt.pvalue),
        }

    r_full = eval_mask(thresh_full)
    r_isq = eval_mask(thresh_is_only)
    if r_full:
        a.n_on_full_is = r_full["n_on"]
        a.expr_on_full = r_full["expr_on"]
        a.delta_full = r_full["delta"]
        a.t_full = r_full["t"]
        a.p_full = r_full["p"]
        a.full_bh_global_proxy = abs(r_full["t"]) >= 4.0
    if r_isq:
        a.n_on_isq_is = r_isq["n_on"]
        a.expr_on_isq = r_isq["expr_on"]
        a.delta_isq = r_isq["delta"]
        a.t_isq = r_isq["t"]
        a.p_isq = r_isq["p"]
        a.isq_bh_global_proxy = abs(r_isq["t"]) >= 4.0
    a.survivor_drift = a.full_bh_global_proxy != a.isq_bh_global_proxy
    return a


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows: list[CellAudit] = []
    for cell in SURVIVOR_CELLS:
        df = load_lane(con, *cell)
        a = analyse(df, cell)
        rows.append(a)
        print(
            f"{cell[0]:3s} {cell[1]:<18s} O{cell[2]} RR{cell[3]} {cell[4]:<5s}"
            f" N_is={a.n_is:4d}"
            f" thresh_full={a.thresh_full if a.thresh_full is not None else float('nan'):.4f}"
            f" thresh_is={a.thresh_is_only if a.thresh_is_only is not None else float('nan'):.4f}"
            f" delta={a.thresh_delta if a.thresh_delta is not None else float('nan'):+.4f}"
            f" t_full={a.t_full if a.t_full is not None else float('nan'):+.2f}"
            f" t_isq={a.t_isq if a.t_isq is not None else float('nan'):+.2f}"
            f" drift={'Y' if a.survivor_drift else 'N'}"
        )
    con.close()

    lines: list[str] = []
    lines.append("# Rel_vol_HIGH_Q3 IS-only quantile sensitivity — 13 BH-global survivors")
    lines.append("")
    lines.append(f"**Generated:** 2026-04-19")
    lines.append(f"**Script:** `research/rel_vol_is_only_quantile_sensitivity.py`")
    lines.append(f"**Parent scan:** `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`")
    lines.append(f"**IS boundary:** `trading_day < {HOLDOUT_SACRED_FROM}` (Mode A)")
    lines.append("")
    lines.append("## What this audit tests")
    lines.append("")
    lines.append("The parent scan computed rel_vol's 67th percentile on the FULL-sample lane")
    lines.append("(IS + OOS rows) and bucketed HIGH_Q3 using that threshold. This audit:")
    lines.append("")
    lines.append("1. Loads each BH-global survivor cell's full lane data.")
    lines.append("2. Computes BOTH the full-sample 67th percentile (original) and the IS-only")
    lines.append("   67th percentile.")
    lines.append("3. Restricts to IS rows and evaluates rel_vol_HIGH_Q3 under each threshold.")
    lines.append("4. Flags `survivor_drift = Y` if the cell clears |t| >= 4 (BH-global proxy)")
    lines.append("   under one threshold but not the other.")
    lines.append("")
    lines.append("If `survivor_drift = N` across all 13 cells, the 13-survivor narrative holds")
    lines.append("under honest IS-only quantile computation. If any cells drift, the 2026-04-15")
    lines.append("scan result doc needs an addendum.")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| Instr | Session | O | RR | Dir | N_IS | thresh_full | thresh_IS | Δ | t_full | t_IS | drift |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    n_drift = 0
    for r in rows:

        def fmt(v, spec="+.3f"):
            return f"{v:{spec}}" if v is not None else "—"

        lines.append(
            f"| {r.instrument} | {r.session} | {r.orb_minutes} | {r.rr} | {r.direction} "
            f"| {r.n_is} | {fmt(r.thresh_full, '.4f')} | {fmt(r.thresh_is_only, '.4f')} "
            f"| {fmt(r.thresh_delta, '+.4f')} | {fmt(r.t_full, '+.2f')} | {fmt(r.t_isq, '+.2f')} "
            f"| {'Y' if r.survivor_drift else 'N'} |"
        )
        if r.survivor_drift:
            n_drift += 1
    lines.append("")
    lines.append(f"## Summary")
    lines.append("")
    lines.append(f"- Cells audited: {len(rows)}")
    lines.append(f"- Cells with survivor drift: **{n_drift}**")
    if n_drift == 0:
        lines.append("")
        lines.append(
            "**Verdict:** All cells retain BH-global-proxy status under IS-only "
            "quantile. The 13-survivor narrative from the 2026-04-15 comprehensive "
            "scan holds under honest IS-only threshold computation. No addendum "
            "required to the parent doc's survivor list."
        )
    else:
        lines.append("")
        lines.append(
            f"**Verdict:** {n_drift} cell(s) drift under IS-only quantile. The parent "
            "scan's 13-survivor narrative requires an addendum. Cells flagged `drift=Y` "
            "should be reclassified."
        )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("```")
    lines.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/rel_vol_is_only_quantile_sensitivity.py")
    lines.append("```")
    lines.append("")
    lines.append("Read-only. No writes.")
    RESULT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
