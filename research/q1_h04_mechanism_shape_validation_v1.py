"""Q1 H04 mechanism-shape validation v1.

Tests whether the H04 confluence signal (rel_vol_HIGH AND F6_INSIDE_PDR) on
MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 short is a real interaction mechanism
by examining the shape of E[pnl_r] across the full rel_vol gradient x F6
state on IS, and then checking whether that shape direction is preserved on
2026 sacred OOS descriptively.

Pre-reg: docs/audit/hypotheses/2026-04-20-q1-h04-mechanism-shape-validation-v1.yaml
Lock SHA: c93ba90d (see YAML reproducibility.commit_sha)
Parent lock SHA: c6ece8a1 (H04 family)
Sibling lock SHA: b5b7bfbf (H04 shadow deployment-shape)

Verdict labels: CONFIRMED / UNVERIFIED_ALIVE / CONTRADICTED / KILL / SCAN_ABORT.
No capital action permitted under any verdict.
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
from research.filter_utils import filter_signal
from research.oos_power import format_power_report, oos_ttest_power
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

LANE_ID = "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5"
SESSION = "COMEX_SETTLE"
FILTER_KEY = "ORB_G5"

PARENT_H04_CELL_EXPR = 0.3855
PARENT_H04_CELL_N = 128
PARENT_H04_DELTA = 0.3579
PARITY_EXPR_TOL = 0.015
PARITY_N_TOL = 3
PARITY_DELTA_TOL = 0.020

RESULT_DOC = Path("docs/audit/results/2026-04-20-q1-h04-mechanism-shape-validation-v1.md")
PREREG_PATH = "docs/audit/hypotheses/2026-04-20-q1-h04-mechanism-shape-validation-v1.yaml"
PREREG_SHA = "c93ba90d"


def _load_lane(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load canonical short-lane rows, triple-joined, orb_minutes-guarded."""
    sql = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_price,
      o.stop_price,
      d.prev_day_high,
      d.prev_day_low,
      d.rel_vol_COMEX_SETTLE         AS rel_vol,
      d.orb_COMEX_SETTLE_high        AS orb_high,
      d.orb_COMEX_SETTLE_low         AS orb_low,
      d.orb_COMEX_SETTLE_size        AS orb_size,
      d.orb_COMEX_SETTLE_break_dir   AS break_dir,
      d.orb_COMEX_SETTLE_vwap        AS orb_vwap,
      d.orb_COMEX_SETTLE_volume      AS orb_volume,
      d.orb_COMEX_SETTLE_break_delay_min AS break_delay_min
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'COMEX_SETTLE'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.5
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day, o.entry_ts
    """
    df = con.sql(sql).to_df()
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    short = df.loc[df["direction"] == "short"].reset_index(drop=True).copy()
    return short


def _apply_canonical_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ORB_G5 via the canonical research.filter_utils helper — RULE 4 of
    institutional-rigor.md and research-truth-protocol.md filter-delegation rule."""
    # filter_signal reads `orb_{session}_*` columns; our DataFrame uses aliased
    # short names.  Recompose the fully-qualified columns for the filter call
    # only — keep the aliased ones for the rest of the analysis.
    df_for_filter = df.copy()
    df_for_filter["orb_COMEX_SETTLE_size"] = df["orb_size"]
    df_for_filter["orb_COMEX_SETTLE_vwap"] = df["orb_vwap"]
    df_for_filter["orb_COMEX_SETTLE_high"] = df["orb_high"]
    df_for_filter["orb_COMEX_SETTLE_low"] = df["orb_low"]
    df_for_filter["orb_COMEX_SETTLE_break_dir"] = df["break_dir"]
    sig = filter_signal(df_for_filter, FILTER_KEY, orb_label=SESSION)
    out = df.loc[sig == 1].reset_index(drop=True).copy()
    return out


def _f6_inside_pdr(df: pd.DataFrame) -> pd.Series:
    """Canonical predicate — matches research/mnq_live_context_overlays_v1.py::_f6_inside_pdr."""
    mid = (df["orb_high"].astype(float) + df["orb_low"].astype(float)) / 2.0
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    return ((mid > pdl) & (mid < pdh)).fillna(False)


def _is_oos_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    td = pd.to_datetime(df["trading_day"])
    is_mask = td < holdout
    return df.loc[is_mask].reset_index(drop=True), df.loc[~is_mask].reset_index(drop=True)


@dataclass
class CellStat:
    q: int
    f6: bool
    n: int
    expr: float
    wr: float


def _quintile_breakpoints(is_df: pd.DataFrame) -> np.ndarray:
    """IS-only quintile breakpoints on rel_vol (q=1..5)."""
    v = is_df["rel_vol"].astype(float).dropna().to_numpy(dtype=np.float64)
    if len(v) < 25:
        raise RuntimeError(f"insufficient IS rows for quintile calibration: {len(v)}")
    return np.asarray(np.nanpercentile(v, [20, 40, 60, 80]), dtype=np.float64)


def _assign_quintile(series: pd.Series, breaks: np.ndarray) -> pd.Series:
    """1..5 by frozen IS breakpoints.  NaN rel_vol -> 0 sentinel (excluded from cells)."""
    v = series.astype(float).values
    q = np.full_like(v, fill_value=np.nan, dtype=float)
    mask = ~np.isnan(v)
    # np.digitize: breaks define 4 boundaries -> 5 bins
    q[mask] = np.digitize(v[mask], breaks, right=False) + 1
    return pd.Series(q, index=series.index)


def _cell_matrix(df: pd.DataFrame, breaks: np.ndarray) -> list[CellStat]:
    q = _assign_quintile(df["rel_vol"], breaks)
    f6 = _f6_inside_pdr(df)
    out: list[CellStat] = []
    for qi in [1, 2, 3, 4, 5]:
        for f in [True, False]:
            mask = (q == qi) & (f6 == f) & q.notna()
            sub = df.loc[mask, "pnl_r"].astype(float)
            n = int(len(sub))
            expr = float(sub.mean()) if n > 0 else float("nan")
            wr = float((sub > 0).mean()) if n > 0 else float("nan")
            out.append(CellStat(q=qi, f6=bool(f), n=n, expr=expr, wr=wr))
    return out


def _spearman_within_f6(cells: list[CellStat], f6: bool) -> tuple[float, float]:
    """Rank correlation of (quintile, ExpR) within F6 row, weighted None.

    Returns (rho, p_perm_10k) using per-cell ExpR (weight = 1 per cell).
    Cells with N=0 or NaN ExpR are dropped.
    """
    xs, ys = [], []
    for c in cells:
        if c.f6 == f6 and c.n > 0 and not np.isnan(c.expr):
            xs.append(c.q)
            ys.append(c.expr)
    if len(xs) < 3:
        return float("nan"), float("nan")
    sp = stats.spearmanr(xs, ys)
    rho = float(sp.statistic)  # type: ignore[attr-defined]
    # 10k permutation p (two-sided)
    rng = np.random.default_rng(42)
    n = 10000
    cnt = 0
    ys_arr = np.asarray(ys, dtype=np.float64)
    xs_arr = np.asarray(xs, dtype=np.float64)
    for _ in range(n):
        perm = rng.permutation(ys_arr)
        sp_perm = stats.spearmanr(xs_arr, perm)
        r = float(sp_perm.statistic)  # type: ignore[attr-defined]
        if not np.isnan(r) and abs(r) >= abs(rho):
            cnt += 1
    return rho, (cnt + 1) / (n + 1)


def _interaction_surplus(cells: list[CellStat]) -> float:
    def c(q: int, f6: bool) -> float:
        for cell in cells:
            if cell.q == q and cell.f6 == f6:
                return cell.expr if cell.n > 0 else float("nan")
        return float("nan")

    t5 = c(5, True)
    t1 = c(1, True)
    f5 = c(5, False)
    f1 = c(1, False)
    return (t5 - t1) - (f5 - f1)


def _interaction_surplus_permutation(df: pd.DataFrame, breaks: np.ndarray, n_perm: int = 10000) -> tuple[float, float]:
    """Permute pnl_r labels and recompute interaction surplus null distribution."""
    q = _assign_quintile(df["rel_vol"], breaks)
    f6 = _f6_inside_pdr(df)
    valid = q.notna() & f6.notna()
    sub = df.loc[valid].reset_index(drop=True)
    qv = q.loc[valid].reset_index(drop=True).astype(int)
    fv = f6.loc[valid].reset_index(drop=True).astype(bool)
    pnl = sub["pnl_r"].astype(float).to_numpy(dtype=np.float64)

    def compute_surplus(pnl_array: np.ndarray) -> float:
        df_tmp = pd.DataFrame({"q": qv, "f6": fv, "pnl_r": pnl_array})
        def mean_cell(qi: int, f6v: bool) -> float:
            m = (df_tmp["q"] == qi) & (df_tmp["f6"] == f6v)
            return float(df_tmp.loc[m, "pnl_r"].mean()) if m.any() else float("nan")
        return (mean_cell(5, True) - mean_cell(1, True)) - (mean_cell(5, False) - mean_cell(1, False))

    observed = compute_surplus(pnl)
    if np.isnan(observed):
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pnl)
        s = compute_surplus(perm)
        if not np.isnan(s) and abs(s) >= abs(observed):
            cnt += 1
    return observed, (cnt + 1) / (n_perm + 1)


def _parity_check(is_prefilter_df: pd.DataFrame, is_filtered_df: pd.DataFrame) -> dict:
    """Reproduce H04 filtered-pass cell per parent methodology.

    Parent's P67 is computed on the IS-only short-lane PRE-filter universe
    (see research/mnq_live_context_overlays_v1.py::_calibrate_rel_vol_threshold
    called on `cmx_short_base` which is the short subset prior to ORB_G5).
    The rv_HI AND F6 cell is then evaluated on the filtered subset.
    """
    v = is_prefilter_df["rel_vol"].astype(float).dropna().to_numpy(dtype=np.float64)
    p67 = float(np.nanpercentile(v, 67))
    rv_hi = (is_filtered_df["rel_vol"].astype(float) > p67).fillna(False)
    f6 = _f6_inside_pdr(is_filtered_df)
    on = is_filtered_df.loc[rv_hi & f6, "pnl_r"].astype(float)
    off = is_filtered_df.loc[~(rv_hi & f6), "pnl_r"].astype(float)
    delta = float(on.mean() - off.mean()) if len(on) > 0 and len(off) > 0 else float("nan")
    return {
        "p67_is": p67,
        "n_on": int(len(on)),
        "expr_on": float(on.mean()) if len(on) > 0 else float("nan"),
        "n_off": int(len(off)),
        "expr_off": float(off.mean()) if len(off) > 0 else float("nan"),
        "delta": delta,
    }


def _render_cell_table(cells: list[CellStat], title: str) -> str:
    lines = [f"### {title}", "", "| quintile | F6_TRUE N | F6_TRUE ExpR | F6_TRUE WR | F6_FALSE N | F6_FALSE ExpR | F6_FALSE WR |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    by_q: dict[int, dict[bool, CellStat]] = {}
    for c in cells:
        by_q.setdefault(c.q, {})[c.f6] = c
    for q in sorted(by_q):
        t = by_q[q].get(True)
        f = by_q[q].get(False)
        def fmt(c: CellStat | None, col: str) -> str:
            if c is None or c.n == 0:
                return "-"
            if col == "n":
                return str(c.n)
            if col == "expr":
                return f"{c.expr:+.4f}"
            if col == "wr":
                return f"{c.wr:.3f}"
            return "-"
        lines.append(
            f"| Q{q} | {fmt(t,'n')} | {fmt(t,'expr')} | {fmt(t,'wr')} "
            f"| {fmt(f,'n')} | {fmt(f,'expr')} | {fmt(f,'wr')} |"
        )
    return "\n".join(lines)


def _year_table(is_df: pd.DataFrame, breaks: np.ndarray) -> str:
    q = _assign_quintile(is_df["rel_vol"], breaks)
    f6 = _f6_inside_pdr(is_df)
    is_df2 = is_df.copy()
    is_df2["q"] = q
    is_df2["f6"] = f6
    is_df2["year"] = is_df2["trading_day"].astype(str).str[:4]
    lines = ["### IS year-by-year — top-F6-Q5 cell vs rest",
             "", "| year | N_top | ExpR_top | N_rest | ExpR_rest | Δ |",
             "|---|---:|---:|---:|---:|---:|"]
    for yr, g in is_df2.groupby("year"):
        top = g[(g["q"] == 5) & (g["f6"])]
        rest = g.loc[~g.index.isin(top.index)]
        dt = float(top["pnl_r"].mean() - rest["pnl_r"].mean()) if len(top) > 0 and len(rest) > 0 else float("nan")
        lines.append(
            f"| {yr} | {len(top)} | "
            f"{(top['pnl_r'].mean() if len(top)>0 else float('nan')):+.4f} | "
            f"{len(rest)} | "
            f"{(rest['pnl_r'].mean() if len(rest)>0 else float('nan')):+.4f} | "
            f"{dt:+.4f} |"
        )
    return "\n".join(lines)


def _month_table(oos_df: pd.DataFrame, breaks: np.ndarray) -> str:
    if len(oos_df) == 0:
        return "### OOS per-month — (no OOS rows)"
    q = _assign_quintile(oos_df["rel_vol"], breaks)
    f6 = _f6_inside_pdr(oos_df)
    oos = oos_df.copy()
    oos["q"] = q
    oos["f6"] = f6
    oos["month"] = pd.to_datetime(oos["trading_day"]).dt.strftime("%Y-%m")
    lines = ["### OOS per-month — top-F6-Q5 cell (descriptive, power-qualified)",
             "", "| month | N_top | ExpR_top | N_rest | ExpR_rest |",
             "|---|---:|---:|---:|---:|"]
    for m, g in oos.groupby("month"):
        top = g[(g["q"] == 5) & (g["f6"])]
        rest = g.loc[~g.index.isin(top.index)]
        lines.append(
            f"| {m} | {len(top)} | "
            f"{(top['pnl_r'].mean() if len(top)>0 else float('nan')):+.4f} | "
            f"{len(rest)} | "
            f"{(rest['pnl_r'].mean() if len(rest)>0 else float('nan')):+.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        lane = _load_lane(con)
    finally:
        con.close()
    filtered = _apply_canonical_filter(lane)
    is_df, oos_df = _is_oos_split(filtered)
    is_prefilter_df, _ = _is_oos_split(lane)

    # Parity vs parent H04 — parent calibrated P67 on IS-only short PRE-filter rows
    parity = _parity_check(is_prefilter_df, is_df)

    # Quintile breakpoints frozen on IS
    breaks = _quintile_breakpoints(is_df)

    # Cell matrices
    is_cells = _cell_matrix(is_df, breaks)
    oos_cells = _cell_matrix(oos_df, breaks)

    # Shape metrics
    rho_is, rho_is_p = _spearman_within_f6(is_cells, f6=True)
    rho_oos, rho_oos_p = _spearman_within_f6(oos_cells, f6=True)
    surplus_is, surplus_is_p = _interaction_surplus_permutation(is_df, breaks)
    surplus_oos, surplus_oos_p = _interaction_surplus_permutation(oos_df, breaks) if len(oos_df) > 0 else (float("nan"), float("nan"))

    # Top-cell power — compute OOS power to detect IS interaction-surplus effect
    top_is = next(c for c in is_cells if c.q == 5 and c.f6)
    top_oos = next(c for c in oos_cells if c.q == 5 and c.f6)
    pooled_std = float(filtered["pnl_r"].std())
    # IS delta = top-cell ExpR minus rest-of-filtered ExpR (the deployable comparison)
    rest_expr_is = float(is_df.loc[
        ~((_assign_quintile(is_df["rel_vol"], breaks) == 5) & _f6_inside_pdr(is_df)),
        "pnl_r"
    ].astype(float).mean())
    is_delta_top = top_is.expr - rest_expr_is if top_is.n > 0 else float("nan")
    n_oos_rest = max(sum(c.n for c in oos_cells) - top_oos.n, 2)
    if top_oos.n >= 2 and pooled_std > 0 and not np.isnan(is_delta_top):
        power_dict = oos_ttest_power(
            is_delta=is_delta_top,
            is_pooled_std=pooled_std,
            n_oos_a=max(top_oos.n, 2),
            n_oos_b=n_oos_rest,
        )
        power_top = power_dict["power"]
        cohens_d = power_dict["cohen_d"]
        power_report = format_power_report(power_dict)
    else:
        power_top = float("nan")
        cohens_d = float("nan")
        power_report = "OOS top-cell N<2 — power computation skipped (STATISTICALLY_USELESS)"

    # Verdict logic
    parity_ok = (
        abs(parity["n_on"] - PARENT_H04_CELL_N) <= PARITY_N_TOL
        and abs(parity["expr_on"] - PARENT_H04_CELL_EXPR) <= PARITY_EXPR_TOL
        and abs(parity["delta"] - PARENT_H04_DELTA) <= PARITY_DELTA_TOL
    )
    is_shape_ok = (not np.isnan(rho_is) and rho_is >= 0.5 and surplus_is > 0)
    oos_direction_preserved = (not np.isnan(surplus_oos)) and np.sign(surplus_oos) == np.sign(surplus_is)
    pooled_top_n = top_oos.n

    if not parity_ok:
        verdict = "SCAN_ABORT"
        verdict_reason = "Parity check failed — integrity violation."
    elif not is_shape_ok:
        verdict = "KILL"
        verdict_reason = f"IS shape does not hold: rho_IS={rho_is:.3f} surplus_IS={surplus_is:+.4f}"
    elif (top_oos.n >= 10 and not np.isnan(top_oos.expr) and top_oos.expr <= -0.15
          and not np.isnan(surplus_oos) and surplus_oos <= -0.15):
        verdict = "CONTRADICTED"
        verdict_reason = "K_kill_4 — OOS top cell catastrophic contradiction."
    elif oos_direction_preserved and pooled_top_n >= 30:
        verdict = "CONFIRMED"
        verdict_reason = "IS shape holds; OOS direction preserves with pooled top-N>=30."
    else:
        verdict = "UNVERIFIED_ALIVE"
        verdict_reason = (
            f"IS shape holds; OOS direction {'preserves' if oos_direction_preserved else 'does not preserve'}; "
            f"pooled top-N={pooled_top_n} (<30 power-floor, descriptive only per RULE 3.3)."
        )

    # Result doc
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    parts.append(f"# Q1 — H04 mechanism-shape validation v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}` (LOCKED, commit_sha=`{PREREG_SHA}`)")
    parts.append(f"**Script:** `research/q1_h04_mechanism_shape_validation_v1.py`")
    parts.append(f"**Lane:** `{LANE_ID}` short")
    parts.append(f"**IS window:** trading_day < {HOLDOUT_SACRED_FROM}")
    parts.append(f"**OOS window:** trading_day >= {HOLDOUT_SACRED_FROM}")
    parts.append("")
    parts.append(f"## Verdict: **{verdict}**")
    parts.append("")
    parts.append(f"> {verdict_reason}")
    parts.append("")
    parts.append("## Parity vs parent H04 result")
    parts.append("")
    parts.append("| metric | parent | this run | tolerance | pass |")
    parts.append("|---|---:|---:|---:|:---:|")
    parts.append(f"| H04 cell N   | {PARENT_H04_CELL_N} | {parity['n_on']} | ±{PARITY_N_TOL} | {'YES' if abs(parity['n_on']-PARENT_H04_CELL_N)<=PARITY_N_TOL else 'NO'} |")
    parts.append(f"| H04 cell ExpR| {PARENT_H04_CELL_EXPR:+.4f} | {parity['expr_on']:+.4f} | ±{PARITY_EXPR_TOL} | {'YES' if abs(parity['expr_on']-PARENT_H04_CELL_EXPR)<=PARITY_EXPR_TOL else 'NO'} |")
    parts.append(f"| H04 Δ (on−off)| {PARENT_H04_DELTA:+.4f} | {parity['delta']:+.4f} | ±{PARITY_DELTA_TOL} | {'YES' if abs(parity['delta']-PARENT_H04_DELTA)<=PARITY_DELTA_TOL else 'NO'} |")
    parts.append(f"| P67 IS threshold (parent frozen 2.012172) | 2.012172 | {parity['p67_is']:.6f} | — | — |")
    parts.append("")
    parts.append("## Quintile breakpoints (IS-only, frozen)")
    parts.append("")
    parts.append("| edge | rel_vol_COMEX_SETTLE |")
    parts.append("|---|---:|")
    for lbl, v in zip(["P20", "P40", "P60", "P80"], breaks):
        parts.append(f"| {lbl} | {v:.6f} |")
    parts.append("")
    parts.append("## Cell matrices")
    parts.append("")
    parts.append(_render_cell_table(is_cells, "IS 5×2 matrix (quintile × F6)"))
    parts.append("")
    parts.append(_render_cell_table(oos_cells, "2026 OOS 5×2 matrix (quintile × F6)"))
    parts.append("")
    parts.append("## Shape metrics")
    parts.append("")
    parts.append("| metric | value | p (10k perm) |")
    parts.append("|---|---:|---:|")
    parts.append(f"| Spearman ρ within F6=TRUE, IS  | {rho_is:+.4f} | {rho_is_p:.4f} |")
    parts.append(f"| Spearman ρ within F6=TRUE, OOS | {rho_oos:+.4f} | {rho_oos_p:.4f} |")
    parts.append(f"| Interaction surplus, IS        | {surplus_is:+.4f} | {surplus_is_p:.4f} |")
    parts.append(f"| Interaction surplus, OOS       | {surplus_oos:+.4f} | {surplus_oos_p:.4f} |")
    parts.append("")
    parts.append("## OOS power — top cell (Q5 × F6_TRUE)")
    parts.append("")
    parts.append(f"- IS top-cell ExpR: {top_is.expr:+.4f} (N={top_is.n})")
    parts.append(f"- OOS top-cell ExpR: {top_oos.expr:+.4f} (N={top_oos.n})")
    parts.append(f"- Cohen's d (top_IS vs 0): {cohens_d:+.4f}")
    parts.append(f"- OOS power vs IS effect: {power_top:.1%}")
    parts.append(f"- Power report: `{power_report}`")
    parts.append("")
    parts.append(_year_table(is_df, breaks))
    parts.append("")
    parts.append(_month_table(oos_df, breaks))
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No capital action, allocator change, or sizing modification.")
    parts.append("- Does not bypass the merged H04 shadow prereg's gate_1 (N>=30 on OOS fires of the binary confluence).")
    parts.append("- Does not re-fit quintile breakpoints; the IS-only breakpoints are now frozen and citable by any follow-on.")
    parts.append("")
    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print(f"VERDICT: {verdict}")
    print(f"REASON: {verdict_reason}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
