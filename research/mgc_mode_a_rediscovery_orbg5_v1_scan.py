#!/usr/bin/env python3
"""MGC Mode A rediscovery — ORB_G5 long RR1.5 across 4 sessions (K=4).

Pre-registered at:
  docs/audit/hypotheses/2026-04-19-mgc-mode-a-rediscovery-orbg5-v1.yaml
  LOCKED at commit e227ceb3.

Reads ONLY canonical tables (daily_features + orb_outcomes). No writes.
Canonical filter delegation via research.filter_utils.filter_signal.

Output: docs/audit/results/2026-04-19-mgc-mode-a-rediscovery-orbg5-v1-scan.md
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as _sstats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from research.filter_utils import filter_signal

RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-19-mgc-mode-a-rediscovery-orbg5-v1-scan.md"

CELLS = [
    {"id": "H1_LONDON_METALS", "session": "LONDON_METALS"},
    {"id": "H2_COMEX_SETTLE",  "session": "COMEX_SETTLE"},
    {"id": "H3_US_DATA_1000",  "session": "US_DATA_1000"},
    {"id": "H4_EUROPE_FLOW",   "session": "EUROPE_FLOW"},
]

INSTRUMENT = "MGC"
ORB_MINUTES = 5
RR = 1.5
DIRECTION = "long"
FILTER_KEY = "ORB_G5"

# Pre-reg baselines (Mode A IS, computed 2026-04-19 — K2 smoke-test source)
PREREG_BASELINES = {
    "LONDON_METALS_unfiltered": {"N": 469, "ExpR": -0.1528},
    "COMEX_SETTLE_unfiltered":  {"N": 431, "ExpR": -0.2012},
    "US_DATA_1000_unfiltered":  {"N": 439, "ExpR": -0.0298},
    "EUROPE_FLOW_unfiltered":   {"N": 493, "ExpR": -0.1111},
    "LONDON_METALS_ORB_G5_on":  {"N": 41,  "ExpR": 0.1793},
    "COMEX_SETTLE_ORB_G5_on":   {"N": 33,  "ExpR": 0.0470},
    "US_DATA_1000_ORB_G5_on":   {"N": 133, "ExpR": 0.0671},
    "EUROPE_FLOW_ORB_G5_on":    {"N": 26,  "ExpR": 0.3377},
}
K2_TOLERANCE = 0.001  # same-path smoke-test, tight


@dataclass
class CellResult:
    id: str
    session: str
    n_base: int = 0
    expr_base: float | None = None
    n_on: int = 0
    n_off: int = 0
    expr_on: float | None = None
    expr_off: float | None = None
    wr_on: float | None = None
    wr_off: float | None = None
    t_is: float | None = None
    raw_p: float | None = None
    bootstrap_p: float | None = None
    delta_is: float | None = None
    fire_rate: float | None = None
    tautology_corr_orbsize: float | None = None
    tautology_corr_atr: float | None = None
    tautology_corr_ovnrng: float | None = None
    wr_spread: float | None = None
    per_year_positive: int = 0
    years_breakdown: dict[int, dict[str, Any]] = field(default_factory=dict)
    q_family: float | None = None
    passes_bh_family: bool = False
    n_oos_on: int = 0
    expr_oos_on: float | None = None
    delta_oos: float | None = None
    dir_match: bool | None = None
    extreme_fire: bool = False
    tautology: bool = False
    arithmetic_only: bool = False
    positive_mean_floor: bool = False
    k2_base_match: bool | None = None
    k2_on_match: bool | None = None
    gate_results: dict[str, bool] = field(default_factory=dict)
    h1_pass: bool = False
    verdict: str = "UNKNOWN"


def t_test(pnl: np.ndarray) -> tuple[float | None, float | None]:
    if len(pnl) < 2:
        return None, None
    m = float(np.mean(pnl))
    s = float(np.std(pnl, ddof=1))
    if s == 0:
        return None, None
    t = m / (s / math.sqrt(len(pnl)))
    p = float(2.0 * (1.0 - _sstats.t.cdf(abs(t), df=len(pnl) - 1)))
    return float(t), p


def block_bootstrap(pnl: np.ndarray, block: int = 5, B: int = 10_000, seed: int = 42) -> float | None:
    n = len(pnl)
    if n < block * 2:
        return None
    obs = float(abs(np.mean(pnl)))
    centered = pnl - np.mean(pnl)
    rng = np.random.default_rng(seed)
    nblocks = int(math.ceil(n / block))
    ex = 0
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=nblocks)
        boot = np.concatenate([centered[s:s + block] for s in starts])[:n]
        if abs(float(np.mean(boot))) >= obs:
            ex += 1
    return (ex + 1) / (B + 1)


def bh_qvals(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(m, dtype=float)
    running_min = float("inf")
    for i in range(m - 1, -1, -1):
        idx = int(order[i])
        p = pvals[idx]
        raw = p * m / (i + 1)
        running_min = min(running_min, raw)
        q[idx] = min(1.0, running_min)
    return q.tolist()


def corr_with(df: pd.DataFrame, fire: np.ndarray, col: str) -> float | None:
    if col not in df.columns:
        return None
    vals = df[col].astype(float).to_numpy()
    fire_f = np.asarray(fire).astype(float)
    valid = ~np.isnan(vals)
    if valid.sum() < 30:
        return None
    v = vals[valid]
    f = fire_f[valid]
    if f.sum() == 0 or f.sum() == len(f) or np.std(v) == 0:
        return None
    r = float(np.corrcoef(f, v)[0, 1])
    return None if math.isnan(r) else r


def analyze(con: duckdb.DuckDBPyConnection, cell: dict) -> CellResult:
    sess = cell["session"]
    sql = f"""
    SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND d.orb_{sess}_break_dir=? AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(sql, [INSTRUMENT, sess, ORB_MINUTES, RR, DIRECTION]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["is_is"] = df["trading_day"] < HOLDOUT_SACRED_FROM
    df["is_oos"] = df["trading_day"] >= HOLDOUT_SACRED_FROM

    res = CellResult(id=cell["id"], session=sess)
    is_df = df[df["is_is"]].reset_index(drop=True)
    oos_df = df[df["is_oos"]].reset_index(drop=True)

    res.n_base = len(is_df)
    if res.n_base > 0:
        res.expr_base = float(is_df["pnl_r"].mean())

    # K2 smoke
    base_key = f"{sess}_unfiltered"
    if base_key in PREREG_BASELINES and res.expr_base is not None:
        res.k2_base_match = abs(res.expr_base - PREREG_BASELINES[base_key]["ExpR"]) < K2_TOLERANCE

    fire_is = np.asarray(filter_signal(is_df, FILTER_KEY, sess)).astype(bool)
    on_pnl = is_df.loc[fire_is, "pnl_r"].astype(float).to_numpy()
    off_pnl = is_df.loc[~fire_is, "pnl_r"].astype(float).to_numpy()
    res.n_on = int(fire_is.sum())
    res.n_off = int((~fire_is).sum())

    if res.n_on > 0:
        res.expr_on = float(np.mean(on_pnl))
        res.wr_on = float(np.mean(is_df.loc[fire_is, "outcome"].astype(str) == "win"))
        res.fire_rate = res.n_on / res.n_base if res.n_base > 0 else None
        t, p = t_test(on_pnl)
        res.t_is, res.raw_p = t, p
        res.bootstrap_p = block_bootstrap(on_pnl)
        res.positive_mean_floor = res.expr_on > 0

        on_key = f"{sess}_ORB_G5_on"
        if on_key in PREREG_BASELINES:
            res.k2_on_match = abs(res.expr_on - PREREG_BASELINES[on_key]["ExpR"]) < K2_TOLERANCE

    if res.n_off > 0:
        res.expr_off = float(np.mean(off_pnl))
        res.wr_off = float(np.mean(is_df.loc[~fire_is, "outcome"].astype(str) == "win"))

    if res.expr_on is not None and res.expr_base is not None:
        res.delta_is = res.expr_on - res.expr_base
    if res.wr_on is not None and res.wr_off is not None:
        res.wr_spread = res.wr_on - res.wr_off

    # T0 tautology — against orb_size (self-referential expected), atr_20 (structural), overnight_range (structural)
    size_col = f"orb_{sess}_size"
    res.tautology_corr_orbsize = corr_with(is_df, fire_is, size_col)
    res.tautology_corr_atr = corr_with(is_df, fire_is, "atr_20")
    res.tautology_corr_ovnrng = corr_with(is_df, fire_is, "overnight_range")
    # ORB_G5 is self-referentially tautological with orb_size (expected). The pre-reg
    # § T0 tautology clause says cross-filter tautology matters, not self-tautology.
    cross_corrs = [c for c in [res.tautology_corr_atr, res.tautology_corr_ovnrng] if c is not None]
    res.tautology = any(abs(c) > 0.70 for c in cross_corrs)

    if res.fire_rate is not None:
        res.extreme_fire = res.fire_rate < 0.05 or res.fire_rate > 0.95
    if res.wr_spread is not None and res.delta_is is not None:
        res.arithmetic_only = abs(res.wr_spread) < 0.03 and abs(res.delta_is) > 0.10

    # Per-year
    is_df["_year"] = pd.to_datetime(is_df["trading_day"]).dt.year
    for yr in sorted(is_df["_year"].unique()):
        yr_mask = (is_df["_year"] == yr).to_numpy()
        yr_fire = fire_is & yr_mask
        n = int(yr_fire.sum())
        if n == 0:
            res.years_breakdown[int(yr)] = {"n": 0, "expr": None, "positive": None}
            continue
        yr_expr = float(is_df.loc[yr_fire, "pnl_r"].astype(float).mean())
        pos = yr_expr > 0
        res.years_breakdown[int(yr)] = {"n": n, "expr": yr_expr, "positive": pos}
        if n >= 10 and pos:
            res.per_year_positive += 1

    # OOS descriptive
    if len(oos_df) > 0:
        fire_oos = np.asarray(filter_signal(oos_df, FILTER_KEY, sess)).astype(bool)
        res.n_oos_on = int(fire_oos.sum())
        if res.n_oos_on > 0:
            res.expr_oos_on = float(oos_df.loc[fire_oos, "pnl_r"].astype(float).mean())
            expr_oos_base = float(oos_df["pnl_r"].astype(float).mean())
            res.delta_oos = res.expr_oos_on - expr_oos_base
            if res.delta_is is not None and res.delta_oos is not None:
                res.dir_match = (res.delta_is > 0 and res.delta_oos > 0) or (res.delta_is < 0 and res.delta_oos < 0)

    return res


def eval_gates(cells: list[CellResult]) -> None:
    pvals = [c.raw_p if c.raw_p is not None else 1.0 for c in cells]
    qs = bh_qvals(pvals)
    for c, q in zip(cells, qs):
        c.q_family = q
        c.passes_bh_family = q < 0.05

    for c in cells:
        g = {}
        g["bh_pass_family"] = bool(c.passes_bh_family)
        g["abs_t_IS_ge_3"] = bool(c.t_is is not None and abs(c.t_is) >= 3.0)
        g["N_IS_on_ge_100"] = bool(c.n_on >= 100)
        g["years_positive_ge_3"] = bool(c.per_year_positive >= 3)
        g["bootstrap_p_lt_0.10"] = bool(c.bootstrap_p is not None and c.bootstrap_p < 0.10)
        g["ExpR_on_IS_gt_0"] = bool(c.positive_mean_floor)
        g["not_tautology"] = not c.tautology
        g["not_extreme_fire"] = not c.extreme_fire
        g["not_arithmetic_only"] = not c.arithmetic_only
        c.gate_results = g
        c.h1_pass = all(g.values())
        c.verdict = "CONTINUE" if c.h1_pass else "KILL"


def _fmt(x, p=4):
    if x is None: return "—"
    if isinstance(x, float) and math.isnan(x): return "nan"
    if isinstance(x, bool): return "Y" if x else "N"
    if isinstance(x, float): return f"{x:.{p}f}"
    return str(x)


def render(cells: list[CellResult]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_cont = sum(1 for c in cells if c.verdict == "CONTINUE")
    lines: list[str] = []
    lines.append("# MGC Mode A rediscovery — ORB_G5 long RR1.5 K=4 scan")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mgc-mode-a-rediscovery-orbg5-v1.yaml` (LOCKED, commit_sha=e227ceb3)")
    lines.append(f"**Script:** `research/mgc_mode_a_rediscovery_orbg5_v1_scan.py`")
    lines.append(f"**IS window:** `trading_day < {HOLDOUT_SACRED_FROM}` (Mode A)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Cells: {len(cells)} | CONTINUE: {n_cont} | KILL: {len(cells) - n_cont}")
    lines.append("")
    k2_ok = all((c.k2_base_match is not False and c.k2_on_match is not False) for c in cells)
    lines.append(f"**K2 baseline sanity smoke-test:** {'PASS' if k2_ok else 'FAIL'} (same-path reproducibility only; see pre-reg § Baseline cross-check).")
    lines.append("")
    lines.append("## Per-cell IS results")
    lines.append("")
    lines.append("| Cell | Session | N_base | N_on | Fire% | ExpR_base | ExpR_on | Δ_IS | t | raw_p | boot_p | q_family | years_pos |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        lines.append(
            f"| {c.id} | {c.session} | {c.n_base} | {c.n_on} | "
            f"{_fmt(c.fire_rate, 3)} | {_fmt(c.expr_base)} | {_fmt(c.expr_on)} | "
            f"{_fmt(c.delta_is)} | {_fmt(c.t_is, 3)} | {_fmt(c.raw_p)} | "
            f"{_fmt(c.bootstrap_p)} | {_fmt(c.q_family)} | {c.per_year_positive} |"
        )
    lines.append("")
    lines.append("## Gate breakdown")
    lines.append("")
    keys = ["bh_pass_family", "abs_t_IS_ge_3", "N_IS_on_ge_100", "years_positive_ge_3",
            "bootstrap_p_lt_0.10", "ExpR_on_IS_gt_0", "not_tautology", "not_extreme_fire", "not_arithmetic_only"]
    lines.append("| Cell | " + " | ".join(keys) + " | Verdict |")
    lines.append("|---|" + "|".join(["---"] * (len(keys) + 1)) + "|")
    for c in cells:
        row = [c.id] + ["Y" if c.gate_results.get(k) else "N" for k in keys] + [c.verdict]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Flags & T0 (cross-filter tautology — ORB_G5 is trivially correlated with orb_size; check vs atr_20, overnight_range)")
    lines.append("")
    lines.append("| Cell | fire_rate | corr_orbsize (self, expected ~1) | corr_atr | corr_ovnrng | tautology | extreme_fire | arithmetic_only |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for c in cells:
        lines.append(
            f"| {c.id} | {_fmt(c.fire_rate, 3)} | "
            f"{_fmt(c.tautology_corr_orbsize, 3)} | {_fmt(c.tautology_corr_atr, 3)} | "
            f"{_fmt(c.tautology_corr_ovnrng, 3)} | {_fmt(c.tautology)} | "
            f"{_fmt(c.extreme_fire)} | {_fmt(c.arithmetic_only)} |"
        )
    lines.append("")
    lines.append("## OOS descriptive (NOT used to select or tune)")
    lines.append("")
    lines.append("| Cell | N_OOS_on | ExpR_OOS_on | Δ_OOS | dir_match |")
    lines.append("|---|---:|---:|---:|---|")
    for c in cells:
        lines.append(
            f"| {c.id} | {c.n_oos_on} | {_fmt(c.expr_oos_on)} | {_fmt(c.delta_oos)} | {_fmt(c.dir_match)} |"
        )
    lines.append("")
    lines.append("## Per-year IS breakdown")
    lines.append("")
    yrs = sorted({y for c in cells for y in c.years_breakdown})
    lines.append("| Cell | " + " | ".join(str(y) for y in yrs) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(yrs)) + "|")
    for c in cells:
        cells_str = []
        for y in yrs:
            b = c.years_breakdown.get(y)
            if not b or b["n"] == 0:
                cells_str.append("—")
            elif b["n"] < 10:
                cells_str.append(f"N={b['n']}")
            else:
                sign = "+" if b["positive"] else "-"
                cells_str.append(f"{sign}{_fmt(b['expr'], 3)}(N={b['n']})")
        lines.append("| " + c.id + " | " + " | ".join(cells_str) + " |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if n_cont == 0 and k2_ok:
        lines.append("**Verdict: KILL per pre-reg K1.** Zero of 4 cells pass all gate clauses. MGC ORB_G5 long RR1.5 on the 4 pre-registered sessions does NOT yield a Pathway A Chordia-validated edge on 3.5-year Mode A IS. Honest negative evidence on MGC cross-instrument-mirror hypothesis. Pre-reg explicitly anticipated this outcome; baselines (approx_t 0.23-1.47) predicted a KILL. No re-runs with different thresholds.")
    elif n_cont >= 1 and k2_ok:
        lines.append(f"**Verdict: CONTINUE on {n_cont} cell(s).** Candidates for committee review.")
    else:
        lines.append("**Verdict: HARNESS FAIL per K2.** Non-authoritative.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mgc_mode_a_rediscovery_orbg5_v1_scan.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        cells = [analyze(con, c) for c in CELLS]
    finally:
        con.close()
    eval_gates(cells)
    for c in cells:
        print(f"{c.id} N_on={c.n_on} ExpR_on={_fmt(c.expr_on)} t={_fmt(c.t_is, 3)} q={_fmt(c.q_family)} -> {c.verdict}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(cells), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
