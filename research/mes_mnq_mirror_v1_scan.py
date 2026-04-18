#!/usr/bin/env python3
"""MES mirror of MNQ US_DATA_1000 VWAP_MID_ALIGNED long — K=2 family scan.

Pre-registered at:
  docs/audit/hypotheses/2026-04-19-mes-mnq-mirror-vwap-mid-aligned-us-data-1000-v1.yaml

Reads ONLY canonical tables (daily_features + orb_outcomes). No writes to
validated_setups or experimental_strategies. Emits a single result markdown.

Filter delegation: canonical `research.filter_utils.filter_signal` per
research-truth-protocol.md § Canonical filter delegation. No inline filter
re-implementation.

Holdout: Mode A, boundary imported from trading_app.holdout_policy
(not hardcoded). IS = trading_day < HOLDOUT_SACRED_FROM.

Output: docs/audit/results/2026-04-19-mes-mnq-mirror-vwap-mid-aligned-us-data-1000-v1-scan.md

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mes_mnq_mirror_v1_scan.py
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

# =============================================================================
# PRE-REGISTERED FAMILY (locked by YAML commit 9b9826ee)
# =============================================================================

PREREG_YAML = PROJECT_ROOT / "docs/audit/hypotheses/2026-04-19-mes-mnq-mirror-vwap-mid-aligned-us-data-1000-v1.yaml"
RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-19-mes-mnq-mirror-vwap-mid-aligned-us-data-1000-v1-scan.md"

# Cell definitions frozen at pre-reg lock (commit 9b9826ee)
CELLS = [
    {"id": "H1_RR1.5", "instrument": "MES", "session": "US_DATA_1000",
     "orb_minutes": 15, "rr": 1.5, "direction": "long", "filter_key": "VWAP_MID_ALIGNED"},
    {"id": "H2_RR2.0", "instrument": "MES", "session": "US_DATA_1000",
     "orb_minutes": 15, "rr": 2.0, "direction": "long", "filter_key": "VWAP_MID_ALIGNED"},
]

ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

# Baselines from pre-reg § baseline — used for K2 harness cross-check.
PREREG_BASELINES = {
    ("MES", "US_DATA_1000", 15, 1.5, "long", "unfiltered"): {"N": 835, "ExpR": -0.0012},
    ("MES", "US_DATA_1000", 15, 2.0, "long", "unfiltered"): {"N": 758, "ExpR": -0.0603},
    ("MES", "US_DATA_1000", 15, 1.5, "long", "VWAP_MID_ALIGNED"): {"N": 492, "ExpR": 0.0617},
    ("MES", "US_DATA_1000", 15, 2.0, "long", "VWAP_MID_ALIGNED"): {"N": 443, "ExpR": 0.0217},
}

K2_TOLERANCE = 0.01  # pre-reg kill K2 threshold

# =============================================================================
# DATA LOADING
# =============================================================================


def load_cell(con: duckdb.DuckDBPyConnection, cell: dict[str, Any]) -> pd.DataFrame:
    sess = cell["session"]
    sql = f"""
    SELECT o.trading_day, o.pnl_r, o.outcome,
           d.overnight_range, d.atr_20,
           d.orb_{sess}_high, d.orb_{sess}_low, d.orb_{sess}_vwap,
           d.orb_{sess}_break_dir, d.orb_{sess}_size
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = ?
      AND o.confirm_bars = ?
      AND o.rr_target = ?
      AND d.orb_{sess}_break_dir = ?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(
        sql,
        [cell["instrument"], sess, cell["orb_minutes"],
         ENTRY_MODEL, CONFIRM_BARS, cell["rr"], cell["direction"]],
    ).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["is_is"] = df["trading_day"] < HOLDOUT_SACRED_FROM
    df["is_oos"] = df["trading_day"] >= HOLDOUT_SACRED_FROM
    return df


# =============================================================================
# STATISTICS
# =============================================================================


def t_test_vs_zero(pnl: np.ndarray) -> tuple[float | None, float | None]:
    if len(pnl) < 2:
        return None, None
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    if std == 0.0:
        return None, None
    t = mean / (std / math.sqrt(len(pnl)))
    raw_p = float(2.0 * (1.0 - _sstats.t.cdf(abs(t), df=len(pnl) - 1)))
    return float(t), raw_p


def block_bootstrap_p(pnl: np.ndarray, block: int = 5, B: int = 10_000,
                      seed: int = 42) -> float | None:
    """Moving-block bootstrap under centered-H0 (mean subtracted).

    Null: true mean is 0. Test statistic: observed mean. Resample blocks
    of length `block` from mean-centered pnl, compute resampled means,
    two-sided p from proportion exceeding the observed absolute mean.

    Per backtesting-methodology.md historical-failure-log (2026-04-15:
    block bootstrap preserved joint structure), we resample the pnl
    ONLY; there is no mask here since the statistic is unconditional.
    """
    n = len(pnl)
    if n < block * 2:
        return None
    observed = float(abs(np.mean(pnl)))
    centered = pnl - np.mean(pnl)  # enforce H0: mean = 0
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    exceedances = 0
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        samples: list[np.ndarray] = []
        for s in starts:
            samples.append(centered[s:s + block])
        boot = np.concatenate(samples)[:n]
        if abs(float(np.mean(boot))) >= observed:
            exceedances += 1
    # Phipson & Smyth (2010) +1 correction
    return (exceedances + 1) / (B + 1)


def bh_fdr_qvalues(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg monotone q-values (matches statsmodels fdr_bh)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(pvalues)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(m)
    q = np.empty(m, dtype=float)
    running_min = float("inf")
    # iterate from largest rank down
    for i in range(m - 1, -1, -1):
        idx = order[i]
        p = pvalues[idx]
        raw = p * m / (i + 1)
        running_min = min(running_min, raw)
        q[idx] = min(1.0, running_min)
    return q.tolist()


def tautology_corr_with_orb_size(df: pd.DataFrame, fire_mask: np.ndarray,
                                 sess: str) -> float | None:
    """T0: Pearson corr between fire binary and orb_size over IS window."""
    sz_col = f"orb_{sess}_size"
    sz = df[sz_col].astype(float).values
    fire = np.asarray(fire_mask).astype(float)
    valid = ~np.isnan(sz)
    sz_v = sz[valid]
    fire_v = fire[valid]
    if fire_v.sum() == 0 or fire_v.sum() == len(fire_v) or np.std(sz_v) == 0:
        return None
    r = float(np.corrcoef(fire_v, sz_v)[0, 1])
    if math.isnan(r):
        return None
    return r


# =============================================================================
# CELL ANALYSIS
# =============================================================================


@dataclass
class CellResult:
    id: str
    instrument: str
    session: str
    orb_minutes: int
    rr: float
    direction: str
    filter_key: str

    # IS window
    n_is_base: int = 0
    expr_is_base: float | None = None
    n_is_on: int = 0
    n_is_off: int = 0
    expr_is_on: float | None = None
    expr_is_off: float | None = None
    wr_is_on: float | None = None
    wr_is_off: float | None = None
    std_is_on: float | None = None
    t_is: float | None = None
    raw_p_is: float | None = None
    bootstrap_p: float | None = None
    delta_is: float | None = None
    fire_rate_is: float | None = None
    tautology_corr: float | None = None
    wr_spread: float | None = None
    per_year_positive: int = 0
    per_year_total: int = 0
    years_breakdown: dict[int, dict[str, Any]] = field(default_factory=dict)

    # BH-FDR
    q_family: float | None = None
    passes_bh_family: bool = False

    # OOS descriptive
    n_oos_base: int = 0
    n_oos_on: int = 0
    expr_oos_on: float | None = None
    delta_oos: float | None = None
    dir_match: bool | None = None

    # Flags
    extreme_fire: bool = False
    tautology: bool = False
    arithmetic_only: bool = False
    positive_mean_floor: bool = False

    # Gate outcomes
    gate_results: dict[str, bool] = field(default_factory=dict)
    h1_pass: bool = False
    verdict: str = "UNKNOWN"

    # K2 harness cross-check
    k2_base_match: bool | None = None
    k2_on_match: bool | None = None


def analyze_cell(con: duckdb.DuckDBPyConnection, cell: dict[str, Any]) -> CellResult:
    df = load_cell(con, cell)
    res = CellResult(
        id=cell["id"], instrument=cell["instrument"], session=cell["session"],
        orb_minutes=cell["orb_minutes"], rr=cell["rr"], direction=cell["direction"],
        filter_key=cell["filter_key"],
    )

    if len(df) == 0:
        res.verdict = "NO_DATA"
        return res

    is_df = df[df["is_is"]].reset_index(drop=True)
    oos_df = df[df["is_oos"]].reset_index(drop=True)

    # IS unfiltered baseline
    res.n_is_base = len(is_df)
    if res.n_is_base > 0:
        res.expr_is_base = float(is_df["pnl_r"].mean())

    # K2 harness cross-check: IS unfiltered baseline vs pre-reg
    key_base = (cell["instrument"], cell["session"], cell["orb_minutes"],
                cell["rr"], cell["direction"], "unfiltered")
    if key_base in PREREG_BASELINES:
        exp = PREREG_BASELINES[key_base]
        if res.expr_is_base is not None:
            res.k2_base_match = abs(res.expr_is_base - exp["ExpR"]) < K2_TOLERANCE

    # Filter mask on IS
    fire_is = np.asarray(filter_signal(is_df, cell["filter_key"], cell["session"])).astype(bool)
    fire_is = fire_is & is_df["pnl_r"].notna().values
    on_pnl = is_df.loc[fire_is, "pnl_r"].values
    off_pnl = is_df.loc[~fire_is, "pnl_r"].values

    res.n_is_on = int(fire_is.sum())
    res.n_is_off = int((~fire_is).sum())

    if res.n_is_on > 0:
        res.expr_is_on = float(np.mean(on_pnl))
        res.wr_is_on = float(np.mean(is_df.loc[fire_is, "outcome"] == "win"))
        res.std_is_on = float(np.std(on_pnl, ddof=1)) if res.n_is_on > 1 else None
        res.fire_rate_is = res.n_is_on / res.n_is_base if res.n_is_base > 0 else None
        t, p = t_test_vs_zero(on_pnl)
        res.t_is, res.raw_p_is = t, p
        res.bootstrap_p = block_bootstrap_p(on_pnl, block=5, B=10_000, seed=42)
        res.positive_mean_floor = res.expr_is_on > 0

    if res.n_is_off > 0:
        res.expr_is_off = float(np.mean(off_pnl))
        res.wr_is_off = float(np.mean(is_df.loc[~fire_is, "outcome"] == "win"))

    if res.expr_is_on is not None and res.expr_is_base is not None:
        res.delta_is = res.expr_is_on - res.expr_is_base
    if res.wr_is_on is not None and res.wr_is_off is not None:
        res.wr_spread = res.wr_is_on - res.wr_is_off

    # K2 harness cross-check: IS on-filter ExpR vs pre-reg
    key_on = (cell["instrument"], cell["session"], cell["orb_minutes"],
              cell["rr"], cell["direction"], cell["filter_key"])
    if key_on in PREREG_BASELINES:
        exp = PREREG_BASELINES[key_on]
        if res.expr_is_on is not None:
            res.k2_on_match = abs(res.expr_is_on - exp["ExpR"]) < K2_TOLERANCE

    # T0 tautology
    res.tautology_corr = tautology_corr_with_orb_size(is_df, fire_is, cell["session"])
    res.tautology = (
        res.tautology_corr is not None and abs(res.tautology_corr) > 0.70
    )

    # Fire rate flag
    if res.fire_rate_is is not None:
        res.extreme_fire = res.fire_rate_is < 0.05 or res.fire_rate_is > 0.95

    # Arithmetic-only flag
    if (res.wr_spread is not None and res.delta_is is not None
            and abs(res.wr_spread) < 0.03 and abs(res.delta_is) > 0.10):
        res.arithmetic_only = True

    # Per-year stability
    is_df["_year"] = pd.to_datetime(is_df["trading_day"]).dt.year
    for yr in sorted(is_df["_year"].unique()):
        yr_mask = (is_df["_year"] == yr).values
        yr_fire = fire_is & yr_mask
        n = int(yr_fire.sum())
        if n == 0:
            res.years_breakdown[int(yr)] = {"n": 0, "expr": None, "positive": None}
            continue
        yr_expr = float(is_df.loc[yr_fire, "pnl_r"].mean())
        pos = yr_expr > 0
        res.years_breakdown[int(yr)] = {"n": n, "expr": yr_expr, "positive": pos}
        if n >= 10:  # require minimum sample per year for the positive check
            res.per_year_total += 1
            if pos:
                res.per_year_positive += 1

    # OOS descriptive
    res.n_oos_base = len(oos_df)
    if res.n_oos_base > 0:
        fire_oos = np.asarray(filter_signal(oos_df, cell["filter_key"], cell["session"])).astype(bool)
        fire_oos = fire_oos & oos_df["pnl_r"].notna().values
        res.n_oos_on = int(fire_oos.sum())
        if res.n_oos_on > 0:
            res.expr_oos_on = float(oos_df.loc[fire_oos, "pnl_r"].mean())
            expr_oos_base = float(oos_df["pnl_r"].mean())
            res.delta_oos = res.expr_oos_on - expr_oos_base
            if res.delta_is is not None and res.delta_oos is not None:
                res.dir_match = (
                    (res.delta_is > 0 and res.delta_oos > 0)
                    or (res.delta_is < 0 and res.delta_oos < 0)
                )

    return res


# =============================================================================
# GATE EVALUATION
# =============================================================================


def evaluate_gates(cells: list[CellResult]) -> None:
    # BH-FDR at K_family = 2 on raw_p_is
    pvals = [c.raw_p_is if c.raw_p_is is not None else 1.0 for c in cells]
    qs = bh_fdr_qvalues(pvals)
    for c, q in zip(cells, qs):
        c.q_family = q
        c.passes_bh_family = q < 0.05

    for c in cells:
        gates = {}
        gates["bh_pass_family"] = bool(c.passes_bh_family)
        gates["abs_t_IS_ge_3"] = bool(c.t_is is not None and abs(c.t_is) >= 3.0)
        gates["N_IS_on_ge_100"] = bool(c.n_is_on >= 100)
        gates["years_positive_ge_4_of_7"] = bool(
            c.per_year_total > 0 and c.per_year_positive / c.per_year_total >= 4.0 / 7.0
        )
        gates["bootstrap_p_lt_0.10"] = bool(
            c.bootstrap_p is not None and c.bootstrap_p < 0.10
        )
        gates["ExpR_on_IS_gt_0"] = bool(c.positive_mean_floor)

        # flags exclude from survivor
        gates["not_tautology"] = not c.tautology
        gates["not_extreme_fire"] = not c.extreme_fire
        gates["not_arithmetic_only"] = not c.arithmetic_only

        c.gate_results = gates
        c.h1_pass = all(
            gates[k] for k in [
                "bh_pass_family", "abs_t_IS_ge_3", "N_IS_on_ge_100",
                "years_positive_ge_4_of_7", "bootstrap_p_lt_0.10",
                "ExpR_on_IS_gt_0",
                "not_tautology", "not_extreme_fire", "not_arithmetic_only",
            ]
        )
        c.verdict = "CONTINUE" if c.h1_pass else "KILL"


# =============================================================================
# RENDER
# =============================================================================


def _fmt(x, p=4):
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, bool):
        return "Y" if x else "N"
    if isinstance(x, float):
        return f"{x:.{p}f}"
    return str(x)


def render(cells: list[CellResult]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# MES mirror of MNQ US_DATA_1000 VWAP_MID_ALIGNED long — K=2 scan")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mes-mnq-mirror-vwap-mid-aligned-us-data-1000-v1.yaml` (LOCKED, commit_sha=4fd08031)")
    lines.append(f"**Script:** `research/mes_mnq_mirror_v1_scan.py`")
    lines.append(f"**IS window:** `trading_day < {HOLDOUT_SACRED_FROM.isoformat()}` (Mode A, from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    n_continue = sum(1 for c in cells if c.verdict == "CONTINUE")
    n_kill = sum(1 for c in cells if c.verdict == "KILL")
    lines.append(f"Cells: {len(cells)} | CONTINUE: {n_continue} | KILL: {n_kill}")
    lines.append("")

    # K2 harness check summary
    k2_passes = []
    for c in cells:
        k2_passes.append(c.k2_base_match is not False and c.k2_on_match is not False)
    k2_ok = all(k2_passes)
    lines.append(f"**K2 harness cross-check:** {'PASS' if k2_ok else 'FAIL'}")
    if not k2_ok:
        lines.append("")
        lines.append("Harness failure details per cell follow. Treat this run as non-authoritative.")
    lines.append("")

    # Per-cell table
    lines.append("## Per-cell IS results")
    lines.append("")
    lines.append(
        "| Cell | N_on | ExpR_on | WR_on | ExpR_base | Δ_IS | t | raw_p | boot_p | q_family | years+ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        yr = f"{c.per_year_positive}/{c.per_year_total}" if c.per_year_total else "0/0"
        lines.append(
            f"| {c.id} | {c.n_is_on} | {_fmt(c.expr_is_on)} | {_fmt(c.wr_is_on, 3)} | "
            f"{_fmt(c.expr_is_base)} | {_fmt(c.delta_is)} | {_fmt(c.t_is, 3)} | "
            f"{_fmt(c.raw_p_is)} | {_fmt(c.bootstrap_p)} | {_fmt(c.q_family)} | {yr} |"
        )
    lines.append("")

    # Gate breakdown
    lines.append("## Gate breakdown")
    lines.append("")
    gate_keys = [
        "bh_pass_family", "abs_t_IS_ge_3", "N_IS_on_ge_100",
        "years_positive_ge_4_of_7", "bootstrap_p_lt_0.10",
        "ExpR_on_IS_gt_0", "not_tautology", "not_extreme_fire", "not_arithmetic_only",
    ]
    header = "| Cell | " + " | ".join(gate_keys) + " | Verdict |"
    sep = "|---|" + "|".join(["---"] * (len(gate_keys) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in cells:
        row = [c.id]
        for k in gate_keys:
            row.append("✓" if c.gate_results.get(k) else "✗")
        row.append(c.verdict)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Flags
    lines.append("## Flags")
    lines.append("")
    lines.append("| Cell | fire_rate | T0 |corr| | tautology | extreme_fire | arithmetic_only |")
    lines.append("|---|---:|---:|---|---|---|")
    for c in cells:
        lines.append(
            f"| {c.id} | {_fmt(c.fire_rate_is, 3)} | "
            f"{_fmt(abs(c.tautology_corr) if c.tautology_corr is not None else None, 3)} | "
            f"{_fmt(c.tautology)} | {_fmt(c.extreme_fire)} | {_fmt(c.arithmetic_only)} |"
        )
    lines.append("")

    # OOS descriptive
    lines.append("## OOS descriptive (NOT used to select or tune)")
    lines.append("")
    lines.append("| Cell | N_OOS_on | ExpR_on_OOS | Δ_OOS | dir_match |")
    lines.append("|---|---:|---:|---:|---|")
    for c in cells:
        lines.append(
            f"| {c.id} | {c.n_oos_on} | {_fmt(c.expr_oos_on)} | "
            f"{_fmt(c.delta_oos)} | {_fmt(c.dir_match)} |"
        )
    lines.append("")

    # Per-year breakdown
    lines.append("## Per-year IS breakdown")
    lines.append("")
    all_years = sorted({y for c in cells for y in c.years_breakdown})
    hdr = "| Cell | " + " | ".join(str(y) for y in all_years) + " |"
    sep = "|---|" + "|".join(["---:"] * len(all_years)) + "|"
    lines.append(hdr)
    lines.append(sep)
    for c in cells:
        row = [c.id]
        for y in all_years:
            b = c.years_breakdown.get(y)
            if not b:
                row.append("—")
            elif b["n"] < 10:
                row.append(f"N={b['n']}")
            else:
                sign = "+" if b["positive"] else "-"
                row.append(f"{sign}{_fmt(b['expr'], 3)} (N={b['n']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # K2 harness cross-check detail
    lines.append("## K2 harness cross-check detail")
    lines.append("")
    lines.append("| Cell | baseline_prereg | baseline_run | match | on_prereg | on_run | match |")
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for c in cells:
        base_key = (c.instrument, c.session, c.orb_minutes, c.rr, c.direction, "unfiltered")
        on_key = (c.instrument, c.session, c.orb_minutes, c.rr, c.direction, c.filter_key)
        base_pre = PREREG_BASELINES.get(base_key, {}).get("ExpR")
        on_pre = PREREG_BASELINES.get(on_key, {}).get("ExpR")
        lines.append(
            f"| {c.id} | {_fmt(base_pre)} | {_fmt(c.expr_is_base)} | {_fmt(c.k2_base_match)} | "
            f"{_fmt(on_pre)} | {_fmt(c.expr_is_on)} | {_fmt(c.k2_on_match)} |"
        )
    lines.append("")

    # Decision
    lines.append("## Decision")
    lines.append("")
    if k2_ok and n_continue == 0:
        lines.append("**Verdict: KILL per K1.** Zero of 2 cells pass all gate clauses. Honest negative evidence on MES cross-instrument portability of the MNQ VWAP_MID_ALIGNED US_DATA_1000 long signal. No re-runs with different thresholds (forbidden by pre-reg § execution_gate).")
    elif k2_ok and n_continue >= 1:
        lines.append(f"**Verdict: CONTINUE on {n_continue} cell(s).** Passing cells are validated-candidates; do NOT auto-promote to validated_setups — committee review required per pre_registered_criteria.md.")
    elif not k2_ok:
        lines.append("**Verdict: HARNESS FAIL per K2.** Canonical baseline cross-check diverged from pre-reg values. Run is non-authoritative until divergence is traced and fixed.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mes_mnq_mirror_v1_scan.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "No writes to validated_setups or experimental_strategies. No randomness in IS statistics. "
        "Bootstrap p is seeded (seed=42, B=10_000)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    print(f"Pre-reg: {PREREG_YAML.name}")
    print(f"IS window boundary: {HOLDOUT_SACRED_FROM}")
    print()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        cells = [analyze_cell(con, spec) for spec in CELLS]
    finally:
        con.close()

    evaluate_gates(cells)

    for c in cells:
        print(
            f"{c.id} N_on={c.n_is_on} ExpR_on={_fmt(c.expr_is_on)} "
            f"t={_fmt(c.t_is, 3)} p={_fmt(c.raw_p_is)} boot_p={_fmt(c.bootstrap_p)} "
            f"q_family={_fmt(c.q_family)} verdict={c.verdict}"
        )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(cells), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
