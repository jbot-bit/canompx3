#!/usr/bin/env python3
"""Mode A canonical re-validation + pre-registered criterion evaluation.

Phase 3 deliverable + Phase 2.1-extension. Two-layer output per active
validated_setups row:

LAYER 1 — Mode A re-validation.
Recomputes N, ExpR, WR, Sharpe, SD, per-year positivity under STRICT Mode A
IS (trading_day < HOLDOUT_SACRED_FROM) from canonical orb_outcomes +
daily_features, with filter applied via canonical
research.filter_utils.filter_signal. Flags any cell where the canonical
Mode A numbers diverge from the stored validated_setups values by more
than the tolerance thresholds (ΔN > 10% relative OR |ΔExpR| > 0.03
absolute OR |ΔSharpe| > 0.20 OR Mode-B-grandfathered last_trade_day).

LAYER 2 — Pre-registered criterion evaluation (added 2026-04-19).
For each lane under Mode A, compute pass/fail for:
  - Criterion 4 (Chordia t): t_IS >= 3.00 (with-theory) or 3.79 (no-theory)
  - Criterion 7 (Sample size): N_ModeA >= 100
  - Criterion 9 (Era stability): no doctrine era (2015-2019, 2020-2022,
    2023, 2024-2025, 2026) with aggregate-N >= 50 AND aggregate ExpR < -0.05

Locked thresholds from docs/institutional/pre_registered_criteria.md.
Not evaluated here: C1, C2, C3 (require pre-reg file + K), C5 (DSR is
informational-only per Amendment 2.1), C6 (WFE needs OOS Sharpe), C8
(2026 OOS sacred), C10 (data-era compat — filter-class specific), C11
(account-death MC — deployment-time), C12 (SR monitor — post-deploy).

Motivation: per research-truth-protocol.md § "Mode B grandfathered
validated_setups baselines" (2026-04-18), rows with last_trade_day in
[2026-01-01, 2026-04-08] were computed under the prior Mode B holdout
policy and include 2026 Q1 data which is now sacred Mode A OOS. The
stored expectancy_r values are therefore sample-inflated (larger N) and
may be stat-inflated (Mode A OOS included in Mode B IS).

This script writes NOTHING to validated_setups. It is a read-only audit
producing a canonical errata document.

Output: docs/audit/results/2026-04-19-mode-a-criterion-evaluation.md
(NOTE: earlier revision wrote to 2026-04-19-mode-a-revalidation-of-active-
setups.md; that doc is committed-frozen as historical. This script now
writes to the criterion-evaluation path for the extended output.)

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py
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

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from research.filter_utils import filter_signal

RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-19-mode-a-criterion-evaluation.md"
)

# Divergence flagging thresholds
N_RATIO_TOLERANCE = 0.10        # flag if |delta_N / stored_N| > 0.10
EXPR_ABS_TOLERANCE = 0.03       # flag if |delta_ExpR| > 0.03
SHARPE_ABS_TOLERANCE = 0.20     # flag if |delta_Sharpe| > 0.20

# Pre-registered criteria thresholds — LOCKED from
# `docs/institutional/pre_registered_criteria.md`. Any change requires an
# amendment per that file's § Amendment procedure. Do NOT tune here.
#
# Criterion 4 — Chordia t-statistic (pre_registered_criteria.md § Criterion 4):
#   t >= 3.00 with-theory (Harvey-Liu-Zhu 2015; grounding is currently
#     INDIRECT Tier 1 via Chordia p5 — see file note on promotion to DIRECT)
#   t >= 3.79 without-theory (Chordia et al 2018 verbatim Tier 1)
C4_T_WITH_THEORY: float = 3.00
C4_T_NO_THEORY: float = 3.79
#
# Criterion 7 — Sample size (pre_registered_criteria.md § Criterion 7):
#   N >= 100 trades for deployment eligibility (Harvey-Liu 2015 Exhibit 4).
C7_MIN_N: int = 100
#
# Criterion 9 — Era stability (pre_registered_criteria.md § Criterion 9):
#   No era with ExpR < -0.05 and N >= 50 trades.
#   "Era-split into (2015-2019, 2020-2022, 2023, 2024-2025, 2026)" per doctrine —
#   these are doctrine-specified groupings, NOT individual years.
C9_ERA_THRESHOLD: float = -0.05
C9_MIN_N_PER_ERA: int = 50
C9_ERAS: dict[str, tuple[int, int]] = {
    "2015-2019": (2015, 2019),
    "2020-2022": (2020, 2022),
    "2023": (2023, 2023),
    "2024-2025": (2024, 2025),
    "2026": (2026, 2026),  # sacred holdout — no Mode A IS data here by construction
}


@dataclass
class LaneRevalidation:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str | None
    direction: str

    # Stored (from validated_setups, potentially Mode-B grandfathered)
    stored_n: int = 0
    stored_expr: float | None = None
    stored_sharpe: float | None = None
    stored_wr: float | None = None
    stored_last_trade_day: date | None = None

    # Mode A canonical (recomputed)
    mode_a_n: int = 0
    mode_a_expr: float | None = None
    mode_a_sharpe: float | None = None
    mode_a_wr: float | None = None
    mode_a_sd: float | None = None   # per-trade std dev (needed for Criterion 4)

    # Divergence flags
    delta_n: int | None = None
    delta_n_ratio: float | None = None
    delta_expr: float | None = None
    delta_sharpe: float | None = None
    mode_b_contaminated: bool = False
    material_drift: bool = False
    drift_reasons: list[str] = field(default_factory=list)

    # Per-year Mode A breakdown (for WFE / sanity)
    years_positive: int = 0
    years_total: int = 0
    years_breakdown: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Pre-registered criteria evaluation (pre_registered_criteria.md § 4, 7, 9)
    # None = not evaluated (inputs missing); True/False = hard pass/fail.
    c4_t_stat: float | None = None
    c4_pass_with_theory: bool | None = None
    c4_pass_no_theory: bool | None = None
    c7_pass: bool | None = None
    c9_pass: bool | None = None
    # c9_violating_eras contains doctrine-era NAMES (e.g. "2020-2022"), not
    # year integers — see pre_registered_criteria.md § Criterion 9.
    c9_violating_eras: list[str] = field(default_factory=list)
    c9_era_aggregates: dict[str, dict[str, Any]] = field(default_factory=dict)
    criterion_failures: list[str] = field(default_factory=list)


def direction_from_execution_spec(spec: str | None) -> str:
    """Resolve long/short from execution_spec JSON or default to 'long'."""
    if not spec:
        return "long"
    if "short" in str(spec).lower():
        return "short"
    return "long"


def load_active_setups(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target,
               entry_model, confirm_bars, filter_type, sample_size,
               expectancy_r, sharpe_ann, win_rate, last_trade_day,
               execution_spec
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        ORDER BY instrument, orb_label, orb_minutes, rr_target, filter_type
        """
    ).fetchall()
    cols = [
        "strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target",
        "entry_model", "confirm_bars", "filter_type", "sample_size",
        "expectancy_r", "sharpe_ann", "win_rate", "last_trade_day",
        "execution_spec",
    ]
    return [dict(zip(cols, r)) for r in rows]


def compute_mode_a(
    con: duckdb.DuckDBPyConnection, spec: dict[str, Any]
) -> tuple[int, float | None, float | None, float | None, dict[int, dict[str, Any]], float | None]:
    """Recompute (N, ExpR, Sharpe_ann, WR, year_breakdown, SD) under Mode A IS
    (trading_day < HOLDOUT_SACRED_FROM) using canonical filter delegation.

    Returns the 6-tuple where the last element is per-trade std dev — exposed
    so Criterion 4 (Chordia t-stat) can be computed downstream without a
    second pass over the data.
    """
    sess = spec["orb_label"]
    direction = direction_from_execution_spec(spec.get("execution_spec"))
    # SELECT d.* because different canonical filters look up different columns
    # by canonical name (e.g., CostRatioFilter needs orb_{sess}_size + symbol;
    # VWAPBreakDirectionFilter needs orb_{sess}_vwap + orb_{sess}_high/low;
    # OvernightRangeAbsFilter needs overnight_range). Loading all daily_features
    # columns is the only way to remain filter-agnostic without hand-coding a
    # column-needs map per filter. orb_outcomes columns pnl_r/outcome/symbol
    # are selected explicitly since the JOIN aliases could otherwise collide.
    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol,
               d.*
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
          AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    df = con.execute(
        sql,
        [
            spec["instrument"], sess, spec["orb_minutes"],
            spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
            direction, HOLDOUT_SACRED_FROM,
        ],
    ).df()
    if len(df) == 0:
        return 0, None, None, None, {}, None

    filter_type = spec.get("filter_type")

    # CrossAssetATRFilter requires cross_atr_{source}_pct which is NOT in
    # daily_features schema — it is injected at discovery/fitness time by
    # _inject_cross_asset_atrs (canonical impl at trading_app/strategy_discovery.py:978).
    # Mirror that injection here so X_MES_ATR60 et al. evaluate correctly.
    if filter_type and filter_type in ALL_FILTERS:
        filt_obj = ALL_FILTERS[filter_type]
        if isinstance(filt_obj, CrossAssetATRFilter):
            source = filt_obj.source_instrument
            if source != spec["instrument"]:
                src_rows = con.execute(
                    """SELECT trading_day, atr_20_pct FROM daily_features
                       WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL""",
                    [source],
                ).fetchall()
                src_map: dict[date, float] = {}
                for td, pct in src_rows:
                    key = td.date() if hasattr(td, "date") else td
                    src_map[key] = float(pct)
                col = f"cross_atr_{source}_pct"
                df[col] = df["trading_day"].apply(
                    lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
                )

    if filter_type and filter_type != "UNFILTERED":
        try:
            fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
        except Exception as e:
            print(f"  [warn] filter_signal failed for {filter_type} on {sess}: {e}")
            return 0, None, None, None, {}, None
        df_on = df[fire].reset_index(drop=True)
    else:
        df_on = df

    if len(df_on) == 0:
        return 0, None, None, None, {}, None

    pnl = df_on["pnl_r"].astype(float).to_numpy()
    n = len(pnl)
    expr = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1)) if n > 1 else None
    sharpe_per_trade = expr / std if std and std > 0 else None
    wr = float(np.mean(df_on["outcome"].astype(str) == "win"))

    # Annualize Sharpe — conservative: assume ~250 trading days × (trades/day ≈ 1 per eligible day)
    # Per-cell annualization: use observed trades/year estimate
    df_on["_year"] = pd.to_datetime(df_on["trading_day"]).dt.year
    years_sorted = sorted(df_on["_year"].unique())
    if sharpe_per_trade is not None and len(years_sorted) > 0:
        trades_per_year = n / len(years_sorted)
        sharpe_ann = sharpe_per_trade * math.sqrt(trades_per_year)
    else:
        sharpe_ann = None

    year_break: dict[int, dict[str, Any]] = {}
    for yr in years_sorted:
        yr_pnl = df_on.loc[df_on["_year"] == yr, "pnl_r"].astype(float).to_numpy()
        if len(yr_pnl) == 0:
            continue
        yr_expr = float(np.mean(yr_pnl))
        year_break[int(yr)] = {
            "n": len(yr_pnl),
            "expr": yr_expr,
            "positive": yr_expr > 0,
        }

    return n, expr, sharpe_ann, wr, year_break, std


def compute_criterion_flags(rv: LaneRevalidation) -> None:
    """Apply pre_registered_criteria.md Criteria 4, 7, 9 to a revalidated lane.

    Mutates rv in place. Fields stay None when inputs are missing so downstream
    rendering can distinguish "not evaluated" (None) from "evaluated and failed"
    (False).

    Criterion 4 — Chordia t-statistic (pre_registered_criteria.md § Criterion 4):
        Computed: t = ExpR_on_IS / (sd / sqrt(N_on_IS)).
        Report BOTH thresholds per doc: t >= 3.00 (with-theory) and t >= 3.79
        (no-theory). Consumer decides which applies based on lane theory support.
        The doc flags the 3.00 grounding as INDIRECT Tier 1 (via Chordia p5
        referencing Harvey-Liu-Zhu 2015); that note is propagated in the
        rendered output but does not block computation here.

    Criterion 7 — Sample size (pre_registered_criteria.md § Criterion 7):
        PASS if mode_a_n >= C7_MIN_N (Harvey-Liu 2015 Exhibit 4 — 100 deployable).

    Criterion 9 — Era stability (pre_registered_criteria.md § Criterion 9):
        PASS if no year has N >= C9_MIN_N_PER_ERA AND ExpR < C9_ERA_THRESHOLD.
        Years with N < 50 are exempt (not enough data to judge).
    """
    failures: list[str] = []

    # C4 — Chordia t-statistic from Mode A (ExpR, sd, N)
    if (
        rv.mode_a_expr is not None
        and rv.mode_a_sd is not None
        and rv.mode_a_sd > 0
        and rv.mode_a_n > 0
    ):
        se = rv.mode_a_sd / math.sqrt(rv.mode_a_n)
        if se > 0:
            rv.c4_t_stat = rv.mode_a_expr / se
            rv.c4_pass_with_theory = rv.c4_t_stat >= C4_T_WITH_THEORY
            rv.c4_pass_no_theory = rv.c4_t_stat >= C4_T_NO_THEORY
            if not rv.c4_pass_with_theory:
                failures.append(
                    f"C4 t={rv.c4_t_stat:.2f} < {C4_T_WITH_THEORY} (with-theory)"
                )

    # C7 — Sample size
    if rv.mode_a_n > 0:
        rv.c7_pass = rv.mode_a_n >= C7_MIN_N
        if not rv.c7_pass:
            failures.append(f"C7 N={rv.mode_a_n} < {C7_MIN_N}")

    # C9 — Era stability (aggregate years into doctrine eras before testing)
    if rv.years_breakdown:
        rv.c9_era_aggregates = _aggregate_years_to_eras(rv.years_breakdown)
        violating_eras: list[str] = []
        for era_name, agg in rv.c9_era_aggregates.items():
            if (
                agg["n"] >= C9_MIN_N_PER_ERA
                and agg["expr"] < C9_ERA_THRESHOLD
            ):
                violating_eras.append(era_name)
        rv.c9_violating_eras = violating_eras  # preserves doctrine era ordering
        rv.c9_pass = len(violating_eras) == 0
        if not rv.c9_pass:
            failures.append(
                f"C9 era(s) {rv.c9_violating_eras} ExpR<{C9_ERA_THRESHOLD} "
                f"(era-N>={C9_MIN_N_PER_ERA})"
            )

    rv.criterion_failures = failures


def _aggregate_years_to_eras(
    years_breakdown: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-year {n, expr} buckets into the doctrine eras defined by
    C9_ERAS. Returns a dict keyed by era name in doctrine order.

    Weighted mean formula: era_expr = sum(year.n * year.expr) / sum(year.n)
    This reconstructs the true population ExpR over the era from per-year
    aggregates without needing to re-query raw trades.

    Only eras with at least one contributing year are returned (empty eras
    are omitted; their absence is treated as "not evaluable" downstream
    equivalent to N<50).
    """
    result: dict[str, dict[str, Any]] = {}
    for era_name, (start, end) in C9_ERAS.items():
        total_n = 0
        sum_pnl = 0.0
        for yr, bucket in years_breakdown.items():
            if start <= int(yr) <= end:
                total_n += int(bucket["n"])
                sum_pnl += int(bucket["n"]) * float(bucket["expr"])
        if total_n > 0:
            result[era_name] = {
                "n": total_n,
                "expr": sum_pnl / total_n,
                "start_year": start,
                "end_year": end,
            }
    return result


def classify_divergence(rv: LaneRevalidation) -> None:
    reasons: list[str] = []
    if rv.stored_n > 0 and rv.mode_a_n > 0:
        rv.delta_n = rv.mode_a_n - rv.stored_n
        rv.delta_n_ratio = (rv.mode_a_n - rv.stored_n) / rv.stored_n
        if abs(rv.delta_n_ratio) > N_RATIO_TOLERANCE:
            reasons.append(f"|ΔN/N|={abs(rv.delta_n_ratio):.2f}>{N_RATIO_TOLERANCE}")
    if rv.stored_expr is not None and rv.mode_a_expr is not None:
        rv.delta_expr = rv.mode_a_expr - rv.stored_expr
        if abs(rv.delta_expr) > EXPR_ABS_TOLERANCE:
            reasons.append(f"|ΔExpR|={abs(rv.delta_expr):.3f}>{EXPR_ABS_TOLERANCE}")
    if rv.stored_sharpe is not None and rv.mode_a_sharpe is not None:
        rv.delta_sharpe = rv.mode_a_sharpe - rv.stored_sharpe
        if abs(rv.delta_sharpe) > SHARPE_ABS_TOLERANCE:
            reasons.append(f"|ΔSharpe|={abs(rv.delta_sharpe):.2f}>{SHARPE_ABS_TOLERANCE}")

    # Mode-B contamination indicator: last_trade_day after holdout boundary
    if rv.stored_last_trade_day is not None:
        if rv.stored_last_trade_day >= HOLDOUT_SACRED_FROM:
            rv.mode_b_contaminated = True
            reasons.append(
                f"last_trade_day={rv.stored_last_trade_day} >= {HOLDOUT_SACRED_FROM} "
                "(Mode-B grandfathered)"
            )

    rv.drift_reasons = reasons
    rv.material_drift = bool(reasons)


def _fmt(x: Any, places: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{places}f}"
    return str(x)


def _fmt_pass(x: bool | None) -> str:
    """Render a criterion pass/fail cell: None = not evaluated, True = PASS, False = FAIL."""
    if x is None:
        return "—"
    return "PASS" if x else "FAIL"


def render(results: list[LaneRevalidation]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_total = len(results)
    n_drift = sum(1 for r in results if r.material_drift)
    n_mode_b = sum(1 for r in results if r.mode_b_contaminated)
    n_c4_fail_with_theory = sum(1 for r in results if r.c4_pass_with_theory is False)
    n_c7_fail = sum(1 for r in results if r.c7_pass is False)
    n_c9_fail = sum(1 for r in results if r.c9_pass is False)
    n_any_crit_fail = sum(1 for r in results if r.criterion_failures)

    lines: list[str] = []
    lines.append("# Mode A canonical re-validation + pre-registered criterion evaluation")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Script:** `research/mode_a_revalidation_active_setups.py`")
    lines.append(f"**IS boundary:** `trading_day < {HOLDOUT_SACRED_FROM}` (Mode A)")
    lines.append(f"**Canonical filter source:** `research.filter_utils.filter_signal` → `trading_app.config.ALL_FILTERS`")
    lines.append(f"**Criteria source:** `docs/institutional/pre_registered_criteria.md` § Criteria 4, 7, 9 (LOCKED thresholds)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total active lanes re-validated: **{n_total}**")
    lines.append(f"- Lanes with material Mode A drift (|ΔN/N|>10% OR |ΔExpR|>0.03 OR |ΔSharpe|>0.20 OR Mode-B contaminated): **{n_drift}**")
    lines.append(f"- Lanes with last_trade_day >= 2026-01-01 (Mode-B grandfathered): **{n_mode_b}**")
    lines.append("")
    n_c4_fail_no_theory = sum(1 for r in results if r.c4_pass_no_theory is False)
    lines.append("### Pre-registered criterion failures under Mode A (this is the NEW signal)")
    lines.append("")
    lines.append(f"- Lanes failing Criterion 4 at the WITH-THEORY bar (t >= {C4_T_WITH_THEORY}): **{n_c4_fail_with_theory}**")
    lines.append(f"- Lanes failing Criterion 4 at the NO-THEORY bar (t >= {C4_T_NO_THEORY}): **{n_c4_fail_no_theory}**")
    lines.append("  — BOTH counts reported because `pre_registered_criteria.md` § Criterion 4")
    lines.append("  requires per-lane theory-citation evidence before the softer 3.00 bar applies;")
    lines.append("  this report does NOT verify per-lane theory citations. Downstream committee")
    lines.append("  must confirm theory grounding before citing the with-theory count as binding.")
    lines.append(f"- Lanes failing Criterion 7 (Mode A N >= {C7_MIN_N}): **{n_c7_fail}**")
    lines.append(f"- Lanes failing Criterion 9 (no doctrine era with N>={C9_MIN_N_PER_ERA} & ExpR<{C9_ERA_THRESHOLD}): **{n_c9_fail}**")
    lines.append(f"  — Eras per doctrine: {list(C9_ERAS.keys())}")
    lines.append(f"- Lanes failing AT LEAST ONE of {{C4_with_theory, C7, C9}} under Mode A: **{n_any_crit_fail}**")
    lines.append("")
    lines.append("Failing one or more of C4/C7/C9 under Mode A means the lane is not")
    lines.append("deployment-eligible under the locked doctrine without either (a) explicit")
    lines.append("documented theory citation + promotion of the 3.00 with-theory grounding")
    lines.append("to DIRECT (per pre_registered_criteria.md § Criterion 4), OR (b) re-validation")
    lines.append("on a larger population, OR (c) explicit regime-gating to exclude the weak era.")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- N ratio: |ΔN / stored_N| > {N_RATIO_TOLERANCE} → drift flag")
    lines.append(f"- ExpR absolute: |ΔExpR| > {EXPR_ABS_TOLERANCE} → drift flag")
    lines.append(f"- Sharpe absolute: |ΔSharpe_ann| > {SHARPE_ABS_TOLERANCE} → drift flag")
    lines.append(f"- Mode-B contaminated: `last_trade_day >= {HOLDOUT_SACRED_FROM}` → drift flag")
    lines.append(f"- Criterion 4: t_IS >= {C4_T_WITH_THEORY} (with-theory) or >= {C4_T_NO_THEORY} (no-theory)")
    lines.append(f"- Criterion 7: N_ModeA >= {C7_MIN_N}")
    lines.append(f"- Criterion 9: no era with N >= {C9_MIN_N_PER_ERA} and ExpR < {C9_ERA_THRESHOLD}")
    lines.append("")
    lines.append("**Note on Criterion 4 grounding:** `pre_registered_criteria.md` § Criterion 4")
    lines.append(f"flags the {C4_T_WITH_THEORY} with-theory threshold as currently INDIRECT Tier 1")
    lines.append("(via Chordia et al 2018 p5 referencing Harvey-Liu-Zhu 2015). The doc requires")
    lines.append("promotion to DIRECT before any 3.00 ≤ t < 3.79 with-theory candidate is")
    lines.append(f"accepted. Lanes with t in [{C4_T_WITH_THEORY}, {C4_T_NO_THEORY}) are flagged")
    lines.append("below as `c4_with_theory_PASS` / `c4_no_theory_FAIL` so the caveat is explicit.")
    lines.append("")
    lines.append("### How to read this doc (signal vs tripwire)")
    lines.append("")
    lines.append("**Material-drift flag = INFORMATIONAL TRIPWIRE, not retirement signal.**")
    lines.append("Thresholds (10% N / 0.03 ExpR / 0.20 Sharpe / Mode-B grandfather) are")
    lines.append("engineering tolerances, NOT literature-grounded — they detect any")
    lines.append("Mode-A-vs-stored divergence so downstream readers know stored values are")
    lines.append("partially computed on a different IS window. Under Amendment 2.7")
    lines.append("(Mode B → Mode A, 2026-04-08), stored `validated_setups` values pre-date")
    lines.append("the sacred-holdout boundary and will ALWAYS flag material drift on")
    lines.append("recompute — so a 38/38 drift count is EXPECTED, not alarming.")
    lines.append("")
    lines.append("**Retirement-relevant signal = Criterion 4, 7, 9 pass/fail columns below.**")
    lines.append("Those thresholds ARE locked in `pre_registered_criteria.md` — those")
    lines.append("failures are doctrine-grounded and actionable. A lane failing one or")
    lines.append("more of {C4_with_theory, C7, C9} under Mode A is not deployment-eligible")
    lines.append("under the locked doctrine without explicit re-validation, theory-citation")
    lines.append("promotion, or regime-gating.")
    lines.append("")
    lines.append("Treat the Mode A column as the canonical truth going forward; the")
    lines.append("validated_setups rows themselves are NOT mutated by this audit.")
    lines.append("")
    lines.append("## Per-lane re-validation + criterion evaluation")
    lines.append("")
    lines.append("Columns: `t_IS` = Chordia t-stat from Mode A (ExpR/(sd/√N)); `C4_wth` = pass with-theory (t≥3.00); `C4_nth` = pass no-theory (t≥3.79); `C7` = N≥100; `C9` = era stability (no year ExpR<-0.05 with N≥50).")
    lines.append("")
    lines.append("| Instr | Session | Om | RR | Filter | Dir | Stored N / Mode-A N | ΔN/N | Stored ExpR / Mode-A ExpR | ΔExpR | Stored Sh / Mode-A Sh | ΔSh | Yrs+ | Mode-B | t_IS | C4_wth | C4_nth | C7 | C9 | Drift | Crit Fails |")
    lines.append("|---|---|---:|---:|---|---|---|---:|---|---:|---|---:|---:|---|---:|---|---|---|---|---|---|")
    for r in results:
        drift = "DRIFT" if r.material_drift else ""
        mb = "Y" if r.mode_b_contaminated else "N"
        yrs = f"{r.years_positive}/{r.years_total}" if r.years_total else "—"
        t_str = _fmt(r.c4_t_stat, 2)
        c4w = _fmt_pass(r.c4_pass_with_theory)
        c4n = _fmt_pass(r.c4_pass_no_theory)
        c7 = _fmt_pass(r.c7_pass)
        c9 = _fmt_pass(r.c9_pass)
        crit_fails = "; ".join(r.criterion_failures) if r.criterion_failures else ""
        lines.append(
            f"| {r.instrument} | {r.orb_label} | {r.orb_minutes} | {r.rr_target} | "
            f"{r.filter_type or 'UNFILTERED'} | {r.direction} | "
            f"{r.stored_n} / {r.mode_a_n} | {_fmt(r.delta_n_ratio, 2)} | "
            f"{_fmt(r.stored_expr)} / {_fmt(r.mode_a_expr)} | {_fmt(r.delta_expr)} | "
            f"{_fmt(r.stored_sharpe, 2)} / {_fmt(r.mode_a_sharpe, 2)} | {_fmt(r.delta_sharpe, 2)} | "
            f"{yrs} | {mb} | {t_str} | {c4w} | {c4n} | {c7} | {c9} | {drift} | {crit_fails} |"
        )
    lines.append("")
    lines.append("## Criterion-failing lanes — prioritised view")
    lines.append("")
    crit_failing = [r for r in results if r.criterion_failures]
    if not crit_failing:
        lines.append("_No active lane fails C4/C7/C9 under Mode A._")
    else:
        # Sort by severity proxy: first by C7 fails (hardest to fix), then C9, then C4 margin
        def _severity(r: LaneRevalidation) -> tuple[int, int, float]:
            c7_weight = 0 if r.c7_pass is False else 1
            c9_weight = 0 if r.c9_pass is False else 1
            t_gap = (r.c4_t_stat or 0.0) - C4_T_WITH_THEORY  # more negative = worse
            return (c7_weight, c9_weight, t_gap)
        for r in sorted(crit_failing, key=_severity):
            lines.append(f"### {r.instrument} {r.orb_label} O{r.orb_minutes} RR{r.rr_target} {r.filter_type or 'UNFILTERED'} {r.direction}")
            lines.append(f"- `strategy_id`: `{r.strategy_id}`")
            lines.append(f"- Mode A: N={r.mode_a_n} ExpR={_fmt(r.mode_a_expr)} t_IS={_fmt(r.c4_t_stat, 2)} Sh_ann={_fmt(r.mode_a_sharpe, 2)}")
            lines.append(f"- Criterion failures under Mode A: {'; '.join(r.criterion_failures)}")
            if r.c9_violating_eras:
                era_detail = " ".join(
                    f"{era_name}:{_fmt(r.c9_era_aggregates[era_name]['expr'], 3)}(N={r.c9_era_aggregates[era_name]['n']})"
                    for era_name in r.c9_violating_eras
                    if era_name in r.c9_era_aggregates
                )
                lines.append(f"- Violating eras (aggregate): {era_detail}")
            lines.append("")
    lines.append("")
    lines.append("## Materially-drifted lanes — detail")
    lines.append("")
    for r in [x for x in results if x.material_drift]:
        lines.append(f"### {r.instrument} {r.orb_label} O{r.orb_minutes} RR{r.rr_target} {r.filter_type or 'UNFILTERED'} {r.direction}")
        lines.append(f"- `strategy_id`: `{r.strategy_id}`")
        lines.append(f"- Stored: N={r.stored_n} ExpR={_fmt(r.stored_expr)} Sharpe_ann={_fmt(r.stored_sharpe, 2)} WR={_fmt(r.stored_wr, 3)} last_trade_day={r.stored_last_trade_day}")
        lines.append(f"- Mode A: N={r.mode_a_n} ExpR={_fmt(r.mode_a_expr)} Sharpe_ann={_fmt(r.mode_a_sharpe, 2)} WR={_fmt(r.mode_a_wr, 3)}")
        lines.append(f"- Drift reasons: {', '.join(r.drift_reasons)}")
        if r.years_total:
            yr_str = " ".join(
                f"{yr}:{'+' if b['positive'] else '-'}{_fmt(b['expr'], 3)}(N={b['n']})"
                for yr, b in sorted(r.years_breakdown.items())
            )
            lines.append(f"- Mode-A per-year: {yr_str}")
        lines.append("")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py")
    lines.append("```")
    lines.append("")
    lines.append("No writes to validated_setups or experimental_strategies. Output is this")
    lines.append("markdown document only. Numbers reproduce exactly on the same DB state.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        active = load_active_setups(con)
        print(f"Loaded {len(active)} active validated_setups")

        results: list[LaneRevalidation] = []
        for i, spec in enumerate(active, 1):
            direction = direction_from_execution_spec(spec.get("execution_spec"))
            n, expr, sharpe_ann, wr, year_break, sd = compute_mode_a(con, spec)
            yrs_pos = sum(1 for b in year_break.values() if b["positive"] and b["n"] >= 10)
            yrs_tot = sum(1 for b in year_break.values() if b["n"] >= 10)

            rv = LaneRevalidation(
                strategy_id=spec["strategy_id"],
                instrument=spec["instrument"],
                orb_label=spec["orb_label"],
                orb_minutes=spec["orb_minutes"],
                rr_target=spec["rr_target"],
                entry_model=spec["entry_model"],
                confirm_bars=spec["confirm_bars"],
                filter_type=spec["filter_type"],
                direction=direction,
                stored_n=spec["sample_size"] or 0,
                stored_expr=spec["expectancy_r"],
                stored_sharpe=spec["sharpe_ann"],
                stored_wr=spec["win_rate"],
                stored_last_trade_day=spec["last_trade_day"],
                mode_a_n=n,
                mode_a_expr=expr,
                mode_a_sharpe=sharpe_ann,
                mode_a_wr=wr,
                mode_a_sd=sd,
                years_positive=yrs_pos,
                years_total=yrs_tot,
                years_breakdown=year_break,
            )
            classify_divergence(rv)
            compute_criterion_flags(rv)
            results.append(rv)

            flag_str = "!!" if rv.material_drift else "  "
            print(
                f"  {flag_str} {i:2d}/{len(active)} {rv.instrument} {rv.orb_label} "
                f"O{rv.orb_minutes} RR{rv.rr_target} {rv.filter_type or 'UNF':<22} "
                f"N={rv.stored_n:>4}/{rv.mode_a_n:<4} "
                f"ExpR={_fmt(rv.stored_expr, 3)}/{_fmt(rv.mode_a_expr, 3)}"
            )
    finally:
        con.close()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(results), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    drift_count = sum(1 for r in results if r.material_drift)
    print(f"Material drift lanes: {drift_count} / {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
