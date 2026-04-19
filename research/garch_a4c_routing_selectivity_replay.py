"""Stage 1 replay for the locked A4c routing-selectivity allocator hypothesis.

Pre-registration: docs/audit/hypotheses/2026-04-17-garch-a4c-routing-selectivity.yaml
Design doc:      docs/plans/2026-04-17-garch-a4c-routing-selectivity-design.md
Audit grounding: docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md
Framing commit:  1a721e92

Purpose:
  Test whether the locked A4b pre-entry garch composite, re-scored on a
  dimension-neutral routing/selectivity metric (R per filled slot-day),
  improves the canonical shelf allocator over a random-uniform null and a
  trailing-Sharpe comparator, on two independently binding scarcity
  surfaces (A: raw slots @ 5, B: rho-survivor slots @ 3).

Harness:
  - Reuses A4b primitives (_compute_scores, _build_histories, etc.)
  - 5 ranking policies x 2 surfaces = 10 IS replays
  - MANDATORY positive-control gate: runs before candidate math, aborts if fails
  - Destruction shuffle control on the candidate only
  - 2026 OOS: descriptive-only mirror per surface per policy
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from research.garch_a4b_binding_budget_replay import (
    FIXED_STOP_MULTIPLIER,
    IS_START_MONTH,
    LaneScore,
    StrategyHistory,
    _annualized_sharpe,
    _build_garch_pct_by_day,
    _build_histories,
    _build_ranked_allocation,
    _build_session_regime_cache,
    _calendar_days,
    _candidate_garch_expr,
    _compute_scores,
    _first_trading_days_by_month,
    _jaccard_distance,
    _load_strategy_meta,
    _max_drawdown,
    _next_month,
    _pairwise_correlation_as_of,
    _shuffle_garch_pct,
)
from trading_app.lane_allocator import (
    _effective_annual_r,
    build_allocation,
    compute_orb_size_stats,
)

HYPOTHESIS_FILE = "docs/audit/hypotheses/2026-04-17-garch-a4c-routing-selectivity.yaml"
DESIGN_DOC = "docs/plans/2026-04-17-garch-a4c-routing-selectivity-design.md"
AUDIT_REF = "docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md"
FRAMING_COMMIT = "1a721e92"

OUTPUT_MD = Path("docs/audit/results/2026-04-17-garch-a4c-routing-selectivity-replay.md")
OUTPUT_JSON = Path("research/output/garch_a4c_routing_selectivity_replay.json")

# IS window (inherited from A4b)
IS_END_MONTH = date(2025, 12, 1)
HOLDOUT_BOUNDARY = date(2026, 1, 1)

# Pre-declared constants (LOCKED per hypothesis — no post-hoc tuning)
MIN_LIFT_PER_FILL = 0.01
MIN_SHARPE_LIFT = 0.05
MAX_DD_INFLATION = 1.20
SELECTION_CHURN_CAP = 0.50
BIND_PASS_RATIO_GATE = 0.80

# Inherited A4b candidate params (locked, verbatim)
GARCH_HIGH_THRESHOLD = 70.0
MIN_GARCH_HIGH_N = 20
WEIGHT_GARCH = 0.5
WEIGHT_BASELINE = 0.5
SHUFFLE_SEED = 20260417

# DD scalar (inherited from A4b bulenox_50k)
MAX_DD_DOLLARS = 2500.0

# Surfaces: (name, max_slots)
SURFACES: tuple[tuple[str, int], ...] = (
    ("A_raw_slots", 5),
    ("B_rho_survivor_slots", 3),
)

# Rankers evaluated per surface
RANKER_PRIMARY_NULL = "RANDOM_UNIFORM_UNDER_BINDING"
RANKER_SECONDARY = "TRAILING_SHARPE"
RANKER_POSITIVE_CONTROL = "POSITIVE_CONTROL_TRAILING_EXPR"
RANKER_CANDIDATE = "CANDIDATE_GARCH"
RANKER_DESTRUCTION = "DESTRUCTION_SHUFFLE"

ALL_RANKERS = (
    RANKER_PRIMARY_NULL,
    RANKER_SECONDARY,
    RANKER_POSITIVE_CONTROL,
    RANKER_CANDIDATE,
    RANKER_DESTRUCTION,
)


# ---------------------------------------------------------------------------
# Ranker helpers — produce {strategy_id: float | None} from LaneScores
# ---------------------------------------------------------------------------


def _values_random_uniform(
    scores: list[LaneScore],
    rebalance_date: date,
) -> dict[str, float | None]:
    """Seeded random value per deployable lane.

    Seed rule: SHUFFLE_SEED + rebalance_date.toordinal() (matches hypothesis).
    Deterministic — same seed produces same order on re-run.
    """
    rng = random.Random(SHUFFLE_SEED + rebalance_date.toordinal())
    return {s.strategy_id: rng.random() for s in scores}


def _values_trailing_sharpe(scores: list[LaneScore]) -> dict[str, float | None]:
    """Ranking key = sharpe_ann_adj (fallback 0.0 when None or missing)."""
    out: dict[str, float | None] = {}
    for s in scores:
        v = getattr(s, "sharpe_ann_adj", None)
        if v is None:
            v = 0.0
        out[s.strategy_id] = float(v)
    return out


def _values_positive_control(scores: list[LaneScore]) -> dict[str, float | None]:
    """Ranking key = trailing_expr (known-good canonical reference)."""
    return {s.strategy_id: float(s.trailing_expr) for s in scores}


def _values_candidate_garch(
    scores: list[LaneScore],
    histories: dict[str, StrategyHistory],
    rebalance_date: date,
    gp_by_day: dict[tuple[str, date], float | None],
) -> dict[str, float | None]:
    """Locked A4b composite: w1*trailing_expr_garch_high + w2*_effective_annual_r."""
    out: dict[str, float | None] = {}
    for s in scores:
        expr, _n = _candidate_garch_expr(histories[s.strategy_id], rebalance_date, gp_by_day)
        if expr is None:
            out[s.strategy_id] = None
        else:
            out[s.strategy_id] = round(WEIGHT_GARCH * expr + WEIGHT_BASELINE * _effective_annual_r(s), 6)
    return out


def _ranking_values_for_policy(
    policy: str,
    scores: list[LaneScore],
    histories: dict[str, StrategyHistory],
    rebalance_date: date,
    gp_by_day: dict[tuple[str, date], float | None],
    gp_shuffled: dict[tuple[str, date], float | None],
) -> tuple[dict[str, float | None], bool]:
    """Return (ranking_values, exclude_none_ranks) for the given policy."""
    if policy == RANKER_PRIMARY_NULL:
        return _values_random_uniform(scores, rebalance_date), False
    if policy == RANKER_SECONDARY:
        return _values_trailing_sharpe(scores), False
    if policy == RANKER_POSITIVE_CONTROL:
        return _values_positive_control(scores), False
    if policy == RANKER_CANDIDATE:
        return _values_candidate_garch(scores, histories, rebalance_date, gp_by_day), True
    if policy == RANKER_DESTRUCTION:
        return _values_candidate_garch(scores, histories, rebalance_date, gp_shuffled), True
    raise ValueError(f"unknown policy: {policy}")


# ---------------------------------------------------------------------------
# Forward-window accounting with FILL counting
# ---------------------------------------------------------------------------


def _forward_window_accounting(
    selected_ids: list[str],
    histories: dict[str, StrategyHistory],
    calendar_days: list[date],
) -> tuple[list[float], int]:
    """Return (daily_pnl_series, total_fill_count).

    fill = one (selected_lane, day) pair where that lane had non-zero pnl.
    daily_pnl = sum across all selected lanes for each calendar day.
    """
    daily: list[float] = []
    fills = 0
    for day in calendar_days:
        pnl = 0.0
        for sid in selected_ids:
            day_pnl = histories[sid].daily_pnl_r.get(day, 0.0)
            if day_pnl != 0.0:
                fills += 1
            pnl += day_pnl
        daily.append(round(pnl, 6))
    return daily, fills


# ---------------------------------------------------------------------------
# Policy-per-surface replay (one policy x one surface over IS or OOS window)
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    policy: str
    surface: str
    max_slots: int
    window: str  # "IS" or "OOS"
    total_r: float
    total_fills: int
    r_per_fill: float
    sharpe: float
    dd: float
    annualized_r: float
    mean_hit_rate_per_day: float
    trading_days_covered: int
    monthly_outputs: list[dict]
    daily_pnl: list[float]
    selected_by_rebalance: list[list[str]]


def _run_policy_surface(
    *,
    policy: str,
    surface_name: str,
    max_slots: int,
    rebalance_months: list[date],
    month_to_rebalance: dict[date, date],
    month_to_calendar_days: dict[date, list[date]],
    histories: dict[str, StrategyHistory],
    gp_by_day: dict[tuple[str, date], float | None],
    gp_shuffled: dict[tuple[str, date], float | None],
    corr_cache: dict[date, dict],
    orb_stats_cache: dict[date, dict],
    scores_cache: dict[date, list[LaneScore]],
    window_label: str,
) -> PolicyResult:
    monthly_outputs: list[dict] = []
    all_daily: list[float] = []
    all_selected: list[list[str]] = []
    prior_allocation: list[str] | None = None
    total_fills = 0

    for month_start in rebalance_months:
        rebalance_date = month_to_rebalance[month_start]
        scores = scores_cache[rebalance_date]
        corr = corr_cache[rebalance_date]
        orb_stats = orb_stats_cache[rebalance_date]

        ranking_values, exclude_none = _ranking_values_for_policy(
            policy, scores, histories, rebalance_date, gp_by_day, gp_shuffled
        )

        selected = _build_ranked_allocation(
            scores,
            ranking_values=ranking_values,
            max_slots=max_slots,
            max_dd=MAX_DD_DOLLARS,
            orb_size_stats=orb_stats,
            correlation_matrix=corr,
            prior_allocation=prior_allocation,
            exclude_none_ranks=exclude_none,
        )

        next_month = _next_month(month_start)
        end_exclusive = month_to_rebalance.get(next_month)
        if end_exclusive is None:
            # final month: still record selection for churn, but no forward window
            all_selected.append([s.strategy_id for s in selected])
            prior_allocation = [s.strategy_id for s in selected]
            continue

        calendar_days = month_to_calendar_days[month_start]
        selected_ids = [s.strategy_id for s in selected]
        daily_series, fills = _forward_window_accounting(selected_ids, histories, calendar_days)
        total_r = round(sum(daily_series), 6)
        total_fills += fills

        monthly_outputs.append(
            {
                "month": month_start.isoformat(),
                "rebalance_date": rebalance_date.isoformat(),
                "selected_lanes": selected_ids,
                "forward_window_start": rebalance_date.isoformat(),
                "forward_window_end_exclusive": end_exclusive.isoformat(),
                "forward_month_total_r": total_r,
                "forward_month_fills": fills,
                "forward_month_r_per_fill": round(total_r / fills, 6) if fills > 0 else 0.0,
                "forward_month_daily_sharpe": _annualized_sharpe(daily_series),
                "forward_month_slot_hit_rate_per_day": fills / len(calendar_days) if calendar_days else 0.0,
            }
        )
        all_daily.extend(daily_series)
        all_selected.append(selected_ids)
        prior_allocation = selected_ids

    total_r = round(sum(all_daily), 6)
    annualized_r = (total_r * (252.0 / len(all_daily))) if all_daily else 0.0
    r_per_fill = (total_r / total_fills) if total_fills > 0 else 0.0
    mean_hit_rate = (
        sum(m["forward_month_slot_hit_rate_per_day"] for m in monthly_outputs) / len(monthly_outputs)
        if monthly_outputs
        else 0.0
    )

    return PolicyResult(
        policy=policy,
        surface=surface_name,
        max_slots=max_slots,
        window=window_label,
        total_r=total_r,
        total_fills=total_fills,
        r_per_fill=round(r_per_fill, 6),
        sharpe=_annualized_sharpe(all_daily),
        dd=_max_drawdown(all_daily),
        annualized_r=round(annualized_r, 4),
        mean_hit_rate_per_day=round(mean_hit_rate, 4),
        trading_days_covered=len(all_daily),
        monthly_outputs=monthly_outputs,
        daily_pnl=all_daily,
        selected_by_rebalance=all_selected,
    )


# ---------------------------------------------------------------------------
# Binding preflight (re-verify audit finding in-harness, integrity check)
# ---------------------------------------------------------------------------


def _binding_preflight(
    rebalance_months: list[date],
    month_to_rebalance: dict[date, date],
    scores_cache: dict[date, list[LaneScore]],
    corr_cache: dict[date, dict],
) -> dict[str, dict]:
    """Per-surface binding verification. Matches audit methodology."""
    results: dict[str, dict] = {}
    for surface_name, max_slots in SURFACES:
        binds = 0
        total = 0
        supplies = []
        for month_start in rebalance_months:
            rebalance_date = month_to_rebalance[month_start]
            scores = scores_cache[rebalance_date]
            deployable = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
            if surface_name == "A_raw_slots":
                supply = len(deployable)
            elif surface_name == "B_rho_survivor_slots":
                corr = corr_cache[rebalance_date]
                survivors = build_allocation(
                    scores,
                    max_slots=100,
                    max_dd=1e12,
                    stop_multiplier=FIXED_STOP_MULTIPLIER,
                    correlation_matrix=corr,
                )
                supply = len(survivors)
            else:
                raise ValueError(f"unknown surface: {surface_name}")
            supplies.append(supply)
            total += 1
            if supply > max_slots:
                binds += 1
        ratio = binds / total if total else 0.0
        results[surface_name] = {
            "max_slots": max_slots,
            "binds": binds,
            "total": total,
            "bind_ratio": round(ratio, 4),
            "passes_gate": ratio >= BIND_PASS_RATIO_GATE,
            "mean_supply": round(sum(supplies) / len(supplies), 2) if supplies else 0.0,
            "median_supply": sorted(supplies)[len(supplies) // 2] if supplies else 0,
        }
    return results


# ---------------------------------------------------------------------------
# Harness-sanity gate: positive control must beat primary null by lift on both
# ---------------------------------------------------------------------------


def _harness_sanity_gate(
    is_results: dict[str, dict[str, PolicyResult]],
) -> dict[str, object]:
    """Return {'passes_all_surfaces': bool, 'per_surface': {...}}."""
    per_surface: dict[str, dict[str, object]] = {}
    all_pass = True
    for surface_name, _slots in SURFACES:
        pn = is_results[surface_name][RANKER_PRIMARY_NULL]
        pc = is_results[surface_name][RANKER_POSITIVE_CONTROL]
        lift = pc.r_per_fill - pn.r_per_fill
        passes = lift >= MIN_LIFT_PER_FILL
        if not passes:
            all_pass = False
        per_surface[surface_name] = {
            "r_per_fill_primary_null": pn.r_per_fill,
            "r_per_fill_positive_control": pc.r_per_fill,
            "lift": round(lift, 6),
            "min_lift_required": MIN_LIFT_PER_FILL,
            "passes": passes,
        }
    return {"passes_all_surfaces": all_pass, "per_surface": per_surface}


# ---------------------------------------------------------------------------
# Pass/fail evaluation per surface
# ---------------------------------------------------------------------------


def _mean_jaccard_over_rebalances(a_rebalances: list[list[str]], b_rebalances: list[list[str]]) -> float:
    if not a_rebalances or not b_rebalances:
        return 0.0
    pairs = list(zip(a_rebalances, b_rebalances, strict=False))
    vals = [_jaccard_distance(a, b) for a, b in pairs if a or b]
    return sum(vals) / len(vals) if vals else 0.0


def _evaluate_primary(
    surface: str,
    candidate: PolicyResult,
    primary_null: PolicyResult,
    secondary: PolicyResult,
) -> dict[str, object]:
    r_per_fill_delta = candidate.r_per_fill - primary_null.r_per_fill
    sharpe_delta = candidate.sharpe - secondary.sharpe
    dd_ratio = (candidate.dd / primary_null.dd) if primary_null.dd > 0 else float("inf")
    churn = _mean_jaccard_over_rebalances(candidate.selected_by_rebalance, primary_null.selected_by_rebalance)

    r_per_fill_pass = r_per_fill_delta >= MIN_LIFT_PER_FILL
    sharpe_pass = sharpe_delta >= MIN_SHARPE_LIFT
    dd_pass = dd_ratio <= MAX_DD_INFLATION
    churn_pass = churn <= SELECTION_CHURN_CAP

    return {
        "surface": surface,
        "r_per_fill_candidate": candidate.r_per_fill,
        "r_per_fill_primary_null": primary_null.r_per_fill,
        "r_per_fill_delta": round(r_per_fill_delta, 6),
        "r_per_fill_pass": r_per_fill_pass,
        "sharpe_candidate": round(candidate.sharpe, 4),
        "sharpe_secondary": round(secondary.sharpe, 4),
        "sharpe_delta": round(sharpe_delta, 4),
        "sharpe_pass": sharpe_pass,
        "dd_candidate": round(candidate.dd, 4),
        "dd_primary_null": round(primary_null.dd, 4),
        "dd_ratio": round(dd_ratio, 4),
        "dd_pass": dd_pass,
        "churn": round(churn, 4),
        "churn_pass": churn_pass,
        "primary_pass": r_per_fill_pass and sharpe_pass and dd_pass and churn_pass,
    }


def _evaluate_destruction_shuffle(
    surface: str,
    shuffle: PolicyResult,
    primary_null: PolicyResult,
    secondary: PolicyResult,
) -> dict[str, object]:
    """Shuffle MUST fail primary rule (evaluated exactly like the candidate)."""
    return _evaluate_primary(surface, shuffle, primary_null, secondary)


def _verdict_per_surface(eval_primary: dict[str, object]) -> str:
    return "PASS" if eval_primary["primary_pass"] else "FAIL"


def _dual_surface_verdict(
    verdict_A: str,
    verdict_B: str,
    harness_gate: dict[str, object],
    destruction_passes_on_any: bool,
) -> str:
    if not harness_gate["passes_all_surfaces"]:
        return "ABORT"
    if destruction_passes_on_any:
        return "NULL_DATA_MINED"
    if verdict_A == "PASS" and verdict_B == "PASS":
        return "STRONG_PASS"
    if verdict_A == "PASS" and verdict_B == "FAIL":
        return "STANDARD_PASS"
    if verdict_A == "FAIL" and verdict_B == "PASS":
        return "STRUCTURAL_PASS"
    return "NULL"


# ---------------------------------------------------------------------------
# OOS descriptive
# ---------------------------------------------------------------------------


def _oos_descriptive(
    is_primary_eval: dict[str, object],
    oos_candidate: PolicyResult,
    oos_primary_null: PolicyResult,
) -> dict[str, object]:
    oos_delta = oos_candidate.r_per_fill - oos_primary_null.r_per_fill
    is_delta = is_primary_eval["r_per_fill_delta"]
    assert isinstance(is_delta, float)
    if is_delta == 0:
        effect_ratio = None
    else:
        effect_ratio = oos_delta / is_delta
    direction_match = (
        (is_delta > 0 and oos_delta > 0) or (is_delta < 0 and oos_delta < 0) or (is_delta == 0 == oos_delta)
    )
    return {
        "surface": is_primary_eval["surface"],
        "is_r_per_fill_delta": is_delta,
        "oos_r_per_fill_delta": round(oos_delta, 6),
        "direction_match": direction_match,
        "effect_ratio": round(effect_ratio, 4) if effect_ratio is not None else None,
        "effect_ratio_pass_40pct": (effect_ratio is not None and effect_ratio >= 0.40),
        "oos_trading_days": oos_candidate.trading_days_covered,
        "oos_total_fills_candidate": oos_candidate.total_fills,
        "oos_total_fills_primary_null": oos_primary_null.total_fills,
    }


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------


def _emit_markdown(
    *,
    as_of: date,
    preflight: dict[str, dict],
    harness_gate: dict[str, object],
    is_results: dict[str, dict[str, PolicyResult]],
    oos_results: dict[str, dict[str, PolicyResult]],
    primary_eval: dict[str, dict[str, object]],
    destruction_eval: dict[str, dict[str, object]],
    oos_desc: dict[str, dict[str, object]],
    dual_verdict: str,
) -> str:
    lines: list[str] = [
        "# Garch A4c Routing-Selectivity Replay",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**As-of trading day:** {as_of.isoformat()}",
        f"**Pre-registration:** `{HYPOTHESIS_FILE}`",
        f"**Design:** `{DESIGN_DOC}`",
        f"**Framing commit:** `{FRAMING_COMMIT}`",
        f"**Dual-surface verdict:** **{dual_verdict}**",
        "",
        "## Binding preflight (re-verified in-harness)",
        "",
        "| Surface | Max slots | Binds | Total | Ratio | ≥80% gate | Mean supply | Median |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for surface_name, _slots in SURFACES:
        p = preflight[surface_name]
        lines.append(
            f"| {surface_name} | {p['max_slots']} | {p['binds']} | {p['total']} | "
            f"{p['bind_ratio']:.3f} | {'PASS' if p['passes_gate'] else 'FAIL'} | "
            f"{p['mean_supply']} | {p['median_supply']} |"
        )

    lines += [
        "",
        "## Harness-sanity gate (positive control vs primary null)",
        "",
        f"**Rule:** R-per-fill(positive_control) − R-per-fill(primary_null) ≥ {MIN_LIFT_PER_FILL} on BOTH surfaces.",
        f"**Gate outcome:** {'PASS' if harness_gate['passes_all_surfaces'] else 'ABORT'}",
        "",
        "| Surface | R/fill primary null | R/fill positive control | Lift | Required | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    per_surface = harness_gate["per_surface"]
    assert isinstance(per_surface, dict)
    for surface_name, _slots in SURFACES:
        r = per_surface[surface_name]
        assert isinstance(r, dict)
        lines.append(
            f"| {surface_name} | {r['r_per_fill_primary_null']:.6f} | "
            f"{r['r_per_fill_positive_control']:.6f} | {r['lift']:.6f} | "
            f"{r['min_lift_required']:.6f} | {'PASS' if r['passes'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## IS replay — R per filled slot-day by surface and policy",
        "",
        "| Surface | Policy | Trading days | Total R | Fills | R/fill | Sharpe | DD | Ann R | Hit-rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for surface_name, _slots in SURFACES:
        for policy in ALL_RANKERS:
            r = is_results[surface_name][policy]
            lines.append(
                f"| {surface_name} | {policy} | {r.trading_days_covered} | "
                f"{r.total_r:+.2f} | {r.total_fills} | {r.r_per_fill:+.6f} | "
                f"{r.sharpe:+.3f} | {r.dd:.2f} | {r.annualized_r:+.2f} | "
                f"{r.mean_hit_rate_per_day:.3f} |"
            )

    if harness_gate["passes_all_surfaces"]:
        lines += [
            "",
            "## Candidate primary evaluation (per surface)",
            "",
            "Primary pass rule (all must hold):",
            f"- R/fill candidate − R/fill primary_null ≥ {MIN_LIFT_PER_FILL}",
            f"- Sharpe candidate − Sharpe secondary_comparator ≥ {MIN_SHARPE_LIFT}",
            f"- DD candidate / DD primary_null ≤ {MAX_DD_INFLATION}",
            f"- Selection churn (jaccard) ≤ {SELECTION_CHURN_CAP}",
            "",
            "| Surface | ΔR/fill | R/fill pass | ΔSharpe | Sharpe pass | DD ratio | DD pass | Churn | Churn pass | Primary verdict |",
            "|---|---:|:---:|---:|:---:|---:|:---:|---:|:---:|:---:|",
        ]
        for surface_name, _slots in SURFACES:
            e = primary_eval[surface_name]
            lines.append(
                f"| {surface_name} | {e['r_per_fill_delta']:+.6f} | "
                f"{'PASS' if e['r_per_fill_pass'] else 'FAIL'} | "
                f"{e['sharpe_delta']:+.3f} | {'PASS' if e['sharpe_pass'] else 'FAIL'} | "
                f"{e['dd_ratio']:.3f} | {'PASS' if e['dd_pass'] else 'FAIL'} | "
                f"{e['churn']:.3f} | {'PASS' if e['churn_pass'] else 'FAIL'} | "
                f"**{_verdict_per_surface(e)}** |"
            )

        lines += [
            "",
            "## Destruction shuffle control (must FAIL primary on both surfaces)",
            "",
            "| Surface | ΔR/fill | Primary rule verdict | Pass (i.e. failed primary rule) |",
            "|---|---:|:---:|:---:|",
        ]
        for surface_name, _slots in SURFACES:
            d = destruction_eval[surface_name]
            shuffle_passed_primary = d["primary_pass"]
            lines.append(
                f"| {surface_name} | {d['r_per_fill_delta']:+.6f} | "
                f"{'PASS' if shuffle_passed_primary else 'FAIL'} | "
                f"{'PASS (shuffle failed as required)' if not shuffle_passed_primary else '**KILL: shuffle passed**'} |"
            )

        lines += [
            "",
            "## 2026 OOS descriptive (per surface, candidate vs primary null)",
            "",
            "Effect ratio ≥ 0.40 AND direction match required.",
            "",
            "| Surface | IS Δ | OOS Δ | Direction match | Effect ratio | Effect ≥ 0.40 | OOS days | OOS fills cand | OOS fills null |",
            "|---|---:|---:|:---:|---:|:---:|---:|---:|---:|",
        ]
        for surface_name, _slots in SURFACES:
            o = oos_desc[surface_name]
            er_str = f"{o['effect_ratio']:+.4f}" if o["effect_ratio"] is not None else "n/a"
            lines.append(
                f"| {surface_name} | {o['is_r_per_fill_delta']:+.6f} | "
                f"{o['oos_r_per_fill_delta']:+.6f} | "
                f"{'PASS' if o['direction_match'] else 'FAIL'} | "
                f"{er_str} | {'PASS' if o['effect_ratio_pass_40pct'] else 'FAIL'} | "
                f"{o['oos_trading_days']} | {o['oos_total_fills_candidate']} | {o['oos_total_fills_primary_null']} |"
            )
    else:
        lines += [
            "",
            "## Candidate evaluation SKIPPED",
            "",
            "Harness-sanity gate aborted. Positive control failed to beat primary null ",
            "by the pre-declared lift on at least one surface. Per pre-registered ",
            "semantics: do NOT rescue, do NOT re-run, do NOT tune. Return to framing.",
            "",
        ]

    lines += [
        "",
        "## Dual-surface verdict",
        "",
        f"**{dual_verdict}**",
        "",
        "Verdict semantics (locked in hypothesis file):",
        "- STRONG_PASS: routing edge on BOTH raw supply AND operative mechanic",
        "- STANDARD_PASS: raw-supply edge only (A passes, B fails)",
        "- STRUCTURAL_PASS: operative-mechanic edge only (B passes, A fails)",
        "- NULL: both surfaces fail primary rule — no rescue",
        "- NULL_DATA_MINED: destruction shuffle passed primary — candidate is data-mined",
        "- ABORT: harness-sanity gate failed — harness bug, not candidate verdict",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked A4c routing-selectivity replay.")
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    args = parser.parse_args()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        row = con.execute("SELECT MAX(trading_day) FROM daily_features").fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("daily_features is empty")
        as_of: date = row[0]
        metas = _load_strategy_meta(con)
        histories = _build_histories(con, metas, as_of)
        session_regime_cache = _build_session_regime_cache(con, metas, as_of)
        gp_by_day = _build_garch_pct_by_day(con, metas)

        # Rebalance months: IS + OOS, with 2026-01 boundary for final IS forward window
        month_to_rebalance = _first_trading_days_by_month(con, IS_START_MONTH, date(as_of.year, as_of.month, 1))
        if HOLDOUT_BOUNDARY not in month_to_rebalance:
            extra = _first_trading_days_by_month(con, HOLDOUT_BOUNDARY, HOLDOUT_BOUNDARY)
            month_to_rebalance.update(extra)

        # Calendar days per rebalance month's forward window
        month_to_calendar_days: dict[date, list[date]] = {}
        sorted_months = sorted(month_to_rebalance.keys())
        for i, m in enumerate(sorted_months):
            rebalance_date = month_to_rebalance[m]
            if i + 1 < len(sorted_months):
                end_exclusive = month_to_rebalance[sorted_months[i + 1]]
            else:
                # Final month: use as_of + 1 to include most recent data
                from datetime import timedelta

                end_exclusive = as_of + timedelta(days=1)
            month_to_calendar_days[m] = _calendar_days(con, rebalance_date, end_exclusive)
    finally:
        con.close()

    gp_shuffled = _shuffle_garch_pct(gp_by_day, seed=SHUFFLE_SEED)

    # Pre-compute per-rebalance caches (reused across all policies and surfaces)
    is_months: list[date] = [m for m in sorted_months if m <= IS_END_MONTH]
    # OOS: any month with start >= HOLDOUT_BOUNDARY and forward window fully observable
    oos_months: list[date] = [
        m for m in sorted_months if m >= HOLDOUT_BOUNDARY and _next_month(m) in month_to_rebalance
    ]

    print(f"[a4c] as_of: {as_of}")
    print(f"[a4c] metas: {len(metas)}")
    print(f"[a4c] IS months: {len(is_months)}  OOS months: {len(oos_months)}")

    scores_cache: dict[date, list[LaneScore]] = {}
    corr_cache: dict[date, dict] = {}
    orb_stats_cache: dict[date, dict] = {}

    for m in sorted_months:
        rebalance_date = month_to_rebalance[m]
        scores = _compute_scores(rebalance_date, histories, session_regime_cache)
        scores_cache[rebalance_date] = scores
        deployable_ids = [s.strategy_id for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
        corr_cache[rebalance_date] = _pairwise_correlation_as_of(deployable_ids, histories, rebalance_date)
        orb_stats_cache[rebalance_date] = compute_orb_size_stats(rebalance_date)
    print(f"[a4c] per-rebalance caches built: {len(scores_cache)} months")

    # Binding preflight (IS only — matches pre-reg)
    preflight = _binding_preflight(is_months, month_to_rebalance, scores_cache, corr_cache)
    for surface_name in preflight:
        p = preflight[surface_name]
        print(f"[a4c] preflight {surface_name}: bind {p['bind_ratio']:.3f} ({'PASS' if p['passes_gate'] else 'FAIL'})")

    # Run all 5 policies x 2 surfaces on IS
    is_results: dict[str, dict[str, PolicyResult]] = {}
    for surface_name, max_slots in SURFACES:
        is_results[surface_name] = {}
        for policy in ALL_RANKERS:
            print(f"[a4c] IS run: {surface_name} / {policy}")
            is_results[surface_name][policy] = _run_policy_surface(
                policy=policy,
                surface_name=surface_name,
                max_slots=max_slots,
                rebalance_months=is_months,
                month_to_rebalance=month_to_rebalance,
                month_to_calendar_days=month_to_calendar_days,
                histories=histories,
                gp_by_day=gp_by_day,
                gp_shuffled=gp_shuffled,
                corr_cache=corr_cache,
                orb_stats_cache=orb_stats_cache,
                scores_cache=scores_cache,
                window_label="IS",
            )

    harness_gate = _harness_sanity_gate(is_results)
    print(f"[a4c] harness gate: {'PASS' if harness_gate['passes_all_surfaces'] else 'ABORT'}")
    for surface_name in preflight:
        ps = harness_gate["per_surface"]
        assert isinstance(ps, dict)
        s = ps[surface_name]
        assert isinstance(s, dict)
        print(f"[a4c]   {surface_name}: lift {s['lift']:+.6f} (need ≥ {s['min_lift_required']})")

    primary_eval: dict[str, dict[str, object]] = {}
    destruction_eval: dict[str, dict[str, object]] = {}
    oos_results: dict[str, dict[str, PolicyResult]] = {}
    oos_desc: dict[str, dict[str, object]] = {}

    if harness_gate["passes_all_surfaces"]:
        # Evaluate candidate per surface
        for surface_name, _slots in SURFACES:
            primary_eval[surface_name] = _evaluate_primary(
                surface=surface_name,
                candidate=is_results[surface_name][RANKER_CANDIDATE],
                primary_null=is_results[surface_name][RANKER_PRIMARY_NULL],
                secondary=is_results[surface_name][RANKER_SECONDARY],
            )
            destruction_eval[surface_name] = _evaluate_destruction_shuffle(
                surface=surface_name,
                shuffle=is_results[surface_name][RANKER_DESTRUCTION],
                primary_null=is_results[surface_name][RANKER_PRIMARY_NULL],
                secondary=is_results[surface_name][RANKER_SECONDARY],
            )

        # OOS descriptive
        if oos_months:
            for surface_name, max_slots in SURFACES:
                oos_results[surface_name] = {}
                # OOS only needs candidate + primary null (descriptive rule)
                for policy in (RANKER_PRIMARY_NULL, RANKER_CANDIDATE):
                    oos_results[surface_name][policy] = _run_policy_surface(
                        policy=policy,
                        surface_name=surface_name,
                        max_slots=max_slots,
                        rebalance_months=oos_months,
                        month_to_rebalance=month_to_rebalance,
                        month_to_calendar_days=month_to_calendar_days,
                        histories=histories,
                        gp_by_day=gp_by_day,
                        gp_shuffled=gp_shuffled,
                        corr_cache=corr_cache,
                        orb_stats_cache=orb_stats_cache,
                        scores_cache=scores_cache,
                        window_label="OOS",
                    )
                oos_desc[surface_name] = _oos_descriptive(
                    is_primary_eval=primary_eval[surface_name],
                    oos_candidate=oos_results[surface_name][RANKER_CANDIDATE],
                    oos_primary_null=oos_results[surface_name][RANKER_PRIMARY_NULL],
                )

    # Check destruction shuffle doesn't pass primary on either surface
    destruction_passes_on_any = False
    if primary_eval:
        destruction_passes_on_any = any(
            destruction_eval[surface_name]["primary_pass"] for surface_name, _slots in SURFACES
        )

    # Dual-surface verdict
    if harness_gate["passes_all_surfaces"] and primary_eval:
        verdict_A = _verdict_per_surface(primary_eval["A_raw_slots"])
        verdict_B = _verdict_per_surface(primary_eval["B_rho_survivor_slots"])
    else:
        verdict_A = verdict_B = "N/A"
    dual_verdict = _dual_surface_verdict(verdict_A, verdict_B, harness_gate, destruction_passes_on_any)
    print(f"[a4c] dual-surface verdict: {dual_verdict}")

    # Emit MD
    md = _emit_markdown(
        as_of=as_of,
        preflight=preflight,
        harness_gate=harness_gate,
        is_results=is_results,
        oos_results=oos_results,
        primary_eval=primary_eval,
        destruction_eval=destruction_eval,
        oos_desc=oos_desc,
        dual_verdict=dual_verdict,
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")
    print(f"[a4c] wrote {output_md}")

    # Emit JSON (compact audit trail)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": as_of.isoformat(),
        "framing_commit": FRAMING_COMMIT,
        "hypothesis_file": HYPOTHESIS_FILE,
        "preflight": preflight,
        "harness_gate": harness_gate,
        "is_summary": {
            surface: {
                policy: {
                    k: v
                    for k, v in asdict(result).items()
                    if k not in ("monthly_outputs", "daily_pnl", "selected_by_rebalance")
                }
                for policy, result in is_results[surface].items()
            }
            for surface in is_results
        },
        "primary_eval": primary_eval,
        "destruction_eval": destruction_eval,
        "oos_desc": oos_desc,
        "dual_verdict": dual_verdict,
    }
    output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[a4c] wrote {output_json}")


if __name__ == "__main__":
    main()
