"""Allocator Scarcity Surface Audit (A4c scouting — NOT a test).

Ranks 4 candidate scarce-resource surfaces on whether they bind against a
pre-declared budget over 72 IS rebalance dates. Read-only. No utility replay.
No hypothesis pre-registration needed — this is scarcity-surface scouting
before writing an executable A4c hypothesis.

Surfaces and budgets (all pre-declared before seeing results):
  A. Raw slot budget           → max_slots = 5
  B. Correlation-survivor slots → max_slots = 5, supply = rho<0.70 greedy prune
  C. Daily risk-R budget ($)    → $2,500 (bulenox_50k DD scalar, matches A4b)
  D. Contract-integer geometry  → DEMOTED (identical to A at 1-contract minimum)

Bind rule:
  hard gate = supply > budget on >= 80% of rebalance dates
  secondary = supply > 2 * budget (oversubscription indicator, not gate)

Canonical sources reused (no re-encoding):
  trading_app.lane_allocator — LaneScore, build_allocation, correlation helpers
  trading_app.validated_shelf — deployable shelf relation
  research.garch_a4b_binding_budget_replay — harness primitives

Output:
  docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pipeline.cost_model import COST_SPECS
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from research.garch_a4b_binding_budget_replay import (
    FIXED_STOP_MULTIPLIER,
    IS_START_MONTH,
    _build_histories,
    _build_session_regime_cache,
    _compute_scores,
    _first_trading_days_by_month,
    _load_strategy_meta,
    _pairwise_correlation_as_of,
)
from trading_app.lane_allocator import (
    CORRELATION_REJECT_RHO,
    _effective_annual_r,
    build_allocation,
    compute_orb_size_stats,
)

OUTPUT_MD = Path("docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md")

# Pre-declared budgets (locked before results seen)
BUDGET_SLOTS_RAW = 5
BUDGET_SLOTS_CORR = 5
BUDGET_RISK_R_DOLLARS = 2500.0

# Pass rule: supply > budget on >= 80% of rebalance dates
BIND_PASS_RATIO_GATE = 0.80
OVERSUB_MULT = 2  # secondary metric only


def _rho_survivor_count(
    scores: list,
    correlation_matrix: dict,
) -> int:
    """Return the size of the rho-survivor set under greedy <0.70 prune.

    Runs build_allocation with max_slots=100 and max_dd=1e12 so the rho gate is
    the only effective constraint. Matches the canonical allocator rho logic.
    """
    result = build_allocation(
        scores,
        max_slots=100,
        max_dd=1e12,
        stop_multiplier=FIXED_STOP_MULTIPLIER,
        correlation_matrix=correlation_matrix,
    )
    return len(result)


def _risk_r_supply_dollars(
    scores: list,
    orb_stats: dict,
) -> float:
    """Sum of per-lane worst-case $R exposure across deployable lanes.

    Per A4b convention: lane_R = p90_orb_pts * 0.75 * point_value * 1_contract.
    Returns dollar supply (before any budget comparison).
    """
    total = 0.0
    for s in scores:
        cost = COST_SPECS.get(s.instrument)
        if cost is None:
            continue
        _, p90 = orb_stats.get((s.instrument, s.orb_label, s.orb_minutes), (100.0, 100.0))
        total += p90 * FIXED_STOP_MULTIPLIER * cost.point_value
    return total


def _secondary_hit_rate_check(
    scores: list,
    correlation_matrix: dict,
    histories: dict,
    rebalance_date: date,
    forward_end: date,
    max_slots: int = 5,
) -> dict:
    """Slot-hit-rate dimension check across 3 ranking rules.

    Rankers:
      baseline_lit_grounded  — _effective_annual_r (canonical build_allocation)
      trailing_sharpe        — per-lane annualized Sharpe over trailing window
      random_uniform         — seeded random pick of N deployable (with rho gate)

    Hit-rate metric: mean selected-lanes-firing per forward trading day.
    No PnL. Pure firing frequency. Dimension check only.
    """
    import random as _rand

    def _firing_days(sid: str, start: date, end_exclusive: date) -> int:
        trades = histories[sid].trades
        days = {t.trading_day for t in trades if start <= t.trading_day < end_exclusive}
        return len(days)

    def _rank_baseline(xs: list) -> list:
        return sorted(xs, key=lambda s: (0 if s.status == "PROVISIONAL" else 1, _effective_annual_r(s)), reverse=True)

    def _rank_trailing_sharpe(xs: list) -> list:
        def key(s):
            sharpe = s.sharpe_ann_adj if hasattr(s, "sharpe_ann_adj") else 0.0
            if sharpe is None:
                sharpe = 0.0
            return (0 if s.status == "PROVISIONAL" else 1, float(sharpe))

        return sorted(xs, key=key, reverse=True)

    def _rank_random(xs: list) -> list:
        rng = _rand.Random(20260417 + rebalance_date.toordinal())
        shuffled = list(xs)
        rng.shuffle(shuffled)
        return shuffled

    deployable = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
    if not deployable:
        return {"eligible_n": 0}

    def _greedy_select(ranked: list) -> list:
        selected: list = []
        for lane in ranked:
            if len(selected) >= max_slots:
                break
            reject = False
            for sel in selected:
                a, b = (
                    (lane.strategy_id, sel.strategy_id)
                    if lane.strategy_id < sel.strategy_id
                    else (sel.strategy_id, lane.strategy_id)
                )
                rho = correlation_matrix.get((a, b), 0.0)
                if rho > CORRELATION_REJECT_RHO:
                    reject = True
                    break
            if reject:
                continue
            selected.append(lane)
        return selected

    forward_days = max((forward_end - rebalance_date).days, 1)

    sel_baseline = _greedy_select(_rank_baseline(deployable))
    sel_sharpe = _greedy_select(_rank_trailing_sharpe(deployable))
    sel_random = _greedy_select(_rank_random(deployable))

    def _hit_rate(sel: list) -> float:
        if not sel:
            return 0.0
        return sum(_firing_days(s.strategy_id, rebalance_date, forward_end) for s in sel) / forward_days

    return {
        "eligible_n": len(deployable),
        "selected_baseline_n": len(sel_baseline),
        "selected_sharpe_n": len(sel_sharpe),
        "selected_random_n": len(sel_random),
        "hit_rate_baseline": _hit_rate(sel_baseline),
        "hit_rate_sharpe": _hit_rate(sel_sharpe),
        "hit_rate_random": _hit_rate(sel_random),
    }


def _format_md(rows: list[dict], supply_stats: dict, hit_rate_stats: dict) -> str:
    n = len(rows)
    lines: list[str] = [
        "# Allocator Scarcity Surface Audit",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Purpose:** rank 4 candidate scarce-resource surfaces on binding behavior across "
        f"{n} IS rebalance dates (monthly, {IS_START_MONTH.isoformat()} onwards).",
        "**Verdict type:** supply vs pre-declared budget only. No utility replay. No candidate scoring.",
        "",
        "## Pre-declared budgets",
        "",
        f"- Surface A (raw slots): `max_slots = {BUDGET_SLOTS_RAW}`",
        f"- Surface B (rho-survivor slots): `max_slots = {BUDGET_SLOTS_CORR}`",
        f"- Surface C (daily risk-R): `${BUDGET_RISK_R_DOLLARS:.0f}` (bulenox_50k DD scalar)",
        "- Surface D (contract-integer): DEMOTED — identical to A at 1-contract minimum.",
        "",
        f"**Bind pass rule:** supply > budget on ≥{int(BIND_PASS_RATIO_GATE * 100)}% of rebalance dates.",
        f"**Oversubscription secondary:** supply > {OVERSUB_MULT} × budget (reporting only).",
        "",
        "## Per-rebalance supply summary",
        "",
        "| Month | Rebalance | A. Raw slots | B. Rho-survivor | C. Risk-$ | A bind | B bind | C bind |",
        "|---|---|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['month']} | {row['rebalance_date']} | {row['supply_slots']} | "
            f"{row['supply_corr_survivor']} | ${row['supply_risk_r_dollars']:.0f} | "
            f"{'✓' if row['bind_A'] else '✗'} | {'✓' if row['bind_B'] else '✗'} | "
            f"{'✓' if row['bind_C'] else '✗'} |"
        )
    lines += [
        "",
        "## Binding summary (primary gate)",
        "",
        "| Surface | Bind ratio | Passes ≥80% gate? | Mean supply | Median supply | Oversub ratio |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for key, label in [
        ("A", f"A. Raw slots (budget {BUDGET_SLOTS_RAW})"),
        ("B", f"B. Rho-survivor (budget {BUDGET_SLOTS_CORR})"),
        ("C", f"C. Risk-$ (budget ${BUDGET_RISK_R_DOLLARS:.0f})"),
    ]:
        s = supply_stats[key]
        passes = "PASS" if s["bind_ratio"] >= BIND_PASS_RATIO_GATE else "FAIL"
        lines.append(
            f"| {label} | {s['bind_ratio']:.3f} | {passes} | "
            f"{s['mean_supply']:.1f} | {s['median_supply']:.1f} | "
            f"{s['oversub_ratio']:.3f} |"
        )
    lines += [
        "",
        "## Ranking (primary pass, then oversub, then operational-reality tie-break)",
        "",
    ]
    # Rank among those that pass the hard gate; fallbacks below.
    ranked = sorted(
        ["A", "B", "C"],
        key=lambda k: (
            -int(supply_stats[k]["bind_ratio"] >= BIND_PASS_RATIO_GATE),
            -supply_stats[k]["bind_ratio"],
            -supply_stats[k]["oversub_ratio"],
        ),
    )
    rank_labels = {
        "A": "Raw slot budget (budget 5)",
        "B": "Correlation-survivor slot budget (budget 5)",
        "C": "Daily risk-R budget ($2,500)",
    }
    for i, k in enumerate(ranked, 1):
        s = supply_stats[k]
        passes = "PASS" if s["bind_ratio"] >= BIND_PASS_RATIO_GATE else "FAIL"
        lines.append(
            f"{i}. **{rank_labels[k]}** — bind {s['bind_ratio']:.3f} ({passes}), oversub {s['oversub_ratio']:.3f}"
        )
    lines += [
        "",
        "## Secondary: baseline dimension / hit-rate check (last 12 rebalances)",
        "",
        "Slot-hit-rate = mean selected-lanes-firing per forward trading day. No PnL. Pure firing-frequency dimension.",
        "",
        "| Metric | Baseline (lit-grounded) | Trailing-Sharpe | Random-uniform |",
        "|---|---:|---:|---:|",
        f"| Mean selected lanes | {hit_rate_stats['mean_sel_baseline']:.2f} | {hit_rate_stats['mean_sel_sharpe']:.2f} | {hit_rate_stats['mean_sel_random']:.2f} |",
        f"| Mean hit-rate / day | {hit_rate_stats['mean_hr_baseline']:.3f} | {hit_rate_stats['mean_hr_sharpe']:.3f} | {hit_rate_stats['mean_hr_random']:.3f} |",
        "",
        "Dimension interpretation:",
        "- If baseline and random-uniform hit-rates differ materially (≥20%), random-uniform is a structurally different null.",
        "- If baseline and trailing-Sharpe hit-rates are close (≤10% gap), trailing-Sharpe preserves dimension for a selectivity candidate.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocator scarcity surface audit.")
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
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
        month_to_rebalance = _first_trading_days_by_month(con, IS_START_MONTH, date(as_of.year, as_of.month, 1))
        # IS runs through 2025-12; drop any 2026 months
        month_to_rebalance = {m: r for m, r in month_to_rebalance.items() if m.year < 2026}
    finally:
        con.close()

    print(f"[audit] loaded {len(metas)} deployable strategies, {len(month_to_rebalance)} IS rebalance months.")
    print(f"[audit] as_of: {as_of}")

    # Build forward-window boundaries (needed for hit-rate dimension check on last 12 months)
    sorted_months = sorted(month_to_rebalance.keys())
    rebalance_to_next: dict[date, date] = {}
    for i, m in enumerate(sorted_months):
        r = month_to_rebalance[m]
        if i + 1 < len(sorted_months):
            rebalance_to_next[r] = month_to_rebalance[sorted_months[i + 1]]
        else:
            rebalance_to_next[r] = date(2026, 1, 1)  # next IS boundary = holdout

    per_rebalance_rows: list[dict] = []
    # For hit-rate dimension check: only last 12 IS rebalances
    hit_rate_targets = set(sorted_months[-12:]) if len(sorted_months) >= 12 else set(sorted_months)
    hit_rate_samples: list[dict] = []

    for month_start in sorted_months:
        rebalance_date = month_to_rebalance[month_start]
        scores = _compute_scores(rebalance_date, histories, session_regime_cache)
        deployable = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
        supply_slots = len(deployable)

        if supply_slots == 0:
            supply_corr_survivor = 0
            supply_risk_r_dollars = 0.0
        else:
            deployable_ids = [s.strategy_id for s in deployable]
            corr = _pairwise_correlation_as_of(deployable_ids, histories, rebalance_date)
            supply_corr_survivor = _rho_survivor_count(scores, corr)
            orb_stats = compute_orb_size_stats(rebalance_date)
            supply_risk_r_dollars = _risk_r_supply_dollars(deployable, orb_stats)

            if month_start in hit_rate_targets:
                hit_rate_samples.append(
                    _secondary_hit_rate_check(
                        scores=scores,
                        correlation_matrix=corr,
                        histories=histories,
                        rebalance_date=rebalance_date,
                        forward_end=rebalance_to_next[rebalance_date],
                        max_slots=BUDGET_SLOTS_RAW,
                    )
                )

        per_rebalance_rows.append(
            {
                "month": month_start.isoformat(),
                "rebalance_date": rebalance_date.isoformat(),
                "supply_slots": supply_slots,
                "supply_corr_survivor": supply_corr_survivor,
                "supply_risk_r_dollars": supply_risk_r_dollars,
                "bind_A": supply_slots > BUDGET_SLOTS_RAW,
                "bind_B": supply_corr_survivor > BUDGET_SLOTS_CORR,
                "bind_C": supply_risk_r_dollars > BUDGET_RISK_R_DOLLARS,
            }
        )
        print(
            f"[audit] {rebalance_date} slots={supply_slots} rho_survivor={supply_corr_survivor} risk_$={supply_risk_r_dollars:.0f}"
        )

    n = len(per_rebalance_rows)

    def _stats(key_supply: str, key_bind: str, budget: float) -> dict:
        supplies = [r[key_supply] for r in per_rebalance_rows]
        binds = [r[key_bind] for r in per_rebalance_rows]
        oversub = [s > OVERSUB_MULT * budget for s in supplies]
        return {
            "bind_ratio": sum(1 for b in binds if b) / n,
            "oversub_ratio": sum(1 for o in oversub if o) / n,
            "mean_supply": sum(supplies) / n,
            "median_supply": sorted(supplies)[n // 2],
        }

    supply_stats = {
        "A": _stats("supply_slots", "bind_A", BUDGET_SLOTS_RAW),
        "B": _stats("supply_corr_survivor", "bind_B", BUDGET_SLOTS_CORR),
        "C": _stats("supply_risk_r_dollars", "bind_C", BUDGET_RISK_R_DOLLARS),
    }

    if hit_rate_samples:
        valid = [s for s in hit_rate_samples if s.get("eligible_n", 0) > 0]
        if valid:
            hit_rate_stats = {
                "mean_sel_baseline": sum(s["selected_baseline_n"] for s in valid) / len(valid),
                "mean_sel_sharpe": sum(s["selected_sharpe_n"] for s in valid) / len(valid),
                "mean_sel_random": sum(s["selected_random_n"] for s in valid) / len(valid),
                "mean_hr_baseline": sum(s["hit_rate_baseline"] for s in valid) / len(valid),
                "mean_hr_sharpe": sum(s["hit_rate_sharpe"] for s in valid) / len(valid),
                "mean_hr_random": sum(s["hit_rate_random"] for s in valid) / len(valid),
            }
        else:
            hit_rate_stats = {
                k: 0.0
                for k in (
                    "mean_sel_baseline",
                    "mean_sel_sharpe",
                    "mean_sel_random",
                    "mean_hr_baseline",
                    "mean_hr_sharpe",
                    "mean_hr_random",
                )
            }
    else:
        hit_rate_stats = {
            k: 0.0
            for k in (
                "mean_sel_baseline",
                "mean_sel_sharpe",
                "mean_sel_random",
                "mean_hr_baseline",
                "mean_hr_sharpe",
                "mean_hr_random",
            )
        }

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_format_md(per_rebalance_rows, supply_stats, hit_rate_stats), encoding="utf-8")
    print(f"[audit] wrote {output_path}")
    print("[audit] bind ratios:", {k: round(v["bind_ratio"], 3) for k, v in supply_stats.items()})


if __name__ == "__main__":
    main()
