#!/usr/bin/env python3
"""Consolidated retirement-verdict view for the 38 active validated_setups.

Cross-references five 2026-04-19 audit streams (all read-only) into one
per-lane verdict matrix suitable for committee vote:

1. Mode-A criterion evaluation (C4/C7/C9) — computed in-script via
   `research.mode_a_revalidation_active_setups.compute_mode_a` +
   `.compute_criterion_flags`.
2. Regime-drift retirement queue — hardcoded lane IDs from
   `docs/audit/results/2026-04-19-mnq-retirement-queue-committee-action.md`
   (Tier-1 DECAY, Tier-2 REVIEW, HOLD regime-stressed, BETTER-THAN-PEERS).
3. Fire-rate audit — hardcoded flags from
   `docs/audit/results/2026-04-19-fire-rate-audit.md` (Rule 8.1 extreme
   fire-rate ≥95% or 0%; Rule 8.2 arithmetic_only; X_MES_ATR60 pipeline-
   absent caveat).
4. SGP O15/O30 capacity — hardcoded Jaccard finding from
   `docs/audit/results/2026-04-19-sgp-o15-o30-jaccard.md`.
5. Portfolio subset-t + lift-vs-unfiltered sweep (Phase 2.5) — hardcoded
   tier assignments from
   `docs/audit/results/2026-04-19-portfolio-subset-t-sweep.md` and
   `research/output/phase_2_5_portfolio_subset_t_sweep.csv`. Adds subset-t
   tier (1/2/3/4) and Rule 8.3 ARITHMETIC_LIFT flag as a NEW ANNOTATION
   on each lane — does NOT change existing verdict logic; surfaces in
   committee output for additional evidence.

The verdict column is assigned deterministically by rules documented below.
No post-hoc tuning; no new research; no DB writes; read-only audit join.

Output:
  docs/audit/results/2026-04-19-consolidated-retirement-verdict.md

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db uv run python research/consolidated_retirement_verdict.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from research.mode_a_revalidation_active_setups import (
    C4_T_WITH_THEORY,
    C9_MIN_N_PER_ERA,
    C9_ERA_THRESHOLD,
    C7_MIN_N,
    LaneRevalidation,
    classify_divergence,
    compute_criterion_flags,
    compute_mode_a,
    direction_from_execution_spec,
    load_active_setups,
)

RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-19-consolidated-retirement-verdict.md"
)


# --- Hardcoded cross-reference data from committed audit docs -------------
# Each of these sets is sourced verbatim from the committed markdown docs.
# If those docs change, update here and re-run.

# From docs/audit/results/2026-04-19-mnq-retirement-queue-committee-action.md
# Tier-1 DECAY: excess drop > 0.60 vs portfolio-wide -0.41 (lane-specific decay > 1.00 absolute)
RETIREMENT_TIER1: set[str] = {
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM",
    "MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM",
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",  # late Sharpe NEGATIVE → urgent
}
RETIREMENT_TIER1_URGENT: set[str] = {
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",
}

# Tier-2 REVIEW: excess drop 0.10–0.60 vs portfolio (lane drop 0.51–1.00 absolute)
RETIREMENT_TIER2: set[str] = {
    "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
    "MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5",
    "MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM",
}

# HOLD — regime-stressed but within portfolio norm (originally flagged CRITICAL
# by raw-Sharpe-drop framing; reframed via environment-controlled analysis)
RETIREMENT_HOLD: set[str] = {
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100",
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60",
}

# BETTER-THAN-PEERS — Sharpe went UP early→late
RETIREMENT_BETTER: set[str] = {
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100",
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60",
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15",
}


# From docs/audit/results/2026-04-19-fire-rate-audit.md
# Rule 8.1: fire_rate < 5% or > 95%
FIRE_RATE_OVER95: dict[str, float] = {
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5": 99.7,
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5": 99.7,
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15": 99.7,
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12": 98.6,
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12": 98.6,
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5": 95.1,
    "MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5": 95.1,
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5": 95.1,
}

# X_MES_ATR60 lanes showed 0% fire in the Phase 2.2 audit because that audit
# did NOT apply the canonical _inject_cross_asset_atrs pre-processing. My
# mode_a_revalidation_active_setups.compute_mode_a DOES apply the injection
# (lines 182-199), so X_MES_ATR60 metrics in the criterion eval are valid.
# The "0% fire" in fire-rate audit is an audit-scope artifact, NOT a lane defect.
FIRE_RATE_ZERO_INJECTION_ARTIFACT: set[str] = {
    "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60",
}

# Rule 8.2: wr_spread < 3% AND |expR_spread| > 0.10 — cost-screen, not edge
FIRE_RATE_ARITHMETIC_ONLY: set[str] = {
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12",
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5",
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15",
}


# From docs/audit/results/2026-04-19-sgp-o15-o30-jaccard.md
# Jaccard 0.6493 — WARN band (0.50-0.70), moderate redundancy
SGP_JACCARD_PAIR: set[str] = {
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15",
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
}


# From docs/audit/results/2026-04-19-portfolio-subset-t-sweep.md (Phase 2.5)
# Tier 1: subset-t >= 3.00 (Chordia 2018 with-theory, discovery-strict) AND
#         primary_flag = PASS (i.e., N >= 100 also, so lane fully qualifies)
SUBSET_T_TIER1_PASS: set[str] = {
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15",
    "MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8",
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100",
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
}

# Chordia-strict t-pass BUT N<100 (Harvey-Liu deployable floor). Noted
# separately because the signal quality is strong; only N is sample-constrained.
SUBSET_T_TIER1_THIN_N: set[str] = {
    "MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08",
}

# Tier 2: t in [2.58, 3.00) — passes stringent p<0.01 re-audit, misses Chordia
SUBSET_T_TIER2: set[str] = {
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15",
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    "MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
}

# Tier 4: t < 1.96 (fails conventional p<0.05) — HONEST retirement candidates
SUBSET_T_TIER4_FAIL_CONVENTIONAL: set[str] = {
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12",
    "MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM",
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100",
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM",
    "MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60",
    "MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12",
    "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",
}

# Rule 8.3 ARITHMETIC_LIFT (from backtesting-methodology.md addendum 2026-04-19):
# subset ExpR shows >0.10 lift vs unfiltered baseline BUT subset-t fails Chordia.
# Classic MES-class arithmetic illusion. Retain regardless of tier.
SUBSET_T_ARITHMETIC_LIFT: set[str] = {
    "MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15",
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
}


def subset_t_annotation(strategy_id: str) -> str:
    """Return human-readable Phase 2.5 subset-t annotation for a lane."""
    if strategy_id in SUBSET_T_ARITHMETIC_LIFT:
        return "Rule 8.3 ARITHMETIC_LIFT (retire/reframe)"
    if strategy_id in SUBSET_T_TIER1_PASS:
        return "Tier 1 (t>=3.00 Chordia PASS)"
    if strategy_id in SUBSET_T_TIER1_THIN_N:
        return "Tier 1 thin-N (t>=3.00, N<100)"
    if strategy_id in SUBSET_T_TIER2:
        return "Tier 2 (t in [2.58, 3.00))"
    if strategy_id in SUBSET_T_TIER4_FAIL_CONVENTIONAL:
        return "Tier 4 (t<1.96 — fails p<0.05)"
    return "Tier 3 (t in [1.96, 2.58))"


# --- Verdict assignment rules (deterministic, doctrine-grounded) ----------

def assign_verdict(rv: LaneRevalidation) -> tuple[str, list[str]]:
    """Assign a consolidated verdict code + cited reasons for a revalidated lane.

    Verdict codes (most-severe first — single-verdict output; reasons list
    may include lesser concerns):

    - RETIRE_URGENT    — Tier-1 retirement queue with negative late Sharpe
                         (already losing money; vote this week)
    - RETIRE_STANDARD  — Tier-1 retirement queue (excess decay > 0.60 vs portfolio)
    - N_UNDERPOWERED   — C7 FAIL (Mode A N < 100); cannot deploy under doctrine
    - RECLASSIFY_COST  — fire-rate ≥ 95% OR arithmetic_only; per Amendment v3.2
                         DRAFT Criteria 13/14, route to cost-screen registry,
                         not filter-edge registry
    - REVIEW_TIER2     — retirement queue Tier-2 (excess drop 0.10-0.60)
    - REVIEW_CAPACITY  — SGP O15/O30 pair with 65% Jaccard; capacity-split
                         decision required before full parallel deploy
    - REVIEW_C4_WT_FAIL — C4 with-theory FAIL but no other red flags; either
                         lane lacks theory citation (then C4 no-theory 3.79
                         applies and it also fails) or re-validation needed
    - BETTER_THAN_PEERS — Sharpe went UP early→late despite portfolio stress;
                         keep, potentially candidate for scaling
    - KEEP             — passes all evaluated gates, not in retirement queue
    """
    reasons: list[str] = []

    # Primary verdict selection (most-severe first)
    if rv.strategy_id in RETIREMENT_TIER1_URGENT:
        reasons.append("RETIREMENT Tier-1 with NEGATIVE late Sharpe — actively losing")
        verdict = "RETIRE_URGENT"
    elif rv.strategy_id in RETIREMENT_TIER1:
        reasons.append("RETIREMENT Tier-1: excess decay > 0.60 vs portfolio")
        verdict = "RETIRE_STANDARD"
    elif rv.c7_pass is False:
        reasons.append(f"C7 FAIL: Mode A N={rv.mode_a_n} < {C7_MIN_N}")
        verdict = "N_UNDERPOWERED"
    elif rv.strategy_id in FIRE_RATE_OVER95:
        fr = FIRE_RATE_OVER95[rv.strategy_id]
        reasons.append(f"Rule 8.1 fire-rate {fr:.1f}% (>95%)")
        verdict = "RECLASSIFY_COST"
    elif rv.strategy_id in FIRE_RATE_ARITHMETIC_ONLY:
        reasons.append("Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta)")
        verdict = "RECLASSIFY_COST"
    elif rv.strategy_id in RETIREMENT_TIER2:
        reasons.append("RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio")
        verdict = "REVIEW_TIER2"
    elif rv.strategy_id in SGP_JACCARD_PAIR:
        reasons.append("SGP O15/O30 pair: Jaccard 0.65 — capacity review before parallel deploy")
        verdict = "REVIEW_CAPACITY"
    elif rv.strategy_id in RETIREMENT_BETTER:
        reasons.append("Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late")
        verdict = "BETTER_THAN_PEERS"
    elif rv.c4_pass_with_theory is False:
        reasons.append(
            f"C4 with-theory FAIL: t_IS={rv.c4_t_stat:.2f} < {C4_T_WITH_THEORY}"
            if rv.c4_t_stat is not None
            else f"C4 t_IS unavailable"
        )
        verdict = "REVIEW_C4_WT_FAIL"
    else:
        verdict = "KEEP"

    # Secondary reasons — always append if present, regardless of primary verdict
    if rv.c9_pass is False:
        reasons.append(
            f"C9 era stability FAIL: doctrine era(s) {rv.c9_violating_eras} "
            f"ExpR<{C9_ERA_THRESHOLD} (era-N>={C9_MIN_N_PER_ERA})"
        )
    if rv.strategy_id in FIRE_RATE_ZERO_INJECTION_ARTIFACT:
        reasons.append(
            "Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection "
            "in that audit's scope, NOT a lane defect. Criterion eval applies canonical "
            "injection and the lane has real data."
        )
    if rv.mode_b_contaminated:
        reasons.append(
            f"Mode-B grandfathered: stored ExpR last_trade_day={rv.stored_last_trade_day} "
            ">= 2026-01-01; stored values are not Mode-A-clean."
        )
    # Note if lane passes primary verdict but still has C4_WT fail as secondary info
    if verdict != "REVIEW_C4_WT_FAIL" and rv.c4_pass_with_theory is False and rv.c4_t_stat is not None:
        reasons.append(
            f"(secondary) C4 with-theory t_IS={rv.c4_t_stat:.2f} < {C4_T_WITH_THEORY}"
        )

    return verdict, reasons


# --- Rendering -----------------------------------------------------------

VERDICT_ORDER = [
    "RETIRE_URGENT",
    "RETIRE_STANDARD",
    "N_UNDERPOWERED",
    "RECLASSIFY_COST",
    "REVIEW_TIER2",
    "REVIEW_CAPACITY",
    "REVIEW_C4_WT_FAIL",
    "BETTER_THAN_PEERS",
    "KEEP",
]


def render(results: list[tuple[LaneRevalidation, str, list[str]]]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_verdict: dict[str, list[tuple[LaneRevalidation, str, list[str]]]] = {
        v: [] for v in VERDICT_ORDER
    }
    for rv, verdict, reasons in results:
        by_verdict[verdict].append((rv, verdict, reasons))

    lines: list[str] = []
    lines.append("# Consolidated retirement-verdict view — 38 active validated_setups")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(
        "**Script:** `research/consolidated_retirement_verdict.py`"
    )
    lines.append("**Input audit streams:**")
    lines.append(
        "  - Mode-A criterion evaluation (C4/C7/C9) via "
        "`research.mode_a_revalidation_active_setups`"
    )
    lines.append(
        "  - Regime-drift retirement queue "
        "(`docs/audit/results/2026-04-19-mnq-retirement-queue-committee-action.md`)"
    )
    lines.append(
        "  - Fire-rate audit "
        "(`docs/audit/results/2026-04-19-fire-rate-audit.md`)"
    )
    lines.append(
        "  - SGP O15/O30 Jaccard "
        "(`docs/audit/results/2026-04-19-sgp-o15-o30-jaccard.md`)"
    )
    lines.append(
        "  - Portfolio subset-t + lift-vs-unfiltered sweep (Phase 2.5) "
        "(`docs/audit/results/2026-04-19-portfolio-subset-t-sweep.md`)"
    )
    lines.append("")
    lines.append(
        "**Canonical truth:** `orb_outcomes`, `daily_features`, "
        "`trading_app.holdout_policy.HOLDOUT_SACRED_FROM`. "
        "Filters via `research.filter_utils.filter_signal`. "
        "Read-only; no DB writes; no `validated_setups` mutation."
    )
    lines.append("")

    # Summary counts
    lines.append("## Summary")
    lines.append("")
    total = len(results)
    for v in VERDICT_ORDER:
        count = len(by_verdict[v])
        if count:
            lines.append(f"- **{v}:** {count}/{total}")
    lines.append("")

    lines.append("## Verdict rubric (deterministic, doctrine-cited)")
    lines.append("")
    lines.append(
        "| Verdict | Trigger | Action |"
    )
    lines.append("|---|---|---|")
    lines.append(
        "| `RETIRE_URGENT` | Retirement-queue Tier-1 with NEGATIVE late Sharpe | "
        "Vote this week. Actively losing money. |"
    )
    lines.append(
        "| `RETIRE_STANDARD` | Retirement-queue Tier-1 (excess decay > 0.60 vs portfolio) | "
        "Vote this week. Decay exceeds environment. |"
    )
    lines.append(
        f"| `N_UNDERPOWERED` | C7 FAIL: Mode A N < {C7_MIN_N} "
        "(`pre_registered_criteria.md § 7`) | Retire under doctrine — "
        "insufficient N for deployment. |"
    )
    lines.append(
        "| `RECLASSIFY_COST` | Rule 8.1 fire-rate ≥ 95% OR Rule 8.2 "
        "arithmetic_only (Amendment v3.2 DRAFT C13/C14) | Route to "
        "cost-screen registry, not filter-edge registry. |"
    )
    lines.append(
        "| `REVIEW_TIER2` | Retirement-queue Tier-2 (excess decay 0.10-0.60) | "
        "Vote within 2 weeks. |"
    )
    lines.append(
        "| `REVIEW_CAPACITY` | SGP O15/O30 pair (Jaccard 0.65) | "
        "Capacity-split decision required before parallel deploy. |"
    )
    lines.append(
        f"| `REVIEW_C4_WT_FAIL` | C4 with-theory FAIL (t < {C4_T_WITH_THEORY}) "
        "but no other red flags | Either promote C4 grounding to DIRECT "
        "Tier 1 for this lane's theory citation, OR re-validate on larger "
        "sample, OR downgrade to C4 no-theory 3.79 threshold (which this "
        "lane then also fails). |"
    )
    lines.append(
        "| `BETTER_THAN_PEERS` | Sharpe rose early→late under portfolio stress | "
        "Keep; potential scaling candidate (separate pre-reg required). |"
    )
    lines.append(
        "| `KEEP` | Passes all evaluated gates, no retirement queue entry | "
        "Retain. Note: C4 grounding caveat applies — with-theory 3.00 "
        "threshold is INDIRECT Tier 1 per doctrine. |"
    )
    lines.append("")

    lines.append("## Per-lane verdict table")
    lines.append("")
    lines.append(
        "Primary verdict is the first row per lane. Additional reasons (C9 era "
        "fails, Mode-B grandfathered, cross-asset injection artifact, secondary "
        "C4 notes) listed as supplementary evidence."
    )
    lines.append("")
    lines.append(
        "| # | Strategy ID | Verdict | Mode-A N | t_IS | C7 | C9 | Phase 2.5 tier | Primary reason |"
    )
    lines.append("|---|---|---|---:|---:|---|---|---|---|")

    idx = 1
    for v in VERDICT_ORDER:
        for rv, _verdict, reasons in by_verdict[v]:
            t_str = f"{rv.c4_t_stat:+.2f}" if rv.c4_t_stat is not None else "—"
            c7_str = {True: "PASS", False: "FAIL", None: "—"}[rv.c7_pass]
            c9_str = {True: "PASS", False: "FAIL", None: "—"}[rv.c9_pass]
            primary = reasons[0] if reasons else "—"
            p25 = subset_t_annotation(rv.strategy_id)
            lines.append(
                f"| {idx} | `{rv.strategy_id}` | **{v}** | "
                f"{rv.mode_a_n} | {t_str} | {c7_str} | {c9_str} | {p25} | {primary} |"
            )
            idx += 1
    lines.append("")

    # Per-verdict detailed sections
    for v in VERDICT_ORDER:
        lanes = by_verdict[v]
        if not lanes:
            continue
        lines.append(f"## {v} — detail")
        lines.append("")
        for rv, _, reasons in lanes:
            expr_str = f"{rv.mode_a_expr:+.4f}" if rv.mode_a_expr is not None else "—"
            t_str = f"{rv.c4_t_stat:+.2f}" if rv.c4_t_stat is not None else "—"
            sh_str = f"{rv.mode_a_sharpe:+.2f}" if rv.mode_a_sharpe is not None else "—"
            lines.append(f"### `{rv.strategy_id}`")
            lines.append("")
            lines.append(
                f"- Mode-A: N={rv.mode_a_n} ExpR={expr_str} t_IS={t_str} Sh_ann={sh_str}"
            )
            lines.append(
                f"- Phase 2.5 subset-t sweep: {subset_t_annotation(rv.strategy_id)}"
            )
            lines.append(
                f"- Years positive: {rv.years_positive}/{rv.years_total}"
            )
            if rv.c9_violating_eras:
                era_detail = " ".join(
                    f"{era}:{rv.c9_era_aggregates[era]['expr']:+.3f}"
                    f"(N={rv.c9_era_aggregates[era]['n']})"
                    for era in rv.c9_violating_eras
                    if era in rv.c9_era_aggregates
                )
                lines.append(f"- Violating eras: {era_detail}")
            lines.append("- Evidence:")
            for reason in reasons:
                lines.append(f"  - {reason}")
            lines.append("")

    lines.append("## Scope — criteria NOT evaluated here")
    lines.append("")
    lines.append(
        "Per `pre_registered_criteria.md`, the 12 doctrine criteria include "
        "several not computable from this audit's inputs:"
    )
    lines.append("")
    lines.append(
        "- **C1 (pre-reg file)**: requires checking `docs/audit/hypotheses/` "
        "for each lane's original pre-reg. Historical lanes pre-date Phase 0 "
        "literature grounding (2026-04-07); they are research-provisional per "
        "Amendment 2.7."
    )
    lines.append(
        "- **C2 (MinBTL)**: requires discovery run's trial count. Not stored."
    )
    lines.append(
        "- **C3 (BH-FDR)**: requires discovery hypothesis family. Not stored."
    )
    lines.append(
        "- **C5 (DSR)**: downgraded to INFORMATIONAL-only per Amendment 2.1 "
        "because N_eff unresolved."
    )
    lines.append(
        "- **C6 (WFE)**: requires OOS Sharpe computation under Mode A. Not in scope."
    )
    lines.append(
        "- **C8 (2026 OOS)**: 2026 is sacred holdout under Amendment 2.7."
    )
    lines.append(
        "- **C10 (data-era compat)**: filter-class specific (volume filters on "
        "MICRO era only). Not a standalone per-lane check."
    )
    lines.append(
        "- **C11 (account-death Monte Carlo)**: deployment-time gate, not "
        "audit-time."
    )
    lines.append(
        "- **C12 (Shiryaev-Roberts monitor)**: post-deployment drift gate, "
        "not audit-time."
    )
    lines.append("")

    lines.append("## Committee action matrix")
    lines.append("")
    lines.append(
        "| Action | Verdict codes | Count | Timing |"
    )
    lines.append("|---|---|---:|---|")
    counts = {v: len(by_verdict[v]) for v in VERDICT_ORDER}
    lines.append(
        f"| Immediate retire vote | `RETIRE_URGENT` + `RETIRE_STANDARD` | "
        f"{counts['RETIRE_URGENT'] + counts['RETIRE_STANDARD']} | This week |"
    )
    lines.append(
        f"| Immediate N-floor retire | `N_UNDERPOWERED` | "
        f"{counts['N_UNDERPOWERED']} | This week |"
    )
    lines.append(
        f"| Route to cost-screen registry | `RECLASSIFY_COST` | "
        f"{counts['RECLASSIFY_COST']} | Gated on Amendment v3.2 lock |"
    )
    lines.append(
        f"| Next-sprint review | `REVIEW_TIER2` + `REVIEW_CAPACITY` + "
        f"`REVIEW_C4_WT_FAIL` | "
        f"{counts['REVIEW_TIER2'] + counts['REVIEW_CAPACITY'] + counts['REVIEW_C4_WT_FAIL']} | "
        f"Within 2 weeks |"
    )
    lines.append(
        f"| Keep, potential scaling | `BETTER_THAN_PEERS` | "
        f"{counts['BETTER_THAN_PEERS']} | No action required |"
    )
    lines.append(
        f"| Keep, no action | `KEEP` | {counts['KEEP']} | No action required |"
    )
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(
        "DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db "
        "uv run python research/consolidated_retirement_verdict.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Read-only audit. Numbers reproduce exactly on the same DB state. "
        "If source audit docs change, update hardcoded cross-reference "
        "sets at the top of the script and re-run."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        active = load_active_setups(con)
        print(f"Loaded {len(active)} active validated_setups")

        results: list[tuple[LaneRevalidation, str, list[str]]] = []
        for i, spec in enumerate(active, 1):
            direction = direction_from_execution_spec(spec.get("execution_spec"))
            n, expr, sharpe_ann, wr, year_break, sd = compute_mode_a(con, spec)
            yrs_pos = sum(
                1 for b in year_break.values() if b["positive"] and b["n"] >= 10
            )
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
            verdict, reasons = assign_verdict(rv)
            results.append((rv, verdict, reasons))

            print(
                f"  {i:2d}/{len(active)} {verdict:<18} {rv.strategy_id}"
            )
    finally:
        con.close()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(results), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")

    # Summary
    from collections import Counter
    counts: Counter[str] = Counter(v for _, v, _ in results)
    print("\nVerdict counts:")
    for v in VERDICT_ORDER:
        print(f"  {v:<20} {counts.get(v, 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
