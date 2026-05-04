"""Additivity audit for PARKed Pathway-B candidates (D2, D4) — 2026-04-29.

Purpose
-------
Run the canonical lane-correlation gate
``trading_app.lane_correlation.check_candidate_correlation`` against the
deployed ``topstep_50k_mnq_auto`` book for the 2 Pathway-B candidates that
landed PARK_PENDING_OOS_POWER on 2026-04-28:

- D2 B-MES-EUR  (pre-reg: docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml)
- D4 B-MNQ-COX  (pre-reg: docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml)

Outputs a verdict matrix per candidate:
  candidate_id | worst_rho | worst_subset | gate_pass | reject_reasons | disposition

Where ``disposition`` is:
  - PASS_ADDITIVITY        — eligible for Phase E once OOS power clears
  - FAIL_ADDITIVITY        — RULE 7 KILL even if OOS PASS

Read-only: no writes to validated_setups, allocator, or live_config.
Fail-closed: zero candidate-pnl rows aborts before correlation call.

@pre-reg: candidate-only canonical registration (OVNRNG_PCT_GT80, GARCH_VOL_PCT_GT70)
@scratch-policy: include-as-zero (canonical post-Stage-5 outcome_builder; pnl_r=COALESCE(pnl_r, 0.0))
@research-source: docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.lane_correlation import (
    RHO_REJECT_THRESHOLD,
    SUBSET_REJECT_THRESHOLD,
    CorrelationReport,
    check_candidate_correlation,
)
from trading_app.strategy_fitness import _load_strategy_outcomes

PROFILE_ID = "topstep_50k_mnq_auto"

# Candidate definitions — one dict per parked Pathway-B cell.
# The ``filter_type`` keys MUST be canonical entries in trading_app.config.ALL_FILTERS;
# they are registered hypothesis-scoped (NOT in BASE_GRID_FILTERS / get_filters_for_grid).
CANDIDATES: list[dict[str, Any]] = [
    {
        "label": "D2_B-MES-EUR",
        "strategy_id": "MES_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_PCT_GT80_O15",
        "instrument": "MES",
        "orb_label": "EUROPE_FLOW",
        "orb_minutes": 15,
        "rr_target": 1.0,
        "entry_model": "E2",
        "confirm_bars": 1,
        "filter_type": "OVNRNG_PCT_GT80",
        "pre_reg": "docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml",
    },
    {
        "label": "D4_B-MNQ-COX",
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_GARCH_VOL_PCT_GT70",
        "instrument": "MNQ",
        "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5,
        "rr_target": 1.0,
        "entry_model": "E2",
        "confirm_bars": 1,
        "filter_type": "GARCH_VOL_PCT_GT70",
        "pre_reg": "docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml",
    },
]


def _verify_candidate_outcomes(con: duckdb.DuckDBPyConnection, cand: dict[str, Any]) -> int:
    """Fail-closed pre-flight: candidate must have non-zero outcomes.

    Returns N rows. Raises if zero (would silently produce ``rho=0`` /
    ``gate_pass=True`` from the canonical gate, which is misleading).
    """
    outcomes = _load_strategy_outcomes(
        con,
        instrument=cand["instrument"],
        orb_label=cand["orb_label"],
        orb_minutes=cand["orb_minutes"],
        entry_model=cand["entry_model"],
        rr_target=cand["rr_target"],
        confirm_bars=cand["confirm_bars"],
        filter_type=cand["filter_type"],
    )
    n = len(outcomes)
    if n == 0:
        raise RuntimeError(
            f"FAIL-CLOSED: candidate {cand['label']} ({cand['filter_type']}) "
            f"resolved to ZERO trades via _load_strategy_outcomes. "
            f"Cannot run additivity audit. Verify filter registration "
            f"and pre-reg cell predicate."
        )
    return n


def _disposition(report: CorrelationReport) -> str:
    """Map canonical CorrelationReport to human-readable disposition."""
    if report.gate_pass:
        return "PASS_ADDITIVITY"
    return "FAIL_ADDITIVITY"


def main() -> int:
    print("=" * 78)
    print("Parked Pathway-B Additivity Audit — 2026-04-29")
    print(f"Profile: {PROFILE_ID}")
    print(f"Thresholds: rho > {RHO_REJECT_THRESHOLD}, subset_coverage > {SUBSET_REJECT_THRESHOLD} (same-session only)")
    print("=" * 78)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)

    matrix_rows: list[dict[str, Any]] = []

    try:
        for cand in CANDIDATES:
            print(f"\n--- Candidate: {cand['label']} ---")
            print(f"  strategy_id : {cand['strategy_id']}")
            print(f"  filter_type : {cand['filter_type']}")
            print(f"  pre_reg     : {cand['pre_reg']}")

            n = _verify_candidate_outcomes(con, cand)
            print(f"  N (canonical filter applied): {n}")

            report = check_candidate_correlation(
                candidate_lane=cand,
                profile_id=PROFILE_ID,
                con=con,
            )

            print(f"  Pairs against {len(report.pairs)} deployed lanes")
            for pair in report.pairs:
                tag = "REJECT" if pair.reject else "OK"
                print(
                    f"    [{tag}] vs {pair.deployed_id}: "
                    f"shared={pair.shared_days}, rho={pair.pearson_rho:+.4f}, "
                    f"subset_cov={pair.subset_coverage:.1%} | {pair.reason}"
                )

            disp = _disposition(report)
            row = {
                "candidate": cand["label"],
                "strategy_id": cand["strategy_id"],
                "filter_type": cand["filter_type"],
                "candidate_n_days": report.pairs[0].candidate_days if report.pairs else 0,
                "worst_rho": report.worst_rho,
                "worst_subset": report.worst_subset,
                "gate_pass": report.gate_pass,
                "reject_reasons": " ; ".join(report.reject_reasons) if report.reject_reasons else "(none)",
                "disposition": disp,
            }
            matrix_rows.append(row)
            print(f"  -> {disp} | worst_rho={report.worst_rho:+.4f}, worst_subset={report.worst_subset:.1%}")

    finally:
        con.close()

    # Verdict matrix
    print("\n" + "=" * 78)
    print("VERDICT MATRIX")
    print("=" * 78)
    headers = ["candidate", "filter_type", "N_days", "worst_rho", "worst_subset", "gate_pass", "disposition"]
    print(" | ".join(f"{h:<22}" for h in headers))
    print("-" * 78)
    for row in matrix_rows:
        print(
            " | ".join(
                [
                    f"{row['candidate']:<22}",
                    f"{row['filter_type']:<22}",
                    f"{row['candidate_n_days']:<22}",
                    f"{row['worst_rho']:+.4f}".ljust(22),
                    f"{row['worst_subset']:.1%}".ljust(22),
                    f"{str(row['gate_pass']):<22}",
                    f"{row['disposition']:<22}",
                ]
            )
        )
    print("=" * 78)
    print("\nReject reasons by candidate:")
    for row in matrix_rows:
        print(f"  {row['candidate']}: {row['reject_reasons']}")

    # Output absolute path of result doc target
    result_doc = Path("docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md").resolve()
    print(f"\nResult doc target: {result_doc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
