from __future__ import annotations

import csv
from pathlib import Path

from scripts.tools import chordia_unlock_batch as cub
from scripts.tools import lane_bench_state as lbs


def _candidate(
    strategy_id: str,
    *,
    status: str = "DEPLOY",
    reason: str = "ok",
    verdict: str | None = "MISSING",
    c8: str | None = "PASSED",
    annual: float = 10.0,
    expr: float = 0.05,
    instrument: str = "MNQ",
    session: str = "NYSE_OPEN",
) -> cub.UnlockCandidate:
    return cub.UnlockCandidate(
        strategy_id=strategy_id,
        instrument=instrument,
        session=session,
        orb_minutes=15,
        entry_model="E2",
        rr_target=1.5,
        filter_type="NO_FILTER",
        confirm_bars=1,
        trailing_expr=expr,
        trailing_n=100,
        annual_r_estimate=annual,
        status=status,
        status_reason=reason,
        chordia_verdict=verdict,
        chordia_audit_age_days=None,
        c8_oos_status=c8,
    )


def test_missing_chordia_positive_lane_is_exact_replay_work() -> None:
    rows = lbs.build_bench_rows(
        [_candidate("A")],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )

    assert rows[0].state == "EXACT_LANE_READY_FOR_REPLAY"
    assert rows[0].primary_blocker == "MISSING_CHORDIA"
    assert rows[0].next_action == "RUN_STRICT_UNLOCK"


def test_chordia_pass_not_active_is_allocator_eligible_bench() -> None:
    rows = lbs.build_bench_rows(
        [_candidate("A", verdict="PASS_CHORDIA")],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )

    assert rows[0].state == "ALLOCATOR_ELIGIBLE_BENCH"
    assert rows[0].primary_blocker == "NONE"
    assert rows[0].next_action == "EVALUATE_ALLOCATION_SLOT"


def test_blockers_precede_missing_chordia() -> None:
    candidates = [
        _candidate("C8", c8="NEGATIVE_OOS_EXPR"),
        _candidate("UNSAFE", status="PAUSE", reason="live tradeability gate: close-selected"),
        _candidate("NEG", annual=-1.0, expr=-0.01),
    ]

    rows = lbs.build_bench_rows(
        candidates,
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )

    assert [(r.strategy_id, r.state, r.primary_blocker) for r in rows] == [
        ("C8", "PARKED", "C8_OOS"),
        ("UNSAFE", "PARKED", "LIVE_TRADEABILITY"),
        ("NEG", "PARKED", "NEGATIVE_CURRENT_EDGE"),
    ]


def test_active_lane_with_blocker_is_review_not_silently_ok() -> None:
    rows = lbs.build_bench_rows(
        [_candidate("ACTIVE_BAD", verdict="FAIL_CHORDIA")],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset({"ACTIVE_BAD"}),
        family_hints={},
    )

    assert rows[0].state == "LIVE_ACTIVE_REVIEW"
    assert rows[0].primary_blocker == "CHORDIA_FAIL"
    assert rows[0].next_action == "REMOVE_OR_REPLACE_ACTIVE_LANE"


def test_report_declares_read_only_state_counts() -> None:
    rows = lbs.build_bench_rows(
        [
            _candidate("A"),
            _candidate("B", verdict="PASS_PROTOCOL_A"),
        ],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )

    report = lbs.render_report(rows, profile_id="topstep_50k_mnq_auto", source_notes=("unit test",))

    assert "read-only lane bench state surface" in report
    assert "does not mutate live allocation" in report
    assert "state counts" in report
    assert "EXACT_LANE_READY_FOR_REPLAY" in report


def test_load_family_hints_from_prior_bench_csv(tmp_path: Path) -> None:
    path = tmp_path / "bench.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "instrument",
                "session",
                "orb_minutes",
                "entry_model",
                "rr_target",
                "filter_type",
                "family",
                "family_role",
                "family_priority",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "instrument": "MNQ",
                "session": "NYSE_OPEN",
                "orb_minutes": "15",
                "entry_model": "E2",
                "rr_target": "1.5",
                "filter_type": "NO_FILTER",
                "family": "MNQ NYSE_OPEN baseline_orb",
                "family_role": "deployable_candidate",
                "family_priority": "0",
            }
        )

    hints = lbs.load_family_hints_from_bench(path)
    rows = lbs.build_bench_rows(
        [_candidate("A")],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints=hints,
    )

    assert rows[0].family == "MNQ NYSE_OPEN baseline_orb"
    assert rows[0].family_role == "deployable_candidate"
    assert rows[0].family_priority == 0
