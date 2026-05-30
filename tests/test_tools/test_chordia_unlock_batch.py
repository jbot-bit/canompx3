from __future__ import annotations

from pathlib import Path

from scripts.tools import chordia_unlock_batch as cub


def _candidate(
    strategy_id: str,
    *,
    instrument: str = "MNQ",
    session: str = "NYSE_OPEN",
    orb_minutes: int = 15,
    rr_target: float = 1.5,
    filter_type: str = "NO_FILTER",
    status: str = "DEPLOY",
    chordia_verdict: str | None = "MISSING",
    annual_r: float = 12.0,
    trailing_expr: float = 0.05,
) -> cub.UnlockCandidate:
    return cub.UnlockCandidate(
        strategy_id=strategy_id,
        instrument=instrument,
        session=session,
        orb_minutes=orb_minutes,
        entry_model="E2",
        rr_target=rr_target,
        filter_type=filter_type,
        confirm_bars=1,
        trailing_expr=trailing_expr,
        trailing_n=84,
        annual_r_estimate=annual_r,
        status=status,
        status_reason="candidate",
        chordia_verdict=chordia_verdict,
        chordia_audit_age_days=None,
        c8_oos_status="PASSED",
    )


def test_core_inventory_hints_rank_missing_chordia_candidates_first() -> None:
    hints = {
        cub.candidate_key("MNQ", "NYSE_OPEN", 15, "E2", 1.5, "NO_FILTER"): cub.FamilyHint(
            instrument="MNQ",
            session="NYSE_OPEN",
            orb_minutes=15,
            entry_model="E2",
            rr_target=1.5,
            filter_type="NO_FILTER",
            family="MNQ NYSE_OPEN baseline_orb",
            family_role="deployable_candidate",
            family_median_exp_r=0.069,
            family_oos_median=0.051,
            priority=0,
        )
    }
    candidates = (
        _candidate("MNQ_US_DATA_1000_E2_RR2_CB1_NO_FILTER_O30", session="US_DATA_1000", orb_minutes=30, annual_r=30.0),
        _candidate("MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER_O15", annual_r=8.0),
    )

    rows = cub.build_unlock_queue(
        candidates,
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN", "US_DATA_1000"}),
        active_strategy_ids=frozenset(),
        family_hints=hints,
        limit=100,
    )

    assert [row.strategy_id for row in rows] == [
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER_O15",
        "MNQ_US_DATA_1000_E2_RR2_CB1_NO_FILTER_O30",
    ]
    assert rows[0].action == "RUN_STRICT_UNLOCK"
    assert rows[0].family_priority == 0


def test_queue_excludes_active_profile_blocked_and_non_missing_by_default() -> None:
    candidates = (
        _candidate("ACTIVE"),
        _candidate("MES_BLOCKED", instrument="MES"),
        _candidate("SESSION_BLOCKED", session="CME_REOPEN"),
        _candidate("ALREADY_PASS", chordia_verdict="PASS_CHORDIA"),
        _candidate("NEGATIVE_NOW", annual_r=-1.0, trailing_expr=-0.01),
        _candidate("READY"),
    )

    rows = cub.build_unlock_queue(
        candidates,
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset({"ACTIVE"}),
        family_hints={},
        limit=100,
    )

    assert [row.strategy_id for row in rows] == ["READY"]


def test_report_declares_no_live_mutation_and_next_actions() -> None:
    rows = (
        cub.UnlockQueueRow.from_candidate(
            rank=1,
            candidate=_candidate("READY"),
            family_hint=None,
            active=False,
        ),
    )

    report = cub.render_report(
        rows,
        profile_id="topstep_50k_mnq_auto",
        source_notes=("unit test",),
    )

    assert "does not mutate live allocation" in report
    assert "does not mutate chordia_audit_log" in report
    assert "RUN_STRICT_UNLOCK" in report


def test_inventory_hints_do_not_elevate_single_cell_killed_family(tmp_path: Path) -> None:
    (tmp_path / "family_summary.csv").write_text(
        "\n".join(
            [
                "family,mechanism,role,instrument,session,cells,median_exp_r,mean_exp_r,best_exp_r,"
                "bh_pass_cells,t3_cells,deployable_candidates,provisional_cells,unsupported_cells,min_n_is,max_n_is,median_oos",
                "baseline_orb,ORB,standalone,MNQ,SINGAPORE_OPEN,18,-0.01,0.01,0.20,1,1,1,0,17,100,200,0.12",
            ]
        ),
        encoding="utf-8",
    )
    cells = tmp_path / "cells.csv"
    cells.write_text(
        "\n".join(
            [
                "family,mechanism,role,variant,instrument,session,entry_model,orb_minutes,rr_target,"
                "k_family,n_is,exp_r_is,t_stat,p_value,q_value,wfe,era_dead,n_oos,exp_r_oos,mean_cost_to_risk,classification",
                "baseline_orb,ORB,standalone,NO_FILTER,MNQ,SINGAPORE_OPEN,E2,30,2.0,"
                "594,1702,0.112,4.29,0.001,0.01,1.38,False,85,0.13,0.02,DEPLOYABLE_CANDIDATE_NEEDS_FORMAL_PREREG",
            ]
        ),
        encoding="utf-8",
    )

    hints = cub.load_family_hints_from_cells(cells)

    assert hints == {}
