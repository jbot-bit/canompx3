from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import build_mnq_discovery_frontier


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_frontier_keeps_expected_candidate_kinds(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "audit" / "results"
    _mkfile(
        results / "2026-04-22-mnq-layered-candidate-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,signal,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,winrate_on_is,winrate_off_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is,priority_rank",
                "US_DATA_1000,1.0,long,F5_BELOW_PDL,136,745,0.32,-0.01,0.33,0.69,0.52,4.0,0.0,8,27,-0.02,-0.06,0.04,True,TAKE,0.01,0.33,10336",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-prior-day-family-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,members,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is",
                "US_DATA_1000,1.0,long,TAKE_DOWNSIDE_DISPLACEMENT,F2 OR F5,205,676,0.23,-0.02,0.25,3.3,0.0,10,25,0.17,-0.14,0.32,True,TAKE,0.0,0.25",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-geometry-transfer-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,is_solved_lane,bh_p,abs_delta_is",
                "COMEX_SETTLE,1.0,long,PD_CLEAR_LONG,303,484,0.18,0.01,0.17,2.6,0.0,15,19,0.13,-0.10,0.23,True,False,0.02,0.17",
            ]
        ),
    )

    frontier = build_mnq_discovery_frontier.build_frontier(
        tmp_path,
        tmp_path / ".session" / "mnq_discovery_frontier_ledger.json",
    )

    assert frontier["candidate_count"] == 3
    assert frontier["queued_count"] == 3
    assert {row["candidate_kind"] for row in frontier["candidates"]} == {"family", "transfer", "cell"}
    assert [row["candidate_kind"] for row in frontier["review_batch"]] == ["family", "transfer"]
    assert frontier["kind_counts"] == {"family": 1, "transfer": 1, "cell": 1}


def test_build_frontier_applies_ledger_status(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "audit" / "results"
    _mkfile(
        results / "2026-04-22-mnq-layered-candidate-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,signal,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,winrate_on_is,winrate_off_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is,priority_rank",
                "US_DATA_1000,1.0,long,F5_BELOW_PDL,136,745,0.32,-0.01,0.33,0.69,0.52,4.0,0.0,8,27,-0.02,-0.06,0.04,True,TAKE,0.01,0.33,10336",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-prior-day-family-board-v1.csv",
        "orb_label,rr_target,direction,family,members,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is\n",
    )
    _mkfile(
        results / "2026-04-22-mnq-geometry-transfer-board-v1.csv",
        "orb_label,rr_target,direction,family,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,is_solved_lane,bh_p,abs_delta_is\n",
    )
    ledger = tmp_path / ".session" / "mnq_discovery_frontier_ledger.json"
    _mkfile(
        ledger,
        json.dumps(
            {
                "cell::US_DATA_1000::1.0::long::F5_BELOW_PDL::TAKE": {
                    "frontier_decision": "parked",
                    "summary": "Already reviewed.",
                }
            }
        ),
    )

    frontier = build_mnq_discovery_frontier.build_frontier(tmp_path, ledger)

    assert frontier["candidates"][0]["status"] == "parked"
    assert frontier["candidates"][0]["last_note"] == "Already reviewed."


def test_build_frontier_review_batch_diversifies_candidate_kinds(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "audit" / "results"
    _mkfile(
        results / "2026-04-22-mnq-layered-candidate-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,signal,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,winrate_on_is,winrate_off_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is,priority_rank",
                "NYSE_OPEN,1.5,long,F6_INSIDE_PDR,492,338,-0.00,0.21,-0.21,0.42,0.50,-2.4,0.0,21,7,-0.47,0.23,-0.70,True,AVOID,0.21,0.21,10150",
                "NYSE_OPEN,1.5,short,F2_NEAR_PDL_15__AND__F5_BELOW_PDL,40,779,0.50,0.08,0.41,0.62,0.45,2.1,0.03,2,38,0.23,-0.03,0.26,True,TAKE,0.30,0.41,10384",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-prior-day-family-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,members,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is",
                "NYSE_OPEN,1.5,long,AVOID_CONGESTION,F3 OR F6,633,197,0.05,0.19,-0.13,-1.3,0.17,21,7,0.06,0.76,-0.70,True,AVOID,0.35,0.13",
                "NYSE_OPEN,1.5,long,TAKE_OVERHEAD_BREAK,F1 OR F4,320,510,0.17,0.03,0.14,1.6,0.09,7,21,0.75,0.05,0.70,True,TAKE,0.29,0.14",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-geometry-transfer-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,is_solved_lane,bh_p,abs_delta_is",
                "COMEX_SETTLE,1.0,long,PD_GO_LONG,390,397,0.14,0.02,0.12,1.8,0.0,17,17,0.12,-0.10,0.22,True,False,0.15,0.12",
                "EUROPE_FLOW,1.0,long,PD_GO_LONG,211,547,0.11,0.01,0.09,1.3,0.18,10,27,0.51,0.18,0.33,True,False,0.33,0.09",
            ]
        ),
    )

    frontier = build_mnq_discovery_frontier.build_frontier(
        tmp_path,
        tmp_path / ".session" / "mnq_discovery_frontier_ledger.json",
    )

    kinds = [row["candidate_kind"] for row in frontier["review_batch"][:4]]
    assert "family" in kinds
    assert "transfer" in kinds
    assert "cell" in kinds


def test_build_frontier_excludes_role_inconsistent_rows(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "audit" / "results"
    _mkfile(
        results / "2026-04-22-mnq-layered-candidate-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,signal,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,winrate_on_is,winrate_off_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is,priority_rank",
                "CME_PRECLOSE,1.0,long,F5_BELOW_PDL,100,300,0.08,0.15,-0.07,0.50,0.55,-0.9,0.7,5,10,-0.02,0.10,-0.12,True,TAKE,0.73,0.07,10000",
                "NYSE_OPEN,1.5,long,F6_INSIDE_PDR,492,338,-0.00,0.21,-0.21,0.42,0.50,-2.4,0.0,21,7,-0.47,0.23,-0.70,True,AVOID,0.21,0.21,10150",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-prior-day-family-board-v1.csv",
        "orb_label,rr_target,direction,family,members,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is\n",
    )
    _mkfile(
        results / "2026-04-22-mnq-geometry-transfer-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,is_solved_lane,bh_p,abs_delta_is",
                "CME_PRECLOSE,1.0,long,PD_GO_LONG,370,300,0.15,0.19,-0.04,-0.5,0.7,13,10,-0.28,0.05,-0.33,True,False,0.73,0.04",
            ]
        ),
    )

    frontier = build_mnq_discovery_frontier.build_frontier(
        tmp_path,
        tmp_path / ".session" / "mnq_discovery_frontier_ledger.json",
    )

    candidate_ids = [row["candidate_id"] for row in frontier["candidates"]]
    assert "cell::CME_PRECLOSE::1.0::long::F5_BELOW_PDL::TAKE" not in candidate_ids
    assert "transfer::CME_PRECLOSE::1.0::long::PD_GO_LONG" not in candidate_ids
    assert "cell::NYSE_OPEN::1.5::long::F6_INSIDE_PDR::AVOID" in candidate_ids


def test_build_frontier_excludes_doc_resolved_candidates(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "audit" / "results"
    _mkfile(
        results / "2026-04-22-mnq-layered-candidate-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,signal,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,winrate_on_is,winrate_off_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is,priority_rank",
                "US_DATA_1000,1.0,long,F3_NEAR_PIVOT_50,618,263,-0.03,0.22,-0.25,0.51,0.64,-3.7,0.0,19,16,-0.18,0.09,-0.27,True,AVOID,0.01,0.25,10251",
                "NYSE_OPEN,1.5,long,F6_INSIDE_PDR,492,338,-0.00,0.21,-0.21,0.42,0.50,-2.4,0.0,21,7,-0.47,0.23,-0.70,True,AVOID,0.21,0.21,10150",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-prior-day-family-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,members,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is",
                "US_DATA_1000,1.0,long,AVOID_CONGESTION,F3 OR F6,649,232,-0.02,0.22,-0.24,-3.3,0.0,22,13,-0.20,0.20,-0.40,True,AVOID,0.0,0.24",
                "NYSE_OPEN,1.5,long,AVOID_CONGESTION,F3 OR F6,633,197,0.05,0.19,-0.13,-1.3,0.17,21,7,0.06,0.76,-0.70,True,AVOID,0.35,0.13",
            ]
        ),
    )
    _mkfile(
        results / "2026-04-22-mnq-geometry-transfer-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,family,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,is_solved_lane,bh_p,abs_delta_is",
                "COMEX_SETTLE,1.0,long,PD_CLEAR_LONG,303,484,0.18,0.01,0.17,2.6,0.0,15,19,0.13,-0.10,0.23,True,False,0.02,0.17",
                "COMEX_SETTLE,1.0,long,PD_GO_LONG,390,397,0.14,0.02,0.12,1.8,0.0,17,17,0.12,-0.10,0.22,True,False,0.15,0.12",
            ]
        ),
    )
    _mkfile(
        tmp_path / "docs" / "plans" / "2026-04-22-mnq-usdata1000-geometry-family-register.md",
        "LOCALLY SOLVED ENOUGH\nCLOSED as non-promotable exact-cell\n",
    )
    _mkfile(
        results / "2026-04-22-mnq-comex-pd-clear-long-take-v1.md",
        "PD_CLEAR_LONG survives the full bridge and promotes.\n",
    )

    frontier = build_mnq_discovery_frontier.build_frontier(
        tmp_path,
        tmp_path / ".session" / "mnq_discovery_frontier_ledger.json",
    )

    candidate_ids = [row["candidate_id"] for row in frontier["candidates"]]
    assert "family::US_DATA_1000::1.0::long::AVOID_CONGESTION" not in candidate_ids
    assert "cell::US_DATA_1000::1.0::long::F3_NEAR_PIVOT_50::AVOID" not in candidate_ids
    assert "transfer::COMEX_SETTLE::1.0::long::PD_CLEAR_LONG" not in candidate_ids
    assert "family::NYSE_OPEN::1.5::long::AVOID_CONGESTION" in candidate_ids
    assert "transfer::COMEX_SETTLE::1.0::long::PD_GO_LONG" in candidate_ids
