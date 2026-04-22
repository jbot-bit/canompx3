from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import render_mnq_discovery_capsule


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_capsule_prioritizes_unsolved_hiroi_frontier(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "audit" / "results"
    _mkfile(
        results / "2026-04-22-mnq-layered-candidate-board-v1.csv",
        "\n".join(
            [
                "orb_label,rr_target,direction,signal,n_on_is,n_off_is,expr_on_is,expr_off_is,delta_is,winrate_on_is,winrate_off_is,welch_t,welch_p,n_on_oos,n_off_oos,expr_on_oos,expr_off_oos,delta_oos,same_sign_oos,role,bh_p,abs_delta_is,priority_rank",
                "US_DATA_1000,1.0,long,F5_BELOW_PDL,136,745,0.32,-0.01,0.33,0.69,0.52,4.0,0.0,8,27,-0.02,-0.06,0.04,True,TAKE,0.01,0.33,10336",
                "NYSE_OPEN,1.5,long,F3_NEAR_PIVOT_15,216,614,-0.11,0.15,-0.26,0.37,0.48,-2.8,0.0,4,24,-0.38,0.33,-0.72,True,AVOID,0.09,0.26,10252",
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
                "COMEX_SETTLE,1.0,long,PD_GO_LONG,390,397,0.14,0.02,0.12,1.8,0.0,17,17,0.12,-0.10,0.22,True,False,0.15,0.12",
                "US_DATA_1000,1.0,long,PD_GO_LONG,324,400,0.19,-0.07,0.26,3.9,0.0,15,20,0.62,0.00,0.62,True,True,0.0,0.26",
            ]
        ),
    )
    state_path = tmp_path / ".session" / "mnq_autonomous_discovery_state.json"
    _mkfile(
        state_path,
        json.dumps(
            {
                "history": [
                    {
                        "summary": "Parked exact geometry queue pending non-geometry shortlist.",
                        "status": "parked",
                        "next_focus": "Refresh non-geometry shortlist.",
                        "continue_running": False,
                    }
                ]
            }
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

    capsule = render_mnq_discovery_capsule.build_capsule(tmp_path, state_path)

    assert "MNQ hiROI Discovery Capsule" in capsule
    assert "HANDOFF.md` is a cross-tool baton" in capsule
    assert "Top Family Candidates" in capsule
    assert "Review Batch" in capsule
    assert "Top Transfer Candidates" in capsule
    assert "Top Cell Candidates" in capsule
    assert "- families: 0" in capsule
    assert "- transfers: 1" in capsule
    assert "- cells: 1" in capsule
    assert "US_DATA_1000 RR1.0 long | TAKE F5_BELOW_PDL" not in capsule
    assert "US_DATA_1000 RR1.0 long | TAKE TAKE_DOWNSIDE_DISPLACEMENT" not in capsule
    assert "COMEX_SETTLE RR1.0 long | TAKE PD_CLEAR_LONG" not in capsule
    assert "NYSE_OPEN RR1.5 long | AVOID F3_NEAR_PIVOT_15" in capsule
    assert "COMEX_SETTLE RR1.0 long | TAKE PD_GO_LONG" in capsule
    assert "Refresh non-geometry shortlist." in capsule
