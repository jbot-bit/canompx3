from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import live_readiness_report


def test_build_live_readiness_report_merges_allocator_and_lifecycle(tmp_path: Path, monkeypatch) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "profile_id": "topstep_50k_mnq_auto",
                "rebalance_date": "2026-05-03T06:07:00+00:00",
                "trailing_window_months": 3,
                "all_scores_count": 42,
                "lanes": [
                    {
                        "strategy_id": "SID_A",
                        "instrument": "MNQ",
                        "orb_label": "COMEX_SETTLE",
                        "orb_minutes": 5,
                        "rr_target": 1.0,
                        "filter_type": "OVNRNG_100",
                        "status": "DEPLOY",
                        "status_reason": "selected",
                        "chordia_verdict": "PASS_CHORDIA",
                    }
                ],
                "paused": [
                    {
                        "strategy_id": "SID_B",
                        "instrument": "MNQ",
                        "orb_label": "NYSE_OPEN",
                        "orb_minutes": 5,
                        "rr_target": 1.0,
                        "filter_type": "COST_LT12",
                        "status": "PAUSED",
                        "status_reason": "correlation gate",
                    }
                ],
                "stale": [
                    {
                        "strategy_id": "SID_C",
                        "instrument": "MNQ",
                        "orb_label": "US_DATA_1000",
                        "orb_minutes": 5,
                        "rr_target": 1.0,
                        "filter_type": "X_MES_ATR60",
                        "status": "STALE",
                        "status_reason": "needs replay",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        live_readiness_report,
        "resolve_profile_id",
        lambda *_args, **_kwargs: "topstep_50k_mnq_auto",
    )
    monkeypatch.setattr(
        live_readiness_report,
        "get_profile_lane_definitions",
        lambda _profile_id: [
            {"strategy_id": "SID_A", "instrument": "MNQ", "orb_label": "COMEX_SETTLE", "orb_minutes": 5},
            {"strategy_id": "SID_B", "instrument": "MNQ", "orb_label": "NYSE_OPEN", "orb_minutes": 5},
        ],
    )
    monkeypatch.setattr(
        live_readiness_report,
        "_load_validated_strategy_ids",
        lambda _db_path: ["SID_A", "SID_B", "SID_D"],
    )
    monkeypatch.setattr(
        live_readiness_report,
        "read_lifecycle_state",
        lambda *_args, **_kwargs: {
            "criterion11": {"gate_ok": True, "gate_msg": "pass", "report_age_days": 1},
            "criterion12": {"valid": True, "counts": {"ALARM": 1}, "state_age_days": 0},
            "pauses": {"paused_count": 1, "paused_strategy_ids": ["SID_B"]},
            "conditional_overlays": {"available": True, "overlays": []},
            "blocked_strategy_ids": ["SID_B"],
            "blocked_reason_by_strategy": {"SID_B": "Paused pending review"},
            "strategy_states": {
                "SID_A": {"blocked": False, "block_source": None, "block_reason": None, "sr_status": "CONTINUE"},
                "SID_B": {
                    "blocked": True,
                    "block_source": "pause",
                    "block_reason": "Paused pending review",
                    "sr_status": "ALARM",
                    "paused": True,
                    "pause_reason": "manual pause",
                },
                "SID_C": {
                    "blocked": False,
                    "block_source": None,
                    "block_reason": None,
                    "sr_status": "NO_DATA",
                },
            },
        },
    )
    monkeypatch.setattr(live_readiness_report, "_git_branch", lambda _root: "feature/live-readiness")
    monkeypatch.setattr(live_readiness_report, "_git_head", lambda _root: "abc1234")

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["deployment_summary"]["deployed_count"] == 2
    assert report["deployment_summary"]["validated_active_count"] == 3
    assert report["deployment_summary"]["validated_not_deployed"] == ["SID_D"]
    assert report["allocator_summary"]["profile_match"] is True
    assert report["allocator_summary"]["rebalance_date"] == "2026-05-03T06:07:00+00:00"
    assert len(report["active_lanes"]) == 1
    assert report["active_lanes"][0]["strategy_id"] == "SID_A"
    assert report["allocator_summary"]["paused_lanes"][0]["lifecycle_block_reason"] == "Paused pending review"
    assert report["allocator_summary"]["stale_lanes"][0]["status_reason"] == "needs replay"


def test_build_live_readiness_report_falls_back_when_allocator_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(live_readiness_report, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k")
    monkeypatch.setattr(
        live_readiness_report,
        "get_profile_lane_definitions",
        lambda _profile_id: [
            {
                "strategy_id": "SID_Z",
                "instrument": "MES",
                "orb_label": "NYSE_OPEN",
                "orb_minutes": 5,
                "rr_target": 1.0,
                "filter_type": "NONE",
            }
        ],
    )
    monkeypatch.setattr(live_readiness_report, "_load_validated_strategy_ids", lambda _db_path: ["SID_Z"])
    monkeypatch.setattr(
        live_readiness_report,
        "read_lifecycle_state",
        lambda *_args, **_kwargs: {
            "criterion11": {"gate_ok": False, "gate_msg": "blocked", "report_age_days": None},
            "criterion12": {"valid": False, "counts": {}, "state_age_days": None},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
            "conditional_overlays": {"available": False, "overlays": []},
            "blocked_strategy_ids": ["SID_Z"],
            "blocked_reason_by_strategy": {"SID_Z": "Criterion 12 SR ALARM — manual review required"},
            "strategy_states": {
                "SID_Z": {
                    "blocked": True,
                    "block_source": "sr_alarm",
                    "block_reason": "Criterion 12 SR ALARM — manual review required",
                    "sr_status": "ALARM",
                    "paused": False,
                    "pause_reason": None,
                }
            },
        },
    )
    monkeypatch.setattr(live_readiness_report, "_git_branch", lambda _root: "feature/live-readiness")
    monkeypatch.setattr(live_readiness_report, "_git_head", lambda _root: "abc1234")

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=tmp_path / "missing.json",
    )

    assert report["allocator_summary"]["available"] is False
    assert len(report["active_lanes"]) == 1
    assert report["active_lanes"][0]["allocator_bucket"] == "profile_config"
    assert report["active_lanes"][0]["lifecycle_blocked"] is True

    markdown = live_readiness_report._render_markdown(report)

    assert "Live Readiness Report" in markdown
    assert "Criterion 11" in markdown
    assert "Criterion 12" in markdown
