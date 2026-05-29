from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from scripts.tools import live_readiness_report


def _write_allocator(
    path: Path,
    *,
    profile_id: str = "topstep_50k_mnq_auto",
    lane_status: str = "DEPLOY",
) -> None:
    path.write_text(
        json.dumps(
            {
                "profile_id": profile_id,
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
                        "status": lane_status,
                        "status_reason": "selected",
                        "chordia_verdict": "PASS_CHORDIA",
                    }
                ],
                "paused": [],
                "stale": [],
            }
        ),
        encoding="utf-8",
    )


def _install_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    allocation_path: Path,
    *,
    criterion11: dict[str, object] | None = None,
    criterion12: dict[str, object] | None = None,
    strategy_state: dict[str, object] | None = None,
    telemetry: dict[str, object] | None = None,
    stages: dict[str, object] | None = None,
    validated_ids: list[str] | None = None,
) -> None:
    _write_allocator(allocation_path)
    monkeypatch.setattr(
        live_readiness_report,
        "resolve_profile_id",
        lambda *_args, **_kwargs: "topstep_50k_mnq_auto",
    )
    monkeypatch.setattr(
        live_readiness_report,
        "get_profile_lane_definitions",
        lambda _profile_id: [
            {
                "strategy_id": "SID_A",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "orb_minutes": 5,
                "rr_target": 1.0,
                "filter_type": "OVNRNG_100",
            }
        ],
    )
    monkeypatch.setattr(
        live_readiness_report,
        "_load_validated_strategy_ids",
        lambda _db_path: validated_ids or ["SID_A"],
    )
    monkeypatch.setattr(
        live_readiness_report,
        "get_profile",
        lambda _profile_id: SimpleNamespace(
            profile_id="topstep_50k_mnq_auto",
            firm="topstep",
            account_size=50_000,
            copies=1,
            daily_loss_dollars=450.0,
        ),
    )
    monkeypatch.setattr(
        live_readiness_report,
        "read_lifecycle_state",
        lambda *_args, **_kwargs: {
            "criterion11": criterion11 or {"gate_ok": True, "gate_msg": "pass", "report_age_days": 1},
            "criterion12": criterion12 or {"valid": True, "counts": {"ALARM": 0}, "state_age_days": 0},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
            "conditional_overlays": {"available": True, "overlays": []},
            "blocked_strategy_ids": ["SID_A"] if (strategy_state or {}).get("blocked") else [],
            "blocked_reason_by_strategy": (
                {"SID_A": str((strategy_state or {}).get("block_reason"))}
                if (strategy_state or {}).get("block_reason")
                else {}
            ),
            "strategy_states": {
                "SID_A": {
                    "blocked": False,
                    "block_source": None,
                    "block_reason": None,
                    "sr_status": "CONTINUE",
                    "sr_review_outcome": "watch_review_pass",
                    **(strategy_state or {}),
                }
            },
        },
    )
    monkeypatch.setattr(live_readiness_report, "_git_branch", lambda _root: "test-branch")
    monkeypatch.setattr(live_readiness_report, "_git_head", lambda _root: "deadbeef")
    monkeypatch.setattr(
        live_readiness_report,
        "evaluate_telemetry_maturity",
        lambda *_args, **_kwargs: (
            telemetry
            or {
                "verdict": "TELEMETRY_MATURE",
                "instrument": "MNQ",
                "profile_id": "topstep_50k_mnq_auto",
                "scope": "profile",
                "profile_scoped": True,
                "n_unique_trading_days": 30,
                "min_required": 30,
                "trading_days": ["2026-05-01"],
                "signal_files_scanned": 3,
                "records_scanned": 80,
                "records_qualifying": 40,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        live_readiness_report,
        "_evaluate_live_stage_acceptance",
        lambda: (
            stages
            or {
                "stages": [
                    {
                        "path": "docs/runtime/stages/2026-05-22-live-bar-ring-chart.md",
                        "green": True,
                        "status_text": "CLOSED",
                    },
                    {
                        "path": "docs/runtime/stages/2026-05-26-ring-orphan-startup-sweep.md",
                        "green": True,
                        "status_text": "CLOSED",
                    },
                ]
            }
        ),
        raising=False,
    )


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


def test_paused_status_inside_allocator_lanes_is_not_active(tmp_path: Path, monkeypatch) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "profile_id": "topstep_50k_mnq_auto",
                "rebalance_date": "2026-05-03",
                "lanes": [
                    {
                        "strategy_id": "SID_A",
                        "instrument": "MNQ",
                        "orb_label": "COMEX_SETTLE",
                        "orb_minutes": 5,
                        "rr_target": 1.0,
                        "filter_type": "OVNRNG_100",
                        "status": "DEPLOY",
                    },
                    {
                        "strategy_id": "SID_B",
                        "instrument": "MNQ",
                        "orb_label": "NYSE_OPEN",
                        "orb_minutes": 5,
                        "rr_target": 1.0,
                        "filter_type": "COST_LT12",
                        "status": "PAUSE",
                        "status_reason": "SR alarm",
                    },
                ],
                "paused": [],
                "stale": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(live_readiness_report, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k_mnq_auto")
    monkeypatch.setattr(
        live_readiness_report,
        "get_profile_lane_definitions",
        lambda _profile_id: [{"strategy_id": "SID_A"}, {"strategy_id": "SID_B"}],
    )
    monkeypatch.setattr(live_readiness_report, "_load_validated_strategy_ids", lambda _db_path: ["SID_A", "SID_B"])
    monkeypatch.setattr(
        live_readiness_report,
        "read_lifecycle_state",
        lambda *_args, **_kwargs: {
            "criterion11": {"gate_ok": True},
            "criterion12": {"valid": True},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
            "conditional_overlays": {"available": True, "overlays": []},
            "blocked_strategy_ids": [],
            "blocked_reason_by_strategy": {},
            "strategy_states": {
                "SID_A": {"blocked": False},
                "SID_B": {"blocked": True, "block_reason": "SR alarm"},
            },
        },
    )
    monkeypatch.setattr(live_readiness_report, "_git_branch", lambda _root: "test")
    monkeypatch.setattr(live_readiness_report, "_git_head", lambda _root: "abc")

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert [lane["strategy_id"] for lane in report["active_lanes"]] == ["SID_A"]
    assert [lane["strategy_id"] for lane in report["allocator_summary"]["paused_lanes"]] == ["SID_B"]


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


def test_falls_back_to_profile_config_when_allocator_profile_mismatched(tmp_path: Path, monkeypatch) -> None:
    """Allocator JSON for a different profile must NOT surface its lanes as active.

    Operator-facing report must fail-closed on profile mismatch — otherwise the
    requested-profile banner would render lanes from a different profile, a
    silent integrity violation. Mismatch stays visible via
    allocator_summary["profile_match"] = False so operators see the problem.
    """
    allocation_path = tmp_path / "lane_allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "profile_id": "topstep_50k_mes_signal",  # different profile
                "rebalance_date": "2026-05-03T06:07:00+00:00",
                "lanes": [
                    {
                        "strategy_id": "WRONG_PROFILE_LANE",
                        "instrument": "MES",
                        "orb_label": "NYSE_OPEN",
                        "orb_minutes": 5,
                        "rr_target": 1.0,
                        "filter_type": "COST_LT12",
                        "status": "DEPLOY",
                        "status_reason": "selected",
                    }
                ],
                "paused": [],
                "stale": [],
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
            {
                "strategy_id": "REQUESTED_PROFILE_LANE",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "orb_minutes": 5,
                "rr_target": 1.0,
                "filter_type": "OVNRNG_100",
            }
        ],
    )
    monkeypatch.setattr(
        live_readiness_report,
        "_load_validated_strategy_ids",
        lambda _db_path: ["REQUESTED_PROFILE_LANE"],
    )
    monkeypatch.setattr(
        live_readiness_report,
        "read_lifecycle_state",
        lambda *_args, **_kwargs: {
            "criterion11": {"gate_ok": True, "gate_msg": "pass", "report_age_days": 1},
            "criterion12": {"valid": True, "counts": {}, "state_age_days": 0},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
            "conditional_overlays": {"available": True, "overlays": []},
            "blocked_strategy_ids": [],
            "blocked_reason_by_strategy": {},
            "strategy_states": {
                "REQUESTED_PROFILE_LANE": {
                    "blocked": False,
                    "block_source": None,
                    "block_reason": None,
                    "sr_status": "CONTINUE",
                },
            },
        },
    )
    monkeypatch.setattr(live_readiness_report, "_git_branch", lambda _root: "test-branch")
    monkeypatch.setattr(live_readiness_report, "_git_head", lambda _root: "deadbeef")

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["allocator_summary"]["profile_match"] is False
    assert report["allocator_summary"]["allocation_profile_id"] == "topstep_50k_mes_signal"
    assert len(report["active_lanes"]) == 1
    assert report["active_lanes"][0]["strategy_id"] == "REQUESTED_PROFILE_LANE"
    assert report["active_lanes"][0]["allocator_bucket"] == "profile_config"
    wrong_ids = {lane["strategy_id"] for lane in report["active_lanes"]}
    assert "WRONG_PROFILE_LANE" not in wrong_ids


def test_strict_zero_warn_blocks_when_telemetry_below_floor(tmp_path: Path, monkeypatch) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        telemetry={
            "verdict": "UNVERIFIED_INSUFFICIENT_TELEMETRY",
            "instrument": "MNQ",
            "n_unique_trading_days": 8,
            "min_required": 30,
            "trading_days": ["2026-05-01", "2026-05-02"],
            "signal_files_scanned": 2,
            "records_scanned": 12,
            "records_qualifying": 8,
        },
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["telemetry_maturity"]["verdict"] == "UNVERIFIED_INSUFFICIENT_TELEMETRY"
    assert report["strict_zero_warn"]["green"] is False
    assert any("telemetry" in blocker.lower() for blocker in report["strict_zero_warn"]["blockers"])
    markdown = live_readiness_report._render_markdown(report)
    assert "Strict zero-warn" in markdown
    assert "Telemetry" in markdown


def test_strict_zero_warn_blocks_mature_instrument_global_telemetry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        telemetry={
            "verdict": "TELEMETRY_MATURE",
            "instrument": "MNQ",
            "n_unique_trading_days": 30,
            "min_required": 30,
            "trading_days": ["2026-05-01"],
            "signal_files_scanned": 30,
            "records_scanned": 120,
            "records_qualifying": 30,
        },
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["telemetry_maturity"]["verdict"] == "TELEMETRY_MATURE"
    assert report["telemetry_maturity"]["scope"] == "instrument_global"
    assert report["telemetry_maturity"]["profile_scoped"] is False
    assert report["strict_zero_warn"]["green"] is False
    assert any("not profile-scoped" in blocker.lower() for blocker in report["strict_zero_warn"]["blockers"])


def test_strict_zero_warn_blocks_when_active_lane_blocked_and_sr_alarm(tmp_path: Path, monkeypatch) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        strategy_state={
            "blocked": True,
            "block_source": "lifecycle",
            "block_reason": "manual hold",
            "sr_status": "ALARM",
            "sr_review_outcome": None,
        },
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["active_lanes"][0]["lifecycle_blocked"] is True
    assert report["active_lanes"][0]["sr_status"] == "ALARM"
    assert report["strict_zero_warn"]["green"] is False
    # An un-reviewed SR alarm IS the cause of the lifecycle block — it is reported
    # once via the richer SR-alarm entry, not double-counted as a separate
    # "lifecycle blocked" line for the same lane.
    blockers = report["strict_zero_warn"]["blockers"]
    sr_alarm_blockers = [b for b in blockers if "sr alarm" in b.lower() and "SID_A" in b]
    lifecycle_blockers = [b for b in blockers if "lifecycle blocked" in b.lower() and "SID_A" in b]
    assert len(sr_alarm_blockers) == 1
    assert lifecycle_blockers == []


def test_strict_zero_warn_blocks_any_active_sr_alarm_even_when_watch_reviewed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        strategy_state={
            "sr_status": "ALARM",
            "sr_review_outcome": "watch",
        },
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["active_lanes"][0]["lifecycle_blocked"] is False
    assert report["active_lanes"][0]["sr_status"] == "ALARM"
    assert report["active_lanes"][0]["sr_review_outcome"] == "watch"
    assert report["strict_zero_warn"]["green"] is False
    assert any("watch reviewed" in blocker.lower() for blocker in report["strict_zero_warn"]["blockers"])


def test_strict_zero_warn_counts_one_active_alarm_once(tmp_path: Path, monkeypatch) -> None:
    """One active SR alarm produces exactly one blocker entry, not a per-lane
    entry plus a redundant Criterion 12 aggregate entry for the same alarm."""
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        criterion12={"valid": True, "counts": {"ALARM": 1}, "state_age_days": 0},
        strategy_state={
            "sr_status": "ALARM",
            "sr_review_outcome": "watch",
        },
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    blockers = report["strict_zero_warn"]["blockers"]
    alarm_related = [b for b in blockers if "alarm" in b.lower()]
    assert len(alarm_related) == 1
    assert "SID_A" in alarm_related[0]
    # The standalone Criterion 12 aggregate must NOT fire when the alarm is
    # already covered by an active-lane entry.
    assert not any("alarm count" in b.lower() or "alarm not on active lane" in b.lower() for b in blockers)


def test_strict_zero_warn_flags_alarm_not_on_active_lane(tmp_path: Path, monkeypatch) -> None:
    """Fail-closed coverage: an SR alarm count exceeding the active-lane alarms
    (i.e. an orphan alarm on a strategy dropped from the active set) still
    produces a blocker via the narrowed Criterion 12 aggregate."""
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        # Two alarms reported by C12, but the single active lane carries no alarm.
        criterion12={"valid": True, "counts": {"ALARM": 2}, "state_age_days": 0},
        strategy_state={"sr_status": "CONTINUE", "sr_review_outcome": None},
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    blockers = report["strict_zero_warn"]["blockers"]
    assert report["strict_zero_warn"]["green"] is False
    aggregate = [b for b in blockers if "alarm not on active lane" in b.lower()]
    assert len(aggregate) == 1
    assert "(2)" in aggregate[0]


def test_strict_zero_warn_flags_alarm_count_mismatch_failclosed(tmp_path: Path, monkeypatch) -> None:
    """Fail-closed integrity: an active lane in ALARM that the SR-file count
    omits (count < active-lane alarms) must still surface the aggregate rather
    than be silently suppressed by a negative subtraction."""
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        # SR file reports 0 alarms, but the active lane shows ALARM — a divergence
        # that must not be silently swallowed.
        criterion12={"valid": True, "counts": {"ALARM": 0}, "state_age_days": 0},
        strategy_state={"sr_status": "ALARM", "sr_review_outcome": "watch"},
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    blockers = report["strict_zero_warn"]["blockers"]
    assert report["strict_zero_warn"]["green"] is False
    # The per-lane SR-alarm entry fires AND the mismatch aggregate fires.
    assert any("sr alarm" in b.lower() and "SID_A" in b for b in blockers)
    assert any("alarm not on active lane" in b.lower() for b in blockers)


def test_strict_zero_warn_blocks_when_live_stage_pending(tmp_path: Path, monkeypatch) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(
        monkeypatch,
        allocation_path,
        stages={
            "stages": [
                {
                    "path": "docs/runtime/stages/2026-05-22-live-bar-ring-chart.md",
                    "green": False,
                    "status_text": "IMPLEMENTATION",
                },
                {
                    "path": "docs/runtime/stages/2026-05-26-ring-orphan-startup-sweep.md",
                    "green": True,
                    "status_text": "CLOSED",
                },
            ]
        },
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["live_stage_acceptance"]["stages"][0]["green"] is False
    assert report["strict_zero_warn"]["green"] is False
    assert any("live stage" in blocker.lower() for blocker in report["strict_zero_warn"]["blockers"])


def test_strict_zero_warn_blocks_multi_copy_without_shadow_loss_protection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    allocation_path = tmp_path / "lane_allocation.json"
    _install_happy_path(monkeypatch, allocation_path)
    monkeypatch.setattr(
        live_readiness_report,
        "get_profile",
        lambda _profile_id: SimpleNamespace(
            profile_id="topstep_50k_mnq_auto",
            firm="topstep",
            account_size=50_000,
            copies=2,
            daily_loss_dollars=450.0,
        ),
    )

    report = live_readiness_report.build_live_readiness_report(
        db_path=tmp_path / "gold.db",
        allocation_path=allocation_path,
    )

    assert report["profile_launch"]["copies"] == 2
    assert report["profile_launch"]["shadow_copy_loss_protection"] is False
    assert report["strict_zero_warn"]["green"] is False
    assert any("copies>1" in blocker for blocker in report["strict_zero_warn"]["blockers"])


def test_main_strict_zero_warn_exits_nonzero_when_not_green(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        live_readiness_report,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "git_head": "deadbeef",
            "strict_zero_warn": {"green": False, "blockers": ["telemetry not mature"]},
        },
    )
    monkeypatch.setattr(live_readiness_report, "_render_text", lambda _report: "Live Readiness")
    monkeypatch.setattr("sys.argv", ["live_readiness_report.py", "--strict-zero-warn"])

    with pytest.raises(SystemExit) as excinfo:
        live_readiness_report.main()

    assert excinfo.value.code == 1
    assert "Live Readiness" in capsys.readouterr().out


def test_main_strict_zero_warn_returns_zero_when_green(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        live_readiness_report,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "git_head": "deadbeef",
            "strict_zero_warn": {"green": True, "blockers": []},
        },
    )
    monkeypatch.setattr(live_readiness_report, "_render_text", lambda _report: "Live Readiness")
    monkeypatch.setattr("sys.argv", ["live_readiness_report.py", "--strict-zero-warn"])

    live_readiness_report.main()

    assert "Live Readiness" in capsys.readouterr().out


def test_json_cli_stdout_is_parseable_without_warning_preamble(tmp_path: Path) -> None:
    # CI has no gold.db by policy (CLAUDE.md "local disk, no cloud sync"), so
    # the subprocess must run against a seeded temp DB rather than the dev's
    # local gold.db. validated_setups is the only hard DB dependency in the
    # report's build path (_load_validated_strategy_ids); seeding it with the
    # two columns the deployable-shelf predicate reads is sufficient — all
    # downstream lifecycle/survival/overlay readers fail-soft on missing tables.
    # DUCKDB_PATH is honored only when the file exists (pipeline/paths.py).
    seed_db = tmp_path / "gold.db"
    seed_con = duckdb.connect(str(seed_db))
    try:
        seed_con.execute("CREATE TABLE validated_setups (strategy_id VARCHAR, status VARCHAR)")
        seed_con.execute("INSERT INTO validated_setups VALUES ('MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100', 'active')")
    finally:
        seed_con.close()

    env = {**os.environ, "DUCKDB_PATH": str(seed_db)}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/live_readiness_report.py",
            "--profile",
            "topstep_50k_mnq_auto",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.lstrip().startswith("{")
    parsed = json.loads(result.stdout)
    assert parsed["profile_id"] == "topstep_50k_mnq_auto"
