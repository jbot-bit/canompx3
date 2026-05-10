from __future__ import annotations

from pathlib import Path

import duckdb

from scripts.tools import adversarial_stress_gate as gate
from scripts.tools import stress_test_chain_integrity


class _DummyContext:
    def __enter__(self):
        return object()

    def __exit__(self, *_exc):
        return False


def _live_report(*, blocked: bool = False) -> dict:
    return {
        "profile_id": "topstep_50k_mnq_auto",
        "git_head": "abc1234",
        "git_branch": "main",
        "deployment_summary": {
            "profile_id": "topstep_50k_mnq_auto",
            "deployed_count": 1,
            "validated_active_count": 99,
            "deployed_not_validated": [],
            "validated_not_deployed": ["OTHER"],
        },
        "criterion11": {"gate_ok": True},
        "criterion12": {"valid": True, "reason": None, "state_age_days": 0},
        "pauses": {"paused_count": 1 if blocked else 0},
        "active_lanes": [
            {
                "strategy_id": "SID_A",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "lifecycle_blocked": blocked,
                "lifecycle_block_reason": "SR alarm" if blocked else None,
                "sr_status": None,
                "sr_review_outcome": None,
                "sr_review_summary": None,
                "paused": blocked,
            }
        ],
        "allocator_summary": {
            "available": True,
            "profile_match": True,
            "active_lanes": [],
            "paused_lanes": [],
            "stale_lanes": [],
        },
    }


def _strategy_row(**overrides) -> dict:
    row = {
        "strategy_id": "SID_A",
        "instrument": "MNQ",
        "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5,
        "entry_model": "E2",
        "rr_target": 1.5,
        "confirm_bars": 1,
        "filter_type": "OVNRNG_100",
        "status": "active",
        "deployment_scope": "deployable",
        "sample_size": 200,
        "years_tested": 5,
        "expectancy_r": 0.2,
        "oos_exp_r": 0.18,
        "wfe": 1.0,
        "dsr_score": 0.0,
        "fdr_adjusted_p": 0.001,
        "discovery_k": 1000,
        "slippage_validation_status": None,
        "c8_oos_status": None,
        "robustness_status": "WHITELISTED",
        "trade_tier": "CORE",
        "pbo": 0.0,
    }
    row.update(overrides)
    return row


def _counts() -> dict:
    return {
        "raw_validated": 847,
        "deployable_rows": 847,
        "unique_streams": 353,
        "edge_families": 527,
        "edge_families_non_purged": 356,
        "rr_locked_non_purged": 346,
        "by_instrument": [{"instrument": "MNQ", "raw_validated": 786}],
    }


def _patch_db(monkeypatch, *, live=None, row=None, fdr=None, counts=None) -> None:
    monkeypatch.setattr(gate, "_connect_ro", lambda _db_path: _DummyContext())
    monkeypatch.setattr(gate, "load_counts", lambda _con: counts or _counts())
    monkeypatch.setattr(gate, "_load_strategy_rows", lambda _con, _ids: {"SID_A": row or _strategy_row()})
    monkeypatch.setattr(
        gate,
        "current_k_fdr",
        lambda _con, _ids: fdr or {"SID_A": {"current_pass": True, "current_adj_p": 0.001}},
    )
    monkeypatch.setattr(gate, "build_live_readiness_report", lambda **_kwargs: live or _live_report())
    monkeypatch.setattr(
        gate,
        "build_deployability_audit",
        lambda **_kwargs: {
            "summary": {"deployable_candidates": 1, "total_candidates": 1},
            "strategies": [
                {"strategy_id": "SID_A", "deployable": True, "verdict": "DEPLOYABLE_CANDIDATE", "issues": []}
            ],
        },
    )


def test_paused_active_lane_blocks(monkeypatch):
    _patch_db(monkeypatch, live=_live_report(blocked=True))

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["verdict"] == gate.BLOCKED
    assert any(b["id"] == "selected_lane_paused_or_blocked" for b in report["hard_blockers"])


def test_selected_current_k_fdr_failure_blocks(monkeypatch):
    _patch_db(
        monkeypatch,
        fdr={"SID_A": {"current_pass": False, "current_adj_p": 0.051, "stored_adj_p": 0.049}},
    )

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["verdict"] == gate.BLOCKED
    assert any(b["id"] == "selected_lane_current_k_fdr_fail" for b in report["hard_blockers"])


def test_invalid_criterion12_blocks(monkeypatch):
    live = _live_report()
    live["criterion12"] = {"valid": False, "reason": "db identity mismatch", "state_age_days": None}
    _patch_db(monkeypatch, live=live, row=_strategy_row(slippage_validation_status="PASSED", c8_oos_status="PASSED"))

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["verdict"] == gate.BLOCKED
    assert any(b["id"] == "criterion12_invalid" for b in report["hard_blockers"])


def test_reviewed_sr_alarm_is_edge_case_not_hard_block(monkeypatch):
    live = _live_report()
    live["active_lanes"][0]["sr_status"] = "ALARM"
    live["active_lanes"][0]["sr_review_outcome"] = "watch"
    live["active_lanes"][0]["sr_review_summary"] = "reviewed watch"
    _patch_db(monkeypatch, live=live, row=_strategy_row(slippage_validation_status="PASSED", c8_oos_status="PASSED"))

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["verdict"] == gate.GO
    assert any(e["id"] == "selected_lane_sr_alarm_watch" for e in report["edge_cases"])


def test_drift_failure_is_no_go(monkeypatch):
    _patch_db(monkeypatch)
    results = {
        "check_drift": gate.CommandResult("check_drift", [], 1, "FAIL", ""),
        "fdr_integrity": gate.CommandResult("fdr_integrity", [], 0, "PASS", ""),
        "chain_integrity": gate.CommandResult("chain_integrity", [], 0, "PASS", ""),
    }

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        runner=lambda name, _argv, _timeout: results[name],
        run_external_checks=True,
    )

    assert report["verdict"] == gate.NO_GO
    assert any(b["id"] == "drift_failure" for b in report["hard_blockers"])


def test_timeout_fails_closed_as_blocked(monkeypatch):
    _patch_db(monkeypatch)
    results = {
        "check_drift": gate.CommandResult("check_drift", [], None, "", "", timed_out=True),
        "fdr_integrity": gate.CommandResult("fdr_integrity", [], 0, "PASS", ""),
        "chain_integrity": gate.CommandResult("chain_integrity", [], 0, "PASS", ""),
    }

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        runner=lambda name, _argv, _timeout: results[name],
        run_external_checks=True,
    )

    assert report["verdict"] == gate.BLOCKED
    assert any(b["id"] == "tool_timeout" for b in report["hard_blockers"])


def test_chain_integrity_suspect_without_critical_is_edge_case(monkeypatch):
    _patch_db(monkeypatch)
    results = {
        "check_drift": gate.CommandResult("check_drift", [], 0, "PASS", ""),
        "fdr_integrity": gate.CommandResult("fdr_integrity", [], 0, "PASS", ""),
        "chain_integrity": gate.CommandResult(
            "chain_integrity",
            [],
            1,
            "OVERALL: SUSPECT  (critical=0 warnings=4)",
            "",
        ),
    }

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        runner=lambda name, _argv, _timeout: results[name],
        run_external_checks=True,
    )

    assert report["verdict"] == gate.BLOCKED
    assert not any(b["id"] == "chain_integrity_not_clean" for b in report["hard_blockers"])
    assert any(e["id"] == "chain_integrity_warn" for e in report["edge_cases"])


def test_raw_validated_count_is_not_reported_as_capacity(monkeypatch):
    _patch_db(monkeypatch)

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["counts"]["raw_validated"] == 847
    assert report["counts"]["unique_streams"] == 353
    assert report["counts"]["rr_locked_non_purged"] == 346
    assert "capacity" not in report["counts"]


def test_missing_validation_fields_are_labelled_silences(monkeypatch):
    _patch_db(monkeypatch, row=_strategy_row(slippage_validation_status=None, c8_oos_status=None))

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["verdict"] == gate.BLOCKED
    missing = [s for s in report["silences"] if s["id"] == "missing_validation_field"]
    assert {s["field"] for s in missing} == {"slippage_validation_status", "c8_oos_status"}


def test_full_shelf_deployability_blocks_selected_lane(monkeypatch):
    _patch_db(monkeypatch, row=_strategy_row(slippage_validation_status="PASSED", c8_oos_status="PASSED"))
    monkeypatch.setattr(
        gate,
        "build_deployability_audit",
        lambda **_kwargs: {
            "summary": {"deployable_candidates": 0, "total_candidates": 1},
            "strategies": [
                {
                    "strategy_id": "SID_A",
                    "deployable": False,
                    "verdict": "BLOCKED_SLIPPAGE",
                    "issues": [{"id": "slippage_missing", "severity": "hard"}],
                }
            ],
        },
    )

    report = gate.build_gate_report(
        profile_id="topstep_50k_mnq_auto",
        db_path=Path("unused.db"),
        run_external_checks=False,
    )

    assert report["verdict"] == gate.BLOCKED
    assert any(b["id"] == "selected_lane_not_full_shelf_deployable" for b in report["hard_blockers"])


def test_non5m_validated_rows_are_warn_not_fail_by_default(tmp_path: Path):
    db_path = tmp_path / "stress.db"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE validated_setups (
                strategy_id VARCHAR,
                instrument VARCHAR,
                orb_label VARCHAR,
                orb_minutes INTEGER,
                filter_type VARCHAR,
                sample_size INTEGER,
                expectancy_r DOUBLE,
                status VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO validated_setups VALUES
            ('SID_O15', 'MNQ', 'US_DATA_1000', 15, 'VWAP_MID_ALIGNED', 100, 0.2, 'active')
            """
        )

        verdict, findings, data = stress_test_chain_integrity.test_t4_orb_minutes_contamination(con)
        assert verdict == "WARN"
        assert data["legacy_bug_scan"] is False
        assert any("stale_harness_reconciled" in f for f in findings)

        legacy_verdict, _, legacy_data = stress_test_chain_integrity.test_t4_orb_minutes_contamination(
            con,
            legacy_bug_scan=True,
        )
        assert legacy_verdict == "FAIL"
        assert legacy_data["legacy_bug_scan"] is True
    finally:
        con.close()
