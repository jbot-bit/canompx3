import duckdb

from scripts.audits import AuditPhase
from scripts.audits import phase_7_live_trading as phase7


def _lane(strategy_id: str) -> dict:
    return {
        "profile_id": "topstep_50k_mnq_auto",
        "strategy_id": strategy_id,
        "instrument": "MNQ",
        "orb_label": "EUROPE_FLOW",
        "entry_model": "E2",
        "rr_target": 1.5,
        "confirm_bars": 1,
        "filter_type": "ORB_G5",
        "orb_minutes": 5,
        "lane_name": "EUROPE_FLOW",
        "stop_multiplier": 0.75,
        "is_half_size": False,
        "shadow_only": False,
        "execution_notes": "",
        "max_orb_size_pts": 120.0,
    }


def _mkcon():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE validated_setups (
            strategy_id TEXT,
            instrument TEXT,
            orb_label TEXT,
            entry_model TEXT,
            filter_type TEXT,
            status TEXT,
            deployment_scope TEXT
        )
        """
    )
    con.execute(
        """
        CREATE VIEW deployable_validated_setups AS
        SELECT *
        FROM validated_setups
        WHERE LOWER(status) = 'active'
          AND LOWER(COALESCE(deployment_scope, 'deployable')) = 'deployable'
        """
    )
    return con


def test_phase7_passes_when_active_profile_lanes_match_deployable_shelf(monkeypatch):
    con = _mkcon()
    con.execute(
        """
        INSERT INTO validated_setups VALUES
        ('MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', 'MNQ', 'EUROPE_FLOW', 'E2', 'ORB_G5', 'active', 'deployable')
        """
    )
    audit = AuditPhase(phase_num=7, name="Live Trading Readiness")
    monkeypatch.setattr(phase7, "get_active_profile_ids", lambda **_: ["topstep_50k_mnq_auto"])
    monkeypatch.setattr(
        phase7,
        "get_profile_lane_definitions",
        lambda profile_id: [_lane("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5")],
    )

    phase7._check_live_config_coherence(audit, con)

    assert audit.findings == []


def test_phase7_flags_orphan_when_active_profile_lane_missing_from_deployable_shelf(monkeypatch):
    con = _mkcon()
    audit = AuditPhase(phase_num=7, name="Live Trading Readiness")
    monkeypatch.setattr(phase7, "get_active_profile_ids", lambda **_: ["topstep_50k_mnq_auto"])
    monkeypatch.setattr(
        phase7,
        "get_profile_lane_definitions",
        lambda profile_id: [_lane("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5")],
    )

    phase7._check_live_config_coherence(audit, con)

    assert len(audit.findings) == 1
    assert audit.findings[0].tag == "ORPHAN_STRATEGY"


def test_phase7_flags_parity_violation_when_lane_metadata_mismatches_shelf(monkeypatch):
    con = _mkcon()
    con.execute(
        """
        INSERT INTO validated_setups VALUES
        ('MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', 'MNQ', 'NYSE_OPEN', 'E2', 'ORB_G5', 'active', 'deployable')
        """
    )
    audit = AuditPhase(phase_num=7, name="Live Trading Readiness")
    monkeypatch.setattr(phase7, "get_active_profile_ids", lambda **_: ["topstep_50k_mnq_auto"])
    monkeypatch.setattr(
        phase7,
        "get_profile_lane_definitions",
        lambda profile_id: [_lane("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5")],
    )

    phase7._check_live_config_coherence(audit, con)

    tags = {finding.tag for finding in audit.findings}
    assert "PARITY_VIOLATION" in tags
