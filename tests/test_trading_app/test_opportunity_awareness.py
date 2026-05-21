from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_app import opportunity_awareness


def _lane(strategy_id: str, *, session: str = "COMEX_SETTLE", status: str = "DEPLOY") -> dict:
    return {
        "strategy_id": strategy_id,
        "instrument": "MNQ",
        "orb_label": session,
        "orb_minutes": 5,
        "entry_model": "E2",
        "rr_target": 1.5,
        "confirm_bars": 1,
        "filter_type": "OVNRNG_100",
        "status": status,
        "status_reason": "Session HOT (+0.05)",
        "trailing_expr": 0.2412,
        "session_regime": "HOT",
        "chordia_verdict": "PASS_CHORDIA",
        "p90_orb_pts": 48.5,
    }


def test_build_opportunity_snapshot_classifies_prime_and_blocked(monkeypatch):
    prime = _lane("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100")
    blocked = _lane("MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15", session="US_DATA_1000")

    monkeypatch.setattr(opportunity_awareness, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k_mnq_auto")
    monkeypatch.setattr(opportunity_awareness, "get_profile_lane_definitions", lambda _profile: [prime, blocked])

    snapshot = opportunity_awareness.build_opportunity_snapshot(
        "topstep_50k_mnq_auto",
        trading_day=date(2026, 5, 13),
        lifecycle={
            "strategy_states": {
                blocked["strategy_id"]: {
                    "blocked": True,
                    "block_source": "sr_alarm",
                    "block_reason": "Criterion 12 SR ALARM",
                    "sr_status": "ALARM",
                }
            }
        },
        allocation_payload={"lanes": [prime, blocked], "paused": []},
    )

    by_id = {row.strategy_id: row for row in snapshot.lanes}
    assert by_id[prime["strategy_id"]].opportunity_tier == "PRIME_SHADOW"
    assert by_id[prime["strategy_id"]].blockers == ()
    assert by_id[blocked["strategy_id"]].opportunity_tier == "BLOCKED"
    assert "Criterion 12 SR ALARM" in by_id[blocked["strategy_id"]].blockers
    assert snapshot.summary["prime_shadow_count"] == 1
    assert snapshot.summary["blocked_count"] == 1


def test_refresh_opportunity_state_writes_shadow_envelope(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    lane = _lane("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100")

    monkeypatch.setattr(opportunity_awareness, "STATE_DIR", state_dir)
    monkeypatch.setattr(opportunity_awareness, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k_mnq_auto")
    monkeypatch.setattr(opportunity_awareness, "get_profile", lambda _profile: object())
    monkeypatch.setattr(opportunity_awareness, "get_profile_lane_definitions", lambda _profile: [lane])
    monkeypatch.setattr(opportunity_awareness, "build_profile_fingerprint", lambda _profile: "profile-fingerprint")
    monkeypatch.setattr(opportunity_awareness, "build_db_identity", lambda _db_path: "db-identity")
    monkeypatch.setattr(opportunity_awareness, "build_code_fingerprint", lambda _paths: "code-identity")
    monkeypatch.setattr(opportunity_awareness, "get_git_head", lambda _root=None: "testsha")

    state = opportunity_awareness.refresh_opportunity_state(
        "topstep_50k_mnq_auto",
        db_path=Path("/tmp/gold.db"),
        today=date(2026, 5, 13),
        lifecycle={"strategy_states": {}},
        allocation_payload={"lanes": [lane], "paused": []},
    )

    assert state["state_type"] == "opportunity_awareness_shadow"
    assert state["payload"]["summary"]["prime_shadow_count"] == 1
    assert opportunity_awareness.get_opportunity_state_path("topstep_50k_mnq_auto").exists()


def test_lane_code_paths_hash_only_source_files():
    paths = opportunity_awareness._lane_code_paths()

    assert opportunity_awareness.LANE_ALLOCATION_PATH not in paths
    assert all(path.suffix == ".py" for path in paths)


def test_describe_opportunity_awareness_names_relevant_lanes():
    state = {
        "available": True,
        "valid": True,
        "summary": {
            "lane_count": 3,
            "prime_shadow_count": 1,
            "watch_count": 1,
            "blocked_count": 1,
        },
        "lanes": [
            {
                "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "opportunity_tier": "PRIME_SHADOW",
                "trailing_expr": 0.241,
            },
            {
                "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
                "instrument": "MNQ",
                "orb_label": "NYSE_OPEN",
                "opportunity_tier": "WATCH",
                "warnings": ["Allocation status is PROVISIONAL"],
            },
            {
                "strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP",
                "instrument": "MNQ",
                "orb_label": "US_DATA_1000",
                "opportunity_tier": "BLOCKED",
                "blockers": ["Criterion 12 SR ALARM"],
            },
        ],
    }

    status, detail = opportunity_awareness.describe_opportunity_awareness(state)

    assert status == "warn"
    assert "1 PRIME_SHADOW" in detail
    assert "prime: COMEX_SETTLE/MNQ" in detail
    assert "watch: NYSE_OPEN/MNQ (Allocation status is PROVISIONAL)" in detail
    assert "blocked: US_DATA_1000/MNQ (Criterion 12 SR ALARM)" in detail


def test_read_opportunity_state_refreshes_when_lifecycle_is_supplied(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lane = _lane("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100")

    monkeypatch.setattr(opportunity_awareness, "STATE_DIR", state_dir)
    monkeypatch.setattr(opportunity_awareness, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k_mnq_auto")
    monkeypatch.setattr(opportunity_awareness, "get_profile", lambda _profile: object())
    monkeypatch.setattr(opportunity_awareness, "get_profile_lane_definitions", lambda _profile: [lane])
    monkeypatch.setattr(opportunity_awareness, "build_profile_fingerprint", lambda _profile: "profile-fingerprint")
    monkeypatch.setattr(opportunity_awareness, "build_db_identity", lambda _db_path: "db-identity")
    monkeypatch.setattr(opportunity_awareness, "build_code_fingerprint", lambda _paths: "code-identity")
    monkeypatch.setattr(opportunity_awareness, "get_git_head", lambda _root=None: "testsha")
    monkeypatch.setattr(opportunity_awareness, "_load_allocation_payload", lambda: {"lanes": [lane], "paused": []})

    opportunity_awareness.refresh_opportunity_state(
        "topstep_50k_mnq_auto",
        db_path=Path("/tmp/gold.db"),
        today=date(2026, 5, 13),
        lifecycle={"strategy_states": {}},
        allocation_payload={"lanes": [lane], "paused": []},
    )

    refreshed = opportunity_awareness.read_opportunity_state(
        "topstep_50k_mnq_auto",
        db_path=Path("/tmp/gold.db"),
        today=date(2026, 5, 13),
        lifecycle={
            "strategy_states": {
                lane["strategy_id"]: {
                    "blocked": True,
                    "block_reason": "Paused by current lifecycle",
                }
            }
        },
    )

    assert refreshed["summary"]["blocked_count"] == 1
    assert refreshed["lanes"][0]["opportunity_tier"] == "BLOCKED"
    assert "Paused by current lifecycle" in refreshed["lanes"][0]["blockers"]
