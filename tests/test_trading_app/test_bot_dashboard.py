"""Focused tests for dashboard metadata, operator state, and legacy compatibility."""

import asyncio
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from trading_app.live import bot_dashboard
from trading_app.live.bot_dashboard import (
    _build_operator_payload,
    _choose_operator_profile,
    _derive_operator_state,
    _legacy_lanes_to_lane_cards,
    _parse_preflight_output,
    _profile_session_ambiguity,
    _strategy_meta,
)
from trading_app.live.bot_state import build_state_snapshot
from trading_app.prop_profiles import ACCOUNT_PROFILES, DailyLaneSpec


def _with_shared_nyse_open(profile_id: str):
    """Return a patched copy of profile_id with MNQ + MES lanes on NYSE_OPEN.

    The 2026-04-19 rebuild of topstep_50k_type_a removed the second NYSE_OPEN
    lane that these tests relied on for shared-session ambiguity coverage.
    The logic under test (_legacy_lanes_to_lane_cards, _profile_session_
    ambiguity, _resolve_session_lane) still needs a shared-session profile
    to exercise — construct one explicitly rather than depending on live
    profile drift.
    """
    base = ACCOUNT_PROFILES[profile_id]
    extra_lanes = tuple(base.daily_lanes) + (
        DailyLaneSpec(
            "MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50",
            "MNQ",
            "NYSE_OPEN",
            max_orb_size_pts=117.8,
        ),
        DailyLaneSpec(
            "MES_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12",
            "MES",
            "NYSE_OPEN",
            max_orb_size_pts=60.0,
        ),
    )
    # Replace any existing NYSE_OPEN lane so the two lanes above are the
    # only NYSE_OPEN lanes (guarantees shared-session ambiguity).
    non_nyse = tuple(lane for lane in base.daily_lanes if lane.orb_label != "NYSE_OPEN")
    return replace(
        base,
        daily_lanes=non_nyse + extra_lanes[-2:],
        allowed_instruments=frozenset(base.allowed_instruments | {"MNQ", "MES"}),
        allowed_sessions=frozenset(base.allowed_sessions | {"NYSE_OPEN"}),
    )


def test_build_state_snapshot_uses_explicit_trading_day_for_session_times():
    strategy = SimpleNamespace(
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL",
        instrument="MNQ",
        orb_label="NYSE_OPEN",
        filter_type="ATR70_VOL",
        rr_target=1.0,
        orb_minutes=5,
        entry_model="E2",
        confirm_bars=1,
    )

    snapshot = build_state_snapshot(
        mode="SIGNAL",
        instrument="MNQ",
        contract="MNQM6",
        trading_day=date(2026, 4, 3),
        account_id=0,
        account_name="profile_topstep_50k_mnq_auto",
        daily_pnl_r=0.0,
        daily_loss_limit_r=-5.0,
        max_equity_dd_r=None,
        bars_received=12,
        strategies=[strategy],
        active_trades=[],
        completed_trades=[],
    )

    assert snapshot["trading_day"] == "2026-04-03"
    assert snapshot["lane_cards"][0]["session_time_brisbane"] == "23:30"


def test_strategy_meta_extracts_human_readable_lane_fields():
    meta = _strategy_meta("MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6", date(2026, 4, 3))

    assert meta["instrument_label"] == "MGC"
    assert meta["session_name"] == "CME_REOPEN"
    assert meta["session_time_brisbane"] == "08:00"
    assert meta["entry_model"] == "E2"
    assert meta["rr_target"] == 2.5
    assert meta["confirm_bars"] == 1
    assert meta["filter_type"] == "ORB_G6"
    assert meta["lane_label"] == "MGC CME_REOPEN"


def test_legacy_lanes_reconstruct_all_profile_lanes_and_mark_ambiguous_shared_sessions():
    """Shared-session disambiguation: legacy-lane dict with one NYSE_OPEN
    entry must reconstruct cards for BOTH NYSE_OPEN lanes and mark the
    non-matching one UNKNOWN/cannot-disambiguate.
    """
    synthetic = _with_shared_nyse_open("topstep_50k_type_a")
    mnq_nyse = next(
        lane for lane in synthetic.daily_lanes if lane.strategy_id == "MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50"
    )
    mes_nyse = next(
        lane for lane in synthetic.daily_lanes if lane.strategy_id == "MES_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12"
    )

    with patch.dict(ACCOUNT_PROFILES, {"topstep_50k_type_a": synthetic}):
        cards = _legacy_lanes_to_lane_cards(
            lanes={
                "NYSE_OPEN": {
                    "strategy_id": mnq_nyse.strategy_id,
                    "status": "IN_TRADE",
                    "direction": "long",
                    "entry_price": 21543.25,
                    "current_pnl_r": 0.8,
                    "rr_target": 1.0,
                    "orb_minutes": 5,
                    "filter_type": "OVNRNG_50",
                }
            },
            trading_day=date(2026, 4, 3),
            account_name="profile_topstep_50k_type_a",
        )

    assert len(cards) == len(synthetic.daily_lanes)

    mnq_card = next(card for card in cards if card["strategy_id"] == mnq_nyse.strategy_id)
    mes_card = next(card for card in cards if card["strategy_id"] == mes_nyse.strategy_id)

    assert mnq_card["status"] == "IN_TRADE"
    assert mnq_card["direction"] == "long"
    assert mnq_card["session_time_brisbane"] == "23:30"

    assert mes_card["status"] == "UNKNOWN"
    assert mes_card["status_detail"] is not None
    assert "cannot disambiguate" in mes_card["status_detail"]
    assert mes_card["session_time_brisbane"] == "23:30"


def test_parse_preflight_output_extracts_structured_checks():
    output = """
[1/5] Auth check (projectx)... FAILED: bad creds
[2/5] Portfolio check (MNQ)... OK (6 strategies)
[3/5] Daily features freshness... WARN: stale by 1 day
[4/5] Contract resolution... SKIPPED (auth failed)
[5/5] Component self-tests... OK (all components verified)

Preflight: 2/5 passed
FIX FAILURES before starting a live session.
""".strip()

    parsed = _parse_preflight_output(output)

    assert parsed["passed"] == 2
    assert parsed["total"] == 5
    assert parsed["overall"] == "fail"
    assert parsed["has_failures"] is True
    assert parsed["has_warnings"] is True
    check_map = {check["name"]: check for check in parsed["checks"]}
    assert check_map["Auth check (projectx)"]["status"] == "fail"
    assert check_map["Daily features freshness"]["status"] == "warn"
    assert check_map["Contract resolution"]["status"] == "warn"


def test_derive_operator_state_requires_preflight_before_ready():
    top_state, reason, action = _derive_operator_state(
        raw_mode="STOPPED",
        heartbeat_age_s=9999,
        broker_summary={"enabled_count": 1, "connected_count": 1},
        data_summary={"any_stale": False},
        preflight_summary=None,
    )

    assert top_state == "STOPPED"
    assert "preflight" in reason.lower()
    assert action["id"] == "run_preflight"


def test_derive_operator_state_becomes_ready_after_passing_preflight():
    top_state, _reason, action = _derive_operator_state(
        raw_mode="STOPPED",
        heartbeat_age_s=9999,
        broker_summary={"enabled_count": 1, "connected_count": 1},
        data_summary={"any_stale": False},
        preflight_summary={"status": "pass", "passed": 5, "total": 5},
    )

    assert top_state == "READY"
    assert action["id"] == "start_signal"


def test_derive_operator_state_marks_stale_running_session():
    top_state, reason, action = _derive_operator_state(
        raw_mode="LIVE",
        heartbeat_age_s=180,
        broker_summary={"enabled_count": 1, "connected_count": 1},
        data_summary={"any_stale": False},
        preflight_summary={"status": "pass"},
    )

    assert top_state == "STALE"
    assert "stale" in reason.lower()
    assert action["id"] == "stop_session"


def test_profile_session_ambiguity_passes_for_shared_session_profiles():
    """With MNQ+MES lanes on NYSE_OPEN, ambiguity-detection reports that the
    shared session is supported (not a blocking error) and names the session.
    """
    synthetic = _with_shared_nyse_open("topstep_50k_type_a")
    with patch.dict(ACCOUNT_PROFILES, {"topstep_50k_type_a": synthetic}):
        result = _profile_session_ambiguity("topstep_50k_type_a")

    assert result["status"] == "pass"
    assert "NYSE_OPEN" in result["detail"]
    assert "supported" in result["detail"]


def test_choose_operator_profile_prefers_running_state_then_first_active_auto_profile():
    assert _choose_operator_profile(None, {"account_name": "profile_topstep_50k_mes_auto"}) == "topstep_50k_mes_auto"
    assert _choose_operator_profile(None, {}) == "topstep_50k_mnq_auto"


def test_build_operator_payload_includes_recent_alert_check(monkeypatch):
    monkeypatch.setattr(
        bot_dashboard,
        "read_state",
        lambda: {"mode": "SIGNAL", "heartbeat_age_s": 12, "account_name": "profile_topstep_50k_mnq_auto"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_broker_status",
        lambda: {"enabled_count": 1, "connected_count": 1, "status": "ok"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_data_status",
        lambda: {"status": "ok", "any_stale": False, "instruments": {}},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_alert_summary",
        lambda **_: {
            "status": "ok",
            "alerts": [{"message": "FEED STALE: 180s no data (check 2)"}],
            "total": 1,
            "counts": {"critical": 0, "warning": 1, "info": 0},
            "recent_window_minutes": 30,
            "recent_counts": {"critical": 0, "warning": 1, "info": 0},
            "latest": {"message": "FEED STALE: 180s no data (check 2)"},
        },
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_profile_session_ambiguity",
        lambda profile_id: {"status": "pass", "detail": f"ok:{profile_id}"},
    )
    monkeypatch.setitem(
        bot_dashboard._preflight_cache,
        "topstep_50k_mnq_auto",
        {"status": "pass", "passed": 5, "total": 5},
    )

    payload = _build_operator_payload("topstep_50k_mnq_auto")

    alerts_check = next(check for check in payload["checks"] if check["name"] == "Alerts")
    assert alerts_check["status"] == "warn"
    assert "FEED STALE" in alerts_check["detail"]


def test_build_operator_payload_includes_conditional_overlay_check(monkeypatch):
    monkeypatch.setattr(
        bot_dashboard,
        "read_state",
        lambda: {"mode": "STOPPED", "heartbeat_age_s": 9999, "account_name": "profile_topstep_50k"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_broker_status",
        lambda: {"enabled_count": 1, "connected_count": 1, "status": "ok"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_data_status",
        lambda: {"status": "ok", "any_stale": False, "instruments": {}},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_alert_summary",
        lambda **_: {
            "status": "ok",
            "alerts": [],
            "total": 0,
            "counts": {"critical": 0, "warning": 0, "info": 0},
            "recent_window_minutes": 30,
            "recent_counts": {"critical": 0, "warning": 0, "info": 0},
            "latest": None,
        },
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_profile_session_ambiguity",
        lambda profile_id: {"status": "pass", "detail": f"ok:{profile_id}"},
    )
    monkeypatch.setitem(
        bot_dashboard._preflight_cache,
        "topstep_50k",
        {"status": "pass", "passed": 5, "total": 5},
    )

    overlay_state = {
        "available": True,
        "valid": True,
        "overlays": [
            {
                "overlay_id": "pr48_mgc_cont_exec_v1",
                "valid": True,
                "status": "ready",
                "summary": {"ready_count": 4, "row_count": 18},
            }
        ],
    }
    with patch(
        "trading_app.lifecycle_state.read_lifecycle_state", return_value={"conditional_overlays": overlay_state}
    ):
        payload = _build_operator_payload("topstep_50k")

    overlay_check = next(check for check in payload["checks"] if check["name"] == "Conditional overlays")
    assert overlay_check["status"] == "info"
    assert "pr48_mgc_cont_exec_v1 ready" in overlay_check["detail"]
    assert payload["conditional_overlays"] == overlay_state


def test_api_alerts_returns_recent_runtime_alerts(monkeypatch):
    monkeypatch.setattr(
        bot_dashboard,
        "read_operator_alerts",
        lambda limit=25, profile=None, mode=None: [
            {
                "timestamp_utc": "2026-04-13T09:15:00+00:00",
                "level": "critical",
                "category": "feed_dead",
                "message": "FEED DEAD: all reconnect attempts exhausted for MNQ",
                "profile": profile,
                "mode": mode,
            }
        ],
    )
    monkeypatch.setattr(
        bot_dashboard,
        "summarize_operator_alerts",
        lambda alerts: {
            "total": len(alerts),
            "counts": {"critical": 1, "warning": 0, "info": 0},
            "recent_window_minutes": 30,
            "recent_counts": {"critical": 1, "warning": 0, "info": 0},
            "latest": alerts[0] if alerts else None,
        },
    )

    payload = asyncio.run(bot_dashboard.api_alerts(limit=20, profile="topstep_50k_type_a", mode="SIGNAL"))
    assert payload["alerts"][0]["profile"] == "topstep_50k_type_a"
    assert payload["alerts"][0]["mode"] == "SIGNAL"
