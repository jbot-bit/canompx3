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
    _connection_readiness,
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


_CONNECTED_BROKER_SUMMARY: dict[str, object] = {
    "status": "ok",
    "connections": [{"id": "projectx", "enabled": True, "status": "connected"}],
    "enabled_count": 1,
    "connected_count": 1,
    "error_count": 0,
}


def test_derive_operator_state_allows_start_without_cached_preflight():
    top_state, reason, action = _derive_operator_state(
        raw_mode="STOPPED",
        heartbeat_age_s=9999,
        broker_summary=dict(_CONNECTED_BROKER_SUMMARY),
        data_summary={"any_stale": False},
        preflight_summary=None,
    )

    assert top_state == "READY"
    assert "auto-run readiness checks" in reason
    assert action["id"] == "start_signal"


def test_derive_operator_state_becomes_ready_after_passing_preflight():
    top_state, _reason, action = _derive_operator_state(
        raw_mode="STOPPED",
        heartbeat_age_s=9999,
        broker_summary=dict(_CONNECTED_BROKER_SUMMARY),
        data_summary={"any_stale": False},
        preflight_summary={"status": "pass", "passed": 5, "total": 5},
    )

    assert top_state == "READY"
    assert action["id"] == "start_signal"


def test_derive_operator_state_marks_stale_running_session():
    top_state, reason, action = _derive_operator_state(
        raw_mode="LIVE",
        heartbeat_age_s=180,
        broker_summary=dict(_CONNECTED_BROKER_SUMMARY),
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


def test_connection_readiness_reports_missing_connection():
    broker_summary = {
        "status": "ok",
        "connections": [],
        "enabled_count": 0,
        "connected_count": 0,
        "error_count": 0,
    }

    result = _connection_readiness(broker_summary)

    assert result == {
        "status": "missing",
        "message": "No broker connection configured. Add a connection before starting.",
        "action": "open_connections",
        "connected_count": 0,
        "enabled_count": 0,
    }


def test_build_operator_payload_exposes_connection_readiness_and_blocks_starts(monkeypatch):
    monkeypatch.setattr(
        bot_dashboard,
        "read_state",
        lambda: {"mode": "STOPPED", "heartbeat_age_s": 9999, "account_name": "profile_topstep_50k_mnq_auto"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_broker_status",
        lambda: {"status": "ok", "connections": [], "enabled_count": 0, "connected_count": 0, "error_count": 0},
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

    payload = _build_operator_payload("topstep_50k_mnq_auto")

    assert payload["top_state"] == "BLOCKED"
    assert payload["recommended_action"]["id"] == "open_connections"
    assert payload["connection_readiness"]["status"] == "missing"
    assert "No broker connection configured" in payload["connection_readiness"]["message"]
    assert {"start_signal", "start_demo", "start_live"}.issubset(set(payload["blocked_action_ids"]))
    broker_check = next(check for check in payload["checks"] if check["name"] == "Broker")
    assert broker_check["status"] == "fail"
    assert "Add a connection" in broker_check["detail"]


def _patch_operator_payload_base(monkeypatch, profile_id: str) -> None:
    monkeypatch.setattr(
        bot_dashboard,
        "read_state",
        lambda: {"mode": "STOPPED", "heartbeat_age_s": 9999, "account_name": f"profile_{profile_id}"},
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
        profile_id,
        {"status": "pass", "passed": 5, "total": 5},
    )


def test_build_operator_payload_includes_conditional_overlay_check(monkeypatch):
    _patch_operator_payload_base(monkeypatch, "topstep_50k")

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


def test_build_operator_payload_warns_on_invalid_overlay_status_even_when_envelope_valid(monkeypatch):
    _patch_operator_payload_base(monkeypatch, "topstep_50k")

    overlay_state = {
        "available": True,
        "valid": True,
        "overlays": [
            {
                "overlay_id": "pr48_mgc_cont_exec_v1",
                "valid": True,
                "status": "invalid",
                "reason": "missing breakpoint row",
            }
        ],
    }
    with patch(
        "trading_app.lifecycle_state.read_lifecycle_state", return_value={"conditional_overlays": overlay_state}
    ):
        payload = _build_operator_payload("topstep_50k")

    overlay_check = next(check for check in payload["checks"] if check["name"] == "Conditional overlays")
    assert overlay_check["status"] == "warn"
    assert "missing breakpoint row" in overlay_check["detail"]


def test_build_operator_payload_includes_opportunity_awareness_check(monkeypatch):
    _patch_operator_payload_base(monkeypatch, "topstep_50k_mnq_auto")

    opportunity_state = {
        "available": True,
        "valid": True,
        "summary": {
            "lane_count": 3,
            "prime_shadow_count": 1,
            "watch_count": 1,
            "blocked_count": 1,
        },
        "lanes": [],
    }
    opportunity_state["lanes"] = [
        {
            "instrument": "MNQ",
            "orb_label": "COMEX_SETTLE",
            "opportunity_tier": "PRIME_SHADOW",
        }
    ]
    lifecycle = {
        "conditional_overlays": {"available": False, "valid": True, "overlays": []},
        "opportunity_awareness": opportunity_state,
    }
    with patch("trading_app.lifecycle_state.read_lifecycle_state", return_value=lifecycle):
        payload = _build_operator_payload("topstep_50k_mnq_auto")

    opp_check = next(check for check in payload["checks"] if check["name"] == "Opportunity awareness")
    assert opp_check["status"] == "warn"
    assert "1 PRIME_SHADOW" in opp_check["detail"]
    assert "1 WATCH" in opp_check["detail"]
    assert "1 BLOCKED" in opp_check["detail"]
    assert "prime: COMEX_SETTLE/MNQ" in opp_check["detail"]
    assert payload["opportunity_awareness"] == opportunity_state


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


def test_build_operator_payload_blocks_on_refresh_in_progress(monkeypatch):
    _patch_operator_payload_base(monkeypatch, "topstep_50k_mnq_auto")
    monkeypatch.setattr(
        bot_dashboard,
        "_refresh_snapshot",
        lambda: {"running": True},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": False,
            "raw_mode": "STOPPED",
            "heartbeat_age_s": 9999.0,
            "profile": "topstep_50k_mnq_auto",
            "tracked_alive": False,
        },
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_journal_lock_status",
        lambda: {"locked": False, "detail": "journal available"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_instance_lock_status",
        lambda: {"locked": False, "locks": []},
    )

    payload = _build_operator_payload("topstep_50k_mnq_auto")

    assert payload["top_state"] == "BLOCKED"
    assert payload["recommended_action"]["id"] == "wait_refresh"
    assert "start_signal" in payload["blocked_action_ids"]
    assert payload["busy_reason"] == "Data refresh is in progress. Wait for it to finish."


def test_action_preflight_blocks_while_session_running(monkeypatch):
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": True,
            "raw_mode": "SIGNAL",
            "heartbeat_age_s": 10.0,
            "profile": "topstep_50k_mnq_auto",
            "tracked_alive": True,
        },
    )
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})

    result = asyncio.run(bot_dashboard.action_preflight(profile="topstep_50k_mnq_auto"))

    assert result["status"] == "blocked"
    assert "stop it before preflight" in result["output"].lower()


def test_action_refresh_blocks_while_session_running(monkeypatch):
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": True,
            "raw_mode": "SIGNAL",
            "heartbeat_age_s": 10.0,
            "profile": "topstep_50k_mnq_auto",
            "tracked_alive": True,
        },
    )

    result = asyncio.run(bot_dashboard.action_refresh())

    assert result["status"] == "blocked"
    assert "stop it before refreshing" in result["message"].lower()


def test_action_start_initiates_handoff_for_conflicting_running_session(monkeypatch):
    bot_dashboard._clear_handoff()
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": True,
            "raw_mode": "SIGNAL",
            "heartbeat_age_s": 10.0,
            "profile": "topstep_50k_mnq_auto",
            "tracked_alive": True,
        },
    )
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})
    monkeypatch.setattr(
        bot_dashboard,
        "_journal_lock_status",
        lambda: {"locked": False, "detail": "journal available"},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_instance_lock_status",
        lambda: {"locked": False, "locks": []},
    )

    calls: list[str] = []

    async def _fake_kill():
        calls.append("kill")
        return {"status": "ok"}

    monkeypatch.setattr(bot_dashboard, "action_kill", _fake_kill)

    result = asyncio.run(bot_dashboard.action_start(profile="topstep_50k_mnq_auto", mode="demo"))

    assert result["status"] == "handoff_started"
    assert calls == ["kill"]
    assert result["handoff"]["status"] == "stopping"
    assert bot_dashboard._handoff_state["target_mode"] == "demo"
    bot_dashboard._clear_handoff()


def test_action_start_blocks_when_no_broker_connection(monkeypatch):
    bot_dashboard._clear_handoff()
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": False,
            "raw_mode": "STOPPED",
            "heartbeat_age_s": 9999.0,
            "profile": None,
            "tracked_alive": False,
        },
    )
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_data_status",
        lambda: {"status": "ok", "any_stale": False, "instruments": {}},
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_broker_status",
        lambda: {"status": "ok", "connections": [], "enabled_count": 0, "connected_count": 0, "error_count": 0},
    )

    result = asyncio.run(bot_dashboard.action_start(profile="topstep_50k_mnq_auto", mode="live"))

    assert result["status"] == "blocked"
    assert "No broker connection configured" in result["message"]
    assert result["connection_readiness"]["action"] == "open_connections"


def test_action_start_pins_single_copy_live_pilot_command(monkeypatch, tmp_path):
    bot_dashboard._clear_handoff()
    bot_dashboard._bg_processes.pop("session", None)
    bot_dashboard._bg_processes.pop("_session_logfile", None)
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": False,
            "raw_mode": "STOPPED",
            "heartbeat_age_s": 9999.0,
            "profile": None,
            "tracked_alive": False,
        },
    )
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_data_status",
        lambda: {"status": "ok", "any_stale": False, "instruments": {}},
    )
    monkeypatch.setattr(bot_dashboard, "_collect_broker_status", lambda: dict(_CONNECTED_BROKER_SUMMARY))

    async def fake_prepare(profile, mode="live"):
        return {"status": "pass", "output": "preflight ok"}

    monkeypatch.setattr(bot_dashboard, "_prepare_profile_for_start", fake_prepare)
    monkeypatch.setattr(bot_dashboard, "_ensure_log_dir", lambda: tmp_path)

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **_kw):
            captured["cmd"] = list(cmd)

        def poll(self):
            return None

    monkeypatch.setattr(bot_dashboard.subprocess, "Popen", FakePopen)

    result = asyncio.run(bot_dashboard.action_start(profile="topstep_50k_mnq_auto", mode="live"))

    assert result["status"] == "started"
    assert "--live" in captured["cmd"]
    assert "--auto-confirm" in captured["cmd"]
    assert "--instrument" in captured["cmd"]
    assert "MNQ" in captured["cmd"]
    assert "--copies" in captured["cmd"]
    assert "1" in captured["cmd"]
    assert "--signal-only" not in captured["cmd"]

    log_file = bot_dashboard._bg_processes.pop("_session_logfile", None)
    if log_file is not None:
        log_file.close()
    bot_dashboard._bg_processes.pop("session", None)


def _patch_action_start_to_warn(monkeypatch, tmp_path) -> dict[str, list[str]]:
    """Wire action_start mocks so readiness returns a WARN preflight status.

    Returns the dict that captures the launched command (empty if blocked).
    """
    bot_dashboard._clear_handoff()
    bot_dashboard._bg_processes.pop("session", None)
    bot_dashboard._bg_processes.pop("_session_logfile", None)
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": False,
            "raw_mode": "STOPPED",
            "heartbeat_age_s": 9999.0,
            "profile": None,
            "tracked_alive": False,
        },
    )
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})
    monkeypatch.setattr(
        bot_dashboard,
        "_collect_data_status",
        lambda: {"status": "ok", "any_stale": False, "instruments": {}},
    )
    monkeypatch.setattr(bot_dashboard, "_collect_broker_status", lambda: dict(_CONNECTED_BROKER_SUMMARY))

    async def fake_prepare(profile, mode="live"):
        # A readiness WARN: returncode 0 but a check emitted WARN/SKIPPED.
        return {"status": "warn", "output": "preflight: 1 WARN"}

    monkeypatch.setattr(bot_dashboard, "_prepare_profile_for_start", fake_prepare)
    monkeypatch.setattr(bot_dashboard, "_ensure_log_dir", lambda: tmp_path)

    class FakePopen:
        pid = 4343

        def __init__(self, cmd, **_kw):
            captured["cmd"] = list(cmd)

        def poll(self):
            return None

    monkeypatch.setattr(bot_dashboard.subprocess, "Popen", FakePopen)
    return captured


def test_action_start_live_blocks_on_preflight_warn(monkeypatch, tmp_path):
    """Strict-zero-warn parity: a readiness WARN blocks a real-money launch.

    Restores the gate the retired start_topstep_live_pilot.py enforced via
    --strict-zero-warn. SKIPPED checks normalize to WARN, so this also covers
    a silently-skipped readiness check sliding into live.
    """
    captured = _patch_action_start_to_warn(monkeypatch, tmp_path)

    result = asyncio.run(bot_dashboard.action_start(profile="topstep_50k_mnq_auto", mode="live"))

    assert result["status"] == "blocked"
    assert "cmd" not in captured  # never launched
    bot_dashboard._bg_processes.pop("session", None)
    bot_dashboard._bg_processes.pop("_session_logfile", None)


def _assert_non_live_mode_proceeds_on_warn(monkeypatch, tmp_path, mode: str) -> None:
    """WARN is advisory for signal/demo — they place no live orders, so the
    launch proceeds (a WARN must NOT block non-live modes)."""
    captured = _patch_action_start_to_warn(monkeypatch, tmp_path)

    result = asyncio.run(bot_dashboard.action_start(profile="topstep_50k_mnq_auto", mode=mode))

    assert result["status"] == "started"
    assert captured["cmd"], "expected a launched command for non-live mode on WARN"
    assert "--live" not in captured["cmd"]

    log_file = bot_dashboard._bg_processes.pop("_session_logfile", None)
    if log_file is not None:
        log_file.close()
    bot_dashboard._bg_processes.pop("session", None)


def test_action_start_signal_proceeds_on_preflight_warn(monkeypatch, tmp_path):
    _assert_non_live_mode_proceeds_on_warn(monkeypatch, tmp_path, "signal")


def test_action_start_demo_proceeds_on_preflight_warn(monkeypatch, tmp_path):
    _assert_non_live_mode_proceeds_on_warn(monkeypatch, tmp_path, "demo")


def test_handoff_snapshot_walks_state_machine_to_ready(monkeypatch):
    """Exercise _handoff_snapshot through every transition of the state machine.

    Plan: ~/.claude/plans/inspoect-repoi-instpect-resource-imperative-clarke.md F8.
    Guards against regressions in the handoff FSM introduced by commit 45f50916.
    """
    bot_dashboard._clear_handoff()
    bot_dashboard._set_handoff("topstep_50k_mnq_auto", "demo", "initial")

    # Stage 1: session still running → "stopping"
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": True,
            "raw_mode": "SIGNAL",
            "heartbeat_age_s": 5.0,
            "profile": "topstep_50k_mnq_auto",
            "tracked_alive": True,
        },
    )
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})
    monkeypatch.setattr(bot_dashboard, "_journal_lock_status", lambda: {"locked": False, "detail": "ok"})
    monkeypatch.setattr(bot_dashboard, "_instance_lock_status", lambda: {"locked": False, "locks": []})
    assert bot_dashboard._handoff_snapshot()["status"] == "stopping"

    # Stage 2: session stopped but journal still locked → "waiting_cleanup"
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {
            "running": False,
            "raw_mode": "STOPPED",
            "heartbeat_age_s": 9999.0,
            "profile": None,
            "tracked_alive": False,
        },
    )
    monkeypatch.setattr(
        bot_dashboard,
        "_journal_lock_status",
        lambda: {"locked": True, "detail": "held by pid 1234"},
    )
    assert bot_dashboard._handoff_snapshot()["status"] == "waiting_cleanup"

    # Stage 3: locks clear, refresh running → "waiting_refresh"
    monkeypatch.setattr(bot_dashboard, "_journal_lock_status", lambda: {"locked": False, "detail": "ok"})
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": True})
    assert bot_dashboard._handoff_snapshot()["status"] == "waiting_refresh"

    # Stage 4: refresh done, data stale → "needs_refresh"
    monkeypatch.setattr(bot_dashboard, "_refresh_snapshot", lambda: {"running": False})
    stale_data = {"status": "ok", "any_stale": True, "instruments": {}}
    snap = bot_dashboard._handoff_snapshot(data_summary=stale_data)
    assert snap["status"] == "needs_refresh"

    # Stage 5: fresh data is start-ready; Start auto-runs readiness checks.
    fresh_data = {"status": "ok", "any_stale": False, "instruments": {}}
    snap = bot_dashboard._handoff_snapshot(data_summary=fresh_data, preflight_summary=None)
    assert snap["status"] == "ready_to_start"

    snap = bot_dashboard._handoff_snapshot(data_summary=fresh_data, preflight_summary={"status": "fail"})
    assert snap["status"] == "ready_to_start"

    # Stage 6: an existing passing preflight remains start-ready.
    snap = bot_dashboard._handoff_snapshot(data_summary=fresh_data, preflight_summary={"status": "pass"})
    assert snap["status"] == "ready_to_start"
    assert snap["action"]["id"] == "continue_handoff"
    assert snap["target_mode"] == "demo"

    bot_dashboard._clear_handoff()


def test_preflight_helper_opens_no_duckdb_connection(monkeypatch):
    """Preflight self-test helper must not open any DuckDB connection.

    Commit 45f50916 fixed a Windows lock leak where _run_preflight constructed
    a SessionOrchestrator that owned the journal DB connection and never
    released it. F1-F2 in commit bad97445 stripped orchestrator construction
    entirely, leaving the helper as a notifications + broker-probe surface.

    Regression guard: intercept duckdb.connect and fail immediately if the
    helper attempts any connection. Catches the leak pattern regardless of
    which path (LIVE_JOURNAL_DB_PATH, GOLD_DB_PATH, etc.) the regression hits.

    2026-05-16 update: helper now also runs real bracket + fill-poller probes
    via the `components` kwarg (was hardcoded True before). Thread a stub
    `components` dict so the probes execute against an in-memory router and
    the DB-free guarantee covers the new probe path too.
    """
    import duckdb

    import trading_app.live.notifications as notifications
    from scripts.run_live_session import _run_lightweight_component_self_tests

    monkeypatch.setattr(notifications, "notify", lambda *a, **k: True)

    connect_calls: list[tuple] = []

    def spy_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        raise AssertionError(
            "Preflight helper attempted duckdb.connect — regression of the "
            "45f50916 journal-lock fix. Helper must stay DB-free."
        )

    monkeypatch.setattr(duckdb, "connect", spy_connect)

    # Stub router class — exercises the probe path without any real broker /
    # network / DB call. Mirrors the in-memory routers used by
    # test_run_live_session_preflight.py.
    class _StubAuth:
        def get_token(self) -> str:
            return "tk_stub"

    class _StubRouter:
        def __init__(self, account_id, auth, **_kw):
            self.account_id = account_id
            self.auth = auth

        def supports_native_brackets(self) -> bool:
            return True

        def build_bracket_spec(self, **_kw) -> dict:
            return {"stop": 1, "target": 2}

        def query_order_status(self, _order_id):
            # Mirror the "endpoint exists, returned validation error for sentinel"
            # case used by the canonical probe.
            raise RuntimeError("404 expected for sentinel order_id=0")

    components = {
        "auth": _StubAuth(),
        "router_class": _StubRouter,
        "feed_class": None,
        "contracts_class": None,
        "positions_class": None,
    }

    results = _run_lightweight_component_self_tests(instrument="MNQ", components=components)

    assert connect_calls == []
    assert results["notifications"] is True
    assert results["brackets"] is True
    assert results["fill_poller"] is True


def test_run_preflight_subprocess_pins_single_copy_live_pilot(monkeypatch):
    """Live preflight must use the same effective config as dashboard launch."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_kw):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(bot_dashboard.subprocess, "run", fake_run)

    result = bot_dashboard._run_preflight_subprocess("topstep_50k_mnq_auto")

    assert result["returncode"] == 0
    assert "--signal-only" not in captured["cmd"]
    assert "--live" in captured["cmd"]
    assert "--instrument" in captured["cmd"]
    assert "MNQ" in captured["cmd"]
    assert "--copies" in captured["cmd"]
    assert "1" in captured["cmd"]
    assert "--preflight" in captured["cmd"]
    assert "topstep_50k_mnq_auto" in captured["cmd"]


def test_run_preflight_subprocess_threads_signal_only_in_signal_mode(monkeypatch):
    """Signal-mode preflight must pass --signal-only so telemetry-maturity gate
    auto-passes per the documented signal-only accumulation path.

    Without this flag the Start Signal button fails at the very gate
    signal-only mode is meant to clear — blocking the path that would
    otherwise feed the gate's maturity counter.
    """
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_kw):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(bot_dashboard.subprocess, "run", fake_run)

    result = bot_dashboard._run_preflight_subprocess("topstep_50k_mnq_auto", mode="signal")

    assert result["returncode"] == 0
    assert "--signal-only" in captured["cmd"]
    assert "--preflight" in captured["cmd"]
    # Ordering invariant: --signal-only appended after --preflight (not before
    # the --profile arg) so it never shadows the profile binding.
    assert captured["cmd"].index("--signal-only") > captured["cmd"].index("--preflight")


def test_prepare_profile_for_start_propagates_mode_to_subprocess(monkeypatch):
    """_prepare_profile_for_start must thread its `mode` arg to
    _run_preflight_subprocess so the dashboard's mode-aware Start path stays
    coherent end-to-end. Verified by spying on the subprocess helper.
    """
    captured: dict[str, object] = {}

    def fake_subprocess(profile, mode="live"):
        captured["profile"] = profile
        captured["mode"] = mode
        return {"status": "PASS", "returncode": 0, "output": "ok", "checks": []}

    # _prepare_profile_for_start runs the subprocess helper via
    # asyncio.to_thread; patching the helper directly captures the threaded
    # call without needing a subprocess.run stub.
    monkeypatch.setattr(bot_dashboard, "_run_preflight_subprocess", fake_subprocess)
    monkeypatch.setattr(
        bot_dashboard,
        "_run_control_refresh_subprocess",
        lambda profile: {"status": "pass", "output": "ok"},
    )

    result = asyncio.run(bot_dashboard._prepare_profile_for_start("topstep_50k_mnq_auto", mode="signal"))

    assert captured["profile"] == "topstep_50k_mnq_auto"
    assert captured["mode"] == "signal"
    assert result["status"] != "error"


def test_dashboard_live_pilot_copy_is_explicit_and_professional():
    html = (bot_dashboard.PROJECT_ROOT / "trading_app" / "live" / "bot_dashboard.html").read_text(encoding="utf-8")

    assert "HOLD TO GO LIVE" in html
    assert "pilot-contract" in html
    assert "hero-pilot" in html
    assert "NYSE parked" in html
    assert "Topstep MNQ &middot; one protected primary account" in html
    assert "Topstep MNQ &middot; 1 copy &middot; real orders &middot; gates run first" in html
    assert "Live pilot: <b>MNQ</b> &middot; 3 lanes &middot; 1 copy &middot; NYSE: parked" in html
    assert "Broker account pending" in html
    assert "renderAccounts(lastAccountsData)" in html
    assert "Combine" not in html
    assert "🔒" not in html


# --- live_health dashboard reader (Phase 2 read-only operator visibility) ---------------


def _payload_monkey_baseline(monkeypatch):
    """Shared monkeypatch baseline so live_health tests do not pull broker state."""
    monkeypatch.setattr(
        bot_dashboard,
        "read_state",
        lambda: {"mode": "SIGNAL", "heartbeat_age_s": 5, "account_name": "topstep_50k_mnq_auto"},
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


def test_dashboard_payload_includes_live_health_when_snapshot_present(monkeypatch):
    """Snapshot file present + fresh → payload surfaces broker_status verbatim."""
    _payload_monkey_baseline(monkeypatch)
    sample = {
        "auth_healthy": True,
        "brackets_probe": True,
        "fill_poller_probe": True,
        "broker_status": "ok",
        "fill_polls_run": 17,
        "fill_polls_confirmed": 3,
        "fill_polls_failed": 0,
        "snapshot_ts_utc": "2026-05-16T01:23:45+00:00",
    }
    monkeypatch.setattr(bot_dashboard, "read_live_health", lambda: sample)

    payload = _build_operator_payload("topstep_50k_mnq_auto")

    assert payload["live_health"] == sample
    assert payload["live_health"]["broker_status"] == "ok"
    assert payload["live_health"]["fill_polls_run"] == 17


def test_dashboard_payload_live_health_unknown_when_snapshot_missing(monkeypatch):
    """Snapshot missing → reader returns unknown; dashboard surfaces it as such (fail-closed)."""
    _payload_monkey_baseline(monkeypatch)
    monkeypatch.setattr(
        bot_dashboard,
        "read_live_health",
        lambda: {"broker_status": "unknown", "reason": "snapshot_missing"},
    )

    payload = _build_operator_payload("topstep_50k_mnq_auto")

    assert payload["live_health"]["broker_status"] == "unknown"
    assert payload["live_health"]["reason"] == "snapshot_missing"


def test_read_live_health_returns_unknown_when_file_corrupt(tmp_path, monkeypatch):
    """End-to-end: corrupt JSON on disk → read_live_health returns unknown payload."""
    from trading_app.live import bot_state

    corrupt_file = tmp_path / "live_health.json"
    corrupt_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(bot_state, "LIVE_HEALTH_FILE", corrupt_file)

    result = bot_state.read_live_health()

    assert result["broker_status"] == "unknown"
    assert "read_error" in result["reason"]


def test_read_live_health_returns_unknown_when_snapshot_stale(tmp_path, monkeypatch):
    """Snapshot older than LIVE_HEALTH_STALE_AFTER_SECS → broker_status forced to unknown."""
    import json
    from datetime import UTC, datetime, timedelta

    from trading_app.live import bot_state

    stale_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    stale_file = tmp_path / "live_health.json"
    stale_file.write_text(
        json.dumps({"broker_status": "ok", "auth_healthy": True, "snapshot_ts_utc": stale_ts}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bot_state, "LIVE_HEALTH_FILE", stale_file)

    result = bot_state.read_live_health()

    assert result["broker_status"] == "unknown"
    assert result["reason"].startswith("snapshot_stale:")


def test_write_then_read_live_health_round_trips(tmp_path, monkeypatch):
    """Atomic write + read round-trip preserves operator-visibility fields."""
    from trading_app.live import bot_state

    target = tmp_path / "live_health.json"
    monkeypatch.setattr(bot_state, "LIVE_HEALTH_FILE", target)

    bot_state.write_live_health(
        {
            "auth_healthy": True,
            "brackets_probe": True,
            "fill_poller_probe": False,
            "broker_status": "degraded",
            "fill_polls_run": 5,
            "fill_polls_confirmed": 2,
            "fill_polls_failed": 1,
        }
    )

    result = bot_state.read_live_health()

    assert result["broker_status"] == "degraded"
    assert result["fill_poller_probe"] is False
    assert result["fill_polls_failed"] == 1
    assert "snapshot_ts_utc" in result  # writer stamps it
