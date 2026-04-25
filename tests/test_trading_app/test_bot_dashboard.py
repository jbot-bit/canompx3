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

    # Stage 5: data fresh, no/failed preflight → "needs_preflight"
    fresh_data = {"status": "ok", "any_stale": False, "instruments": {}}
    snap = bot_dashboard._handoff_snapshot(data_summary=fresh_data, preflight_summary=None)
    assert snap["status"] == "needs_preflight"

    snap = bot_dashboard._handoff_snapshot(data_summary=fresh_data, preflight_summary={"status": "fail"})
    assert snap["status"] == "needs_preflight"

    # Stage 6: data fresh + preflight pass → "ready_to_start"
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
    entirely, leaving the helper as a notifications-only probe.

    Regression guard: intercept duckdb.connect and fail immediately if the
    helper attempts any connection. Catches the leak pattern regardless of
    which path (LIVE_JOURNAL_DB_PATH, GOLD_DB_PATH, etc.) the regression hits.
    """
    import duckdb

    from scripts.run_live_session import _run_lightweight_component_self_tests

    import trading_app.live.notifications as notifications

    monkeypatch.setattr(notifications, "notify", lambda *a, **k: True)

    connect_calls: list[tuple] = []

    def spy_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        raise AssertionError(
            "Preflight helper attempted duckdb.connect — regression of the "
            "45f50916 journal-lock fix. Helper must stay DB-free."
        )

    monkeypatch.setattr(duckdb, "connect", spy_connect)

    results = _run_lightweight_component_self_tests(instrument="MNQ")

    assert connect_calls == []
    assert results["notifications"] is True
    assert results["brackets"] is True
    assert results["fill_poller"] is True
