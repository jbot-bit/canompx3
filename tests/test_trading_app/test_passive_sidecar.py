import asyncio

import pytest

from trading_app.live import bot_state
from trading_app.live.passive_sidecar import alert_engine as passive_alert_engine
from trading_app.live.passive_sidecar import dashboard_feed, policy_gate
from trading_app.live.passive_sidecar.data_consumer import PassiveSidecarDataConsumer


def test_policy_gate_blocks_by_default(monkeypatch):
    monkeypatch.delenv("LIVE_PASSIVE_SIDECAR_ALLOWED", raising=False)

    with pytest.raises(policy_gate.PassiveSidecarPolicyError):
        policy_gate.assert_passive_sidecar_allowed()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_policy_gate_accepts_explicit_truthy_values(monkeypatch, value):
    monkeypatch.setenv("LIVE_PASSIVE_SIDECAR_ALLOWED", value)
    policy_gate.assert_passive_sidecar_allowed()


def test_data_consumer_checks_policy_before_auth_factory(monkeypatch):
    monkeypatch.delenv("LIVE_PASSIVE_SIDECAR_ALLOWED", raising=False)

    def _boom():
        raise AssertionError("auth factory must not be called when policy gate is blocked")

    consumer = PassiveSidecarDataConsumer(auth_factory=_boom)
    with pytest.raises(policy_gate.PassiveSidecarPolicyError):
        asyncio.run(consumer.start([123]))


def test_dashboard_feed_writes_passive_snapshot_without_touching_bot_state(tmp_path, monkeypatch):
    passive_path = tmp_path / "runtime" / "passive_sidecar_state.json"
    bot_path = tmp_path / "bot_state.json"
    monkeypatch.setattr(dashboard_feed, "STATE_PATH", passive_path)
    monkeypatch.setattr(bot_state, "STATE_FILE", bot_path)

    snapshot = dashboard_feed.build_dashboard_snapshot(
        {
            "connection_status": "connected",
            "accounts_by_id": {1: {"canTrade": True}},
            "positions_by_contract": {},
            "orders_by_id": {},
            "trades_by_id": {},
            "last_event_utc": "2026-05-02T00:00:00+00:00",
        },
        policy_gate_state="blocked",
    )
    dashboard_feed.write_passive_sidecar_state(snapshot)

    assert passive_path.exists()
    assert not bot_path.exists()
    payload = dashboard_feed.read_passive_sidecar_state()
    assert payload["policy_gate_status"] == "blocked"
    assert payload["connection_status"] == "connected"
    assert "heartbeat_utc" in payload


def test_passive_alert_engine_records_connection_failure(monkeypatch):
    recorded: list[dict[str, object]] = []

    def _record(**kwargs):
        recorded.append(kwargs)
        return kwargs

    monkeypatch.setattr(passive_alert_engine, "record_operator_alert", _record)

    alerts = passive_alert_engine.evaluate_projection({"connection_status": "dead", "last_error": "boom"})

    assert alerts
    assert recorded
    assert recorded[0]["source"] == "passive_sidecar"
    assert "PASSIVE SIDECAR CONNECTION DOWN" in recorded[0]["message"]


def test_passive_alert_engine_records_account_position_mismatch(monkeypatch):
    recorded: list[dict[str, object]] = []

    def _record(**kwargs):
        recorded.append(kwargs)
        return kwargs

    monkeypatch.setattr(passive_alert_engine, "record_operator_alert", _record)

    alerts = passive_alert_engine.evaluate_projection(
        {
            "connection_status": "subscribed",
            "accounts_by_id": {7: {"id": 7, "canTrade": False}},
            "positions_by_contract": {"7:MGCM6": {"accountId": 7, "size": 1}},
        }
    )

    assert alerts
    assert any("ACCOUNT/POSITION MISMATCH" in str(item["message"]) for item in recorded)
