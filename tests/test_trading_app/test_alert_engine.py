from trading_app.live import alert_engine


def test_classify_operator_alert_marks_feed_dead_critical():
    level, category = alert_engine.classify_operator_alert("FEED DEAD: all reconnect attempts exhausted for MNQ")

    assert level == "critical"
    assert category == "feed_dead"


def test_record_read_and_summarize_operator_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_engine, "ALERTS_PATH", tmp_path / "operator_alerts.jsonl")

    alert_engine.record_operator_alert(
        message="HEARTBEAT: session alive, 4 strategies loaded (SIGNAL)",
        instrument="MNQ",
        profile="topstep_50k_mnq_auto",
        mode="SIGNAL",
        source="session_orchestrator",
        trading_day="2026-04-13",
    )
    alert_engine.record_operator_alert(
        message="FEED STALE: 180s no data (check 2)",
        instrument="MNQ",
        profile="topstep_50k_mnq_auto",
        mode="SIGNAL",
        source="session_orchestrator",
        trading_day="2026-04-13",
    )

    alerts = alert_engine.read_operator_alerts(limit=10)

    assert len(alerts) == 2
    assert alerts[0]["category"] == "feed_stale"
    assert alerts[0]["level"] == "warning"
    assert alerts[1]["category"] == "heartbeat"
    assert alerts[1]["instrument"] == "MNQ"

    summary = alert_engine.summarize_operator_alerts(alerts)
    assert summary["total"] == 2
    assert summary["counts"]["warning"] == 1
    assert summary["counts"]["info"] == 1
    assert summary["latest"]["message"].startswith("FEED STALE")


def test_read_operator_alerts_filters_by_normalized_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_engine, "ALERTS_PATH", tmp_path / "operator_alerts.jsonl")

    alert_engine.record_operator_alert(
        message="HEARTBEAT: topstep session alive",
        instrument="MNQ",
        profile="profile_topstep_50k_mnq_auto",
        mode="SIGNAL",
    )
    alert_engine.record_operator_alert(
        message="HEARTBEAT: tradeify session alive",
        instrument="MGC",
        profile="tradeify_50k_type_b",
        mode="SIGNAL",
    )

    alerts = alert_engine.read_operator_alerts(limit=10, profile="topstep_50k_mnq_auto")

    assert len(alerts) == 1
    assert alerts[0]["instrument"] == "MNQ"
