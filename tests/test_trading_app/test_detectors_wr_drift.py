"""TDD tests for Phase 6e Alert 3 WR Drift detector.

Locks window gate (n_trades >= wr_window_trades) + strict drift gate
((baseline_wr - rolling_wr) * 100 > wr_delta_pp) per 2026-02-08 spec
line 424, plus canonical marker "WR DRIFT".
"""

from dataclasses import replace

from trading_app.live.alert_engine import classify_operator_alert
from trading_app.live.detectors.wr_drift import check_wr_drift
from trading_app.live.monitor_thresholds import MonitorThresholds


def test_no_alert_when_under_window_even_if_drift_large():
    # n_trades = 49 < window 50; baseline 60%, rolling 20% (40pp drop) -- still no alert
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.20,
            baseline_wr=0.60,
            n_trades=49,
            thresholds=MonitorThresholds(),
        )
        == []
    )


def test_no_alert_when_rolling_equals_baseline():
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.60,
            baseline_wr=0.60,
            n_trades=80,
            thresholds=MonitorThresholds(),
        )
        == []
    )


def test_no_alert_when_rolling_above_baseline():
    # Better than expected -- no drift alert
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.75,
            baseline_wr=0.60,
            n_trades=80,
            thresholds=MonitorThresholds(),
        )
        == []
    )


def test_no_alert_when_drift_below_delta():
    # baseline 0.60, rolling 0.55 => 5pp drop, threshold is 10pp, should not fire
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.55,
            baseline_wr=0.60,
            n_trades=80,
            thresholds=MonitorThresholds(),
        )
        == []
    )


def test_no_alert_when_drift_equals_delta():
    # 2026-02-08 spec line 424: strict "< baseline - 10pp"; exactly 10pp drop does NOT fire
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.50,
            baseline_wr=0.60,
            n_trades=80,
            thresholds=MonitorThresholds(),
        )
        == []
    )


def test_alert_when_drift_just_past_delta():
    # 10.1pp drop is strictly above the 10pp threshold -> fires
    messages = check_wr_drift(
        strategy_id="mnq_nyse_open_e2",
        rolling_wr=0.499,
        baseline_wr=0.60,
        n_trades=80,
        thresholds=MonitorThresholds(),
    )
    assert len(messages) == 1


def test_alert_when_drift_exceeds_delta():
    # baseline 0.60, rolling 0.40 => 20pp drop
    messages = check_wr_drift(
        strategy_id="mnq_nyse_open_e2",
        rolling_wr=0.40,
        baseline_wr=0.60,
        n_trades=80,
        thresholds=MonitorThresholds(),
    )
    assert len(messages) == 1


def test_alert_when_n_trades_exactly_window():
    # n_trades == window_trades (50) is inclusive
    messages = check_wr_drift(
        strategy_id="mnq_nyse_open_e2",
        rolling_wr=0.40,
        baseline_wr=0.60,
        n_trades=50,
        thresholds=MonitorThresholds(),
    )
    assert len(messages) == 1


def test_alert_message_contains_strategy_id():
    messages = check_wr_drift(
        strategy_id="mnq_nyse_open_e2_cb1_rr1.5",
        rolling_wr=0.40,
        baseline_wr=0.60,
        n_trades=80,
        thresholds=MonitorThresholds(),
    )
    assert "mnq_nyse_open_e2_cb1_rr1.5" in messages[0]


def test_alert_message_contains_canonical_marker():
    messages = check_wr_drift(
        strategy_id="mnq_nyse_open_e2",
        rolling_wr=0.40,
        baseline_wr=0.60,
        n_trades=80,
        thresholds=MonitorThresholds(),
    )
    assert "WR DRIFT" in messages[0]


def test_alert_message_contains_rolling_and_baseline_and_delta():
    messages = check_wr_drift(
        strategy_id="mnq_nyse_open_e2",
        rolling_wr=0.40,
        baseline_wr=0.60,
        n_trades=80,
        thresholds=MonitorThresholds(),
    )
    text = messages[0]
    # rolling 40%, baseline 60%, delta 20pp -- caller formats; tolerate either pct or fraction form
    assert ("40" in text) or ("0.40" in text)
    assert ("60" in text) or ("0.60" in text)
    assert "20" in text


def test_respects_injected_wr_window_override():
    tight = replace(MonitorThresholds(), wr_window_trades=200)
    # n=80 now under tighter window -> no alert
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.40,
            baseline_wr=0.60,
            n_trades=80,
            thresholds=tight,
        )
        == []
    )


def test_respects_injected_wr_delta_override():
    lenient = replace(MonitorThresholds(), wr_delta_pp=25.0)
    # 20pp drop is now UNDER the lenient 25pp threshold -> no alert
    assert (
        check_wr_drift(
            strategy_id="mnq_nyse_open_e2",
            rolling_wr=0.40,
            baseline_wr=0.60,
            n_trades=80,
            thresholds=lenient,
        )
        == []
    )


def test_classifier_routes_wr_drift_to_warning():
    level, category = classify_operator_alert(
        "WR DRIFT: mnq_nyse_open_e2 rolling_wr=40% baseline=60% drop=20pp after 80 trades"
    )
    assert level == "warning"
    assert category == "wr_drift"
