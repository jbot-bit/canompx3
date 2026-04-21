"""TDD tests for Phase 6e Alert 2 Circuit Break detector.

Locks at-or-below (<=) semantics and canonical marker "DAILY CIRCUIT BREAK".
"""

from dataclasses import replace

from trading_app.live.alert_engine import classify_operator_alert
from trading_app.live.detectors.circuit_break import check_circuit_break
from trading_app.live.monitor_thresholds import MonitorThresholds


def test_no_alert_when_daily_r_above_halt():
    assert check_circuit_break(daily_r=+0.5, thresholds=MonitorThresholds()) == []


def test_no_alert_when_daily_r_above_halt_but_under_warn():
    # daily_r = -4.0 is below warn (-3.0) but above halt (-5.0) -- only warn fires, not this
    assert check_circuit_break(daily_r=-4.0, thresholds=MonitorThresholds()) == []


def test_no_alert_when_daily_r_just_above_halt():
    assert check_circuit_break(daily_r=-4.99, thresholds=MonitorThresholds()) == []


def test_alert_when_daily_r_exactly_at_halt():
    # halt is at-or-below (<=); exactly -5.0 DOES fire
    messages = check_circuit_break(daily_r=-5.0, thresholds=MonitorThresholds())
    assert len(messages) == 1


def test_alert_when_daily_r_below_halt():
    messages = check_circuit_break(daily_r=-6.25, thresholds=MonitorThresholds())
    assert len(messages) == 1


def test_alert_message_contains_canonical_marker():
    messages = check_circuit_break(daily_r=-6.25, thresholds=MonitorThresholds())
    assert "DAILY CIRCUIT BREAK" in messages[0]


def test_alert_message_contains_observed_daily_r():
    messages = check_circuit_break(daily_r=-6.25, thresholds=MonitorThresholds())
    assert "-6.25" in messages[0]


def test_alert_message_contains_threshold():
    messages = check_circuit_break(daily_r=-6.25, thresholds=MonitorThresholds())
    assert "-5.0" in messages[0]


def test_respects_injected_threshold_override():
    tight = replace(MonitorThresholds(), daily_pnl_halt_r=-2.0)
    # -1.5 is ABOVE the tightened halt (-2.0), should NOT fire
    assert check_circuit_break(daily_r=-1.5, thresholds=tight) == []
    # -2.5 is BELOW the tightened halt (-2.0), SHOULD fire
    assert len(check_circuit_break(daily_r=-2.5, thresholds=tight)) == 1


def test_classifier_routes_daily_circuit_break_to_critical():
    level, category = classify_operator_alert("DAILY CIRCUIT BREAK: daily_r=-6.25 threshold=-5.0")
    assert level == "critical"
    assert category == "daily_circuit_break"
