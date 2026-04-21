"""TDD tests for Phase 6e Alert 1 Drawdown detector.

Locks strict-less-than semantics and canonical marker string ("DRAWDOWN WARN").
"""

from trading_app.live.detectors.drawdown import check_drawdown
from trading_app.live.monitor_thresholds import MonitorThresholds


def test_no_alert_when_daily_r_above_threshold():
    assert check_drawdown(daily_r=+0.5, thresholds=MonitorThresholds()) == []


def test_no_alert_when_daily_r_slightly_above_threshold():
    # threshold = -3.0; -2.99 is above
    assert check_drawdown(daily_r=-2.99, thresholds=MonitorThresholds()) == []


def test_no_alert_when_daily_r_exactly_at_threshold():
    # strict less-than: -3.0 exactly does NOT trigger
    assert check_drawdown(daily_r=-3.0, thresholds=MonitorThresholds()) == []


def test_alert_when_daily_r_below_threshold():
    messages = check_drawdown(daily_r=-3.42, thresholds=MonitorThresholds())
    assert len(messages) == 1


def test_alert_message_contains_canonical_marker():
    messages = check_drawdown(daily_r=-3.42, thresholds=MonitorThresholds())
    assert "DRAWDOWN WARN" in messages[0]


def test_alert_message_contains_observed_daily_r():
    messages = check_drawdown(daily_r=-3.42, thresholds=MonitorThresholds())
    assert "-3.42" in messages[0]


def test_alert_message_contains_threshold():
    messages = check_drawdown(daily_r=-3.42, thresholds=MonitorThresholds())
    assert "-3.0" in messages[0]


def test_respects_injected_threshold_override():
    # custom thresholds injected -- detector must not read canonical hardcoded values
    # (wrap default -- frozen dataclass -- and build a custom instance via replace-style swap
    # by constructing fresh with a different default-bound scenario: here we assert the
    # detector's firing depends on thresholds.daily_pnl_warn_r, not a literal -3.0)
    from dataclasses import replace

    loose = replace(MonitorThresholds(), daily_pnl_warn_r=-10.0)
    # -3.42 is now ABOVE the loose threshold (-10.0), should NOT fire
    assert check_drawdown(daily_r=-3.42, thresholds=loose) == []


def test_no_alert_when_daily_r_is_nan():
    # Institutional-rigor Rule #6: NaN treated as missing, not data.
    assert check_drawdown(daily_r=float("nan"), thresholds=MonitorThresholds()) == []


def test_no_alert_when_daily_r_is_none():
    # Upstream may pass None when no rolling value computed yet.
    assert check_drawdown(daily_r=None, thresholds=MonitorThresholds()) == []


def test_classifier_routes_drawdown_warn_to_warning():
    from trading_app.live.alert_engine import classify_operator_alert

    level, category = classify_operator_alert("DRAWDOWN WARN: daily_r=-3.42 threshold=-3.0")
    assert level == "warning"
    assert category == "drawdown_warn"
