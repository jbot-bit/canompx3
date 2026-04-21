"""TDD tests for Phase 6e Alert 4a ExpR Ratio detector.

Locks the pre-reg contract from
docs/runtime/stages/phase-6e-detector-expr-ratio.md:
  Fire iff n_trades >= window AND baseline_expr > 0 AND
           rolling_expr < expr_ratio_threshold * baseline_expr.
Strict-less-than per 2026-02-08 Phase 6 spec line 425.
Severity: CRITICAL.
"""

from dataclasses import replace

from trading_app.live.alert_engine import classify_operator_alert
from trading_app.live.detectors.expr_ratio import check_expr_ratio
from trading_app.live.monitor_thresholds import MonitorThresholds


def _call(**overrides):
    defaults = dict(
        strategy_id="mnq_nyse_open_e2",
        rolling_expr=0.05,
        baseline_expr=0.30,
        n_trades=80,
        thresholds=MonitorThresholds(),
    )
    defaults.update(overrides)
    return check_expr_ratio(**defaults)


def test_no_alert_when_under_window_even_if_deep_decay():
    # rolling = 0 (total decay) but only 49 trades -> UNVERIFIED
    assert _call(rolling_expr=0.0, n_trades=49) == []


def test_no_alert_when_n_trades_exactly_window_but_above_band():
    # n=50 (exactly window, inclusive) and rolling above 50% of baseline -> no alert
    assert _call(rolling_expr=0.20, baseline_expr=0.30, n_trades=50) == []


def test_no_alert_when_rolling_equals_half_baseline():
    # rolling_expr == 0.5 * baseline_expr exactly -> strict `<` gate, no fire
    assert _call(rolling_expr=0.15, baseline_expr=0.30) == []


def test_no_alert_when_rolling_just_above_half_baseline():
    # rolling = 0.151 > 0.5 * 0.30 = 0.15 -> no fire
    assert _call(rolling_expr=0.151, baseline_expr=0.30) == []


def test_alert_when_rolling_just_below_half_baseline():
    # rolling = 0.149 < 0.5 * 0.30 = 0.15 -> fires
    messages = _call(rolling_expr=0.149, baseline_expr=0.30)
    assert len(messages) == 1


def test_alert_when_rolling_deeply_below_band():
    # rolling = 0 while baseline = 0.30 -> 100% decay -> fires
    messages = _call(rolling_expr=0.0, baseline_expr=0.30)
    assert len(messages) == 1


def test_alert_when_rolling_negative():
    # rolling = -0.10 (strategy losing money per trade) -> fires as critical
    messages = _call(rolling_expr=-0.10, baseline_expr=0.30)
    assert len(messages) == 1


def test_no_alert_when_baseline_nonpositive_zero():
    # baseline_expr == 0 -> ratio gate undefined; detector returns [] per pre-reg
    assert _call(rolling_expr=-0.50, baseline_expr=0.0) == []


def test_no_alert_when_baseline_nonpositive_negative():
    # baseline_expr < 0 -> ratio gate undefined; detector returns []
    assert _call(rolling_expr=-0.50, baseline_expr=-0.10) == []


def test_no_alert_when_rolling_above_baseline():
    # rolling > baseline (better than expected) -> no alert
    assert _call(rolling_expr=0.40, baseline_expr=0.30) == []


def test_alert_message_contains_strategy_id():
    messages = _call(
        strategy_id="mnq_nyse_open_e2_cb1_rr1.5",
        rolling_expr=0.0,
        baseline_expr=0.30,
    )
    assert "mnq_nyse_open_e2_cb1_rr1.5" in messages[0]


def test_alert_message_contains_canonical_marker():
    messages = _call(rolling_expr=0.0, baseline_expr=0.30)
    assert "EXPR DRIFT" in messages[0]


def test_alert_message_contains_rolling_and_baseline_and_ratio():
    messages = _call(rolling_expr=0.10, baseline_expr=0.30, n_trades=80)
    text = messages[0]
    # operator-readable: expect both ExpR values and a ratio/percentage
    assert "0.10" in text or "0.1" in text
    assert "0.30" in text or "0.3" in text


def test_respects_injected_window_override():
    tight = replace(MonitorThresholds(), expr_window_trades=200)
    # n=80 now under tighter window -> no alert despite deep decay
    assert _call(rolling_expr=0.0, baseline_expr=0.30, n_trades=80, thresholds=tight) == []


def test_respects_injected_ratio_override():
    lenient = replace(MonitorThresholds(), expr_ratio_threshold=0.10)
    # rolling=0.05, baseline=0.30; 0.05 < 0.10*0.30=0.03? No, 0.05 > 0.03 -> no alert
    assert _call(rolling_expr=0.05, baseline_expr=0.30, thresholds=lenient) == []
    # rolling=0.02 < 0.03 -> fires
    messages = _call(rolling_expr=0.02, baseline_expr=0.30, thresholds=lenient)
    assert len(messages) == 1


def test_no_alert_when_rolling_expr_is_nan():
    # Institutional-rigor Rule #6: NaN treated as missing, not data.
    assert _call(rolling_expr=float("nan"), baseline_expr=0.30) == []


def test_no_alert_when_baseline_expr_is_nan():
    assert _call(rolling_expr=0.05, baseline_expr=float("nan")) == []


def test_classifier_routes_expr_drift_to_critical():
    level, category = classify_operator_alert(
        "EXPR DRIFT: mnq_nyse_open_e2 rolling_expr=0.05R baseline=0.30R ratio=0.17 after 80 trades"
    )
    assert level == "critical"
    assert category == "expr_drift"
