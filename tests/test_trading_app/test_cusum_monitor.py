from trading_app.live.cusum_monitor import CUSUMMonitor


def test_no_alarm_on_good_performance():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(10):
        monitor.update(+0.5)
    assert not monitor.alarm_triggered


def test_alarm_triggered_on_persistent_losses():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(20):
        monitor.update(-1.0)
    assert monitor.alarm_triggered


def test_drift_severity_positive_when_losing():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(5):
        monitor.update(-1.0)
    assert monitor.drift_severity > 0


def test_clear_resets_state():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(20):
        monitor.update(-1.0)
    monitor.clear()
    assert not monitor.alarm_triggered
    assert monitor.cusum_neg == 0.0
    assert monitor.cusum_pos == 0.0


def test_alarm_triggers_once_not_repeatedly():
    """update() returns True only on the first trigger, not on subsequent calls."""
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    triggered = []
    for _ in range(30):
        if monitor.update(-1.0):
            triggered.append(True)
    assert len(triggered) == 1


def test_trade_count_increments():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(5):
        monitor.update(0.1)
    assert monitor.n_trades == 5


def test_cusum_lower_sigma_triggers_alarm_sooner():
    """CUSUM with lower sigma should trigger alarm sooner on same drift.

    z = (actual_r - expected_r) / std_r, so lower std_r amplifies z-scores.
    With actual_r=-0.5, expected_r=0.3:
      conservative (std_r=1.0): z=-0.8/trade → 4 trades = cusum -3.2 (no alarm)
      sensitive    (std_r=0.5): z=-1.6/trade → 3 trades = cusum -4.8 (alarm at >4.0)
    """
    conservative = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    sensitive = CUSUMMonitor(expected_r=0.3, std_r=0.5, threshold=4.0)

    # Feed identical losing trades
    for _ in range(4):
        conservative.update(-0.5)
        sensitive.update(-0.5)

    # Sensitive (lower sigma) should alarm first
    assert sensitive.alarm_triggered is True
    assert conservative.alarm_triggered is False
