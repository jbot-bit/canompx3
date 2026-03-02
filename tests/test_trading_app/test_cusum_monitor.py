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
