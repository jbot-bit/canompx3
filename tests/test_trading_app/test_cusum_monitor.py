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


def test_std_r_calibrated_per_strategy():
    """CUSUM std_r must be computed from win_rate + rr_target, not hardcoded 1.0."""
    from trading_app.live.performance_monitor import PerformanceMonitor
    from trading_app.portfolio import PortfolioStrategy

    s = PortfolioStrategy(
        strategy_id="TEST_RR3",
        instrument="MGC",
        orb_label="CME_REOPEN",
        entry_model="E2",
        rr_target=3.0,
        confirm_bars=1,
        filter_type="ORB_G5",
        expectancy_r=0.20,
        win_rate=0.30,
        sample_size=200,
        sharpe_ratio=1.0,
        max_drawdown_r=5.0,
        median_risk_points=3.0,
        stop_multiplier=1.0,
        source="test",
        weight=1.0,
    )
    monitor = PerformanceMonitor([s])
    cusum = monitor.get_cusum("TEST_RR3")
    # std_r for WR=0.30, RR=3.0, ExpR=0.20:
    # sqrt(0.30*(3.0-0.20)^2 + 0.70*(-1-0.20)^2) ≈ 1.764
    assert cusum.std_r > 1.5, f"std_r={cusum.std_r} — should be ~1.76, not 1.0"
    assert cusum.std_r < 2.0


def test_std_r_rr1_stays_near_one():
    """RR1.0 strategies should have std_r ≈ 1.0 (validates formula doesn't break them)."""
    from trading_app.live.performance_monitor import PerformanceMonitor
    from trading_app.portfolio import PortfolioStrategy

    s = PortfolioStrategy(
        strategy_id="TEST_RR1",
        instrument="MGC",
        orb_label="CME_REOPEN",
        entry_model="E2",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="ORB_G5",
        expectancy_r=0.05,
        win_rate=0.55,
        sample_size=200,
        sharpe_ratio=1.0,
        max_drawdown_r=5.0,
        median_risk_points=3.0,
        stop_multiplier=1.0,
        source="test",
        weight=1.0,
    )
    monitor = PerformanceMonitor([s])
    cusum = monitor.get_cusum("TEST_RR1")
    assert 0.9 < cusum.std_r < 1.1
