from __future__ import annotations

import duckdb
from datetime import date, timedelta

from trading_app.live.sr_monitor import ShiryaevRobertsMonitor, calibrate_sr_threshold
from trading_app import sr_monitor


def test_sr_no_alarm_on_good_performance():
    monitor = ShiryaevRobertsMonitor(expected_r=0.3, std_r=1.0, threshold=10.0, delta=-1.0)
    for _ in range(10):
        monitor.update(0.8)
    assert monitor.alarm_triggered is False


def test_sr_alarm_on_persistent_losses():
    monitor = ShiryaevRobertsMonitor(expected_r=0.3, std_r=1.0, threshold=10.0, delta=-1.0)
    for _ in range(20):
        monitor.update(-1.0)
        if monitor.alarm_triggered:
            break
    assert monitor.alarm_triggered is True
    assert monitor.alarm_ratio >= 1.0


def test_sr_clear_resets_state():
    monitor = ShiryaevRobertsMonitor(expected_r=0.3, std_r=1.0, threshold=10.0, delta=-1.0)
    for _ in range(20):
        monitor.update(-1.0)
        if monitor.alarm_triggered:
            break
    monitor.clear()
    assert monitor.alarm_triggered is False
    assert monitor.sr_stat == 0.0


def test_calibrate_sr_threshold_targets_arl_near_60():
    threshold = calibrate_sr_threshold(target_arl=60, n_paths=400, max_steps=800, seed=0)
    assert 20.0 < threshold < 45.0


def test_prepare_monitor_inputs_prefers_live_baseline_after_50_paper_trades():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE paper_trades (
            strategy_id VARCHAR,
            trading_day DATE,
            pnl_r DOUBLE
        )
        """
    )
    rows = []
    start = date(2026, 1, 1)
    for i in range(55):
        pnl = 0.1 if i % 2 == 0 else 0.3
        rows.append(("SID1", start + timedelta(days=i), pnl))
    con.executemany("INSERT INTO paper_trades VALUES (?, ?, ?)", rows)

    monitor, trades, baseline_source, stream_source = sr_monitor.prepare_monitor_inputs(
        con,
        "SID1",
        {
            "mu0": 0.9,
            "sigma": 2.0,
            "instrument": "MNQ",
            "orb_label": "NYSE_CLOSE",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G8",
        },
    )

    assert baseline_source == "paper_trades_first_50"
    assert stream_source == "paper_trades"
    assert len(trades) == 5
    assert abs(monitor.expected_r - 0.2) < 1e-9
    assert monitor.std_r > 0


def test_prepare_monitor_inputs_falls_back_to_validated_baseline_and_canonical(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE paper_trades (
            strategy_id VARCHAR,
            trading_day DATE,
            pnl_r DOUBLE
        )
        """
    )

    monkeypatch.setattr(
        sr_monitor,
        "_load_canonical_forward_trades",
        lambda con, params: [1.0, -1.0, 0.5],
    )

    monitor, trades, baseline_source, stream_source = sr_monitor.prepare_monitor_inputs(
        con,
        "SID2",
        {
            "mu0": 0.25,
            "sigma": 1.5,
            "instrument": "MGC",
            "orb_label": "CME_REOPEN",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G6",
        },
    )

    assert baseline_source == "validated_backtest"
    assert stream_source == "canonical_forward"
    assert trades == [1.0, -1.0, 0.5]
    assert monitor.expected_r == 0.25
    assert monitor.std_r == 1.5


def test_apply_alarm_pauses_only_alarm_rows(monkeypatch):
    calls = []

    def _fake_pause(profile_id, strategy_id, reason, expires, source):
        calls.append((profile_id, strategy_id, reason, expires, source))
        return True

    monkeypatch.setattr("trading_app.lane_ctl.pause_strategy_id", _fake_pause)

    results = [
        {
            "strategy_id": "SID_ALARM",
            "status": "ALARM",
            "sr_stat": 42.0,
            "threshold": 31.96,
            "baseline_source": "validated_backtest",
            "stream_source": "canonical_forward",
        },
        {
            "strategy_id": "SID_OK",
            "status": "CONTINUE",
            "sr_stat": 2.0,
            "threshold": 31.96,
            "baseline_source": "validated_backtest",
            "stream_source": "canonical_forward",
        },
    ]

    applied = sr_monitor.apply_alarm_pauses(
        results,
        profile_id="topstep_50k_mnq_auto",
        pause_days=30,
        as_of=date(2026, 4, 10),
    )

    assert applied == 1
    assert len(calls) == 1
    assert calls[0][0] == "topstep_50k_mnq_auto"
    assert calls[0][1] == "SID_ALARM"
    assert calls[0][4] == "sr_monitor"
