from __future__ import annotations

import json
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


def test_run_monitor_writes_state_envelope(tmp_path, monkeypatch, capsys):
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE validated_setups (status VARCHAR)")
    con.execute("INSERT INTO validated_setups VALUES ('active')")
    con.execute("CREATE TABLE paper_trades (strategy_id VARCHAR, trading_day DATE, pnl_r DOUBLE)")
    con.execute("CREATE TABLE orb_outcomes (trading_day DATE)")
    con.execute("INSERT INTO orb_outcomes VALUES (DATE '2026-04-09')")
    con.execute("CREATE TABLE daily_features (trading_day DATE)")
    con.execute("INSERT INTO daily_features VALUES (DATE '2026-04-09')")

    monkeypatch.setattr(sr_monitor, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sr_monitor.duckdb, "connect", lambda *_args, **_kwargs: con)
    monkeypatch.setattr(sr_monitor, "resolve_profile_id", lambda: "topstep_50k_mnq_auto")
    monkeypatch.setattr(sr_monitor, "get_profile", lambda _pid: object())
    monkeypatch.setattr(
        sr_monitor,
        "_build_lanes",
        lambda: {
            "SID1": {
                "mu0": 0.2,
                "sigma": 1.0,
                "instrument": "MNQ",
                "orb_label": "NYSE_CLOSE",
                "orb_minutes": 5,
                "entry_model": "E2",
                "rr_target": 1.0,
                "confirm_bars": 1,
                "filter_type": "ORB_G8",
                "label": "L1 NYSE_CLOSE ORB_G8",
            }
        },
    )
    monkeypatch.setattr(
        sr_monitor,
        "prepare_monitor_inputs",
        lambda _con, _sid, _params: (
            ShiryaevRobertsMonitor(expected_r=0.2, std_r=1.0, threshold=99.0, delta=-1.0),
            [0.3, 0.4],
            "validated_backtest",
            "paper_trades",
        ),
    )
    monkeypatch.setattr(sr_monitor, "build_profile_fingerprint", lambda _profile: "pfp")
    monkeypatch.setattr(sr_monitor, "build_code_fingerprint", lambda _paths: "codeid")
    monkeypatch.setattr(sr_monitor, "get_git_head", lambda _root: "abc123")

    sr_monitor.run_monitor(apply_pauses=False, pause_days=30)
    capsys.readouterr()

    payload = json.loads((tmp_path / "sr_state.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["state_type"] == "sr_monitor"
    assert payload["canonical_inputs"]["profile_id"] == "topstep_50k_mnq_auto"
    assert payload["canonical_inputs"]["profile_fingerprint"] == "pfp"
    assert payload["canonical_inputs"]["lane_ids"] == ["SID1"]
    assert payload["canonical_inputs"]["db_identity"]
    assert payload["canonical_inputs"]["code_fingerprint"] == "codeid"
    assert payload["freshness"]["max_age_days"] == 2
    assert payload["payload"]["results"][0]["strategy_id"] == "SID1"


def test_sr_code_paths_include_shared_derived_state():
    paths = sr_monitor._sr_code_paths()

    assert any(path.name == "sr_monitor.py" for path in paths)
    assert any(path.name == "derived_state.py" for path in paths)


def test_run_monitor_reports_sr_at_alarm_not_at_stream_end():
    """Contract: when a stream crosses threshold mid-way and then recovers,
    the monitor breaks at first-crossing, reports ALARM, and the persisted
    sr_stat is the alarm-trigger value (not the post-recovery final value).

    Regression guard for the F-3 reconciliation in
    docs/audit/results/2026-04-19-sr-monitor-stream-audit.md: a path-walk
    reconstruction that computes final sr_stat without breaking on alarm
    can disagree with the live monitor by multiple orders of magnitude
    (e.g. NYSE_OPEN COST_LT12 on 2026-04-19: live 32.69 ALARM vs
    audit 0.82 CONTINUE — same math, different reporting mode).

    If this test fails, the run_monitor loop semantic has drifted and any
    downstream consumer that treats `sr_stat` as "current health" may be
    misreading alarm records as active signals (or vice versa).
    """
    monitor = ShiryaevRobertsMonitor(expected_r=0.1, std_r=1.0, threshold=10.0, delta=-1.0)

    # Stream: adverse run drives SR above threshold, then recovery pulls it back.
    adverse_run = [-1.0] * 20
    recovery_run = [2.0] * 30
    stream = adverse_run + recovery_run

    status = "NO_DATA"
    alarm_trade = None
    for i, trade_r in enumerate(stream, 1):
        if monitor.update(trade_r):
            status = "ALARM"
            alarm_trade = i
            break
    else:
        if stream:
            status = "CONTINUE"

    assert status == "ALARM", "Stream must cross threshold for this regression guard"
    assert alarm_trade is not None and alarm_trade <= len(adverse_run)
    assert monitor.sr_stat >= monitor.threshold, (
        "Reported sr_stat at break must be the alarm-trigger value "
        f"(got {monitor.sr_stat:.3f} vs threshold {monitor.threshold:.2f})"
    )

    # Confirm the "post-recovery final" value would be much smaller if the
    # loop had continued — this is what the audit path-walk computed.
    continued = ShiryaevRobertsMonitor(expected_r=0.1, std_r=1.0, threshold=10.0, delta=-1.0)
    for trade_r in stream:
        continued.update(trade_r)
    assert continued.sr_stat < monitor.sr_stat, (
        "Post-recovery final sr_stat should be below the alarm-trigger value; "
        "if they are equal, the recovery branch of this regression guard is "
        "not exercising the reporting-mode difference it is meant to cover."
    )
