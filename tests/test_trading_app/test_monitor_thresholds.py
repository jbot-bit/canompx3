"""Phase 6e monitor threshold contract tests.

Locks the 11 numeric values from docs/plans/2026-04-21-phase-6e-monitoring-design.md § 4.
Any drift from § 4 here fails these tests -- per pre_registered_criteria.md
no-post-hoc-relaxation rule.
"""

import dataclasses
from pathlib import Path

import pytest

from trading_app.live.monitor_thresholds import MonitorThresholds


def test_instantiable_with_defaults():
    t = MonitorThresholds()
    assert t is not None


def test_alert_1_drawdown_daily_pnl_warn():
    assert MonitorThresholds().daily_pnl_warn_r == -3.0


def test_alert_2_circuit_break_daily_pnl_halt():
    assert MonitorThresholds().daily_pnl_halt_r == -5.0


def test_alert_3_wr_window_size():
    assert MonitorThresholds().wr_window_trades == 50


def test_alert_3_wr_delta_pp():
    assert MonitorThresholds().wr_delta_pp == 10.0


def test_alert_4_expr_window_size():
    assert MonitorThresholds().expr_window_trades == 50


def test_alert_4_expr_ratio_threshold():
    assert MonitorThresholds().expr_ratio_threshold == 0.50


def test_alert_4_sr_alarm_arl0():
    assert MonitorThresholds().sr_alarm_arl0 == 1000


def test_alert_5_orb_size_rolling_window():
    assert MonitorThresholds().orb_size_rolling_days == 30


def test_alert_5_orb_size_median_ratio():
    assert MonitorThresholds().orb_size_median_ratio == 2.0


def test_alert_6_missing_data_ratio():
    assert MonitorThresholds().missing_data_ratio == 0.80


def test_alert_7_strategy_stale_days():
    assert MonitorThresholds().stale_days == 30


def test_frozen_dataclass_rejects_mutation():
    t = MonitorThresholds()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.daily_pnl_warn_r = -99.0


def test_module_has_revalidated_for_annotation():
    source = Path(__file__).resolve().parents[2] / "trading_app" / "live" / "monitor_thresholds.py"
    text = source.read_text(encoding="utf-8")
    assert "@revalidated-for" in text, (
        "Phase 6e threshold module must carry @revalidated-for provenance annotation per CLAUDE.md Research Provenance Rule."
    )


def test_module_cites_design_doc():
    source = Path(__file__).resolve().parents[2] / "trading_app" / "live" / "monitor_thresholds.py"
    text = source.read_text(encoding="utf-8")
    assert "2026-04-21-phase-6e-monitoring-design.md" in text, "Module must cite the authoritative design doc (§ 4)."
