from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime

import pytest

from trading_app.account_survival import (
    DailyScenario,
    SurvivalRules,
    check_survival_report_gate,
    evaluate_profile_survival,
    get_survival_report_path,
    simulate_survival,
)


def _rules(
    *,
    dd_type: str = "eod_trailing",
    daily_loss_limit: float | None = None,
    consistency_rule: float | None = None,
    freeze_at_balance: float | None = None,
    contracts_per_trade_micro: int = 1,
    topstep_day1_max_lots: int | None = None,
) -> SurvivalRules:
    return SurvivalRules(
        profile_id="topstep_50k_mnq_auto",
        firm="topstep",
        account_size=50_000,
        dd_type=dd_type,
        starting_balance=0.0,
        dd_limit_dollars=500.0,
        daily_loss_limit=daily_loss_limit,
        consistency_rule=consistency_rule,
        freeze_at_balance=freeze_at_balance,
        contracts_per_trade_micro=contracts_per_trade_micro,
        topstep_day1_max_lots=topstep_day1_max_lots,
    )


def test_simulate_survival_daily_loss_breach():
    scenarios = [
        DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=-150.0,
            positive_pnl_dollars=0.0,
            active_lane_count=1,
        )
    ]
    result = simulate_survival(
        scenarios,
        _rules(daily_loss_limit=100.0),
        horizon_days=1,
        n_paths=8,
        seed=0,
    )

    assert result["dd_survival_probability"] == 0.0
    assert result["daily_loss_breach_probability"] == 1.0
    assert result["trailing_dd_breach_probability"] == 0.0
    assert result["operational_pass_probability"] == 0.0


def test_simulate_survival_consistency_breach_blocks_operational_pass():
    scenarios = [
        DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=200.0,
            positive_pnl_dollars=200.0,
            active_lane_count=1,
        )
    ]
    result = simulate_survival(
        scenarios,
        _rules(consistency_rule=0.40),
        horizon_days=1,
        n_paths=1,
        seed=0,
    )

    assert result["dd_survival_probability"] == 1.0
    assert result["consistency_breach_probability"] == 1.0
    assert result["consistency_pass_probability"] == 0.0
    assert result["operational_pass_probability"] == 0.0


def test_simulate_survival_rejects_intraday_trailing_daily_approximation():
    scenarios = [
        DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=25.0,
            positive_pnl_dollars=25.0,
            active_lane_count=1,
        )
    ]
    with pytest.raises(ValueError, match="intraday_trailing"):
        simulate_survival(
            scenarios,
            _rules(dd_type="intraday_trailing"),
            horizon_days=5,
            n_paths=10,
            seed=0,
        )


def test_check_survival_report_gate_blocks_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is False
    assert "no Criterion 11 survival report" in msg


def test_check_survival_report_gate_enforces_freshness_and_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "profile_id": "topstep_50k_mnq_auto",
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "as_of_date": "2026-02-01",
                    "horizon_days": 90,
                    "n_paths": 10000,
                    "seed": 0,
                    "source_days": 50,
                    "source_start": "2025-01-01",
                    "source_end": "2026-02-01",
                    "dd_survival_probability": 0.9,
                    "operational_pass_probability": 0.65,
                    "consistency_pass_probability": None,
                    "trailing_dd_breach_probability": 0.1,
                    "daily_loss_breach_probability": 0.0,
                    "consistency_breach_probability": 0.0,
                    "scaling_feasible": True,
                    "intraday_approximated": False,
                    "min_operational_pass_probability": 0.7,
                    "gate_pass": False,
                    "p50_final_balance": 10.0,
                    "p05_final_balance": -100.0,
                    "p95_final_balance": 100.0,
                    "p50_total_pnl": 10.0,
                    "p05_total_pnl": -100.0,
                    "p95_total_pnl": 100.0,
                    "p50_max_dd": 50.0,
                    "p95_max_dd": 200.0,
                    "median_best_day": 25.0,
                },
                "rules": asdict(_rules()),
                "metadata": {"profile_id": "topstep_50k_mnq_auto"},
            }
        )
    )

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
        max_age_days=30,
    )

    assert ok is False
    assert "30d" in msg


def test_check_survival_report_gate_blocks_low_probability_even_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "profile_id": "topstep_50k_mnq_auto",
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "as_of_date": "2026-04-09",
                    "horizon_days": 90,
                    "n_paths": 10000,
                    "seed": 0,
                    "source_days": 50,
                    "source_start": "2025-01-01",
                    "source_end": "2026-04-09",
                    "dd_survival_probability": 0.92,
                    "operational_pass_probability": 0.61,
                    "consistency_pass_probability": None,
                    "trailing_dd_breach_probability": 0.08,
                    "daily_loss_breach_probability": 0.01,
                    "consistency_breach_probability": 0.0,
                    "scaling_feasible": True,
                    "intraday_approximated": False,
                    "min_operational_pass_probability": 0.7,
                    "gate_pass": False,
                    "p50_final_balance": 200.0,
                    "p05_final_balance": -300.0,
                    "p95_final_balance": 900.0,
                    "p50_total_pnl": 200.0,
                    "p05_total_pnl": -300.0,
                    "p95_total_pnl": 900.0,
                    "p50_max_dd": 150.0,
                    "p95_max_dd": 450.0,
                    "median_best_day": 80.0,
                },
                "rules": asdict(_rules()),
                "metadata": {"profile_id": "topstep_50k_mnq_auto"},
            }
        )
    )

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is False
    assert "61.0% < 70%" in msg


def test_check_survival_report_gate_passes_clean_report(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "profile_id": "topstep_50k_mnq_auto",
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "as_of_date": "2026-04-09",
                    "horizon_days": 90,
                    "n_paths": 10000,
                    "seed": 0,
                    "source_days": 50,
                    "source_start": "2025-01-01",
                    "source_end": "2026-04-09",
                    "dd_survival_probability": 0.92,
                    "operational_pass_probability": 0.78,
                    "consistency_pass_probability": None,
                    "trailing_dd_breach_probability": 0.08,
                    "daily_loss_breach_probability": 0.01,
                    "consistency_breach_probability": 0.0,
                    "scaling_feasible": True,
                    "intraday_approximated": False,
                    "min_operational_pass_probability": 0.7,
                    "gate_pass": True,
                    "p50_final_balance": 200.0,
                    "p05_final_balance": -300.0,
                    "p95_final_balance": 900.0,
                    "p50_total_pnl": 200.0,
                    "p05_total_pnl": -300.0,
                    "p95_total_pnl": 900.0,
                    "p50_max_dd": 150.0,
                    "p95_max_dd": 450.0,
                    "median_best_day": 80.0,
                },
                "rules": asdict(_rules()),
                "metadata": {"profile_id": "topstep_50k_mnq_auto"},
            }
        )
    )

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is True
    assert "Criterion 11 pass" in msg


def test_evaluate_profile_survival_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)

    def fake_load(_profile_id, *, as_of_date, db_path=None):
        scenarios = [
            DailyScenario(
                trading_day="2026-01-02",
                total_pnl_dollars=50.0,
                positive_pnl_dollars=50.0,
                active_lane_count=1,
            )
        ]
        metadata = {
            "profile_id": "topstep_50k_mnq_auto",
            "source_start": "2026-01-02",
            "source_end": str(as_of_date),
            "source_days": 1,
            "lane_ids": ["MNQ_TEST"],
            "instruments": ["MNQ"],
        }
        return scenarios, metadata

    monkeypatch.setattr("trading_app.account_survival._load_profile_daily_scenarios", fake_load)

    summary = evaluate_profile_survival(
        "topstep_50k_mnq_auto",
        as_of_date=date(2026, 4, 9),
        horizon_days=90,
        n_paths=32,
        seed=0,
        write_state=True,
    )

    assert summary.profile_id == "topstep_50k_mnq_auto"
    assert summary.gate_pass is True

    payload = json.loads(get_survival_report_path("topstep_50k_mnq_auto").read_text())
    assert payload["summary"]["profile_id"] == "topstep_50k_mnq_auto"
    assert payload["summary"]["gate_pass"] is True
    assert payload["metadata"]["source_days"] == 1
