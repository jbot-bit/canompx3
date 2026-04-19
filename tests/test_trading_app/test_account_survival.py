from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from trading_app.account_survival import (
    DailyScenario,
    SurvivalRules,
    TradePath,
    _build_profile_fingerprint,
    _current_survival_canonical_inputs,
    _load_lane_daily_pnl,
    _scenario_from_trade_paths,
    check_survival_report_gate,
    evaluate_profile_survival,
    get_survival_report_path,
    read_survival_report_state,
    simulate_survival,
)
from trading_app.prop_profiles import get_profile


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


def _canonical_inputs() -> dict[str, object]:
    return {
        "profile_id": "topstep_50k_mnq_auto",
        "profile_fingerprint": "profile-fingerprint",
        "lane_ids": ["MNQ_TEST"],
        "db_path": "/tmp/gold.db",
        "db_identity": "db-identity",
        "code_fingerprint": "code-identity",
    }


def _survival_envelope(
    *,
    as_of_date: str,
    operational_pass_probability: float,
    gate_pass: bool,
    canonical_inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "state_type": "account_survival",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_head": "abc123",
        "tool": "account_survival",
        "canonical_inputs": canonical_inputs or _canonical_inputs(),
        "freshness": {
            "as_of_date": as_of_date,
            "max_age_days": 30,
        },
        "payload": {
            "summary": {
                "profile_id": "topstep_50k_mnq_auto",
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "as_of_date": as_of_date,
                "horizon_days": 90,
                "n_paths": 10000,
                "seed": 0,
                "source_days": 50,
                "source_start": "2025-01-01",
                "source_end": as_of_date,
                "dd_survival_probability": 0.92,
                "operational_pass_probability": operational_pass_probability,
                "consistency_pass_probability": None,
                "trailing_dd_breach_probability": 0.08,
                "daily_loss_breach_probability": 0.01,
                "scaling_breach_probability": 0.0,
                "consistency_breach_probability": 0.0,
                "scaling_feasible": True,
                "intraday_approximated": False,
                "path_model": "trade_path_conservative",
                "min_operational_pass_probability": 0.7,
                "gate_pass": gate_pass,
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
            "metadata": {
                "source_start": "2025-01-01",
                "source_end": as_of_date,
                "source_days": 50,
                "instruments": ["MNQ"],
            },
        },
    }


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


def test_simulate_survival_intraday_trailing_uses_intraday_high_for_dd():
    scenarios = [
        DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=0.0,
            positive_pnl_dollars=0.0,
            active_lane_count=1,
            min_balance_delta_dollars=60.0,
            max_balance_delta_dollars=120.0,
        )
    ]
    rules = SurvivalRules(
        profile_id="mffu_50k",
        firm="mffu",
        account_size=50_000,
        dd_type="intraday_trailing",
        starting_balance=0.0,
        dd_limit_dollars=50.0,
        daily_loss_limit=None,
        consistency_rule=None,
        freeze_at_balance=None,
        contracts_per_trade_micro=1,
        topstep_day1_max_lots=None,
    )

    result = simulate_survival(scenarios, rules, horizon_days=1, n_paths=4, seed=0)

    assert result["trailing_dd_breach_probability"] == 1.0
    assert result["operational_pass_probability"] == 0.0
    assert result["path_model"] == "trade_path_conservative"


def test_simulate_survival_scaling_breach_blocks_operational_pass():
    scenarios = [
        DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=10.0,
            positive_pnl_dollars=10.0,
            active_lane_count=3,
            max_open_lots=3,
        )
    ]

    result = simulate_survival(
        scenarios,
        _rules(topstep_day1_max_lots=2),
        horizon_days=1,
        n_paths=8,
        seed=0,
    )

    assert result["scaling_breach_probability"] == 1.0
    assert result["scaling_feasible"] is False
    assert result["operational_pass_probability"] == 0.0


def test_scenario_from_trade_paths_tracks_conservative_intraday_bounds_and_lots():
    """Verify _scenario_from_trade_paths aggregates per-instrument contracts.

    Updated 2026-04-11 for the F-1 false-alarm fix: the scenario now tracks
    raw contracts per instrument and computes max_open_lots via
    aggregate-then-ceiling (matching the canonical "20 micros = 2 lots" rule),
    NOT per-trade ceiling-then-sum. This test uses 10-MNQ and 20-MNQ
    positions so the concurrent window (10+20=30 MNQ micros) resolves to
    exactly 3 mini-equivalent lots — the same value the prior test
    asserted, but for a different (canonical) reason. The prior test
    asserted lots=1+lots=2 = 3 via simple arithmetic; the canonical rule
    would give ceil((10+20)/10) = 3 lots for the same real exposure.
    """
    trades = [
        TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="A",
            entry_ts=datetime(2026, 1, 2, 10, 0, tzinfo=UTC),
            exit_ts=datetime(2026, 1, 2, 11, 0, tzinfo=UTC),
            pnl_dollars=40.0,
            mae_dollars=20.0,
            mfe_dollars=60.0,
            lots=1,
            contracts=10,
            instrument="MNQ",
        ),
        TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="B",
            entry_ts=datetime(2026, 1, 2, 10, 30, tzinfo=UTC),
            exit_ts=datetime(2026, 1, 2, 10, 45, tzinfo=UTC),
            pnl_dollars=-10.0,
            mae_dollars=15.0,
            mfe_dollars=5.0,
            lots=2,
            contracts=20,
            instrument="MNQ",
        ),
    ]

    scenario = _scenario_from_trade_paths(date(2026, 1, 2), trades)

    assert scenario.total_pnl_dollars == 30.0
    assert scenario.min_balance_delta_dollars == -35.0
    assert scenario.max_balance_delta_dollars == 65.0
    # 10 + 20 = 30 MNQ micros concurrent → ceil(30/10) = 3 mini-equivalent lots
    assert scenario.max_open_lots == 3


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
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(_survival_envelope(as_of_date="2026-02-01", operational_pass_probability=0.65, gate_pass=False))
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
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(_survival_envelope(as_of_date="2026-04-09", operational_pass_probability=0.61, gate_pass=False))
    )

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is False
    assert "61.0% < 70%" in msg


def test_check_survival_report_gate_passes_clean_report(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(_survival_envelope(as_of_date="2026-04-09", operational_pass_probability=0.78, gate_pass=True))
    )

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is True
    assert "Criterion 11 pass" in msg


def test_check_survival_report_gate_blocks_profile_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    stale_inputs = {**_canonical_inputs(), "profile_fingerprint": "stale-fingerprint"}
    report_path.write_text(
        json.dumps(
            _survival_envelope(
                as_of_date="2026-04-09",
                operational_pass_probability=0.78,
                gate_pass=True,
                canonical_inputs=stale_inputs,
            )
        )
    )

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is False
    assert "profile fingerprint mismatch" in msg


def test_check_survival_report_gate_blocks_lane_id_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    stale_inputs = {**_canonical_inputs(), "lane_ids": ["STALE_LANE"]}
    report_path.write_text(
        json.dumps(
            _survival_envelope(
                as_of_date="2026-04-09",
                operational_pass_probability=0.78,
                gate_pass=True,
                canonical_inputs=stale_inputs,
            )
        )
    )

    ok, msg = check_survival_report_gate("topstep_50k_mnq_auto", today=date(2026, 4, 10))

    assert ok is False
    assert "lane_ids mismatch" in msg


def test_read_survival_report_state_marks_legacy_payload_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(
        json.dumps(
            {
                "summary": {"profile_id": "topstep_50k_mnq_auto", "as_of_date": "2026-04-09"},
                "rules": asdict(_rules()),
                "metadata": {"source_days": 5},
            }
        )
    )

    state = read_survival_report_state("topstep_50k_mnq_auto", today=date(2026, 4, 10))

    assert state["available"] is True
    assert state["valid"] is False
    assert state["reason"] == "legacy state: missing versioned envelope"


def test_current_survival_canonical_inputs_fingerprints_shared_helper(monkeypatch):
    captured: list[Path] = []

    def fake_build_code_fingerprint(paths):
        captured.extend(paths)
        return "code-identity"

    monkeypatch.setattr("trading_app.account_survival.build_code_fingerprint", fake_build_code_fingerprint)
    monkeypatch.setattr("trading_app.account_survival.build_db_identity", lambda _db_path: "db-identity")

    inputs = _current_survival_canonical_inputs("topstep_50k_mnq_auto", db_path=Path("/tmp/gold.db"))

    assert inputs["code_fingerprint"] == "code-identity"
    assert any(path.name == "derived_state.py" for path in captured)


def test_load_lane_daily_pnl_uses_effective_profile_stop_multiplier(monkeypatch):
    monkeypatch.setattr(
        "trading_app.account_survival._load_strategy_snapshot",
        lambda _con, _sid: {
            "instrument": "MNQ",
            "orb_label": "NYSE_CLOSE",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G8",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(
        "trading_app.account_survival._load_strategy_outcomes",
        lambda *_args, **_kwargs: [
            {
                "trading_day": date(2026, 1, 2),
                "outcome": "win",
                "entry_price": 100.0,
                "stop_price": 99.0,
                "pnl_r": 1.0,
            }
        ],
    )
    monkeypatch.setattr("trading_app.account_survival.get_cost_spec", lambda _instrument: object())
    monkeypatch.setattr("trading_app.account_survival.risk_in_dollars", lambda *_args, **_kwargs: 100.0)

    def fake_apply_tight_stop(outcomes, stop_multiplier, _cost_spec):
        assert stop_multiplier == 0.75
        return [
            {
                **outcomes[0],
                "pnl_r": 0.5,
            }
        ]

    monkeypatch.setattr("trading_app.account_survival.apply_tight_stop", fake_apply_tight_stop)

    daily = _load_lane_daily_pnl(
        con=None,
        strategy_id="MNQ_TEST",
        as_of_date=date(2026, 4, 9),
        effective_stop_multiplier=0.75,
    )

    assert daily == {date(2026, 1, 2): 50.0}


def test_evaluate_profile_survival_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )

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
    assert payload["state_type"] == "account_survival"
    assert payload["payload"]["summary"]["profile_id"] == "topstep_50k_mnq_auto"
    assert payload["payload"]["summary"]["gate_pass"] is True
    assert payload["payload"]["metadata"]["source_days"] == 1
    assert payload["canonical_inputs"]["profile_fingerprint"] == _canonical_inputs()["profile_fingerprint"]
