from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from trading_app.account_survival import (
    STRICT_DD_BUDGET_FRACTION_EXPRESS,
    STRICT_DD_BUDGET_FRACTION_SELF_FUNDED,
    DailyScenario,
    SurvivalRules,
    TradePath,
    _build_profile_fingerprint,
    _build_rules,
    _current_survival_canonical_inputs,
    _load_lane_daily_pnl,
    _load_lane_trade_paths,
    _scenario_from_trade_paths,
    check_survival_report_gate,
    effective_strict_dd_budget,
    evaluate_profile_survival,
    get_survival_report_path,
    main,
    read_survival_report_state,
    simulate_survival,
)
from trading_app.prop_profiles import get_profile


def _require_canonical_gold_db() -> None:
    """Skip measured live-DB pins when the canonical DB is not provisioned."""
    import trading_app.account_survival as asv

    if not asv.GOLD_DB_PATH.exists():
        pytest.skip(f"NEED_DB canonical gold.db unavailable at {asv.GOLD_DB_PATH}")


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
                "strict_account_gate_pass": gate_pass,
                "effective_dd_budget_dollars": 1600.0,
                "historical_daily_loss_breach_days": [],
                "historical_daily_loss_breach_count": 0,
                "historical_max_observed_90d_dd_dollars": 450.0,
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


def test_build_rules_uses_profile_daily_loss_and_reports_strict_dd_budget():
    profile = get_profile("topstep_50k_mnq_auto")

    rules = _build_rules(profile)

    assert rules.daily_loss_limit == 450.0
    assert rules.dd_limit_dollars == 2000.0
    # Express-funded strict budget = 0.90 belt on the $2,000 MLL = $1,800
    # (operator risk-knob 2026-06-04; raised from the prior arbitrary 0.80/$1,600).
    assert STRICT_DD_BUDGET_FRACTION_EXPRESS == 0.90
    assert effective_strict_dd_budget(profile, rules) == 1800.0


def test_effective_strict_dd_budget_is_profile_aware_and_fails_closed():
    """Resolver: express belt 0.90; self-funded relaxed 1.00; fail-closed to express."""
    express = get_profile("topstep_50k_mnq_auto")
    express_rules = _build_rules(express)
    assert express.is_express_funded is True
    assert effective_strict_dd_budget(express, express_rules) == round(
        express_rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_EXPRESS, 2
    )

    self_funded = get_profile("self_funded_tradovate")
    self_rules = _build_rules(self_funded)
    assert self_funded.is_express_funded is False
    # Stage 2: self-funded budget comes from the profile's OWN self-imposed DD
    # halt (risk-first SOURCE), NOT the prop-firm tier figure.
    assert self_funded.self_imposed_dd_dollars == 3_000.0
    assert effective_strict_dd_budget(self_funded, self_rules) == round(
        self_funded.self_imposed_dd_dollars * STRICT_DD_BUDGET_FRACTION_SELF_FUNDED, 2
    )
    # The LEAK guard: the self-funded budget must NOT equal the prop number
    # (tier.max_dd at either fraction). tier.max_dd = $6,000 here, so the old
    # `dd_limit_dollars * 1.00` path would have returned $6,000 — 2× the operator's
    # actual -$3,000 halt. Prove we are de-coupled from the prop figure.
    assert self_rules.dd_limit_dollars == 6_000.0  # the prop-shaped tier number
    assert effective_strict_dd_budget(self_funded, self_rules) != round(
        self_rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_SELF_FUNDED, 2
    )
    assert effective_strict_dd_budget(self_funded, self_rules) != round(
        self_rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_EXPRESS, 2
    )
    assert STRICT_DD_BUDGET_FRACTION_SELF_FUNDED >= STRICT_DD_BUDGET_FRACTION_EXPRESS

    # Fail-closed (a): a profile whose express flag is missing/None resolves to the
    # STRICTER express belt, never the relaxed self-funded one.
    class _NoFlag:
        pass

    assert effective_strict_dd_budget(_NoFlag(), express_rules) == round(
        express_rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_EXPRESS, 2
    )

    # Fail-closed (b): a SELF-FUNDED profile that omits the risk-first source
    # (self_imposed_dd_dollars) must fall back to the STRICTER express belt on the
    # firm number — never to the looser prop figure at full fraction.
    class _SelfFundedNoSource:
        is_express_funded = False
        self_imposed_dd_dollars = None

    express_belt_on_prop = round(self_rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_EXPRESS, 2)
    assert effective_strict_dd_budget(_SelfFundedNoSource(), self_rules) == express_belt_on_prop

    # Fail-closed (c): malformed self_imposed_dd_dollars MUST resolve to the
    # express belt, never to a garbage budget. `bool` is a subclass of `int`
    # (True > 0 is True) — a stray True must NOT yield a $1.00 budget. NaN, 0,
    # negatives, and non-numerics all fail-close too.
    for bad in (True, False, float("nan"), 0, -100.0, "3000"):

        class _Bad:
            is_express_funded = False
            self_imposed_dd_dollars = bad

        assert effective_strict_dd_budget(_Bad(), self_rules) == express_belt_on_prop, (
            f"malformed self_imposed_dd_dollars={bad!r} did not fail-close to the express belt"
        )


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
            risk_points=10.0,  # not sizing-sensitive here (no size_model) — present to satisfy the required field
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
            risk_points=10.0,  # not sizing-sensitive here (no size_model) — present to satisfy the required field
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


def test_check_survival_report_gate_blocks_strict_historical_daily_loss_breaches(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )
    envelope = _survival_envelope(as_of_date="2026-04-09", operational_pass_probability=0.78, gate_pass=True)
    envelope["payload"]["summary"]["strict_account_gate_pass"] = False
    envelope["payload"]["summary"]["historical_daily_loss_breach_days"] = ["2025-03-03", "2025-08-14"]
    envelope["payload"]["summary"]["historical_daily_loss_breach_count"] = 2
    envelope["payload"]["summary"]["historical_max_observed_90d_dd_dollars"] = 1701.0
    report_path = get_survival_report_path("topstep_50k_mnq_auto")
    report_path.write_text(json.dumps(envelope))

    ok, msg = check_survival_report_gate(
        "topstep_50k_mnq_auto",
        today=date(2026, 4, 10),
    )

    assert ok is False
    assert "strict account diagnostics failed" in msg
    assert "historical_daily_loss_days=2" in msg
    assert "max_90d_dd=$1,701/$1,600" in msg


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
    assert "strict_account=PASS" in msg


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


def test_load_lane_trade_paths_applies_orb_cap_skip_boundary_like_live(monkeypatch):
    monkeypatch.setattr(
        "trading_app.account_survival._load_strategy_snapshot",
        lambda _con, _sid: {
            "instrument": "MNQ",
            "orb_label": "NYSE_OPEN",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.5,
            "confirm_bars": 1,
            "filter_type": "COST_LT10",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(
        "trading_app.account_survival._load_strategy_outcomes",
        lambda *_args, **_kwargs: [
            {
                "trading_day": date(2026, 1, 2),
                "outcome": "win",
                "entry_price": 20000.0,
                "stop_price": 19851.0,
                "pnl_r": 1.0,
            },
            {
                "trading_day": date(2026, 1, 3),
                "outcome": "win",
                "entry_price": 20000.0,
                "stop_price": 19850.0,
                "pnl_r": 1.0,
            },
            {
                "trading_day": date(2026, 1, 4),
                "outcome": "win",
                "entry_price": 20000.0,
                "stop_price": 19849.0,
                "pnl_r": 1.0,
            },
        ],
    )
    monkeypatch.setattr("trading_app.account_survival.get_cost_spec", lambda _instrument: object())
    monkeypatch.setattr("trading_app.account_survival.risk_in_dollars", lambda *_args, **_kwargs: 100.0)

    capped = _load_lane_trade_paths(
        con=None,
        strategy_id="MNQ_TEST",
        as_of_date=date(2026, 4, 9),
        max_orb_size_pts=150.0,
    )
    uncapped = _load_lane_trade_paths(
        con=None,
        strategy_id="MNQ_TEST",
        as_of_date=date(2026, 4, 9),
        max_orb_size_pts=None,
    )

    assert [trade.trading_day for trade in capped] == [date(2026, 1, 2)]
    assert [trade.trading_day for trade in uncapped] == [
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 4),
    ]


def test_load_lane_trade_paths_applies_orb_cap_after_tight_stop(monkeypatch):
    monkeypatch.setattr(
        "trading_app.account_survival._load_strategy_snapshot",
        lambda _con, _sid: {
            "instrument": "MNQ",
            "orb_label": "NYSE_OPEN",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.5,
            "confirm_bars": 1,
            "filter_type": "COST_LT10",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(
        "trading_app.account_survival._load_strategy_outcomes",
        lambda *_args, **_kwargs: [
            {
                "trading_day": date(2026, 1, 2),
                "outcome": "win",
                "entry_price": 20000.0,
                "stop_price": 19700.0,
                "pnl_r": 1.0,
            }
        ],
    )
    monkeypatch.setattr("trading_app.account_survival.get_cost_spec", lambda _instrument: object())
    monkeypatch.setattr("trading_app.account_survival.risk_in_dollars", lambda *_args, **_kwargs: 100.0)

    def fake_apply_tight_stop(outcomes, stop_multiplier, _cost_spec):
        assert stop_multiplier == 0.5
        return [{**outcomes[0], "stop_price": 19851.0}]

    monkeypatch.setattr("trading_app.account_survival.apply_tight_stop", fake_apply_tight_stop)

    trades = _load_lane_trade_paths(
        con=None,
        strategy_id="MNQ_TEST",
        as_of_date=date(2026, 4, 9),
        effective_stop_multiplier=0.5,
        max_orb_size_pts=150.0,
    )

    assert [trade.trading_day for trade in trades] == [date(2026, 1, 2)]


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
    # Hermetic: the C11 sizing-parity guard opens gold.db via build_profile_portfolio,
    # which is absent in CI and would fail-closed (gate_pass=False). Mock it to a clean
    # pass so this positive-path test exercises survival math, not DB availability.
    # (Mirrors the violation-path mock in test_..._gate_fails_closed_on_sizing_parity_violation.)
    monkeypatch.setattr(
        "trading_app.account_survival._assert_sizing_parity",
        lambda _pid: (True, "ok"),
    )

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


def test_evaluate_profile_survival_records_strict_historical_account_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )

    def fake_load(_profile_id, *, as_of_date, db_path=None):
        scenarios = [
            DailyScenario(
                trading_day="2025-01-02",
                total_pnl_dollars=100.0,
                positive_pnl_dollars=100.0,
                active_lane_count=1,
            ),
            DailyScenario(
                trading_day="2025-01-03",
                total_pnl_dollars=-1701.0,
                positive_pnl_dollars=0.0,
                active_lane_count=1,
                min_balance_delta_dollars=-1701.0,
            ),
        ]
        metadata = {
            "profile_id": "topstep_50k_mnq_auto",
            "source_start": "2025-01-02",
            "source_end": str(as_of_date),
            "source_days": len(scenarios),
            "lane_ids": ["MNQ_TEST"],
            "instruments": ["MNQ"],
        }
        return scenarios, metadata

    monkeypatch.setattr("trading_app.account_survival._load_profile_daily_scenarios", fake_load)

    summary = evaluate_profile_survival(
        "topstep_50k_mnq_auto",
        as_of_date=date(2025, 12, 31),
        horizon_days=90,
        n_paths=16,
        seed=0,
        write_state=False,
    )

    assert summary.gate_pass is False
    assert summary.strict_account_gate_pass is False
    # Express belt now $1,800 (0.90 × $2,000). The synthetic DD here is $1,701,
    # which is BELOW the new budget — so the strict gate fails purely on the
    # daily-loss breach day, not on DD magnitude. (Under the old $1,600 belt it
    # failed on both.) The breach alone keeps strict_account_gate_pass False.
    assert summary.effective_dd_budget_dollars == 1800.0
    assert summary.historical_max_observed_90d_dd_dollars == 1701.0
    assert summary.historical_max_observed_90d_dd_dollars <= summary.effective_dd_budget_dollars
    assert summary.historical_daily_loss_breach_days == ["2025-01-03"]
    assert summary.historical_daily_loss_breach_count == 1
    assert not get_survival_report_path("topstep_50k_mnq_auto").exists()


def test_account_survival_no_write_state_cli_fails_operational_gate_and_reports_strict_diagnostics(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs", lambda *_args, **_kwargs: _canonical_inputs()
    )

    def fake_load(_profile_id, *, as_of_date, db_path=None):
        scenarios = [
            DailyScenario(
                trading_day="2025-01-03",
                total_pnl_dollars=-1701.0,
                positive_pnl_dollars=0.0,
                active_lane_count=1,
                min_balance_delta_dollars=-1701.0,
            )
        ]
        metadata = {
            "profile_id": "topstep_50k_mnq_auto",
            "source_start": "2025-01-03",
            "source_end": str(as_of_date),
            "source_days": len(scenarios),
            "lane_ids": ["MNQ_TEST"],
            "instruments": ["MNQ"],
        }
        return scenarios, metadata

    monkeypatch.setattr("trading_app.account_survival._load_profile_daily_scenarios", fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "account_survival",
            "--profile",
            "topstep_50k_mnq_auto",
            "--as-of",
            "2025-12-31",
            "--paths",
            "8",
            "--no-write-state",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert not get_survival_report_path("topstep_50k_mnq_auto").exists()
    out = capsys.readouterr().out
    assert "gate=FAIL" in out
    assert "Expectancy edge: not evaluated by Criterion 11 account survival" in out
    assert "Strict diagnostics: effective_dd_budget=$1,800" in out
    assert "Prop-account path safety=FAIL" in out
    assert "Final deployability gate=FAIL" in out
    assert "Historical daily-loss breach days (1): 2025-01-03" in out


# ── Capital fix D — Criterion 11 code fingerprint must cover live-risk paths ──
# A cached C11 PASS must invalidate when the live ORB-cap / sizing / routing code
# changes, not just account_survival.py + derived_state.py.


def test_criterion11_fingerprint_covers_live_risk_execution_paths():
    from trading_app.account_survival import _criterion11_code_paths

    names = {p.name for p in _criterion11_code_paths()}
    # Original two must remain.
    assert "account_survival.py" in names
    assert "derived_state.py" in names
    # Live-risk execution dependencies that can drift the real DD/cap/sizing.
    for required in (
        "prop_profiles.py",
        "portfolio.py",
        "execution_engine.py",
        "session_orchestrator.py",
    ):
        assert required in names, (
            f"C11 code fingerprint must include {required} so a change to live "
            f"risk behaviour invalidates a cached PASS; got {sorted(names)}"
        )


def test_criterion11_fingerprint_paths_all_exist():
    from trading_app.account_survival import _criterion11_code_paths

    for p in _criterion11_code_paths():
        assert p.exists(), f"fingerprint path does not exist: {p}"


# ── Fork #2 (2026-06-07 capital review): the PROFILE fingerprint must change ──
# when a survival-verdict config field changes, or a cached PASS goes stale
# silently. These two fields are read by the survival sim
# (effective_strict_dd_budget reads self_imposed_dd_dollars; _build_rules reads
# daily_loss_dollars) but were absent from build_profile_fingerprint.
def test_profile_fingerprint_changes_on_self_imposed_dd_dollars():
    base = get_profile("self_funded_tradovate")
    assert base.self_imposed_dd_dollars == 3_000.0  # guard the fixture
    mutated = replace(base, self_imposed_dd_dollars=6_000.0)
    assert _build_profile_fingerprint(base) != _build_profile_fingerprint(mutated), (
        "loosening self_imposed_dd_dollars (3k→6k) must invalidate the profile "
        "fingerprint — it changes the self-funded survival DD budget"
    )


def test_profile_fingerprint_changes_on_daily_loss_dollars():
    # Find any live profile with a daily_loss_dollars set, else inject one.
    base = get_profile("self_funded_tradovate")
    a = replace(base, daily_loss_dollars=450.0)
    b = replace(base, daily_loss_dollars=900.0)
    assert _build_profile_fingerprint(a) != _build_profile_fingerprint(b), (
        "changing daily_loss_dollars must invalidate the profile fingerprint — "
        "it changes the sim's daily-loss circuit breaker"
    )


# ── D-3 sizing parity (Stage 1): the survival sim now sizes like the live engine ──
# (vol-scaled, capped). The guard no longer FORBIDS max_contracts > 1 — the honest
# sim fails the operational gate at unsafe size on its own. The guard's narrower job
# is to PROVE the sim CAN size like the engine: the portfolio builds and the equity
# that feeds the sizer is positive. It fails CLOSED only when parity is unprovable.


def test_sizing_parity_ok_when_portfolio_builds_with_positive_equity(monkeypatch):
    from types import SimpleNamespace

    from trading_app import account_survival as asv

    fake_portfolio = SimpleNamespace(
        account_equity=25000.0,
        strategies=[
            SimpleNamespace(strategy_id="A", max_contracts=1),
            SimpleNamespace(strategy_id="B", max_contracts=3),  # cap > 1 is NO LONGER a violation
        ],
    )
    monkeypatch.setattr(asv, "build_profile_portfolio", lambda **_kw: fake_portfolio)
    ok, msg = asv._assert_sizing_parity("topstep_50k_mnq_auto")
    assert ok is True
    assert "OK" in msg


def test_sizing_parity_fails_closed_on_builder_error(monkeypatch):
    from trading_app import account_survival as asv

    def boom(**_kw):
        raise RuntimeError("cannot build portfolio")

    monkeypatch.setattr(asv, "build_profile_portfolio", boom)
    ok, msg = asv._assert_sizing_parity("topstep_50k_mnq_auto")
    assert ok is False
    assert "parity" in msg.lower()


def test_sizing_parity_fails_closed_on_nonpositive_equity(monkeypatch):
    from types import SimpleNamespace

    from trading_app import account_survival as asv

    # express-funded XFA accounts can present account_equity 0.0 → would zero the
    # sizer → DD=$0 → a FALSE survival PASS. Must fail closed.
    fake_portfolio = SimpleNamespace(account_equity=0.0, strategies=[])
    monkeypatch.setattr(asv, "build_profile_portfolio", lambda **_kw: fake_portfolio)
    ok, msg = asv._assert_sizing_parity("topstep_50k_mnq_auto")
    assert ok is False
    assert "equity" in msg.lower()


def test_evaluate_profile_survival_gate_fails_closed_on_sizing_parity_violation(monkeypatch):
    """End-to-end: a clean-passing profile is forced FAIL by the D-3 parity guard.

    Also guards against a regression where the gate branch referenced an
    undefined ``log`` (NameError) — this exercises that exact code path.
    """
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs",
        lambda *_a, **_k: _canonical_inputs(),
    )

    def fake_load(_profile_id, *, as_of_date, db_path=None):
        scenarios = [
            DailyScenario(
                trading_day="2025-01-02",
                total_pnl_dollars=100.0,
                positive_pnl_dollars=100.0,
                active_lane_count=1,
            ),
        ]
        metadata = {
            "profile_id": "topstep_50k_mnq_auto",
            "source_start": "2025-01-02",
            "source_end": str(as_of_date),
            "source_days": len(scenarios),
            "lane_ids": ["MNQ_TEST"],
            "instruments": ["MNQ"],
        }
        return scenarios, metadata

    monkeypatch.setattr("trading_app.account_survival._load_profile_daily_scenarios", fake_load)
    # Force the parity guard to report a violation (parity unprovable).
    monkeypatch.setattr(
        "trading_app.account_survival._assert_sizing_parity",
        lambda _pid: (False, "D-3 sizing parity unprovable: test"),
    )

    summary = evaluate_profile_survival(
        "topstep_50k_mnq_auto",
        as_of_date=date(2025, 12, 31),
        horizon_days=90,
        n_paths=16,
        seed=0,
        write_state=False,
    )

    assert summary.gate_pass is False, "C11 gate must fail closed when D-3 sizing parity is violated"


def test_sizing_context_carries_engine_inputs_and_resolves_cap():
    from trading_app.account_survival import SizingContext

    ctx = SizingContext(
        account_equity=25000.0,
        risk_per_trade_pct=2.0,
        account_size=50000,
        max_contracts_by_strategy={"A": 3, "B": 1},
    )
    assert ctx.account_equity == 25000.0
    assert ctx.risk_per_trade_pct == 2.0
    assert ctx.account_size == 50000
    assert ctx.max_contracts_for("A") == 3
    assert ctx.max_contracts_for("B") == 1
    assert ctx.max_contracts_for("UNKNOWN") == 1  # fail closed to 1, never unbounded


def test_lane_atr_by_day_returns_per_day_atr_map():
    import duckdb

    from trading_app.account_survival import _lane_atr_by_day

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE daily_features (symbol VARCHAR, orb_minutes INT, trading_day DATE, atr_20 DOUBLE)")
    con.execute(
        "INSERT INTO daily_features VALUES "
        "('MNQ',5,DATE '2026-01-02',12.5),"
        "('MNQ',15,DATE '2026-01-02',12.5),"  # other aperture, deduped by orb_minutes=5
        "('MNQ',5,DATE '2026-01-05',NULL)"  # NULL atr -> omitted from map
    )
    m = _lane_atr_by_day(con, "MNQ", 5, {date(2026, 1, 2), date(2026, 1, 5)})
    assert m[date(2026, 1, 2)] == 12.5
    assert date(2026, 1, 5) not in m  # NULL not mapped; caller falls back to vol_scalar=1.0


def test_lane_median_atr_delegates_to_canonical_trailing_helper(monkeypatch):
    """Task 5: the median provider delegates per-day to the CANONICAL trailing
    helper paper_trader._get_median_atr_20 (no re-encoded SQL); only truthy
    medians are mapped (0.0 omitted -> caller falls back to vol_scalar=1.0)."""
    import trading_app.account_survival as asv

    calls = []

    def fake_median(con, instrument, trading_day, lookback_days=252):
        calls.append((instrument, trading_day))
        return 0.0 if trading_day == date(2026, 1, 3) else 10.0

    monkeypatch.setattr("trading_app.paper_trader._get_median_atr_20", fake_median)
    m = asv._lane_median_atr(None, "MNQ", {date(2026, 1, 2), date(2026, 1, 3)})
    assert m == {date(2026, 1, 2): 10.0}  # 0.0 omitted
    assert set(d for _, d in calls) == {date(2026, 1, 2), date(2026, 1, 3)}
    assert all(i == "MNQ" for i, _ in calls)


def test_d3_median_is_trailing_not_full_history(monkeypatch):
    """Task 5b oracle (e): the median must be point-in-time (a trade on day D
    must not see ATR on/after D). Proven by delegating to the canonical helper,
    whose SQL uses `trading_day < ?` (paper_trader.py)."""
    import trading_app.account_survival as asv

    seen = []

    def fake_median(con, instrument, trading_day, lookback_days=252):
        seen.append(trading_day)
        return 10.0

    monkeypatch.setattr("trading_app.paper_trader._get_median_atr_20", fake_median)
    asv._lane_median_atr(None, "MNQ", {date(2026, 1, 10)})
    assert seen == [date(2026, 1, 10)]  # the trade day is handed to the trailing fn


def test_d3_null_atr_day_falls_back_to_scalar_one_not_fail(monkeypatch):
    """Task 5b oracle (c): a priced day with NULL/missing atr uses vol_scalar=1.0
    (engine parity) — NOT a structural failure; sizing still produces a valid
    count (floored to 1 at cap=1)."""
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext

    base = [
        asv.TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="L1",
            entry_ts=None,
            exit_ts=None,
            pnl_dollars=40.0,
            mae_dollars=20.0,
            mfe_dollars=60.0,
            lots=1,
            risk_points=10.0,  # planned entry-to-stop basis the sizer now reads
            contracts=1,
            instrument="MNQ",
        )
    ]
    monkeypatch.setattr(
        asv,
        "_load_strategy_snapshot",
        lambda con, sid: {
            "instrument": "MNQ",
            "orb_label": "X",
            "orb_minutes": 5,
            "entry_model": "m",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "NO_FILTER",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    monkeypatch.setattr(asv, "_lane_atr_by_day", lambda *a, **k: {})  # NULL/missing atr
    monkeypatch.setattr(asv, "_lane_median_atr", lambda *a, **k: {})
    ctx = SizingContext(
        account_equity=50_000.0, risk_per_trade_pct=2.0, account_size=50_000, max_contracts_by_strategy={"L1": 2}
    )
    out = asv._load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)
    assert out[0].contracts >= 1  # vol_scalar=1.0 fallback still sizes, does not zero/raise
    assert out[0].pnl_dollars >= 40.0


def test_d3_structural_atr_loss_fails_closed():
    """Task 5b oracle (f): if ATR cannot be obtained structurally (query raises
    on a missing table), the sizing path must NOT silently swallow it — it must
    propagate so the gate fails closed."""
    import duckdb
    import pytest

    from trading_app.account_survival import _lane_atr_by_day

    con = duckdb.connect(":memory:")  # no daily_features table
    with pytest.raises(duckdb.CatalogException):
        _lane_atr_by_day(con, "MNQ", 5, {date(2026, 1, 2)})


# ---------------------------------------------------------------------------
# Task 3: opt-in size_model gate in _load_lane_trade_paths
# ---------------------------------------------------------------------------


def test_load_lane_trade_paths_scales_all_fields_when_size_model_given(monkeypatch):
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext, _load_lane_trade_paths

    base = [
        asv.TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="L1",
            entry_ts=None,
            exit_ts=None,
            pnl_dollars=40.0,
            mae_dollars=20.0,
            mfe_dollars=60.0,
            lots=1,
            # Sizing now reads PLANNED risk_points (entry-to-stop), NOT mae_dollars.
            # risk_points=10 on MNQ (point_value=2) → with $500k equity @ 2% the sizer
            # is far above cap=2, so cap binds at n=2 (the binding constraint asserted).
            risk_points=10.0,
            contracts=1,
            instrument="MNQ",
        )
    ]
    monkeypatch.setattr(
        asv,
        "_load_strategy_snapshot",
        lambda con, sid: {
            "instrument": "MNQ",
            "orb_label": "X",
            "orb_minutes": 5,
            "entry_model": "m",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "NO_FILTER",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    monkeypatch.setattr(asv, "_lane_atr_by_day", lambda *a, **k: {})  # no ATR -> vol_scalar 1.0
    monkeypatch.setattr(asv, "_lane_median_atr", lambda *a, **k: {})
    # Use high equity so sizer returns >= cap=2; cap is the binding constraint.
    ctx = SizingContext(
        account_equity=500_000.0, risk_per_trade_pct=2.0, account_size=50_000, max_contracts_by_strategy={"L1": 2}
    )
    scaled = _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)
    t = scaled[0]
    assert t.contracts == 2
    assert t.pnl_dollars == 80.0
    assert t.mae_dollars == 40.0
    assert t.mfe_dollars == 120.0


def test_load_lane_trade_paths_cap1_floors_wide_stop_trade_to_one_not_zero(monkeypatch):
    """Regression for the D-3 cap=1 byte-identity defect: a real historical trade
    whose stop is wide enough that the engine sizer returns n=0 must be floored to
    1 contract at cap=1 (not zeroed), so cap=1 is byte-identical to today's
    1-micro behavior. Without the floor the trade's pnl_dollars*0 silently drops
    its drawdown contribution (measured: 1091/2048 day mismatches, dd 1535->2032)."""
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext, _load_lane_trade_paths

    # Sizing reads PLANNED risk_points (entry-to-stop), NOT mae_dollars. A very wide
    # planned stop: risk_points=10000 on MNQ (point_value=2) => $1000 budget (2% of
    # $50k) / (10000pts * $2) = 0.05 -> sizer returns n=0, which must floor to 1.
    # mae_dollars=20000 is kept so the byte-identity assertion (mae_dollars unchanged
    # at cap=1) still pins that the DD contribution is carried, not zeroed.
    base = [
        asv.TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="L1",
            entry_ts=None,
            exit_ts=None,
            pnl_dollars=40.0,
            mae_dollars=20000.0,
            mfe_dollars=60.0,
            lots=1,
            risk_points=10000.0,
            contracts=1,
            instrument="MNQ",
        )
    ]
    monkeypatch.setattr(
        asv,
        "_load_strategy_snapshot",
        lambda con, sid: {
            "instrument": "MNQ",
            "orb_label": "X",
            "orb_minutes": 5,
            "entry_model": "m",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "NO_FILTER",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    monkeypatch.setattr(asv, "_lane_atr_by_day", lambda *a, **k: {})
    monkeypatch.setattr(asv, "_lane_median_atr", lambda *a, **k: {})
    ctx = SizingContext(
        account_equity=50_000.0, risk_per_trade_pct=2.0, account_size=50_000, max_contracts_by_strategy={"L1": 1}
    )
    out = _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)
    # cap=1 => floored to exactly 1; fields byte-identical to the unscaled trade.
    assert out[0].contracts == 1, "wide-stop trade was zeroed instead of floored to 1"
    assert out[0].pnl_dollars == 40.0
    assert out[0].mae_dollars == 20000.0
    assert out[0].mfe_dollars == 60.0


def test_survival_sizes_on_planned_risk_points_not_realized_mae_value_oracle(monkeypatch):
    """D-3 VALUE-oracle: prove the survival sim sizes on the SAME planned
    entry-to-stop basis the live engine uses — NOT realized MAE.

    The drift parity check verifies both files IMPORT the canonical sizer, but a
    shared sizer fed DIFFERENT inputs still diverges. This drives a representative
    WINNER (mae_r small) through the REAL ``_load_lane_trade_paths`` sizing loop
    and asserts the contract count matches the planned-risk-points (engine) basis,
    NOT the realized-MAE basis. The two bases are chosen to give clearly different,
    UN-clamped contract counts (high equity, high cap) so a regression to the mae
    basis flips the assertion. (This is the test the import-parity drift check
    cannot cover; verified to FAIL on the injected old basis — integrity §7.)
    """
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext, _load_lane_trade_paths, get_cost_spec
    from trading_app.portfolio import compute_position_size_vol_scaled

    cost = get_cost_spec("MNQ")  # point_value = 2.0
    equity, risk_pct, vol_scalar = 50_000.0, 2.0, 1.0  # budget = $1,000

    # Representative WINNER: planned stop is 125pts wide (engine basis), but drew
    # down only mae_r=0.4 of it. Both contract counts are chosen BELOW the XFA cap
    # (5 for a 50k account) so the clamp does not mask the divergence — the BASIS,
    # not the cap, decides the count.
    risk_points_engine = 250.0  # the engine basis: abs(entry - stop)
    risk_dollars = risk_points_engine * cost.point_value  # 250 * 2 = $500
    mae_dollars = 0.4 * risk_dollars  # $200 — what the sim stores; mae_r = 0.4
    risk_points_old_mae = abs(mae_dollars) / cost.point_value  # 100.0 pts (the buggy basis)

    # Sanity: the two bases produce DIFFERENT contract counts, both <= XFA cap (5).
    n_engine = compute_position_size_vol_scaled(equity, risk_pct, risk_points_engine, cost, vol_scalar)
    n_old_mae = compute_position_size_vol_scaled(equity, risk_pct, risk_points_old_mae, cost, vol_scalar)
    assert n_engine == 2, n_engine  # 1000 / (250*2)
    assert n_old_mae == 5, n_old_mae  # 1000 / (100*2) — 2.5x over-size, capital-unsafe
    assert n_old_mae > n_engine

    base = [
        asv.TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="L1",
            entry_ts=None,
            exit_ts=None,
            pnl_dollars=2.0 * risk_dollars,
            mae_dollars=mae_dollars,
            mfe_dollars=2.0 * risk_dollars,
            lots=1,
            risk_points=risk_points_engine,  # the planned basis the prod constructor carries
            contracts=1,
            instrument="MNQ",
        )
    ]
    monkeypatch.setattr(
        asv,
        "_load_strategy_snapshot",
        lambda con, sid: {
            "instrument": "MNQ",
            "orb_label": "X",
            "orb_minutes": 5,
            "entry_model": "m",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "NO_FILTER",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    monkeypatch.setattr(asv, "_lane_atr_by_day", lambda *a, **k: {})  # vol_scalar = 1.0
    monkeypatch.setattr(asv, "_lane_median_atr", lambda *a, **k: {})
    # Lane cap high so the lane cap never binds; account_size=50000 → XFA cap=5,
    # and both contract counts (2 and 5) are <= 5, so the BASIS decides contracts.
    ctx = SizingContext(
        account_equity=equity, risk_per_trade_pct=risk_pct, account_size=50_000, max_contracts_by_strategy={"L1": 5000}
    )
    out = _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)

    # The production sizing loop must size on the carried planned risk_points
    # (engine basis), NOT the realized-MAE basis. A regression to the old basis
    # would make contracts == n_old_mae (1000) and flip this assertion.
    assert out[0].contracts == n_engine, (
        f"survival sim sized {out[0].contracts} contracts; expected engine basis {n_engine}, "
        f"NOT realized-MAE basis {n_old_mae}"
    )
    assert out[0].contracts != n_old_mae, "survival sim must NOT size on the realized-MAE basis"


def test_load_lane_trade_paths_unchanged_when_size_model_none(monkeypatch):
    import trading_app.account_survival as asv
    from trading_app.account_survival import _load_lane_trade_paths

    base = [
        asv.TradePath(
            trading_day=date(2026, 1, 2),
            strategy_id="L1",
            entry_ts=None,
            exit_ts=None,
            pnl_dollars=40.0,
            mae_dollars=20.0,
            mfe_dollars=60.0,
            lots=1,
            risk_points=10.0,  # carried through unchanged when size_model is None
            contracts=1,
            instrument="MNQ",
        )
    ]
    monkeypatch.setattr(
        asv,
        "_load_strategy_snapshot",
        lambda con, sid: {
            "instrument": "MNQ",
            "orb_label": "X",
            "orb_minutes": 5,
            "entry_model": "m",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "NO_FILTER",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    out = _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3))
    assert out[0].contracts == 1
    assert out[0].pnl_dollars == 40.0


def test_load_lane_trade_paths_fails_closed_on_nonpositive_equity(monkeypatch):
    import pytest

    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext, _load_lane_trade_paths

    monkeypatch.setattr(
        asv,
        "_load_strategy_snapshot",
        lambda con, sid: {
            "instrument": "MNQ",
            "orb_label": "X",
            "orb_minutes": 5,
            "entry_model": "m",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "NO_FILTER",
            "stop_multiplier": 1.0,
        },
    )
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: [])
    ctx = SizingContext(account_equity=0.0, risk_per_trade_pct=2.0, account_size=50_000, max_contracts_by_strategy={})
    with pytest.raises(ValueError, match="account_equity"):
        _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)


# ---------------------------------------------------------------------------
# Task 4 — SizingContext wired into survival-private scenario builder
# ---------------------------------------------------------------------------


def test_profile_scenarios_pass_size_model_but_daily_pnl_does_not(monkeypatch):
    """Survival path scales (size_model non-None for every lane); the
    correlation/allocation entry (_load_lane_daily_pnl) stays raw (size_model None)."""
    import trading_app.account_survival as asv

    _require_canonical_gold_db()
    captured = []
    real = asv._load_lane_trade_paths

    def spy(con, sid, *, as_of_date, effective_stop_multiplier=None, max_orb_size_pts=None, size_model=None):
        captured.append((sid, size_model))
        return real(
            con,
            sid,
            as_of_date=as_of_date,
            effective_stop_multiplier=effective_stop_multiplier,
            max_orb_size_pts=max_orb_size_pts,
            size_model=size_model,
        )

    monkeypatch.setattr(asv, "_load_lane_trade_paths", spy)

    asv._load_profile_daily_scenarios("topstep_50k_mnq_auto", as_of_date=date(2026, 6, 1))
    # survival path passed a non-None SizingContext for EVERY lane it loaded
    assert captured, "expected at least one lane loaded"
    assert all(sm is not None for _, sm in captured), captured
    assert all(isinstance(sm, asv.SizingContext) for _, sm in captured)


def test_load_lane_daily_pnl_passes_no_size_model(monkeypatch):
    """_load_lane_daily_pnl (correlation/allocation path) must keep size_model=None."""
    import duckdb

    import trading_app.account_survival as asv
    from trading_app.prop_profiles import get_profile_lane_definitions

    _require_canonical_gold_db()
    captured = []
    real = asv._load_lane_trade_paths

    def spy(con, sid, *, as_of_date, effective_stop_multiplier=None, max_orb_size_pts=None, size_model=None):
        captured.append((sid, size_model))
        return real(
            con,
            sid,
            as_of_date=as_of_date,
            effective_stop_multiplier=effective_stop_multiplier,
            max_orb_size_pts=max_orb_size_pts,
            size_model=size_model,
        )

    monkeypatch.setattr(asv, "_load_lane_trade_paths", spy)

    lane = get_profile_lane_definitions("topstep_50k_mnq_auto")[0]
    con = duckdb.connect(str(asv.GOLD_DB_PATH), read_only=True)
    try:
        asv._load_lane_daily_pnl(
            con,
            lane["strategy_id"],
            as_of_date=date(2026, 6, 1),
        )
    finally:
        con.close()

    assert captured, "expected at least one call to _load_lane_trade_paths"
    assert all(sm is None for _, sm in captured), captured


# ── Task 8: measured-behavior integration pins (D-3 seam Stage 1) ─────────────
# These hit the REAL gold.db production path. The numbers below were MEASURED on
# 2026-06-07 via the canonical `_scenarios_for_context` seam (not a harness):
#   profile=topstep_50k_mnq_auto, as_of=2026-06-01, horizon=90, n_paths=10000, seed=7
#   cap=1 -> n_scen=2048, rolling_dd=1535.22, op_pass_prob=0.9997
#   cap=2 -> n_scen=2048, rolling_dd=3568.02, op_pass_prob=0.3563
# rolling_dd scales 2.32x (super-linear: per-day vol-scaling lets high-vol days
# size past 2x before the cap binds — the plan's "~2.0x" came from a prior
# throwaway no-floor harness and is corrected here by real measurement).
_D3_PID = "topstep_50k_mnq_auto"
_D3_AS_OF = date(2026, 6, 1)
_D3_GOLDEN_CAP1_ROLLING_DD = 1535.22  # known live baseline ($1,535.22 vs $1,800 budget)


def _d3_scenarios_at_cap(cap: int):
    """Build common-support scenarios for the live profile at a forced contract cap."""
    import duckdb

    import trading_app.account_survival as asv
    from trading_app.prop_profiles import get_profile_lane_definitions, load_allocation_lanes

    _require_canonical_gold_db()
    profile = asv.get_profile(_D3_PID)
    lane_defs = get_profile_lane_definitions(_D3_PID)
    instruments = sorted({ln["instrument"] for ln in lane_defs})
    lane_specs = profile.daily_lanes or load_allocation_lanes(profile.profile_id)
    effective_stop_by_strategy = {
        ln.strategy_id: float(ln.planned_stop_multiplier or profile.stop_multiplier) for ln in lane_specs
    }
    pf = asv.build_profile_portfolio(profile_id=_D3_PID)
    ctx = asv.SizingContext(
        account_equity=pf.account_equity,
        risk_per_trade_pct=pf.risk_per_trade_pct,
        account_size=profile.account_size,
        max_contracts_by_strategy={s.strategy_id: cap for s in pf.strategies},
    )
    con = duckdb.connect(str(asv.GOLD_DB_PATH), read_only=True)
    asv.configure_connection(con)
    try:
        scenarios, _meta = asv._scenarios_for_context(
            con,
            profile=profile,
            lane_defs=lane_defs,
            instruments=instruments,
            effective_stop_by_strategy=effective_stop_by_strategy,
            as_of_date=_D3_AS_OF,
            size_model=ctx,
        )
    finally:
        con.close()
    return scenarios


def test_d3_cap1_rolling_dd_reconciles_to_live_baseline():
    """cap=1 via the seam must reconcile to the known live DD ($1,535.22).

    This is the load-bearing 'no live verdict change today' pin: with every lane
    at max_contracts=1 (today's reality) the gate projects exactly the production
    drawdown, byte-for-byte, despite the new vol-sizing machinery.
    """
    import trading_app.account_survival as asv

    scen = _d3_scenarios_at_cap(1)
    dd = asv._max_observed_rolling_drawdown(scen, horizon_days=90)
    assert dd == _D3_GOLDEN_CAP1_ROLLING_DD, (
        f"cap=1 rolling_dd {dd} != live baseline {_D3_GOLDEN_CAP1_ROLLING_DD} "
        "— the seam changed a live verdict (it must not at cap=1)"
    )


def test_d3_rolling_dd_scales_superlinearly_and_pass_prob_drops_at_cap2():
    """MEASURED 2026-06-07: cap=2 rolling_dd ~2.32x cap=1; op_pass_prob 0.9997->0.3563.

    Proves the seam is wired: sizing like the engine makes the gate correctly fail
    closed at unsafe size. DD scaling is super-linear (per-day vol-scaling), so the
    pin is a measured band, not a hardcoded 2.0x.
    """
    import trading_app.account_survival as asv

    s1 = _d3_scenarios_at_cap(1)
    s2 = _d3_scenarios_at_cap(2)
    dd1 = asv._max_observed_rolling_drawdown(s1, horizon_days=90)
    dd2 = asv._max_observed_rolling_drawdown(s2, horizon_days=90)
    # measured 2.32x; band tolerates seed/data drift without admitting a flat or
    # runaway scaling regression.
    assert 2.0 * dd1 <= dd2 <= 2.6 * dd1, f"cap2/cap1 dd ratio {dd2 / dd1:.3f} outside [2.0, 2.6]"

    profile = asv.get_profile(_D3_PID)
    rules = asv._with_consistency_rule(asv._build_rules(profile), profile)
    p1 = asv.simulate_survival(s1, rules, horizon_days=90, n_paths=10000, seed=7)["operational_pass_probability"]
    p2 = asv.simulate_survival(s2, rules, horizon_days=90, n_paths=10000, seed=7)["operational_pass_probability"]
    assert p2 < p1, f"op_pass_prob must drop with size: cap1={p1} cap2={p2}"
    assert p1 > 0.90, f"cap=1 must still operationally pass on the live profile (got {p1})"
    assert p2 < 0.50, f"cap=2 must flip the operational gate False on the live profile (got {p2})"


# ---------------------------------------------------------------------------
# Survival-cap sweep Stage 1 (D-3 follow-on) — sweep + gate extract + persist
# ---------------------------------------------------------------------------


def test_contiguous_safe_ceiling_all_pass():
    """All caps pass -> ceiling is the top probed cap."""
    import trading_app.account_survival as asv

    per_cap = [{"contracts": n, "gate_pass": True} for n in range(1, 6)]
    assert asv._contiguous_safe_ceiling(per_cap) == 5


def test_contiguous_safe_ceiling_breaks_at_first_failure():
    """Pass 1,2 then fail 3 -> ceiling 2, even if a LATER cap passes (non-monotonic)."""
    import trading_app.account_survival as asv

    per_cap = [
        {"contracts": 1, "gate_pass": True},
        {"contracts": 2, "gate_pass": True},
        {"contracts": 3, "gate_pass": False},
        {"contracts": 4, "gate_pass": True},  # must NOT be honored — gap above failure
    ]
    assert asv._contiguous_safe_ceiling(per_cap) == 2


def test_contiguous_safe_ceiling_fails_closed_to_zero_when_cap1_fails():
    """Even cap=1 fails -> ceiling 0 (profile cannot survive at any size)."""
    import trading_app.account_survival as asv

    per_cap = [{"contracts": 1, "gate_pass": False}, {"contracts": 2, "gate_pass": True}]
    assert asv._contiguous_safe_ceiling(per_cap) == 0


def test_evaluate_gate_matches_inline_verdict_math():
    """_evaluate_gate reproduces the inline verdict the eval used to compute.

    Builds a synthetic scenario set + sim result and asserts the extracted gate
    helper resolves operational/strict/overall pass exactly as the prior inline
    math (round-to-4 op_pass >= floor; zero historical breach days; rolling DD
    within budget; AND sizing_parity). One definition of "survives" (rigor s.4).
    """
    import trading_app.account_survival as asv

    profile = asv.get_profile("topstep_50k_mnq_auto")
    rules = asv._with_consistency_rule(asv._build_rules(profile), profile)
    # A single benign day: tiny positive pnl, no daily-loss breach, trivial DD.
    scenarios = [
        asv.DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=50.0,
            positive_pnl_dollars=50.0,
            active_lane_count=1,
            min_balance_delta_dollars=-10.0,
            max_balance_delta_dollars=60.0,
        )
    ]
    result = asv.simulate_survival(scenarios, rules, horizon_days=1, n_paths=64, seed=0)
    gate = asv._evaluate_gate(
        scenarios,
        rules,
        result,
        profile,
        min_survival_probability=asv.MIN_SURVIVAL_PROBABILITY,
        sizing_parity_ok=True,
    )
    # Re-derive the verdict independently from the same primitives.
    expected_op = round(result["operational_pass_probability"], 4) >= asv.MIN_SURVIVAL_PROBABILITY
    expected_breaches = asv._historical_daily_loss_breach_days(scenarios, rules)
    expected_dd = asv._max_observed_rolling_drawdown(scenarios, horizon_days=asv.STRICT_DD_HORIZON_DAYS)
    expected_budget = asv.effective_strict_dd_budget(profile, rules)
    expected_strict = len(expected_breaches) == 0 and expected_dd <= expected_budget
    assert gate.operational_gate_pass == expected_op
    assert gate.strict_account_gate_pass == expected_strict
    assert gate.gate_pass == (expected_op and expected_strict and True)


def test_evaluate_gate_fails_when_sizing_parity_false():
    """sizing_parity_ok=False forces gate_pass False regardless of sim outcome."""
    import trading_app.account_survival as asv

    profile = asv.get_profile("topstep_50k_mnq_auto")
    rules = asv._with_consistency_rule(asv._build_rules(profile), profile)
    scenarios = [
        asv.DailyScenario(
            trading_day="2026-01-02",
            total_pnl_dollars=50.0,
            positive_pnl_dollars=50.0,
            active_lane_count=1,
            min_balance_delta_dollars=-10.0,
            max_balance_delta_dollars=60.0,
        )
    ]
    result = asv.simulate_survival(scenarios, rules, horizon_days=1, n_paths=64, seed=0)
    gate = asv._evaluate_gate(
        scenarios,
        rules,
        result,
        profile,
        min_survival_probability=asv.MIN_SURVIVAL_PROBABILITY,
        sizing_parity_ok=False,
    )
    assert gate.gate_pass is False


def test_persist_sweep_round_trips_into_existing_c11_envelope(tmp_path, monkeypatch):
    """Persisting a sweep adds a survival_cap_sweep block without clobbering summary."""
    import trading_app.account_survival as asv

    monkeypatch.setattr(asv, "STATE_DIR", tmp_path)
    # Avoid touching the real DB/profile fingerprints — stub the canonical-inputs
    # and git_head so the re-stamp is deterministic and DB-free.
    monkeypatch.setattr(asv, "_current_survival_canonical_inputs", lambda *_a, **_k: _canonical_inputs())
    monkeypatch.setattr(asv, "get_git_head", lambda *_a, **_k: "test-head")

    pid = "topstep_50k_mnq_auto"
    report_path = asv.get_survival_report_path(pid)
    base = _survival_envelope(as_of_date="2026-06-01", operational_pass_probability=0.99, gate_pass=True)
    report_path.write_text(json.dumps(base), encoding="utf-8")

    sweep = asv.SurvivalCapSweepResult(
        profile_id=pid,
        ceiling_probed=3,
        survival_safe_ceiling=2,
        sizing_parity_ok=True,
        sizing_parity_msg="ok",
        per_cap=[
            {"contracts": 1, "gate_pass": True},
            {"contracts": 2, "gate_pass": True},
            {"contracts": 3, "gate_pass": False},
        ],
    )
    asv._persist_sweep_into_c11_envelope(pid, sweep)

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    block = raw["payload"]["survival_cap_sweep"]
    assert block["survival_safe_ceiling"] == 2
    assert block["ceiling_probed"] == 3
    assert block["horizon_days"] == 90
    assert block["n_paths"] == 10_000
    assert block["seed"] == 0
    assert block["min_survival_probability"] == 0.70
    assert block["as_of_date"] == ""
    # Original summary untouched.
    assert raw["payload"]["summary"]["operational_pass_probability"] == 0.99


def test_persist_sweep_raises_when_base_report_missing(tmp_path, monkeypatch):
    """No base C11 report -> persist raises (never silently masks a missing report)."""
    import trading_app.account_survival as asv

    monkeypatch.setattr(asv, "STATE_DIR", tmp_path)
    sweep = asv.SurvivalCapSweepResult(
        profile_id="topstep_50k_mnq_auto",
        ceiling_probed=1,
        survival_safe_ceiling=1,
        sizing_parity_ok=True,
        sizing_parity_msg="ok",
        per_cap=[{"contracts": 1, "gate_pass": True}],
    )
    with pytest.raises(FileNotFoundError):
        asv._persist_sweep_into_c11_envelope("topstep_50k_mnq_auto", sweep)
