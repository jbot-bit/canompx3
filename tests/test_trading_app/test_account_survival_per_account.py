"""Behavior tests for evaluate_per_account_survival (Stage 3a).

Per-account C11 survival for a non-uniform account_contracts map. Each account
trades the SAME lanes at a DIFFERENT contract count; the function proves each
divergent belt is survivable by re-running the CANONICAL gate at each distinct
contract scale via SizingContext — NOT the vestigial contracts_per_trade_micro.

Design mirrors test_account_survival_sweep.py (institutional-rigor § 4 — never
re-encode canonical math): _evaluate_gate runs REAL; the test drives it by the
ONE input the function does not compute — the simulate_survival result dict's
operational_pass_probability. The gold.db-touching seams are mocked; the config
seams (resolve_profile_id / get_profile / _build_rules) stay REAL.
"""

from __future__ import annotations

import pytest

from trading_app.account_survival import (
    DailyScenario,
    PerAccountSurvivalResult,
    evaluate_per_account_survival,
)

PROFILE_ID = "topstep_50k_mnq_auto"


def _passing_scenarios() -> list[DailyScenario]:
    return [
        DailyScenario(
            trading_day=f"2026-01-{d:02d}",
            total_pnl_dollars=10.0,
            positive_pnl_dollars=10.0,
            active_lane_count=1,
        )
        for d in range(1, 6)
    ]


def _result(operational_pass_probability: float) -> dict:
    return {"operational_pass_probability": operational_pass_probability}


def _install_common_mocks(monkeypatch, *, scenarios=None):
    scen = scenarios if scenarios is not None else _passing_scenarios()

    monkeypatch.setattr(
        "trading_app.account_survival._assert_sizing_parity",
        lambda _pid: (True, "ok"),
    )

    class _FakePortfolio:
        account_equity = 50_000.0
        risk_per_trade_pct = 0.5

        class _Strat:
            def __init__(self, sid):
                self.strategy_id = sid

        strategies = [_Strat("MNQ_TEST")]

    monkeypatch.setattr(
        "trading_app.account_survival.build_profile_portfolio",
        lambda *, profile_id: _FakePortfolio(),
    )

    class _DummyCon:
        def close(self):
            pass

    monkeypatch.setattr("trading_app.account_survival.duckdb.connect", lambda *a, **k: _DummyCon())
    monkeypatch.setattr("trading_app.account_survival.configure_connection", lambda _con: None)
    monkeypatch.setattr(
        "trading_app.account_survival._scenarios_for_context",
        lambda *a, **k: (list(scen), {}),
    )
    return scen


# ----------------------------------------------------------------------------- guards


def test_empty_map_raises():
    with pytest.raises(ValueError, match="must be non-empty"):
        evaluate_per_account_survival(PROFILE_ID, {})


def test_non_positive_contracts_raises():
    with pytest.raises(ValueError, match="must be int >= 1"):
        evaluate_per_account_survival(PROFILE_ID, {101: 0})


# ----------------------------------------------------------------------------- per-account fan-out


def test_one_verdict_per_account_distinct_counts_share_verdict(monkeypatch):
    # Map: 101@1, 202@3, 303@3. Two DISTINCT counts {1, 3} → simulate_survival is
    # called exactly twice; 202 and 303 (both @3) share the count-3 verdict.
    _install_common_mocks(monkeypatch)
    calls = {"n": 0}

    def _fake_sim(*a, **k):
        calls["n"] += 1
        return _result(0.95)

    monkeypatch.setattr("trading_app.account_survival.simulate_survival", _fake_sim)

    res = evaluate_per_account_survival(PROFILE_ID, {101: 1, 202: 3, 303: 3}, n_paths=16)

    assert isinstance(res, PerAccountSurvivalResult)
    assert calls["n"] == 2, "simulate_survival must run once per DISTINCT contract count"
    assert {e["account_id"] for e in res.per_account} == {101, 202, 303}
    by_aid = {e["account_id"]: e for e in res.per_account}
    assert by_aid[101]["contracts"] == 1
    assert by_aid[202]["contracts"] == by_aid[303]["contracts"] == 3
    assert res.all_pass is True


def test_one_account_failing_count_marks_all_pass_false(monkeypatch):
    # 101@1 passes, 202@3 fails → all_pass False, and only the @3 account fails.
    _install_common_mocks(monkeypatch)
    seen_counts: list[int] = []

    def _fake_sim(*a, **k):
        # The function probes counts in sorted order {1, 3}; first call = count 1.
        idx = len(seen_counts)
        seen_counts.append(idx)
        return _result(0.95 if idx == 0 else 0.40)  # count 1 PASS, count 3 FAIL

    monkeypatch.setattr("trading_app.account_survival.simulate_survival", _fake_sim)

    res = evaluate_per_account_survival(PROFILE_ID, {101: 1, 202: 3}, n_paths=16)

    by_aid = {e["account_id"]: e for e in res.per_account}
    assert by_aid[101]["gate_pass"] is True
    assert by_aid[202]["gate_pass"] is False
    assert res.all_pass is False


def test_sizing_context_carries_distinct_count_not_vestigial_field(monkeypatch):
    # C2 guard: the per-account contract scale enters via SizingContext
    # (max_contracts_by_strategy), NOT SurvivalRules.contracts_per_trade_micro.
    # Capture the SizingContext handed to _scenarios_for_context per call.
    _install_common_mocks(monkeypatch)
    captured: list[dict] = []

    real_scen = _passing_scenarios()

    def _capture_scen(*a, **k):
        sm = k["size_model"]
        captured.append(dict(sm.max_contracts_by_strategy))
        return (list(real_scen), {})

    monkeypatch.setattr("trading_app.account_survival._scenarios_for_context", _capture_scen)
    monkeypatch.setattr("trading_app.account_survival.simulate_survival", lambda *a, **k: _result(0.95))

    evaluate_per_account_survival(PROFILE_ID, {101: 1, 202: 4}, n_paths=16)

    # Distinct counts {1, 4} → two SizingContexts, each capping every strategy at
    # that account's contract count.
    caps = sorted({next(iter(c.values())) for c in captured})
    assert caps == [1, 4]
    # Each context caps ALL strategies uniformly at the swept count.
    for c in captured:
        assert len(set(c.values())) == 1


def test_no_state_persisted(monkeypatch, tmp_path):
    # Read-only EVIDENCE — must not write the C11 envelope (no clamp lift).
    _install_common_mocks(monkeypatch)
    monkeypatch.setattr("trading_app.account_survival.simulate_survival", lambda *a, **k: _result(0.95))
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)

    evaluate_per_account_survival(PROFILE_ID, {101: 1, 202: 2}, n_paths=16)

    # Nothing written under the state dir.
    assert list(tmp_path.iterdir()) == []
