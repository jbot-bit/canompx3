"""Behavior tests for the D-3 survival-cap sweep (Stage 1).

Covers ``trading_app.account_survival.sweep_survival_cap`` and its helpers
``_contiguous_safe_ceiling`` and ``_persist_sweep_into_c11_envelope``.

Design (institutional-rigor § 4 — never re-encode canonical math):
    * ``_evaluate_gate`` and ``_contiguous_safe_ceiling`` run REAL in every test.
      The test NEVER re-implements the pass criteria — it drives the gate by
      controlling the ONE input the sweep does not compute itself: the
      ``simulate_survival`` result dict's ``operational_pass_probability``.
    * The flip lever is a call-counter on the ``simulate_survival`` mock. The
      sweep calls it strictly in order n=1,2,...,ceiling, so call N maps 1:1 to
      contract count ``n`` — deterministic, reads nothing fragile.
    * Mocked scenarios are tiny-positive PnL so the REAL strict-account gate
      (``historical_max_observed_90d_dd_dollars`` / daily-loss breach days)
      always PASSes; the gate verdict then rides purely on the per-n
      ``operational_pass_probability`` we thread in. Verified empirically before
      writing these tests (real ``_evaluate_gate``: opp=0.95 -> PASS, opp=0.40 ->
      FAIL on the express profile, strict gate True in both).

Hermetic: gold.db is absent in CI, so the DB seams (``build_profile_portfolio``,
``_assert_sizing_parity``, ``duckdb.connect``, ``_scenarios_for_context``) are
mocked. The config seams (``resolve_profile_id`` / ``get_profile`` /
``_build_rules`` / ``get_profile_lane_definitions``) are left REAL against a real
profile id for fidelity — they read prop_profiles, not the DB.
"""

from __future__ import annotations

import json

import pytest

from trading_app.account_survival import (
    DailyScenario,
    SurvivalCapSweepResult,
    _build_rules,
    _contiguous_safe_ceiling,
    _evaluate_gate,
    _persist_sweep_into_c11_envelope,
    _with_consistency_rule,
    get_survival_report_path,
    sweep_survival_cap,
)
from trading_app.prop_profiles import get_profile

PROFILE_ID = "topstep_50k_mnq_auto"


# ----------------------------------------------------------------------------- helpers


def _passing_scenarios() -> list[DailyScenario]:
    """Tiny-positive days: REAL strict-account gate passes (DD=0, no breaches)."""
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
    """Minimal ``simulate_survival`` result dict honoring the keys the sweep reads.

    ``_evaluate_gate`` reads only ``operational_pass_probability`` off the result
    dict (the strict-account half is computed from ``scenarios``). The other keys
    the real ``simulate_survival`` returns are not read by the gate or the sweep
    body, so the minimal stub is contract-complete for what is consumed.
    """
    return {"operational_pass_probability": operational_pass_probability}


def _install_common_mocks(monkeypatch, *, scenarios=None):
    """Mock the gold.db-touching seams; leave config/gate/ceiling math REAL."""
    scen = scenarios if scenarios is not None else _passing_scenarios()

    # sizing-parity opens gold.db via build_profile_portfolio — clean pass.
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

    monkeypatch.setattr(
        "trading_app.account_survival.duckdb.connect",
        lambda *a, **k: _DummyCon(),
    )
    monkeypatch.setattr(
        "trading_app.account_survival.configure_connection",
        lambda _con: None,
    )

    # scenarios are invariant to n in these tests (the flip rides on the sim
    # result, not the scenarios) — return the same controlled set every call.
    monkeypatch.setattr(
        "trading_app.account_survival._scenarios_for_context",
        lambda *a, **k: (list(scen), {}),
    )
    return scen


# ----------------------------------------------------------------------------- _contiguous_safe_ceiling (pure unit)


def test_contiguous_safe_ceiling_all_pass():
    per_cap = [{"contracts": n, "gate_pass": True} for n in range(1, 4)]
    assert _contiguous_safe_ceiling(per_cap) == 3


def test_contiguous_safe_ceiling_cap1_fail_returns_zero():
    per_cap = [{"contracts": 1, "gate_pass": False}, {"contracts": 2, "gate_pass": True}]
    assert _contiguous_safe_ceiling(per_cap) == 0


def test_contiguous_safe_ceiling_non_contiguous_pass_not_honored():
    # pass at 1, fail at 2, pass at 3 -> a gap above a failure is NOT trusted.
    per_cap = [
        {"contracts": 1, "gate_pass": True},
        {"contracts": 2, "gate_pass": False},
        {"contracts": 3, "gate_pass": True},
    ]
    assert _contiguous_safe_ceiling(per_cap) == 1


# ----------------------------------------------------------------------------- sweep_survival_cap


def test_sweep_ceiling_below_1_raises():
    with pytest.raises(ValueError, match="ceiling must be >= 1"):
        sweep_survival_cap(PROFILE_ID, ceiling=0, write_state=False)


def test_sweep_cap1_byte_parity_with_single_gate(monkeypatch):
    """ceiling=1 sweep verdict == a single REAL _evaluate_gate on identical inputs.

    Proves the sweep routes the cap=1 verdict through the SAME canonical gate (no
    re-encode): both this test's direct _evaluate_gate call and the sweep's
    internal call see identical (scenarios, rules, result, parity) and must agree.
    """
    scen = _install_common_mocks(monkeypatch)
    monkeypatch.setattr(
        "trading_app.account_survival.simulate_survival",
        lambda *a, **k: _result(0.95),
    )

    sweep = sweep_survival_cap(PROFILE_ID, ceiling=1, n_paths=16, write_state=False)

    # Independently resolve the canonical gate on the same inputs.
    profile = get_profile(PROFILE_ID)
    rules = _with_consistency_rule(_build_rules(profile), profile)
    expected = _evaluate_gate(
        scen,
        rules,
        _result(0.95),
        profile,
        min_survival_probability=sweep.min_survival_probability,
        sizing_parity_ok=True,
    )

    assert len(sweep.per_cap) == 1
    entry = sweep.per_cap[0]
    assert entry["contracts"] == 1
    # Field-level equality on the gate verdict (not a brittle whole-object compare).
    assert entry["gate_pass"] == expected.gate_pass
    assert entry["operational_gate_pass"] == expected.operational_gate_pass
    assert entry["strict_account_gate_pass"] == expected.strict_account_gate_pass
    assert entry["operational_pass_probability"] == expected.operational_pass_probability
    assert sweep.survival_safe_ceiling == (1 if expected.gate_pass else 0)
    # Guard the test itself: it must exercise a real PASS, not a trivial tie.
    assert expected.gate_pass is True


def test_sweep_returns_contiguous_safe_ceiling(monkeypatch):
    """Synthetic flip: gate passes at n<=2, fails at n=3 -> ceiling == 2."""
    _install_common_mocks(monkeypatch)
    # Call N (1-indexed) maps to contract n. Pass for n in {1,2}, fail at n>=3.
    calls = {"n": 0}

    def _fake_sim(*a, **k):
        calls["n"] += 1
        return _result(0.95 if calls["n"] <= 2 else 0.40)

    monkeypatch.setattr("trading_app.account_survival.simulate_survival", _fake_sim)

    sweep = sweep_survival_cap(PROFILE_ID, ceiling=5, n_paths=16, write_state=False)

    assert [e["contracts"] for e in sweep.per_cap] == [1, 2, 3, 4, 5]
    assert [e["gate_pass"] for e in sweep.per_cap] == [True, True, False, False, False]
    assert sweep.survival_safe_ceiling == 2


def test_sweep_fail_closed_when_cap1_fails(monkeypatch):
    """cap=1 gate FAIL -> survival_safe_ceiling == 0 (fail-closed signal)."""
    _install_common_mocks(monkeypatch)
    monkeypatch.setattr(
        "trading_app.account_survival.simulate_survival",
        lambda *a, **k: _result(0.40),  # below 0.70 floor -> operational gate fails
    )

    sweep = sweep_survival_cap(PROFILE_ID, ceiling=3, n_paths=16, write_state=False)

    assert sweep.per_cap[0]["gate_pass"] is False
    assert sweep.survival_safe_ceiling == 0


def test_sweep_non_contiguous_pass_not_honored(monkeypatch):
    """pass at 1, fail at 2, pass at 3 -> ceiling == 1 (gap above failure ignored)."""
    _install_common_mocks(monkeypatch)
    calls = {"n": 0}

    def _fake_sim(*a, **k):
        calls["n"] += 1
        # PASS, FAIL, PASS
        return _result(0.95 if calls["n"] in (1, 3) else 0.40)

    monkeypatch.setattr("trading_app.account_survival.simulate_survival", _fake_sim)

    sweep = sweep_survival_cap(PROFILE_ID, ceiling=3, n_paths=16, write_state=False)

    assert [e["gate_pass"] for e in sweep.per_cap] == [True, False, True]
    assert sweep.survival_safe_ceiling == 1


# ----------------------------------------------------------------------------- _persist_sweep_into_c11_envelope


def _sweep_result() -> SurvivalCapSweepResult:
    return SurvivalCapSweepResult(
        profile_id=PROFILE_ID,
        ceiling_probed=3,
        survival_safe_ceiling=2,
        sizing_parity_ok=True,
        sizing_parity_msg="ok",
        per_cap=[
            {"contracts": 1, "gate_pass": True},
            {"contracts": 2, "gate_pass": True},
            {"contracts": 3, "gate_pass": False},
        ],
        horizon_days=90,
        n_paths=16,
        seed=0,
        min_survival_probability=0.70,
        as_of_date="2026-06-11",
    )


def _write_base_report(path, *, payload_extra=None):
    """Write a minimal C11 envelope with an existing summary/rules/metadata."""
    payload = {
        "summary": {"profile_id": PROFILE_ID, "gate_pass": True},
        "rules": {"dd_limit_dollars": 2000.0},
        "metadata": {"source_days": 5},
    }
    if payload_extra:
        payload.update(payload_extra)
    path.write_text(
        json.dumps(
            {
                "state_type": "account_survival",
                "payload": payload,
                "canonical_inputs": {"profile_id": PROFILE_ID},
                "freshness": {"as_of_date": "2026-06-11"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_persist_sweep_roundtrips_into_c11_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    monkeypatch.setattr(
        "trading_app.account_survival._current_survival_canonical_inputs",
        lambda *_a, **_k: {"profile_id": PROFILE_ID, "code_fingerprint": "fp"},
    )
    report_path = get_survival_report_path(PROFILE_ID)
    _write_base_report(report_path)

    _persist_sweep_into_c11_envelope(PROFILE_ID, _sweep_result())

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    payload = raw["payload"]
    # New block merged.
    sweep_block = payload["survival_cap_sweep"]
    assert sweep_block["ceiling_probed"] == 3
    assert sweep_block["survival_safe_ceiling"] == 2
    assert sweep_block["per_cap"][0]["contracts"] == 1
    assert "computed_at_utc" in sweep_block
    # Existing blocks left intact.
    assert payload["summary"] == {"profile_id": PROFILE_ID, "gate_pass": True}
    assert payload["rules"] == {"dd_limit_dollars": 2000.0}
    assert payload["metadata"] == {"source_days": 5}
    # Envelope re-stamped through build_state_envelope.
    assert raw["state_type"] == "account_survival"
    assert "canonical_inputs" in raw and "freshness" in raw


def test_persist_sweep_raises_when_base_report_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    # No base report written -> must raise, never silently mask.
    with pytest.raises(FileNotFoundError, match="base C11 report missing"):
        _persist_sweep_into_c11_envelope(PROFILE_ID, _sweep_result())


def test_persist_sweep_raises_on_unversioned_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_app.account_survival.STATE_DIR", tmp_path)
    report_path = get_survival_report_path(PROFILE_ID)
    # Legacy/corrupt: no versioned dict payload.
    report_path.write_text(json.dumps({"state_type": "account_survival"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no versioned payload"):
        _persist_sweep_into_c11_envelope(PROFILE_ID, _sweep_result())
