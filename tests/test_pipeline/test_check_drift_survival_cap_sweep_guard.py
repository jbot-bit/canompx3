"""Mutation-proof tests for ``check_live_contract_cap_traces_to_swept_ceiling``.

Survival-cap sweep Stage 1 (D-3). The check is a BLOCKING capital gate: a live
contract cap above 1 (``DEPLOYED_MAX_CONTRACTS_CLAMP`` in portfolio.py) must trace
to a persisted, survival-PASSED swept ceiling for every active profile. While the
clamp is 1 it is a no-op PASS.

Fail-direction (feedback_capital_guard_fail_direction_matters): every unresolved
condition must BLOCK (a false PASS would let real capital size up on unproven
survival math). These tests inject known violations and prove each blocks
(integrity-guardian.md § 7 — never trust a check without injection).
"""

from __future__ import annotations

import json

import pytest

import pipeline.check_drift as cd


def _envelope_with_sweep(ceiling: int | None) -> dict:
    """A versioned C11 envelope; includes a survival_cap_sweep block iff ceiling is not None."""
    payload: dict = {"summary": {"gate_pass": True}, "rules": {}, "metadata": {}}
    if ceiling is not None:
        payload["survival_cap_sweep"] = {
            "ceiling_probed": max(ceiling, 1),
            "survival_safe_ceiling": ceiling,
            "sizing_parity_ok": True,
            "horizon_days": 90,
            "n_paths": 10_000,
            "seed": 0,
            "min_survival_probability": 0.70,
            "per_cap": [],
        }
    return {
        "schema_version": 1,
        "state_type": "account_survival",
        "canonical_inputs": {},
        "freshness": {"as_of_date": "2026-06-01", "max_age_days": 30},
        "payload": payload,
    }


@pytest.fixture
def one_active_profile(tmp_path, monkeypatch):
    """Wire the check to a single active profile whose report lives in tmp_path.

    Returns a writer ``write(envelope|None, *, valid=True)`` so each test sets up
    the persisted state it wants to inject.
    """
    pid = "topstep_test"
    report_path = tmp_path / f"account_survival_{pid}.json"

    import trading_app.account_survival as asv

    monkeypatch.setattr(asv, "get_survival_report_path", lambda _pid=None: report_path)
    monkeypatch.setattr("trading_app.prop_profiles.get_active_profile_ids", lambda: [pid])

    def _state(*, valid: bool):
        # read_survival_report_state is monkeypatched per-test for validity; default valid.
        return {"valid": valid, "reason": None if valid else "stale", "summary": {}}

    def write(envelope, *, valid=True):
        if envelope is None:
            if report_path.exists():
                report_path.unlink()
        else:
            report_path.write_text(json.dumps(envelope), encoding="utf-8")
        monkeypatch.setattr(asv, "read_survival_report_state", lambda _pid=None: _state(valid=valid))

    return write


def test_real_repo_passes_at_clamp_1():
    """The REAL repo (clamp==1) → no-op PASS, zero violations."""
    assert cd._deployed_max_contracts_clamp_literal(cd.PROJECT_ROOT) == 1
    assert cd.check_live_contract_cap_traces_to_swept_ceiling() == []


def test_clamp_unreadable_fails_closed(monkeypatch):
    """Cannot resolve the clamp literal → BLOCK (fail closed)."""
    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: None)
    violations = cd.check_live_contract_cap_traces_to_swept_ceiling()
    assert len(violations) == 1
    assert "cannot resolve DEPLOYED_MAX_CONTRACTS_CLAMP" in violations[0]


def test_clamp_above_1_with_no_sweep_blocks(monkeypatch, one_active_profile):
    """clamp>1 but NO persisted sweep → BLOCK."""
    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: 2)
    one_active_profile(_envelope_with_sweep(None), valid=True)
    violations = cd.check_live_contract_cap_traces_to_swept_ceiling()
    assert len(violations) == 1
    assert "NO persisted survival_cap_sweep" in violations[0]


def test_clamp_above_1_with_ceiling_below_clamp_blocks(monkeypatch, one_active_profile):
    """clamp>1 and survival_safe_ceiling < clamp → BLOCK."""
    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: 3)
    one_active_profile(_envelope_with_sweep(2), valid=True)
    violations = cd.check_live_contract_cap_traces_to_swept_ceiling()
    assert len(violations) == 1
    assert "survival_safe_ceiling=2 < clamp=3" in violations[0]


def test_clamp_above_1_with_ceiling_at_or_above_clamp_passes(monkeypatch, one_active_profile):
    """clamp>1 and survival_safe_ceiling >= clamp → PASS (the proven-safe case)."""
    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: 2)
    one_active_profile(_envelope_with_sweep(3), valid=True)
    assert cd.check_live_contract_cap_traces_to_swept_ceiling() == []


def test_clamp_above_1_with_noncanonical_sweep_config_blocks(monkeypatch, one_active_profile):
    """A weaker ad hoc sweep must not justify a live clamp lift."""
    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: 2)
    env = _envelope_with_sweep(3)
    env["payload"]["survival_cap_sweep"]["min_survival_probability"] = 0.50
    one_active_profile(env, valid=True)
    violations = cd.check_live_contract_cap_traces_to_swept_ceiling()
    assert len(violations) == 1
    assert "min_survival_probability=0.5 is not canonical" in violations[0]


def test_clamp_above_1_with_invalid_report_blocks(monkeypatch, one_active_profile):
    """clamp>1 and the survival report is stale/invalid → BLOCK."""
    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: 2)
    one_active_profile(_envelope_with_sweep(5), valid=False)
    violations = cd.check_live_contract_cap_traces_to_swept_ceiling()
    assert len(violations) == 1
    assert "not valid/current" in violations[0]


def test_deployed_max_contracts_clamp_literal_reads_portfolio():
    """The AST reader returns the live clamp int literal (1 today)."""
    val = cd._deployed_max_contracts_clamp_literal(cd.PROJECT_ROOT)
    assert val == 1


def test_clamp_literal_missing_module_returns_none(tmp_path):
    """No portfolio.py in the given root → None (caller fails closed)."""
    assert cd._deployed_max_contracts_clamp_literal(tmp_path) is None


def test_corrupted_report_blocks_and_warns(monkeypatch, one_active_profile, capsys):
    """clamp>1 and a corrupted (unparseable) report → BLOCK + WARN (not silent).

    Closes the adversarial-audit finding (2026-06-10): _read_sweep_block must not
    silently swallow a read/parse error as "sweep missing".
    """
    import trading_app.account_survival as asv

    monkeypatch.setattr(cd, "_deployed_max_contracts_clamp_literal", lambda _root: 2)
    # Valid envelope state (so the check proceeds past the validity gate) but the
    # on-disk file the sweep-block reader opens is corrupted JSON.
    one_active_profile(_envelope_with_sweep(5), valid=True)
    report_path = asv.get_survival_report_path("topstep_test")
    report_path.write_text("{not valid json", encoding="utf-8")

    violations = cd.check_live_contract_cap_traces_to_swept_ceiling()
    assert len(violations) == 1
    assert "NO persisted survival_cap_sweep" in violations[0]  # fail-closed BLOCK
    assert "unreadable" in capsys.readouterr().out  # surfaced, not silent
