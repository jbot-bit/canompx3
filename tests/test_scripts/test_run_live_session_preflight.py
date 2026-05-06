"""Tests for `scripts.run_live_session._run_preflight` after the
list-of-callables refactor.

Closes debt-ledger entry `preflight-checks-total-hardcode`. Validates that:
- `checks_total = len(PREFLIGHT_CHECKS)` is dynamic (adding a check
  auto-bumps the count and the summary).
- A failing injected check is counted toward the total (fail-closed).
- Existing happy-path and auth-fail-cascade behavior is preserved.

Per institutional-rigor § 1, the load-bearing tests for the close-out
are #1 (dynamic count) and #2 (injected fail counted) — they prove the
debt is gone. #3 and #4 are regression coverage so the refactor is
behavior-identical to the manual constant.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_live_session as rls  # noqa: E402


def _build_portfolio():
    """Minimal portfolio fixture matching the attrs `_check_portfolio`
    and `_check_daily_features` read."""
    strategy = SimpleNamespace(
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        orb_label="NYSE_OPEN",
        entry_model="E2",
        rr_target=1.0,
        orb_minutes=5,
        win_rate=0.55,
        expectancy_r=0.183,
        sample_size=200,
        filter_type="COST_LT12",
    )
    return SimpleNamespace(strategies=[strategy])


@pytest.fixture
def all_pass_components():
    """Mock broker components for the auth + contracts + notifications path."""
    auth = SimpleNamespace(get_token=lambda: "tk_abcdef1234567890")
    contracts_cls = lambda **_kw: SimpleNamespace(  # noqa: E731
        resolve_front_month=lambda _instrument: "MNQU6"
    )
    return {"auth": auth, "contracts_class": contracts_cls}


@pytest.fixture
def stub_daily_features(monkeypatch):
    """Stub SessionOrchestrator._build_daily_features_row → ATR populated."""
    monkeypatch.setattr(
        rls.SessionOrchestrator,
        "_build_daily_features_row",
        staticmethod(lambda _td, _instr, orb_minutes=5: {"atr_20": 50.0, "atr_vel_regime": "NORMAL"}),
    )


# ---------- LOAD-BEARING tests for the LOW-1 close-out ----------


def test_checks_total_equals_len_checks(monkeypatch, capsys, all_pass_components, stub_daily_features):
    """Inject a 7th check; the [i/N] header MUST show /7, not /6.

    This is the canonical LOW-1 evidence: removing the manual checks_total
    constant means adding a check auto-updates every count site.
    """
    extra_check = lambda ctx: rls.CheckResult(True, "OK (injected)")  # noqa: E731
    extra_check.__doc__ = "Injected extra check"
    monkeypatch.setattr(rls, "PREFLIGHT_CHECKS", [*rls.PREFLIGHT_CHECKS, extra_check])
    monkeypatch.setattr(rls, "create_broker_components", lambda *a, **kw: all_pass_components, raising=False)
    # Patch the broker_factory imports inside _check_auth via attribute lookup.
    import trading_app.live.broker_factory as bf

    monkeypatch.setattr(bf, "create_broker_components", lambda *a, **kw: all_pass_components)
    monkeypatch.setattr(bf, "get_broker_name", lambda: "topstep")
    monkeypatch.setattr(
        rls,
        "_run_lightweight_component_self_tests",
        lambda *, instrument: {"notifications": True, "brackets": True, "fill_poller": True},
    )

    # Stub TradeJournal so check_trade_journal returns OK without filesystem.
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    rls._run_preflight("MNQ", "topstep", demo=True, portfolio=_build_portfolio())
    out = capsys.readouterr().out
    assert "[1/7]" in out, f"first check header must read [1/7] when 7 checks registered:\n{out}"
    assert "[7/7]" in out
    assert "Preflight: 7/7 passed" in out


def test_known_failing_check_counted_toward_total(monkeypatch, capsys, all_pass_components, stub_daily_features):
    """Inject a fail-always check; total must be 7, passed must be 6, return False."""
    fail_check = lambda ctx: rls.CheckResult(False, "FAILED: synthetic")  # noqa: E731
    fail_check.__doc__ = "Synthetic fail"
    monkeypatch.setattr(rls, "PREFLIGHT_CHECKS", [*rls.PREFLIGHT_CHECKS, fail_check])

    import trading_app.live.broker_factory as bf

    monkeypatch.setattr(bf, "create_broker_components", lambda *a, **kw: all_pass_components)
    monkeypatch.setattr(bf, "get_broker_name", lambda: "topstep")
    monkeypatch.setattr(
        rls,
        "_run_lightweight_component_self_tests",
        lambda *, instrument: {"notifications": True, "brackets": True, "fill_poller": True},
    )
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    result = rls._run_preflight("MNQ", "topstep", demo=True, portfolio=_build_portfolio())
    out = capsys.readouterr().out
    assert "Preflight: 6/7 passed" in out
    assert "FIX FAILURES" in out
    assert result is False


# ---------- Behavioral regression: refactor preserves observable shape ----------


def test_all_pass_smoke(monkeypatch, capsys, all_pass_components, stub_daily_features):
    """All 6 checks pass with valid mocks; final stdout contains the
    'All clear' summary and bool=True."""
    import trading_app.live.broker_factory as bf

    monkeypatch.setattr(bf, "create_broker_components", lambda *a, **kw: all_pass_components)
    monkeypatch.setattr(bf, "get_broker_name", lambda: "topstep")
    monkeypatch.setattr(
        rls,
        "_run_lightweight_component_self_tests",
        lambda *, instrument: {"notifications": True, "brackets": True, "fill_poller": True},
    )
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    result = rls._run_preflight("MNQ", "topstep", demo=True, portfolio=_build_portfolio())
    out = capsys.readouterr().out
    assert "Preflight: 6/6 passed" in out
    assert "All clear" in out
    assert result is True
    # Header shape preserved.
    for i in range(1, 7):
        assert f"[{i}/6]" in out


def test_auth_fail_cascades(monkeypatch, capsys, stub_daily_features):
    """Auth raises → check 1 FAILED, check 4 SKIPPED (auth failed),
    check 5 FAILED (auth failed), check 6 still runs OK."""
    import trading_app.live.broker_factory as bf

    def _raise(*_a, **_kw):
        raise RuntimeError("auth refused")

    monkeypatch.setattr(bf, "create_broker_components", _raise)
    monkeypatch.setattr(bf, "get_broker_name", lambda: "topstep")
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    result = rls._run_preflight("MNQ", "topstep", demo=True, portfolio=_build_portfolio())
    out = capsys.readouterr().out
    # Check 1 failed inline.
    assert "FAILED: auth refused" in out
    # Check 4 reports SKIPPED on the inline line.
    assert "SKIPPED (auth failed)" in out
    # Check 5 reports the auth-failed message.
    assert "FAILED: auth failed" in out
    # Check 6 (trade journal) is independent of components and still runs OK.
    assert "[6/6] Trade journal health" in out
    # Final return reflects 3 failures.
    assert result is False


# ---------- Contract on the canonical helpers ----------


def test_preflight_checks_is_an_ordered_list_of_six():
    """Lock the canonical ordering at the test layer alongside the dataclass."""
    names = [c.__name__ for c in rls.PREFLIGHT_CHECKS]
    assert names == [
        "_check_auth",
        "_check_portfolio",
        "_check_daily_features",
        "_check_contracts",
        "_check_notifications",
        "_check_trade_journal",
    ]


def test_no_hardcoded_checks_total_constant():
    """The literal `checks_total = 6` MUST be gone. Source-grep is cheap
    and surfaces accidental reintroduction during a future merge."""
    src = (ROOT / "scripts" / "run_live_session.py").read_text(encoding="utf-8")
    assert "checks_total = 6" not in src
    assert "checks_total = len(PREFLIGHT_CHECKS)" in src
