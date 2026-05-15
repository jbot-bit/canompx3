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
    """Inject an extra check; the [i/N] header MUST reflect the new count.

    This is the canonical LOW-1 evidence: removing the manual checks_total
    constant means adding a check auto-updates every count site. Total is
    `len(PREFLIGHT_CHECKS) + 1` to keep the assertion robust as new
    production checks land.
    """
    baseline = len(rls.PREFLIGHT_CHECKS)
    new_total = baseline + 1
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
    assert f"[1/{new_total}]" in out, f"first header must read [1/{new_total}] with {new_total} checks:\n{out}"
    assert f"[{new_total}/{new_total}]" in out
    assert f"Preflight: {new_total}/{new_total} passed" in out


def test_known_failing_check_counted_toward_total(monkeypatch, capsys, all_pass_components, stub_daily_features):
    """Inject a fail-always check; total = baseline+1, passed = baseline, return False.

    Baseline is `len(rls.PREFLIGHT_CHECKS)` to honor the dynamic-count contract
    (the whole point of `preflight-checks-total-hardcode` close-out).
    """
    baseline = len(rls.PREFLIGHT_CHECKS)
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
    assert f"Preflight: {baseline}/{baseline + 1} passed" in out
    assert "FIX FAILURES" in out
    assert result is False


# ---------- Behavioral regression: refactor preserves observable shape ----------


def test_all_pass_smoke(monkeypatch, capsys, all_pass_components, stub_daily_features):
    """All baseline checks pass with valid mocks; final stdout contains the
    'All clear' summary and bool=True. Total derives from len(PREFLIGHT_CHECKS)."""
    n = len(rls.PREFLIGHT_CHECKS)
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
    assert f"Preflight: {n}/{n} passed" in out
    assert "All clear" in out
    assert result is True
    # Header shape preserved.
    for i in range(1, n + 1):
        assert f"[{i}/{n}]" in out


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
    # Trade journal is independent of components and still runs OK.
    n = len(rls.PREFLIGHT_CHECKS)
    assert f"[6/{n}] Trade journal health" in out
    # Final return reflects failures (auth, contracts, notifications).
    assert result is False


# ---------- Contract on the canonical helpers ----------


def test_preflight_checks_is_an_ordered_list():
    """Lock the canonical ordering at the test layer alongside the dataclass.

    Order matters: _check_auth must run first (populates ctx.components for
    contracts / notifications / copy-trading), and _check_copy_trading_accounts
    must run last (depends on ctx.components AND inspects profile.copies).
    """
    names = [c.__name__ for c in rls.PREFLIGHT_CHECKS]
    assert names == [
        "_check_auth",
        "_check_portfolio",
        "_check_daily_features",
        "_check_contracts",
        "_check_notifications",
        "_check_trade_journal",
        "_check_copy_trading_accounts",
    ]


def test_no_hardcoded_checks_total_constant():
    """The literal `checks_total = 6` MUST be gone. Source-grep is cheap
    and surfaces accidental reintroduction during a future merge."""
    src = (ROOT / "scripts" / "run_live_session.py").read_text(encoding="utf-8")
    assert "checks_total = 6" not in src
    assert "checks_total = len(PREFLIGHT_CHECKS)" in src


# ---------- A.6.5: Copy-trading account-resolution check ----------


def _make_copy_trading_ctx(profile_id, requested_account_id, all_accounts=None):
    """Build a PreflightContext + components stub for the copy-trading check.

    `all_accounts` is the list the stubbed contracts_class.resolve_all_account_ids()
    will return (None → empty list).
    """
    if all_accounts is None:
        all_accounts = []

    class _StubContracts:
        def __init__(self, auth, demo):
            self._auth = auth
            self._demo = demo

        def resolve_all_account_ids(self):
            return list(all_accounts)

    components = {
        "auth": SimpleNamespace(get_token=lambda: "tk_stub_xxxxx"),
        "contracts_class": _StubContracts,
    }
    return rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=None,
        components=components,
        profile_id=profile_id,
        requested_account_id=requested_account_id,
    )


def test_copy_trading_check_skipped_when_no_profile():
    ctx = _make_copy_trading_ctx(profile_id=None, requested_account_id=None)
    result = rls._check_copy_trading_accounts(ctx)
    assert result.passed is True
    assert "SKIPPED" in result.message
    assert "no profile" in result.message


def test_copy_trading_check_skipped_when_copies_le_1(monkeypatch):
    """Single-account profile → SKIPPED, no broker calls."""
    fake_profile = SimpleNamespace(copies=1)
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"single_copy_profile": fake_profile},
    )
    ctx = _make_copy_trading_ctx(profile_id="single_copy_profile", requested_account_id=None)
    result = rls._check_copy_trading_accounts(ctx)
    assert result.passed is True
    assert "SKIPPED" in result.message
    assert "copies=1" in result.message


def test_copy_trading_check_fails_on_unknown_requested_account_id(monkeypatch):
    """copies>1 + requested_account_id not in broker accounts → FAIL.

    This is the exact bug-class that crashed live-start on 2026-05-15 09:04
    (logs/session.log) and motivated commit a0b3c24b. Preflight must catch it
    BEFORE the operator clicks Start Live.
    """
    fake_profile = SimpleNamespace(copies=2)
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"multi_copy_profile": fake_profile},
    )
    ctx = _make_copy_trading_ctx(
        profile_id="multi_copy_profile",
        requested_account_id=999999,
        all_accounts=[(21944866, "PA-XFA-001"), (21944867, "PA-XFA-002")],
    )
    result = rls._check_copy_trading_accounts(ctx)
    assert result.passed is False
    assert "FAILED" in result.message
    assert "not in the broker's discovered" in result.message


def test_copy_trading_check_passes_with_none_account_id(monkeypatch):
    """copies>1 + None (auto-discover, dashboard 'Start Live' default) → PASS."""
    fake_profile = SimpleNamespace(copies=2)
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"multi_copy_profile": fake_profile},
    )
    ctx = _make_copy_trading_ctx(
        profile_id="multi_copy_profile",
        requested_account_id=None,
        all_accounts=[(21944866, "PA-XFA-001"), (21944867, "PA-XFA-002")],
    )
    result = rls._check_copy_trading_accounts(ctx)
    assert result.passed is True
    assert "OK" in result.message
    assert "copies=2" in result.message
    assert "2 accounts" in result.message


def test_copy_trading_check_fails_when_auth_failed():
    """If _check_auth already failed (ctx.components is None), copy-trading
    check cannot run — must report SKIPPED with passed=False so the operator
    sees the unverified state."""
    fake_profile = SimpleNamespace(copies=2)
    # No need to monkeypatch ACCOUNT_PROFILES — check exits before profile lookup
    # only if ctx.components is None. We build ctx WITHOUT components set.
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=None,
        components=None,
        profile_id="multi_copy_profile",
        requested_account_id=None,
    )
    # Profile lookup happens before components check — patch to keep the test
    # focused on the components=None branch.
    import trading_app.prop_profiles as pp

    original = pp.ACCOUNT_PROFILES
    pp.ACCOUNT_PROFILES = {"multi_copy_profile": fake_profile}  # type: ignore[misc]
    try:
        result = rls._check_copy_trading_accounts(ctx)
    finally:
        pp.ACCOUNT_PROFILES = original  # type: ignore[misc]
    assert result.passed is False
    assert "auth failed" in result.message
