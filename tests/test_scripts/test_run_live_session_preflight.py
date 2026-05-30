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


@pytest.fixture
def stub_capital_state_ok(monkeypatch):
    """Make a profiled routing session fully valid for the full-preflight smoke
    tests, now that _check_survival_report / _check_sr_state fail-closed without
    a profile (baf99cfe H2 fix). The smoke tests launch with
    profile_id='topstep_50k_mnq_auto' to satisfy the new profile-required gate;
    this fixture supplies the two collaborators that profiled run then needs:

    - read_lifecycle_state → C11 gate_ok / C12 valid (the gates under fix).
    - ACCOUNT_PROFILES['topstep_50k_mnq_auto'].copies=1 so the unrelated
      copy-trading / shadow-MLL checks SKIP. The real profile is copies=2, which
      would (correctly) trip _check_shadow_copy_loss_protection — out of scope
      for these count-derivation smoke tests, so we pin a single-copy stub.
    """
    monkeypatch.setattr(
        "trading_app.lifecycle_state.read_lifecycle_state",
        lambda profile_id=None: {
            "criterion11": {"gate_ok": True, "operational_pass_probability": 0.85},
            "criterion12": {"available": True, "valid": True, "counts": {"ALARM": 0, "NO_DATA": 0}},
        },
    )
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"topstep_50k_mnq_auto": SimpleNamespace(copies=1)},
    )


@pytest.fixture
def stub_telemetry_mature(monkeypatch, tmp_path):
    """Synthesize >=30 distinct MNQ trading_days so _check_telemetry_maturity passes.

    Smoke tests that exercise the full PREFLIGHT_CHECKS list need this -- without
    it the real signal logs at repo root would be evaluated (currently 11<30)
    and the telemetry check would correctly return FAILED, breaking the smoke.
    """
    import json
    from datetime import UTC, datetime, timedelta

    from trading_app.live.telemetry_maturity import MIN_TELEMETRY_TRADING_DAYS

    sig_dir = tmp_path / "sig"
    sig_dir.mkdir()
    for i in range(MIN_TELEMETRY_TRADING_DAYS):
        day = (datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        rec = {
            "ts": f"{day}T20:00:00+00:00",
            "instrument": "MNQ",
            "type": "SESSION_START",
            "contract": "CON.F.US.MNQ.M26",
            "mode": "signal_only",
        }
        (sig_dir / f"live_signals_{day}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(rls.SessionOrchestrator, "SIGNALS_DIR", sig_dir)


# ---------- LOAD-BEARING tests for the LOW-1 close-out ----------


def test_checks_total_equals_len_checks(
    monkeypatch, capsys, all_pass_components, stub_daily_features, stub_telemetry_mature, stub_capital_state_ok
):
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
        lambda *, instrument, components=None: {"notifications": True, "brackets": True, "fill_poller": True},
    )

    # Stub TradeJournal so check_trade_journal returns OK without filesystem.
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    rls._run_preflight("MNQ", "topstep", demo=True, portfolio=_build_portfolio(), profile_id="topstep_50k_mnq_auto")
    out = capsys.readouterr().out
    assert f"[1/{new_total}]" in out, f"first header must read [1/{new_total}] with {new_total} checks:\n{out}"
    assert f"[{new_total}/{new_total}]" in out
    assert f"Preflight: {new_total}/{new_total} passed" in out


def test_known_failing_check_counted_toward_total(
    monkeypatch, capsys, all_pass_components, stub_daily_features, stub_telemetry_mature, stub_capital_state_ok
):
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
        lambda *, instrument, components=None: {"notifications": True, "brackets": True, "fill_poller": True},
    )
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    result = rls._run_preflight(
        "MNQ", "topstep", demo=True, portfolio=_build_portfolio(), profile_id="topstep_50k_mnq_auto"
    )
    out = capsys.readouterr().out
    assert f"Preflight: {baseline}/{baseline + 1} passed" in out
    assert "FIX FAILURES" in out
    assert result is False


# ---------- Behavioral regression: refactor preserves observable shape ----------


def test_all_pass_smoke(
    monkeypatch, capsys, all_pass_components, stub_daily_features, stub_telemetry_mature, stub_capital_state_ok
):
    """All baseline checks pass with valid mocks; final stdout contains the
    'All clear' summary and bool=True. Total derives from len(PREFLIGHT_CHECKS)."""
    n = len(rls.PREFLIGHT_CHECKS)
    import trading_app.live.broker_factory as bf

    monkeypatch.setattr(bf, "create_broker_components", lambda *a, **kw: all_pass_components)
    monkeypatch.setattr(bf, "get_broker_name", lambda: "topstep")
    monkeypatch.setattr(
        rls,
        "_run_lightweight_component_self_tests",
        lambda *, instrument, components=None: {"notifications": True, "brackets": True, "fill_poller": True},
    )
    import trading_app.live.trade_journal as tj

    monkeypatch.setattr(tj, "TradeJournal", lambda *a, **kw: SimpleNamespace(is_healthy=True))

    result = rls._run_preflight(
        "MNQ", "topstep", demo=True, portfolio=_build_portfolio(), profile_id="topstep_50k_mnq_auto"
    )
    out = capsys.readouterr().out
    assert f"Preflight: {n}/{n} passed" in out
    assert "All clear" in out
    assert result is True
    # Header shape preserved.
    for i in range(1, n + 1):
        assert f"[{i}/{n}]" in out


def test_auth_fail_cascades(monkeypatch, capsys, stub_daily_features, stub_telemetry_mature):
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
    assert f"[8/{n}] Trade journal health" in out
    # Final return reflects failures (auth, contracts, notifications).
    assert result is False


def test_survival_report_check_blocks_capital_profile(monkeypatch):
    """Criterion 11 must fail closed for profile demo/live paths."""
    mock_lifecycle = {
        "criterion11": {
            "gate_ok": False,
            "gate_msg": "BLOCKED: no Criterion 11 survival report",
        },
        "criterion12": {"available": True, "valid": True, "counts": {}},
    }
    monkeypatch.setattr(
        "trading_app.lifecycle_state.read_lifecycle_state",
        lambda profile_id=None: mock_lifecycle,
    )
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=_build_portfolio(),
        profile_id="topstep_50k_mnq_auto",
    )

    result = rls._check_survival_report(ctx)

    assert result.passed is False
    assert "Criterion 11" in result.message


def test_sr_state_check_blocks_missing_state(monkeypatch):
    """Criterion 12 missing state must fail closed for profile demo/live paths."""
    mock_lifecycle = {
        "criterion11": {"gate_ok": True, "operational_pass_probability": 0.8},
        "criterion12": {"available": False, "valid": False, "reason": "missing", "counts": {}},
    }
    monkeypatch.setattr(
        "trading_app.lifecycle_state.read_lifecycle_state",
        lambda profile_id=None: mock_lifecycle,
    )
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=_build_portfolio(),
        profile_id="topstep_50k_mnq_auto",
    )

    result = rls._check_sr_state(ctx)

    assert result.passed is False
    assert "Criterion 12 SR state missing" in result.message


def test_capital_state_checks_skip_signal_only(monkeypatch):
    """Signal-only stays open because it is the path that accumulates evidence."""

    def _raise(*_args, **_kwargs):
        raise AssertionError("signal-only should not read lifecycle state")

    monkeypatch.setattr("trading_app.lifecycle_state.read_lifecycle_state", _raise)
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=_build_portfolio(),
        profile_id="topstep_50k_mnq_auto",
        signal_only=True,
    )

    assert rls._check_survival_report(ctx).passed is True
    assert rls._check_sr_state(ctx).passed is True


def test_capital_state_checks_fail_closed_when_routing_without_profile(monkeypatch):
    """baf99cfe H2 defect regression: a --demo/--live session with no --profile
    routes orders (signal_only=False) but C11/C12 lifecycle state is profile-keyed.

    Pre-fix both gates SKIPPED (passed=True) with "no profile - raw-baseline path",
    letting an order-routing session pass with zero survival/SR evidence. They
    MUST now FAIL-CLOSED. read_lifecycle_state must never be reached (there is no
    profile to key on) — patch it to raise so any read is a test failure.
    """

    def _raise(*_args, **_kwargs):
        raise AssertionError("must not read lifecycle state when profile_id is None")

    monkeypatch.setattr("trading_app.lifecycle_state.read_lifecycle_state", _raise)
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,  # order routing active (signal_only defaults False)
        portfolio=_build_portfolio(),
        profile_id=None,
    )

    survival = rls._check_survival_report(ctx)
    sr_state = rls._check_sr_state(ctx)

    assert survival.passed is False, "C11 must fail-closed when routing without a profile"
    assert "--profile" in survival.message
    assert "Criterion 11" in survival.message
    assert sr_state.passed is False, "C12 must fail-closed when routing without a profile"
    assert "--profile" in sr_state.message
    assert "Criterion 12" in sr_state.message


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
        "_check_survival_report",
        "_check_sr_state",
        "_check_daily_features",
        "_check_contracts",
        "_check_notifications",
        "_check_trade_journal",
        "_check_repo_drift_for_live",
        "_check_telemetry_maturity",
        "_check_live_readiness_report",
        "_check_copy_trading_accounts",
        "_check_shadow_copy_loss_protection",
    ]


def test_no_hardcoded_checks_total_constant():
    """The literal `checks_total = 6` MUST be gone. Source-grep is cheap
    and surfaces accidental reintroduction during a future merge."""
    src = (ROOT / "scripts" / "run_live_session.py").read_text(encoding="utf-8")
    assert "checks_total = 6" not in src
    assert "checks_total = len(PREFLIGHT_CHECKS)" in src


# ---------- A.6.5: Copy-trading account-resolution check ----------


def _make_copy_trading_ctx(profile_id, requested_account_id, all_accounts=None, requested_copies=0):
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
        requested_copies=requested_copies,
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


def test_copy_trading_check_honors_requested_single_copy(monkeypatch):
    """--copies 1 overrides profile.copies for the first live pilot."""
    fake_profile = SimpleNamespace(copies=2)
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"multi_copy_profile": fake_profile},
    )
    ctx = _make_copy_trading_ctx(
        profile_id="multi_copy_profile",
        requested_account_id=None,
        all_accounts=[(21944866, "PA-XFA-001"), (21944867, "PA-XFA-002")],
        requested_copies=1,
    )
    result = rls._check_copy_trading_accounts(ctx)
    assert result.passed is True
    assert "SKIPPED" in result.message
    assert "copies=1" in result.message


def test_copy_trading_check_skipped_when_signal_only(monkeypatch):
    """copies=2, signal_only=True → SKIPPED (no broker account-resolution call).

    When --preflight --signal-only is used, the live-start path skips all
    copy-trading account resolution, so the preflight check must match.
    """
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
    ctx.signal_only = True
    result = rls._check_copy_trading_accounts(ctx)
    assert result.passed is True
    assert "SKIPPED" in result.message
    assert "signal-only" in result.message


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


def test_shadow_copy_loss_protection_blocks_live_multi_copy(monkeypatch):
    fake_profile = SimpleNamespace(copies=2)
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"multi_copy_profile": fake_profile},
    )
    ctx = _make_copy_trading_ctx(profile_id="multi_copy_profile", requested_account_id=None)
    ctx.demo = False
    result = rls._check_shadow_copy_loss_protection(ctx)
    assert result.passed is False
    assert "SHADOW-MLL" in result.message
    assert "copies=2" in result.message


def test_shadow_copy_loss_protection_allows_live_single_copy_override(monkeypatch):
    fake_profile = SimpleNamespace(copies=2)
    monkeypatch.setattr(
        "trading_app.prop_profiles.ACCOUNT_PROFILES",
        {"multi_copy_profile": fake_profile},
    )
    ctx = _make_copy_trading_ctx(
        profile_id="multi_copy_profile",
        requested_account_id=None,
        requested_copies=1,
    )
    ctx.demo = False
    result = rls._check_shadow_copy_loss_protection(ctx)
    assert result.passed is True
    assert "copies=1" in result.message


def test_live_readiness_report_receives_single_copy_override(monkeypatch):
    captured = {}

    def _fake_report(**kwargs):
        captured.update(kwargs)
        return {"strict_zero_warn": {"green": True, "blockers": []}}

    monkeypatch.setattr("scripts.tools.live_readiness_report.build_live_readiness_report", _fake_report)
    ctx = _make_copy_trading_ctx(
        profile_id="topstep_50k_mnq_auto",
        requested_account_id=None,
        requested_copies=1,
    )
    ctx.demo = False

    result = rls._check_live_readiness_report(ctx)

    assert result.passed is True
    assert captured["profile_id"] == "topstep_50k_mnq_auto"
    assert captured["effective_copies"] == 1


def test_repo_drift_gate_blocks_dirty_live_repo(monkeypatch):
    monkeypatch.setattr(
        rls.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="## main...origin/main\n M scripts/run_live_session.py\n",
            stderr="",
        ),
    )
    ctx = rls.PreflightContext(instrument="MNQ", broker_name="topstep", demo=False, portfolio=None)
    result = rls._check_repo_drift_for_live(ctx)
    assert result.passed is False
    assert "repo dirty" in result.message


def test_dashboard_auto_launch_disabled_for_dashboard_origin(monkeypatch):
    monkeypatch.setenv("CANOMPX3_DASHBOARD_ORIGIN", "1")
    assert rls._should_launch_dashboard() is False


def test_dashboard_auto_launch_enabled_for_cli_origin(monkeypatch):
    monkeypatch.delenv("CANOMPX3_DASHBOARD_ORIGIN", raising=False)
    assert rls._should_launch_dashboard() is True


# ---------- 2026-05-16: bracket + fill-poller probes (real, not stubbed) ----------
#
# Background: preflight historically hardcoded results["brackets"] = True and
# results["fill_poller"] = True. That produced a misleading "7/7 PASS" line
# even when the bracket/fill-poller paths would have failed at live-start.
# These tests lock in the new behaviour: the probes exercise the real router
# class and surface PASS/FAIL in the summary so an operator sees the gap
# before clicking Start Live.


class _BracketSupportingRouter:
    """Router stub that advertises bracket support and returns a spec."""

    def __init__(self, account_id, auth, **_kw):
        self.account_id = account_id
        self.auth = auth

    def supports_native_brackets(self) -> bool:
        return True

    def build_bracket_spec(self, **_kw) -> dict:
        return {"stop": 1, "target": 2}

    def query_order_status(self, _order_id):
        # Emulate "endpoint exists, returned validation error for sentinel" —
        # mirrors ProjectX returning 404/422 for order_id=0.
        raise RuntimeError("404: order not found (expected for sentinel order_id=0)")


class _BrokenBracketRouter(_BracketSupportingRouter):
    def build_bracket_spec(self, **_kw):
        return None  # broker advertised support but cannot build — production verifier flags this


class _RaisingBracketRouter(_BracketSupportingRouter):
    def build_bracket_spec(self, **_kw):
        raise RuntimeError("auth not loaded for bracket build")


class _NoPollRouter(_BracketSupportingRouter):
    def query_order_status(self, _order_id):
        raise NotImplementedError("broker does not support order status polling")


class _BracketUnsupportedRouter(_BracketSupportingRouter):
    """Broker does not advertise bracket support — probe must return True
    (matches production `_verify_brackets` `if not supports_native_brackets`
    branch)."""

    def supports_native_brackets(self) -> bool:
        return False


def _components_with_router(router_cls):
    """Build the same shape `create_broker_components` returns: dict with
    `auth` (instance) and `router_class` (class)."""
    auth = SimpleNamespace(get_token=lambda: "tk_probe1234567890")
    return {
        "auth": auth,
        "router_class": router_cls,
        "feed_class": None,
        "contracts_class": SimpleNamespace,
        "positions_class": SimpleNamespace,
    }


def test_probe_brackets_passes_when_spec_is_built():
    """Working bracket spec → PASS. The happy path that prior to this fix was
    silently rubber-stamped True regardless of what the broker would do."""
    components = _components_with_router(_BracketSupportingRouter)
    assert rls._probe_brackets(components) is True


def test_probe_brackets_passes_when_broker_has_no_native_brackets():
    """Production `_verify_brackets` returns True when the broker advertises
    no bracket support (signal-only / Tradovate-without-brackets case). The
    probe must mirror that contract."""
    components = _components_with_router(_BracketUnsupportedRouter)
    assert rls._probe_brackets(components) is True


def test_probe_brackets_fails_when_spec_is_none():
    """Broker advertised support but `build_bracket_spec` returned None —
    production verifier flags this as a FAIL because the entry would land
    without crash protection. Preflight must surface it the same way."""
    components = _components_with_router(_BrokenBracketRouter)
    assert rls._probe_brackets(components) is False


def test_probe_brackets_fails_when_build_raises():
    """Any exception during the bracket build is fail-closed."""
    components = _components_with_router(_RaisingBracketRouter)
    assert rls._probe_brackets(components) is False


def test_probe_brackets_fails_when_components_none():
    """No broker components (auth failed upstream) → bracket probe must
    surface FAIL, not silently pass."""
    assert rls._probe_brackets(None) is False


def test_probe_fill_poller_passes_on_endpoint_error():
    """Production `_verify_fill_poller` treats any non-NotImplementedError as
    "endpoint exists, returned an expected error for sentinel order_id=0".
    The probe must mirror that contract."""
    components = _components_with_router(_BracketSupportingRouter)
    assert rls._probe_fill_poller(components) is True


def test_probe_fill_poller_fails_on_not_implemented():
    """`NotImplementedError` is the ONLY signal that the broker genuinely
    cannot poll for fills. Production sets `_poller_active = False` and
    raises; the probe surfaces this as FAIL so the operator sees it."""
    components = _components_with_router(_NoPollRouter)
    assert rls._probe_fill_poller(components) is False


def test_probe_fill_poller_fails_when_components_none():
    assert rls._probe_fill_poller(None) is False


def test_self_tests_threads_components_into_probes(monkeypatch):
    """The new contract: `_check_notifications` MUST forward `ctx.components`
    into `_run_lightweight_component_self_tests` so the probes hit a real
    router. Regression guard against future refactors dropping the kwarg."""
    captured = {}

    def _spy(*, instrument, components=None):
        captured["instrument"] = instrument
        captured["components"] = components
        return {"notifications": True, "brackets": True, "fill_poller": True}

    monkeypatch.setattr(rls, "_run_lightweight_component_self_tests", _spy)
    components = _components_with_router(_BracketSupportingRouter)
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=None,
        components=components,
    )
    result = rls._check_notifications(ctx)
    assert result.passed is True
    assert captured["instrument"] == "MNQ"
    assert captured["components"] is components, "components dict must be forwarded by reference"


def test_check_notifications_summary_surfaces_per_component_status(monkeypatch):
    """The inline summary string must spell out every probe (PASS or FAIL),
    not just lump them under "WARNINGS". This is the operator-visibility
    contract that motivated the fix."""
    monkeypatch.setattr(
        rls,
        "_run_lightweight_component_self_tests",
        lambda *, instrument, components=None: {
            "notifications": True,
            "brackets": False,
            "fill_poller": False,
        },
    )
    components = _components_with_router(_BracketSupportingRouter)
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=None,
        components=components,
    )
    result = rls._check_notifications(ctx)
    # Preflight is not blocking — the live SessionOrchestrator re-runs the
    # verifiers and that path is authoritative. But the failure MUST be visible.
    assert "brackets:FAIL" in result.message
    assert "fill_poller:FAIL" in result.message
    assert "notifications:PASS" in result.message


def test_check_notifications_summary_lists_all_pass_components(monkeypatch):
    """Even on the happy path, every component is listed by name with PASS so
    the operator can scan the line and confirm each subsystem was exercised."""
    monkeypatch.setattr(
        rls,
        "_run_lightweight_component_self_tests",
        lambda *, instrument, components=None: {
            "notifications": True,
            "brackets": True,
            "fill_poller": True,
        },
    )
    components = _components_with_router(_BracketSupportingRouter)
    ctx = rls.PreflightContext(
        instrument="MNQ",
        broker_name="topstep",
        demo=True,
        portfolio=None,
        components=components,
    )
    result = rls._check_notifications(ctx)
    assert result.passed is True
    assert "brackets:PASS" in result.message
    assert "fill_poller:PASS" in result.message
    assert "notifications:PASS" in result.message


def test_no_hardcoded_self_test_stubs():
    """Source-level grep — the literal `results["brackets"] = True` /
    `results["fill_poller"] = True` lines MUST be gone. Surfaces accidental
    reintroduction during a future merge."""
    src = (ROOT / "scripts" / "run_live_session.py").read_text(encoding="utf-8")
    assert 'results["brackets"] = True' not in src, "preflight is rubber-stamping bracket self-test results again"
    assert 'results["fill_poller"] = True' not in src, (
        "preflight is rubber-stamping fill-poller self-test results again"
    )
    # And the new probe functions must exist.
    assert "_probe_brackets" in src
    assert "_probe_fill_poller" in src
