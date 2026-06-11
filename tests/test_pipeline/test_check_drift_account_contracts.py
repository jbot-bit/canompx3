"""Known-violation injection tests for check_account_contracts_feasibility (Stage 3a).

The drift check is the static, fast config-feasibility FLOOR for per-account
contract maps. The deep proof (modeled DD at scale) is the per-account C11
survival re-run — this check guards shape + positivity + belt reachability so a
config regression fails loud BEFORE survival is even run.

Strategy: monkeypatch ACCOUNT_PROFILES with a synthetic profile dict so the check
sees a known violation. The check imports ACCOUNT_PROFILES at call time (function
body), so patching the prop_profiles symbol is sufficient and hermetic (no gold.db).
"""

from __future__ import annotations

import pipeline.check_drift as cd
from trading_app.prop_profiles import AccountProfile, get_account_tier


def _profile(**kw) -> AccountProfile:
    base = dict(profile_id="t", firm="topstep", account_size=50000, is_express_funded=True)
    base.update(kw)
    return AccountProfile(**base)


def _patch_profiles(monkeypatch, profiles: dict) -> None:
    # The check does `from trading_app.prop_profiles import ACCOUNT_PROFILES`
    # inside its body, so patching the source module symbol is what it sees.
    import trading_app.prop_profiles as pp

    monkeypatch.setattr(pp, "ACCOUNT_PROFILES", profiles)


def test_clean_when_no_profile_sets_account_contracts():
    # Real profile set: none populate the field at Stage 3a. Check is silent.
    assert cd.check_account_contracts_feasibility() == []


def test_uniform_one_map_is_clean(monkeypatch):
    # contracts == 1 carries no divergent-belt feasibility risk.
    p = _profile(account_contracts=((202, 1), (303, 1)))
    _patch_profiles(monkeypatch, {"t": p})
    assert cd.check_account_contracts_feasibility() == []


def test_scaled_worst_trade_breaching_mll_is_caught(monkeypatch):
    # tier.max_dd for topstep 50k == $2000. 3 contracts × max_risk_per_trade=$1000
    # = $3000 >= MLL → a single scaled worst-trade can breach the broker MLL.
    tier = get_account_tier("topstep", 50000)
    assert tier.max_dd == 2000  # guard the test's own premise
    p = _profile(account_contracts=((202, 3),), max_risk_per_trade=1000.0)
    _patch_profiles(monkeypatch, {"t": p})
    violations = cd.check_account_contracts_feasibility()
    assert any(">= broker MLL" in v for v in violations), violations
    assert any("account 202" in v and "tier.max_dd=$2000" in v for v in violations)


def test_scaled_worst_trade_overshooting_belt_is_caught(monkeypatch):
    # belt $450, max_risk_per_trade $200, 3 contracts → scaled $600 > belt $450,
    # but $600 < MLL $2000 (so ONLY the belt-overshoot fires, not the MLL one).
    p = _profile(
        account_contracts=((202, 3),),
        max_risk_per_trade=200.0,
        daily_loss_dollars=450.0,
    )
    _patch_profiles(monkeypatch, {"t": p})
    violations = cd.check_account_contracts_feasibility()
    assert any("daily_loss_dollars belt" in v and "account 202" in v for v in violations), violations
    # The MLL breach must NOT fire ($600 < $2000).
    assert not any(">= broker MLL" in v for v in violations)


def test_feasible_scaled_map_is_clean(monkeypatch):
    # 3 contracts × $100 = $300 < belt $450 < MLL $2000 → feasible, silent.
    p = _profile(
        account_contracts=((202, 3),),
        max_risk_per_trade=100.0,
        daily_loss_dollars=450.0,
    )
    _patch_profiles(monkeypatch, {"t": p})
    assert cd.check_account_contracts_feasibility() == []


def test_check_is_registered_and_blocking():
    # The check must be wired into the drift registry (CHECKS) as BLOCKING.
    entry = next(
        (t for t in cd.CHECKS if "Per-account contract maps must be shape-valid" in t[0]),
        None,
    )
    assert entry is not None, "check_account_contracts_feasibility not registered in CHECKS"
    warn_only = entry[2]
    assert warn_only is False, "feasibility check must be blocking, not warn-only"
