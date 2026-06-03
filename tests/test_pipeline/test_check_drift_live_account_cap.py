"""Live-account-cap drift guard tests (MFFU Layer C, Option A).

Operator 2026-05-31: prop plans force Evaluation -> Sim Funded -> Live Funded and
the live stage caps accounts. The cap is a firm operational FACT, modeled as DATA
in PropFirmSpec.firm_specific_rules['max_live_accounts'] (NOT behavioral logic).
check_live_funded_firms_declare_max_live_accounts asserts that every firm backing
a live_funded PayoutPolicy declares that cap, so it can never sit as undocumented
prose (a silent-failure / lying-field class per institutional-rigor.md §§ 5-6).

These tests prove the guard:
  1. passes clean against the real repo state (Topstep now declares it),
  2. fails when the backing spec omits max_live_accounts,
  3. fails loud (not silent-skip) on a firm-name mismatch,
  4. fails loud if the live_funded universe vanishes (layout change),
  5. passes when firm_specific_rules declares the cap.

The guard imports the live PAYOUT_POLICIES / PROP_FIRM_SPECS registries inside the
function body, so injection tests monkeypatch those dicts on their SOURCE modules
(where the local import resolves at call time).
"""

from __future__ import annotations

from dataclasses import replace

import trading_app.prop_firm_policies as pfp
import trading_app.prop_profiles as pp
from pipeline.check_drift import check_live_funded_firms_declare_max_live_accounts


def test_guard_passes_on_real_repo_state():
    """Real repo: Topstep + MFFU live caps are declared as data."""
    violations = check_live_funded_firms_declare_max_live_accounts()
    assert violations == [], "live-account-cap guard must pass clean: " + "; ".join(violations)


def test_guard_fails_when_backing_spec_omits_cap(monkeypatch):
    """Teeth: a live_funded policy whose spec has no max_live_accounts is drift."""
    real_spec = pp.PROP_FIRM_SPECS["topstep"]
    # Strip firm_specific_rules from the topstep spec (frozen dataclass -> replace()).
    stripped = replace(real_spec, firm_specific_rules=None)
    monkeypatch.setitem(pp.PROP_FIRM_SPECS, "topstep", stripped)

    violations = check_live_funded_firms_declare_max_live_accounts()
    assert violations, "spec missing max_live_accounts must produce a violation"
    assert any("max_live_accounts" in v and "topstep" in v for v in violations)


def test_guard_fails_when_cap_declared_none(monkeypatch):
    """A firm_specific_rules dict that exists but sets max_live_accounts=None still fails."""
    real_spec = pp.PROP_FIRM_SPECS["topstep"]
    nulled = replace(real_spec, firm_specific_rules={"max_live_accounts": None})
    monkeypatch.setitem(pp.PROP_FIRM_SPECS, "topstep", nulled)

    violations = check_live_funded_firms_declare_max_live_accounts()
    assert any("max_live_accounts" in v for v in violations)


def test_guard_fails_loud_on_firm_name_mismatch(monkeypatch):
    """Fail-closed: a live_funded policy whose firm has no PROP_FIRM_SPECS entry."""
    real_policy = pfp.PAYOUT_POLICIES["topstep_live_funded"]
    orphan = replace(real_policy, firm="ghost_firm_no_spec")
    monkeypatch.setitem(pfp.PAYOUT_POLICIES, "topstep_live_funded", orphan)

    violations = check_live_funded_firms_declare_max_live_accounts()
    assert any("no matching PROP_FIRM_SPECS entry" in v for v in violations)


def test_guard_fails_loud_when_no_live_funded_policy(monkeypatch):
    """Layout change: if no live_funded policy is modeled, say so rather than pass."""
    sim_only = {k: v for k, v in pfp.PAYOUT_POLICIES.items() if v.stage != "live_funded"}
    monkeypatch.setattr(pfp, "PAYOUT_POLICIES", sim_only)

    violations = check_live_funded_firms_declare_max_live_accounts()
    assert any("no PayoutPolicy with stage='live_funded'" in v for v in violations)


def test_guard_passes_when_cap_declared(monkeypatch):
    """A well-formed spec with max_live_accounts declared passes."""
    real_spec = pp.PROP_FIRM_SPECS["topstep"]
    good = replace(real_spec, firm_specific_rules={"max_live_accounts": 1})
    monkeypatch.setitem(pp.PROP_FIRM_SPECS, "topstep", good)

    violations = check_live_funded_firms_declare_max_live_accounts()
    assert violations == []
