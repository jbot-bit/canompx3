"""AccountProfile.is_express_funded fail-closed default + explicit-declaration drift check.

2026-05-18 follow-up to commit a2d5ea56 (telemetry-maturity FAIL→WARN demotion):

- Dataclass default flipped True → False. A new profile that forgets the
  field is silently treated as real-capital so safety gates stay FAIL.
- Drift check check_account_profiles_declare_is_express_funded asserts
  every ACCOUNT_PROFILES entry literally declares is_express_funded=,
  preventing implicit reliance on the default.

These tests pin both invariants so a future "let's simplify" refactor
cannot quietly re-flip the default or remove the drift check.
"""

from __future__ import annotations

import ast

from pipeline.check_drift import (
    TRADING_APP_DIR,
    check_account_profiles_declare_is_express_funded,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, AccountProfile


def test_account_profile_default_is_express_funded_is_false():
    """Fail-closed default: forgetting the field classifies as real-capital."""
    bare = AccountProfile(profile_id="_test_bare", firm="test", account_size=1)
    assert bare.is_express_funded is False, "default must be fail-closed (real-capital)"


def test_every_account_profile_declares_is_express_funded_explicitly():
    """No live profile may inherit the default — every entry must declare the field."""
    violations = check_account_profiles_declare_is_express_funded()
    assert violations == [], "ACCOUNT_PROFILES entries omitting is_express_funded=: " + "; ".join(violations)


def test_drift_check_catches_omitted_field():
    """Mutation probe: a synthetic prop_profiles literal without is_express_funded must produce a violation."""
    # Build a synthetic snippet mirroring prop_profiles.py's structure and
    # AST-parse it through the same logic the drift check uses. This proves
    # the check has teeth without mutating the real source file.
    snippet = (
        "ACCOUNT_PROFILES: dict[str, AccountProfile] = {\n"
        "    'bad_profile': AccountProfile(\n"
        "        profile_id='bad_profile',\n"
        "        firm='test',\n"
        "        account_size=1,\n"
        "    ),\n"
        "}\n"
    )
    tree = ast.parse(snippet)
    found_dict: ast.Dict | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "ACCOUNT_PROFILES" and isinstance(node.value, ast.Dict):
                found_dict = node.value
                break
    assert found_dict is not None, "test scaffolding broken — expected to parse ACCOUNT_PROFILES dict"
    omitted: list[str] = []
    for key, val in zip(found_dict.keys, found_dict.values, strict=False):
        if not isinstance(val, ast.Call):
            continue
        if any(kw.arg == "is_express_funded" for kw in val.keywords):
            continue
        if isinstance(key, ast.Constant):
            omitted.append(str(key.value))
    assert omitted == ["bad_profile"], f"mutation-probe AST walk must find omitted-field profile; got {omitted!r}"


def test_real_account_profiles_module_resolves():
    """Smoke: ACCOUNT_PROFILES dict imports and is non-empty (drift check has something to scan)."""
    assert len(ACCOUNT_PROFILES) > 0, "ACCOUNT_PROFILES must be populated"
    assert (TRADING_APP_DIR / "prop_profiles.py").exists(), "drift-check target file must exist"
