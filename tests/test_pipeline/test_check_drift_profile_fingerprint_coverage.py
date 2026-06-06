"""Profile-fingerprint field-coverage drift-guard tests (Fork #2, 2026-06-07).

check_profile_fingerprint_field_coverage pins the CONFIG-VALUE fingerprint:
every AccountProfile field the survival sim reads (via profile.<attr> OR
getattr(profile, "...")) — minus a reviewed label-only allowlist — must be in
build_profile_fingerprint's payload, or an operator could change it (e.g. loosen
self_imposed_dd_dollars) while a cached PASS stayed valid (silent stale-PASS).

The check reads trading_app/account_survival.py + trading_app/derived_state.py
directly, so injection tests point it at a temp root via the project_root param —
no mutation of real source.

Proves the guard:
  1. passes clean against the real repo state,
  2. has teeth: fails when a sim-read field is missing from the fingerprint,
  3. catches the getattr() access form (the form that motivated the fix),
  4. respects the allowlist (label-only field read but absent → no violation),
  5. fails closed on missing module / unparseable / unlocatable payload.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.check_drift import check_profile_fingerprint_field_coverage

# Minimal account_survival.py: reads two survival-verdict fields, one via direct
# attribute access, one via getattr (mirroring the real defensive form).
_SURVIVAL_SRC = """\
def _build(profile):
    budget = getattr(profile, "self_imposed_dd_dollars", None)
    dll = profile.daily_loss_dollars
    label = profile.profile_id  # allowlisted label-only read
    return budget, dll, label
"""

# Minimal derived_state.py: a build_profile_fingerprint with a `payload` dict.
_DERIVED_GOOD = """\
def build_profile_fingerprint(profile):
    payload = {
        "profile_id": profile.profile_id,
        "self_imposed_dd_dollars": profile.self_imposed_dd_dollars,
        "daily_loss_dollars": profile.daily_loss_dollars,
    }
    return payload
"""

# Same, but daily_loss_dollars dropped from the payload — the silent stale-PASS bug.
_DERIVED_MISSING_ATTR_FIELD = """\
def build_profile_fingerprint(profile):
    payload = {
        "profile_id": profile.profile_id,
        "self_imposed_dd_dollars": profile.self_imposed_dd_dollars,
    }
    return payload
"""

# Same, but self_imposed_dd_dollars dropped — the field read via getattr().
_DERIVED_MISSING_GETATTR_FIELD = """\
def build_profile_fingerprint(profile):
    payload = {
        "profile_id": profile.profile_id,
        "daily_loss_dollars": profile.daily_loss_dollars,
    }
    return payload
"""

# No `payload` dict literal — fail-closed (None payload keys).
_DERIVED_NO_PAYLOAD = """\
def build_profile_fingerprint(profile):
    return hash(profile.profile_id)
"""


def _root(tmp_path: Path, *, survival: str = _SURVIVAL_SRC, derived: str = _DERIVED_GOOD) -> Path:
    app = tmp_path / "trading_app"
    app.mkdir()
    (app / "account_survival.py").write_text(survival, encoding="utf-8")
    (app / "derived_state.py").write_text(derived, encoding="utf-8")
    return tmp_path


def test_passes_against_real_repo_state():
    # No project_root → checks the actual repo. Must be clean after the fix.
    assert check_profile_fingerprint_field_coverage() == []


def test_passes_on_well_formed_temp_root(tmp_path):
    assert check_profile_fingerprint_field_coverage(project_root=_root(tmp_path)) == []


def test_teeth_missing_direct_attr_field(tmp_path):
    root = _root(tmp_path, derived=_DERIVED_MISSING_ATTR_FIELD)
    violations = check_profile_fingerprint_field_coverage(project_root=root)
    assert violations, "dropping daily_loss_dollars from the fingerprint must be caught"
    assert "daily_loss_dollars" in violations[0]


def test_teeth_missing_getattr_field(tmp_path):
    # The getattr() access form is the one that motivated the fix — the scan
    # must see it, not just direct attribute access.
    root = _root(tmp_path, derived=_DERIVED_MISSING_GETATTR_FIELD)
    violations = check_profile_fingerprint_field_coverage(project_root=root)
    assert violations, "dropping a getattr-read field from the fingerprint must be caught"
    assert "self_imposed_dd_dollars" in violations[0]


def test_allowlisted_label_field_not_required(tmp_path):
    # profile_id is read but on the allowlist; a fingerprint WITHOUT it must not
    # violate (proven via a survival src that ONLY reads profile_id).
    survival = "def _f(profile):\n    return profile.profile_id\n"
    derived = "def build_profile_fingerprint(profile):\n    payload = {'firm': profile.firm}\n    return payload\n"
    root = _root(tmp_path, survival=survival, derived=derived)
    assert check_profile_fingerprint_field_coverage(project_root=root) == []


def test_fail_closed_missing_module(tmp_path):
    # Only derived_state.py present, account_survival.py missing.
    app = tmp_path / "trading_app"
    app.mkdir()
    (app / "derived_state.py").write_text(_DERIVED_GOOD, encoding="utf-8")
    violations = check_profile_fingerprint_field_coverage(project_root=tmp_path)
    assert violations and "missing module" in violations[0]


def test_fail_closed_unparseable(tmp_path):
    root = _root(tmp_path, survival="def broken(:\n")
    violations = check_profile_fingerprint_field_coverage(project_root=root)
    assert violations and "parse error" in violations[0]


def test_fail_closed_no_payload_literal(tmp_path):
    root = _root(tmp_path, derived=_DERIVED_NO_PAYLOAD)
    violations = check_profile_fingerprint_field_coverage(project_root=root)
    assert violations and "payload" in violations[0]
