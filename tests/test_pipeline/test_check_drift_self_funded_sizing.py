"""Self-funded sizing doctrine — marker drift-guard tests.

HARD DOCTRINE (.claude/rules/self-funded-sizing-doctrine.md, operator 2026-05-31):
prop-firm contract caps apply ONLY to prop survival/rule sims; they must NEVER
bound self_funded (personal-capital) book-building or earning capacity. The drift
check check_prop_caps_do_not_leak_into_self_funded is the marker layer that pins
that intent and fails loud if a self_funded tier is added without the
@margin-guard-not-earnings-cap marker, or if the doctrine file is deleted.

These tests prove the guard:
  1. passes clean against the real repo state,
  2. has teeth (fails on an injected marker-stripped source),
  3. fails loud if the doctrine file is missing,
  4. fails loud if the marker sits AFTER the first self_funded tier (scope hole).

The guard reads source files directly, so the injection tests point it at a
temp copy via PROJECT_ROOT monkeypatch — no mutation of the real source.
"""

from __future__ import annotations

from pathlib import Path

import pipeline.check_drift as cd
from pipeline.check_drift import check_prop_caps_do_not_leak_into_self_funded

# A minimal prop_profiles.py-shaped source: the marker comment introduces the
# self_funded tier block, exactly as the real file does.
_GOOD_SRC = """\
ACCOUNT_TIERS = {
    ("topstep", 50_000): PropFirmAccount("topstep", 50_000, 2_000, 5, 50),
    # @margin-guard-not-earnings-cap
    # doctrine: self_funded caps are a margin/sanity guard, not an earnings ceiling.
    ("self_funded", 2_500): PropFirmAccount("self_funded", 2_500, 375, 0, 1, 125),
    ("self_funded", 50_000): PropFirmAccount("self_funded", 50_000, 10_000, 2, 20, 2_500),
}
"""

_NO_MARKER_SRC = """\
ACCOUNT_TIERS = {
    ("topstep", 50_000): PropFirmAccount("topstep", 50_000, 2_000, 5, 50),
    ("self_funded", 2_500): PropFirmAccount("self_funded", 2_500, 375, 0, 1, 125),
    ("self_funded", 50_000): PropFirmAccount("self_funded", 50_000, 10_000, 2, 20, 2_500),
}
"""

_MARKER_TOO_LATE_SRC = """\
ACCOUNT_TIERS = {
    ("self_funded", 2_500): PropFirmAccount("self_funded", 2_500, 375, 0, 1, 125),
    # @margin-guard-not-earnings-cap
    ("self_funded", 50_000): PropFirmAccount("self_funded", 50_000, 10_000, 2, 20, 2_500),
}
"""


def _fake_root(tmp_path: Path, src: str, *, write_doctrine: bool = True) -> Path:
    """Build a temp PROJECT_ROOT with a prop_profiles.py and (optionally) the doctrine file."""
    (tmp_path / "trading_app").mkdir(parents=True, exist_ok=True)
    (tmp_path / "trading_app" / "prop_profiles.py").write_text(src, encoding="utf-8")
    if write_doctrine:
        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True, exist_ok=True)
        (rules / "self-funded-sizing-doctrine.md").write_text("# doctrine\n", encoding="utf-8")
    return tmp_path


def test_guard_passes_on_real_repo_state():
    """The real repo must satisfy the doctrine guard (marker present + doctrine file present)."""
    violations = check_prop_caps_do_not_leak_into_self_funded()
    assert violations == [], "self-funded sizing guard must pass clean on real state: " + "; ".join(violations)


def test_guard_passes_on_well_formed_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "PROJECT_ROOT", _fake_root(tmp_path, _GOOD_SRC))
    assert check_prop_caps_do_not_leak_into_self_funded() == []


def test_guard_fails_when_marker_absent(tmp_path, monkeypatch):
    """Teeth: a self_funded tier with no margin-guard marker is drift."""
    monkeypatch.setattr(cd, "PROJECT_ROOT", _fake_root(tmp_path, _NO_MARKER_SRC))
    violations = check_prop_caps_do_not_leak_into_self_funded()
    assert violations, "marker-stripped source must produce a violation"
    assert any("margin-guard-not-earnings-cap" in v for v in violations)


def test_guard_fails_when_doctrine_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "PROJECT_ROOT", _fake_root(tmp_path, _GOOD_SRC, write_doctrine=False))
    violations = check_prop_caps_do_not_leak_into_self_funded()
    assert any("self-funded-sizing-doctrine.md is MISSING" in v for v in violations)


def test_guard_fails_when_marker_after_first_tier(tmp_path, monkeypatch):
    """Scope hole: marker must INTRODUCE the block; a marker below the first tier leaves it uncovered."""
    monkeypatch.setattr(cd, "PROJECT_ROOT", _fake_root(tmp_path, _MARKER_TOO_LATE_SRC))
    violations = check_prop_caps_do_not_leak_into_self_funded()
    assert any("AFTER the first" in v for v in violations)


def test_guard_fails_loud_when_no_self_funded_tiers(tmp_path, monkeypatch):
    """If the self_funded layout vanished, fail loud rather than silently pass."""
    src = 'ACCOUNT_TIERS = {\n    ("topstep", 50_000): PropFirmAccount("topstep", 50_000, 2_000, 5, 50),\n}\n'
    monkeypatch.setattr(cd, "PROJECT_ROOT", _fake_root(tmp_path, src))
    violations = check_prop_caps_do_not_leak_into_self_funded()
    assert any("could not locate any" in v for v in violations)
