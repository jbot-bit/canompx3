"""Tests for ``check_research_scans_call_guards`` (canary harness deliverable #3).

The meta static-scanner flags research scans that read a canonical layer
(``orb_outcomes`` / ``daily_features``) AND apply a filter or use the E2 entry
model BUT never reference a canonical guard — such a scan could pass a fake edge
because the guard is never invoked.

Mirrors ``tests/test_pipeline/test_check_drift_e2_lookahead.py``: inject
synthetic research files under a temp ``research/`` subtree and assert the check
fires (or not) under the documented policy. ``PROJECT_ROOT`` is monkeypatched so
the whole-tree scan targets the temp dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import check_drift
from scripts.tools import canary_guard_coverage


@pytest.fixture
def fake_research_root(tmp_path: Path, monkeypatch) -> Path:
    """Point check_drift at a temp dir with a research/ subtree."""
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    return research_dir


def test_fires_on_filter_scan_without_guard(fake_research_root: Path) -> None:
    """Reads orb_outcomes + filter_type, no guard → flagged."""
    bad = fake_research_root / "scan_filter_no_guard.py"
    bad.write_text(
        '"""scan."""\n'
        "import duckdb\n"
        "sql = 'SELECT pnl_r FROM orb_outcomes WHERE symbol = ?'\n"
        "filter_type = 'ORB_G5'  # applies a filter inline, never delegates\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert any("scan_filter_no_guard.py" in v for v in violations), violations


def test_fires_on_e2_scan_without_guard(fake_research_root: Path) -> None:
    """Reads daily_features + entry_model='E2' literal, no guard → flagged."""
    bad = fake_research_root / "scan_e2_no_guard.py"
    bad.write_text(
        '"""scan."""\n'
        "sql = 'SELECT pnl_r FROM daily_features JOIN orb_outcomes USING(trading_day)'\n"
        "params = {'entry_model': 'E2'}\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert any("scan_e2_no_guard.py" in v for v in violations), violations


def test_does_not_fire_when_guard_referenced(fake_research_root: Path) -> None:
    """Compliant scan: same shape but delegates to filter_signal → not flagged."""
    good = fake_research_root / "scan_compliant.py"
    good.write_text(
        '"""scan."""\n'
        "from research.filter_utils import filter_signal\n"
        "sql = 'SELECT pnl_r FROM orb_outcomes'\n"
        "filter_type = 'ORB_G5'\n"
        "sig = filter_signal(df, filter_type, orb_label='NYSE_OPEN')\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("scan_compliant.py" in v for v in violations), violations


def test_does_not_fire_when_session_guard_imported(fake_research_root: Path) -> None:
    """session_guard reference is also a valid delegation → not flagged."""
    good = fake_research_root / "scan_session_guard.py"
    good.write_text(
        '"""scan."""\n'
        "from pipeline.session_guard import is_feature_safe\n"
        "sql = 'SELECT pnl_r FROM daily_features'\n"
        "filter_type = 'OVNRNG_100'\n"
        "ok = is_feature_safe('overnight_range', 'LONDON_METALS')\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("scan_session_guard.py" in v for v in violations), violations


def test_passes_with_cleared_marker(fake_research_root: Path) -> None:
    """The opt-out marker exempts a manually-verified scan."""
    good = fake_research_root / "scan_cleared.py"
    good.write_text(
        '"""scan."""\n'
        "# canary-guard-coverage: cleared\n"
        "sql = 'SELECT pnl_r FROM orb_outcomes'\n"
        "filter_type = 'ORB_G5'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("scan_cleared.py" in v for v in violations), violations


def test_does_not_fire_without_canonical_layer(fake_research_root: Path) -> None:
    """A filtering scan that does NOT read a canonical layer is out of scope."""
    safe = fake_research_root / "scan_no_canonical.py"
    safe.write_text(
        '"""scan reading a derived layer only."""\n'
        "sql = 'SELECT * FROM validated_setups'\n"
        "filter_type = 'ORB_G5'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("scan_no_canonical.py" in v for v in violations), violations


def test_does_not_fire_without_filter_or_e2(fake_research_root: Path) -> None:
    """A canonical-reading scan that applies no filter / no E2 needs no guard."""
    safe = fake_research_root / "scan_base_rates.py"
    safe.write_text(
        '"""base-rate scan, no filter."""\nsql = \'SELECT AVG(pnl_r) FROM orb_outcomes GROUP BY orb_label\'\n',
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("scan_base_rates.py" in v for v in violations), violations


def test_archive_subdir_is_skipped(fake_research_root: Path) -> None:
    """Frozen archive/ scans are not retro-flagged."""
    archive = fake_research_root / "archive"
    archive.mkdir()
    frozen = archive / "old_scan.py"
    frozen.write_text(
        '"""frozen scan."""\nsql = \'SELECT pnl_r FROM orb_outcomes\'\nfilter_type = \'ORB_G5\'\n',
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("old_scan.py" in v for v in violations), violations


def test_scripts_research_dir_is_scanned(tmp_path: Path, monkeypatch) -> None:
    """The scanner covers scripts/research/, not just research/."""
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    sr = tmp_path / "scripts" / "research"
    sr.mkdir(parents=True)
    bad = sr / "scan_in_scripts.py"
    bad.write_text(
        '"""scan."""\nsql = \'SELECT pnl_r FROM orb_outcomes\'\nfilter_type = \'ATR_P50\'\n',
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert any("scan_in_scripts.py" in v for v in violations), violations


def test_filter_token_in_comment_does_not_falsely_flag(fake_research_root: Path) -> None:
    """AST precision: 'E2' in a comment/string only, no real filter → not flagged.

    A base-rate scan that mentions E2 in prose but applies no filter and uses no
    E2 entry_model literal must not be flagged (comments are stripped by the AST;
    no _FILTER_TOKENS present).
    """
    safe = fake_research_root / "scan_e2_prose.py"
    safe.write_text(
        '"""Compares E2 vs E1 base rates conceptually."""\n'
        "# we discuss E2 here but apply no filter\n"
        "sql = 'SELECT AVG(pnl_r) FROM orb_outcomes'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scans_call_guards()
    assert not any("scan_e2_prose.py" in v for v in violations), violations


def test_real_repo_scan_returns_list() -> None:
    """Smoke: the scanner runs against the real repo and returns a list.

    The real-repo run IS deliverable #3 (the list of currently-non-compliant
    scan scripts). We assert it returns a list (advisory; the count is captured
    in the result doc, not pinned here so the test does not rot as scans land).
    """
    flagged = canary_guard_coverage.scan_guard_coverage()
    assert isinstance(flagged, list)
    assert all(isinstance(p, str) for p in flagged)
