"""Tests for ``check_research_scratch_policy_annotation``.

Class bug discovered 2026-04-27: research scripts using
``WHERE pnl_r IS NOT NULL`` silently drop scratch rows whose
``pnl_r`` is NULL, inflating measured ExpR by 10-45% on survivor lanes.

These tests inject synthetic research files and assert the drift check
fires (or does not fire) under the canonical policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import check_drift


@pytest.fixture
def fake_research_root(tmp_path: Path, monkeypatch) -> Path:
    """Point check_drift at a temp directory with research/ subtree."""
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    return research_dir


def test_fires_on_unannotated_pnl_r_filter(fake_research_root: Path) -> None:
    """A research script using `pnl_r IS NOT NULL` without policy marker fires."""
    bad = fake_research_root / "scan_unannotated.py"
    bad.write_text(
        '"""scan."""\nimport duckdb\nsql = \'SELECT pnl_r FROM orb_outcomes WHERE pnl_r IS NOT NULL\'\n',
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert any("scan_unannotated.py" in v for v in violations), violations


def test_passes_with_drop_marker(fake_research_root: Path) -> None:
    """Annotation `# scratch-policy: drop` is accepted."""
    good = fake_research_root / "scan_drop.py"
    good.write_text(
        '"""scan with explicit drop policy."""\n'
        "# scratch-policy: drop\n"
        "sql = 'SELECT pnl_r FROM orb_outcomes WHERE pnl_r IS NOT NULL'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert all("scan_drop.py" not in v for v in violations), violations


def test_passes_with_include_as_zero_marker(fake_research_root: Path) -> None:
    good = fake_research_root / "scan_zero.py"
    good.write_text(
        "# scratch-policy: include-as-zero\nsql = 'WHERE pnl_r IS NOT NULL'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert all("scan_zero.py" not in v for v in violations), violations


def test_passes_with_realized_eod_marker(fake_research_root: Path) -> None:
    good = fake_research_root / "scan_eod.py"
    good.write_text(
        "# scratch-policy: realized-eod  (post Stage 5 fix; outcome_builder populates pnl_r)\n"
        "sql = 'WHERE pnl_r IS NOT NULL'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert all("scan_eod.py" not in v for v in violations), violations


def test_passes_when_needle_absent(fake_research_root: Path) -> None:
    """A research script that never uses `pnl_r IS NOT NULL` is fine."""
    clean = fake_research_root / "no_filter.py"
    clean.write_text(
        "sql = 'SELECT pnl_r FROM orb_outcomes WHERE outcome IN (\\'win\\',\\'loss\\')'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert all("no_filter.py" not in v for v in violations), violations


def test_archive_subtree_is_skipped(fake_research_root: Path) -> None:
    """Frozen scans under research/archive/ are exempt (Backtesting Rule 11 audit trail)."""
    archive = fake_research_root / "archive"
    archive.mkdir()
    frozen = archive / "old_scan.py"
    frozen.write_text(
        "sql = 'WHERE pnl_r IS NOT NULL'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert all("old_scan.py" not in v for v in violations), violations


def test_invalid_marker_value_still_fires(fake_research_root: Path) -> None:
    """A `# scratch-policy: bogus` value is not a valid marker."""
    bad = fake_research_root / "bogus_marker.py"
    bad.write_text(
        "# scratch-policy: bogus\nsql = 'WHERE pnl_r IS NOT NULL'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert any("bogus_marker.py" in v for v in violations), violations


def test_case_insensitive_marker(fake_research_root: Path) -> None:
    """Marker matching is case-insensitive."""
    good = fake_research_root / "upper.py"
    good.write_text(
        "# Scratch-Policy: Drop\nsql = 'WHERE pnl_r IS NOT NULL'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_research_scratch_policy_annotation()
    assert all("upper.py" not in v for v in violations), violations
