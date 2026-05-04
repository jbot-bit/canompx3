"""Tests for ``check_e2_lookahead_research_contamination``.

Class bug catalogued 2026-04-28: 18 of 73 E2-using research scripts use
``rel_vol`` / ``break_bar_volume`` / ``break_bar_continues`` / ``break_delay_min``
as predictors. Real-data verification: ~41% of E2 trades have
``entry_ts < break_ts`` so break-bar features are post-entry on that subset.

Registry: ``docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md``.

These tests inject synthetic research files and assert the drift check fires
(or does not fire) under the canonical policy.
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


def test_fires_on_unannotated_e2_with_rel_vol(fake_research_root: Path) -> None:
    """E2 + canonical rel_vol_<SESSION> predictor without policy annotation fires."""
    bad = fake_research_root / "scan_e2_rel_vol.py"
    bad.write_text(
        '"""scan."""\n'
        "import duckdb\n"
        "params = {'entry_model': 'E2'}\n"
        "sql = 'SELECT pnl_r, rel_vol_NYSE_OPEN FROM daily_features ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert any("scan_e2_rel_vol.py" in v for v in violations), violations


def test_does_not_fire_on_clean_replacement_rel_vol_session_norm(
    fake_research_root: Path,
) -> None:
    """Script-local clean-replacement names like `rel_vol_session_norm` do not fire."""
    clean = fake_research_root / "scan_e2_clean_rel_vol.py"
    clean.write_text(
        '"""scan with clean replacement feature."""\n'
        "params = {'entry_model': 'E2'}\n"
        "feature = 'rel_vol_session_norm'  # session-relative, not break-bar\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("scan_e2_clean_rel_vol.py" in v for v in violations), violations


def test_passes_with_cleared_marker(fake_research_root: Path) -> None:
    good = fake_research_root / "scan_e2_cleared.py"
    good.write_text(
        '"""scan."""\n'
        "# e2-lookahead-policy: cleared\n"
        "params = {'entry_model': 'E2'}\n"
        "sql = 'SELECT rel_vol FROM ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("scan_e2_cleared.py" in v for v in violations), violations


def test_passes_with_late_fill_only_marker(fake_research_root: Path) -> None:
    good = fake_research_root / "scan_e2_late_fill.py"
    good.write_text(
        '"""scan."""\n'
        "# e2-lookahead-policy: late-fill-only — filters entry_ts >= break_ts\n"
        "params = {'entry_model': 'E2'}\n"
        "sql = 'SELECT break_bar_volume FROM ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("scan_e2_late_fill.py" in v for v in violations), violations


def test_passes_with_not_predictor_marker(fake_research_root: Path) -> None:
    good = fake_research_root / "scan_e2_not_predictor.py"
    good.write_text(
        '"""scan."""\n'
        "# e2-lookahead-policy: not-predictor — break_delay_min is window-sizing only\n"
        "params = {'entry_model': 'E2'}\n"
        "MAX_DELAY = 'break_delay_min'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("scan_e2_not_predictor.py" in v for v in violations), violations


def test_grandfathered_registry_script_does_not_fire(fake_research_root: Path) -> None:
    """A script in the 2026-04-28 registry allow-list is not double-flagged."""
    grandfathered = fake_research_root / "rel_vol_mechanism_decomposition.py"
    grandfathered.write_text(
        "\"\"\"tainted, no annotation.\"\"\"\nparams = {'entry_model': 'E2'}\nsql = 'SELECT rel_vol FROM ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("rel_vol_mechanism_decomposition.py" in v for v in violations), violations


def test_does_not_fire_on_e2_without_tainted_feature(fake_research_root: Path) -> None:
    """E2 with safe-list features (atr_20, garch_forecast_vol_pct) does not fire."""
    safe = fake_research_root / "scan_e2_safe.py"
    safe.write_text(
        '"""scan."""\n'
        "params = {'entry_model': 'E2'}\n"
        "sql = 'SELECT pnl_r, atr_20_pct, garch_forecast_vol_pct FROM ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("scan_e2_safe.py" in v for v in violations), violations


def test_does_not_fire_on_non_e2_with_rel_vol(fake_research_root: Path) -> None:
    """E1/E3 entry_model with rel_vol does not fire (contamination is E2-specific)."""
    e1 = fake_research_root / "scan_e1_rel_vol.py"
    e1.write_text(
        "\"\"\"scan.\"\"\"\nparams = {'entry_model': 'E1'}\nsql = 'SELECT pnl_r, rel_vol FROM ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("scan_e1_rel_vol.py" in v for v in violations), violations


def test_archive_subdir_is_skipped(fake_research_root: Path) -> None:
    """research/archive/ is frozen and exempt from the check."""
    archive = fake_research_root / "archive"
    archive.mkdir()
    legacy = archive / "legacy_e2_rel_vol.py"
    legacy.write_text(
        "\"\"\"legacy.\"\"\"\nparams = {'entry_model': 'E2'}\nsql = 'SELECT rel_vol FROM ...'\n",
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert not any("legacy_e2_rel_vol.py" in v for v in violations), violations


def test_double_quote_e2_literal_also_fires(fake_research_root: Path) -> None:
    """The check matches both single- and double-quoted E2 literals."""
    bad = fake_research_root / "scan_dq_e2.py"
    bad.write_text(
        '"""scan."""\nparams = {"entry_model": "E2"}\nsql = \'SELECT rel_vol_NYSE_OPEN FROM daily_features ...\'\n',
        encoding="utf-8",
    )
    violations = check_drift.check_e2_lookahead_research_contamination()
    assert any("scan_dq_e2.py" in v for v in violations), violations
