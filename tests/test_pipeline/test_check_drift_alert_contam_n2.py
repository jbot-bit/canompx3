"""Injection tests for ``check_test_writes_to_production_runtime_paths`` (iter 203).

ALERT-CONTAM-N2 class (n=2 incident 2026-05-19): test code that references
canonical production runtime paths as string literals bypasses the
monkeypatch redirect fixture and writes to live operator surfaces.

Per ``memory/feedback_n3_same_class_doctrine_threshold.md`` the n=2 occurrence
promotes this to a mechanical drift check. These tests are the mutation-proof
companion (integrity-guardian.md § 7 — never trust a check without injection).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.check_drift import check_test_writes_to_production_runtime_paths


@pytest.fixture
def fake_tests_root(tmp_path: Path) -> Path:
    """Minimal tests/ tree used to isolate the check from the real test suite."""
    root = tmp_path / "tests"
    root.mkdir()
    return root


def test_clean_tree_passes(fake_tests_root: Path) -> None:
    """Test tree with no canonical path literals → zero violations."""
    (fake_tests_root / "test_something.py").write_text(
        "import pytest\n"
        "def test_example(tmp_path, monkeypatch):\n"
        "    monkeypatch.setattr('module.ALERTS_PATH', tmp_path / 'alerts.jsonl')\n",
        encoding="utf-8",
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert violations == [], violations


def test_alerts_path_literal_fails(fake_tests_root: Path) -> None:
    """Mutation probe: literal operator_alerts.jsonl path in a test triggers the check."""
    bad = fake_tests_root / "test_bad_alerts.py"
    bad.write_text(
        "from pathlib import Path\n"
        "ALERTS = Path('data/runtime/operator_alerts.jsonl')\n"
        "def test_something(): pass\n",
        encoding="utf-8",
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert len(violations) == 1, violations
    assert "test_bad_alerts.py" in violations[0]
    assert "data/runtime/operator_alerts.jsonl" in violations[0]
    assert "ALERT-CONTAM-N2" in violations[0]


def test_state_file_literal_fails(fake_tests_root: Path) -> None:
    """Mutation probe: literal data/bot_state.json path in a test triggers the check."""
    bad = fake_tests_root / "test_bad_state.py"
    bad.write_text(
        "STATE = 'data/bot_state.json'\n"
        "def test_something(): pass\n",
        encoding="utf-8",
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert len(violations) == 1, violations
    assert "test_bad_state.py" in violations[0]
    assert "data/bot_state.json" in violations[0]


def test_live_health_literal_fails(fake_tests_root: Path) -> None:
    """Mutation probe: literal runtime/state/live_health.json path triggers the check."""
    bad = fake_tests_root / "test_bad_health.py"
    bad.write_text(
        "HEALTH = 'runtime/state/live_health.json'\n"
        "def test_something(): pass\n",
        encoding="utf-8",
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert len(violations) == 1, violations
    assert "test_bad_health.py" in violations[0]
    assert "runtime/state/live_health.json" in violations[0]


def test_multiple_violations_reported(fake_tests_root: Path) -> None:
    """Multiple offending files each get one violation row."""
    (fake_tests_root / "test_a.py").write_text(
        "PATH = 'data/runtime/operator_alerts.jsonl'\n", encoding="utf-8"
    )
    (fake_tests_root / "test_b.py").write_text(
        "PATH = 'data/bot_state.json'\n", encoding="utf-8"
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert len(violations) == 2, violations


def test_only_one_violation_per_file(fake_tests_root: Path) -> None:
    """A file referencing multiple forbidden literals gets only one violation (break after first match)."""
    bad = fake_tests_root / "test_multi.py"
    bad.write_text(
        "A = 'data/runtime/operator_alerts.jsonl'\n"
        "B = 'data/bot_state.json'\n"
        "C = 'runtime/state/live_health.json'\n",
        encoding="utf-8",
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert len(violations) == 1, violations


def test_nonexistent_tests_root_returns_empty(tmp_path: Path) -> None:
    """Missing tests root is a valid state (fresh tree) — returns empty, never raises."""
    missing = tmp_path / "does_not_exist"
    violations = check_test_writes_to_production_runtime_paths(missing)
    assert violations == []


def test_nested_test_subdirectory_is_scanned(fake_tests_root: Path) -> None:
    """rglob covers nested test subdirectories, not just the top level."""
    subdir = fake_tests_root / "test_trading_app"
    subdir.mkdir()
    bad = subdir / "test_new_file.py"
    bad.write_text(
        "PATH = 'data/runtime/operator_alerts.jsonl'\n", encoding="utf-8"
    )
    violations = check_test_writes_to_production_runtime_paths(fake_tests_root)
    assert len(violations) == 1, violations
    assert "test_new_file.py" in violations[0]
