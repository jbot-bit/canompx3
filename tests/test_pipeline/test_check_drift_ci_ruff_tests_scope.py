"""Regression tests for `check_ci_ruff_lint_covers_tests`.

Pins the 2026-06-08 gate-scope-divergence fix: CI's `ruff check` (lint) must
cover `tests/`, mirroring `ruff format --check`. Without it, a lint defect in a
test file (e.g. a `B017` blind `pytest.raises(Exception)`) stays green on CI but
blocks the next merge that stages it through the path-agnostic pre-commit hook.

Verifies the guard by known-violation injection (institutional-rigor.md sec 11):
a guard that never fails on a real violation is itself a silent failure.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.check_drift import check_ci_ruff_lint_covers_tests


def test_real_ci_yml_passes() -> None:
    """The committed `.github/workflows/*.yml` must satisfy the guard."""
    assert check_ci_ruff_lint_covers_tests() == []


def test_ruff_check_without_tests_is_caught(tmp_path: Path) -> None:
    """A `ruff check` line that omits `tests/` is the exact regression we fixed."""
    wf = tmp_path / "ci.yml"
    wf.write_text(
        "jobs:\n  lint:\n    steps:\n      - run: uv run ruff check pipeline/ trading_app/ scripts/\n",
        encoding="utf-8",
    )
    violations = check_ci_ruff_lint_covers_tests(workflows_dir=tmp_path)
    assert len(violations) == 1
    assert "tests/" in violations[0]


def test_ruff_check_with_tests_passes(tmp_path: Path) -> None:
    """The corrected invocation (with `tests/`) must pass."""
    wf = tmp_path / "ci.yml"
    wf.write_text(
        "jobs:\n  lint:\n    steps:\n      - run: uv run ruff check pipeline/ trading_app/ scripts/ tests/\n",
        encoding="utf-8",
    )
    assert check_ci_ruff_lint_covers_tests(workflows_dir=tmp_path) == []


def test_ruff_format_line_is_not_a_false_positive(tmp_path: Path) -> None:
    """`ruff format` (not `ruff check`) must never trip the lint-scope guard."""
    wf = tmp_path / "ci.yml"
    wf.write_text(
        "jobs:\n  lint:\n    steps:\n      - run: uv run ruff format --check pipeline/ trading_app/ scripts/\n",
        encoding="utf-8",
    )
    assert check_ci_ruff_lint_covers_tests(workflows_dir=tmp_path) == []


def test_commented_old_scope_is_ignored(tmp_path: Path) -> None:
    """A commented narration of the old scope must not fire."""
    wf = tmp_path / "ci.yml"
    wf.write_text(
        "jobs:\n  lint:\n    steps:\n"
        "      # old: uv run ruff check pipeline/ trading_app/ scripts/\n"
        "      - run: uv run ruff check pipeline/ trading_app/ scripts/ tests/\n",
        encoding="utf-8",
    )
    assert check_ci_ruff_lint_covers_tests(workflows_dir=tmp_path) == []


def test_missing_workflows_dir_fails_open(tmp_path: Path) -> None:
    """Absent workflows directory → no enforcement, no crash (fail-open)."""
    assert check_ci_ruff_lint_covers_tests(workflows_dir=tmp_path / "nope") == []
