"""Mutation-probe tests for Check #159
(``check_canonical_inline_copies_have_parity_check``).

Stage 2 of the canonical-inline-copy-parity-bug-class defense. Each test
deliberately introduces ONE failure mode of the meta-check and asserts the
expected violation surfaces. Sibling-coverage doctrine per
``memory/feedback_regex_alternation_sibling_coverage.md`` -- one test per
documented failure path of the check.

Tests
-----
test_clean_registry_passes
    Baseline: with the on-disk registry, the meta-check returns no
    violations. Pins the canonical happy path.

test_empty_registry_violates
    CANONICAL_INLINE_COPIES = [] -> dedicated violation message instead
    of a silent pass.

test_missing_parity_check_function_violates
    Registry entry points at a function name not present in
    pipeline.check_drift module globals -> orphan violation.

test_missing_test_file_violates
    Registry entry points at a test file path that does not exist on
    disk -> sibling-coverage doctrine violation.

test_insufficient_test_functions_violates
    Test file exists but contains fewer ``def test_`` functions than the
    entry has gated_constants -> sibling-coverage doctrine violation.

test_non_inline_copy_pair_violates
    Registry contains an object that is not an InlineCopyPair -> type
    violation. Guards against operator passing a dict or namedtuple by
    accident.

test_registry_import_failure_failopen
    The check fail-closes (returns a violation) when the registry module
    cannot be imported. Asserts the meta-check never silently passes when
    Layer 2 is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pipeline.canonical_inline_copies import (
    CANONICAL_INLINE_COPIES,
    InlineCopyPair,
)
from pipeline.check_drift import check_canonical_inline_copies_have_parity_check


def _snapshot() -> list[InlineCopyPair]:
    return list(CANONICAL_INLINE_COPIES)


@pytest.fixture
def reset_registry():
    """Restore the registry list after each test, regardless of failure."""
    original = _snapshot()
    yield
    CANONICAL_INLINE_COPIES.clear()
    CANONICAL_INLINE_COPIES.extend(original)


def test_clean_registry_passes(reset_registry):
    """Baseline: on-disk registry produces zero violations."""
    violations = check_canonical_inline_copies_have_parity_check()
    assert violations == [], "Clean registry should produce no violations. Got: " + "\n".join(violations)


def test_empty_registry_violates(reset_registry):
    """Empty list is a structural error, not a silent pass."""
    CANONICAL_INLINE_COPIES.clear()
    violations = check_canonical_inline_copies_have_parity_check()
    assert len(violations) == 1
    assert "CANONICAL_INLINE_COPIES is empty" in violations[0]
    assert "fast_lane_promote_threshold" in violations[0]


def test_missing_parity_check_function_violates(reset_registry):
    """Registry points at a non-existent parity-check function."""
    CANONICAL_INLINE_COPIES.clear()
    CANONICAL_INLINE_COPIES.append(
        InlineCopyPair(
            name="orphan_test_pair",
            inline_site="pipeline/canonical_inline_copies.py",
            canonical_source="docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml",
            gated_constants=("FOO",),
            parity_check="check_function_that_does_not_exist_xyz",
            test_file=("tests/test_pipeline/test_check_drift_fast_lane_promote_threshold_parity.py"),
            bug_class_anchor="memory/feedback_canonical_inline_copy_parity_bug_class.md",
        )
    )
    violations = check_canonical_inline_copies_have_parity_check()
    assert any(
        "check_function_that_does_not_exist_xyz" in v and "not found" in v and "orphan_test_pair" in v
        for v in violations
    ), f"Expected orphan-function violation, got: {violations}"


def test_missing_test_file_violates(reset_registry):
    """Registry points at a test file that does not exist on disk."""
    CANONICAL_INLINE_COPIES.clear()
    CANONICAL_INLINE_COPIES.append(
        InlineCopyPair(
            name="missing_test_file_pair",
            inline_site="scripts/research/fast_lane_promote_queue.py",
            canonical_source="docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml",
            gated_constants=("FOO",),
            parity_check="check_fast_lane_promote_threshold_parity",
            test_file=("tests/test_pipeline/test_file_that_does_not_exist_xyz.py"),
            bug_class_anchor="memory/feedback_canonical_inline_copy_parity_bug_class.md",
        )
    )
    violations = check_canonical_inline_copies_have_parity_check()
    assert any("test_file_that_does_not_exist_xyz.py" in v and "not found" in v for v in violations), (
        f"Expected missing-test-file violation, got: {violations}"
    )


def test_insufficient_test_functions_violates(reset_registry, tmp_path: Path):
    """Test file has fewer test functions than gated_constants count."""
    # Create a tiny test file with one `def test_` function, but claim
    # six gated constants. The meta-check should fire sibling-coverage
    # violation.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    rel_test_file = "tests/test_pipeline/_synthetic_undertested.py"
    abs_test_file = PROJECT_ROOT / rel_test_file
    abs_test_file.write_text(
        "def test_only_one_function():\n    assert True\n",
        encoding="utf-8",
    )
    try:
        CANONICAL_INLINE_COPIES.clear()
        CANONICAL_INLINE_COPIES.append(
            InlineCopyPair(
                name="undertested_pair",
                inline_site="scripts/research/fast_lane_promote_queue.py",
                canonical_source="docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml",
                gated_constants=("A", "B", "C", "D", "E", "F"),
                parity_check="check_fast_lane_promote_threshold_parity",
                test_file=rel_test_file,
                bug_class_anchor="memory/feedback_canonical_inline_copy_parity_bug_class.md",
            )
        )
        violations = check_canonical_inline_copies_have_parity_check()
        assert any("sibling-coverage" in v and "expected >= 6" in v and "found 1" in v for v in violations), (
            f"Expected sibling-coverage violation, got: {violations}"
        )
    finally:
        if abs_test_file.exists():
            abs_test_file.unlink()


def test_non_inline_copy_pair_violates(reset_registry):
    """A non-InlineCopyPair object in the registry is a type violation."""
    CANONICAL_INLINE_COPIES.clear()
    CANONICAL_INLINE_COPIES.append({"name": "not_a_pair"})  # type: ignore[arg-type]
    violations = check_canonical_inline_copies_have_parity_check()
    assert any("not an InlineCopyPair" in v for v in violations), f"Expected type violation, got: {violations}"


def test_registry_import_failure_failopen(monkeypatch):
    """Import error on the registry module surfaces as a violation, not a silent pass."""
    # Simulate the registry module being missing/broken by removing it
    # from sys.modules and inserting a path that will reraise on import.
    monkeypatch.delitem(sys.modules, "pipeline.canonical_inline_copies", raising=False)

    class _BrokenLoader:
        def find_module(self, name, path=None):
            if name == "pipeline.canonical_inline_copies":
                return self
            return None

        def load_module(self, name):
            raise ImportError("simulated registry import failure for test")

    # Install a custom meta_path finder that breaks ONLY the target module.
    import importlib.abc
    import importlib.machinery

    class _BlockingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "pipeline.canonical_inline_copies":
                raise ImportError("simulated registry import failure for test")
            return None

    finder = _BlockingFinder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path], raising=False)

    violations = check_canonical_inline_copies_have_parity_check()
    assert len(violations) >= 1
    assert any("could not import pipeline.canonical_inline_copies" in v for v in violations), (
        f"Expected import-failure violation, got: {violations}"
    )


def test_seed_entry_is_fast_lane_promote_threshold():
    """Belt-and-suspenders: the registry must seed the Stage 1 anchor.

    If somebody clears the registry to fix a different bug, this test pins
    the canonical first entry so we don't silently lose the Layer-1 anchor.
    """
    assert len(CANONICAL_INLINE_COPIES) >= 1
    first = CANONICAL_INLINE_COPIES[0]
    assert first.name == "fast_lane_promote_threshold"
    assert first.parity_check == "check_fast_lane_promote_threshold_parity"
    assert len(first.gated_constants) == 6
