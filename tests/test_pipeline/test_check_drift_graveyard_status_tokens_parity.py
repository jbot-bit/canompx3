"""Injection tests for check_graveyard_status_tokens_parity (Check #172).

The graveyard-digest parser reads the status-token alternation from a
``## Status Token Doctrine`` block in ``chatgpt_bundle/06_RD_GRAVEYARD.md``
rather than inlining the list. This check asserts:

  - The doctrine block parses to a non-empty ``status_tokens`` list.
  - Every status token surfaced by a real heading in the file is declared
    in the doctrine.
  - No duplicate tokens in the doctrine.

Class anchor: [[canonical-inline-copy-parity-bug-class]] (9th confirmed
instance, 2026-05-20 — Stage 2A.2 follow-up).

Background:
  memory/feedback_canonical_inline_copy_parity_bug_class.md
  chatgpt_bundle/06_RD_GRAVEYARD.md § Status Token Doctrine
  scripts/research/fast_lane_graveyard_digest.py::_load_status_tokens
"""

from __future__ import annotations

from pathlib import Path

from pipeline.check_drift import check_graveyard_status_tokens_parity

# ----------------------------------------------------------------------
# Clean-state baseline
# ----------------------------------------------------------------------


def test_clean_state_passes() -> None:
    """Real graveyard doctrine block must cover every status token used
    by real headings in the same file."""
    violations = check_graveyard_status_tokens_parity()
    assert violations == [], f"unexpected parity violations on clean state: {violations}"


# ----------------------------------------------------------------------
# Fail-closed: missing canonical source
# ----------------------------------------------------------------------


def test_missing_graveyard_file_fails_closed(tmp_path: Path) -> None:
    """If 06_RD_GRAVEYARD.md is unreachable the check returns a single
    violation rather than silently passing."""
    forged = tmp_path / "does-not-exist.md"
    violations = check_graveyard_status_tokens_parity(graveyard_path=forged)
    assert violations
    assert len(violations) == 1
    assert "canonical graveyard source missing" in violations[0]


# ----------------------------------------------------------------------
# Injection 1: doctrine block missing entirely
# ----------------------------------------------------------------------


def test_missing_doctrine_block_is_caught(tmp_path: Path) -> None:
    """A graveyard file lacking the ``## Status Token Doctrine`` block
    must produce a violation."""
    forged = tmp_path / "graveyard.md"
    forged.write_text(
        "# R&D Graveyard\n\n## ML attempt — DEAD\n\nSome prose.\n",
        encoding="utf-8",
    )
    violations = check_graveyard_status_tokens_parity(graveyard_path=forged)
    assert violations
    assert any("Status Token Doctrine" in v for v in violations)
    assert any("canonical-inline-copy-parity-bug-class" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 2: doctrine block parses to empty list
# ----------------------------------------------------------------------


def test_empty_status_tokens_is_caught(tmp_path: Path) -> None:
    """An empty ``status_tokens:`` list must be flagged — parser would
    capture every heading as UNKNOWN."""
    forged = tmp_path / "graveyard.md"
    forged.write_text(
        "# Graveyard\n\n## Status Token Doctrine\n\n```yaml\nstatus_tokens: []\n```\n\n## ML — DEAD\n",
        encoding="utf-8",
    )
    violations = check_graveyard_status_tokens_parity(graveyard_path=forged)
    assert violations
    assert any("empty" in v.lower() or "missing" in v.lower() for v in violations)


# ----------------------------------------------------------------------
# Injection 3: heading uses a status token not in the doctrine
# ----------------------------------------------------------------------


def test_undeclared_status_token_in_heading_is_caught(tmp_path: Path) -> None:
    """A heading using a status token the doctrine doesn't list must
    fire a violation that names the undeclared token."""
    forged = tmp_path / "graveyard.md"
    forged.write_text(
        "# Graveyard\n\n"
        "## Status Token Doctrine\n\n"
        "```yaml\n"
        "status_tokens:\n"
        "  - DEAD\n"
        "  - NO-GO\n"
        "```\n\n"
        "## VWAP X — PURGED 2026-06-01\n",
        encoding="utf-8",
    )
    violations = check_graveyard_status_tokens_parity(graveyard_path=forged)
    assert violations
    assert any("PURGED" in v for v in violations), f"violation should name the undeclared token PURGED: {violations}"


# ----------------------------------------------------------------------
# Injection 4: duplicate tokens in the doctrine
# ----------------------------------------------------------------------


def test_duplicate_tokens_in_doctrine_is_caught(tmp_path: Path) -> None:
    """Duplicate entries in ``status_tokens:`` flagged for cleanup."""
    forged = tmp_path / "graveyard.md"
    forged.write_text(
        "# Graveyard\n\n"
        "## Status Token Doctrine\n\n"
        "```yaml\n"
        "status_tokens:\n"
        "  - DEAD\n"
        "  - NO-GO\n"
        "  - DEAD\n"
        "```\n\n"
        "## ML — DEAD\n",
        encoding="utf-8",
    )
    violations = check_graveyard_status_tokens_parity(graveyard_path=forged)
    assert violations
    assert any("duplicate" in v.lower() for v in violations)
    assert any("'DEAD'" in v for v in violations)


# ----------------------------------------------------------------------
# Negative: declared token in heading does NOT fire
# ----------------------------------------------------------------------


def test_declared_token_in_heading_passes(tmp_path: Path) -> None:
    """A minimal valid file with one declared token + one matching heading
    must pass."""
    forged = tmp_path / "graveyard.md"
    forged.write_text(
        "# Graveyard\n\n"
        "## Status Token Doctrine\n\n"
        "```yaml\n"
        "status_tokens:\n"
        "  - DEAD\n"
        "  - NO-GO\n"
        "  - PAUSED\n"
        "```\n\n"
        "## ML attempt — DEAD\n"
        "## Filter X — NO-GO\n",
        encoding="utf-8",
    )
    violations = check_graveyard_status_tokens_parity(graveyard_path=forged)
    assert violations == [], f"unexpected violations on valid synthetic graveyard: {violations}"
