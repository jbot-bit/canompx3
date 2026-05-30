"""Tests for ralph_build_ledger.py v2 additions.

Covers:
- _file_hash returns sha256[:16] and empty string on missing file
- compute_files_audited populates file_hash + git_sha fields
- compute_known_acceptable_patterns deduplicates by (file, finding_type)
- build_ledger integration: new fields present in output
"""

import hashlib
from pathlib import Path

from scripts.tools.ralph_build_ledger import (
    _file_hash,
    build_ledger,
    compute_files_audited,
    compute_known_acceptable_patterns,
    parse_iterations,
)


# ── _file_hash ────────────────────────────────────────────────────────


def test_file_hash_known_content(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()[:16]
    assert _file_hash(f) == expected


def test_file_hash_missing_file(tmp_path: Path) -> None:
    assert _file_hash(tmp_path / "nonexistent.py") == ""


def test_file_hash_length(tmp_path: Path) -> None:
    f = tmp_path / "bar.py"
    f.write_bytes(b"x" * 1000)
    h = _file_hash(f)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ── compute_files_audited ─────────────────────────────────────────────

_MINIMAL_HISTORY = """\
## Iteration 1 — 2026-03-09
- Phase: fix
- Classification: [judgment]
- Target: {rel_path}:42
- Finding: hardcoded instrument list
- Action: replaced with canonical import
- Blast radius: 1 file
- Verification: PASS
- Commit: abc1234
"""


def test_compute_files_audited_has_hash_field(tmp_path: Path) -> None:
    # Create a real file so _file_hash can read it
    src = tmp_path / "pipeline" / "foo.py"
    src.parent.mkdir()
    src.write_bytes(b"# content")
    rel = "pipeline/foo.py"

    history = _MINIMAL_HISTORY.format(rel_path=rel)
    iterations = parse_iterations(history)
    result = compute_files_audited(iterations, tmp_path)

    assert rel in result
    assert "file_hash" in result[rel]
    assert "git_sha" in result[rel]
    expected_hash = hashlib.sha256(b"# content").hexdigest()[:16]
    assert result[rel]["file_hash"] == expected_hash


def test_compute_files_audited_missing_file_gives_empty_hash(tmp_path: Path) -> None:
    rel = "pipeline/does_not_exist.py"
    history = _MINIMAL_HISTORY.format(rel_path=rel)
    iterations = parse_iterations(history)
    result = compute_files_audited(iterations, tmp_path)

    assert rel in result
    assert result[rel]["file_hash"] == ""
    assert result[rel]["git_sha"] == ""  # no git repo in tmp_path


# ── compute_known_acceptable_patterns ─────────────────────────────────

_ACCEPTABLE_HISTORY = """\
## Iteration 10 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: trading_app/foo.py:30
- Finding: silent failure in broad_except handler
- Action: acceptable per project standards
- Blast radius: 1 file
- Verification: PASS
- Commit: NONE

## Iteration 11 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: trading_app/foo.py:55
- Finding: another broad_except at adapter boundary
- Action: acceptable per project standards
- Blast radius: 1 file
- Verification: PASS
- Commit: NONE

## Iteration 12 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: trading_app/bar.py:10
- Finding: hardcoded canonical violation — instrument list
- Action: acceptable heuristic per session
- Blast radius: 1 file
- Verification: PASS
- Commit: NONE
"""


def test_known_acceptable_deduplicates_same_file_and_type() -> None:
    iterations = parse_iterations(_ACCEPTABLE_HISTORY)
    result = compute_known_acceptable_patterns(iterations)
    # Both iter 10 and iter 11 target foo.py — "silent failure in broad_except handler"
    # classifies as silent_failure (first match in FINDING_TYPE_RULES). Both share
    # (file=trading_app/foo.py, finding_type=silent_failure) — should deduplicate to ONE entry.
    foo_silent = [p for p in result if p["file"] == "trading_app/foo.py" and p["finding_type"] == "silent_failure"]
    assert len(foo_silent) == 1


def test_known_acceptable_includes_different_files() -> None:
    iterations = parse_iterations(_ACCEPTABLE_HISTORY)
    result = compute_known_acceptable_patterns(iterations)
    files = {p["file"] for p in result}
    assert "trading_app/foo.py" in files
    assert "trading_app/bar.py" in files


def test_known_acceptable_empty_on_no_accept_verdicts() -> None:
    history = """\
## Iteration 1 — 2026-03-09
- Phase: fix
- Classification: [judgment]
- Target: pipeline/foo.py:1
- Finding: hardcoded instrument list
- Blast radius: 1 file
- Verification: PASS
- Commit: abc1234
"""
    iterations = parse_iterations(history)
    result = compute_known_acceptable_patterns(iterations)
    assert result == []


# ── build_ledger integration ───────────────────────────────────────────


def test_build_ledger_has_known_acceptable_patterns_key(tmp_path: Path) -> None:
    history = _ACCEPTABLE_HISTORY
    ledger = build_ledger(history, project_root=tmp_path)
    assert "known_acceptable_patterns" in ledger
    assert isinstance(ledger["known_acceptable_patterns"], list)


def test_build_ledger_files_audited_has_new_fields(tmp_path: Path) -> None:
    src = tmp_path / "trading_app" / "foo.py"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"# foo")

    ledger = build_ledger(_ACCEPTABLE_HISTORY, project_root=tmp_path)
    entry = ledger["files_audited"].get("trading_app/foo.py", {})
    assert "file_hash" in entry
    assert "git_sha" in entry
    assert entry["file_hash"] == hashlib.sha256(b"# foo").hexdigest()[:16]
