"""Tests for trading_app/phase_4_discovery_gates.py — Phase 4 Stage 4.1c.

Covers both write-side gates:

- ``check_git_cleanliness`` — git subprocess integration, mocked for the
  common unit tests and exercised end-to-end with a real temp git repo
  for the integration test.
- ``check_single_use`` — read-only query against an in-memory DuckDB
  fixture. No mocking needed — in-memory DuckDB is fast and faithful.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from unittest.mock import patch

import duckdb
import pytest

from trading_app.hypothesis_loader import HypothesisLoaderError
from trading_app.phase_4_discovery_gates import (
    check_git_cleanliness,
    check_single_use,
)


def _make_fixture_db() -> duckdb.DuckDBPyConnection:
    """In-memory DB mirroring the relevant experimental_strategies columns."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE experimental_strategies (
            strategy_id TEXT,
            hypothesis_file_sha TEXT,
            created_at TIMESTAMPTZ,
            orb_minutes INTEGER DEFAULT 5
        )
    """)
    return con


# ---------------------------------------------------------------------------
# check_git_cleanliness — mocked subprocess unit tests
# ---------------------------------------------------------------------------


class TestCheckGitCleanlinessMocked:
    """Fast unit tests with subprocess.run mocked — covers the branching
    logic without a real git repo."""

    def _mock_run(self, *, ls_returncode: int, diff_returncode: int):
        """Build a side_effect function that returns (ls_files, diff) results."""
        call_count = {"n": 0}

        def side_effect(args, **_kwargs):
            call_count["n"] += 1
            result = subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")
            if "ls-files" in args:
                result.returncode = ls_returncode
            elif "diff" in args:
                result.returncode = diff_returncode
            return result

        return side_effect

    def test_tracked_and_clean_passes(self, tmp_path):
        f = tmp_path / "h.yaml"
        f.write_text("metadata: {}\n", encoding="utf-8")
        with patch(
            "trading_app.phase_4_discovery_gates.subprocess.run",
            side_effect=self._mock_run(ls_returncode=0, diff_returncode=0),
        ):
            check_git_cleanliness(f)  # no raise

    def test_untracked_file_rejects(self, tmp_path):
        f = tmp_path / "h.yaml"
        f.write_text("metadata: {}\n", encoding="utf-8")
        with patch(
            "trading_app.phase_4_discovery_gates.subprocess.run",
            side_effect=self._mock_run(ls_returncode=1, diff_returncode=0),
        ):
            with pytest.raises(HypothesisLoaderError, match="not tracked"):
                check_git_cleanliness(f)

    def test_dirty_file_rejects(self, tmp_path):
        f = tmp_path / "h.yaml"
        f.write_text("metadata: {}\n", encoding="utf-8")
        with patch(
            "trading_app.phase_4_discovery_gates.subprocess.run",
            side_effect=self._mock_run(ls_returncode=0, diff_returncode=1),
        ):
            with pytest.raises(HypothesisLoaderError, match="uncommitted changes"):
                check_git_cleanliness(f)

    def test_missing_file_rejects_without_subprocess(self, tmp_path):
        """A non-existent file fails at the is_file() guard BEFORE any
        subprocess call — proven by the mock never being invoked."""
        nonexistent = tmp_path / "nope.yaml"
        with patch(
            "trading_app.phase_4_discovery_gates.subprocess.run"
        ) as mock_run:
            with pytest.raises(HypothesisLoaderError, match="not found"):
                check_git_cleanliness(nonexistent)
            mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# check_git_cleanliness — end-to-end integration with real temp git repo
# ---------------------------------------------------------------------------


class TestCheckGitCleanlinessIntegration:
    """One end-to-end test against a real temp git repo to verify the
    mocked unit tests don't diverge from real git behavior. Covers the
    full tracked+committed → edit → clean workflow in one scenario."""

    def test_real_git_tracked_clean_dirty_sequence(self, tmp_path, monkeypatch):
        # Build a minimal git repo in tmp_path
        monkeypatch.chdir(tmp_path)
        subprocess.run(["git", "init", "-q"], check=True, cwd=tmp_path)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            check=True,
            cwd=tmp_path,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=True,
            cwd=tmp_path,
        )

        hypothesis_file = tmp_path / "test_hypothesis.yaml"
        hypothesis_file.write_text(
            "metadata:\n  name: test\n",
            encoding="utf-8",
        )

        # Stage 1: untracked file → should reject
        with pytest.raises(HypothesisLoaderError, match="not tracked"):
            check_git_cleanliness(hypothesis_file)

        # Stage 2: add + commit → should pass
        subprocess.run(["git", "add", str(hypothesis_file)], check=True, cwd=tmp_path)
        subprocess.run(
            ["git", "commit", "-q", "-m", "pre-register test"],
            check=True,
            cwd=tmp_path,
        )
        check_git_cleanliness(hypothesis_file)  # no raise

        # Stage 3: edit the file without committing → should reject
        hypothesis_file.write_text(
            "metadata:\n  name: test\n  edited: true\n",
            encoding="utf-8",
        )
        with pytest.raises(HypothesisLoaderError, match="uncommitted"):
            check_git_cleanliness(hypothesis_file)

        # Stage 4: commit the edit → should pass again
        subprocess.run(["git", "add", str(hypothesis_file)], check=True, cwd=tmp_path)
        subprocess.run(
            ["git", "commit", "-q", "-m", "edit"],
            check=True,
            cwd=tmp_path,
        )
        check_git_cleanliness(hypothesis_file)  # no raise


# ---------------------------------------------------------------------------
# check_single_use — in-memory DuckDB fixture tests
# ---------------------------------------------------------------------------


class TestCheckSingleUse:
    """Fresh SHA passes; previously-used SHA fails with identifying info."""

    def test_fresh_sha_passes(self):
        con = _make_fixture_db()
        check_single_use("fresh_sha_" + "a" * 56, con)  # no raise

    def test_empty_db_passes(self):
        con = _make_fixture_db()
        # Table exists but empty — any SHA is "unused"
        check_single_use("any_sha", con)

    def test_sha_already_used_once_fails(self):
        con = _make_fixture_db()
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at) VALUES (?, ?, ?)",
            ["s1", "used_sha", datetime(2026, 4, 8, 10, 0, 0, tzinfo=UTC)],
        )
        with pytest.raises(HypothesisLoaderError, match="already been used"):
            check_single_use("used_sha", con)

    def test_error_message_includes_count_and_timestamp(self):
        con = _make_fixture_db()
        # Use distinct DATES so the min-timestamp assertion is robust to
        # DuckDB's timezone display (DuckDB renders TIMESTAMPTZ in the
        # session's local timezone in str output, which on Brisbane shifts
        # UTC times by +10 hours — would break an hour-precise assertion).
        first = datetime(2026, 4, 5, 10, 0, 0, tzinfo=UTC)
        second = datetime(2026, 4, 6, 14, 0, 0, tzinfo=UTC)
        third = datetime(2026, 4, 7, 18, 0, 0, tzinfo=UTC)
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at) VALUES (?, ?, ?)",
            ["s1", "multi_use_sha", second],
        )
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at) VALUES (?, ?, ?)",
            ["s2", "multi_use_sha", first],  # earliest
        )
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at) VALUES (?, ?, ?)",
            ["s3", "multi_use_sha", third],
        )
        try:
            check_single_use("multi_use_sha", con)
            pytest.fail("expected rejection")
        except HypothesisLoaderError as e:
            msg = str(e)
            assert "3" in msg  # count
            # MIN(created_at) should be the earliest date (2026-04-05).
            # Later dates must NOT appear — that would indicate min() was
            # picking the wrong row. Date-level assertion is timezone-safe
            # since the 3 dates are all 24+ hours apart.
            assert "2026-04-05" in msg
            assert "2026-04-06" not in msg
            assert "2026-04-07" not in msg
            assert "supersede" in msg.lower()

    def test_other_sha_rows_do_not_collide(self):
        """Inserting rows with a DIFFERENT SHA must not affect the query."""
        con = _make_fixture_db()
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at) VALUES (?, ?, ?)",
            ["s1", "different_sha", datetime(2026, 4, 8, 10, 0, 0, tzinfo=UTC)],
        )
        # Query a different SHA — should pass
        check_single_use("my_target_sha", con)

    def test_sql_injection_attempt_is_parameterized(self):
        """Parameter binding prevents SQL injection via the SHA string.

        A malicious SHA like "'; DROP TABLE experimental_strategies; --"
        must be treated as a literal string, not executed."""
        con = _make_fixture_db()
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at) VALUES (?, ?, ?)",
            ["s1", "legitimate_sha", datetime(2026, 4, 8, 10, 0, 0, tzinfo=UTC)],
        )
        # The injection string is not in the DB, so this should pass
        check_single_use(
            "'; DROP TABLE experimental_strategies; --",
            con,
        )
        # Confirm the table still exists and the legitimate row is still there
        count = con.execute(
            "SELECT COUNT(*) FROM experimental_strategies"
        ).fetchone()
        assert count is not None
        assert count[0] == 1

    def test_different_orb_minutes_does_not_block(self):
        """A hypothesis file used at orb_minutes=15 must NOT block a run
        at orb_minutes=30 — they cover disjoint subsets of the family."""
        con = _make_fixture_db()
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at, orb_minutes) "
            "VALUES (?, ?, ?, ?)",
            ["s1_O15", "shared_sha", datetime(2026, 4, 13, 9, 0, 0, tzinfo=UTC), 15],
        )
        # orb_minutes=30 with same SHA — should pass (different aperture)
        check_single_use("shared_sha", con, orb_minutes=30)

    def test_same_orb_minutes_still_blocks(self):
        """A hypothesis file used at orb_minutes=15 MUST block a re-run
        at the same orb_minutes=15 — that IS a same-aperture re-run."""
        con = _make_fixture_db()
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at, orb_minutes) "
            "VALUES (?, ?, ?, ?)",
            ["s1_O15", "shared_sha", datetime(2026, 4, 13, 9, 0, 0, tzinfo=UTC), 15],
        )
        with pytest.raises(HypothesisLoaderError, match="already been used"):
            check_single_use("shared_sha", con, orb_minutes=15)

    def test_legacy_no_orb_minutes_still_blocks_globally(self):
        """When orb_minutes is None (legacy), ANY row with the SHA blocks."""
        con = _make_fixture_db()
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, hypothesis_file_sha, created_at, orb_minutes) "
            "VALUES (?, ?, ?, ?)",
            ["s1_O15", "sha_legacy", datetime(2026, 4, 13, 9, 0, 0, tzinfo=UTC), 15],
        )
        with pytest.raises(HypothesisLoaderError, match="already been used"):
            check_single_use("sha_legacy", con)
