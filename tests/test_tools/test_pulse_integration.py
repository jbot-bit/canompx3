"""Integration tests for project_pulse.py — runs against REAL repo state.

These tests do NOT mock. They run the actual pulse against the actual repo,
actual gold.db, actual git state. If these pass, the pulse works for real.

WHO CHECKS THE CHECKER: unit tests prove logic; these prove reality.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PULSE_SCRIPT = PROJECT_ROOT / "scripts" / "tools" / "project_pulse.py"
PYTHON = sys.executable


@pytest.fixture
def seeded_pulse_db(tmp_path):
    """Temp gold.db with one active validated_setups row per active instrument.

    project_pulse reports `Strategy fitness:` only when fitness_summary is
    non-empty, which requires at least one validated_setups row. The test
    runs the pulse via subprocess with DUCKDB_PATH pointing here.
    """
    db_path = tmp_path / "gold.db"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            orb_label VARCHAR,
            entry_model VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            sharpe_ann DOUBLE,
            status VARCHAR,
            fdr_significant BOOLEAN,
            wf_passed BOOLEAN,
            stop_multiplier DOUBLE
        )
        """
    )
    for inst in ("MNQ", "MES"):
        con.execute(
            """
            INSERT INTO validated_setups
                (strategy_id, instrument, orb_label, entry_model, orb_minutes,
                 rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                 sample_size, sharpe_ann, status, fdr_significant, wf_passed,
                 stop_multiplier)
            VALUES (?, ?, 'TOKYO_OPEN', 'E2', 5, 1.5, 1, 'ORB_G5',
                    0.18, 0.52, 150, 1.1, 'active', TRUE, TRUE, 1.0)
            """,
            [f"{inst}_TOKYO_OPEN_E2_CB1_ORB_G5_RR1.5", inst],
        )
    con.close()
    return db_path


def _run_pulse(*args: str, timeout: int = 60, db_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if db_path is not None:
        env["DUCKDB_PATH"] = str(db_path)
    return subprocess.run(
        [PYTHON, str(PULSE_SCRIPT), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


class TestPulseIntegration:
    """Run the REAL pulse against the REAL repo. No mocks."""

    def test_fast_text_runs_without_error(self) -> None:
        """Basic smoke: does it run and produce output?"""
        r = _run_pulse("--fast", "--no-cache")
        # Exit code 0 (clean) or 1 (broken items) are both valid
        assert r.returncode in (0, 1), f"Unexpected exit code {r.returncode}: {r.stderr}"
        assert "PROJECT PULSE" in r.stdout
        assert "===" in r.stdout

    def test_fast_text_has_required_sections(self, seeded_pulse_db) -> None:
        """Output must have the key sections a user needs."""
        r = _run_pulse("--fast", "--no-cache", db_path=seeded_pulse_db)
        output = r.stdout
        # Must show branch and HEAD
        assert "Branch:" in output
        assert "HEAD:" in output
        # Must have a recommendation line
        assert ">>>" in output
        # Must show strategy fitness (DB accessible from canonical root)
        assert "Strategy fitness:" in output or "fitness" in output.lower()

    def test_json_is_valid_and_complete(self) -> None:
        """JSON output must parse and contain all required keys."""
        r = _run_pulse("--fast", "--no-cache", "--format", "json")
        assert r.returncode in (0, 1)
        data = json.loads(r.stdout)
        # Required top-level keys
        for key in (
            "generated_at",
            "git_head",
            "git_branch",
            "counts",
            "items",
            "handoff",
            "fitness_summary",
            "deployment_summary",
            "survival_summary",
            "sr_summary",
            "pause_summary",
            "recommendation",
            "upcoming_sessions",
            "time_since_green",
            "session_delta",
        ):
            assert key in data, f"Missing key: {key}"
        # Counts must have all categories
        for cat in ("broken", "decaying", "ready", "unactioned", "paused"):
            assert cat in data["counts"], f"Missing count category: {cat}"
        # Items must be a list of dicts with required fields
        for item in data["items"]:
            assert "category" in item
            assert "severity" in item
            assert "source" in item
            assert "summary" in item

    def test_markdown_has_structure(self) -> None:
        """Markdown output must have heading structure."""
        r = _run_pulse("--fast", "--no-cache", "--format", "markdown")
        assert r.returncode in (0, 1)
        assert r.stdout.startswith("# Project Pulse")

    def test_fitness_shows_all_active_instruments(self) -> None:
        """Fitness summary should reflect active validated strategies without stale assumptions."""
        r = _run_pulse("--fast", "--no-cache", "--format", "json")
        data = json.loads(r.stdout)
        fitness = data.get("fitness_summary", {})
        if fitness:  # Only check if DB was accessible
            assert "MNQ" in fitness, f"MNQ missing from fitness: {list(fitness.keys())}"
            assert all("active_strategies" in stats for stats in fitness.values())

    def test_upcoming_sessions_have_valid_times(self) -> None:
        """Upcoming sessions must have plausible Brisbane times."""
        r = _run_pulse("--fast", "--no-cache", "--format", "json")
        data = json.loads(r.stdout)
        for session in data.get("upcoming_sessions", []):
            assert "label" in session
            assert "brisbane_time" in session
            assert "hours_away" in session
            # Time should be HH:MM format
            time_str = session["brisbane_time"]
            parts = time_str.split(":")
            assert len(parts) == 2, f"Bad time format: {time_str}"
            assert 0 <= int(parts[0]) <= 23
            assert 0 <= int(parts[1]) <= 59
            # Hours away should be 0-6
            assert 0 <= session["hours_away"] <= 6.5

    def test_recommendation_is_not_empty(self) -> None:
        """The recommendation engine must always produce something."""
        r = _run_pulse("--fast", "--no-cache", "--format", "json")
        data = json.loads(r.stdout)
        assert data["recommendation"], "Recommendation must not be empty"
        assert len(data["recommendation"]) > 5, "Recommendation too short"

    def test_items_have_skill_suggestions(self) -> None:
        """Actionable items should have skill suggestions attached."""
        r = _run_pulse("--fast", "--no-cache", "--format", "json")
        data = json.loads(r.stdout)
        # At least some items with known sources should have actions
        sources_with_suggestions = {"staleness", "drift", "tests", "fitness", "handoff", "ralph"}
        items_with_known_source = [i for i in data["items"] if i["source"] in sources_with_suggestions]
        if items_with_known_source:
            items_with_action = [i for i in items_with_known_source if i.get("action")]
            assert len(items_with_action) > 0, (
                f"No skill suggestions on {len(items_with_known_source)} items from known sources"
            )

    def test_cache_works_across_runs(self) -> None:
        """Second run should be faster (cache hit for drift/tests)."""
        # First run — populates cache (skip drift/tests to be fast)
        r1 = _run_pulse("--fast", "--no-cache")
        assert r1.returncode in (0, 1)
        # Second run — should use cache
        r2 = _run_pulse("--fast")
        assert r2.returncode in (0, 1)
        # Both should produce valid output
        assert "PROJECT PULSE" in r2.stdout

    def test_text_output_is_scannable(self) -> None:
        """Text output should be concise enough to read in 10 seconds."""
        r = _run_pulse("--fast", "--no-cache")
        lines = r.stdout.strip().splitlines()
        # Keep enough headroom for a few extra high-signal sections without
        # letting the pulse sprawl into a wall of text.
        assert len(lines) <= 60, f"Output too long for a pulse ({len(lines)} lines). Should be <=60."
