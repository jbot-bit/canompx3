"""Tests for scripts.tools.project_pulse."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.tools import project_pulse
from scripts.tools.project_pulse import (
    PulseItem,
    PulseReport,
    _find_memory_md,
    _read_expensive_cache,
    _write_expensive_cache,
    build_pulse,
    collect_action_queue,
    collect_deployment_state,
    collect_drift,
    collect_git_state,
    collect_handoff,
    collect_lifecycle_control,
    collect_pause_state,
    collect_ralph_deferred,
    collect_session_claims,
    collect_sr_state,
    collect_staleness,
    collect_survival_state,
    collect_system_identity,
    collect_tests,
    collect_worktrees,
    format_json,
    format_markdown,
    format_text,
)


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# collect_handoff
# ---------------------------------------------------------------------------


class TestCollectHandoff:
    def test_missing_handoff(self, tmp_path: Path) -> None:
        context, items = collect_handoff(tmp_path)
        assert any(i.category == "broken" and "missing" in i.summary.lower() for i in items)
        assert context == {}

    def test_extracts_metadata_and_next_steps(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "HANDOFF.md",
            "\n".join(
                [
                    "## Last Session",
                    "- **Tool:** Claude",
                    "- **Date:** 2026-03-17",
                    "- **Summary:** Built pulse",
                    "",
                    "## Next Steps — Active",
                    "1. Phase 1: do thing",
                    "2. Phase 2: do other thing",
                    "",
                    "## Blockers / Warnings",
                    "- All good here",
                    "- Pre-existing test failure: broken thing",
                ]
            ),
        )
        context, items = collect_handoff(tmp_path)
        assert context["tool"] == "Claude"
        assert context["date"] == "2026-03-17"
        assert context["summary"] == "Built pulse"
        assert len(context["next_steps"]) == 2
        # "failure" keyword → broken item
        assert any(i.category == "broken" and "failure" in i.summary.lower() for i in items)
        # "All good here" has no blocker keywords → not surfaced
        assert not any("All good" in i.summary for i in items)

    def test_no_blockers_section(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "HANDOFF.md",
            "\n".join(
                [
                    "## Last Session",
                    "- **Tool:** Codex",
                    "- **Date:** 2026-03-18",
                    "- **Summary:** Research",
                ]
            ),
        )
        context, items = collect_handoff(tmp_path)
        assert context["tool"] == "Codex"
        assert items == []

    def test_extracts_current_rolling_update_format(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "HANDOFF.md",
            "\n".join(
                [
                    "## Update (2026-04-11 — Concurrency Guardrails v1)",
                    "",
                    "### Headline",
                    "",
                    "Implemented same-branch mutating session enforcement.",
                    "",
                    "### Next move",
                    "",
                    "Highest-value next step remains:",
                    "",
                    "- Criterion 11 v2",
                    "- Derived state contract follow-up",
                ]
            ),
        )

        context, items = collect_handoff(tmp_path)

        assert context["tool"] == "Update log"
        assert context["date"] == "2026-04-11"
        assert context["summary"] == "Implemented same-branch mutating session enforcement."
        assert context["next_steps"] == ["Criterion 11 v2", "Derived state contract follow-up"]
        assert items == []

    def test_extracts_priority_order_next_steps_from_rolling_update(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "HANDOFF.md",
            "\n".join(
                [
                    "## Update (2026-04-09 evening — Full Discovery Methodology Overhaul + 6-Strategy Portfolio)",
                    "",
                    "### Where we are",
                    "",
                    "Session redesigned discovery methodology from scratch.",
                    "",
                    "### Next steps (priority order)",
                    "",
                    "#### 1. Review tiered portfolio doc (IMMEDIATE)",
                    "Read `docs/plans/2026-04-09-portfolio-tiered.md`.",
                    "",
                    "#### 2. Deploy Tier 1 portfolio",
                    "Update `trading_app/prop_profiles.py` with 6 Tier 1 strategies.",
                ]
            ),
        )

        context, _items = collect_handoff(tmp_path)

        assert context["tool"] == "Update log"
        assert context["date"] == "2026-04-09"
        assert context["summary"] == "Full Discovery Methodology Overhaul + 6-Strategy Portfolio"
        assert context["next_steps"] == [
            "Review tiered portfolio doc (IMMEDIATE) — Read docs/plans/2026-04-09-portfolio-tiered.md.",
            "Deploy Tier 1 portfolio — Update trading_app/prop_profiles.py with 6 Tier 1 strategies.",
        ]

    def test_skips_empty_placeholder_update_and_uses_first_substantive_one(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "HANDOFF.md",
            "\n".join(
                [
                    "## Update (2026-04-11 — placeholder)",
                    "",
                    "## Update (2026-04-11 — Concurrency Guardrails v1)",
                    "",
                    "### Headline",
                    "",
                    "Implemented same-branch mutating session enforcement.",
                ]
            ),
        )

        context, _items = collect_handoff(tmp_path)

        assert context["tool"] == "Update log"
        assert context["date"] == "2026-04-11"
        assert context["summary"] == "Implemented same-branch mutating session enforcement."


# ---------------------------------------------------------------------------
# collect_git_state
# ---------------------------------------------------------------------------


class TestCollectGitState:
    def test_clean_repo(self, tmp_path: Path) -> None:
        mock_status = MagicMock(returncode=0, stdout="")
        mock_stash = MagicMock(returncode=0, stdout="")
        with patch.object(project_pulse, "_run_git", side_effect=[mock_status, mock_stash]):
            items = collect_git_state(tmp_path)
        assert items == []

    def test_dirty_files_excludes_cache(self, tmp_path: Path) -> None:
        mock_status = MagicMock(returncode=0, stdout=" M foo.py\n?? .pulse_cache.json\n M bar.py\n")
        mock_stash = MagicMock(returncode=0, stdout="")
        with patch.object(project_pulse, "_run_git", side_effect=[mock_status, mock_stash]):
            items = collect_git_state(tmp_path)
        assert len(items) == 1
        assert "2 uncommitted" in items[0].summary  # foo.py and bar.py, not cache

    def test_stashes_detected(self, tmp_path: Path) -> None:
        mock_status = MagicMock(returncode=0, stdout="")
        mock_stash = MagicMock(returncode=0, stdout="stash@{0}: WIP on main\nstash@{1}: old work\n")
        with patch.object(project_pulse, "_run_git", side_effect=[mock_status, mock_stash]):
            items = collect_git_state(tmp_path)
        assert len(items) == 1
        assert "2 git stash" in items[0].summary


class TestCollectSessionClaims:
    def test_reports_safe_parallel_claims(self, tmp_path: Path) -> None:
        mock_preflight = MagicMock()
        mock_preflight.list_claims.return_value = [
            MagicMock(tool="codex", branch="wt-codex-a", mode="mutating"),
            MagicMock(tool="claude", branch="wt-claude-b", mode="mutating"),
        ]
        with patch.dict("sys.modules", {"session_preflight": mock_preflight}):
            items = collect_session_claims(tmp_path)
        assert len(items) == 1
        assert items[0].category == "paused"
        assert "parallel appears isolated" in items[0].summary

    def test_reports_dangerous_same_branch_mutating_claims(self, tmp_path: Path) -> None:
        mock_preflight = MagicMock()
        mock_preflight.list_claims.return_value = [
            MagicMock(tool="codex", branch="main", mode="mutating"),
            MagicMock(tool="claude", branch="main", mode="mutating"),
        ]
        with patch.dict("sys.modules", {"session_preflight": mock_preflight}):
            items = collect_session_claims(tmp_path)
        assert len(items) == 1
        assert items[0].category == "decaying"
        assert "dangerous same-branch mutating claims" in items[0].summary


# ---------------------------------------------------------------------------
# collect_drift
# ---------------------------------------------------------------------------


class TestCollectDrift:
    def test_clean_drift(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "pipeline" / "check_drift.py", "")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="NO DRIFT DETECTED: 67 passed")
            items = collect_drift(tmp_path)
        assert items == []

    def test_drift_failure(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "pipeline" / "check_drift.py", "")
        stdout = "Check 5: something...\n  FAILED:\n    violation 1\nDRIFT DETECTED: 1 violation(s)"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout=stdout)
            items = collect_drift(tmp_path)
        assert len(items) == 1
        assert items[0].category == "broken"
        # Only "FAILED:" line counted, not the "DRIFT DETECTED" summary line
        assert "1 violation" in items[0].summary

    def test_drift_timeout(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "pipeline" / "check_drift.py", "")
        from subprocess import TimeoutExpired

        with patch("subprocess.run", side_effect=TimeoutExpired("cmd", 60)):
            items = collect_drift(tmp_path)
        assert len(items) == 1
        assert "timed out" in items[0].summary

    def test_missing_script(self, tmp_path: Path) -> None:
        items = collect_drift(tmp_path)
        assert items == []


# ---------------------------------------------------------------------------
# collect_tests
# ---------------------------------------------------------------------------


class TestCollectTests:
    def test_tests_pass(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="42 passed")
            items = collect_tests(tmp_path)
        assert items == []

    def test_tests_fail(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="FAILED tests/test_foo.py::test_bar\n1 failed, 41 passed",
            )
            items = collect_tests(tmp_path)
        assert len(items) == 1
        assert items[0].category == "broken"

    def test_tests_timeout_is_paused_not_broken(self, tmp_path: Path) -> None:
        """A suite that exceeds the 120s pulse budget is paused (budget
        mismatch), not broken. Broken is reserved for real failures."""
        (tmp_path / "tests").mkdir()
        from subprocess import TimeoutExpired

        with patch("subprocess.run", side_effect=TimeoutExpired("cmd", 120)):
            items = collect_tests(tmp_path)
        assert len(items) == 1
        assert items[0].category == "paused"
        assert items[0].severity == "low"
        assert "skipped" in items[0].summary.lower()


# ---------------------------------------------------------------------------
# collect_staleness
# ---------------------------------------------------------------------------


class TestCollectStaleness:
    def test_db_missing(self, tmp_path: Path) -> None:
        items = collect_staleness(tmp_path, tmp_path / "nonexistent.db")
        assert len(items) == 1
        assert items[0].category == "broken"
        assert "not found" in items[0].summary

    def test_stale_instruments(self, tmp_path: Path) -> None:
        db_path = tmp_path / "gold.db"
        db_path.touch()
        mock_engine = MagicMock(return_value={"stale_steps": ["outcome_builder", "discovery"]})
        mock_asset_configs = MagicMock()
        mock_asset_configs.ACTIVE_ORB_INSTRUMENTS = ["MGC"]
        mock_asset_configs.DEPLOYABLE_ORB_INSTRUMENTS = ["MGC"]
        mock_pipeline_status = MagicMock(staleness_engine=mock_engine)
        with (
            patch.dict(
                "sys.modules",
                {
                    "pipeline.asset_configs": mock_asset_configs,
                    "pipeline_status": mock_pipeline_status,
                },
            ),
            patch("duckdb.connect") as mock_connect,
        ):
            mock_con = MagicMock()
            mock_connect.return_value = mock_con
            items = collect_staleness(tmp_path, db_path)
        assert isinstance(items, list)
        # With mocked staleness_engine returning stale steps, we expect a decaying item
        assert any(i.category == "decaying" for i in items)

    def test_research_only_validated_setups_step_suppressed(self, tmp_path: Path) -> None:
        """Research-only instruments (active but not deployable) must not surface
        the 'validated_setups' stale step, because the empty shelf is by-design.
        Any OTHER stale step for the same instrument still alerts."""
        db_path = tmp_path / "gold.db"
        db_path.touch()
        mock_engine = MagicMock(return_value={"stale_steps": ["validated_setups"]})
        mock_asset_configs = MagicMock()
        mock_asset_configs.ACTIVE_ORB_INSTRUMENTS = ["MGC"]
        mock_asset_configs.DEPLOYABLE_ORB_INSTRUMENTS = []  # MGC not deployable
        mock_pipeline_status = MagicMock(staleness_engine=mock_engine)
        with (
            patch.dict(
                "sys.modules",
                {
                    "pipeline.asset_configs": mock_asset_configs,
                    "pipeline_status": mock_pipeline_status,
                },
            ),
            patch("duckdb.connect") as mock_connect,
        ):
            mock_con = MagicMock()
            mock_connect.return_value = mock_con
            items = collect_staleness(tmp_path, db_path)
        # Only stale step was validated_setups → research-only → filtered → no alert
        assert not any(i.source == "staleness" for i in items)

    def test_research_only_other_steps_still_alert(self, tmp_path: Path) -> None:
        """For a research-only instrument, non-validated_setups stale steps still
        surface — only the validated_setups entry is filtered."""
        db_path = tmp_path / "gold.db"
        db_path.touch()
        mock_engine = MagicMock(return_value={"stale_steps": ["validated_setups", "outcome_builder"]})
        mock_asset_configs = MagicMock()
        mock_asset_configs.ACTIVE_ORB_INSTRUMENTS = ["MGC"]
        mock_asset_configs.DEPLOYABLE_ORB_INSTRUMENTS = []
        mock_pipeline_status = MagicMock(staleness_engine=mock_engine)
        with (
            patch.dict(
                "sys.modules",
                {
                    "pipeline.asset_configs": mock_asset_configs,
                    "pipeline_status": mock_pipeline_status,
                },
            ),
            patch("duckdb.connect") as mock_connect,
        ):
            mock_con = MagicMock()
            mock_connect.return_value = mock_con
            items = collect_staleness(tmp_path, db_path)
        decaying = [i for i in items if i.source == "staleness"]
        assert len(decaying) == 1
        # outcome_builder survives the filter; validated_setups does not.
        assert "outcome_builder" in decaying[0].summary
        assert "validated_setups" not in decaying[0].summary

    def test_deployable_validated_setups_step_still_alerts(self, tmp_path: Path) -> None:
        """Deployable instruments still get the validated_setups alert — the
        filter is ONLY for research-only instruments."""
        db_path = tmp_path / "gold.db"
        db_path.touch()
        mock_engine = MagicMock(return_value={"stale_steps": ["validated_setups"]})
        mock_asset_configs = MagicMock()
        mock_asset_configs.ACTIVE_ORB_INSTRUMENTS = ["MES"]
        mock_asset_configs.DEPLOYABLE_ORB_INSTRUMENTS = ["MES"]
        mock_pipeline_status = MagicMock(staleness_engine=mock_engine)
        with (
            patch.dict(
                "sys.modules",
                {
                    "pipeline.asset_configs": mock_asset_configs,
                    "pipeline_status": mock_pipeline_status,
                },
            ),
            patch("duckdb.connect") as mock_connect,
        ):
            mock_con = MagicMock()
            mock_connect.return_value = mock_con
            items = collect_staleness(tmp_path, db_path)
        decaying = [i for i in items if i.source == "staleness"]
        assert len(decaying) == 1
        assert "validated_setups" in decaying[0].summary


# ---------------------------------------------------------------------------
# collect_fitness_fast — deployable vs active scoping
# ---------------------------------------------------------------------------


class TestCollectFitnessFastDeployable:
    """Ensure fitness_fast alerts iterate DEPLOYABLE_ORB_INSTRUMENTS, not ACTIVE."""

    def test_research_only_empty_shelf_no_alert(self, tmp_path: Path) -> None:
        """Research-only instrument with empty shelf must NOT produce a
        '0 active validated strategies' alert."""
        from scripts.tools.project_pulse import collect_fitness_fast

        db_path = tmp_path / "gold.db"
        db_path.touch()
        mock_asset_configs = MagicMock()
        mock_asset_configs.DEPLOYABLE_ORB_INSTRUMENTS = ["MES", "MNQ"]
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("MES", 2),
            ("MNQ", 27),
        ]
        with (
            patch.dict("sys.modules", {"pipeline.asset_configs": mock_asset_configs}),
            patch("duckdb.connect", return_value=mock_con),
            patch.object(project_pulse, "deployable_validated_relation", return_value="vs"),
        ):
            summary, items = collect_fitness_fast(db_path)
        # MES + MNQ both covered → no alerts. MGC not in DEPLOYABLE → not alerted.
        assert summary == {
            "MES": {"active_strategies": 2},
            "MNQ": {"active_strategies": 27},
        }
        assert items == []

    def test_deployable_instrument_empty_shelf_alerts(self, tmp_path: Path) -> None:
        """A deployable instrument with an empty shelf MUST alert."""
        from scripts.tools.project_pulse import collect_fitness_fast

        db_path = tmp_path / "gold.db"
        db_path.touch()
        mock_asset_configs = MagicMock()
        mock_asset_configs.DEPLOYABLE_ORB_INSTRUMENTS = ["MES", "MNQ"]
        mock_con = MagicMock()
        # Only MES has rows — MNQ is missing entirely
        mock_con.execute.return_value.fetchall.return_value = [("MES", 2)]
        with (
            patch.dict("sys.modules", {"pipeline.asset_configs": mock_asset_configs}),
            patch("duckdb.connect", return_value=mock_con),
            patch.object(project_pulse, "deployable_validated_relation", return_value="vs"),
        ):
            summary, items = collect_fitness_fast(db_path)
        assert summary == {"MES": {"active_strategies": 2}}
        assert len(items) == 1
        assert items[0].source == "fitness"
        assert "MNQ" in items[0].summary
        assert "0 active validated" in items[0].summary


# ---------------------------------------------------------------------------
# collect_action_queue
# ---------------------------------------------------------------------------


class TestCollectActionQueue:
    def test_parses_open_items(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "memory" / "MEMORY.md",
            "\n".join(
                [
                    "# Memory",
                    "## ACTION QUEUE",
                    "1. ~~**Full rebuild**~~ — DONE.",
                    "2. **CUSUM-based fitness** — Faster regime break detection.",
                    "3. **ATR-normalized sizing** — Carver approach. Scale with certainty.",
                    "## Other Section",
                ]
            ),
        )
        with patch.object(project_pulse, "_find_memory_md", return_value=tmp_path / "memory" / "MEMORY.md"):
            items = collect_action_queue(tmp_path)
        assert len(items) == 2  # item 1 is strikethrough (done)
        assert items[0].summary == "CUSUM-based fitness"
        assert items[1].summary == "ATR-normalized sizing"
        assert all(i.category == "ready" for i in items)

    def test_no_memory_file(self, tmp_path: Path) -> None:
        with patch.object(project_pulse, "_find_memory_md", return_value=None):
            items = collect_action_queue(tmp_path)
        assert items == []

    def test_truncates_long_items(self, tmp_path: Path) -> None:
        long_item = "A" * 100
        _mkfile(
            tmp_path / "memory" / "MEMORY.md",
            f"## ACTION QUEUE\n1. **{long_item}** — description\n## End",
        )
        with patch.object(project_pulse, "_find_memory_md", return_value=tmp_path / "memory" / "MEMORY.md"):
            items = collect_action_queue(tmp_path)
        assert len(items) == 1
        assert len(items[0].summary) <= 80


# ---------------------------------------------------------------------------
# collect_ralph_deferred
# ---------------------------------------------------------------------------


class TestCollectRalphDeferred:
    def test_parses_open_findings(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "docs" / "ralph-loop" / "deferred-findings.md",
            "\n".join(
                [
                    "# Deferred",
                    "## Open Findings",
                    "| ID | Iter | Severity | Target | Description | Reason |",
                    "|----|------|----------|--------|-------------|--------|",
                    "| DF-04 | 12 | LOW | rolling.py:304 | Dormant orb_minutes | annotated |",
                    "## Won't Fix",
                ]
            ),
        )
        items = collect_ralph_deferred(tmp_path)
        assert len(items) == 1
        assert items[0].source == "ralph"
        assert "DF-04" in items[0].summary
        assert items[0].severity == "low"

    def test_no_deferred_file(self, tmp_path: Path) -> None:
        items = collect_ralph_deferred(tmp_path)
        assert items == []

    def test_empty_open_findings(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "docs" / "ralph-loop" / "deferred-findings.md",
            "\n".join(
                [
                    "## Open Findings",
                    "| ID | Iter | Severity | Target | Description | Reason |",
                    "|----|------|----------|--------|-------------|--------|",
                    "## Won't Fix",
                ]
            ),
        )
        items = collect_ralph_deferred(tmp_path)
        assert items == []


# ---------------------------------------------------------------------------
# collect_worktrees
# ---------------------------------------------------------------------------


class TestCollectWorktrees:
    def test_detects_worktree(self, tmp_path: Path) -> None:
        meta = tmp_path / ".worktrees" / "claude" / "my-task" / ".canompx3-worktree.json"
        _mkfile(
            meta,
            json.dumps({"tool": "claude", "name": "my-task", "purpose": "Build", "branch": "wt-my-task"}),
        )
        items = collect_worktrees(tmp_path)
        assert len(items) == 1
        assert "my-task" in items[0].summary
        assert items[0].category == "paused"

    def test_no_worktrees_dir(self, tmp_path: Path) -> None:
        items = collect_worktrees(tmp_path)
        assert items == []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestExpensiveCache:
    def test_write_and_read(self, tmp_path: Path) -> None:
        drift = [PulseItem("broken", "high", "drift", "drift failed")]
        tests = [PulseItem("broken", "high", "tests", "test failed")]
        _write_expensive_cache(tmp_path, "abc123", drift, tests, None)

        result = _read_expensive_cache(tmp_path, "abc123")
        assert result is not None
        assert result["head"] == "abc123"
        assert len(result["drift_items"]) == 1
        assert len(result["test_items"]) == 1

    def test_cache_miss_on_head_change(self, tmp_path: Path) -> None:
        _write_expensive_cache(tmp_path, "abc123", [], [], None)
        result = _read_expensive_cache(tmp_path, "def456")
        assert result is None

    def test_merge_preserves_existing(self, tmp_path: Path) -> None:
        """Partial run preserves previously cached values."""
        drift = [PulseItem("broken", "high", "drift", "drift failed")]
        _write_expensive_cache(tmp_path, "abc123", drift, [], None)

        # Second run skips drift (None), runs tests
        existing = _read_expensive_cache(tmp_path, "abc123")
        test_items = [PulseItem("broken", "high", "tests", "test failed")]
        _write_expensive_cache(tmp_path, "abc123", None, test_items, existing)

        result = _read_expensive_cache(tmp_path, "abc123")
        assert result is not None
        # Drift preserved from first run
        assert len(result["drift_items"]) == 1
        assert result["drift_items"][0]["source"] == "drift"
        # Tests from second run
        assert len(result["test_items"]) == 1
        assert result["test_items"][0]["source"] == "tests"

    def test_corrupted_cache_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / ".pulse_cache.json").write_text("not json", encoding="utf-8")
        assert _read_expensive_cache(tmp_path, "abc") is None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _sample_report() -> PulseReport:
    return PulseReport(
        generated_at="2026-03-18T00:00:00+00:00",
        cache_hit=False,
        git_head="abc123",
        git_branch="main",
        system_identity={
            "canonical_repo_root": "/repo",
            "canonical_db_path": "/repo/gold.db",
            "selected_db_path": "/repo/gold.db",
            "db_override_active": False,
            "live_journal_db_path": "/repo/live_journal.db",
            "active_orb_instruments": ["MES", "MGC", "MNQ"],
            "authority_map_doc": "docs/governance/system_authority_map.md",
            "doctrine_docs": ["CLAUDE.md", "TRADING_RULES.md"],
            "backbone_modules": ["pipeline/system_authority.py", "pipeline/db_contracts.py"],
            "published_relations": {
                "active": "active_validated_setups",
                "deployable": "deployable_validated_setups",
            },
        },
        items=[
            PulseItem("broken", "high", "drift", "Drift FAILED"),
            PulseItem("decaying", "medium", "staleness", "MGC: 2 stale steps"),
            PulseItem("ready", "low", "action_queue", "CUSUM fitness"),
            PulseItem("paused", "low", "git", "3 uncommitted file(s)"),
        ],
        handoff_tool="Claude",
        handoff_date="2026-03-17",
        handoff_summary="Did work",
        handoff_next_steps=["Phase 1: build", "Phase 2: test"],
        fitness_summary={"MGC": {"active_strategies": 573}},
        deployment_summary={
            "profile_id": "topstep_50k_mnq_auto",
            "deployed_count": 5,
            "validated_active_count": 6,
            "validated_not_deployed": ["MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8"],
        },
        survival_summary={
            "gate_ok": True,
            "as_of_date": "2026-04-09",
            "report_age_days": 1,
            "operational_pass_probability": 0.8723,
        },
        sr_summary={
            "counts": {"CONTINUE": 5, "ALARM": 0, "NO_DATA": 0},
            "stream_counts": {"canonical_forward": 4, "paper_trades": 1},
            "state_age_days": 0,
        },
        pause_summary={"paused_count": 0},
        recommendation="Fix: Drift FAILED → /verify",
    )


class TestFormatText:
    def test_contains_all_sections(self) -> None:
        text = format_text(_sample_report())
        assert "PROJECT PULSE" in text
        assert "System identity:" in text
        assert "Live control:" in text
        assert "FIX NOW" in text
        assert "ACT SOON" in text
        assert "ON DECK" in text
        assert "PAUSED" in text
        assert "Strategy fitness" in text
        assert "MGC: 573 active" in text
        assert "Fix: Drift FAILED" in text  # recommendation line

    def test_all_clear_message(self) -> None:
        report = PulseReport(
            generated_at="now",
            cache_hit=False,
            git_head="abc",
            git_branch="main",
            recommendation="All clear — start new work",
        )
        text = format_text(report)
        assert "All clear" in text


class TestFormatJson:
    def test_valid_json(self) -> None:
        output = format_json(_sample_report())
        data = json.loads(output)
        assert data["system_identity"]["canonical_db_path"] == "/repo/gold.db"
        assert data["counts"]["broken"] == 1
        assert data["counts"]["decaying"] == 1
        assert data["handoff"]["tool"] == "Claude"
        assert data["deployment_summary"]["deployed_count"] == 5
        assert data["survival_summary"]["gate_ok"] is True
        assert data["sr_summary"]["counts"]["CONTINUE"] == 5
        assert len(data["items"]) == 4


class TestFormatMarkdown:
    def test_markdown_structure(self) -> None:
        md = format_markdown(_sample_report())
        assert md.startswith("# Project Pulse")
        assert "## System Identity" in md
        assert "## Live Control" in md
        assert "## FIX NOW" in md
        assert "## ACT SOON" in md
        assert "## Strategy Fitness" in md


class TestCollectSystemIdentity:
    def test_collects_linked_system_identity(self) -> None:
        root = Path("/repo/wt")
        canonical = Path("/repo")
        db_path = Path("/repo/gold.db")
        snapshot = MagicMock()
        snapshot.git.canonical_root = "/repo"
        snapshot.git.selected_root = "/repo/wt"
        snapshot.git.branch = "main"
        snapshot.git.head_sha = "abc123"
        snapshot.git.dirty_count = 0
        snapshot.git.in_linked_worktree = True
        snapshot.db.canonical_db_path = "/repo/gold.db"
        snapshot.db.selected_db_path = "/repo/gold.db"
        snapshot.db.db_override_active = False
        snapshot.db.live_journal_db_path = "/repo/live_journal.db"
        snapshot.authority.active_orb_instruments = ["MGC", "MNQ"]
        snapshot.authority.authority_map_doc = "docs/governance/system_authority_map.md"
        snapshot.authority.doctrine_docs = ["CLAUDE.md", "TRADING_RULES.md"]
        snapshot.authority.backbone_modules = ["pipeline/system_authority.py", "pipeline/system_context.py"]
        snapshot.authority.published_relations = {
            "active": "active_validated_setups",
            "deployable": "deployable_validated_setups",
        }
        snapshot.interpreter.context = "codex-wsl"
        snapshot.interpreter.current_python = "/repo/.venv-wsl/bin/python"
        snapshot.interpreter.current_prefix = "/repo/.venv-wsl"
        snapshot.interpreter.expected_python = "/repo/.venv-wsl/bin/python"
        snapshot.interpreter.expected_prefix = "/repo/.venv-wsl"
        snapshot.interpreter.matches_expected = True
        snapshot.active_stages = []
        snapshot.claims = []

        decision = MagicMock()
        decision.allowed = True
        decision.warnings = []
        decision.applicable_controls = ["pipeline/check_drift.py", "pipeline/system_context.py"]

        with (
            patch("pipeline.system_context.build_system_context", return_value=snapshot),
            patch("pipeline.system_context.evaluate_system_policy", return_value=decision),
        ):
            summary, items = collect_system_identity(root, canonical, db_path)

        assert items == []
        assert summary is not None
        assert summary["canonical_db_path"] == "/repo/gold.db"
        assert summary["active_orb_instruments"] == ["MGC", "MNQ"]
        assert summary["published_relations"]["deployable"] == "deployable_validated_setups"
        assert summary["authority_map_doc"] == "docs/governance/system_authority_map.md"
        assert summary["interpreter"]["current_prefix"] == "/repo/.venv-wsl"
        assert summary["interpreter"]["expected_prefix"] == "/repo/.venv-wsl"
        assert summary["interpreter"]["matches_expected"] is True
        assert summary["policy"]["allowed"] is True


# ---------------------------------------------------------------------------
# build_pulse integration (mocked externals)
# ---------------------------------------------------------------------------


class TestBuildPulse:
    def test_fast_mode_skips_drift_and_tests(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "## Last Session\n- **Tool:** Test\n- **Date:** today\n- **Summary:** test")
        _mkfile(tmp_path / ".git" / "HEAD", "ref: refs/heads/main")
        db_path = tmp_path / "gold.db"
        db_path.touch()

        with (
            patch.object(project_pulse, "_canonical_repo_root", return_value=tmp_path),
            patch.object(project_pulse, "_git_head", return_value="abc123"),
            patch.object(project_pulse, "_git_branch", return_value="main"),
            patch.object(project_pulse, "_run_git", return_value=MagicMock(returncode=0, stdout="")),
            patch.object(project_pulse, "collect_system_identity", return_value=({}, [])),
            patch.object(project_pulse, "collect_staleness", return_value=[]),
            patch.object(project_pulse, "collect_fitness_fast", return_value=({}, [])),
            patch.object(project_pulse, "collect_deployment_state", return_value=(None, [])),
            patch.object(project_pulse, "collect_lifecycle_control", return_value=(None, None, None, [])),
            patch.object(project_pulse, "collect_worktrees", return_value=[]),
            patch.object(project_pulse, "collect_session_claims", return_value=[]),
            patch.object(project_pulse, "collect_action_queue", return_value=[]),
            patch.object(project_pulse, "collect_ralph_deferred", return_value=[]),
            patch.object(project_pulse, "collect_drift") as mock_drift,
            patch.object(project_pulse, "collect_tests") as mock_tests,
        ):
            report = build_pulse(tmp_path, db_path=db_path, skip_drift=True, skip_tests=True)

        mock_drift.assert_not_called()
        mock_tests.assert_not_called()
        assert isinstance(report, PulseReport)


class TestDeploymentState:
    def test_detects_validated_but_not_deployed(self) -> None:
        db_path = Path("unused.db")
        mock_profile_module = MagicMock()
        mock_profile_module.resolve_profile_id.return_value = "topstep_50k_mnq_auto"
        mock_profile_module.get_profile_lane_definitions.return_value = [
            {"strategy_id": "A"},
            {"strategy_id": "B"},
        ]
        with (
            patch.dict("sys.modules", {"trading_app.prop_profiles": mock_profile_module}),
            patch("duckdb.connect") as mock_connect,
            patch.object(Path, "exists", return_value=True),
        ):
            mock_con = MagicMock()
            mock_con.execute.return_value.fetchall.return_value = [("A",), ("B",), ("C",)]
            mock_connect.return_value = mock_con
            summary, items = collect_deployment_state(db_path)

        assert summary is not None
        assert summary["validated_not_deployed"] == ["C"]
        assert any(i.source == "deployment" and i.category == "ready" for i in items)


class TestControlSummaries:
    def test_survival_state_pass_summary(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {
                "profile_id": "topstep_50k_mnq_auto",
                "gate_ok": True,
                "gate_msg": "Criterion 11 pass: operational 85.0%, as_of=2026-04-09, age=1d, paths=10000",
                "as_of_date": "2026-04-09",
                "generated_at_utc": "2026-04-09T22:39:14.876962+00:00",
                "report_age_days": 1,
                "operational_pass_probability": 0.85,
                "n_paths": 10000,
                "horizon_days": 90,
                "gate_pass": True,
            },
            "criterion12": {"available": True, "valid": True, "counts": {"ALARM": 0}, "state_age_days": 0},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            summary, items = collect_survival_state()

        assert summary is not None
        assert summary["gate_ok"] is True
        assert items == []

    def test_survival_state_block_recommends_control_state_refresh(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {
                "profile_id": "topstep_50k_mnq_auto",
                "gate_ok": False,
                "gate_msg": "BLOCKED: Criterion 11 state code fingerprint mismatch. Re-run account survival.",
                "reason": "code fingerprint mismatch",
                "report_age_days": 0,
            },
            "criterion12": {"available": True, "valid": True, "counts": {"ALARM": 0}, "state_age_days": 0},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            _summary, items = collect_survival_state()

        assert any(i.source == "criterion11" and "refresh_control_state.py" in str(i.action) for i in items)

    def test_sr_state_alarm_item(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {"gate_ok": True, "report_age_days": 0},
            "criterion12": {
                "profile_id": "topstep_50k_mnq_auto",
                "available": True,
                "valid": True,
                "state_date": "2026-04-10",
                "state_age_days": 0,
                "counts": {"ALARM": 1, "CONTINUE": 1},
                "stream_counts": {"paper_trades": 1, "canonical_forward": 1},
                "apply_pauses": False,
                "status_by_strategy": {"SID_A": "ALARM", "SID_B": "CONTINUE"},
                "alarm_strategy_ids": ["SID_A"],
                "no_data_strategy_ids": [],
            },
            "strategy_states": {
                "SID_A": {"sr_status": "ALARM", "sr_review_outcome": None, "paused": False},
                "SID_B": {"sr_status": "CONTINUE", "paused": False},
            },
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            summary, items = collect_sr_state(Path("/tmp/repo/gold.db"))

        assert summary is not None
        assert summary["counts"]["ALARM"] == 1
        assert summary["unresolved_alarm_count"] == 1
        assert any(i.source == "sr_monitor" and i.category == "decaying" for i in items)

    def test_sr_state_mismatch_degrades_without_trusting_payload(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {"gate_ok": True, "report_age_days": 0},
            "criterion12": {
                "profile_id": "topstep_50k_mnq_auto",
                "available": True,
                "valid": False,
                "reason": "profile fingerprint mismatch",
                "counts": {},
                "stream_counts": {},
                "status_by_strategy": {},
                "alarm_strategy_ids": [],
                "no_data_strategy_ids": [],
            },
            "strategy_states": {},
            "pauses": {"paused_count": 0, "paused_strategy_ids": []},
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            summary, items = collect_sr_state(Path("/tmp/repo/gold.db"))

        assert summary is None
        assert any(i.source == "sr_monitor" and i.category == "decaying" for i in items)
        assert any("mismatched/legacy" in i.summary for i in items)
        assert any("refresh_control_state.py" in str(i.action) for i in items if i.source == "sr_monitor")

    def test_pause_state_reports_paused(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {"gate_ok": True, "report_age_days": 0},
            "criterion12": {"available": True, "valid": True, "counts": {"ALARM": 0}, "state_age_days": 0},
            "strategy_states": {},
            "pauses": {
                "profile_id": "topstep_50k_mnq_auto",
                "paused_count": 2,
                "paused_strategy_ids": ["A", "B"],
            },
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            summary, items = collect_pause_state()
        assert summary is not None
        assert summary["paused_count"] == 2
        assert any(i.source == "pauses" and i.category == "paused" for i in items)

    def test_collect_lifecycle_control_returns_one_read_for_all_summaries(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {"gate_ok": True, "report_age_days": 0},
            "criterion12": {
                "available": True,
                "valid": True,
                "counts": {"ALARM": 1, "CONTINUE": 2},
                "state_age_days": 0,
            },
            "strategy_states": {
                "SID_A": {"sr_status": "ALARM", "sr_review_outcome": None, "paused": False},
                "SID_B": {"sr_status": "CONTINUE", "paused": False},
            },
            "pauses": {"profile_id": "topstep_50k_mnq_auto", "paused_count": 1, "paused_strategy_ids": ["SID_A"]},
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            survival, sr, pauses, items = collect_lifecycle_control(Path("/tmp/repo/gold.db"))

        assert survival is not None
        assert sr is not None
        assert pauses is not None
        mock_lifecycle.read_lifecycle_state.assert_called_once()
        assert any(i.source == "sr_monitor" for i in items)
        assert any(i.source == "pauses" for i in items)

    def test_reviewed_watch_alarm_is_summarized_but_not_actionable(self) -> None:
        mock_lifecycle = MagicMock()
        mock_lifecycle.read_lifecycle_state.return_value = {
            "criterion11": {"gate_ok": True, "report_age_days": 0},
            "criterion12": {
                "available": True,
                "valid": True,
                "counts": {"ALARM": 1, "CONTINUE": 2},
                "state_age_days": 0,
                "stream_counts": {"canonical_forward": 3},
            },
            "strategy_states": {
                "SID_A": {"sr_status": "ALARM", "sr_review_outcome": "watch", "paused": False},
                "SID_B": {"sr_status": "CONTINUE", "paused": False},
            },
            "pauses": {"profile_id": "topstep_50k_mnq_auto", "paused_count": 0, "paused_strategy_ids": []},
        }
        with patch.dict("sys.modules", {"trading_app.lifecycle_state": mock_lifecycle}):
            summary, items = collect_sr_state(Path("/tmp/repo/gold.db"))

        assert summary is not None
        assert summary["reviewed_watch_count"] == 1
        assert summary["unresolved_alarm_count"] == 0
        assert items == []


# ---------------------------------------------------------------------------
# v2 features: recommendation, momentum, conflicts, skill suggestions
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_broken_takes_priority(self) -> None:
        from scripts.tools.project_pulse import _compute_recommendation

        report = PulseReport(
            generated_at="now",
            cache_hit=False,
            git_head="abc",
            git_branch="main",
            items=[
                PulseItem("broken", "high", "drift", "Drift FAILED", action="/verify"),
                PulseItem("ready", "low", "action_queue", "CUSUM fitness"),
            ],
            recommendation="",
        )
        rec = _compute_recommendation(report)
        assert rec.startswith("Fix:")
        assert "/verify" in rec

    def test_runtime_snapshot_refresh_beats_upcoming_session(self) -> None:
        from scripts.tools.project_pulse import _compute_recommendation

        report = PulseReport(
            generated_at="now",
            cache_hit=False,
            git_head="abc",
            git_branch="main",
            items=[
                PulseItem(
                    "paused",
                    "medium",
                    "runtime_snapshot",
                    "Runtime snapshot missing or stale",
                    action="python scripts/tools/refresh_runtime_snapshot.py",
                )
            ],
            upcoming_sessions=[{"label": "TOKYO_OPEN", "hours_away": 1.5, "instruments": {"MGC": 3}}],
        )
        rec = _compute_recommendation(report)
        assert rec.startswith("Refresh:")
        assert "refresh_runtime_snapshot.py" in rec

    def test_upcoming_session_before_decay(self) -> None:
        from scripts.tools.project_pulse import _compute_recommendation

        report = PulseReport(
            generated_at="now",
            cache_hit=False,
            git_head="abc",
            git_branch="main",
            items=[PulseItem("decaying", "low", "fitness", "MGC: 5 WATCH")],
            upcoming_sessions=[{"label": "TOKYO_OPEN", "hours_away": 1.5, "instruments": {"MGC": 3}}],
        )
        rec = _compute_recommendation(report)
        assert "TOKYO_OPEN" in rec
        assert "/trade-book" in rec

    def test_all_clear(self) -> None:
        from scripts.tools.project_pulse import _compute_recommendation

        report = PulseReport(generated_at="now", cache_hit=False, git_head="abc", git_branch="main")
        rec = _compute_recommendation(report)
        assert "All clear" in rec


class TestSkillSuggestions:
    def test_attaches_action_to_staleness(self) -> None:
        from scripts.tools.project_pulse import _attach_skill_suggestions

        items = [PulseItem("decaying", "medium", "staleness", "MGC: 2 stale steps")]
        _attach_skill_suggestions(items)
        assert items[0].action == "/rebuild-outcomes MGC"

    def test_attaches_action_to_drift(self) -> None:
        from scripts.tools.project_pulse import _attach_skill_suggestions

        items = [PulseItem("broken", "high", "drift", "Drift FAILED")]
        _attach_skill_suggestions(items)
        assert items[0].action == "/verify"

    def test_preserves_existing_action(self) -> None:
        from scripts.tools.project_pulse import _attach_skill_suggestions

        items = [PulseItem("broken", "high", "drift", "Drift FAILED", action="custom")]
        _attach_skill_suggestions(items)
        assert items[0].action == "custom"


class TestWorkstreamMomentum:
    def test_stalled_detection(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import _workstream_momentum

        data = {"branch": "wt-old-task", "created_at": "2026-03-10T00:00:00+00:00"}
        with patch.object(project_pulse, "_run_git", return_value=MagicMock(returncode=0, stdout="0")):
            result = _workstream_momentum(data, tmp_path)
        assert "STALLED" in result

    def test_active_workstream(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import _workstream_momentum

        data = {"branch": "wt-active", "created_at": "2026-03-16T00:00:00+00:00"}
        with patch.object(project_pulse, "_run_git", return_value=MagicMock(returncode=0, stdout="5")):
            result = _workstream_momentum(data, tmp_path)
        assert "5 commit(s)" in result
        assert "STALLED" not in result


class TestWorktreeConflicts:
    def test_detects_overlap(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import collect_worktree_conflicts

        wt_a = tmp_path / ".worktrees" / "claude" / "task-a"
        wt_b = tmp_path / ".worktrees" / "claude" / "task-b"
        _mkfile(wt_a / ".canompx3-worktree.json", json.dumps({"branch": "wt-a", "name": "task-a"}))
        _mkfile(wt_b / ".canompx3-worktree.json", json.dumps({"branch": "wt-b", "name": "task-b"}))

        def mock_git(root, *args):
            # args is ("diff", "--name-only", "main...wt-a") or similar
            joined = " ".join(str(a) for a in args)
            if "wt-a" in joined:
                return MagicMock(returncode=0, stdout="config.py\nutils.py\n")
            if "wt-b" in joined:
                return MagicMock(returncode=0, stdout="config.py\nrouter.py\n")
            return MagicMock(returncode=0, stdout="")

        with patch.object(project_pulse, "_run_git", side_effect=mock_git):
            items = collect_worktree_conflicts(tmp_path)
        assert len(items) == 1
        assert "config.py" in items[0].summary
        assert items[0].category == "decaying"

    def test_no_overlap(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import collect_worktree_conflicts

        wt_a = tmp_path / ".worktrees" / "claude" / "task-a"
        _mkfile(wt_a / ".canompx3-worktree.json", json.dumps({"branch": "wt-a", "name": "task-a"}))

        with patch.object(project_pulse, "_run_git", return_value=MagicMock(returncode=0, stdout="foo.py\n")):
            items = collect_worktree_conflicts(tmp_path)
        assert items == []  # only 1 worktree, no overlap possible


class TestTimeSinceGreen:
    def test_records_green(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import _update_time_since_green

        result = _update_time_since_green(tmp_path, is_green=True)
        assert result == "now"
        # Cache file should have last_green
        cache = json.loads((tmp_path / ".pulse_cache.json").read_text(encoding="utf-8"))
        assert "last_green" in cache

    def test_reports_time_since(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime, timedelta

        from scripts.tools.project_pulse import CACHE_FILE, _update_time_since_green

        # Write a green timestamp 2 hours ago
        two_hours_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        (tmp_path / CACHE_FILE).write_text(json.dumps({"last_green": two_hours_ago}), encoding="utf-8")
        result = _update_time_since_green(tmp_path, is_green=False)
        assert result is not None
        assert "h ago" in result


class TestFormatJsonV2:
    def test_includes_v2_fields(self) -> None:
        report = _sample_report()
        report.upcoming_sessions = [{"label": "TOKYO", "hours_away": 2}]
        report.time_since_green = "3h ago"
        report.session_delta = ["Since last: 2 commits"]
        output = format_json(report)
        data = json.loads(output)
        assert data["recommendation"] is not None
        assert data["upcoming_sessions"][0]["label"] == "TOKYO"
        assert data["time_since_green"] == "3h ago"
        assert data["deployment_summary"]["profile_id"] == "topstep_50k_mnq_auto"
        assert "Since last" in data["session_delta"][0]


# ---------------------------------------------------------------------------
# collect_session_delta, collect_upcoming_sessions, _resolve_db_path
# ---------------------------------------------------------------------------


class TestSessionDelta:
    def test_first_run_creates_marker(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import collect_session_delta

        with patch.object(project_pulse, "_git_head", return_value="abc123"):
            lines = collect_session_delta(tmp_path, tmp_path, tool_name="codex")
        # First run — no prior marker, so no delta
        assert lines == []
        # But marker was written
        marker = tmp_path / ".pulse_last_session.json"
        assert marker.exists()
        data = json.loads(marker.read_text(encoding="utf-8"))
        assert data["head"] == "abc123"
        assert data["tool"] == "codex"

    def test_detects_new_commits(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import collect_session_delta

        # Write a prior marker with old HEAD
        marker = tmp_path / ".pulse_last_session.json"
        marker.write_text(
            json.dumps({"head": "old123", "tool": "codex", "at": "2026-03-17T00:00:00+00:00"}),
            encoding="utf-8",
        )

        def mock_git(root, *args):
            joined = " ".join(str(a) for a in args)
            if "rev-parse" in joined:
                return MagicMock(returncode=0, stdout="new456")
            if "log" in joined and "old123" in joined:
                return MagicMock(returncode=0, stdout="new456 feat: something\nabc789 fix: other\n")
            return MagicMock(returncode=0, stdout="new456")

        with patch.object(project_pulse, "_run_git", side_effect=mock_git):
            with patch.object(project_pulse, "_git_head", return_value="new456"):
                lines = collect_session_delta(tmp_path, tmp_path)
        assert len(lines) > 0
        assert "codex" in lines[0]  # shows which tool had last session


class TestResolveDbPath:
    def test_canonical_gold_db(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import _resolve_db_path

        db = tmp_path / "gold.db"
        db.touch()
        result = _resolve_db_path(tmp_path, tmp_path)
        assert result == db

    def test_fallback_to_root(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import _resolve_db_path

        canonical = tmp_path / "canonical"
        canonical.mkdir()
        # No gold.db anywhere — falls back to root/gold.db
        with patch.dict("sys.modules", {"pipeline": None, "pipeline.paths": None}):
            result = _resolve_db_path(tmp_path, canonical)
        assert result.name == "gold.db"


class TestUpcomingSessions:
    def test_graceful_on_import_failure(self, tmp_path: Path) -> None:
        from scripts.tools.project_pulse import collect_upcoming_sessions

        # When pipeline isn't importable, should return empty list
        with patch.dict("sys.modules", {"pipeline.dst": None}):
            result = collect_upcoming_sessions(tmp_path / "nonexistent.db")
        assert result == []


class TestCliBootstrap:
    def test_script_help_runs_via_direct_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [sys.executable, "scripts/tools/project_pulse.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Project pulse" in result.stdout
