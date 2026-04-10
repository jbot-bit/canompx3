"""Tests for scripts.tools.project_pulse."""

from __future__ import annotations

import json
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
    collect_pause_state,
    collect_ralph_deferred,
    collect_session_claims,
    collect_sr_state,
    collect_staleness,
    collect_survival_state,
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
        assert "## Live Control" in md
        assert "## FIX NOW" in md
        assert "## ACT SOON" in md
        assert "## Strategy Fitness" in md


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
            patch.object(project_pulse, "collect_staleness", return_value=[]),
            patch.object(project_pulse, "collect_fitness_fast", return_value=({}, [])),
            patch.object(project_pulse, "collect_deployment_state", return_value=(None, [])),
            patch.object(project_pulse, "collect_survival_state", return_value=(None, [])),
            patch.object(project_pulse, "collect_sr_state", return_value=(None, [])),
            patch.object(project_pulse, "collect_pause_state", return_value=(None, [])),
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
        mock_profile_module = MagicMock()
        mock_profile_module.resolve_profile_id.return_value = "topstep_50k_mnq_auto"
        mock_survival_module = MagicMock()
        mock_survival_module.check_survival_report_gate.return_value = (
            True,
            "Criterion 11 pass: operational 87.2%, as_of=2026-04-09, age=1d, paths=10000",
        )
        report_payload = {
            "summary": {
                "generated_at_utc": "2026-04-09T22:39:14.876962+00:00",
                "as_of_date": "2026-04-09",
                "operational_pass_probability": 0.8723,
                "n_paths": 10000,
                "horizon_days": 90,
                "gate_pass": True,
            }
        }
        with (
            patch.dict(
                "sys.modules",
                {
                    "trading_app.prop_profiles": mock_profile_module,
                    "trading_app.account_survival": mock_survival_module,
                },
            ),
            patch.object(project_pulse, "PROJECT_ROOT", Path("/tmp/repo")),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(report_payload)),
        ):
            mock_survival_module.get_survival_report_path.return_value = Path("/tmp/repo/data/state/report.json")
            summary, items = collect_survival_state()

        assert summary is not None
        assert summary["gate_ok"] is True
        assert items == []

    def test_sr_state_alarm_item(self) -> None:
        mock_profile_module = MagicMock()
        mock_profile_module.resolve_profile_id.return_value = "topstep_50k_mnq_auto"
        mock_profile_module.get_profile.return_value = object()
        mock_profile_module.get_profile_lane_definitions.return_value = [
            {"strategy_id": "SID_A"},
            {"strategy_id": "SID_B"},
        ]
        mock_derived_state = MagicMock()
        mock_derived_state.build_profile_fingerprint.return_value = "pfp"
        mock_derived_state.build_db_identity.return_value = "dbid"
        mock_derived_state.build_code_fingerprint.return_value = "codeid"
        mock_derived_state.validate_state_envelope.return_value = (
            True,
            None,
            {
                "canonical_inputs": {"profile_id": "topstep_50k_mnq_auto"},
                "freshness": {"as_of_date": "2026-04-10", "max_age_days": 2},
                "payload": {
                    "apply_pauses": False,
                    "results": [
                        {"status": "ALARM", "stream_source": "paper_trades"},
                        {"status": "CONTINUE", "stream_source": "canonical_forward"},
                    ],
                },
            },
        )
        payload = {
            "schema_version": 1,
            "state_type": "sr_monitor",
            "generated_at_utc": "2026-04-10T00:00:00+00:00",
            "git_head": "abc123",
            "tool": "sr_monitor",
            "canonical_inputs": {},
            "freshness": {},
            "payload": {},
        }
        with (
            patch.dict(
                "sys.modules",
                {
                    "trading_app.prop_profiles": mock_profile_module,
                    "trading_app.derived_state": mock_derived_state,
                },
            ),
            patch.object(project_pulse, "PROJECT_ROOT", Path("/tmp/repo")),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(payload)),
        ):
            summary, items = collect_sr_state(Path("/tmp/repo/gold.db"))

        assert summary is not None
        assert summary["counts"]["ALARM"] == 1
        assert any(i.source == "sr_monitor" and i.category == "decaying" for i in items)

    def test_sr_state_mismatch_degrades_without_trusting_payload(self) -> None:
        mock_profile_module = MagicMock()
        mock_profile_module.resolve_profile_id.return_value = "topstep_50k_mnq_auto"
        mock_profile_module.get_profile.return_value = object()
        mock_profile_module.get_profile_lane_definitions.return_value = [{"strategy_id": "SID_A"}]
        mock_derived_state = MagicMock()
        mock_derived_state.build_profile_fingerprint.return_value = "pfp"
        mock_derived_state.build_db_identity.return_value = "dbid"
        mock_derived_state.build_code_fingerprint.return_value = "codeid"
        mock_derived_state.validate_state_envelope.return_value = (False, "profile_fingerprint_mismatch", None)

        with (
            patch.dict(
                "sys.modules",
                {
                    "trading_app.prop_profiles": mock_profile_module,
                    "trading_app.derived_state": mock_derived_state,
                },
            ),
            patch.object(project_pulse, "PROJECT_ROOT", Path("/tmp/repo")),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps({"legacy": True})),
        ):
            summary, items = collect_sr_state(Path("/tmp/repo/gold.db"))

        assert summary is None
        assert any(i.source == "sr_monitor" and i.category == "decaying" for i in items)
        assert any("mismatched/legacy" in i.summary for i in items)

    def test_pause_state_reports_paused(self) -> None:
        mock_profile_module = MagicMock()
        mock_profile_module.resolve_profile_id.return_value = "topstep_50k_mnq_auto"
        mock_lane_ctl = MagicMock()
        mock_lane_ctl.get_paused_strategy_ids.return_value = {"A", "B"}
        with patch.dict(
            "sys.modules",
            {
                "trading_app.prop_profiles": mock_profile_module,
                "trading_app.lane_ctl": mock_lane_ctl,
            },
        ):
            summary, items = collect_pause_state()
        assert summary is not None
        assert summary["paused_count"] == 2
        assert any(i.source == "pauses" and i.category == "paused" for i in items)


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
