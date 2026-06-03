from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools import chordia_batch_cycle as cycle
from scripts.tools.chordia_evidence_factory import ParsedResult
from scripts.tools.chordia_replay_batch_bridge import BridgePlanRow, ReplayRunResult


def _audit_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "default_has_theory: false",
                "audit_freshness_days: 90",
                "theory_grants: []",
                "audits:",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _plan() -> tuple[BridgePlanRow, ...]:
    return (
        BridgePlanRow(
            rank=1,
            strategy_id="MNQ_TEST_E2_RR1.0_CB1_NO_FILTER",
            factory_status="PREREG_DRAFT_READY",
            bridge_status="READY_TO_ACTIVATE",
            source_draft_path="artifacts/research/factory/prereg_drafts/test.draft.yaml",
            active_hypothesis_path="docs/audit/hypotheses/test.yaml",
            result_md_path="docs/audit/results/test.md",
            runner_command="python research/chordia_strict_unlock_v1.py --hypothesis-file docs/audit/hypotheses/test.yaml",
            note="Ready",
        ),
    )


def test_cycle_requires_fresh_run_before_apply_reviewed(tmp_path: Path) -> None:
    args = cycle.parse_args(
        [
            "--factory-dir",
            str(tmp_path),
            "--batch-id",
            "batch_001",
            "--apply-reviewed",
        ]
    )

    with pytest.raises(ValueError, match="requires --run"):
        cycle.run_cycle(args)


def test_cycle_runs_bridge_and_applies_reviewed_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    _audit_log(audit_log)
    output_dir = tmp_path / "cycle"

    monkeypatch.setattr(cycle, "plan_acceptance", lambda **_: _plan())
    monkeypatch.setattr(cycle, "activate_ready_drafts", lambda plan: plan)
    monkeypatch.setattr(
        cycle,
        "run_strict_replays",
        lambda plan, max_runs=None: (
            ReplayRunResult(
                strategy_id="MNQ_TEST_E2_RR1.0_CB1_NO_FILTER",
                hypothesis_path="docs/audit/hypotheses/test.yaml",
                result_md_path="docs/audit/results/test.md",
                returncode=0,
                stdout_tail="ok",
                stderr_tail="",
            ),
        ),
    )
    monkeypatch.setattr(
        cycle,
        "parse_successful_results",
        lambda plan, runs: (
            ParsedResult(
                path=tmp_path / "test.md",
                strategy_id="MNQ_TEST_E2_RR1.0_CB1_NO_FILTER",
                verdict="PASS_CHORDIA",
                threshold=3.79,
                has_theory=False,
                t_stat=4.1,
                sample_size=1000,
                exp_r=0.1,
            ),
        ),
    )

    args = cycle.parse_args(
        [
            "--factory-dir",
            str(tmp_path / "factory"),
            "--batch-id",
            "batch_001",
            "--output-dir",
            str(output_dir),
            "--activate",
            "--run",
            "--apply-reviewed",
            "--audit-log",
            str(audit_log),
            "--reviewed-by",
            "test",
        ]
    )

    result = cycle.run_cycle(args)

    assert result.plan_count == 1
    assert result.activated_count == 1
    assert result.run_count == 1
    assert result.applied_count == 1
    assert result.live_mutation is False
    assert result.validated_setups_mutation is False
    assert "strategy_id: MNQ_TEST_E2_RR1.0_CB1_NO_FILTER" in audit_log.read_text(encoding="utf-8")
    assert (output_dir / "audit_log_proposal.yaml").is_file()


def test_cycle_refuses_to_apply_when_any_replay_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    _audit_log(audit_log)

    monkeypatch.setattr(cycle, "plan_acceptance", lambda **_: _plan())
    monkeypatch.setattr(cycle, "activate_ready_drafts", lambda plan: plan)
    monkeypatch.setattr(
        cycle,
        "run_strict_replays",
        lambda plan, max_runs=None: (
            ReplayRunResult(
                strategy_id="MNQ_TEST_E2_RR1.0_CB1_NO_FILTER",
                hypothesis_path="docs/audit/hypotheses/test.yaml",
                result_md_path="docs/audit/results/test.md",
                returncode=2,
                stdout_tail="",
                stderr_tail="refused",
            ),
        ),
    )
    monkeypatch.setattr(cycle, "parse_successful_results", lambda plan, runs: ())

    args = cycle.parse_args(
        [
            "--factory-dir",
            str(tmp_path / "factory"),
            "--batch-id",
            "batch_001",
            "--output-dir",
            str(tmp_path / "cycle"),
            "--activate",
            "--run",
            "--apply-reviewed",
            "--audit-log",
            str(audit_log),
        ]
    )

    with pytest.raises(RuntimeError, match="refusing to apply reviewed audit rows"):
        cycle.run_cycle(args)

    assert "MNQ_TEST_E2_RR1.0_CB1_NO_FILTER" not in audit_log.read_text(encoding="utf-8")


def test_cycle_refreshes_derived_surfaces_after_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    _audit_log(audit_log)
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(cycle, "plan_acceptance", lambda **_: ())
    monkeypatch.setattr(cycle, "activate_ready_drafts", lambda plan: ())
    monkeypatch.setattr(cycle, "run_strict_replays", lambda plan, max_runs=None: ())
    monkeypatch.setattr(cycle, "parse_successful_results", lambda plan, runs: ())
    monkeypatch.setattr(cycle, "fast_lane_status_main", lambda argv: calls.append(("status", list(argv))) or 0)
    monkeypatch.setattr(cycle, "lane_bench_state_main", lambda argv: calls.append(("bench", list(argv))) or 0)
    monkeypatch.setattr(cycle, "evidence_factory_main", lambda argv: calls.append(("factory", list(argv))) or 0)

    bench_dir = tmp_path / "bench"
    factory_dir = tmp_path / "factory_next"
    args = cycle.parse_args(
        [
            "--factory-dir",
            str(tmp_path / "factory"),
            "--batch-id",
            "batch_002",
            "--output-dir",
            str(tmp_path / "cycle"),
            "--run",
            "--refresh-fast-lane-status",
            "--refresh-bench-output-dir",
            str(bench_dir),
            "--family-hints-csv",
            "artifacts/research/lane_bench_state_2026_05_30/bench.csv",
            "--refresh-factory-output-dir",
            str(factory_dir),
            "--factory-limit",
            "0",
            "--max-family-priority",
            "5",
            "--audit-log",
            str(audit_log),
        ]
    )

    result = cycle.run_cycle(args)

    assert result.refreshed_fast_lane_status is True
    assert result.refreshed_bench_dir is not None
    assert result.refreshed_factory_dir is not None
    assert [name for name, _ in calls] == ["status", "bench", "factory"]
    assert ["--write"] in [argv for _, argv in calls]
