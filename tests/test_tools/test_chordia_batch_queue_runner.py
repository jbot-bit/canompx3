from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.tools import chordia_batch_queue_runner as runner
from scripts.tools.chordia_batch_cycle import BatchCycleResult


def _cycle_result(*, idx: int, applied: int = 1, run_count: int = 1) -> BatchCycleResult:
    return BatchCycleResult(
        factory_dir=f"factory_{idx}",
        batch_id="batch_001",
        output_dir=f"artifacts/research/cycle_{idx}",
        plan_count=1,
        bridge_status_counts={"READY_TO_ACTIVATE": 1},
        activated_count=1,
        run_count=run_count,
        run_returncode_counts={0: run_count},
        parsed_result_count=run_count,
        applied_count=applied,
        skipped_existing_count=0,
        refreshed_fast_lane_status=True,
        refreshed_bench_dir=f"artifacts/research/bench_{idx}",
        refreshed_factory_dir=f"artifacts/research/factory_next_{idx}",
    )


def _args(tmp_path: Path, *, max_batches: int = 2) -> Namespace:
    return runner.parse_args(
        [
            "--initial-factory-dir",
            str(tmp_path / "factory_0"),
            "--run-label",
            "test_queue",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--max-batches",
            str(max_batches),
            "--activate",
            "--run",
            "--apply-reviewed",
            "--refresh-fast-lane-status",
            "--family-hints-csv",
            str(tmp_path / "hints.csv"),
            "--today",
            "2026-05-31",
            "--rebalance-date",
            "2026-05-31",
        ]
    )


def test_queue_runner_chains_refreshed_factory_until_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Namespace] = []

    monkeypatch.setattr(runner, "_batch_ready_count", lambda factory_dir, batch_id: 1)

    def fake_run_cycle(args: Namespace) -> BatchCycleResult:
        calls.append(args)
        return _cycle_result(idx=len(calls), applied=2, run_count=3)

    monkeypatch.setattr(runner, "run_cycle", fake_run_cycle)

    result = runner.run_queue(_args(tmp_path, max_batches=2))

    assert result.cycles_run == 2
    assert result.stopped_reason == "MAX_BATCHES_REACHED"
    assert result.total_run_count == 6
    assert result.total_applied_count == 4
    assert result.final_factory_dir == "artifacts/research/factory_next_2"
    assert calls[0].factory_dir == tmp_path / "factory_0"
    assert calls[1].factory_dir == Path("artifacts/research/factory_next_1")
    assert calls[0].output_dir != calls[1].output_dir


def test_queue_runner_stops_without_running_when_no_ready_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_batch_ready_count", lambda factory_dir, batch_id: 0)
    monkeypatch.setattr(
        runner,
        "run_cycle",
        lambda args: (_ for _ in ()).throw(AssertionError("run_cycle should not be called")),
    )

    result = runner.run_queue(_args(tmp_path, max_batches=3))

    assert result.cycles_run == 0
    assert result.stopped_reason == "NO_READY_WORK"
    assert result.total_run_count == 0
    assert result.final_factory_dir == str(tmp_path / "factory_0").replace("\\", "/")


def test_queue_runner_rejects_non_positive_max_batches(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _args(tmp_path, max_batches=0)


def test_queue_runner_script_help_runs_from_file_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/tools/chordia_batch_queue_runner.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--initial-factory-dir" in result.stdout
