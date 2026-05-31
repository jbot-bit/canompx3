from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.tools import bounded_benchmark_harness


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["trading_day", "baseline", "pnl_r", "selection_date"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_bounded_benchmark_requires_fixed_baseline_set(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.csv"
    _write_rows(
        source,
        [
            {"trading_day": "2025-12-01", "baseline": "naive", "pnl_r": "1.0", "selection_date": "2025-01-01"},
            {"trading_day": "2025-12-02", "baseline": "core_lane", "pnl_r": "0.5", "selection_date": "2025-01-01"},
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        bounded_benchmark_harness.build_bounded_benchmark_report(
            input_csv=source,
            train_end="2025-12-31",
            holdout_start="2026-01-01",
            cost_model="fixed_commission_slippage_v1",
        )

    assert "missing fixed baseline(s): trend" in str(excinfo.value)


def test_bounded_benchmark_rejects_holdout_selection_leakage(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.csv"
    _write_rows(
        source,
        [
            {"trading_day": "2025-12-01", "baseline": "naive", "pnl_r": "1.0", "selection_date": "2025-01-01"},
            {"trading_day": "2025-12-01", "baseline": "trend", "pnl_r": "0.3", "selection_date": "2025-01-01"},
            {"trading_day": "2025-12-01", "baseline": "core_lane", "pnl_r": "0.5", "selection_date": "2026-01-10"},
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        bounded_benchmark_harness.build_bounded_benchmark_report(
            input_csv=source,
            train_end="2025-12-31",
            holdout_start="2026-01-01",
            cost_model="fixed_commission_slippage_v1",
        )

    assert "holdout selection leakage" in str(excinfo.value)


def test_bounded_benchmark_writes_json_and_markdown_with_failed_controls(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.csv"
    _write_rows(
        source,
        [
            {"trading_day": "2025-12-01", "baseline": "naive", "pnl_r": "1.0", "selection_date": "2025-01-01"},
            {"trading_day": "2026-01-02", "baseline": "naive", "pnl_r": "-0.2", "selection_date": "2025-01-01"},
            {"trading_day": "2025-12-01", "baseline": "trend", "pnl_r": "0.3", "selection_date": "2025-01-01"},
            {"trading_day": "2026-01-02", "baseline": "trend", "pnl_r": "0.4", "selection_date": "2025-01-01"},
            {"trading_day": "2025-12-01", "baseline": "core_lane", "pnl_r": "0.5", "selection_date": "2025-01-01"},
            {"trading_day": "2026-01-02", "baseline": "core_lane", "pnl_r": "0.1", "selection_date": "2025-01-01"},
        ],
    )

    report = bounded_benchmark_harness.build_bounded_benchmark_report(
        input_csv=source,
        train_end="2025-12-31",
        holdout_start="2026-01-01",
        cost_model="fixed_commission_slippage_v1",
    )
    paths = bounded_benchmark_harness.write_bounded_benchmark_artifacts(report, tmp_path / "out")

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["controls"]["fixed_baselines"]["passed"] is True
    assert payload["controls"]["holdout_leakage_guard"]["passed"] is True
    assert payload["controls"]["failed_controls"] == []
    assert payload["baselines"]["core_lane"]["holdout"]["n"] == 1
    assert "Bounded Benchmark Harness" in paths["markdown"].read_text(encoding="utf-8")
