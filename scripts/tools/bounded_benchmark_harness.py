#!/usr/bin/env python3
"""Bounded benchmark harness for EV-3 proof artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

FIXED_BASELINES = ("naive", "trend", "core_lane")


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"invalid {field}: {value}") from exc


def _read_rows(input_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"trading_day", "baseline", "pnl_r", "selection_date"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"benchmark input missing column(s): {', '.join(sorted(missing))}")
        for index, row in enumerate(reader, start=2):
            baseline = str(row["baseline"]).strip()
            if baseline not in FIXED_BASELINES:
                raise SystemExit(f"row {index}: unsupported baseline: {baseline}")
            rows.append(
                {
                    "trading_day": _parse_date(str(row["trading_day"]).strip(), "trading_day"),
                    "baseline": baseline,
                    "pnl_r": float(row["pnl_r"]),
                    "selection_date": _parse_date(str(row["selection_date"]).strip(), "selection_date"),
                }
            )
    return rows


def _metrics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "total_r": 0.0, "mean_r": None, "win_rate": None}
    wins = sum(1 for value in values if value > 0)
    return {
        "n": len(values),
        "total_r": round(sum(values), 6),
        "mean_r": round(sum(values) / len(values), 6),
        "win_rate": round(wins / len(values), 6),
    }


def _baseline_metrics(rows: list[dict[str, Any]], train_end: date, holdout_start: date) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"train": [], "holdout": [], "all": []})
    for row in rows:
        bucket = "holdout" if row["trading_day"] >= holdout_start else "train"
        if train_end < row["trading_day"] < holdout_start:
            bucket = "gap"
        if bucket != "gap":
            grouped[row["baseline"]][bucket].append(row["pnl_r"])
        grouped[row["baseline"]]["all"].append(row["pnl_r"])
    return {
        baseline: {
            "train": _metrics(grouped[baseline]["train"]),
            "holdout": _metrics(grouped[baseline]["holdout"]),
            "all": _metrics(grouped[baseline]["all"]),
        }
        for baseline in FIXED_BASELINES
    }


def build_bounded_benchmark_report(
    *,
    input_csv: Path,
    train_end: str,
    holdout_start: str,
    cost_model: str,
) -> dict[str, Any]:
    train_end_date = _parse_date(train_end, "train_end")
    holdout_start_date = _parse_date(holdout_start, "holdout_start")
    if holdout_start_date <= train_end_date:
        raise SystemExit("holdout_start must be after train_end")

    rows = _read_rows(input_csv)
    present = {row["baseline"] for row in rows}
    missing = [baseline for baseline in FIXED_BASELINES if baseline not in present]
    if missing:
        raise SystemExit(f"missing fixed baseline(s): {', '.join(missing)}")

    leakage_rows = [
        row for row in rows if row["baseline"] == "core_lane" and row["selection_date"] >= holdout_start_date
    ]
    if leakage_rows:
        raise SystemExit(f"holdout selection leakage: {len(leakage_rows)} core_lane row(s)")

    controls = {
        "fixed_baselines": {"passed": True, "required": list(FIXED_BASELINES)},
        "fixed_split": {
            "passed": True,
            "train_end": train_end,
            "holdout_start": holdout_start,
        },
        "cost_model": {"passed": bool(cost_model), "id": cost_model},
        "no_parameter_rescue": {"passed": True, "note": "no optimizer or parameter override is exposed"},
        "holdout_leakage_guard": {"passed": True, "selection_date_lt": holdout_start},
        "failed_controls": [],
    }
    controls["failed_controls"] = [
        name for name, control in controls.items() if isinstance(control, dict) and not control["passed"]
    ]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(input_csv),
        "benchmark_set": list(FIXED_BASELINES),
        "constraints": {
            "fixed_date_split_declared_before_run": True,
            "fixed_cost_model": cost_model,
            "no_parameter_rescue": True,
            "no_2026_holdout_selection_leakage": True,
            "all_failed_controls_included": True,
        },
        "controls": controls,
        "baselines": _baseline_metrics(rows, train_end_date, holdout_start_date),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bounded Benchmark Harness",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Source: `{report['source']}`",
        f"- Benchmark set: {', '.join(f'`{item}`' for item in report['benchmark_set'])}",
        f"- Cost model: `{report['constraints']['fixed_cost_model']}`",
        f"- Failed controls: {', '.join(report['controls']['failed_controls']) or 'none'}",
        "",
        "## Baselines",
        "",
        "| baseline | train n | train mean R | holdout n | holdout mean R |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for baseline in FIXED_BASELINES:
        data = report["baselines"][baseline]
        lines.append(
            "| "
            f"{baseline} | {data['train']['n']} | {data['train']['mean_r']} | "
            f"{data['holdout']['n']} | {data['holdout']['mean_r']} |"
        )
    return "\n".join(lines) + "\n"


def write_bounded_benchmark_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bounded_benchmark_report.json"
    markdown_path = output_dir / "bounded_benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded EV-3 benchmark harness.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--holdout-start", required=True)
    parser.add_argument("--cost-model", default="fixed_commission_slippage_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = build_bounded_benchmark_report(
        input_csv=Path(args.input_csv),
        train_end=args.train_end,
        holdout_start=args.holdout_start,
        cost_model=args.cost_model,
    )
    write_bounded_benchmark_artifacts(report, Path(args.output_dir))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
