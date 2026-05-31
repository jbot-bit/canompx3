"""Drain reviewed Chordia replay work through repeatable batch cycles.

This is a thin orchestrator over ``scripts/tools/chordia_batch_cycle.py``. It
keeps the strict replay/audit-log workflow resume-safe by refreshing the bench
and factory after every applied batch, then taking the next ``batch_001`` from
the refreshed factory.

It does not mutate live allocation or ``validated_setups``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.chordia_batch_cycle import BatchCycleResult, run_cycle


@dataclass(frozen=True)
class QueueRunResult:
    initial_factory_dir: str
    final_factory_dir: str
    batch_id: str
    cycles_run: int
    stopped_reason: str
    total_plan_count: int
    total_activated_count: int
    total_run_count: int
    total_parsed_result_count: int
    total_applied_count: int
    total_skipped_existing_count: int
    latest_refreshed_bench_dir: str | None
    latest_refreshed_factory_dir: str | None
    cycle_output_dirs: tuple[str, ...]
    live_mutation: bool = False
    validated_setups_mutation: bool = False


def _safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text).strip("_")


def _path_text(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _batch_ready_count(factory_dir: Path, batch_id: str) -> int:
    batch_path = factory_dir / "batches" / f"{batch_id}.csv"
    if not batch_path.is_file():
        return 0
    with batch_path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for row in csv.DictReader(fh) if row.get("factory_status") == "PREREG_DRAFT_READY")


def _cycle_output_dir(output_root: Path, label: str, cycle_index: int) -> Path:
    return output_root / f"chordia_batch_cycle_{label}_{cycle_index:03d}"


def _bench_output_dir(output_root: Path, label: str, cycle_index: int) -> Path:
    return output_root / f"lane_bench_state_{label}_{cycle_index:03d}_apply"


def _factory_output_dir(output_root: Path, label: str, cycle_index: int, max_family_priority: int) -> Path:
    return output_root / f"chordia_evidence_factory_{label}_{cycle_index:03d}_apply_p{max_family_priority}"


def _make_cycle_args(args: argparse.Namespace, *, factory_dir: Path, cycle_index: int) -> argparse.Namespace:
    label = _safe_label(args.run_label)
    bench_dir = _bench_output_dir(args.output_root, label, cycle_index)
    factory_out = _factory_output_dir(args.output_root, label, cycle_index, args.max_family_priority)
    return argparse.Namespace(
        factory_dir=factory_dir,
        batch_id=args.batch_id,
        output_dir=_cycle_output_dir(args.output_root, label, cycle_index),
        hypotheses_dir=args.hypotheses_dir,
        allow_existing=args.allow_existing,
        activate=args.activate,
        run=args.run,
        max_runs=args.max_runs_per_batch,
        apply_reviewed=args.apply_reviewed,
        audit_log=args.audit_log,
        reviewed_by=args.reviewed_by,
        refresh_fast_lane_status=args.refresh_fast_lane_status,
        refresh_bench_output_dir=bench_dir,
        profile=args.profile,
        rebalance_date=args.rebalance_date,
        family_hints_csv=args.family_hints_csv,
        refresh_factory_output_dir=factory_out,
        today=args.today,
        factory_limit=args.factory_limit,
        factory_batch_size=args.factory_batch_size,
        max_family_priority=args.max_family_priority,
        include_non_default_stop=args.include_non_default_stop,
        format="json",
    )


def run_queue(args: argparse.Namespace) -> QueueRunResult:
    current_factory = args.initial_factory_dir
    cycles: list[BatchCycleResult] = []
    stopped_reason = "MAX_BATCHES_REACHED"

    for cycle_index in range(1, args.max_batches + 1):
        if _batch_ready_count(current_factory, args.batch_id) < 1:
            stopped_reason = "NO_READY_WORK"
            break

        result = run_cycle(_make_cycle_args(args, factory_dir=current_factory, cycle_index=cycle_index))
        cycles.append(result)
        if not result.refreshed_factory_dir:
            stopped_reason = "NO_REFRESHED_FACTORY"
            break
        current_factory = Path(result.refreshed_factory_dir)

    latest = cycles[-1] if cycles else None
    return QueueRunResult(
        initial_factory_dir=_path_text(args.initial_factory_dir),
        final_factory_dir=_path_text(current_factory),
        batch_id=args.batch_id,
        cycles_run=len(cycles),
        stopped_reason=stopped_reason,
        total_plan_count=sum(item.plan_count for item in cycles),
        total_activated_count=sum(item.activated_count for item in cycles),
        total_run_count=sum(item.run_count for item in cycles),
        total_parsed_result_count=sum(item.parsed_result_count for item in cycles),
        total_applied_count=sum(item.applied_count for item in cycles),
        total_skipped_existing_count=sum(item.skipped_existing_count for item in cycles),
        latest_refreshed_bench_dir=latest.refreshed_bench_dir if latest else None,
        latest_refreshed_factory_dir=latest.refreshed_factory_dir if latest else None,
        cycle_output_dirs=tuple(item.output_dir for item in cycles),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-factory-dir", required=True, type=Path)
    parser.add_argument("--batch-id", default="batch_001")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-root", default=Path("artifacts/research"), type=Path)
    parser.add_argument("--max-batches", required=True, type=_positive_int)
    parser.add_argument("--hypotheses-dir", default=Path("docs/audit/hypotheses"), type=Path)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-runs-per-batch", type=_positive_int)
    parser.add_argument("--apply-reviewed", action="store_true")
    parser.add_argument("--audit-log", default=Path("docs/runtime/chordia_audit_log.yaml"), type=Path)
    parser.add_argument("--reviewed-by", default="chordia_batch_queue_runner")
    parser.add_argument("--refresh-fast-lane-status", action="store_true")
    parser.add_argument("--profile", default="topstep_50k_mnq_auto")
    parser.add_argument("--rebalance-date", default=date.today().isoformat())
    parser.add_argument("--family-hints-csv", type=Path)
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--factory-limit", type=int, default=0)
    parser.add_argument("--factory-batch-size", type=_positive_int, default=25)
    parser.add_argument("--max-family-priority", type=int, default=5)
    parser.add_argument("--include-non-default-stop", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_queue(args)
    if args.format == "json":
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print(f"Queue run: {result.initial_factory_dir} / {result.batch_id}")
        print(f"Cycles run: {result.cycles_run}; stopped: {result.stopped_reason}")
        print(f"Runs: {result.total_run_count}; applied audit rows: {result.total_applied_count}")
        print(f"Final factory: {result.final_factory_dir}")
        print("live_mutation=false validated_setups_mutation=false")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
