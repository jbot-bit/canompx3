"""Run one reviewed Chordia replay batch cycle.

This is the repeatable operator loop over the smaller Chordia tools:

1. Plan/activate a factory batch.
2. Optionally run strict replay.
3. Emit bridge artifacts and proposal-only audit rows.
4. Optionally apply reviewed rows to ``docs/runtime/chordia_audit_log.yaml``.
5. Optionally refresh derived bench/factory/status surfaces.

It does not mutate live allocation or ``validated_setups``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.chordia_audit_log_apply import apply_reviewed_proposal  # noqa: E402
from scripts.tools.chordia_evidence_factory import main as evidence_factory_main  # noqa: E402
from scripts.tools.chordia_replay_batch_bridge import (  # noqa: E402
    activate_ready_drafts,
    parse_successful_results,
    plan_acceptance,
    run_strict_replays,
    write_bridge_artifacts,
)
from scripts.tools.fast_lane_status import main as fast_lane_status_main  # noqa: E402
from scripts.tools.lane_bench_state import main as lane_bench_state_main  # noqa: E402


@dataclass(frozen=True)
class BatchCycleResult:
    factory_dir: str
    batch_id: str
    output_dir: str
    plan_count: int
    bridge_status_counts: dict[str, int]
    activated_count: int
    run_count: int
    run_returncode_counts: dict[int, int]
    parsed_result_count: int
    applied_count: int
    skipped_existing_count: int
    refreshed_fast_lane_status: bool
    refreshed_bench_dir: str | None
    refreshed_factory_dir: str | None
    live_mutation: bool = False
    validated_setups_mutation: bool = False


def _artifact_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _default_output_dir(batch_id: str) -> Path:
    stamp = datetime.now(UTC).date().isoformat().replace("-", "_")
    return _REPO_ROOT / "artifacts" / "research" / f"chordia_batch_cycle_{stamp}_{batch_id}"


def run_cycle(args: argparse.Namespace) -> BatchCycleResult:
    if args.apply_reviewed and not args.run:
        raise ValueError("--apply-reviewed requires --run so proposals come from fresh measured results.")

    output_dir = args.output_dir or _default_output_dir(args.batch_id)
    plan = plan_acceptance(
        factory_dir=args.factory_dir,
        batch_id=args.batch_id,
        hypotheses_dir=args.hypotheses_dir,
        allow_existing=args.allow_existing,
    )
    activated = activate_ready_drafts(plan) if args.activate else ()
    runs = run_strict_replays(plan, max_runs=args.max_runs) if args.run else ()
    parsed = parse_successful_results(plan, runs) if runs else ()
    write_bridge_artifacts(plan, output_dir=output_dir, results=runs, parsed_results=parsed)

    applied_count = 0
    skipped_existing_count = 0
    if args.apply_reviewed:
        failed_runs = [row for row in runs if row.returncode != 0]
        if failed_runs:
            failed_ids = ", ".join(row.strategy_id for row in failed_runs[:5])
            extra = "" if len(failed_runs) <= 5 else f" (+{len(failed_runs) - 5} more)"
            raise RuntimeError(
                "strict replay batch had non-zero return codes; "
                f"refusing to apply reviewed audit rows for partial batch: {failed_ids}{extra}"
            )
        apply_result = apply_reviewed_proposal(
            proposal_path=output_dir / "audit_log_proposal.yaml",
            audit_log_path=args.audit_log,
            reviewed=True,
            reviewed_by=args.reviewed_by,
            write=True,
        )
        applied_count = apply_result.applied_count
        skipped_existing_count = apply_result.skipped_existing_count

    refreshed_fast_lane_status = False
    if args.refresh_fast_lane_status:
        rc = fast_lane_status_main(["--write"])
        if rc != 0:
            raise RuntimeError(f"fast_lane_status refresh failed with exit code {rc}")
        refreshed_fast_lane_status = True

    refreshed_bench_dir: str | None = None
    if args.refresh_bench_output_dir:
        bench_args = [
            "--profile",
            args.profile,
            "--rebalance-date",
            args.rebalance_date,
            "--output-dir",
            str(args.refresh_bench_output_dir),
        ]
        if args.family_hints_csv:
            bench_args.extend(["--family-hints-csv", str(args.family_hints_csv)])
        with contextlib.redirect_stdout(io.StringIO()):
            rc = lane_bench_state_main(bench_args)
        if rc != 0:
            raise RuntimeError(f"lane_bench_state refresh failed with exit code {rc}")
        refreshed_bench_dir = _artifact_rel(args.refresh_bench_output_dir)

    refreshed_factory_dir: str | None = None
    if args.refresh_factory_output_dir:
        if not args.refresh_bench_output_dir:
            raise ValueError("--refresh-factory-output-dir requires --refresh-bench-output-dir.")
        factory_args = [
            "--bench-csv",
            str(args.refresh_bench_output_dir / "bench.csv"),
            "--output-dir",
            str(args.refresh_factory_output_dir),
            "--today",
            args.today,
            "--limit",
            str(args.factory_limit),
            "--batch-size",
            str(args.factory_batch_size),
            "--max-family-priority",
            str(args.max_family_priority),
        ]
        if args.include_non_default_stop:
            factory_args.append("--include-non-default-stop")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = evidence_factory_main(factory_args)
        if rc != 0:
            raise RuntimeError(f"chordia_evidence_factory refresh failed with exit code {rc}")
        refreshed_factory_dir = _artifact_rel(args.refresh_factory_output_dir)

    return BatchCycleResult(
        factory_dir=_artifact_rel(args.factory_dir),
        batch_id=args.batch_id,
        output_dir=_artifact_rel(output_dir),
        plan_count=len(plan),
        bridge_status_counts=dict(sorted(Counter(row.bridge_status for row in plan).items())),
        activated_count=len(activated),
        run_count=len(runs),
        run_returncode_counts=dict(sorted(Counter(row.returncode for row in runs).items())),
        parsed_result_count=len(parsed),
        applied_count=applied_count,
        skipped_existing_count=skipped_existing_count,
        refreshed_fast_lane_status=refreshed_fast_lane_status,
        refreshed_bench_dir=refreshed_bench_dir,
        refreshed_factory_dir=refreshed_factory_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-dir", required=True, type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hypotheses-dir", default=_REPO_ROOT / "docs" / "audit" / "hypotheses", type=Path)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--apply-reviewed", action="store_true")
    parser.add_argument("--audit-log", default=_REPO_ROOT / "docs" / "runtime" / "chordia_audit_log.yaml", type=Path)
    parser.add_argument("--reviewed-by", default="chordia_batch_cycle")
    parser.add_argument("--refresh-fast-lane-status", action="store_true")
    parser.add_argument("--refresh-bench-output-dir", type=Path)
    parser.add_argument("--profile", default="topstep_50k_mnq_auto")
    parser.add_argument("--rebalance-date", default=date.today().isoformat())
    parser.add_argument("--family-hints-csv", type=Path)
    parser.add_argument("--refresh-factory-output-dir", type=Path)
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--factory-limit", type=int, default=0)
    parser.add_argument("--factory-batch-size", type=int, default=25)
    parser.add_argument("--max-family-priority", type=int, default=5)
    parser.add_argument("--include-non-default-stop", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_cycle(args)
    if args.format == "json":
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print(f"Batch cycle: {_artifact_rel(args.factory_dir)} / {args.batch_id}")
        print(f"Bridge statuses: {result.bridge_status_counts}")
        print(f"Runs: {result.run_count}, returncodes: {result.run_returncode_counts}")
        print(f"Applied audit rows: {result.applied_count}; skipped existing: {result.skipped_existing_count}")
        print(f"Wrote artifacts to {result.output_dir}")
        print("live_mutation=false validated_setups_mutation=false")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
