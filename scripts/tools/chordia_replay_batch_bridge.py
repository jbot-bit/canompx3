"""Bridge Chordia factory batches into reviewed strict-replay execution.

The bridge has three explicit stages:

1. Plan acceptance from a factory ``batch_###.csv``.
2. Activate ready draft preregs by copying them into an active hypotheses
   directory with ``execution_gate.allowed_now=true``.
3. Optionally run the existing strict Chordia runner and parse measured result
   markdown into an audit-log proposal.

It never appends to ``docs/runtime/chordia_audit_log.yaml`` and never mutates
live allocation or ``validated_setups``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.chordia_evidence_factory import (  # noqa: E402
    ParsedResult,
    build_audit_log_proposals,
    parse_result_md,
)

STRICT_RUNNER = "research/chordia_strict_unlock_v1.py"
DEFAULT_HYPOTHESES_DIR = _REPO_ROOT / "docs" / "audit" / "hypotheses"


@dataclass(frozen=True)
class BridgePlanRow:
    rank: int
    strategy_id: str
    factory_status: str
    bridge_status: str
    source_draft_path: str
    active_hypothesis_path: str
    result_md_path: str
    runner_command: str
    note: str


@dataclass(frozen=True)
class ReplayRunResult:
    strategy_id: str
    hypothesis_path: str
    result_md_path: str
    returncode: int
    stdout_tail: str
    stderr_tail: str


def _artifact_rel(path: Path, *, repo_root: Path = _REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_path(path_text: str, *, repo_root: Path = _REPO_ROOT) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _active_path_for(source_draft: Path, hypotheses_dir: Path) -> Path:
    name = source_draft.name
    if name.endswith(".draft.yaml"):
        name = name.removesuffix(".draft.yaml") + ".yaml"
    return hypotheses_dir / name


def _result_path_for(active_hypothesis: Path, *, repo_root: Path = _REPO_ROOT) -> Path:
    return repo_root / "docs" / "audit" / "results" / f"{active_hypothesis.stem}.md"


def _read_batch(factory_dir: Path, batch_id: str) -> list[dict[str, str]]:
    path = factory_dir / "batches" / f"{batch_id}.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def plan_acceptance(
    *,
    factory_dir: Path,
    batch_id: str,
    hypotheses_dir: Path = DEFAULT_HYPOTHESES_DIR,
    repo_root: Path = _REPO_ROOT,
    allow_existing: bool = False,
) -> tuple[BridgePlanRow, ...]:
    rows: list[BridgePlanRow] = []
    for row in _read_batch(factory_dir, batch_id):
        rank = int(float(row.get("rank") or 0))
        strategy_id = row.get("strategy_id") or ""
        factory_status = row.get("factory_status") or ""
        source_draft_text = row.get("draft_path") or ""

        if factory_status != "PREREG_DRAFT_READY":
            rows.append(
                BridgePlanRow(
                    rank=rank,
                    strategy_id=strategy_id,
                    factory_status=factory_status,
                    bridge_status="SKIP_FACTORY_STATUS",
                    source_draft_path=source_draft_text,
                    active_hypothesis_path="",
                    result_md_path="",
                    runner_command="",
                    note=f"Factory status is {factory_status}; bridge only activates PREREG_DRAFT_READY rows.",
                )
            )
            continue

        source_draft = (
            repo_root / source_draft_text if not Path(source_draft_text).is_absolute() else Path(source_draft_text)
        )
        active_path = _active_path_for(source_draft, hypotheses_dir)
        result_path = _result_path_for(active_path, repo_root=repo_root)
        if not source_draft.is_file():
            status = "BLOCKED_MISSING_DRAFT"
            note = "Source draft file is missing."
        elif active_path.exists() and not allow_existing:
            status = "BLOCKED_ACTIVE_EXISTS"
            note = "Active hypothesis file already exists; pass --allow-existing to reuse."
        else:
            status = "READY_TO_ACTIVATE"
            note = "Ready for activation and strict replay."

        rel_active = _artifact_rel(active_path, repo_root=repo_root)
        rows.append(
            BridgePlanRow(
                rank=rank,
                strategy_id=strategy_id,
                factory_status=factory_status,
                bridge_status=status,
                source_draft_path=_artifact_rel(source_draft, repo_root=repo_root),
                active_hypothesis_path=rel_active if status != "BLOCKED_MISSING_DRAFT" else "",
                result_md_path=_artifact_rel(result_path, repo_root=repo_root)
                if status != "BLOCKED_MISSING_DRAFT"
                else "",
                runner_command=f"python {STRICT_RUNNER} --hypothesis-file {rel_active}"
                if status in {"READY_TO_ACTIVATE", "BLOCKED_ACTIVE_EXISTS"}
                else "",
                note=note,
            )
        )
    return tuple(rows)


def activate_ready_drafts(
    plan: Sequence[BridgePlanRow],
    *,
    repo_root: Path = _REPO_ROOT,
) -> tuple[BridgePlanRow, ...]:
    activated: list[BridgePlanRow] = []
    for row in plan:
        if row.bridge_status != "READY_TO_ACTIVATE":
            continue
        source = _resolve_path(row.source_draft_path, repo_root=repo_root)
        target = _resolve_path(row.active_hypothesis_path, repo_root=repo_root)
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Draft prereg must be a YAML mapping: {source}")
        gate = data.setdefault("execution_gate", {})
        if not isinstance(gate, dict):
            raise ValueError(f"execution_gate must be a mapping: {source}")
        gate["allowed_now"] = True
        data["bridge_activation"] = {
            "source_draft": row.source_draft_path,
            "activated_at": datetime.now(UTC).isoformat(),
            "bridge": "scripts/tools/chordia_replay_batch_bridge.py",
            "mutation_scope": "active prereg copy only; no audit-log/live mutation",
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100),
            encoding="utf-8",
        )
        activated.append(row)
    return tuple(activated)


def run_strict_replays(
    plan: Sequence[BridgePlanRow],
    *,
    repo_root: Path = _REPO_ROOT,
    max_runs: int | None = None,
) -> tuple[ReplayRunResult, ...]:
    runnable = [row for row in plan if row.bridge_status in {"READY_TO_ACTIVATE", "BLOCKED_ACTIVE_EXISTS"}]
    if max_runs is not None:
        runnable = runnable[:max_runs]
    results: list[ReplayRunResult] = []
    for row in runnable:
        hypothesis_path = _resolve_path(row.active_hypothesis_path, repo_root=repo_root)
        if not hypothesis_path.is_file():
            raise FileNotFoundError(f"Active hypothesis file is missing; activate first: {hypothesis_path}")
        proc = subprocess.run(
            [sys.executable, STRICT_RUNNER, "--hypothesis-file", str(hypothesis_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            ReplayRunResult(
                strategy_id=row.strategy_id,
                hypothesis_path=_artifact_rel(hypothesis_path, repo_root=repo_root),
                result_md_path=row.result_md_path,
                returncode=proc.returncode,
                stdout_tail=_tail(proc.stdout),
                stderr_tail=_tail(proc.stderr),
            )
        )
    return tuple(results)


def _tail(text: str, *, max_chars: int = 2000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _write_plan_csv(path: Path, plan: Sequence[BridgePlanRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(BridgePlanRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in plan:
            writer.writerow(asdict(row))


def _write_run_csv(path: Path, results: Sequence[ReplayRunResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(ReplayRunResult.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_bridge_artifacts(
    plan: Sequence[BridgePlanRow],
    *,
    output_dir: Path,
    results: Sequence[ReplayRunResult],
    parsed_results: Sequence[ParsedResult] = (),
    audit_date: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_plan_csv(output_dir / "activation_plan.csv", plan)
    _write_run_csv(output_dir / "strict_replay_runs.csv", results)
    proposal_date = audit_date or datetime.now().astimezone().date().isoformat()
    proposals = build_audit_log_proposals(parsed_results, audit_date=proposal_date)
    (output_dir / "audit_log_proposal.yaml").write_text(
        yaml.safe_dump(
            {
                "proposal_only": True,
                "target": "docs/runtime/chordia_audit_log.yaml",
                "audits": [asdict(proposal) for proposal in proposals],
                "note": "Review before appending to runtime audit log.",
            },
            sort_keys=False,
            allow_unicode=False,
            width=100,
        ),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "plan_count": len(plan),
        "bridge_status_counts": dict(sorted(Counter(row.bridge_status for row in plan).items())),
        "strict_replay_execution": bool(results),
        "strict_replay_returncode_counts": dict(sorted(Counter(row.returncode for row in results).items())),
        "proposal_count": len(proposals),
        "chordia_log_mutation": False,
        "live_mutation": False,
        "validated_setups_mutation": False,
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False, width=100),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_successful_results(
    plan: Sequence[BridgePlanRow],
    runs: Sequence[ReplayRunResult],
    *,
    repo_root: Path = _REPO_ROOT,
) -> tuple[ParsedResult, ...]:
    successful_ids = {run.strategy_id for run in runs if run.returncode == 0}
    parsed: list[ParsedResult] = []
    for row in plan:
        if row.strategy_id not in successful_ids or not row.result_md_path:
            continue
        path = _resolve_path(row.result_md_path, repo_root=repo_root)
        if path.exists():
            parsed.append(parse_result_md(path))
    return tuple(parsed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-dir", required=True, type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hypotheses-dir", type=Path, default=DEFAULT_HYPOTHESES_DIR)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = plan_acceptance(
        factory_dir=args.factory_dir,
        batch_id=args.batch_id,
        hypotheses_dir=args.hypotheses_dir,
        allow_existing=args.allow_existing,
    )
    if args.activate:
        activate_ready_drafts(plan)
    runs: tuple[ReplayRunResult, ...] = ()
    parsed: tuple[ParsedResult, ...] = ()
    if args.run:
        runs = run_strict_replays(plan, max_runs=args.max_runs)
        parsed = parse_successful_results(plan, runs)
    output_dir = args.output_dir or (
        _REPO_ROOT
        / "artifacts"
        / "research"
        / f"chordia_replay_batch_bridge_{datetime.now(UTC).date().isoformat().replace('-', '_')}_{args.batch_id}"
    )
    write_bridge_artifacts(plan, output_dir=output_dir, results=runs, parsed_results=parsed)
    if args.format == "json":
        print(json.dumps([asdict(row) for row in plan], indent=2, sort_keys=True))
    else:
        print(f"Bridge plan rows: {len(plan)}")
        print(f"Status counts: {dict(sorted(Counter(row.bridge_status for row in plan).items()))}")
        print(f"Strict replay runs: {len(runs)}")
        print(f"Wrote artifacts to {_artifact_rel(output_dir)}")
    return 1 if any(run.returncode != 0 for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
