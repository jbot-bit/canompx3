"""Build read-only Chordia replay evidence-factory artifacts.

This tool consumes the lane bench state CSV and turns missing strict replay
work into operator-ready draft preregs, run manifests, and audit-log proposals.
It does not run the strict replay, mutate live allocation, append to
``docs/runtime/chordia_audit_log.yaml``, or write ``validated_setups``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trading_app.strategy_discovery import parse_stop_multiplier  # noqa: E402

REPLAY_STATE = "EXACT_LANE_READY_FOR_REPLAY"
MISSING_BLOCKER = "MISSING_CHORDIA"
STRICT_THRESHOLD_PROSE = "Criterion 4 no-theory strict threshold (t >= 3.79)"
STRICT_RUNNER = "research/chordia_strict_unlock_v1.py"
DEFAULT_BENCH_CSV = _REPO_ROOT / "artifacts" / "research" / "lane_bench_state_2026_05_30" / "bench.csv"
DEFAULT_STOP_MULTIPLIER = 1.0


@dataclass(frozen=True)
class FactoryCandidate:
    rank: int
    strategy_id: str
    instrument: str
    session: str
    orb_minutes: int
    entry_model: str
    rr_target: float
    filter_type: str
    confirm_bars: int
    trailing_expr: float
    trailing_n: int
    annual_r_estimate: float
    family: str | None
    family_role: str | None
    family_priority: int

    @classmethod
    def from_bench_row(cls, row: Mapping[str, str]) -> FactoryCandidate:
        return cls(
            rank=_int(row.get("rank")),
            strategy_id=str(row.get("strategy_id") or "").strip(),
            instrument=str(row.get("instrument") or "").strip().upper(),
            session=str(row.get("session") or "").strip().upper(),
            orb_minutes=_int(row.get("orb_minutes")),
            entry_model=str(row.get("entry_model") or "").strip().upper(),
            rr_target=_float(row.get("rr_target")),
            filter_type=str(row.get("filter_type") or "NO_FILTER").strip().upper(),
            confirm_bars=_int(row.get("confirm_bars")),
            trailing_expr=_float(row.get("trailing_expr")),
            trailing_n=_int(row.get("trailing_n")),
            annual_r_estimate=_float(row.get("annual_r_estimate")),
            family=_optional_text(row.get("family")),
            family_role=_optional_text(row.get("family_role")),
            family_priority=_int(row.get("family_priority"), default=5),
        )


@dataclass(frozen=True)
class ReplayWorkItem:
    rank: int
    strategy_id: str
    factory_status: str
    next_action: str
    draft_path: str | None
    runner_command: str | None
    stop_multiplier: float
    family_priority: int


@dataclass(frozen=True)
class BatchShard:
    batch_id: str
    batch_index: int
    total_batches: int
    item_count: int
    ready_count: int
    blocked_count: int
    first_rank: int
    last_rank: int
    manifest_path: str = ""
    command_path: str = ""


@dataclass(frozen=True)
class ParsedResult:
    path: Path
    strategy_id: str
    verdict: str
    threshold: float | None
    has_theory: bool
    t_stat: float | None
    sample_size: int | None
    exp_r: float | None


@dataclass(frozen=True)
class AuditLogProposal:
    strategy_id: str
    audit_date: str
    verdict: str
    has_theory: bool
    t_stat: float | None
    sample_size: int | None
    source_result: str
    note: str


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _int(value: object, *, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return default


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return default


def _float_or_none(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _bool_from_text(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _slug(strategy_id: str) -> str:
    return strategy_id.lower().replace("_", "-").replace(".", "-")


def _artifact_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _candidate_draft_path(candidate: FactoryCandidate, output_dir: Path, today: str) -> Path:
    return output_dir / "prereg_drafts" / f"{today}-{_slug(candidate.strategy_id)}-chordia-unlock-v1.draft.yaml"


def read_bench_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _is_replay_row(row: Mapping[str, str], *, max_family_priority: int) -> bool:
    return (
        str(row.get("state") or "").strip().upper() == REPLAY_STATE
        and str(row.get("primary_blocker") or "").strip().upper() == MISSING_BLOCKER
        and _int(row.get("family_priority"), default=5) <= max_family_priority
    )


def plan_replay_work(
    rows: Sequence[Mapping[str, str]],
    *,
    limit: int,
    max_family_priority: int = 0,
    include_non_default_stop: bool = False,
    output_dir: Path | None = None,
    today: str | None = None,
) -> tuple[ReplayWorkItem, ...]:
    stamp = today or date.today().isoformat()
    base_dir = output_dir or Path(".")
    candidates = [
        FactoryCandidate.from_bench_row(row)
        for row in rows
        if _is_replay_row(row, max_family_priority=max_family_priority)
    ]
    candidates.sort(key=lambda c: (c.rank, c.family_priority, c.strategy_id))
    selected = candidates if limit <= 0 else candidates[:limit]

    planned: list[ReplayWorkItem] = []
    for candidate in selected:
        stop_multiplier = float(parse_stop_multiplier(candidate.strategy_id))
        if stop_multiplier != DEFAULT_STOP_MULTIPLIER and not include_non_default_stop:
            planned.append(
                ReplayWorkItem(
                    rank=candidate.rank,
                    strategy_id=candidate.strategy_id,
                    factory_status="BLOCKED_NON_DEFAULT_STOP",
                    next_action=(
                        "Requires outcome_builder rebuild at stop_multiplier="
                        f"{stop_multiplier:g} and a stop-specific strict replay runner before Chordia audit."
                    ),
                    draft_path=None,
                    runner_command=None,
                    stop_multiplier=stop_multiplier,
                    family_priority=candidate.family_priority,
                )
            )
            continue

        draft_path = _candidate_draft_path(candidate, base_dir, stamp)
        draft_path_text = _artifact_rel(draft_path) if output_dir is not None else str(draft_path).replace("\\", "/")
        planned.append(
            ReplayWorkItem(
                rank=candidate.rank,
                strategy_id=candidate.strategy_id,
                factory_status="PREREG_DRAFT_READY",
                next_action="Review draft, move to active hypotheses if accepted, then run strict Chordia replay.",
                draft_path=draft_path_text,
                runner_command=f"python {STRICT_RUNNER} --hypothesis-file {draft_path_text}",
                stop_multiplier=stop_multiplier,
                family_priority=candidate.family_priority,
            )
        )
    return tuple(planned)


def plan_batch_shards(work_items: Sequence[ReplayWorkItem], *, batch_size: int) -> tuple[BatchShard, ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    chunks = list(_chunked(work_items, batch_size))
    total_batches = len(chunks)
    shards: list[BatchShard] = []
    for idx, chunk in enumerate(chunks, start=1):
        ready = sum(1 for item in chunk if item.factory_status == "PREREG_DRAFT_READY")
        blocked = len(chunk) - ready
        batch_id = f"batch_{idx:03d}"
        shards.append(
            BatchShard(
                batch_id=batch_id,
                batch_index=idx,
                total_batches=total_batches,
                item_count=len(chunk),
                ready_count=ready,
                blocked_count=blocked,
                first_rank=chunk[0].rank,
                last_rank=chunk[-1].rank,
                manifest_path=f"batches/{batch_id}.csv",
                command_path=f"batches/{batch_id}_commands.ps1",
            )
        )
    return tuple(shards)


def _chunked(items: Sequence[ReplayWorkItem], size: int) -> Iterable[tuple[ReplayWorkItem, ...]]:
    for start in range(0, len(items), size):
        yield tuple(items[start : start + size])


def build_prereg_draft(candidate: FactoryCandidate, *, today: str) -> dict[str, Any]:
    strategy_id = candidate.strategy_id
    family_note = candidate.family or "lane_bench_state replay queue"
    return {
        "metadata": {
            "theory_grant": False,
            "name": f"{_slug(strategy_id).replace('-', '_')}_chordia_unlock_v1",
            "purpose": (
                "Evidence-factory draft for a bounded exact-lane Chordia strict replay. "
                "This is not an active prereg until an operator reviews and moves it."
            ),
            "date_locked": f"{today}T00:00:00+10:00",
            "holdout_date": "2026-01-01",
            "total_expected_trials": 1,
            "testing_mode": "individual",
            "research_question_type": "conditional_role",
            "template_version": "chordia_strict_v1",
            "is_triage_screen": False,
            "validation_status_explicit": (
                "DRAFT -- not preregistered. Keep no-theory strict threshold unless "
                "a separately reviewed theory_citation is deliberately added."
            ),
        },
        "execution": {
            "mode": "bounded_runner",
            "entrypoint": STRICT_RUNNER,
        },
        "authority": {
            "primary": [
                "RESEARCH_RULES.md",
                ".claude/rules/backtesting-methodology.md",
                ".claude/rules/research-truth-protocol.md",
                "docs/institutional/pre_registered_criteria.md",
                "docs/runtime/chordia_audit_log.yaml",
            ],
            "notes": [
                "Authored by scripts/tools/chordia_evidence_factory.py.",
                "The factory is proposal-only and does not append to the audit log.",
                f"Threshold basis: {STRICT_THRESHOLD_PROSE}.",
            ],
        },
        "scope": {
            "strategy_id": strategy_id,
            "instrument": candidate.instrument,
            "session": candidate.session,
            "orb_minutes": candidate.orb_minutes,
            "entry_model": candidate.entry_model,
            "confirm_bars": candidate.confirm_bars,
            "rr_target": candidate.rr_target,
            "direction": "pooled",
            "filter_type": candidate.filter_type,
            "filter_source": "lane_bench_state",
            "out_of_scope": [
                "sibling RR/aperture/confirm-bar variants",
                "live allocation",
                "validated_setups promotion",
                "audit-log append before measured result review",
            ],
        },
        "data_policy": {
            "is_window": {
                "description": "trading_day < HOLDOUT_SACRED_FROM",
                "constant_source": "trading_app.holdout_policy.HOLDOUT_SACRED_FROM",
                "locked_boundary": "2026-01-01",
            },
            "oos_window": {
                "description": "trading_day >= HOLDOUT_SACRED_FROM",
                "policy": "descriptive OOS sign/power review; not a live deployment criterion by itself",
            },
            "tuning_against_oos": False,
            "canonical_layers_only": True,
            "scratch_policy": "realized-eod",
            "scratch_handling": "COALESCE(pnl_r, 0.0) -- never WHERE pnl_r IS NOT NULL",
        },
        "grounding": {
            "filter_grounding_status": {
                "verdict": "UNSUPPORTED",
                "basis": (
                    "Evidence factory cannot author literature citations. "
                    "theory_grant=false keeps the strict no-theory Chordia hurdle."
                ),
            },
        },
        "upstream_discovery_provenance": {
            "role": "REPLAY_QUEUE_PROVENANCE_ONLY",
            "source": "lane_bench_state",
            "family": family_note,
            "family_role": candidate.family_role,
            "family_priority": candidate.family_priority,
            "trailing_expr": candidate.trailing_expr,
            "trailing_n": candidate.trailing_n,
            "annual_r_estimate": candidate.annual_r_estimate,
            "note": (
                "Bench status earns the right to an exact-lane replay only. "
                "Fresh strict runner output is the only valid Chordia verdict source."
            ),
        },
        "primary_schema": {
            "family_cells": [
                {
                    "id": strategy_id,
                    "strategy_id": strategy_id,
                    "instrument": candidate.instrument,
                    "session": candidate.session,
                    "orb_minutes": candidate.orb_minutes,
                    "rr_target": candidate.rr_target,
                    "filter": candidate.filter_type,
                }
            ],
            "k_family": 1,
            "k_global": 1,
            "k_lane": 1,
            "k_session": 1,
            "chordia_threshold_basis": STRICT_THRESHOLD_PROSE,
            "promotion_gate": "PASS_CHORDIA only; live allocation requires separate deployment-readiness gates",
        },
        "hypotheses": [
            {
                "name": f"{_slug(strategy_id)}-strict-chordia",
                "economic_basis": (
                    "Bounded exact-lane replay of a lane-bench candidate. The question is whether "
                    "this exact configured lane clears the strict no-theory Chordia hurdle on "
                    "canonical Mode-A data; no sibling rescue is allowed."
                ),
                "role": {
                    "kind": "standalone",
                    "parent": (
                        f"Exact {strategy_id} canonical lane replay on orb_outcomes x daily_features."
                    ),
                    "comparator": (
                        "Replay-selected trades versus the strict Chordia threshold and descriptive "
                        "OOS sign monitor; no sibling rescue or parameter variation."
                    ),
                    "primary_metric": "selected_trade_mean_r",
                    "promotion_target": "Chordia audit-log proposal only after measured result review.",
                },
                "filter": {
                    "type": candidate.filter_type,
                },
                "scope": {
                    "instruments": [candidate.instrument],
                    "sessions": [candidate.session],
                    "rr_targets": [candidate.rr_target],
                    "entry_models": [candidate.entry_model],
                    "confirm_bars": [candidate.confirm_bars],
                    "stop_multipliers": [DEFAULT_STOP_MULTIPLIER],
                    "orb_minutes": [candidate.orb_minutes],
                },
                "expected_trial_count": 1,
                "pass_criteria": {
                    "primary_metric": "IS t-stat",
                    "threshold": 3.79,
                    "min_is_trades": 100,
                    "min_exp_r": 0.0,
                },
                "kill_criteria": [
                    "IS t < 3.79",
                    "IS ExpR <= 0",
                    "N_IS_on < 100",
                    "OOS sign flip with descriptive OOS N >= 30",
                ],
            }
        ],
        "trial_budget": {
            "primary_selection_trials": 1,
            "schema_locked_before_any_metric": True,
            "minbtl_bound": "K=1 exact-lane replay; upstream queue provenance carries discovery multiplicity risk.",
        },
        "total_hypothesis_count": 1,
        "total_expected_trials": 1,
        "methodology_rules_applied": {
            "rule_1_temporal_alignment": {"application": "Strict runner resolves canonical ORB windows."},
            "rule_3_is_oos_discipline": {"application": "HOLDOUT_SACRED_FROM=2026-01-01 remains locked."},
            "rule_9_canonical_layers": {"application": "Strict runner reads orb_outcomes joined to daily_features."},
            "rule_10_pre_registration": {
                "application": "Draft must be reviewed and moved into active hypotheses before execution."
            },
        },
        "execution_gate": {
            "allowed_now": False,
            "execution_surface": "Windows PowerShell with .venv",
            "operator_must_review_before_flip": [
                "accept no-theory strict 3.79 hurdle or author a grounded theory citation",
                "confirm default stop_multiplier support",
                "confirm OOS descriptive power and era stability expectations",
            ],
            "forbidden_now": [
                "any deployment action",
                "writes to chordia_audit_log.yaml until result MD exists and is reviewed",
                "validated_setups promotion",
            ],
        },
    }


def write_prereg_draft(prereg: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(prereg), sort_keys=False, allow_unicode=False, width=100), encoding="utf-8")


def parse_result_md(path: Path) -> ParsedResult:
    text = path.read_text(encoding="utf-8")
    strategy_id = _match_required(
        text,
        r"^#\s+Chordia strict unlock audit\s+(?:-|--|---|\u2014)\s*([A-Za-z0-9_.-]+)",
        "strategy_id",
    )
    verdict = _match_required(text, r"\*\*MEASURED verdict:\*\*\s*`?([A-Z0-9_]+)`?", "verdict")
    threshold = _float_or_none(_match_optional(text, r"\*\*MEASURED threshold applied:\*\*\s*`?([0-9.]+)`?"))
    has_theory = _bool_from_text(
        _match_optional(text, r"\*\*MEASURED loader has_theory:\*\*\s*`?(True|False)`?"),
        default=False,
    )
    sample_size, exp_r, t_stat = _parse_is_table_row(text)
    if sample_size is None:
        sample_size = _int_or_none(_match_optional(text, r"\bN=(\d+)\b"))
    if exp_r is None:
        exp_r = _float_or_none(_match_optional(text, r"\bExpR=([-+]?[0-9]*\.?[0-9]+)"))
    return ParsedResult(
        path=path,
        strategy_id=strategy_id,
        verdict=verdict,
        threshold=threshold,
        has_theory=has_theory,
        t_stat=t_stat,
        sample_size=sample_size,
        exp_r=exp_r,
    )


def _match_required(text: str, pattern: str, field: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse {field} from Chordia result markdown")
    return match.group(1)


def _match_optional(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_is_table_row(text: str) -> tuple[int | None, float | None, float | None]:
    for line in text.splitlines():
        if not line.strip().startswith("| IS |"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 10:
            return None, None, None
        return _int_or_none(cells[2]), _float_or_none(cells[6]), _float_or_none(cells[9])
    return None, None, None


def build_audit_log_proposals(
    parsed_results: Iterable[ParsedResult],
    *,
    audit_date: str,
) -> tuple[AuditLogProposal, ...]:
    proposals: list[AuditLogProposal] = []
    for result in parsed_results:
        proposals.append(
            AuditLogProposal(
                strategy_id=result.strategy_id,
                audit_date=audit_date,
                verdict=result.verdict,
                has_theory=result.has_theory,
                t_stat=result.t_stat,
                sample_size=result.sample_size,
                source_result=_artifact_rel(result.path),
                note=(
                    "PROPOSAL_ONLY from measured result; review before appending to "
                    "docs/runtime/chordia_audit_log.yaml."
                ),
            )
        )
    return tuple(proposals)


def write_factory_artifacts(
    rows: Sequence[Mapping[str, str]],
    *,
    output_dir: Path,
    today: str,
    limit: int,
    max_family_priority: int = 0,
    include_non_default_stop: bool = False,
    result_paths: Sequence[Path] = (),
    bench_csv: Path | None = None,
    batch_size: int = 25,
) -> tuple[ReplayWorkItem, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_items = plan_replay_work(
        rows,
        limit=limit,
        max_family_priority=max_family_priority,
        include_non_default_stop=include_non_default_stop,
        output_dir=output_dir,
        today=today,
    )
    candidates_by_id = {
        FactoryCandidate.from_bench_row(row).strategy_id: FactoryCandidate.from_bench_row(row)
        for row in rows
        if _is_replay_row(row, max_family_priority=max_family_priority)
    }

    for item in work_items:
        if item.factory_status != "PREREG_DRAFT_READY" or item.draft_path is None:
            continue
        candidate = candidates_by_id[item.strategy_id]
        write_prereg_draft(build_prereg_draft(candidate, today=today), _REPO_ROOT / item.draft_path)

    _write_run_manifest(output_dir / "run_manifest.csv", work_items)
    shards = plan_batch_shards(work_items, batch_size=batch_size)
    _write_batch_artifacts(output_dir, work_items, shards=shards, batch_size=batch_size)
    parsed = tuple(parse_result_md(path) for path in result_paths)
    proposals = build_audit_log_proposals(parsed, audit_date=today)
    (output_dir / "audit_log_proposal.yaml").write_text(
        yaml.safe_dump(
            {
                "proposal_only": True,
                "target": "docs/runtime/chordia_audit_log.yaml",
                "audits": [asdict(proposal) for proposal in proposals],
                "note": "Empty audits list means no result MD inputs were supplied.",
            },
            sort_keys=False,
            allow_unicode=False,
            width=100,
        ),
        encoding="utf-8",
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "today": today,
        "bench_csv": _artifact_rel(bench_csv) if bench_csv else None,
        "work_item_count": len(work_items),
        "factory_status_counts": dict(sorted(Counter(item.factory_status for item in work_items).items())),
        "batch_count": len(shards),
        "batch_size": batch_size,
        "result_md_count": len(result_paths),
        "proposal_count": len(proposals),
        "live_mutation": False,
        "validated_setups_mutation": False,
        "chordia_log_mutation": False,
        "strict_replay_execution": False,
        "include_non_default_stop": include_non_default_stop,
        "max_family_priority": max_family_priority,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_report(work_items, proposals=proposals, batch_size=batch_size),
        encoding="utf-8",
    )
    return work_items


def _write_batch_artifacts(
    output_dir: Path,
    work_items: Sequence[ReplayWorkItem],
    *,
    shards: Sequence[BatchShard],
    batch_size: int,
) -> None:
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "batch_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(BatchShard.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for shard in shards:
            writer.writerow(asdict(shard))

    for shard, items in zip(shards, _chunked(work_items, batch_size), strict=True):
        _write_run_manifest(output_dir / shard.manifest_path, items)
        lines = [
            "# Proposal-only strict replay command shard.",
            "# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.",
            "# This file is not executed by the evidence factory.",
            "",
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            "",
        ]
        for item in items:
            if item.factory_status != "PREREG_DRAFT_READY" or item.runner_command is None:
                continue
            lines.append(f"# {item.strategy_id}")
            lines.append(item.runner_command)
        lines.append("")
        (output_dir / shard.command_path).write_text("\n".join(lines), encoding="utf-8")


def _write_run_manifest(path: Path, work_items: Sequence[ReplayWorkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(ReplayWorkItem.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in work_items:
            writer.writerow(asdict(item))


def render_report(
    work_items: Sequence[ReplayWorkItem],
    *,
    proposals: Sequence[AuditLogProposal],
    batch_size: int = 25,
) -> str:
    status_counts = dict(sorted(Counter(item.factory_status for item in work_items).items()))
    batch_count = len(plan_batch_shards(work_items, batch_size=batch_size)) if work_items else 0
    lines = [
        "# Chordia Evidence Factory",
        "",
        "This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run "
        "strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.",
        "",
        "## Summary",
        "",
        f"- work items: {len(work_items)}",
        f"- status counts: {status_counts}",
        f"- batch count at size {batch_size}: {batch_count}",
        f"- audit-log proposals: {len(proposals)}",
        "- non-default stop lanes are blocked by default because the current strict runner audits default-stop orb_outcomes only",
        "",
        "## Work Items",
        "",
        "| rank | status | strategy_id | stop | next_action |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for item in work_items:
        lines.append(
            f"| {item.rank} | {item.factory_status} | `{item.strategy_id}` | "
            f"{item.stop_multiplier:g} | {item.next_action} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-csv", type=Path, default=DEFAULT_BENCH_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-family-priority", type=int, default=0)
    parser.add_argument("--include-non-default-stop", action="store_true")
    parser.add_argument("--result-md", type=Path, action="append", default=[])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        _REPO_ROOT / "artifacts" / "research" / f"chordia_evidence_factory_{args.today.replace('-', '_')}"
    )
    rows = read_bench_csv(args.bench_csv)
    work_items = write_factory_artifacts(
        rows,
        output_dir=output_dir,
        today=args.today,
        limit=args.limit,
        max_family_priority=args.max_family_priority,
        include_non_default_stop=args.include_non_default_stop,
        result_paths=tuple(args.result_md),
        bench_csv=args.bench_csv,
        batch_size=args.batch_size,
    )
    if args.format == "json":
        print(json.dumps([asdict(item) for item in work_items], indent=2, sort_keys=True))
    else:
        print((output_dir / "report.md").read_text(encoding="utf-8"))
        print(f"Wrote artifacts to {_artifact_rel(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
