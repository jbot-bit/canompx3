"""Classify deployable-shelf rows into a read-only lane bench state.

This is an operator/readiness surface, not deployment authority. It does not
promote lanes, write live allocation JSON, mutate validated_setups, or append
to docs/runtime/chordia_audit_log.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.chordia_unlock_batch import (  # noqa: E402
    DEPLOY_CLASS_STATUSES,
    FamilyHint,
    UnlockCandidate,
    candidate_from_lane_score,
    candidate_key,
    load_family_hints_from_cells,
)
from trading_app.lane_allocator import apply_c8_gate, apply_live_tradeability_gate, compute_lane_scores  # noqa: E402
from trading_app.prop_profiles import ACCOUNT_PROFILES, resolve_allocation_json  # noqa: E402

PASS_VERDICTS = frozenset({"PASS_CHORDIA", "PASS_PROTOCOL_A"})
FAIL_VERDICTS = frozenset({"FAIL_CHORDIA", "FAIL_BOTH"})


@dataclass(frozen=True)
class BenchRow:
    rank: int
    strategy_id: str
    state: str
    primary_blocker: str
    next_action: str
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
    status: str
    status_reason: str
    chordia_verdict: str
    chordia_audit_age_days: int | None
    c8_oos_status: str | None
    family: str | None
    family_role: str | None
    family_priority: int
    active_in_profile: bool


def _norm_verdict(verdict: str | None) -> str:
    if verdict is None or verdict == "":
        return "MISSING"
    return verdict.upper()


def _family_hint_for(
    candidate: UnlockCandidate,
    family_hints: dict[tuple[str, str, int, str, str, str], FamilyHint],
) -> FamilyHint | None:
    return family_hints.get(
        candidate_key(
            candidate.instrument,
            candidate.session,
            candidate.orb_minutes,
            candidate.entry_model,
            candidate.rr_target,
            candidate.filter_type,
        )
    )


def load_family_hints_from_bench(path: Path) -> dict[tuple[str, str, int, str, str, str], FamilyHint]:
    hints: dict[tuple[str, str, int, str, str, str], FamilyHint] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            family = (row.get("family") or "").strip()
            family_role = (row.get("family_role") or "").strip()
            priority_raw = row.get("family_priority")
            if not family and not family_role and priority_raw in (None, ""):
                continue
            key = candidate_key(
                str(row["instrument"]),
                str(row["session"]),
                int(float(row["orb_minutes"])),
                str(row["entry_model"]),
                float(row["rr_target"]),
                str(row["filter_type"]),
            )
            hints[key] = FamilyHint(
                instrument=str(row["instrument"]),
                session=str(row["session"]),
                orb_minutes=int(float(row["orb_minutes"])),
                entry_model=str(row["entry_model"]),
                rr_target=float(row["rr_target"]),
                filter_type=str(row["filter_type"]),
                family=family or "UNSPECIFIED",
                family_role=family_role or "unknown",
                family_median_exp_r=None,
                family_oos_median=None,
                priority=int(float(priority_raw if priority_raw not in (None, "") else 5)),
            )
    return hints


def _profile_blocker(
    candidate: UnlockCandidate,
    *,
    allowed_instruments: frozenset[str] | None,
    allowed_sessions: frozenset[str] | None,
) -> str | None:
    if allowed_instruments is not None and candidate.instrument not in allowed_instruments:
        return "PROFILE_INSTRUMENT"
    if allowed_sessions is not None and candidate.session not in allowed_sessions:
        return "PROFILE_SESSION"
    return None


def _candidate_blocker(
    candidate: UnlockCandidate,
    *,
    allowed_instruments: frozenset[str] | None,
    allowed_sessions: frozenset[str] | None,
) -> str:
    profile = _profile_blocker(
        candidate,
        allowed_instruments=allowed_instruments,
        allowed_sessions=allowed_sessions,
    )
    if profile is not None:
        return profile

    reason = candidate.status_reason or ""
    c8 = candidate.c8_oos_status
    if c8 is not None and c8 != "PASSED":
        return "C8_OOS"
    if reason.startswith("live tradeability gate"):
        return "LIVE_TRADEABILITY"
    if reason.startswith("Session regime COLD"):
        return "REGIME_COLD"
    if candidate.status == "STALE":
        return "STALE"
    if candidate.status not in DEPLOY_CLASS_STATUSES:
        return "NON_DEPLOYABLE_STATUS"
    if candidate.trailing_expr <= 0.0 or candidate.annual_r_estimate <= 0.0:
        return "NEGATIVE_CURRENT_EDGE"

    verdict = _norm_verdict(candidate.chordia_verdict)
    if verdict == "MISSING":
        return "MISSING_CHORDIA"
    if verdict == "PARK":
        return "CHORDIA_PARK"
    if verdict in FAIL_VERDICTS:
        return "CHORDIA_FAIL"
    if verdict in PASS_VERDICTS:
        return "NONE"
    return "UNKNOWN_CHORDIA_VERDICT"


def _state_action(blocker: str, *, active: bool, verdict: str) -> tuple[str, str]:
    if active and blocker != "NONE":
        return "LIVE_ACTIVE_REVIEW", "REMOVE_OR_REPLACE_ACTIVE_LANE"
    if active:
        return "LIVE_ACTIVE", "MONITOR_ACTIVE_LANE"
    if blocker == "NONE":
        if verdict == "PASS_PROTOCOL_A":
            return "ALLOCATOR_ELIGIBLE_BENCH", "EVALUATE_1_CONTRACT_ALLOCATION_SLOT"
        return "ALLOCATOR_ELIGIBLE_BENCH", "EVALUATE_ALLOCATION_SLOT"
    if blocker == "MISSING_CHORDIA":
        return "EXACT_LANE_READY_FOR_REPLAY", "RUN_STRICT_UNLOCK"
    if blocker == "CHORDIA_FAIL":
        return "KILLED", "DO_NOT_REOPEN_WITHOUT_NEW_PREREG"
    if blocker == "CHORDIA_PARK":
        return "PARKED", "REVIEW_PARK_THEN_RERUN_IF_PREREGISTERED"
    if blocker == "C8_OOS":
        return "PARKED", "RESOLVE_C8_OR_REVALIDATE"
    if blocker == "LIVE_TRADEABILITY":
        return "PARKED", "REDESIGN_OR_PARK_FILTER_CLASS"
    if blocker == "REGIME_COLD":
        return "PARKED", "RECHECK_REGIME_NEXT_REBALANCE"
    if blocker in {"PROFILE_INSTRUMENT", "PROFILE_SESSION"}:
        return "PARKED", "USE_DIFFERENT_PROFILE_OR_SKIP"
    if blocker == "NEGATIVE_CURRENT_EDGE":
        return "PARKED", "WAIT_FOR_POSITIVE_CURRENT_EDGE"
    return "PARKED", "INVESTIGATE_BLOCKER"


def build_bench_rows(
    candidates: list[UnlockCandidate],
    *,
    allowed_instruments: frozenset[str] | None,
    allowed_sessions: frozenset[str] | None,
    active_strategy_ids: frozenset[str],
    family_hints: dict[tuple[str, str, int, str, str, str], FamilyHint],
) -> tuple[BenchRow, ...]:
    rows: list[BenchRow] = []
    for candidate in candidates:
        active = candidate.strategy_id in active_strategy_ids
        verdict = _norm_verdict(candidate.chordia_verdict)
        blocker = _candidate_blocker(
            candidate,
            allowed_instruments=allowed_instruments,
            allowed_sessions=allowed_sessions,
        )
        state, action = _state_action(blocker, active=active, verdict=verdict)
        hint = _family_hint_for(candidate, family_hints)
        rows.append(
            BenchRow(
                rank=0,
                strategy_id=candidate.strategy_id,
                state=state,
                primary_blocker=blocker,
                next_action=action,
                instrument=candidate.instrument,
                session=candidate.session,
                orb_minutes=candidate.orb_minutes,
                entry_model=candidate.entry_model,
                rr_target=candidate.rr_target,
                filter_type=candidate.filter_type,
                confirm_bars=candidate.confirm_bars,
                trailing_expr=candidate.trailing_expr,
                trailing_n=candidate.trailing_n,
                annual_r_estimate=candidate.annual_r_estimate,
                status=candidate.status,
                status_reason=candidate.status_reason,
                chordia_verdict=verdict,
                chordia_audit_age_days=candidate.chordia_audit_age_days,
                c8_oos_status=candidate.c8_oos_status,
                family=hint.family if hint else None,
                family_role=hint.family_role if hint else None,
                family_priority=hint.priority if hint else 5,
                active_in_profile=active,
            )
        )

    rows.sort(
        key=lambda row: (
            _state_sort_key(row.state),
            row.family_priority,
            -row.annual_r_estimate,
            -row.trailing_expr,
            row.strategy_id,
        )
    )
    return tuple(_replace_rank(row, rank=i) for i, row in enumerate(rows, start=1))


def _state_sort_key(state: str) -> int:
    order = {
        "LIVE_ACTIVE_REVIEW": 0,
        "LIVE_ACTIVE": 1,
        "ALLOCATOR_ELIGIBLE_BENCH": 2,
        "EXACT_LANE_READY_FOR_REPLAY": 3,
        "PARKED": 4,
        "KILLED": 5,
    }
    return order.get(state, 9)


def _replace_rank(row: BenchRow, *, rank: int) -> BenchRow:
    data = asdict(row)
    data["rank"] = rank
    return BenchRow(**data)


def state_counts(rows: list[BenchRow] | tuple[BenchRow, ...]) -> dict[str, int]:
    return dict(sorted(Counter(row.state for row in rows).items()))


def blocker_counts(rows: list[BenchRow] | tuple[BenchRow, ...]) -> dict[str, int]:
    return dict(sorted(Counter(row.primary_blocker for row in rows).items()))


def render_report(
    rows: list[BenchRow] | tuple[BenchRow, ...],
    *,
    profile_id: str,
    source_notes: tuple[str, ...],
) -> str:
    row_list = list(rows)
    lines = [
        f"# Lane Bench State - {profile_id}",
        "",
        "This is a read-only lane bench state surface. It does not mutate live allocation JSON, "
        "validated_setups, or chordia_audit_log.yaml, and it does not make any lane live-selectable by itself.",
        "",
        "## Summary",
        "",
        f"- rows: {len(row_list)}",
        f"- state counts: {state_counts(row_list)}",
        f"- blocker counts: {blocker_counts(row_list)}",
    ]
    if source_notes:
        lines.extend(["", "## Sources", ""])
        lines.extend(f"- {note}" for note in source_notes)
    lines.extend(
        [
            "",
            "## Bench",
            "",
            "| rank | state | blocker | strategy_id | verdict | action |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in row_list[:150]:
        lines.append(
            f"| {row.rank} | {row.state} | {row.primary_blocker} | "
            f"`{row.strategy_id}` | {row.chordia_verdict} | {row.next_action} |"
        )
    return "\n".join(lines) + "\n"


def _active_strategy_ids(profile_id: str) -> frozenset[str]:
    result = resolve_allocation_json(profile_id)
    data = result.data
    if not isinstance(data, dict):
        return frozenset()
    out: set[str] = set()
    for entry in data.get("lanes", []):
        if not isinstance(entry, dict):
            continue
        sid = entry.get("strategy_id")
        status = str(entry.get("status") or "DEPLOY").upper()
        if sid and status in DEPLOY_CLASS_STATUSES:
            out.add(str(sid))
    return frozenset(out)


def _allowed_profile_sets(profile_id: str) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    profile = ACCOUNT_PROFILES[profile_id]
    allowed_instruments = frozenset(profile.allowed_instruments) if profile.allowed_instruments is not None else None
    allowed_sessions = frozenset(profile.allowed_sessions) if profile.allowed_sessions is not None else None
    return allowed_instruments, allowed_sessions


def _write_artifacts(
    rows: tuple[BenchRow, ...],
    *,
    output_dir: Path,
    profile_id: str,
    source_notes: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_id": profile_id,
        "row_count": len(rows),
        "state_counts": state_counts(rows),
        "blocker_counts": blocker_counts(rows),
        "live_mutation": False,
        "validated_setups_mutation": False,
        "chordia_log_mutation": False,
        "args": serializable_args,
        "source_notes": list(source_notes),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_report(rows, profile_id=profile_id, source_notes=source_notes),
        encoding="utf-8",
    )
    with (output_dir / "bench.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(BenchRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="topstep_50k_mnq_auto")
    parser.add_argument("--rebalance-date", default=date.today().isoformat())
    parser.add_argument("--inventory-cells", type=Path)
    parser.add_argument("--family-hints-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile_id = args.profile
    rebalance_date = date.fromisoformat(args.rebalance_date)
    source_notes = [f"rebalance_date={rebalance_date.isoformat()}"]

    family_hints: dict[tuple[str, str, int, str, str, str], FamilyHint] = {}
    if args.inventory_cells:
        family_hints = load_family_hints_from_cells(args.inventory_cells)
        source_notes.append(f"inventory_cells={args.inventory_cells}")
    if args.family_hints_csv:
        family_hints.update(load_family_hints_from_bench(args.family_hints_csv))
        source_notes.append(f"family_hints_csv={args.family_hints_csv}")

    scores = compute_lane_scores(rebalance_date, db_path=args.db_path)
    scores = apply_live_tradeability_gate(apply_c8_gate(scores))
    candidates = [candidate_from_lane_score(score) for score in scores]
    allowed_instruments, allowed_sessions = _allowed_profile_sets(profile_id)
    rows = build_bench_rows(
        candidates,
        allowed_instruments=allowed_instruments,
        allowed_sessions=allowed_sessions,
        active_strategy_ids=_active_strategy_ids(profile_id),
        family_hints=family_hints,
    )

    if args.output_dir:
        _write_artifacts(
            rows,
            output_dir=args.output_dir,
            profile_id=profile_id,
            source_notes=tuple(source_notes),
            args=args,
        )

    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    else:
        print(render_report(rows, profile_id=profile_id, source_notes=tuple(source_notes)))
        if args.output_dir:
            print(f"Wrote artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
