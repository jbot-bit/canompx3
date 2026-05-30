"""Build a read-only batch queue for Chordia strict unlock work.

This tool does not promote strategies, write live allocation JSON, or mutate
``docs/runtime/chordia_audit_log.yaml``. It turns the current deployable shelf
plus optional family-level research inventory hints into an exact-lane work
queue so missing strict replay verdicts can be processed in batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trading_app.lane_allocator import (  # noqa: E402
    LaneScore,
    apply_c8_gate,
    apply_live_tradeability_gate,
    compute_lane_scores,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, resolve_allocation_json  # noqa: E402

DEPLOY_CLASS_STATUSES = frozenset({"DEPLOY", "RESUME", "PROVISIONAL"})
DEFAULT_UNLOCK_VERDICTS = frozenset({"MISSING"})
CORE_MNQ_BASELINE_SESSIONS = frozenset({"NYSE_OPEN", "US_DATA_1000"})


@dataclass(frozen=True)
class UnlockCandidate:
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
    status: str
    status_reason: str
    chordia_verdict: str | None
    chordia_audit_age_days: int | None
    c8_oos_status: str | None


@dataclass(frozen=True)
class FamilyHint:
    instrument: str
    session: str
    orb_minutes: int
    entry_model: str
    rr_target: float
    filter_type: str
    family: str
    family_role: str
    family_median_exp_r: float | None
    family_oos_median: float | None
    priority: int


@dataclass(frozen=True)
class FamilySummary:
    median_exp_r: float | None
    median_oos: float | None
    deployable_candidates: int
    t3_cells: int


@dataclass(frozen=True)
class UnlockQueueRow:
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
    status: str
    chordia_verdict: str
    chordia_audit_age_days: int | None
    c8_oos_status: str | None
    family: str | None
    family_role: str | None
    family_priority: int
    family_median_exp_r: float | None
    family_oos_median: float | None
    active_in_profile: bool
    action: str

    @classmethod
    def from_candidate(
        cls,
        *,
        rank: int,
        candidate: UnlockCandidate,
        family_hint: FamilyHint | None,
        active: bool,
    ) -> UnlockQueueRow:
        verdict = _norm_verdict(candidate.chordia_verdict)
        if verdict == "MISSING":
            action = "RUN_STRICT_UNLOCK"
        elif verdict == "PARK":
            action = "REVIEW_PARK_THEN_RERUN_IF_PREREGISTERED"
        else:
            action = "REVIEW_EXISTING_VERDICT"
        return cls(
            rank=rank,
            strategy_id=candidate.strategy_id,
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
            chordia_verdict=verdict,
            chordia_audit_age_days=candidate.chordia_audit_age_days,
            c8_oos_status=candidate.c8_oos_status,
            family=family_hint.family if family_hint else None,
            family_role=family_hint.family_role if family_hint else None,
            family_priority=family_hint.priority if family_hint else 5,
            family_median_exp_r=family_hint.family_median_exp_r if family_hint else None,
            family_oos_median=family_hint.family_oos_median if family_hint else None,
            active_in_profile=active,
            action=action,
        )


def candidate_key(
    instrument: str,
    session: str,
    orb_minutes: int,
    entry_model: str,
    rr_target: float,
    filter_type: str,
) -> tuple[str, str, int, str, str, str]:
    return (
        instrument.upper(),
        session.upper(),
        int(orb_minutes),
        entry_model.upper(),
        f"{float(rr_target):.4g}",
        (filter_type or "NO_FILTER").upper(),
    )


def _norm_verdict(verdict: str | None) -> str:
    if verdict is None or verdict == "":
        return "MISSING"
    return verdict.upper()


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_zero(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _priority_for_inventory_row(row: dict[str, str], summary: FamilySummary | None) -> int | None:
    classification = (row.get("classification") or "").upper()
    family = (row.get("family") or "").lower()
    variant = (row.get("variant") or "NO_FILTER").upper()
    instrument = (row.get("instrument") or "").upper()
    session = (row.get("session") or "").upper()
    entry_model = (row.get("entry_model") or "").upper()

    if not (
        classification.startswith("DEPLOYABLE_CANDIDATE")
        or classification.startswith("RESEARCH_PROVISIONAL")
    ):
        return None
    if summary is not None:
        if (
            family == "baseline_orb"
            and variant == "NO_FILTER"
            and instrument == "MNQ"
            and session in CORE_MNQ_BASELINE_SESSIONS
            and entry_model == "E2"
            and summary.deployable_candidates >= 6
            and (summary.median_exp_r or 0.0) > 0.0
            and (summary.median_oos or 0.0) > 0.0
        ):
            return 0
        if (
            family == "baseline_orb"
            and variant == "NO_FILTER"
            and instrument == "MNQ"
            and session == "CME_PRECLOSE"
            and entry_model == "E2"
            and summary.deployable_candidates >= 3
            and (summary.median_exp_r or 0.0) > 0.0
        ):
            return 4
        return None
    if (
        family == "baseline_orb"
        and variant == "NO_FILTER"
        and instrument == "MNQ"
        and session in CORE_MNQ_BASELINE_SESSIONS
        and entry_model == "E2"
    ):
        return 0
    if classification.startswith("DEPLOYABLE_CANDIDATE"):
        return 2
    return 4


def _load_family_summary(path: Path) -> dict[tuple[str, str, str], FamilySummary]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], FamilySummary] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (
                (row.get("family") or "").lower(),
                (row.get("instrument") or "").upper(),
                (row.get("session") or "").upper(),
            )
            out[key] = FamilySummary(
                median_exp_r=_float_or_none(row.get("median_exp_r")),
                median_oos=_float_or_none(row.get("median_oos")),
                deployable_candidates=_int_or_zero(row.get("deployable_candidates")),
                t3_cells=_int_or_zero(row.get("t3_cells")),
            )
    return out


def load_family_hints_from_cells(path: Path) -> dict[tuple[str, str, int, str, str, str], FamilyHint]:
    summary = _load_family_summary(path.with_name("family_summary.csv"))
    hints: dict[tuple[str, str, int, str, str, str], FamilyHint] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            instrument = (row.get("instrument") or "").upper()
            session = (row.get("session") or "").upper()
            family = row.get("family") or "unknown"
            summary_key = (family.lower(), instrument, session)
            family_summary = summary.get(summary_key)
            priority = _priority_for_inventory_row(row, family_summary)
            if priority is None:
                continue
            median_exp_r = (
                family_summary.median_exp_r if family_summary else _float_or_none(row.get("exp_r_is"))
            )
            median_oos = (
                family_summary.median_oos if family_summary else _float_or_none(row.get("exp_r_oos"))
            )
            key = candidate_key(
                instrument,
                session,
                _int_or_zero(row.get("orb_minutes")),
                row.get("entry_model") or "",
                _float_or_none(row.get("rr_target")) or 0.0,
                row.get("variant") or "NO_FILTER",
            )
            hint = FamilyHint(
                instrument=instrument,
                session=session,
                orb_minutes=key[2],
                entry_model=key[3],
                rr_target=float(key[4]),
                filter_type=key[5],
                family=f"{instrument} {session} {family}",
                family_role="deployable_candidate"
                if (row.get("classification") or "").upper().startswith("DEPLOYABLE_CANDIDATE")
                else "research_provisional",
                family_median_exp_r=median_exp_r,
                family_oos_median=median_oos,
                priority=priority,
            )
            existing = hints.get(key)
            if existing is None or hint.priority < existing.priority:
                hints[key] = hint
    return hints


def candidate_from_lane_score(score: LaneScore) -> UnlockCandidate:
    return UnlockCandidate(
        strategy_id=score.strategy_id,
        instrument=score.instrument,
        session=score.orb_label,
        orb_minutes=score.orb_minutes,
        entry_model=score.entry_model,
        rr_target=score.rr_target,
        filter_type=score.filter_type,
        confirm_bars=score.confirm_bars,
        trailing_expr=score.trailing_expr,
        trailing_n=score.trailing_n,
        annual_r_estimate=score.annual_r_estimate,
        status=score.status,
        status_reason=score.status_reason,
        chordia_verdict=score.chordia_verdict,
        chordia_audit_age_days=score.chordia_audit_age_days,
        c8_oos_status=score.c8_oos_status,
    )


def build_unlock_queue(
    candidates: Iterable[UnlockCandidate],
    *,
    allowed_instruments: frozenset[str] | None,
    allowed_sessions: frozenset[str] | None,
    active_strategy_ids: frozenset[str],
    family_hints: dict[tuple[str, str, int, str, str, str], FamilyHint],
    limit: int,
    unlock_verdicts: frozenset[str] = DEFAULT_UNLOCK_VERDICTS,
    include_active: bool = False,
) -> tuple[UnlockQueueRow, ...]:
    selected: list[tuple[UnlockCandidate, FamilyHint | None, bool]] = []
    norm_unlock_verdicts = frozenset(v.upper() for v in unlock_verdicts)
    for candidate in candidates:
        if candidate.status not in DEPLOY_CLASS_STATUSES:
            continue
        active = candidate.strategy_id in active_strategy_ids
        if active and not include_active:
            continue
        if allowed_instruments is not None and candidate.instrument not in allowed_instruments:
            continue
        if allowed_sessions is not None and candidate.session not in allowed_sessions:
            continue
        if _norm_verdict(candidate.chordia_verdict) not in norm_unlock_verdicts:
            continue
        if candidate.trailing_expr <= 0.0 or candidate.annual_r_estimate <= 0.0:
            continue
        key = candidate_key(
            candidate.instrument,
            candidate.session,
            candidate.orb_minutes,
            candidate.entry_model,
            candidate.rr_target,
            candidate.filter_type,
        )
        selected.append((candidate, family_hints.get(key), active))

    selected.sort(
        key=lambda item: (
            item[1].priority if item[1] else 5,
            -float(item[0].annual_r_estimate or 0.0),
            -float(item[0].trailing_expr or 0.0),
            -int(item[0].trailing_n or 0),
            item[0].strategy_id,
        )
    )
    return tuple(
        UnlockQueueRow.from_candidate(rank=i, candidate=candidate, family_hint=hint, active=active)
        for i, (candidate, hint, active) in enumerate(selected[:limit], start=1)
    )


def render_report(rows: Iterable[UnlockQueueRow], *, profile_id: str, source_notes: tuple[str, ...]) -> str:
    row_list = list(rows)
    priority_counts: dict[int, int] = {}
    for row in row_list:
        priority_counts[row.family_priority] = priority_counts.get(row.family_priority, 0) + 1
    lines = [
        f"# Chordia Unlock Batch Queue - {profile_id}",
        "",
        "This artifact is a read-only work queue. It does not mutate live allocation JSON, "
        "does not mutate chordia_audit_log.yaml, and does not make any lane live-selectable by itself.",
        "",
        "## Summary",
        "",
        f"- candidates: {len(row_list)}",
        "- capital rule: strict Chordia/live-readiness gates remain binding",
        "- priority 0: family-backed MNQ NYSE_OPEN/US_DATA_1000 baseline rows",
        "- priority 4: provisional family rows needing extra cushion/cost proof",
        "- priority 5: validated-shelf rows without inventory-family priority",
        f"- priority counts: {dict(sorted(priority_counts.items()))}",
        "- next step: run exact-lane strict replay/prereg work for queued rows, then update doctrine logs through the existing review path",
    ]
    if source_notes:
        lines.extend(["", "## Sources", ""])
        lines.extend(f"- {note}" for note in source_notes)

    lines.extend(
        [
            "",
            "## Queue",
            "",
            "| rank | strategy_id | family_priority | annual_r | exp_r | n | verdict | action |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in row_list[:100]:
        lines.append(
            f"| {row.rank} | `{row.strategy_id}` | {row.family_priority} | "
            f"{row.annual_r_estimate:.2f} | {row.trailing_expr:.4f} | "
            f"{row.trailing_n} | {row.chordia_verdict} | {row.action} |"
        )
    return "\n".join(lines) + "\n"


def _active_strategy_ids(profile_id: str) -> frozenset[str]:
    result = resolve_allocation_json(profile_id)
    data = result.data
    if not isinstance(data, dict):
        return frozenset()
    ids: set[str] = set()
    for entry in data.get("lanes", []):
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "DEPLOY").upper()
        sid = entry.get("strategy_id")
        if sid and status in DEPLOY_CLASS_STATUSES:
            ids.add(str(sid))
    return frozenset(ids)


def _allowed_profile_sets(profile_id: str) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    profile = ACCOUNT_PROFILES[profile_id]
    allowed_instruments = (
        frozenset(profile.allowed_instruments) if profile.allowed_instruments is not None else None
    )
    allowed_sessions = frozenset(profile.allowed_sessions) if profile.allowed_sessions is not None else None
    return allowed_instruments, allowed_sessions


def _write_artifacts(
    rows: tuple[UnlockQueueRow, ...],
    *,
    output_dir: Path,
    profile_id: str,
    source_notes: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_id": profile_id,
        "candidate_count": len(rows),
        "live_mutation": False,
        "chordia_log_mutation": False,
        "args": serializable_args,
        "source_notes": list(source_notes),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_report(rows, profile_id=profile_id, source_notes=source_notes),
        encoding="utf-8",
    )
    with (output_dir / "queue.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(UnlockQueueRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="topstep_50k_mnq_auto")
    parser.add_argument("--rebalance-date", default=date.today().isoformat())
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--inventory-cells", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--include-active", action="store_true")
    parser.add_argument("--include-park", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rebalance_date = date.fromisoformat(args.rebalance_date)
    profile_id = args.profile
    allowed_instruments, allowed_sessions = _allowed_profile_sets(profile_id)
    source_notes = [f"rebalance_date={rebalance_date.isoformat()}"]

    family_hints: dict[tuple[str, str, int, str, str, str], FamilyHint] = {}
    if args.inventory_cells:
        family_hints = load_family_hints_from_cells(args.inventory_cells)
        source_notes.append(f"inventory_cells={args.inventory_cells}")

    scores = compute_lane_scores(rebalance_date, db_path=args.db_path)
    scores = apply_live_tradeability_gate(apply_c8_gate(scores))
    candidates = [candidate_from_lane_score(score) for score in scores]
    unlock_verdicts = {"MISSING", "PARK"} if args.include_park else {"MISSING"}
    rows = build_unlock_queue(
        candidates,
        allowed_instruments=allowed_instruments,
        allowed_sessions=allowed_sessions,
        active_strategy_ids=_active_strategy_ids(profile_id),
        family_hints=family_hints,
        limit=args.limit,
        unlock_verdicts=frozenset(unlock_verdicts),
        include_active=args.include_active,
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
