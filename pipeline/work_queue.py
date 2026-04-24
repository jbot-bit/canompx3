"""Canonical active-work queue and local session-lease helpers.

Tracked state lives in ``docs/runtime/action-queue.yaml``.
Local multi-terminal ownership lives in ``.session/work_queue_leases.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_SCHEMA_VERSION = 1
LEASE_SCHEMA_VERSION = 1

QueueClass = Literal["research", "runtime", "deploy", "ops", "tooling", "docs", "debt"]
QueueStatus = Literal["ready", "active", "blocked", "waiting_observation", "parked", "closed", "superseded"]
QueuePriority = Literal["P1", "P2", "P3"]

OPEN_STATUSES: set[QueueStatus] = {"ready", "active", "blocked", "waiting_observation"}
BATON_STATUSES: set[QueueStatus] = {"ready", "active", "blocked", "waiting_observation"}
PRIORITY_RANK = {"P1": 0, "P2": 1, "P3": 2}
STATUS_RANK = {
    "active": 0,
    "ready": 1,
    "blocked": 2,
    "waiting_observation": 3,
    "parked": 4,
    "closed": 5,
    "superseded": 6,
}

HANDOFF_HEADER = """# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.
"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkQueueItem(StrictModel):
    id: str
    title: str
    class_: QueueClass = Field(alias="class")
    status: QueueStatus
    priority: QueuePriority
    close_before_new_work: bool = False
    owner_hint: str | None = None
    last_verified_at: str
    freshness_sla_days: int = 7
    next_action: str
    exit_criteria: str
    blocked_by: list[str] = Field(default_factory=list)
    decision_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    notes_ref: str | None = None
    override_note: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def is_baton_active(self) -> bool:
        return self.status in BATON_STATUSES


class WorkQueue(StrictModel):
    schema_version: int = QUEUE_SCHEMA_VERSION
    updated_at: str | None = None
    items: list[WorkQueueItem] = Field(default_factory=list)


class SessionLease(StrictModel):
    session_id: str
    tool: str
    branch: str
    worktree: str
    claimed_item_ids: list[str] = Field(default_factory=list)
    last_heartbeat_at: str


class LeaseCollection(StrictModel):
    schema_version: int = LEASE_SCHEMA_VERSION
    leases: list[SessionLease] = Field(default_factory=list)


class QueueItemSummary(StrictModel):
    id: str
    title: str
    status: QueueStatus
    priority: QueuePriority
    close_before_new_work: bool = False


class LeaseConflict(StrictModel):
    item_id: str
    session_id: str
    tool: str
    branch: str
    worktree: str


class WorkQueueSnapshot(StrictModel):
    exists: bool
    path: str
    handoff_matches_rendered: bool | None = None
    open_count: int = 0
    close_first_open_count: int = 0
    stale_count: int = 0
    top_items: list[QueueItemSummary] = Field(default_factory=list)
    close_first_items: list[QueueItemSummary] = Field(default_factory=list)
    stale_items: list[QueueItemSummary] = Field(default_factory=list)
    lease_conflicts: list[LeaseConflict] = Field(default_factory=list)


def queue_path(root: Path) -> Path:
    return root / "docs" / "runtime" / "action-queue.yaml"


def lease_path(root: Path) -> Path:
    return root / ".session" / "work_queue_leases.json"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if len(value) == 10:
            return datetime.fromisoformat(f"{value}T00:00:00+00:00")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def load_queue(root: Path = PROJECT_ROOT) -> WorkQueue:
    path = queue_path(root)
    if not path.exists():
        return WorkQueue(updated_at=None, items=[])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data.get("updated_at"), datetime):
        data["updated_at"] = data["updated_at"].astimezone(UTC).isoformat()
    for item in data.get("items", []) or []:
        last_verified = item.get("last_verified_at")
        if hasattr(last_verified, "isoformat"):
            item["last_verified_at"] = last_verified.isoformat()
    return WorkQueue.model_validate(data)


def save_queue(queue: WorkQueue, root: Path = PROJECT_ROOT) -> Path:
    path = queue_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = queue.model_dump(by_alias=True, mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def load_leases(root: Path = PROJECT_ROOT) -> LeaseCollection:
    path = lease_path(root)
    if not path.exists():
        return LeaseCollection()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LeaseCollection()
    return LeaseCollection.model_validate(data)


def save_leases(collection: LeaseCollection, root: Path = PROJECT_ROOT) -> Path:
    path = lease_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def fresh_leases(
    root: Path = PROJECT_ROOT, *, now: datetime | None = None, freshness_hours: int = 8
) -> list[SessionLease]:
    cutoff = (now or _utc_now()) - timedelta(hours=freshness_hours)
    leases = []
    for lease in load_leases(root).leases:
        last_seen = _parse_dt(lease.last_heartbeat_at)
        if last_seen is not None and last_seen >= cutoff:
            leases.append(lease)
    return leases


def sorted_items(items: list[WorkQueueItem]) -> list[WorkQueueItem]:
    return sorted(
        items,
        key=lambda item: (
            PRIORITY_RANK.get(item.priority, 99),
            0 if item.close_before_new_work else 1,
            STATUS_RANK.get(item.status, 99),
            item.title.lower(),
        ),
    )


def item_summary(item: WorkQueueItem) -> QueueItemSummary:
    return QueueItemSummary(
        id=item.id,
        title=item.title,
        status=item.status,
        priority=item.priority,
        close_before_new_work=item.close_before_new_work,
    )


def stale_items(queue: WorkQueue, *, now: datetime | None = None) -> list[WorkQueueItem]:
    reference = now or _utc_now()
    stale: list[WorkQueueItem] = []
    for item in queue.items:
        if not item.is_open:
            continue
        last_verified = _parse_dt(item.last_verified_at)
        if last_verified is None:
            stale.append(item)
            continue
        if last_verified + timedelta(days=max(item.freshness_sla_days, 0)) < reference:
            stale.append(item)
    return sorted_items(stale)


def top_baton_items(queue: WorkQueue, *, limit: int = 3) -> list[WorkQueueItem]:
    candidates = [item for item in queue.items if item.is_baton_active]
    return sorted_items(candidates)[:limit]


def close_first_open_items(queue: WorkQueue) -> list[WorkQueueItem]:
    items = [item for item in queue.items if item.is_open and item.close_before_new_work]
    return sorted_items(items)


def queue_snapshot(root: Path = PROJECT_ROOT, *, now: datetime | None = None) -> WorkQueueSnapshot:
    path = queue_path(root)
    if not path.exists():
        return WorkQueueSnapshot(exists=False, path=str(path))

    queue = load_queue(root)
    stale = stale_items(queue, now=now)
    close_first = close_first_open_items(queue)
    top = top_baton_items(queue)
    conflicts = lease_conflicts(root=root, candidate_item_ids=[item.id for item in top], session_id=None)
    open_count = len([item for item in queue.items if item.is_open])

    handoff_path = root / "HANDOFF.md"
    rendered = render_handoff_text(root) if handoff_path.exists() else None
    matches = (
        None
        if rendered is None
        else handoff_path.read_text(encoding="utf-8", errors="replace").strip() == rendered.strip()
    )

    return WorkQueueSnapshot(
        exists=True,
        path=str(path),
        handoff_matches_rendered=matches,
        open_count=open_count,
        close_first_open_count=len(close_first),
        stale_count=len(stale),
        top_items=[item_summary(item) for item in top],
        close_first_items=[item_summary(item) for item in close_first],
        stale_items=[item_summary(item) for item in stale],
        lease_conflicts=conflicts,
    )


def _find_item(queue: WorkQueue, item_id: str) -> WorkQueueItem:
    for item in queue.items:
        if item.id == item_id:
            return item
    raise KeyError(f"Unknown queue item: {item_id}")


def claim_requires_override(queue: WorkQueue, claimed_item_id: str) -> bool:
    close_first_ids = {item.id for item in close_first_open_items(queue)}
    return bool(close_first_ids and claimed_item_id not in close_first_ids)


def lease_conflicts(
    *,
    root: Path = PROJECT_ROOT,
    candidate_item_ids: list[str],
    session_id: str | None,
) -> list[LeaseConflict]:
    conflicts: list[LeaseConflict] = []
    wanted = set(candidate_item_ids)
    if not wanted:
        return conflicts
    for lease in fresh_leases(root):
        if session_id and lease.session_id == session_id:
            continue
        overlap = sorted(wanted.intersection(lease.claimed_item_ids))
        for item_id in overlap:
            conflicts.append(
                LeaseConflict(
                    item_id=item_id,
                    session_id=lease.session_id,
                    tool=lease.tool,
                    branch=lease.branch,
                    worktree=lease.worktree,
                )
            )
    return conflicts


def upsert_lease(
    *,
    root: Path = PROJECT_ROOT,
    session_id: str,
    tool: str,
    branch: str,
    worktree: str,
    claimed_item_ids: list[str],
    heartbeat_at: str | None = None,
) -> SessionLease:
    collection = load_leases(root)
    lease = SessionLease(
        session_id=session_id,
        tool=tool,
        branch=branch,
        worktree=worktree,
        claimed_item_ids=sorted(set(claimed_item_ids)),
        last_heartbeat_at=heartbeat_at or _utc_now().isoformat(),
    )
    remaining = [existing for existing in collection.leases if existing.session_id != session_id]
    collection.leases = [*remaining, lease]
    save_leases(collection, root)
    return lease


def heartbeat_lease(root: Path = PROJECT_ROOT, *, session_id: str) -> SessionLease | None:
    collection = load_leases(root)
    for idx, existing in enumerate(collection.leases):
        if existing.session_id != session_id:
            continue
        updated = existing.model_copy(update={"last_heartbeat_at": _utc_now().isoformat()})
        collection.leases[idx] = updated
        save_leases(collection, root)
        return updated
    return None


def close_item(
    root: Path = PROJECT_ROOT,
    *,
    item_id: str,
    status: Literal["closed", "superseded"],
    override_note: str | None = None,
) -> WorkQueueItem:
    queue = load_queue(root)
    item = _find_item(queue, item_id)
    updated = item.model_copy(
        update={
            "status": status,
            "last_verified_at": _utc_now().date().isoformat(),
            "override_note": override_note or item.override_note,
        }
    )
    queue.items = [updated if existing.id == item_id else existing for existing in queue.items]
    queue.updated_at = _utc_now().isoformat()
    save_queue(queue, root)
    return updated


def record_override(root: Path = PROJECT_ROOT, *, item_id: str, note: str) -> WorkQueueItem:
    queue = load_queue(root)
    item = _find_item(queue, item_id)
    updated = item.model_copy(update={"override_note": note, "last_verified_at": _utc_now().date().isoformat()})
    queue.items = [updated if existing.id == item_id else existing for existing in queue.items]
    queue.updated_at = _utc_now().isoformat()
    save_queue(queue, root)
    return updated


def claim_item(
    root: Path = PROJECT_ROOT,
    *,
    item_id: str,
    session_id: str,
    tool: str,
    branch: str,
    worktree: str,
    override_note: str | None = None,
) -> SessionLease:
    queue = load_queue(root)
    item = _find_item(queue, item_id)
    if item.status in {"closed", "superseded"}:
        raise ValueError(f"Cannot claim queue item {item_id}: status={item.status}")

    conflicts = lease_conflicts(root=root, candidate_item_ids=[item_id], session_id=session_id)
    if conflicts:
        detail = ", ".join(f"{conflict.tool}:{conflict.session_id}" for conflict in conflicts)
        raise ValueError(f"Queue item already claimed by a fresh session: {detail}")

    if claim_requires_override(queue, item_id):
        if not override_note:
            raise ValueError(
                "Close-first carry-over items remain open. Record an override note before claiming new work."
            )
        record_override(root, item_id=item_id, note=override_note)

    return upsert_lease(
        root=root,
        session_id=session_id,
        tool=tool,
        branch=branch,
        worktree=worktree,
        claimed_item_ids=[item_id],
    )


def release_session(root: Path = PROJECT_ROOT, *, session_id: str) -> None:
    collection = load_leases(root)
    original_count = len(collection.leases)
    collection.leases = [lease for lease in collection.leases if lease.session_id != session_id]
    if len(collection.leases) != original_count:
        save_leases(collection, root)


def _parse_handoff_metadata(handoff_path: Path) -> tuple[str | None, str | None, str | None]:
    if not handoff_path.exists():
        return None, None, None
    tool = date = summary = None
    for line in handoff_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("- **Tool:** "):
            tool = line.removeprefix("- **Tool:** ").strip()
        elif line.startswith("- **Date:** "):
            date = line.removeprefix("- **Date:** ").strip()
        elif line.startswith("- **Summary:** "):
            summary = line.removeprefix("- **Summary:** ").strip()
        if tool and date and summary:
            break
    return tool, date, summary


def render_handoff_text(
    root: Path = PROJECT_ROOT,
    *,
    tool: str | None = None,
    date: str | None = None,
    summary: str | None = None,
) -> str:
    queue = load_queue(root)
    handoff_path = root / "HANDOFF.md"
    existing_tool, existing_date, existing_summary = _parse_handoff_metadata(handoff_path)
    resolved_tool = tool or existing_tool or "Codex"
    resolved_date = date or existing_date or _utc_now().date().isoformat()
    resolved_summary = (
        summary or existing_summary or "Queue-backed baton refreshed from canonical active-work registry."
    )

    next_steps = [f"{item.title} — {item.next_action}" for item in top_baton_items(queue)]
    stale = stale_items(queue)
    close_first = close_first_open_items(queue)
    conflicts = lease_conflicts(
        root=root, candidate_item_ids=[item.id for item in top_baton_items(queue)], session_id=None
    )

    lines = [
        HANDOFF_HEADER.rstrip(),
        "",
        "## Last Session",
        f"- **Tool:** {resolved_tool}",
        f"- **Date:** {resolved_date}",
        f"- **Summary:** {resolved_summary}",
    ]

    if next_steps:
        lines.extend(["", "## Next Steps — Active"])
        for idx, step in enumerate(next_steps, start=1):
            lines.append(f"{idx}. {step}")

    warnings: list[str] = []
    if close_first:
        warnings.append("Close-first carry-over items remain open: " + ", ".join(item.id for item in close_first[:5]))
    if stale:
        warnings.append("Stale queue items need re-verification: " + ", ".join(item.id for item in stale[:5]))
    if conflicts:
        warnings.append(
            "Fresh session lease overlap on active items: "
            + ", ".join(f"{conflict.item_id}:{conflict.tool}" for conflict in conflicts[:5])
        )

    if warnings:
        lines.extend(["", "## Blockers / Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Durable References",
            "- `docs/runtime/action-queue.yaml`",
            "- `docs/runtime/decision-ledger.md`",
            "- `docs/runtime/debt-ledger.md`",
            "- `docs/plans/2026-04-22-handoff-baton-compaction.md`",
            "",
        ]
    )
    return "\n".join(lines)


def write_rendered_handoff(
    root: Path = PROJECT_ROOT,
    *,
    tool: str | None = None,
    date: str | None = None,
    summary: str | None = None,
) -> Path:
    handoff = root / "HANDOFF.md"
    handoff.write_text(render_handoff_text(root, tool=tool, date=date, summary=summary), encoding="utf-8")
    return handoff
