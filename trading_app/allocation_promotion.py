"""Promotion utilities for moving vetted proposal rows into lane allocation.

This module is intentionally profile/instrument agnostic. Research runners can
decide which candidates are eligible; this layer only converts explicit
``PASS_*`` promotion records into a runtime allocation patch while preserving
allocation safety invariants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

DEPLOYABLE_STATUSES = {"DEPLOY", "PROVISIONAL"}
PASS_DECISIONS = {"PASS_ADD", "PASS_REPLACE"}
PASS_CHORDIA_VERDICTS = {"PASS_CHORDIA", "PASS_PROTOCOL_A"}


class AllocationPromotionError(ValueError):
    """Raised when a proposal row cannot safely mutate an allocation."""


@dataclass(frozen=True)
class PromotionCandidate:
    profile_id: str
    strategy_id: str
    decision: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    filter_type: str
    status: str
    status_reason: str
    chordia_verdict: str
    chordia_audit_age_days: int | None
    annual_r: float | None
    trailing_expr: float | None
    trailing_n: int | None
    trailing_wr: float | None
    months_negative: int | None
    session_regime: str | None
    avg_orb_pts: float | None
    p90_orb_pts: float | None
    replacement_target: str | None
    replacement_target_status: str | None
    source_path: str
    account_risk_detail: str
    confirm_bars: int = 1


@dataclass(frozen=True)
class PromotionPatchResult:
    allocation: dict[str, Any]
    promoted: tuple[str, ...]
    removed_lane_ids: tuple[str, ...]
    removed_block_ids: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(self.allocation, indent=2, sort_keys=False) + "\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocationPromotionError(message)


def _lane_from_candidate(candidate: PromotionCandidate) -> dict[str, Any]:
    lane: dict[str, Any] = {
        "strategy_id": candidate.strategy_id,
        "instrument": candidate.instrument,
        "orb_label": candidate.orb_label,
        "orb_minutes": candidate.orb_minutes,
        "rr_target": candidate.rr_target,
        "filter_type": candidate.filter_type,
        "annual_r": candidate.annual_r,
        "trailing_expr": candidate.trailing_expr,
        "trailing_n": candidate.trailing_n,
        "trailing_wr": candidate.trailing_wr,
        "months_negative": candidate.months_negative,
        "session_regime": candidate.session_regime,
        "status": candidate.status,
        "status_reason": candidate.status_reason,
        "chordia_verdict": candidate.chordia_verdict,
        "chordia_audit_age_days": candidate.chordia_audit_age_days,
        "promotion_source": candidate.source_path,
        "promotion_decision": candidate.decision,
        "promotion_account_risk": candidate.account_risk_detail,
    }
    if candidate.avg_orb_pts is not None:
        lane["avg_orb_pts"] = candidate.avg_orb_pts
    if candidate.p90_orb_pts is not None:
        lane["p90_orb_pts"] = candidate.p90_orb_pts
    return lane


def _same_runtime_slot(left: dict[str, Any], right: PromotionCandidate) -> bool:
    return left.get("instrument") == right.instrument and left.get("orb_label") == right.orb_label


def _validate_candidate(candidate: PromotionCandidate, allocation: dict[str, Any]) -> None:
    profile_id = allocation.get("profile_id")
    _require(candidate.profile_id == profile_id, f"profile mismatch for {candidate.strategy_id}: {candidate.profile_id} != {profile_id}")
    _require(candidate.decision in PASS_DECISIONS, f"{candidate.strategy_id}: decision {candidate.decision!r} is not promotable")
    _require(
        candidate.status in DEPLOYABLE_STATUSES,
        f"{candidate.strategy_id}: status {candidate.status!r} is not runtime deployable",
    )
    _require(
        candidate.chordia_verdict in PASS_CHORDIA_VERDICTS,
        f"{candidate.strategy_id}: Chordia verdict {candidate.chordia_verdict!r} is not promotable",
    )
    _require(candidate.chordia_audit_age_days is not None, f"{candidate.strategy_id}: missing Chordia audit age")
    _require(candidate.p90_orb_pts is not None, f"{candidate.strategy_id}: missing p90_orb_pts for risk loader")
    if candidate.decision == "PASS_REPLACE":
        _require(
            bool(candidate.replacement_target),
            f"{candidate.strategy_id}: PASS_REPLACE requires replacement_target",
        )


def apply_promotions(
    allocation: dict[str, Any],
    candidates: list[PromotionCandidate],
    *,
    rebalance_date: date,
) -> PromotionPatchResult:
    """Apply vetted promotion candidates to a lane allocation payload.

    The function is pure: callers decide whether and where to write the result.
    """
    _require(bool(candidates), "no promotion candidates supplied")
    next_allocation = json.loads(json.dumps(allocation))
    lanes = list(next_allocation.get("lanes") or [])
    paused = list(next_allocation.get("paused") or [])
    stale = list(next_allocation.get("stale") or [])

    promoted: list[str] = []
    removed_lane_ids: list[str] = []
    removed_block_ids: list[str] = []

    for candidate in candidates:
        _validate_candidate(candidate, next_allocation)

        remove_ids = {candidate.strategy_id}
        if candidate.replacement_target:
            remove_ids.add(candidate.replacement_target)

        kept_lanes: list[dict[str, Any]] = []
        for lane in lanes:
            sid = str(lane.get("strategy_id", ""))
            remove = sid in remove_ids or _same_runtime_slot(lane, candidate)
            if remove:
                removed_lane_ids.append(sid)
            else:
                kept_lanes.append(lane)
        lanes = kept_lanes

        kept_paused: list[dict[str, Any]] = []
        for row in paused:
            sid = str(row.get("strategy_id", ""))
            if sid in remove_ids:
                removed_block_ids.append(sid)
            else:
                kept_paused.append(row)
        paused = kept_paused

        kept_stale: list[dict[str, Any]] = []
        for row in stale:
            sid = str(row.get("strategy_id", ""))
            if sid in remove_ids:
                removed_block_ids.append(sid)
            else:
                kept_stale.append(row)
        stale = kept_stale

        lanes.append(_lane_from_candidate(candidate))
        promoted.append(candidate.strategy_id)

    next_allocation["rebalance_date"] = rebalance_date.isoformat()
    next_allocation["lanes"] = lanes
    next_allocation["paused"] = paused
    next_allocation["stale"] = stale
    next_allocation["promotion_patch"] = {
        "applied_on": rebalance_date.isoformat(),
        "promoted": promoted,
        "removed_lane_ids": sorted(set(removed_lane_ids)),
        "removed_block_ids": sorted(set(removed_block_ids)),
    }

    return PromotionPatchResult(
        allocation=next_allocation,
        promoted=tuple(promoted),
        removed_lane_ids=tuple(sorted(set(removed_lane_ids))),
        removed_block_ids=tuple(sorted(set(removed_block_ids))),
    )
