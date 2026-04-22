#!/usr/bin/env python3
"""Build a deterministic MNQ hiROI frontier queue from board outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str | None) -> float:
    raw = (value or "").strip()
    if not raw or raw.lower() == "nan":
        return float("-inf")
    return float(raw)


def _as_int(value: str | None) -> int:
    raw = (value or "").strip()
    return int(raw) if raw else 0


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _priority_score(*, base: float, abs_delta_is: float, n_on_oos: int) -> float:
    return round(base + (abs_delta_is * 1000.0) + (min(n_on_oos, 50) * 10.0), 3)


def _role_consistent(role: str, delta_is: float, delta_oos: float) -> bool:
    if role == "TAKE":
        return delta_is > 0.0 and delta_oos > 0.0
    if role == "AVOID":
        return delta_is < 0.0 and delta_oos < 0.0
    return False


def _evidence_score(*, bh_p: float, delta_is: float, delta_oos: float, n_on_oos: int) -> float:
    score = abs(delta_is) * 700.0 + abs(delta_oos) * 400.0 + min(n_on_oos, 50) * 12.0
    if bh_p != float("-inf"):
        if bh_p <= 0.01:
            score += 140.0
        elif bh_p <= 0.05:
            score += 90.0
        elif bh_p <= 0.10:
            score += 45.0
        elif bh_p <= 0.25:
            score += 10.0
        elif bh_p >= 0.50:
            score -= 40.0
    if n_on_oos < 5:
        score -= 60.0
    return round(score, 3)


def _status_rank(status: str) -> int:
    return {
        "queued": 0,
        "deferred": 1,
        "advanced": 2,
        "parked": 3,
        "killed": 4,
    }.get(status, 9)


def _candidate_sort_key(row: dict) -> tuple[int, float, int]:
    return (
        _status_rank(str(row["status"])),
        -float(row["priority_score"]),
        -int(row["n_on_oos"]),
    )


def _build_review_batch(candidates: list[dict], per_kind: int = 2, max_total: int = 6) -> list[dict]:
    """Return a diversified shortlist across family / transfer / cell lanes.

    This prevents the worker from exhausting one candidate kind just because the
    frontier uses broad class weights.
    """
    review_batch: list[dict] = []
    used_ids: set[str] = set()
    used_lanes: set[str] = set()

    for kind in ("family", "transfer", "cell"):
        picked = 0
        for row in candidates:
            if row["candidate_kind"] != kind or row["status"] != "queued":
                continue
            if row["candidate_id"] in used_ids or row["lane"] in used_lanes:
                continue
            if int(row["n_on_oos"]) < 5:
                continue
            review_batch.append(row)
            used_ids.add(row["candidate_id"])
            used_lanes.add(row["lane"])
            picked += 1
            if picked >= per_kind or len(review_batch) >= max_total:
                break
        if len(review_batch) >= max_total:
            break

    if len(review_batch) < max_total:
        for row in candidates:
            if row["status"] != "queued" or row["candidate_id"] in used_ids:
                continue
            if row["lane"] in used_lanes and len(review_batch) < max_total - 1:
                continue
            review_batch.append(row)
            used_ids.add(row["candidate_id"])
            used_lanes.add(row["lane"])
            if len(review_batch) >= max_total:
                break

    return review_batch


def _load_doc_backed_exclusions(root: Path) -> tuple[set[tuple[str, str, str]], set[str]]:
    """Return lanes/candidates already solved or closed by durable repo docs."""
    solved_lanes: set[tuple[str, str, str]] = set()
    solved_candidates: set[str] = set()

    register_text = _read_text(root / "docs" / "plans" / "2026-04-22-mnq-usdata1000-geometry-family-register.md")
    if "LOCALLY SOLVED ENOUGH" in register_text:
        solved_lanes.add(("US_DATA_1000", "1.0", "long"))

    if "CLOSED as non-promotable exact-cell" in register_text:
        solved_candidates.update(
            {
                "cell::US_DATA_1000::1.0::long::F3_NEAR_PIVOT_50::AVOID",
                "cell::US_DATA_1000::1.0::long::F5_BELOW_PDL::TAKE",
            }
        )

    comex_text = _read_text(root / "docs" / "audit" / "results" / "2026-04-22-mnq-comex-pd-clear-long-take-v1.md")
    if "and promotes." in comex_text:
        solved_candidates.add("transfer::COMEX_SETTLE::1.0::long::PD_CLEAR_LONG")

    return solved_lanes, solved_candidates


def build_frontier(root: Path, ledger_path: Path) -> dict:
    results_root = root / "docs" / "audit" / "results"
    layered_rows = _read_csv(results_root / "2026-04-22-mnq-layered-candidate-board-v1.csv")
    family_rows = _read_csv(results_root / "2026-04-22-mnq-prior-day-family-board-v1.csv")
    transfer_rows = _read_csv(results_root / "2026-04-22-mnq-geometry-transfer-board-v1.csv")
    solved_lanes, solved_candidates = _load_doc_backed_exclusions(root)

    ledger: dict[str, dict] = {}
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ledger = {}

    candidates: list[dict] = []

    for row in family_rows:
        if not _as_bool(row["same_sign_oos"]):
            continue
        lane_key = (row["orb_label"], row["rr_target"], row["direction"])
        if lane_key in solved_lanes:
            continue
        candidate_id = f"family::{row['orb_label']}::{row['rr_target']}::{row['direction']}::{row['family']}"
        if candidate_id in solved_candidates:
            continue
        entry = ledger.get(candidate_id, {})
        delta_is = _as_float(row["delta_is"])
        delta_oos = _as_float(row["delta_oos"])
        role = row["role"]
        if not _role_consistent(role, delta_is, delta_oos):
            continue
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": "family",
                "source": "prior_day_family",
                "lane": f"{row['orb_label']} RR{row['rr_target']} {row['direction']}",
                "thesis": row["family"],
                "role": role,
                "same_sign_oos": True,
                "delta_is": delta_is,
                "delta_oos": delta_oos,
                "n_on_oos": _as_int(row["n_on_oos"]),
                "priority_score": _priority_score(
                    base=3000.0,
                    abs_delta_is=_as_float(row["abs_delta_is"]),
                    n_on_oos=_as_int(row["n_on_oos"]),
                ),
                "evidence_score": _evidence_score(
                    bh_p=_as_float(row["bh_p"]),
                    delta_is=delta_is,
                    delta_oos=delta_oos,
                    n_on_oos=_as_int(row["n_on_oos"]),
                ),
                "status": entry.get("frontier_decision", "queued"),
                "last_note": entry.get("summary", ""),
            }
        )

    for row in transfer_rows:
        if _as_bool(row["is_solved_lane"]) or not _as_bool(row["same_sign_oos"]):
            continue
        candidate_id = f"transfer::{row['orb_label']}::{row['rr_target']}::{row['direction']}::{row['family']}"
        if candidate_id in solved_candidates:
            continue
        entry = ledger.get(candidate_id, {})
        delta_is = _as_float(row["delta_is"])
        delta_oos = _as_float(row["delta_oos"])
        role = "TAKE"
        if not _role_consistent(role, delta_is, delta_oos):
            continue
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": "transfer",
                "source": "geometry_transfer",
                "lane": f"{row['orb_label']} RR{row['rr_target']} {row['direction']}",
                "thesis": row["family"],
                "role": role,
                "same_sign_oos": True,
                "delta_is": delta_is,
                "delta_oos": delta_oos,
                "n_on_oos": _as_int(row["n_on_oos"]),
                "priority_score": _priority_score(
                    base=2000.0,
                    abs_delta_is=_as_float(row["abs_delta_is"]),
                    n_on_oos=_as_int(row["n_on_oos"]),
                ),
                "evidence_score": _evidence_score(
                    bh_p=_as_float(row["bh_p"]),
                    delta_is=delta_is,
                    delta_oos=delta_oos,
                    n_on_oos=_as_int(row["n_on_oos"]),
                ),
                "status": entry.get("frontier_decision", "queued"),
                "last_note": entry.get("summary", ""),
            }
        )

    for row in layered_rows:
        if not _as_bool(row["same_sign_oos"]):
            continue
        lane_key = (row["orb_label"], row["rr_target"], row["direction"])
        if lane_key in solved_lanes:
            continue
        candidate_id = (
            f"cell::{row['orb_label']}::{row['rr_target']}::{row['direction']}::{row['signal']}::{row['role']}"
        )
        if candidate_id in solved_candidates:
            continue
        entry = ledger.get(candidate_id, {})
        delta_is = _as_float(row["delta_is"])
        delta_oos = _as_float(row["delta_oos"])
        role = row["role"]
        if not _role_consistent(role, delta_is, delta_oos):
            continue
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": "cell",
                "source": "layered_candidate",
                "lane": f"{row['orb_label']} RR{row['rr_target']} {row['direction']}",
                "thesis": row["signal"],
                "role": role,
                "same_sign_oos": True,
                "delta_is": delta_is,
                "delta_oos": delta_oos,
                "n_on_oos": _as_int(row["n_on_oos"]),
                "priority_score": _priority_score(
                    base=1000.0,
                    abs_delta_is=_as_float(row["abs_delta_is"]),
                    n_on_oos=_as_int(row["n_on_oos"]),
                ),
                "evidence_score": _evidence_score(
                    bh_p=_as_float(row["bh_p"]),
                    delta_is=delta_is,
                    delta_oos=delta_oos,
                    n_on_oos=_as_int(row["n_on_oos"]),
                ),
                "status": entry.get("frontier_decision", "queued"),
                "last_note": entry.get("summary", ""),
            }
        )

    candidates.sort(
        key=lambda row: (
            _status_rank(str(row["status"])),
            -float(row["evidence_score"]),
            -float(row["priority_score"]),
            -int(row["n_on_oos"]),
        )
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(candidates),
        "queued_count": sum(1 for row in candidates if row["status"] == "queued"),
        "kind_counts": {
            kind: sum(1 for row in candidates if row["candidate_kind"] == kind)
            for kind in ("family", "transfer", "cell")
        },
        "review_batch": _build_review_batch(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Worktree root")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Optional frontier ledger path; defaults to .session/mnq_discovery_frontier_ledger.json",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    ledger_path = (
        args.ledger.resolve()
        if args.ledger is not None
        else root / ".session" / "mnq_discovery_frontier_ledger.json"
    )
    print(json.dumps(build_frontier(root=root, ledger_path=ledger_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
