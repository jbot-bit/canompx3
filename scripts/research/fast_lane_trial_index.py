"""Derived Fast Lane trial index.

The append-only ledger remains the universe-of-trials accounting layer. This
module exposes the V2 counting view over that ledger after correction records
are applied, so scanners/bridges can talk about an index without inventing a
second source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.research.fast_lane_trial_ledger import (
    filter_v2_k_count_rows,
    read_ledger,
    read_trial_corrections,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "docs" / "runtime" / "fast_lane_trial_ledger.yaml"
CORRECTIONS_PATH = REPO_ROOT / "docs" / "runtime" / "fast_lane_trial_corrections.yaml"


def _lane_key(row: dict[str, Any]) -> str:
    lineage = row.get("k_lineage") or {}
    if not isinstance(lineage, dict):
        lineage = {}
    instrument = lineage.get("instrument") or "UNKNOWN"
    orb_label = lineage.get("orb_label") or "UNKNOWN"
    orb_minutes = lineage.get("orb_minutes") or "UNKNOWN"
    return f"{instrument}|{orb_label}|{orb_minutes}"


def _trial_id(row: dict[str, Any]) -> str:
    value = row.get("trial_id")
    if isinstance(value, str) and value:
        return value
    run_id = row.get("run_id")
    return str(run_id) if run_id is not None else ""


def build_trial_index(rows: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic V2 index from ledger rows plus corrections."""
    eligible = filter_v2_k_count_rows(rows, corrections)
    by_structural_hash: dict[str, dict[str, Any]] = {}
    by_lane: dict[str, dict[str, Any]] = {}

    for row in sorted(eligible, key=lambda r: (_trial_id(r), str(r.get("run_id") or ""))):
        tid = _trial_id(row)
        structural_hash = row.get("structural_hash")
        if not isinstance(structural_hash, str) or not structural_hash:
            structural_hash = "UNKNOWN"

        structural_entry = by_structural_hash.setdefault(
            structural_hash,
            {"K_structural": 0, "trial_ids": []},
        )
        structural_entry["K_structural"] += 1
        if tid:
            structural_entry["trial_ids"].append(tid)

        lane = _lane_key(row)
        lane_entry = by_lane.setdefault(lane, {"K_lane": 0, "trial_ids": []})
        lane_entry["K_lane"] += 1
        if tid:
            lane_entry["trial_ids"].append(tid)

    return {
        "schema_version": 1,
        "source": "scripts/research/fast_lane_trial_index.py",
        "total_v2_trials": len(eligible),
        "by_structural_hash": {k: by_structural_hash[k] for k in sorted(by_structural_hash)},
        "by_lane": {k: by_lane[k] for k in sorted(by_lane)},
    }


def build_trial_index_from_paths(
    *,
    ledger_path: Path = LEDGER_PATH,
    corrections_path: Path = CORRECTIONS_PATH,
) -> dict[str, Any]:
    """Load canonical ledger/corrections and return the derived V2 index."""
    ledger = read_ledger(ledger_path)
    rows = ledger.get("entries") or []
    if not isinstance(rows, list):
        raise ValueError(f"fast_lane_trial_index: {ledger_path} `entries` must be a list")
    corrections = read_trial_corrections(corrections_path)
    return build_trial_index(rows, corrections)
