"""MCP server for repo-local strategy validation, fitness, and lane-allocation lookups.

Exposes read-only tools via stdio (fastmcp):
  - get_strategy_readiness: one-call readiness verdict for a single strategy
  - get_lane_allocation_summary: profile-scoped read of docs/runtime/lane_allocation.json
  - get_recent_fitness: rolling regime fitness for a single strategy
  - list_promotable_candidates: validated FIT strategies that are NOT currently allocated

Strict-truth boundary: this server reads canonical surfaces only —
``trading_app/strategy_validator.py`` outputs (``validated_setups``),
``trading_app/strategy_fitness.py`` (``compute_fitness``), and the
``docs/runtime/lane_allocation.json`` rebalance file. It never re-derives
those surfaces.

Overlap with ``gold-db`` MCP is intentional: ``gold-db.get_strategy_fitness``
remains the answer for raw single/portfolio fitness queries; this server
joins fitness with validator + allocator state into one deployment-readiness
verdict so callers do not have to stitch three MCPs together.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _allocation_path() -> Path:
    return PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"


def _load_allocation_doc(allocation_path: Path | None = None) -> dict[str, Any] | None:
    path = allocation_path or _allocation_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _allocation_index(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in doc.get("lanes") or []:
        sid = entry.get("strategy_id")
        if isinstance(sid, str):
            out[sid] = {**entry, "_allocation_state": "active"}
    for entry in doc.get("paused") or []:
        sid = entry.get("strategy_id")
        if isinstance(sid, str) and sid not in out:
            out[sid] = {**entry, "_allocation_state": "paused"}
    return out


def _validated_row(strategy_id: str) -> dict[str, Any] | None:
    """Read the validated_setups row for a strategy, or None if not validated."""
    import duckdb

    from pipeline.paths import GOLD_DB_PATH
    from trading_app.validated_shelf import deployable_validated_relation

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        relation = deployable_validated_relation(con, alias="vs")
        row = con.execute(
            f"""SELECT strategy_id, instrument, orb_label, orb_minutes,
                       entry_model, rr_target, confirm_bars, filter_type,
                       COALESCE(stop_multiplier, 1.0) AS stop_multiplier,
                       sample_size, win_rate, expectancy_r, sharpe_ratio,
                       max_drawdown_r
                FROM {relation}
                WHERE strategy_id = ?
                LIMIT 1""",
            [strategy_id],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in con.description]
        return dict(zip(cols, row, strict=False))


def _list_validated_rows(instrument: str | None) -> list[dict[str, Any]]:
    import duckdb

    from pipeline.paths import GOLD_DB_PATH
    from trading_app.validated_shelf import deployable_validated_relation

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        relation = deployable_validated_relation(con, alias="vs")
        if instrument:
            rows = con.execute(
                f"""SELECT strategy_id, instrument, orb_label, orb_minutes,
                           entry_model, rr_target, confirm_bars, filter_type,
                           COALESCE(stop_multiplier, 1.0) AS stop_multiplier,
                           sample_size, win_rate, expectancy_r, sharpe_ratio,
                           max_drawdown_r
                    FROM {relation}
                    WHERE instrument = ?
                    ORDER BY strategy_id""",
                [instrument],
            ).fetchall()
        else:
            rows = con.execute(
                f"""SELECT strategy_id, instrument, orb_label, orb_minutes,
                           entry_model, rr_target, confirm_bars, filter_type,
                           COALESCE(stop_multiplier, 1.0) AS stop_multiplier,
                           sample_size, win_rate, expectancy_r, sharpe_ratio,
                           max_drawdown_r
                    FROM {relation}
                    ORDER BY strategy_id"""
            ).fetchall()
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, r, strict=False)) for r in rows]


def _compute_fitness_payload(
    strategy_id: str,
    rolling_months: int,
    as_of_date: date | None = None,
) -> dict[str, Any] | dict[str, str]:
    from trading_app.strategy_fitness import compute_fitness

    try:
        score = compute_fitness(
            strategy_id,
            as_of_date=as_of_date,
            rolling_months=rolling_months,
        )
    except (ValueError, KeyError) as exc:
        return {"error": str(exc)}
    payload = asdict(score)
    return payload


def _allocation_staleness(allocation_path: Path | None = None) -> dict[str, Any]:
    from trading_app.lane_allocator import check_allocation_staleness

    status, days_old = check_allocation_staleness(allocation_path=allocation_path)
    return {"status": status, "days_old": days_old}


def _validate_active_instruments() -> set[str]:
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

    return set(ACTIVE_ORB_INSTRUMENTS)


def _readiness_verdict(
    *,
    validated: dict[str, Any] | None,
    fitness: dict[str, Any] | None,
    allocation_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine validator/fitness/allocator state into a single readiness verdict."""
    if validated is None:
        return {
            "verdict": "NOT_VALIDATED",
            "reason": "Strategy is not in validated_setups (deployable shelf).",
        }

    fitness_status = (fitness or {}).get("fitness_status") if fitness and "error" not in fitness else None
    allocation_state = (allocation_entry or {}).get("_allocation_state")
    allocation_status = (allocation_entry or {}).get("status")
    chordia_verdict = (allocation_entry or {}).get("chordia_verdict")
    chordia_age = (allocation_entry or {}).get("chordia_audit_age_days")

    if allocation_state == "active":
        verdict = "DEPLOYED"
        reason = (
            f"Currently allocated (status={allocation_status}, chordia={chordia_verdict}, "
            f"chordia_age_days={chordia_age})."
        )
    elif allocation_state == "paused":
        verdict = "PAUSED"
        reason = (
            f"In allocation file but paused: {(allocation_entry or {}).get('reason', allocation_status or 'unknown')}."
        )
    elif fitness_status == "FIT":
        verdict = "PROMOTABLE"
        reason = "Validated and currently FIT but not in lane_allocation.json."
    elif fitness_status in ("WATCH", "STALE", "DECAY"):
        verdict = f"VALIDATED_BUT_{fitness_status}"
        reason = f"Validated, not allocated, current rolling fitness = {fitness_status}."
    elif fitness_status is None:
        verdict = "VALIDATED_FITNESS_UNAVAILABLE"
        reason = (fitness or {}).get("error", "Fitness could not be computed.")
    else:
        verdict = "VALIDATED_UNKNOWN"
        reason = f"Validated, fitness={fitness_status}, allocation_state={allocation_state}."

    return {"verdict": verdict, "reason": reason}


def _get_strategy_readiness(
    strategy_id: str,
    rolling_months: int = 18,
) -> dict[str, Any]:
    if not strategy_id or not isinstance(strategy_id, str):
        return {"error": "strategy_id is required."}

    validated = _validated_row(strategy_id)
    fitness = _compute_fitness_payload(strategy_id, rolling_months) if validated else None

    allocation_doc = _load_allocation_doc()
    allocation_index = _allocation_index(allocation_doc) if allocation_doc else {}
    allocation_entry = allocation_index.get(strategy_id)

    verdict = _readiness_verdict(
        validated=validated,
        fitness=fitness,
        allocation_entry=allocation_entry,
    )

    payload: dict[str, Any] = {
        "strategy_id": strategy_id,
        "validated": validated,
        "fitness": fitness,
        "allocation_entry": allocation_entry,
        "allocation_rebalance_date": (allocation_doc or {}).get("rebalance_date"),
        "allocation_profile_id": (allocation_doc or {}).get("profile_id"),
        "rolling_months": rolling_months,
        **verdict,
    }
    return payload


def _get_lane_allocation_summary(
    profile_name: str | None = None,
) -> dict[str, Any]:
    doc = _load_allocation_doc()
    if doc is None:
        return {
            "error": "lane_allocation.json missing or unreadable.",
            "path": str(_allocation_path()),
        }

    actual_profile = doc.get("profile_id")
    if profile_name and actual_profile and profile_name != actual_profile:
        return {
            "error": f"Allocation file profile_id={actual_profile!r} does not match requested {profile_name!r}.",
            "path": str(_allocation_path()),
            "actual_profile_id": actual_profile,
        }

    lanes = doc.get("lanes") or []
    paused = doc.get("paused") or []
    staleness = _allocation_staleness()

    return {
        "rebalance_date": doc.get("rebalance_date"),
        "profile_id": actual_profile,
        "trailing_window_months": doc.get("trailing_window_months"),
        "active_count": len(lanes),
        "paused_count": len(paused),
        "all_scores_count": doc.get("all_scores_count"),
        "lanes": lanes,
        "paused": paused,
        "staleness": staleness,
        "path": str(_allocation_path()),
    }


def _get_recent_fitness(
    strategy_id: str,
    rolling_months: int = 18,
) -> dict[str, Any]:
    if not strategy_id or not isinstance(strategy_id, str):
        return {"error": "strategy_id is required."}
    return {
        "strategy_id": strategy_id,
        "rolling_months": rolling_months,
        "fitness": _compute_fitness_payload(strategy_id, rolling_months),
    }


def _list_promotable_candidates(
    instrument: str | None = None,
    rolling_months: int = 18,
    limit: int = 50,
) -> dict[str, Any]:
    active = _validate_active_instruments()
    if instrument and instrument not in active:
        return {
            "error": f"Instrument {instrument!r} is not active for ORB.",
            "active_instruments": sorted(active),
        }

    validated_rows = _list_validated_rows(instrument)
    allocation_doc = _load_allocation_doc()
    allocation_index = _allocation_index(allocation_doc) if allocation_doc else {}
    allocated_ids = {sid for sid, entry in allocation_index.items() if entry.get("_allocation_state") == "active"}

    candidates: list[dict[str, Any]] = []
    for row in validated_rows:
        sid = row["strategy_id"]
        if sid in allocated_ids:
            continue
        fitness = _compute_fitness_payload(sid, rolling_months)
        if isinstance(fitness, dict) and fitness.get("fitness_status") == "FIT":
            candidates.append(
                {
                    "strategy_id": sid,
                    "instrument": row["instrument"],
                    "orb_label": row["orb_label"],
                    "validated_expr": row.get("expectancy_r"),
                    "validated_n": row.get("sample_size"),
                    "rolling_expr": fitness.get("rolling_exp_r"),
                    "rolling_n": fitness.get("rolling_sample"),
                    "rolling_window_months": fitness.get("rolling_window_months"),
                    "fitness_status": fitness.get("fitness_status"),
                }
            )

    def _sort_key(cand: dict[str, Any]) -> float:
        value = cand.get("rolling_expr")
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    candidates.sort(key=_sort_key, reverse=True)

    truncated = candidates[: max(limit, 1)]
    return {
        "instrument": instrument,
        "rolling_months": rolling_months,
        "candidate_count": len(truncated),
        "total_promotable": len(candidates),
        "validated_count": len(validated_rows),
        "currently_allocated_count": len(allocated_ids),
        "candidates": truncated,
        "allocation_rebalance_date": (allocation_doc or {}).get("rebalance_date"),
        "allocation_profile_id": (allocation_doc or {}).get("profile_id"),
    }


def _build_server():
    from fastmcp import FastMCP

    mcp = FastMCP(
        "strategy-lab",
        instructions=(
            "Repo-local read-only strategy-validation, fitness, and lane-allocation MCP for canompx3. "
            "Use it to ask: is this strategy validated, currently fit, currently deployed, or promotable. "
            "Reads validated_setups, compute_fitness, and docs/runtime/lane_allocation.json — does not "
            "re-derive any of them. Overlaps gold-db.get_strategy_fitness intentionally so deployment-"
            "readiness questions can be answered from one server."
        ),
    )

    @mcp.tool()
    def get_strategy_readiness(
        strategy_id: str,
        rolling_months: int = 18,
    ) -> dict[str, Any]:
        """One-call readiness verdict joining validator + fitness + lane allocation.

        Args:
            strategy_id: Full strategy id (e.g. ``MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12``).
            rolling_months: Rolling fitness window in months (default 18).

        Returns:
            ``{verdict, reason, validated, fitness, allocation_entry, ...}`` where
            ``verdict`` is one of NOT_VALIDATED / DEPLOYED / PAUSED / PROMOTABLE /
            VALIDATED_BUT_<STATUS> / VALIDATED_FITNESS_UNAVAILABLE / VALIDATED_UNKNOWN.
        """

        return _get_strategy_readiness(strategy_id=strategy_id, rolling_months=rolling_months)

    @mcp.tool()
    def get_lane_allocation_summary(profile_name: str | None = None) -> dict[str, Any]:
        """Profile-scoped read of the latest lane_allocation.json rebalance.

        Args:
            profile_name: Optional profile id. If supplied and the JSON file is for a
                different profile, the call returns an error rather than mismatched
                lanes.

        Returns:
            Active + paused lanes, staleness verdict, and the underlying file path.
        """

        return _get_lane_allocation_summary(profile_name=profile_name)

    @mcp.tool()
    def get_recent_fitness(
        strategy_id: str,
        rolling_months: int = 18,
    ) -> dict[str, Any]:
        """Rolling regime fitness for a single strategy (thin wrapper over compute_fitness).

        Args:
            strategy_id: Full strategy id.
            rolling_months: Rolling window in months (default 18).
        """

        return _get_recent_fitness(strategy_id=strategy_id, rolling_months=rolling_months)

    @mcp.tool()
    def list_promotable_candidates(
        instrument: str | None = None,
        rolling_months: int = 18,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List validated strategies that are currently FIT but not in the active allocation.

        Args:
            instrument: Optional ACTIVE ORB instrument filter (e.g. ``MNQ``).
            rolling_months: Rolling window in months (default 18).
            limit: Maximum candidates to return (sorted by rolling ExpR desc).
        """

        return _list_promotable_candidates(
            instrument=instrument,
            rolling_months=rolling_months,
            limit=limit,
        )

    return mcp


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
