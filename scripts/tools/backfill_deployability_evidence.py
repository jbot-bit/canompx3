#!/usr/bin/env python3
"""Backfill deployability evidence stamps with explicit provenance.

The default mode is a dry run. This tool is intentionally conservative: it can
persist canonical Criterion 8 statuses, but slippage evidence for MES/MGC/MNQ is
marked as unresolved event-tail evidence unless a row already has a passing
status or a later, instrument/session-specific validator upgrades it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.deployability import SLIPPAGE_PASS_STATUSES  # noqa: E402
from trading_app.strategy_validator import _evaluate_criterion_8_oos  # noqa: E402

Evidence = Literal["all", "c8_oos", "slippage"]

ROUTINE_SLIPPAGE_EVIDENCE: dict[str, dict[str, str]] = {
    "MES": {
        "status": "PENDING_EVENT_TAIL",
        "provenance": "docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md",
        "reason": "routine slippage measured; event-tail debt remains scoped and unresolved",
    },
    "MGC": {
        "status": "PENDING_EVENT_TAIL",
        "provenance": "research/output/mgc_e2_slippage_analysis.json",
        "reason": "routine central tendency measured, but the 2018-01-18 event-tail outlier prevents blanket pass",
    },
    "MNQ": {
        "status": "PENDING_EVENT_TAIL",
        "provenance": "docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v2-gap-fill.md",
        "reason": "routine slippage measured; MNQ event-tail debt remains unresolved",
    },
}


@dataclass(frozen=True)
class EvidenceUpdate:
    strategy_id: str
    instrument: str
    field: str
    old_value: Any
    new_value: Any
    reason: str
    provenance: str | None = None


def _instrument_filter(instruments: set[str] | None) -> tuple[str, list[str]]:
    if not instruments:
        default_instruments = sorted(ACTIVE_ORB_INSTRUMENTS)
        placeholders = ", ".join("?" for _ in default_instruments)
        return f"AND vs.instrument IN ({placeholders})", default_instruments
    placeholders = ", ".join("?" for _ in instruments)
    return f"AND vs.instrument IN ({placeholders})", sorted(instruments)


def _load_rows(
    db_path: Path,
    instruments: set[str] | None,
    strategy_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    instrument_sql, params = _instrument_filter(instruments)
    strategy_sql = ""
    if strategy_ids:
        placeholders = ", ".join("?" for _ in strategy_ids)
        strategy_sql = f"AND vs.strategy_id IN ({placeholders})"
        params.extend(sorted(strategy_ids))
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute(
            f"""
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
                   vs.entry_model, vs.rr_target, vs.confirm_bars, vs.filter_type,
                   LOWER(vs.status) AS status,
                   vs.sample_size, vs.expectancy_r, vs.oos_exp_r,
                   vs.validation_pathway, vs.slippage_validation_status,
                   vs.c8_oos_status
            FROM validated_setups vs
            WHERE LOWER(vs.status) = 'active'
              {instrument_sql}
              {strategy_sql}
            ORDER BY vs.instrument, vs.orb_label, vs.strategy_id
            """,
            params,
        ).fetchall()
        cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, row, strict=False)) for row in rows]


def _is_empty(value: Any) -> bool:
    return value in (None, "")


def _c8_update(row: dict[str, Any], db_path: Path, *, overwrite: bool) -> EvidenceUpdate | None:
    old = row.get("c8_oos_status")
    if not overwrite and not _is_empty(old):
        return None
    result = _evaluate_criterion_8_oos(
        row,
        db_path,
        strict_oos_n=str(row.get("validation_pathway")) == "individual",
    )
    new = result.get("c8_oos_status")
    if _is_empty(new) or new == old:
        return None
    return EvidenceUpdate(
        strategy_id=str(row["strategy_id"]),
        instrument=str(row["instrument"]),
        field="c8_oos_status",
        old_value=old,
        new_value=new,
        reason=str(result.get("reason") or "canonical Criterion 8 evaluator"),
        provenance="trading_app.strategy_validator._evaluate_criterion_8_oos",
    )


def _slippage_update(row: dict[str, Any], *, overwrite: bool) -> EvidenceUpdate | None:
    old = row.get("slippage_validation_status")
    if not overwrite and not _is_empty(old):
        return None
    if str(old).strip().upper() in SLIPPAGE_PASS_STATUSES:
        return None
    evidence = ROUTINE_SLIPPAGE_EVIDENCE.get(str(row.get("instrument")))
    if evidence is None:
        new = "PENDING_MISSING_PILOT"
        reason = "no instrument-level deployability slippage evidence registered"
        provenance = None
    else:
        new = evidence["status"]
        reason = evidence["reason"]
        provenance = evidence["provenance"]
    if new == old:
        return None
    return EvidenceUpdate(
        strategy_id=str(row["strategy_id"]),
        instrument=str(row["instrument"]),
        field="slippage_validation_status",
        old_value=old,
        new_value=new,
        reason=reason,
        provenance=provenance,
    )


def build_backfill_plan(
    *,
    db_path: Path = GOLD_DB_PATH,
    instruments: set[str] | None = None,
    strategy_ids: set[str] | None = None,
    evidence: Evidence = "all",
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = _load_rows(db_path, instruments, strategy_ids)
    updates: list[EvidenceUpdate] = []
    for row in rows:
        if evidence in ("all", "c8_oos"):
            c8 = _c8_update(row, db_path, overwrite=overwrite)
            if c8 is not None:
                updates.append(c8)
        if evidence in ("all", "slippage"):
            slip = _slippage_update(row, overwrite=overwrite)
            if slip is not None:
                updates.append(slip)

    by_field: dict[str, int] = {}
    by_instrument: dict[str, int] = {}
    for update in updates:
        by_field[update.field] = by_field.get(update.field, 0) + 1
        by_instrument[update.instrument] = by_instrument.get(update.instrument, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_path": str(db_path),
        "instruments": sorted(instruments) if instruments else sorted(ACTIVE_ORB_INSTRUMENTS),
        "strategy_ids": sorted(strategy_ids) if strategy_ids else None,
        "evidence": evidence,
        "overwrite": overwrite,
        "dry_run": True,
        "summary": {
            "rows_scanned": len(rows),
            "updates": len(updates),
            "by_field": by_field,
            "by_instrument": by_instrument,
        },
        "updates": [asdict(update) for update in updates],
    }


def apply_backfill_plan(db_path: Path, plan: dict[str, Any]) -> int:
    updates = plan.get("updates", [])
    if not updates:
        return 0
    with duckdb.connect(str(db_path)) as con:
        for update in updates:
            con.execute(
                f"UPDATE validated_setups SET {update['field']} = ? WHERE strategy_id = ?",
                [update["new_value"], update["strategy_id"]],
            )
    return len(updates)


def render_text(plan: dict[str, Any], *, max_rows: int = 40) -> str:
    summary = plan["summary"]
    lines = [
        f"Deployability Evidence Backfill | dry_run={plan['dry_run']} | evidence={plan['evidence']}",
        f"DB: {plan['db_path']}",
        f"Instruments: {plan['instruments']}",
        (
            "Summary: "
            f"rows_scanned={summary['rows_scanned']} "
            f"updates={summary['updates']} "
            f"by_field={summary['by_field']} "
            f"by_instrument={summary['by_instrument']}"
        ),
        "Updates:",
    ]
    for update in plan.get("updates", [])[:max_rows]:
        lines.append(
            "  - "
            f"{update['strategy_id']} {update['field']}: "
            f"{update['old_value']} -> {update['new_value']} | {update['reason']}"
        )
    if len(plan.get("updates", [])) > max_rows:
        lines.append(f"  ... {len(plan['updates']) - max_rows} more")
    return "\n".join(lines)


def _parse_instruments(values: list[str] | None) -> set[str] | None:
    if not values or "ALL" in values:
        return None
    return set(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill deployability evidence statuses.")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH), help="DuckDB path. Defaults to canonical gold.db.")
    parser.add_argument("--instrument", action="append", choices=("ALL", "MES", "MGC", "MNQ"), default=None)
    parser.add_argument(
        "--strategy-id",
        action="append",
        default=None,
        help="Restrict updates to a specific strategy id. Repeat for multiple.",
    )
    parser.add_argument("--evidence", choices=("all", "c8_oos", "slippage"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Recompute non-empty fields too.")
    parser.add_argument("--write", action="store_true", help="Persist updates. Without this, dry-run only.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--max-rows", type=int, default=40)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    db_path = Path(args.db_path)
    plan = build_backfill_plan(
        db_path=db_path,
        instruments=_parse_instruments(args.instrument),
        strategy_ids=set(args.strategy_id) if args.strategy_id else None,
        evidence=args.evidence,
        overwrite=args.overwrite,
    )
    if args.write:
        applied = apply_backfill_plan(db_path, plan)
        plan["dry_run"] = False
        plan["summary"]["applied"] = applied
    rendered = (
        json.dumps(plan, indent=2, sort_keys=True, default=str)
        if args.format == "json"
        else render_text(plan, max_rows=args.max_rows)
    )
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
