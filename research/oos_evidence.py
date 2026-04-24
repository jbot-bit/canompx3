"""Read-only OOS evidence interpreter for standalone candidates.

Classifies validator Criterion-8 OOS outcomes as one of:
    - ``REFUTED_WITH_POWER`` — negative / failed-ratio outcome with OOS
      power >= 0.80 to detect the IS effect.
    - ``UNVERIFIED_SPARSE_OOS`` — negative-looking outcome but OOS power
      below 0.80; cannot refute.
    - ``CONFIRMED`` — positive outcome that passes Criterion-8.
    - ``INSUFFICIENT_DATA`` — OOS sample too small to evaluate.

Power floor enforces `.claude/rules/backtesting-methodology.md` RULE 3.3.

Literature grounding (canonical extracts, not training memory):
    - `docs/institutional/literature/harvey_liu_2015_backtesting.md` —
      OOS haircut framing, power-before-veto principle.
    - `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` —
      finite-data IS/OOS handling, misspecification of binary splits under
      short OOS windows.
    - `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` —
      sample-size-adjusted promotion gate (Criterion 5 cross-check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from research.oos_power import one_sample_n_for_power, one_sample_power, power_verdict
from trading_app.strategy_validator import _evaluate_criterion_8_oos

_C8_STATUS_COMPAT = {
    "passed": "PASSED",
    "pass_through_no_data": "NO_OOS_DATA",
    "pass_through_insufficient_n": "INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH",
}


def _normalize_c8_status(status: str | None) -> str | None:
    if status is None:
        return None
    return _C8_STATUS_COMPAT.get(status, status)


def _load_rows(
    db_path: Path,
    *,
    strategy_id: str | None,
    hypothesis_sha: str | None,
    table: str,
) -> list[dict]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        if hypothesis_sha is not None:
            rows = con.execute(
                "SELECT * FROM experimental_strategies WHERE hypothesis_file_sha = ? ORDER BY strategy_id",
                [hypothesis_sha],
            ).fetchall()
            cols = [desc[0] for desc in con.description]
            return [dict(zip(cols, row, strict=False)) for row in rows]

        if strategy_id is None:
            raise ValueError("strategy_id or hypothesis_sha is required")

        tables = ["experimental_strategies", "validated_setups"] if table == "auto" else [table]
        for source_table in tables:
            rows = con.execute(
                f"SELECT * FROM {source_table} WHERE strategy_id = ?",
                [strategy_id],
            ).fetchall()
            if rows:
                cols = [desc[0] for desc in con.description]
                items = []
                for row in rows:
                    item = dict(zip(cols, row, strict=False))
                    item["_source_table"] = source_table
                    items.append(item)
                return items
    return []


def _normalized_validator_status(row: dict) -> str | None:
    if row.get("_source_table") == "validated_setups":
        return "PASSED"
    return row.get("validation_status")


def _normalized_validator_reason(row: dict) -> str | None:
    if row.get("_source_table") == "validated_setups":
        return row.get("retirement_reason")
    return row.get("rejection_reason") or row.get("validation_notes")


def _estimate_is_std(row: dict) -> float | None:
    expectancy_r = row.get("expectancy_r")
    sharpe_ratio = row.get("sharpe_ratio")
    if expectancy_r is None or sharpe_ratio in (None, 0):
        return None
    try:
        mean_r = float(expectancy_r)
        sr = float(sharpe_ratio)
    except (TypeError, ValueError):
        return None
    if sr == 0:
        return None
    std = abs(mean_r) / abs(sr)
    return std if std > 0 else None


def _derive_interpretation(
    validator_status: str | None,
    validator_reason: str | None,
    c8_status: str | None,
    power_tier: str | None,
) -> str:
    if validator_status == "PASSED":
        return "VALIDATED"
    if c8_status == "NO_OOS_DATA":
        return "INCONCLUSIVE_NO_OOS"
    if validator_status == "REJECTED" and validator_reason and not validator_reason.startswith("criterion_8:"):
        return "REJECTED_NON_OOS"
    if c8_status in {
        "INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH",
        "INSUFFICIENT_N_PATHWAY_B_REJECT",
        "NEGATIVE_OOS_EXPR",
        "FAILED_RATIO",
    }:
        if c8_status in {"NEGATIVE_OOS_EXPR", "FAILED_RATIO"} and power_tier == "CAN_REFUTE":
            return "REFUTED_WITH_POWER"
        return "UNVERIFIED_SPARSE_OOS"
    return "REJECTED_NON_OOS"


def assess_row(row: dict, db_path: Path | None = None) -> dict[str, object]:
    effective_db = db_path or GOLD_DB_PATH
    pathway = row.get("validation_pathway")
    strict = pathway == "individual"
    c8_eval = _evaluate_criterion_8_oos(row, effective_db, strict_oos_n=strict)
    c8_status = _normalize_c8_status(row.get("c8_oos_status")) or c8_eval["c8_oos_status"]
    n_oos = c8_eval["n_oos"]
    is_std = _estimate_is_std(row)

    power = None
    power_tier = None
    n_for_80pct = None
    if is_std is not None and n_oos is not None:
        expectancy_r = float(row.get("expectancy_r") or 0.0)
        effect_size = abs(expectancy_r) / is_std if is_std > 0 else 0.0
        power = one_sample_power(effect_size, int(n_oos)) if n_oos >= 2 else 0.0
        power_tier = power_verdict(power)
        n_for_80pct = one_sample_n_for_power(effect_size) if effect_size > 0 else None

    validator_status = _normalized_validator_status(row)
    validator_reason = _normalized_validator_reason(row)
    interpretation = _derive_interpretation(
        validator_status,
        validator_reason,
        c8_status,
        power_tier,
    )

    return {
        "strategy_id": row.get("strategy_id"),
        "source_table": row.get("_source_table", "experimental_strategies"),
        "validator_status": validator_status,
        "validator_reason": validator_reason,
        "validation_pathway": pathway,
        "c8_oos_status": c8_status,
        "n_is": row.get("sample_size"),
        "expr_is": row.get("expectancy_r"),
        "n_oos": n_oos,
        "expr_oos": c8_eval["oos_expectancy_r"],
        "oos_is_ratio": c8_eval["oos_is_ratio"],
        "oos_power": power,
        "oos_power_tier": power_tier,
        "oos_n_for_80pct_power": n_for_80pct,
        "interpretation": interpretation,
    }


def assess_targets(
    *,
    db_path: Path | None = None,
    strategy_id: str | None = None,
    hypothesis_sha: str | None = None,
    table: str = "auto",
) -> list[dict[str, object]]:
    effective_db = db_path or GOLD_DB_PATH
    rows = _load_rows(
        effective_db,
        strategy_id=strategy_id,
        hypothesis_sha=hypothesis_sha,
        table=table,
    )
    return [assess_row(row, effective_db) for row in rows]


def _format_text(report: dict[str, object]) -> str:
    lines = [
        f"strategy_id: {report['strategy_id']}",
        f"source_table: {report['source_table']}",
        f"validator_status: {report['validator_status']}",
        f"validator_reason: {report['validator_reason']}",
        f"validation_pathway: {report['validation_pathway']}",
        f"c8_oos_status: {report['c8_oos_status']}",
        f"is: N={report['n_is']} ExpR={report['expr_is']}",
        f"oos: N={report['n_oos']} ExpR={report['expr_oos']} ratio={report['oos_is_ratio']}",
        f"oos_power: {report['oos_power']}",
        f"oos_power_tier: {report['oos_power_tier']}",
        f"oos_n_for_80pct_power: {report['oos_n_for_80pct_power']}",
        f"interpretation: {report['interpretation']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only OOS evidence interpreter")
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--strategy-id")
    parser.add_argument("--hypothesis-sha")
    parser.add_argument("--table", choices=["auto", "experimental_strategies", "validated_setups"], default="auto")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    reports = assess_targets(
        db_path=args.db,
        strategy_id=args.strategy_id,
        hypothesis_sha=args.hypothesis_sha,
        table=args.table,
    )
    if args.as_json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for idx, report in enumerate(reports):
            if idx:
                print()
            print(_format_text(report))
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
