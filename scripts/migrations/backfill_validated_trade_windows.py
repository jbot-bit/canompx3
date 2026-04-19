"""Refresh stale trade-window provenance on validated_setups.

Canonical refresh for Check 45 (`check_active_native_trade_windows_match_provenance`).
Re-runs `StrategyTradeWindowResolver.resolve()` against each VALIDATOR_NATIVE active row
and updates stored `first_trade_day / last_trade_day / trade_day_count` if they've
drifted from the canonical recompute. Idempotent: rows matching canonical are skipped.

Does not touch `status`, `promotion_provenance`, `promotion_git_sha`, or any performance
column. Those remain authoritative from the original validation run.

Usage:
    python scripts/migrations/backfill_validated_trade_windows.py --dry-run
    python scripts/migrations/backfill_validated_trade_windows.py
    python scripts/migrations/backfill_validated_trade_windows.py --strategy-id <SID>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.validation_provenance import StrategyTradeWindowResolver

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


@dataclass
class RefreshReport:
    inspected: int
    drifted: int
    updated: int
    details: list[str]


def refresh_validated_trade_windows(
    db_path: Path = GOLD_DB_PATH,
    *,
    strategy_id: str | None = None,
    dry_run: bool = False,
) -> RefreshReport:
    con = duckdb.connect(str(db_path), read_only=dry_run)
    try:
        where = [
            "status = 'active'",
            "promotion_provenance = 'VALIDATOR_NATIVE'",
        ]
        params: list = []
        if strategy_id is not None:
            where.append("strategy_id = ?")
            params.append(strategy_id)

        rows = con.execute(
            f"""
            SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   rr_target, confirm_bars, filter_type,
                   first_trade_day, last_trade_day, trade_day_count
            FROM validated_setups
            WHERE {" AND ".join(where)}
            ORDER BY strategy_id
            """,
            params,
        ).fetchall()

        resolver = StrategyTradeWindowResolver(con)
        drifted: list[tuple] = []
        details: list[str] = []
        for (
            sid,
            instrument,
            orb_label,
            orb_minutes,
            entry_model,
            rr_target,
            confirm_bars,
            filter_type,
            first_day,
            last_day,
            trade_day_count,
        ) in rows:
            canonical = resolver.resolve(
                instrument=instrument,
                orb_label=orb_label,
                orb_minutes=orb_minutes,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                filter_type=filter_type,
            )
            if (
                canonical.first_trade_day == first_day
                and canonical.last_trade_day == last_day
                and canonical.trade_day_count == trade_day_count
            ):
                continue
            drifted.append(
                (
                    sid,
                    canonical.first_trade_day,
                    canonical.last_trade_day,
                    canonical.trade_day_count,
                )
            )
            details.append(
                f"  {sid}: ({first_day}, {last_day}, N={trade_day_count}) -> "
                f"({canonical.first_trade_day}, {canonical.last_trade_day}, N={canonical.trade_day_count})"
            )

        updated = 0
        if drifted and not dry_run:
            for sid, f, l, n in drifted:
                con.execute(
                    """
                    UPDATE validated_setups
                    SET first_trade_day = ?,
                        last_trade_day = ?,
                        trade_day_count = ?
                    WHERE strategy_id = ?
                    """,
                    [f, l, n, sid],
                )
                updated += 1
            con.commit()

        return RefreshReport(
            inspected=len(rows),
            drifted=len(drifted),
            updated=updated,
            details=details,
        )
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh stale validated_setups trade-window provenance (fixes Check 45).",
    )
    parser.add_argument(
        "--db-path",
        default=str(GOLD_DB_PATH),
        help="Path to DuckDB file (default: canonical GOLD_DB_PATH)",
    )
    parser.add_argument(
        "--strategy-id",
        default=None,
        help="Restrict to a single strategy_id (default: all active VALIDATOR_NATIVE rows)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report drift without writing",
    )
    args = parser.parse_args()

    report = refresh_validated_trade_windows(
        db_path=Path(args.db_path),
        strategy_id=args.strategy_id,
        dry_run=args.dry_run,
    )
    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] inspected={report.inspected} drifted={report.drifted} updated={report.updated}")
    if report.details:
        print("Drift detail:")
        for line in report.details:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
