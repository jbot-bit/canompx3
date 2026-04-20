#!/usr/bin/env python3
"""Summarize current live-book attribution for the allocated lanes.

This is a read-only operator/audit surface for the current allocator-managed
book. It intentionally separates:
- modeled priors (`lane_allocation.json`, `validated_setups`)
- realized completed trades (`paper_trades`)
- event-level live execution evidence (`live_signal_events`)

Usage:
    python scripts/tools/live_attribution_report.py
    python scripts/tools/live_attribution_report.py --days 14
    python scripts/tools/live_attribution_report.py --db-path /path/to/gold.db --journal-path /path/to/live_journal.db
    python scripts/tools/live_attribution_report.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH


def load_allocation(allocation_path: Path) -> dict[str, Any]:
    return json.loads(allocation_path.read_text(encoding="utf-8"))


def _safe_connect(path: Path) -> duckdb.DuckDBPyConnection | None:
    if not path.exists():
        return None
    con = duckdb.connect(str(path), read_only=True)
    configure_connection(con)
    return con


def _table_names(con: duckdb.DuckDBPyConnection | None) -> set[str]:
    if con is None:
        return set()
    return {
        row[0]
        for row in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
    }


def _table_columns(con: duckdb.DuckDBPyConnection | None, table_name: str) -> set[str]:
    if con is None:
        return set()
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()}
    except duckdb.Error:
        return set()


def _date_floor(days: int | None) -> datetime | None:
    if days is None:
        return None
    return datetime.now(UTC) - timedelta(days=days)


def _load_modeled_priors(
    con: duckdb.DuckDBPyConnection | None,
    strategy_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if con is None or "validated_setups" not in _table_names(con):
        return {}
    rows = con.execute(
        """
        SELECT strategy_id, expectancy_r, win_rate, sample_size, status
        FROM validated_setups
        WHERE strategy_id IN ({})
        """.format(",".join("?" * len(strategy_ids))),
        strategy_ids,
    ).fetchall()
    return {
        row[0]: {
            "validated_expectancy_r": row[1],
            "validated_win_rate": row[2],
            "validated_sample_size": row[3],
            "validated_status": row[4],
        }
        for row in rows
    }


def _load_completed_trade_stats(
    con: duckdb.DuckDBPyConnection | None,
    strategy_ids: list[str],
    *,
    since: datetime | None,
) -> dict[str, dict[str, Any]]:
    if con is None or "paper_trades" not in _table_names(con):
        return {}

    cols = _table_columns(con, "paper_trades")
    filters = [f"strategy_id IN ({','.join('?' * len(strategy_ids))})"]
    params: list[Any] = list(strategy_ids)
    if since is not None:
        filters.append("entry_time >= ?")
        params.append(since)

    exec_source_expr = "execution_source" if "execution_source" in cols else "'unknown'"
    pnl_dollar_expr = "pnl_dollar" if "pnl_dollar" in cols else "NULL"

    rows = con.execute(
        f"""
        SELECT
            strategy_id,
            COUNT(*) AS completed_trades,
            ROUND(SUM(COALESCE(pnl_r, 0)), 4) AS cum_pnl_r,
            ROUND(AVG(pnl_r), 4) AS avg_pnl_r,
            ROUND(AVG(slippage_ticks), 4) AS avg_slippage_ticks,
            ROUND(SUM(COALESCE({pnl_dollar_expr}, 0)), 2) AS cum_pnl_dollar,
            MAX(trading_day) AS last_trade_day,
            SUM(CASE WHEN {exec_source_expr} = 'live' THEN 1 ELSE 0 END) AS live_rows,
            SUM(CASE WHEN {exec_source_expr} = 'shadow' THEN 1 ELSE 0 END) AS shadow_rows,
            SUM(CASE WHEN {exec_source_expr} = 'backfill' THEN 1 ELSE 0 END) AS backfill_rows
        FROM paper_trades
        WHERE {" AND ".join(filters)}
        GROUP BY strategy_id
        """,
        params,
    ).fetchall()
    return {
        row[0]: {
            "completed_trades": row[1],
            "cum_pnl_r": row[2],
            "avg_pnl_r": row[3],
            "avg_slippage_ticks": row[4],
            "cum_pnl_dollar": row[5],
            "last_trade_day": str(row[6]) if row[6] is not None else None,
            "live_rows": row[7],
            "shadow_rows": row[8],
            "backfill_rows": row[9],
        }
        for row in rows
    }


def _load_event_stats(
    con: duckdb.DuckDBPyConnection | None,
    strategy_ids: list[str],
    *,
    since: datetime | None,
) -> dict[str, dict[str, Any]]:
    if con is None or "live_signal_events" not in _table_names(con):
        return {}

    filters = [f"strategy_id IN ({','.join('?' * len(strategy_ids))})"]
    params: list[Any] = list(strategy_ids)
    if since is not None:
        filters.append("created_at >= ?")
        params.append(since)

    rows = con.execute(
        f"""
        SELECT
            strategy_id,
            COUNT(*) AS total_events,
            SUM(CASE WHEN event_type IN ('ENTRY_SUBMITTED', 'ENTRY_SIGNALLED') THEN 1 ELSE 0 END) AS submitted_events,
            SUM(CASE WHEN event_type = 'ENTRY_FILLED' THEN 1 ELSE 0 END) AS filled_events,
            SUM(CASE
                    WHEN event_type LIKE 'ENTRY_BLOCKED%%'
                      OR event_type IN ('ORB_CAP_SKIP', 'MAX_RISK_SKIP', 'REGIME_PAUSED')
                    THEN 1 ELSE 0
                END) AS skipped_events,
            SUM(CASE
                    WHEN event_type IN ('ENTRY_REJECTED', 'ENTRY_REJECTED_DUPLICATE', 'ENTRY_CANCELLED', 'ENTRY_SUBMIT_FAILED')
                    THEN 1 ELSE 0
                END) AS rejected_events,
            MAX(created_at) AS last_event_at
        FROM live_signal_events
        WHERE {" AND ".join(filters)}
        GROUP BY strategy_id
        """,
        params,
    ).fetchall()
    return {
        row[0]: {
            "total_events": row[1],
            "submitted_events": row[2],
            "filled_events": row[3],
            "skipped_events": row[4],
            "rejected_events": row[5],
            "last_event_at": row[6].isoformat() if row[6] is not None else None,
        }
        for row in rows
    }


def build_report(
    *,
    allocation_path: Path,
    db_path: Path,
    journal_path: Path,
    days: int | None = None,
) -> dict[str, Any]:
    allocation = load_allocation(allocation_path)
    lanes = allocation["lanes"]
    strategy_ids = [lane["strategy_id"] for lane in lanes]
    since = _date_floor(days)

    gold_con = _safe_connect(db_path)
    journal_con = _safe_connect(journal_path)
    try:
        modeled = _load_modeled_priors(gold_con, strategy_ids)
        completed = _load_completed_trade_stats(gold_con, strategy_ids, since=since)
        events = _load_event_stats(journal_con, strategy_ids, since=since)
    finally:
        if gold_con is not None:
            gold_con.close()
        if journal_con is not None:
            journal_con.close()

    rows: list[dict[str, Any]] = []
    for lane in lanes:
        strategy_id = lane["strategy_id"]
        modeled_row = modeled.get(strategy_id, {})
        completed_row = completed.get(strategy_id, {})
        event_row = events.get(strategy_id, {})
        completed_trades = int(completed_row.get("completed_trades", 0) or 0)
        total_events = int(event_row.get("total_events", 0) or 0)
        if completed_trades > 0:
            mechanism_status = "COMPLETED_ROWS"
        elif total_events > 0:
            mechanism_status = "EVENTS_ONLY"
        else:
            mechanism_status = "NO_EVIDENCE"

        rows.append(
            {
                "strategy_id": strategy_id,
                "instrument": lane["instrument"],
                "orb_label": lane["orb_label"],
                "filter_type": lane["filter_type"],
                "rr_target": lane["rr_target"],
                "allocator_trailing_expr": lane["trailing_expr"],
                "allocator_annual_r": lane["annual_r"],
                "allocator_regime": lane["session_regime"],
                "allocator_status": lane["status"],
                "validated_expectancy_r": modeled_row.get("validated_expectancy_r"),
                "validated_win_rate": modeled_row.get("validated_win_rate"),
                "validated_sample_size": modeled_row.get("validated_sample_size"),
                "validated_status": modeled_row.get("validated_status"),
                "completed_trades": completed_trades,
                "cum_pnl_r": completed_row.get("cum_pnl_r", 0.0),
                "avg_pnl_r": completed_row.get("avg_pnl_r"),
                "avg_slippage_ticks": completed_row.get("avg_slippage_ticks"),
                "cum_pnl_dollar": completed_row.get("cum_pnl_dollar"),
                "last_trade_day": completed_row.get("last_trade_day"),
                "live_rows": int(completed_row.get("live_rows", 0) or 0),
                "shadow_rows": int(completed_row.get("shadow_rows", 0) or 0),
                "backfill_rows": int(completed_row.get("backfill_rows", 0) or 0),
                "total_events": total_events,
                "submitted_events": int(event_row.get("submitted_events", 0) or 0),
                "filled_events": int(event_row.get("filled_events", 0) or 0),
                "skipped_events": int(event_row.get("skipped_events", 0) or 0),
                "rejected_events": int(event_row.get("rejected_events", 0) or 0),
                "last_event_at": event_row.get("last_event_at"),
                "mechanism_status": mechanism_status,
            }
        )

    return {
        "report_generated_at": datetime.now(UTC).isoformat(),
        "allocation_path": str(allocation_path),
        "db_path": str(db_path),
        "journal_path": str(journal_path),
        "days": days,
        "profile_id": allocation.get("profile_id"),
        "rebalance_date": allocation.get("rebalance_date"),
        "modeled_prior_warning": (
            "Allocator and validated-setups fields are modeled priors only. "
            "They are comparators for live mechanism audit, not proof."
        ),
        "lanes": rows,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = []
    lines.append("LIVE ATTRIBUTION REPORT")
    lines.append(
        f"profile={report['profile_id']} rebalance_date={report['rebalance_date']} lookback_days={report['days'] or 'ALL'}"
    )
    lines.append(report["modeled_prior_warning"])
    lines.append("")
    lines.append(f"{'Session':<16} {'Status':<14} {'Events':>6} {'Trades':>6} {'AvgR':>8} {'AvgSlip':>8} {'Model':>8}")
    lines.append("-" * 78)
    for lane in report["lanes"]:
        avg_r = f"{lane['avg_pnl_r']:+.4f}" if lane["avg_pnl_r"] is not None else "   n/a  "
        avg_slip = f"{lane['avg_slippage_ticks']:+.2f}" if lane["avg_slippage_ticks"] is not None else "   n/a "
        model = (
            f"{lane['validated_expectancy_r']:+.4f}"
            if lane["validated_expectancy_r"] is not None
            else f"{lane['allocator_trailing_expr']:+.4f}"
        )
        lines.append(
            f"{lane['orb_label']:<16} {lane['mechanism_status']:<14} {lane['total_events']:>6d} "
            f"{lane['completed_trades']:>6d} {avg_r:>8} {avg_slip:>8} {model:>8}"
        )
    lines.append("")
    lines.append("Per-lane detail:")
    for lane in report["lanes"]:
        lines.append(
            f"- {lane['strategy_id']}: events={lane['total_events']} "
            f"(submitted={lane['submitted_events']}, filled={lane['filled_events']}, "
            f"skipped={lane['skipped_events']}, rejected={lane['rejected_events']}), "
            f"completed={lane['completed_trades']}, live_rows={lane['live_rows']}, "
            f"shadow_rows={lane['shadow_rows']}, last_trade={lane['last_trade_day']}, "
            f"last_event={lane['last_event_at']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report live attribution for the current allocated book")
    parser.add_argument(
        "--allocation-path",
        type=Path,
        default=PROJECT_ROOT / "docs/runtime/lane_allocation.json",
        help="Path to current lane_allocation.json",
    )
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH, help="Path to gold.db / paper_trades DB")
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=LIVE_JOURNAL_DB_PATH,
        help="Path to live_journal.db",
    )
    parser.add_argument("--days", type=int, help="Optional lookback window in calendar days")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON instead of text")
    args = parser.parse_args()

    report = build_report(
        allocation_path=args.allocation_path,
        db_path=args.db_path,
        journal_path=args.journal_path,
        days=args.days,
    )
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(render_report(report))


if __name__ == "__main__":
    main()
