#!/usr/bin/env python3
"""Reconcile journaled live orders against broker fills.

This is a read-only attribution gate. It does not fetch broker data; run
``scripts/tools/fetch_broker_fills.py`` first to refresh ``data/broker_fills.jsonl``.
Matching is by ``(account_id, order_id)`` so overlapping order IDs on separate
accounts remain independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FILLS_PATH = PROJECT_ROOT / "data" / "broker_fills.jsonl"


def _default_journal_path() -> Path:
    from pipeline.paths import LIVE_JOURNAL_DB_PATH

    return LIVE_JOURNAL_DB_PATH


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _issue(kind: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, **{k: v for k, v in fields.items() if v is not None}}


def _order_key(*, account_id: Any, order_id: Any) -> tuple[str, str] | None:
    account = _norm(account_id)
    order = _norm(order_id)
    if account is None or order is None:
        return None
    return account, order


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append({"_load_error": f"{path}:{line_no}: {exc}"})
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _select_exprs_for_existing_columns(con: duckdb.DuckDBPyConnection, desired: list[str]) -> str:
    existing = {r[1] for r in con.execute("PRAGMA table_info('live_trades')").fetchall()}
    parts = [col if col in existing else f"NULL AS {col}" for col in desired]
    return ", ".join(parts)


def load_journal_rows(path: Path) -> list[dict[str, Any]]:
    desired = [
        "trade_id",
        "trading_day",
        "strategy_id",
        "profile_id",
        "account_id",
        "copy_id",
        "runtime_session_id",
        "contract_id",
        "session_id",
        "orb_minutes",
        "order_id_entry",
        "order_id_exit",
        "client_order_id",
        "client_order_id_exit",
    ]
    con = duckdb.connect(str(path), read_only=True)
    try:
        has_table = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'live_trades'
            """
        ).fetchone()[0]
        if not has_table:
            return []
        select_list = _select_exprs_for_existing_columns(con, desired)
        rows = con.execute(f"SELECT {select_list} FROM live_trades").fetchall()
        return [dict(zip(desired, row, strict=False)) for row in rows]
    finally:
        con.close()


def reconcile_live_fills(journal_rows: list[dict[str, Any]], broker_fills: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    fills_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for fill in broker_fills:
        if "_load_error" in fill:
            issues.append(_issue("broker_fill_load_error", detail=fill["_load_error"]))
            continue
        key = _order_key(account_id=fill.get("account_id"), order_id=fill.get("order_id"))
        if key is None:
            issues.append(
                _issue(
                    "broker_fill_missing_order_key",
                    fill_id=fill.get("fill_id"),
                    account_id=fill.get("account_id"),
                    order_id=fill.get("order_id"),
                )
            )
            continue
        fills_by_key[key].append(fill)

    journal_order_keys: set[tuple[str, str]] = set()
    matched_broker_fill_rows = 0
    matched_entry_orders = 0
    matched_exit_orders = 0

    for row in journal_rows:
        for leg, field in (("entry", "order_id_entry"), ("exit", "order_id_exit")):
            order_id = _norm(row.get(field))
            if order_id is None:
                continue
            account_id = _norm(row.get("account_id"))
            if account_id is None:
                issues.append(
                    _issue(
                        "journal_order_missing_account_id",
                        leg=leg,
                        trade_id=row.get("trade_id"),
                        trading_day=_norm(row.get("trading_day")),
                        strategy_id=row.get("strategy_id"),
                        order_id=order_id,
                    )
                )
                continue
            key = (account_id, order_id)
            journal_order_keys.add(key)
            matches = fills_by_key.get(key, [])
            if not matches:
                issues.append(
                    _issue(
                        f"unmatched_journal_{leg}_order",
                        trade_id=row.get("trade_id"),
                        trading_day=_norm(row.get("trading_day")),
                        strategy_id=row.get("strategy_id"),
                        account_id=account_id,
                        order_id=order_id,
                    )
                )
                continue
            if leg == "entry":
                matched_entry_orders += 1
            else:
                matched_exit_orders += 1
            matched_broker_fill_rows += len(matches)

    for key, fills in fills_by_key.items():
        if key in journal_order_keys:
            continue
        account_id, order_id = key
        for fill in fills:
            fill_id = _norm(fill.get("fill_id"))
            issues.append(
                _issue(
                    "unmatched_broker_fill",
                    fill_id=fill_id,
                    account_id=account_id,
                    order_id=order_id,
                    instrument=fill.get("instrument"),
                )
            )

    return {
        "journal_rows": len(journal_rows),
        "broker_fills": len([f for f in broker_fills if "_load_error" not in f]),
        "matched_entry_orders": matched_entry_orders,
        "matched_exit_orders": matched_exit_orders,
        "matched_broker_fills": matched_broker_fill_rows,
        "issue_count": len(issues),
        "issues": issues,
    }


def filter_journal_rows_by_trading_day(
    journal_rows: list[dict[str, Any]], trading_day: str | None
) -> list[dict[str, Any]]:
    day = _norm(trading_day)
    if day is None:
        return journal_rows
    return [row for row in journal_rows if _norm(row.get("trading_day")) == day]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-path", type=Path, help="Path to live_journal.db; defaults to pipeline runtime path")
    parser.add_argument("--fills-path", type=Path, default=FILLS_PATH)
    parser.add_argument("--trading-day", help="Restrict journal rows to this YYYY-MM-DD trading day")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)
    journal_path = args.journal_path or _default_journal_path()

    if not journal_path.exists():
        report = {
            "journal_rows": 0,
            "broker_fills": 0,
            "matched_entry_orders": 0,
            "matched_exit_orders": 0,
            "matched_broker_fills": 0,
            "issue_count": 1,
            "issues": [_issue("missing_live_journal", path=str(journal_path))],
        }
    else:
        journal_rows = filter_journal_rows_by_trading_day(load_journal_rows(journal_path), args.trading_day)
        report = reconcile_live_fills(journal_rows, load_jsonl(args.fills_path))

    if args.trading_day:
        report["trading_day"] = args.trading_day

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Live fill reconciliation: "
            f"{report['matched_entry_orders']} entry order(s), "
            f"{report['matched_exit_orders']} exit order(s), "
            f"{report['matched_broker_fills']} broker fill(s) matched; "
            f"{report['issue_count']} issue(s)"
        )
        for issue in report["issues"]:
            print(f"  - {issue['kind']}: {json.dumps(issue, sort_keys=True)}")

    return 0 if report["issue_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
