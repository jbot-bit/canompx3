import json
import subprocess
import sys
from pathlib import Path

from scripts.tools.reconcile_live_fills import filter_journal_rows_by_trading_day, reconcile_live_fills


def test_reconcile_matches_entry_and_exit_by_account_and_order_id():
    rows = [
        {
            "trade_id": "trade-1",
            "strategy_id": "MNQ_TOKYO_OPEN_E2",
            "account_id": "20372221",
            "order_id_entry": "1001",
            "order_id_exit": "1002",
        }
    ]
    fills = [
        {"fill_id": "topstepx-1", "account_id": 20372221, "order_id": 1001},
        {"fill_id": "topstepx-2", "account_id": "20372221", "order_id": "1002"},
    ]

    report = reconcile_live_fills(rows, fills)

    assert report["matched_entry_orders"] == 1
    assert report["matched_exit_orders"] == 1
    assert report["issues"] == []


def test_reconcile_keeps_accounts_independent_when_order_ids_overlap():
    rows = [
        {
            "trade_id": "primary-trade",
            "strategy_id": "MNQ_TOKYO_OPEN_E2",
            "account_id": "111",
            "order_id_entry": "42",
            "order_id_exit": None,
        }
    ]
    fills = [
        {"fill_id": "shadow-fill", "account_id": "222", "order_id": "42"},
    ]

    report = reconcile_live_fills(rows, fills)

    kinds = [issue["kind"] for issue in report["issues"]]
    assert "unmatched_journal_entry_order" in kinds
    assert "unmatched_broker_fill" in kinds


def test_reconcile_flags_journal_orders_without_account_id():
    rows = [
        {
            "trade_id": "trade-1",
            "strategy_id": "MNQ_TOKYO_OPEN_E2",
            "account_id": None,
            "order_id_entry": "1001",
            "order_id_exit": None,
        }
    ]
    fills = [{"fill_id": "topstepx-1", "account_id": "20372221", "order_id": "1001"}]

    report = reconcile_live_fills(rows, fills)

    kinds = [issue["kind"] for issue in report["issues"]]
    assert "journal_order_missing_account_id" in kinds
    assert "unmatched_broker_fill" in kinds


def test_reconcile_treats_partial_fills_as_one_matched_order():
    rows = [
        {
            "trade_id": "trade-1",
            "strategy_id": "MNQ_TOKYO_OPEN_E2",
            "account_id": "20372221",
            "order_id_entry": "1001",
            "order_id_exit": None,
        }
    ]
    fills = [
        {"fill_id": "partial-1", "account_id": "20372221", "order_id": "1001"},
        {"fill_id": "partial-2", "account_id": "20372221", "order_id": "1001"},
    ]

    report = reconcile_live_fills(rows, fills)

    assert report["matched_entry_orders"] == 1
    assert report["matched_broker_fills"] == 2
    assert report["issues"] == []


def test_reconcile_counts_matched_fill_rows_without_fill_ids():
    rows = [
        {
            "trade_id": "trade-1",
            "strategy_id": "MNQ_TOKYO_OPEN_E2",
            "account_id": "20372221",
            "order_id_entry": "1001",
            "order_id_exit": None,
        }
    ]
    fills = [{"account_id": "20372221", "order_id": "1001"}]

    report = reconcile_live_fills(rows, fills)

    assert report["matched_broker_fills"] == 1
    assert report["issues"] == []


def test_filter_journal_rows_by_trading_day_keeps_sessions_independent():
    rows = [
        {"trade_id": "old", "trading_day": "2026-05-31"},
        {"trade_id": "today", "trading_day": "2026-06-01"},
        {"trade_id": "missing-day", "trading_day": None},
    ]

    filtered = filter_journal_rows_by_trading_day(rows, "2026-06-01")

    assert [row["trade_id"] for row in filtered] == ["today"]


def test_cli_runs_when_invoked_as_script_path(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[2]
    missing_journal = tmp_path / "missing_live_journal.db"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/reconcile_live_fills.py",
            "--journal-path",
            str(missing_journal),
            "--json",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["issues"][0]["kind"] == "missing_live_journal"
    assert "ModuleNotFoundError" not in result.stderr
