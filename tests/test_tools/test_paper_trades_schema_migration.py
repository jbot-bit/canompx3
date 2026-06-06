"""
T4 — paper_trades schema migration regression (root-cause fix).

The production gold.db paper_trades table carried 3 columns
(execution_source, pnl_dollar, notes) that the canonical initializer could not
reproduce — they reached prod via an uncommitted ad-hoc ALTER. A fresh rebuild
(CanonMPX_DailyRefresh) would silently drop them, breaking BOTH live trade
logging (log_trade.py writes execution_source='live') AND shadow accumulation
(regime_shadow_runner.py writes execution_source='shadow').

These tests pin the root fix in init_trading_app_schema:
  1. A freshly-built DB HAS the 3 columns with the correct defaults.
  2. Running the initializer twice is a no-op (idempotent migration).
  3. verify_trading_app_schema() reports all_valid=True (parity guard sees the
     columns — closes the detection gap that let the drift hide).
"""

from __future__ import annotations

import duckdb

from trading_app.db_manager import init_trading_app_schema, verify_trading_app_schema

_EXPECTED_NEW = {"execution_source", "pnl_dollar", "notes"}


def _init_schema(db) -> None:
    """init_trading_app_schema creates orb_outcomes with a FK to daily_features,
    so that table must exist first. Mirror the canonical key tuple."""
    with duckdb.connect(str(db)) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS daily_features (
                 symbol VARCHAR, orb_minutes INTEGER, trading_day DATE,
                 UNIQUE (symbol, trading_day, orb_minutes))"""
        )
    init_trading_app_schema(db_path=db)


def _paper_trades_columns(db) -> set[str]:
    with duckdb.connect(str(db), read_only=True) as con:
        return {r[0] for r in con.execute("DESCRIBE paper_trades").fetchall()}


def test_fresh_db_has_migration_columns(tmp_path):
    db = tmp_path / "fresh.db"
    _init_schema(db)
    cols = _paper_trades_columns(db)
    assert cols >= _EXPECTED_NEW, f"missing migration columns: {_EXPECTED_NEW - cols}"


def test_execution_source_default_is_backfill(tmp_path):
    """A row inserted without execution_source defaults to 'backfill' — matching
    production, where all pre-migration rows ARE backfill."""
    db = tmp_path / "default.db"
    _init_schema(db)
    with duckdb.connect(str(db)) as con:
        con.execute(
            """INSERT INTO paper_trades (trading_day, orb_label, strategy_id, instrument)
               VALUES (DATE '2026-06-03', 'COMEX_SETTLE', 'L1', 'MNQ')"""
        )
        src, notes = con.execute("SELECT execution_source, notes FROM paper_trades WHERE strategy_id='L1'").fetchone()
    assert src == "backfill", "execution_source must default to 'backfill'"
    assert notes == "", "notes must default to ''"


def test_init_is_idempotent(tmp_path):
    db = tmp_path / "twice.db"
    _init_schema(db)
    cols_first = _paper_trades_columns(db)
    _init_schema(db)  # second run must be a no-op
    cols_second = _paper_trades_columns(db)
    assert cols_first == cols_second, "second init must not change the schema"


def test_verify_schema_passes_on_fresh_db(tmp_path):
    db = tmp_path / "verify.db"
    _init_schema(db)
    all_valid, violations = verify_trading_app_schema(db_path=db)
    assert all_valid, f"verify_trading_app_schema reported violations: {violations}"
    # Specifically, the parity guard must not flag paper_trades.
    assert not any("paper_trades" in v for v in violations)
