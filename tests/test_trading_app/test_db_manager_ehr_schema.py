"""Tests for EHR Stage 2 — additive validated_setups schema migration.

Verifies the 5 new columns added by `init_trading_app_schema()` per the
EARLY_HOLDOUT_REDISCOVERY PASS 2 plan, Stage 2 (2026-05-17):

- validation_mode TEXT DEFAULT 'STANDARD'
- pseudo_oos_window_start DATE
- pseudo_oos_window_end DATE
- verdict_ceiling TEXT
- cumulative_search_count INTEGER

Acceptance criteria 1-5 of docs/runtime/stages/ehr-stage-2-validated-setups-schema.md.
"""

from datetime import date

import duckdb
import pytest

from trading_app.db_manager import init_trading_app_schema
from trading_app.holdout_policy import EHR_MODE_LABEL, STANDARD_MODE_LABEL


@pytest.fixture
def db_path(tmp_path):
    """Create a temp DuckDB with base pipeline schema (daily_features)."""
    path = tmp_path / "test_ehr_schema.db"
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.commit()
    con.close()
    init_trading_app_schema(db_path=path)
    return path


EHR_COLUMNS = {
    "validation_mode",
    "pseudo_oos_window_start",
    "pseudo_oos_window_end",
    "verdict_ceiling",
    "cumulative_search_count",
}

# Minimal set of NOT NULL columns on validated_setups (see db_manager.py line 221+).
# Used by the round-trip / default / nullable tests so the INSERTs satisfy the
# table's schema without claiming anything about the EHR columns themselves.
_BASE_REQUIRED_COLS = (
    "strategy_id, instrument, orb_label, orb_minutes, rr_target, confirm_bars, "
    "entry_model, filter_type, sample_size, win_rate, expectancy_r, "
    "years_tested, all_years_positive, stress_test_passed, status"
)
_BASE_REQUIRED_VALUES = (
    "?, 'MGC', 'CME_REOPEN', 5, 2.0, 3, 'E1', 'NONE', "
    "100, 0.55, 0.25, 5, TRUE, TRUE, 'ACTIVE'"
)


def _column_types(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    rows = con.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table],
    ).fetchall()
    return {name: dtype for name, dtype in rows}


def test_validated_setups_has_ehr_columns(db_path):
    """Acceptance #1 — all 5 columns present with the right types."""
    con = duckdb.connect(str(db_path), read_only=True)
    types = _column_types(con, "validated_setups")
    con.close()

    missing = EHR_COLUMNS - set(types)
    assert not missing, f"EHR columns missing from validated_setups: {missing}"

    # DuckDB reports types in upper-case; pseudo-OOS window columns are DATE,
    # verdict_ceiling + validation_mode are character, cumulative_search_count
    # is INTEGER. We assert the type family, not the exact spelling, because
    # DuckDB occasionally reports "VARCHAR" vs "TEXT" interchangeably.
    assert types["validation_mode"] in {"VARCHAR", "TEXT"}
    assert types["pseudo_oos_window_start"] == "DATE"
    assert types["pseudo_oos_window_end"] == "DATE"
    assert types["verdict_ceiling"] in {"VARCHAR", "TEXT"}
    assert types["cumulative_search_count"] == "INTEGER"


def test_validation_mode_default_is_standard(db_path):
    """Acceptance #2 — INSERT without validation_mode lands as 'STANDARD'.

    Confirms both the SQL DEFAULT literal and that the literal matches the
    canonical Python constant trading_app.holdout_policy.STANDARD_MODE_LABEL.
    """
    con = duckdb.connect(str(db_path))
    con.execute(
        f"INSERT INTO validated_setups ({_BASE_REQUIRED_COLS}) VALUES ({_BASE_REQUIRED_VALUES})",
        ["test_default"],
    )
    con.commit()
    row = con.execute(
        "SELECT validation_mode FROM validated_setups WHERE strategy_id = 'test_default'"
    ).fetchone()
    con.close()

    assert row is not None
    mode = row[0]
    assert mode == "STANDARD"
    assert mode == STANDARD_MODE_LABEL  # canonical-source parity


def test_ehr_columns_nullable_on_standard_row(db_path):
    """Acceptance #3 — STANDARD rows accept NULL for the 4 EHR-specific columns."""
    con = duckdb.connect(str(db_path))
    con.execute(
        f"INSERT INTO validated_setups ({_BASE_REQUIRED_COLS}) VALUES ({_BASE_REQUIRED_VALUES})",
        ["test_nullable"],
    )
    con.commit()
    row = con.execute(
        """
        SELECT pseudo_oos_window_start, pseudo_oos_window_end,
               verdict_ceiling, cumulative_search_count
        FROM validated_setups WHERE strategy_id = 'test_nullable'
        """
    ).fetchone()
    con.close()

    assert row == (None, None, None, None)


def test_round_trip_ehr_row_preserves_columns(db_path):
    """Acceptance #4 — EHR row insert+select round-trips all 5 columns byte-exact."""
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        INSERT INTO validated_setups (
            strategy_id, instrument, orb_label, orb_minutes, rr_target,
            confirm_bars, entry_model, filter_type,
            sample_size, win_rate, expectancy_r,
            years_tested, all_years_positive, stress_test_passed, status,
            validation_mode, pseudo_oos_window_start, pseudo_oos_window_end,
            verdict_ceiling, cumulative_search_count
        )
        VALUES (
            'test_ehr_row', 'MNQ', 'EUROPE_FLOW', 5, 1.5, 1, 'E2', 'NONE',
            100, 0.55, 0.25, 5, TRUE, TRUE, 'ACTIVE',
            ?, ?, ?, ?, ?
        )
        """,
        [
            EHR_MODE_LABEL,
            date(2025, 1, 1),
            date(2026, 1, 1),
            "RESEARCH_PROVISIONAL",
            342,
        ],
    )
    con.commit()
    row = con.execute(
        """
        SELECT validation_mode, pseudo_oos_window_start, pseudo_oos_window_end,
               verdict_ceiling, cumulative_search_count
        FROM validated_setups WHERE strategy_id = 'test_ehr_row'
        """
    ).fetchone()
    con.close()

    assert row is not None
    assert row[0] == EHR_MODE_LABEL == "EARLY_HOLDOUT_REDISCOVERY"
    assert row[1] == date(2025, 1, 1)
    assert row[2] == date(2026, 1, 1)
    assert row[3] == "RESEARCH_PROVISIONAL"
    assert row[4] == 342
