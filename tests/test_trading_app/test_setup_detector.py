"""
Tests for trading_app.setup_detector module.
"""

import sys
from pathlib import Path
from datetime import date

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.setup_detector import detect_setups
from trading_app.config import NoFilter, OrbSizeFilter


def _setup_db(tmp_path, days_data):
    """Create temp DB with daily_features rows."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    from pipeline.init_db import DAILY_FEATURES_SCHEMA
    con.execute(DAILY_FEATURES_SCHEMA)

    for d in days_data:
        cols = list(d.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        con.execute(
            f"INSERT INTO daily_features ({col_str}) VALUES ({placeholders})",
            list(d.values()),
        )
    con.commit()
    con.close()
    return db_path


class TestDetectSetups:
    """Tests for detect_setups()."""

    def test_no_filter_returns_all_break_days(self, tmp_path):
        """NoFilter returns all days with a break."""
        days = [
            {"trading_day": date(2024, 1, 1), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_high": 2700.0, "orb_0900_low": 2690.0,
             "orb_0900_break_dir": "long", "orb_0900_size": 10.0},
            {"trading_day": date(2024, 1, 2), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_high": 2705.0, "orb_0900_low": 2695.0,
             "orb_0900_break_dir": "short", "orb_0900_size": 10.0},
            {"trading_day": date(2024, 1, 3), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_high": 2710.0, "orb_0900_low": 2700.0},  # no break
        ]
        db_path = _setup_db(tmp_path, days)

        con = duckdb.connect(str(db_path), read_only=True)
        results = detect_setups(con, NoFilter(), "0900", "MGC")
        con.close()

        assert len(results) == 2
        assert results[0][0] == date(2024, 1, 1)
        assert results[1][0] == date(2024, 1, 2)

    def test_orb_size_filter(self, tmp_path):
        """OrbSizeFilter restricts by ORB size."""
        days = [
            {"trading_day": date(2024, 1, 1), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_high": 2700.0, "orb_0900_low": 2697.0,
             "orb_0900_break_dir": "long", "orb_0900_size": 3.0},
            {"trading_day": date(2024, 1, 2), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_high": 2700.0, "orb_0900_low": 2692.0,
             "orb_0900_break_dir": "long", "orb_0900_size": 8.0},
        ]
        db_path = _setup_db(tmp_path, days)

        con = duckdb.connect(str(db_path), read_only=True)
        f = OrbSizeFilter(filter_type="L4", description="<4", max_size=4.0)
        results = detect_setups(con, f, "0900", "MGC")
        con.close()

        assert len(results) == 1
        assert results[0][0] == date(2024, 1, 1)

    def test_date_range_filter(self, tmp_path):
        """Start/end dates restrict results."""
        days = [
            {"trading_day": date(2024, 1, 1), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_break_dir": "long", "orb_0900_high": 2700.0, "orb_0900_low": 2690.0},
            {"trading_day": date(2024, 6, 1), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_break_dir": "long", "orb_0900_high": 2700.0, "orb_0900_low": 2690.0},
            {"trading_day": date(2024, 12, 1), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_break_dir": "long", "orb_0900_high": 2700.0, "orb_0900_low": 2690.0},
        ]
        db_path = _setup_db(tmp_path, days)

        con = duckdb.connect(str(db_path), read_only=True)
        results = detect_setups(
            con, NoFilter(), "0900", "MGC",
            start_date=date(2024, 3, 1), end_date=date(2024, 9, 1),
        )
        con.close()

        assert len(results) == 1
        assert results[0][0] == date(2024, 6, 1)

    def test_empty_result_impossible_filter(self, tmp_path):
        """Filter that matches nothing returns empty list."""
        days = [
            {"trading_day": date(2024, 1, 1), "symbol": "MGC", "orb_minutes": 5,
             "orb_0900_break_dir": "long", "orb_0900_high": 2700.0, "orb_0900_low": 2690.0,
             "orb_0900_size": 10.0},
        ]
        db_path = _setup_db(tmp_path, days)

        con = duckdb.connect(str(db_path), read_only=True)
        # max_size=1 excludes the 10-point ORB
        f = OrbSizeFilter(filter_type="TINY", description="tiny", max_size=1.0)
        results = detect_setups(con, f, "0900", "MGC")
        con.close()

        assert len(results) == 0

    def test_no_days_returns_empty(self, tmp_path):
        """Empty daily_features returns empty list."""
        db_path = _setup_db(tmp_path, [])

        con = duckdb.connect(str(db_path), read_only=True)
        results = detect_setups(con, NoFilter(), "0900", "MGC")
        con.close()

        assert results == []
