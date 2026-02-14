"""
Tests for edge family hash computation and family building.
"""

import hashlib
import pytest
import duckdb
from datetime import date
from pathlib import Path

from trading_app.db_manager import init_trading_app_schema


@pytest.fixture
def db_path(tmp_path):
    """Create temp DB with full schema + test data."""
    path = tmp_path / "test.db"
    con = duckdb.connect(str(path))

    # Minimal daily_features for FK
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.close()

    init_trading_app_schema(db_path=path)

    # Insert test strategies:
    # s1 and s2 share same trade days (same edge, different RR)
    # s3 has different trade days (stricter filter)
    con = duckdb.connect(str(path))

    for sid, orb, rr, filt, expr, shann in [
        ("MGC_0900_E1_RR2.0_CB2_ORB_G5", "0900", 2.0, "ORB_G5", 0.30, 0.8),
        ("MGC_0900_E1_RR2.5_CB2_ORB_G5", "0900", 2.5, "ORB_G5", 0.45, 1.2),
        ("MGC_0900_E1_RR2.0_CB2_ORB_G8", "0900", 2.0, "ORB_G8", 0.60, 1.5),
    ]:
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size,
             win_rate, expectancy_r, sharpe_ann, years_tested,
             all_years_positive, stress_test_passed, status)
            VALUES (?, 'MGC', ?, 5, ?, 2, 'E1', ?, 100, 0.55, ?, ?,
                    3, TRUE, TRUE, 'active')
        """, [sid, orb, rr, filt, expr, shann])

    # s1 and s2: identical trade days
    for sid in [
        "MGC_0900_E1_RR2.0_CB2_ORB_G5",
        "MGC_0900_E1_RR2.5_CB2_ORB_G5",
    ]:
        for d in [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]:
            con.execute(
                "INSERT INTO strategy_trade_days VALUES (?, ?)", [sid, d]
            )

    # s3: different trade days (stricter filter)
    for d in [date(2024, 1, 2), date(2024, 1, 5)]:
        con.execute(
            "INSERT INTO strategy_trade_days VALUES (?, ?)",
            ["MGC_0900_E1_RR2.0_CB2_ORB_G8", d],
        )

    con.commit()
    con.close()
    return path


def _compute_hash(days: list[date]) -> str:
    """Reference hash computation for tests."""
    day_str = ",".join(str(d) for d in sorted(days))
    return hashlib.md5(day_str.encode()).hexdigest()


class TestHashComputation:
    """Hash is deterministic and collision-resistant."""

    def test_same_days_same_hash(self):
        days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
        assert _compute_hash(days) == _compute_hash(days)

    def test_order_independent(self):
        days_a = [date(2024, 1, 5), date(2024, 1, 2), date(2024, 1, 3)]
        days_b = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
        assert _compute_hash(days_a) == _compute_hash(days_b)

    def test_different_days_different_hash(self):
        days_a = [date(2024, 1, 2), date(2024, 1, 3)]
        days_b = [date(2024, 1, 2), date(2024, 1, 5)]
        assert _compute_hash(days_a) != _compute_hash(days_b)

    def test_empty_days_returns_known_sentinel(self):
        assert _compute_hash([]) == hashlib.md5(b"").hexdigest()


class TestBuildEdgeFamilies:
    """Integration: build_edge_families populates tables correctly."""

    def test_groups_by_hash(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute(
            "SELECT family_hash, member_count FROM edge_families ORDER BY member_count DESC"
        ).fetchall()
        con.close()

        # s1 + s2 share a family (2 members), s3 is alone (1 member)
        assert len(families) == 2
        assert families[0][1] == 2
        assert families[1][1] == 1

    def test_head_is_best_expectancy(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        family = con.execute("""
            SELECT head_strategy_id, head_expectancy_r
            FROM edge_families WHERE member_count = 2
        """).fetchone()
        con.close()

        # s2 has better ExpR (0.45 > 0.30)
        assert family[0] == "MGC_0900_E1_RR2.5_CB2_ORB_G5"
        assert family[1] == 0.45

    def test_validated_setups_tagged(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("""
            SELECT strategy_id, family_hash, is_family_head
            FROM validated_setups
            ORDER BY strategy_id
        """).fetchall()
        con.close()

        # All 3 strategies should have a family_hash
        assert all(r[1] is not None for r in rows)

        # Exactly 2 unique hashes
        hashes = {r[1] for r in rows}
        assert len(hashes) == 2

        # s1 and s2 share the same hash
        by_id = {r[0]: (r[1], r[2]) for r in rows}
        assert by_id["MGC_0900_E1_RR2.0_CB2_ORB_G5"][0] == by_id["MGC_0900_E1_RR2.5_CB2_ORB_G5"][0]

        # s2 is head (best ExpR), s1 is not
        assert by_id["MGC_0900_E1_RR2.5_CB2_ORB_G5"][1] is True
        assert by_id["MGC_0900_E1_RR2.0_CB2_ORB_G5"][1] is False

        # s3 is head of its own singleton family
        assert by_id["MGC_0900_E1_RR2.0_CB2_ORB_G8"][1] is True

    def test_trade_day_count(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute("""
            SELECT member_count, trade_day_count
            FROM edge_families ORDER BY member_count DESC
        """).fetchall()
        con.close()

        assert families[0] == (2, 3)  # 2 members, 3 trade days
        assert families[1] == (1, 2)  # 1 member, 2 trade days

    def test_idempotent(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")
        build_edge_families(str(db_path), "MGC")  # Run again

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM edge_families").fetchone()[0]
        con.close()

        assert count == 2  # No duplicates

    def test_no_strategies_returns_zero(self, db_path):
        from scripts.build_edge_families import build_edge_families

        result = build_edge_families(str(db_path), "MCL")
        assert result == 0
