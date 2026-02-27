"""
Tests for edge family hash computation, median head election, and robustness filter.
"""

import hashlib
import pytest
import duckdb
from datetime import date

from trading_app.db_manager import init_trading_app_schema


@pytest.fixture
def db_path(tmp_path):
    """Create temp DB with full schema + test data.

    3 strategies:
    - s1 (RR2.0 G5, ExpR=0.30, ShANN=0.8) and s2 (RR2.5 G5, ExpR=0.45, ShANN=1.2)
      share identical trade days (same edge, different RR) -> 2-member family
    - s3 (RR2.0 G8, ExpR=0.60, ShANN=1.5) has different trade days -> singleton
    """
    path = tmp_path / "test.db"
    con = duckdb.connect(str(path))

    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL, symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL, bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.close()

    init_trading_app_schema(db_path=path)

    con = duckdb.connect(str(path))

    for sid, orb, rr, filt, expr, shann, sample in [
        ("MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G5", "CME_REOPEN", 2.0, "ORB_G5", 0.30, 0.8, 100),
        ("MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G5", "CME_REOPEN", 2.5, "ORB_G5", 0.45, 1.2, 120),
        ("MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G8", "CME_REOPEN", 2.0, "ORB_G8", 0.60, 1.5, 150),
    ]:
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size,
             win_rate, expectancy_r, sharpe_ann, years_tested,
             all_years_positive, stress_test_passed, status)
            VALUES (?, 'MGC', ?, 5, ?, 2, 'E1', ?, ?, 0.55, ?, ?,
                    3, TRUE, TRUE, 'active')
        """, [sid, orb, rr, filt, sample, expr, shann])

    # s1 and s2: identical trade days
    for sid in [
        "MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G5",
        "MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G5",
    ]:
        for d in [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]:
            con.execute(
                "INSERT INTO strategy_trade_days VALUES (?, ?)", [sid, d]
            )

    # s3: different trade days (stricter filter)
    for d in [date(2024, 1, 2), date(2024, 1, 5)]:
        con.execute(
            "INSERT INTO strategy_trade_days VALUES (?, ?)",
            ["MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G8", d],
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


class TestMedianElection:
    """Head elected by median ExpR, not max (Winner's Curse avoidance)."""

    def test_median_head_not_max(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        # 2-member family: ExpR 0.30 and 0.45
        # Median = 0.375. Both equidistant (0.075).
        # Tiebreak: lower strategy_id -> RR2.0 wins
        family = con.execute("""
            SELECT head_strategy_id, head_expectancy_r, median_expectancy_r
            FROM edge_families WHERE member_count = 2
        """).fetchone()
        con.close()

        assert family[0] == "MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G5"
        assert family[1] == 0.30
        assert family[2] == 0.375  # median stored

    def test_singleton_head_is_itself(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        family = con.execute("""
            SELECT head_strategy_id, head_expectancy_r, median_expectancy_r
            FROM edge_families WHERE member_count = 1
        """).fetchone()
        con.close()

        assert family[0] == "MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G8"
        assert family[1] == 0.60
        assert family[2] == 0.60  # median of 1 = itself

    def test_elect_median_head_function(self):
        from scripts.tools.build_edge_families import _elect_median_head

        # 5 members: ExpR = [0.10, 0.20, 0.30, 0.40, 0.50]
        # Median = 0.30, closest = member with 0.30
        members = [
            ("s1", 0.10, 0.5, 100),
            ("s2", 0.20, 0.6, 100),
            ("s3", 0.30, 0.7, 100),
            ("s4", 0.40, 0.8, 100),
            ("s5", 0.50, 0.9, 100),
        ]
        (head_sid, head_expr, _, _), med = _elect_median_head(members)
        assert head_sid == "s3"
        assert head_expr == 0.30
        assert med == 0.30

    def test_elect_median_tiebreak_by_id(self):
        from scripts.tools.build_edge_families import _elect_median_head

        # 2 members equidistant from median -> lower strategy_id wins
        # Use 0.25 and 0.75 so median=0.50 and both are exactly 0.25 away
        members = [
            ("s_beta", 0.75, 0.8, 100),
            ("s_alpha", 0.25, 0.6, 100),
        ]
        (head_sid, _, _, _), med = _elect_median_head(members)
        assert med == 0.5  # median of [0.25, 0.75]
        # Both are 0.25 from median, s_alpha < s_beta
        assert head_sid == "s_alpha"


class TestRobustnessClassification:
    """Family robustness tagging per Duke Protocol #3c."""

    def test_classify_robust(self):
        from scripts.tools.build_edge_families import classify_family
        assert classify_family(5, 1.0, 0.2, 200) == "ROBUST"
        assert classify_family(10, 0.3, 0.5, 50) == "ROBUST"  # N>=5 always ROBUST
        assert classify_family(21, None, None, None) == "ROBUST"

    def test_classify_whitelisted(self):
        from scripts.tools.build_edge_families import classify_family
        # N in [3,4], ShANN>=0.8, CV<=0.5, trades>=50
        assert classify_family(3, 0.9, 0.2, 150) == "WHITELISTED"
        assert classify_family(4, 1.5, 0.5, 50) == "WHITELISTED"   # CV=0.5 boundary
        assert classify_family(3, 0.8, 0.0, 50) == "WHITELISTED"   # ShANN=0.8 boundary

    def test_classify_purged(self):
        from scripts.tools.build_edge_families import classify_family
        # N=1 with quality bar -> SINGLETON; without -> PURGED
        assert classify_family(1, 1.5, 0.0, 200) == "SINGLETON"  # high ShANN + trades
        assert classify_family(1, 0.5, 0.0, 200) == "PURGED"     # ShANN too low
        assert classify_family(1, 1.5, 0.0, 50) == "PURGED"      # trades too low
        # N=2 always PURGED (below WHITELIST_MIN_MEMBERS, not singleton)
        assert classify_family(2, 1.5, 0.1, 200) == "PURGED"   # pair
        # N>=3 but fails a metric
        assert classify_family(3, 0.5, 0.2, 100) == "PURGED"   # ShANN too low
        assert classify_family(3, 0.9, 0.6, 100) == "PURGED"   # CV too high (>0.5)
        assert classify_family(4, 0.9, 0.2, 30) == "PURGED"    # trades too low (<50)
        assert classify_family(3, None, None, 50) == "PURGED"   # missing metrics

    def test_singleton_purged_in_fixture(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        family = con.execute("""
            SELECT robustness_status FROM edge_families
            WHERE member_count = 1
        """).fetchone()
        con.close()

        # s3 is singleton (N=1) with ShANN=1.5 and 150 trades -> SINGLETON tier
        assert family[0] == "SINGLETON"

    def test_robustness_columns_populated(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute("""
            SELECT robustness_status, cv_expectancy, median_expectancy_r,
                   avg_sharpe_ann, min_member_trades, trade_tier
            FROM edge_families ORDER BY member_count DESC
        """).fetchall()
        con.close()

        # 2-member family
        f2 = families[0]
        assert f2[0] in ("ROBUST", "WHITELISTED", "SINGLETON", "PURGED")
        assert f2[1] is not None  # CV computed for 2+ members
        assert f2[2] == 0.375     # median of [0.30, 0.45]
        assert f2[3] is not None  # avg ShANN
        assert f2[4] == 100       # min(100, 120)
        assert f2[5] == "CORE"    # min_trades=100 >= CORE threshold


class TestTradeTier:
    """Trade tier classification: CORE / REGIME / INVALID."""

    def test_classify_core(self):
        from scripts.tools.build_edge_families import classify_trade_tier
        assert classify_trade_tier(100) == "CORE"
        assert classify_trade_tier(500) == "CORE"

    def test_classify_regime(self):
        from scripts.tools.build_edge_families import classify_trade_tier
        assert classify_trade_tier(30) == "REGIME"
        assert classify_trade_tier(99) == "REGIME"

    def test_classify_invalid(self):
        from scripts.tools.build_edge_families import classify_trade_tier
        assert classify_trade_tier(29) == "INVALID"
        assert classify_trade_tier(0) == "INVALID"
        assert classify_trade_tier(None) == "INVALID"

    def test_trade_tier_in_fixture(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute("""
            SELECT member_count, min_member_trades, trade_tier
            FROM edge_families ORDER BY member_count DESC
        """).fetchall()
        con.close()

        # 2-member family: min(100, 120) = 100 -> CORE
        assert families[0] == (2, 100, "CORE")
        # Singleton: 150 trades -> CORE
        assert families[1] == (1, 150, "CORE")


class TestBuildEdgeFamilies:
    """Integration: core family building still works."""

    def test_groups_by_hash(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute(
            "SELECT family_hash, member_count FROM edge_families ORDER BY member_count DESC"
        ).fetchall()
        con.close()

        assert len(families) == 2
        assert families[0][1] == 2
        assert families[1][1] == 1

    def test_validated_setups_tagged(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("""
            SELECT strategy_id, family_hash, is_family_head
            FROM validated_setups ORDER BY strategy_id
        """).fetchall()
        con.close()

        assert all(r[1] is not None for r in rows)
        hashes = {r[1] for r in rows}
        assert len(hashes) == 2

        by_id = {r[0]: (r[1], r[2]) for r in rows}
        # s1 and s2 share hash
        assert by_id["MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G5"][0] == by_id["MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G5"][0]

        # With median election: s1 (RR2.0, ExpR=0.30) is head (closer to median by tiebreak)
        assert by_id["MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G5"][1] is True
        assert by_id["MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G5"][1] is False

        # s3 is head of singleton
        assert by_id["MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G8"][1] is True

    def test_trade_day_count(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute("""
            SELECT member_count, trade_day_count
            FROM edge_families ORDER BY member_count DESC
        """).fetchall()
        con.close()

        assert families[0] == (2, 3)
        assert families[1] == (1, 2)

    def test_idempotent(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")
        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM edge_families").fetchone()[0]
        con.close()

        assert count == 2

    def test_no_strategies_returns_zero(self, db_path):
        from scripts.tools.build_edge_families import build_edge_families

        result = build_edge_families(str(db_path), "MCL")
        assert result == 0
