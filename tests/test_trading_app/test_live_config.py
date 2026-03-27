"""Tests for trading_app.live_config — live portfolio configuration."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from trading_app.live_config import (
    LIVE_MIN_EXPECTANCY_DOLLARS_MULT,
    LIVE_PORTFOLIO,
    LiveStrategySpec,
    _check_dollar_gate,
    _load_best_experimental_variant,
    _load_best_regime_variant,
    build_live_portfolio,
)


@pytest.fixture
def live_config_db(tmp_path):
    """Create temp DB with validated_setups, experimental_strategies, family_rr_locks."""
    db_path = tmp_path / "live_test.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id VARCHAR,
            instrument VARCHAR,
            orb_label VARCHAR,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            sharpe_ratio DOUBLE,
            max_drawdown_r DOUBLE,
            status VARCHAR,
            orb_minutes INTEGER DEFAULT 5,
            stop_multiplier DOUBLE DEFAULT 1.0,
            fdr_significant BOOLEAN,
            noise_risk BOOLEAN,
            oos_exp_r DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE experimental_strategies (
            strategy_id VARCHAR,
            instrument VARCHAR,
            orb_label VARCHAR,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            orb_minutes INTEGER DEFAULT 5,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            sharpe_ratio DOUBLE,
            max_drawdown_r DOUBLE,
            median_risk_points DOUBLE,
            stop_multiplier DOUBLE DEFAULT 1.0
        )
    """)
    from pipeline.init_db import FAMILY_RR_LOCKS_SCHEMA

    con.execute(FAMILY_RR_LOCKS_SCHEMA)
    con.close()
    # Rolling portfolio queries regime_strategies + regime_validated — use real schema.
    from trading_app.regime.schema import init_regime_schema

    init_regime_schema(db_path=db_path)
    return db_path


class TestLiveStrategySpec:
    def test_spec_is_frozen(self):
        spec = LiveStrategySpec("fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", None)
        with pytest.raises(AttributeError):
            spec.tier = "regime"

    def test_spec_fields(self):
        spec = LiveStrategySpec("fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", "rolling")
        assert spec.family_id == "fam1"
        assert spec.regime_gate == "rolling"

    def test_new_fields_default_none(self):
        """New gate fields default to None (no gate active)."""
        spec = LiveStrategySpec("fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", None)
        assert spec.active_months is None
        assert spec.weight_override is None
        assert spec.recovery_expr_threshold is None

    def test_active_months_field(self):
        spec = LiveStrategySpec(
            "fam1",
            "core",
            "TOKYO_OPEN",
            "E1",
            "ORB_G4",
            None,
            active_months=frozenset({11, 12, 1, 2}),
        )
        assert spec.active_months == frozenset({11, 12, 1, 2})
        assert 1 in spec.active_months
        assert 6 not in spec.active_months

    def test_weight_override_field(self):
        spec = LiveStrategySpec(
            "fam1",
            "core",
            "TOKYO_OPEN",
            "E1",
            "ORB_G4",
            None,
            weight_override=0.5,
        )
        assert spec.weight_override == 0.5

    def test_recovery_threshold_field(self):
        spec = LiveStrategySpec(
            "fam1",
            "core",
            "TOKYO_OPEN",
            "E1",
            "ORB_G4",
            None,
            weight_override=0.5,
            recovery_expr_threshold=0.25,
        )
        assert spec.recovery_expr_threshold == 0.25

    def test_recovery_requires_weight_override(self):
        """recovery_expr_threshold without weight_override is invalid."""
        with pytest.raises(ValueError, match="recovery_expr_threshold requires weight_override"):
            LiveStrategySpec(
                "fam1",
                "core",
                "TOKYO_OPEN",
                "E1",
                "ORB_G4",
                None,
                recovery_expr_threshold=0.25,
            )

    def test_invalid_month_rejected(self):
        with pytest.raises(ValueError, match="active_months"):
            LiveStrategySpec(
                "fam1",
                "core",
                "TOKYO_OPEN",
                "E1",
                "ORB_G4",
                None,
                active_months=frozenset({13}),
            )

    def test_weight_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="weight_override"):
            LiveStrategySpec(
                "fam1",
                "core",
                "TOKYO_OPEN",
                "E1",
                "ORB_G4",
                None,
                weight_override=1.5,
            )

    def test_negative_recovery_rejected(self):
        with pytest.raises(ValueError, match="recovery_expr_threshold"):
            LiveStrategySpec(
                "fam1",
                "core",
                "TOKYO_OPEN",
                "E1",
                "ORB_G4",
                None,
                weight_override=0.5,
                recovery_expr_threshold=-0.1,
            )


class TestLivePortfolio:
    def test_portfolio_not_empty(self):
        assert len(LIVE_PORTFOLIO) > 0

    def test_all_entries_are_specs(self):
        for spec in LIVE_PORTFOLIO:
            assert isinstance(spec, LiveStrategySpec)

    def test_tiers_are_valid(self):
        valid_tiers = {"core", "hot", "regime"}
        for spec in LIVE_PORTFOLIO:
            assert spec.tier in valid_tiers, f"{spec.family_id} has invalid tier: {spec.tier}"


class TestLoadBestRegimeVariant:
    def test_found(self, live_config_db):
        """Matching active strategy with locked RR returns dict."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status)
            VALUES (
                'MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G4', 'MGC', 'TOKYO_OPEN', 'E1',
                2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 'active'
            )
        """)
        con.execute("""
            INSERT INTO experimental_strategies (strategy_id, instrument, orb_label,
                entry_model, rr_target, confirm_bars, filter_type, expectancy_r,
                win_rate, sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
            VALUES (
                'MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G4', 'MGC', 'TOKYO_OPEN', 'E1',
                2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 4.2
            )
        """)
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E1', 5, 1, 2.0, 'ONLY_RR',
                    1.2, 3.5, 150, 0.35, 50.0)
        """)
        con.close()

        result, _ = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result is not None
        assert result["strategy_id"] == "MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G4"
        assert result["sharpe_ratio"] == 1.2

    def test_not_found(self, live_config_db):
        """No matching strategy returns None."""
        result, _ = _load_best_regime_variant(live_config_db, "MGC", "9999", "E1", "ORB_G4")
        assert result is None

    def test_inactive_filtered(self, live_config_db):
        """Inactive strategy is not returned even with a matching RR lock."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status)
            VALUES (
                'test_inactive', 'MGC', 'TOKYO_OPEN', 'E1',
                2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 'purged'
            )
        """)
        # Lock exists, but strategy status is 'purged' — should still return None.
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E1', 5, 1, 2.0, 'ONLY_RR',
                    1.2, 3.5, 150, 0.35, 50.0)
        """)
        con.close()

        result, _ = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result is None

    def test_best_expectancy_selected(self, live_config_db):
        """Among locked-RR candidates, highest expectancy_r is returned."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status)
            VALUES
                ('low_expr', 'MGC', 'TOKYO_OPEN', 'E1', 2.0, 1, 'ORB_G4', 0.30, 0.50, 100, 0.8, 4.0, 'active'),
                ('high_expr', 'MGC', 'TOKYO_OPEN', 'E1', 1.5, 2, 'ORB_G4', 0.40, 0.55, 120, 1.5, 3.0, 'active')
        """)
        # Each (confirm_bars) family locked at its own RR — both match INNER JOIN.
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES
                ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E1', 5, 1, 2.0, 'ONLY_RR', 0.8, 4.0, 100, 0.30, 50.0),
                ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E1', 5, 2, 1.5, 'ONLY_RR', 1.5, 3.0, 120, 0.40, 50.0)
        """)
        con.close()

        result, _ = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result["strategy_id"] == "high_expr"

    def test_fdr_significant_preferred_over_higher_expr(self, live_config_db):
        """FDR-significant variant wins over higher-ExpR FDR-failing variant."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status, orb_minutes,
                fdr_significant)
            VALUES
                ('fdr_fail_high', 'MGC', 'NYSE_OPEN', 'E2', 1.0, 1, 'ORB_G8',
                 0.35, 0.54, 144, 1.1, 4.0, 'active', 5, FALSE),
                ('fdr_pass_low', 'MGC', 'NYSE_OPEN', 'E2', 1.0, 1, 'ORB_G8',
                 0.22, 0.60, 472, 0.8, 6.0, 'active', 30, TRUE)
        """)
        # Two apertures (5m and 30m) each locked at RR=1.0.
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES
                ('MGC', 'NYSE_OPEN', 'ORB_G8', 'E2', 5, 1, 1.0, 'ONLY_RR',
                 1.1, 4.0, 144, 0.35, 15.0),
                ('MGC', 'NYSE_OPEN', 'ORB_G8', 'E2', 30, 1, 1.0, 'ONLY_RR',
                 0.8, 6.0, 472, 0.22, 47.0)
        """)
        con.close()

        result, _ = _load_best_regime_variant(live_config_db, "MGC", "NYSE_OPEN", "E2", "ORB_G8")
        assert result is not None
        # FDR-significant variant (ExpR=0.22) must win over FDR-failing (ExpR=0.35).
        assert result["strategy_id"] == "fdr_pass_low"

    def test_unlocked_rr_excluded(self, live_config_db):
        """Strategy at non-locked RR is excluded by INNER JOIN."""
        con = duckdb.connect(str(live_config_db))
        # Strategy at RR2.5, but family locked at RR1.0 — must NOT return.
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status)
            VALUES (
                'wrong_rr', 'MGC', 'TOKYO_OPEN', 'E1',
                2.5, 1, 'ORB_G4', 0.50, 0.55, 200, 1.5, 3.0, 'active'
            )
        """)
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E1', 5, 1, 1.0, 'SHARPE_DD',
                    1.2, 5.0, 180, 0.20, 50.0)
        """)
        con.close()

        result, _ = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result is None  # RR2.5 != locked RR1.0

    def test_locked_rr_only_returned(self, live_config_db):
        """Only the locked RR variant is returned from a multi-RR family."""
        con = duckdb.connect(str(live_config_db))
        # Two strategies, same family, different RR. Lock at RR1.5.
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status)
            VALUES
                ('rr10', 'MGC', 'TOKYO_OPEN', 'E1', 1.0, 1, 'ORB_G4', 0.45, 0.60, 200, 1.8, 3.0, 'active'),
                ('rr15', 'MGC', 'TOKYO_OPEN', 'E1', 1.5, 1, 'ORB_G4', 0.30, 0.52, 180, 1.2, 5.0, 'active')
        """)
        # Lock at RR1.5 (via JK-MaxExpR criterion).
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E1', 5, 1, 1.5, 'SHARPE_DD',
                    1.2, 5.0, 180, 0.30, 50.0)
        """)
        con.close()

        result, _ = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result is not None
        assert result["strategy_id"] == "rr15"
        assert result["rr_target"] == 1.5


class TestJKFallbackRR:
    """JK-equal liveability tiebreaker: when locked RR fails LIVE_MIN,
    try JK-equal alternatives that pass."""

    def _insert_family(self, con, rr, sharpe, expr, n=200, status="active"):
        """Insert a validated strategy + experimental_strategies row + RR lock."""
        sid = f"MGC_TOKYO_OPEN_E2_RR{rr}_CB1_ORB_G4"
        con.execute(
            """INSERT INTO validated_setups
               (strategy_id, instrument, orb_label, entry_model, rr_target,
                confirm_bars, filter_type, expectancy_r, win_rate, sample_size,
                sharpe_ratio, max_drawdown_r, status, orb_minutes, noise_risk, oos_exp_r)
               VALUES (?, 'MGC', 'TOKYO_OPEN', 'E2', ?, 1, 'ORB_G4', ?, 0.55, ?,
                       ?, 5.0, ?, 5, FALSE, 0.15)""",
            [sid, rr, expr, n, sharpe, status],
        )
        con.execute(
            """INSERT INTO experimental_strategies
               (strategy_id, instrument, orb_label, entry_model, rr_target,
                confirm_bars, filter_type, orb_minutes, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
               VALUES (?, 'MGC', 'TOKYO_OPEN', 'E2', ?, 1, 'ORB_G4', 5, ?, 0.55,
                       ?, ?, 5.0, 6.3)""",
            [sid, rr, expr, n, sharpe],
        )

    def test_no_fallback_when_locked_rr_passes(self, live_config_db):
        """When locked RR passes LIVE_MIN, no fallback fires."""
        con = duckdb.connect(str(live_config_db))
        self._insert_family(con, 1.0, 0.24, 0.25)  # passes LIVE_MIN
        self._insert_family(con, 1.5, 0.23, 0.30)
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E2', 5, 1, 1.0, 'MAX_SHARPE',
                    0.24, 5.0, 200, 0.25, 50.0)
        """)
        con.close()

        result, fallback_note = _load_best_regime_variant(
            live_config_db,
            "MGC",
            "TOKYO_OPEN",
            "E2",
            "ORB_G4",
        )
        assert result is not None
        assert result["rr_target"] == 1.0  # locked RR used directly
        assert fallback_note is None  # no fallback

    def test_fallback_fires_when_locked_rr_fails_live_min(self, live_config_db):
        """When locked RR fails LIVE_MIN but JK-equal alt passes, fallback fires."""
        con = duckdb.connect(str(live_config_db))
        self._insert_family(con, 1.0, 0.236, 0.186)  # fails LIVE_MIN (0.186 < 0.22)
        self._insert_family(con, 1.5, 0.228, 0.235)  # passes LIVE_MIN, JK-equal
        self._insert_family(con, 2.0, 0.208, 0.257)  # passes LIVE_MIN, JK-equal
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E2', 5, 1, 1.0, 'MAX_SHARPE',
                    0.236, 5.0, 200, 0.186, 50.0)
        """)
        con.close()

        result, fallback_note = _load_best_regime_variant(
            live_config_db,
            "MGC",
            "TOKYO_OPEN",
            "E2",
            "ORB_G4",
        )
        assert result is not None
        assert result["rr_target"] == 1.5  # highest Sharpe among JK-equal gate-passers
        assert fallback_note is not None
        assert "JK_FALLBACK" in fallback_note
        assert "locked_rr=1.0" in fallback_note
        assert "fallback_rr=1.5" in fallback_note

    def test_no_fallback_when_no_jk_equal_passes(self, live_config_db):
        """When locked RR fails and all alternatives also fail LIVE_MIN, returns None."""
        con = duckdb.connect(str(live_config_db))
        self._insert_family(con, 1.0, 0.24, 0.15)  # fails LIVE_MIN
        self._insert_family(con, 1.5, 0.23, 0.18)  # also fails LIVE_MIN
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E2', 5, 1, 1.0, 'MAX_SHARPE',
                    0.24, 5.0, 200, 0.15, 50.0)
        """)
        con.close()

        result, fallback_note = _load_best_regime_variant(
            live_config_db,
            "MGC",
            "TOKYO_OPEN",
            "E2",
            "ORB_G4",
        )
        assert result is None
        assert fallback_note is None

    def test_fallback_excludes_jk_unequal(self, live_config_db):
        """Alt with significantly worse Sharpe (JK p < 0.05) is NOT used as fallback."""
        con = duckdb.connect(str(live_config_db))
        # Locked: high Sharpe, fails LIVE_MIN
        self._insert_family(con, 1.0, 2.5, 0.15, n=500)
        # Alt: MUCH worse Sharpe (JK will reject), passes LIVE_MIN
        self._insert_family(con, 4.0, 0.3, 0.30, n=500)
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E2', 5, 1, 1.0, 'MAX_SHARPE',
                    2.5, 5.0, 500, 0.15, 50.0)
        """)
        con.close()

        result, fallback_note = _load_best_regime_variant(
            live_config_db,
            "MGC",
            "TOKYO_OPEN",
            "E2",
            "ORB_G4",
        )
        # JK should reject this alt — significantly worse Sharpe
        assert result is None
        assert fallback_note is None

    def test_fallback_note_has_audit_fields(self, live_config_db):
        """Fallback note contains all required audit fields."""
        con = duckdb.connect(str(live_config_db))
        self._insert_family(con, 1.0, 0.236, 0.186)
        self._insert_family(con, 1.5, 0.228, 0.235)
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E2', 5, 1, 1.0, 'MAX_SHARPE',
                    0.236, 5.0, 200, 0.186, 50.0)
        """)
        con.close()

        _, fallback_note = _load_best_regime_variant(
            live_config_db,
            "MGC",
            "TOKYO_OPEN",
            "E2",
            "ORB_G4",
        )
        assert fallback_note is not None
        # Required audit fields
        assert "locked_rr=" in fallback_note
        assert "fallback_rr=" in fallback_note
        assert "jk_p=" in fallback_note
        assert "ExpR=" in fallback_note
        assert "Sharpe=" in fallback_note
        assert "reason=" in fallback_note


class TestCheckDollarGate:
    """_check_dollar_gate: 4 branches — None guard, fail, pass, exception."""

    def _variant(self, expectancy_r: float, median_risk_points: float | None) -> dict:
        return {"expectancy_r": expectancy_r, "median_risk_points": median_risk_points}

    def test_none_guard_blocks(self):
        """Missing median_risk_points must BLOCK (fail-closed, not skip gate)."""
        passes, note = _check_dollar_gate(self._variant(0.10, None), "MGC")
        assert passes is False
        assert "blocked" in note

    def test_fails_when_exp_below_threshold(self):
        """Tiny ORB on MGC: exp$ well below 1.3x RT cost must fail."""
        # MGC: point_value=$10, total_friction=$5.74
        # median=0.5pt → 1R$=0.5*10=$5.00; exp$=0.10*5.00=$0.50
        # threshold=1.3*5.74=$7.46 → should fail
        passes, note = _check_dollar_gate(self._variant(0.10, 0.5), "MGC")
        assert passes is False
        assert "dollar gate failed" not in note  # note uses "Exp$" format, not "failed"
        assert "Exp$" in note or "<" in note

    def test_passes_when_exp_meets_threshold(self):
        """Large ORB on MGC: exp$ well above 1.3x RT cost must pass."""
        # MGC: median=5pt → 1R$=5*10=$50.00; exp$=0.30*50.00=$15.00
        # threshold=$7.46 → should pass
        passes, note = _check_dollar_gate(self._variant(0.30, 5.0), "MGC")
        assert passes is True
        assert ">=" in note

    def test_exception_in_cost_spec_blocks(self):
        """Unknown instrument raises in get_cost_spec — must BLOCK (fail-closed)."""
        passes, note = _check_dollar_gate(self._variant(0.10, 3.0), "UNKNOWN_INSTRUMENT_XYZ")
        assert passes is False
        assert "BLOCKED" in note

    def test_one_r_excludes_friction(self):
        """1R dollars = median_risk_pts * point_value only — friction must NOT be added."""
        from pipeline.cost_model import get_cost_spec

        spec = get_cost_spec("MGC")
        median = 3.0
        correct_one_r = median * spec.point_value  # $30.00
        wrong_one_r = median * spec.point_value + spec.total_friction  # $35.74

        # Choose ExpR so the result distinguishes the two formulas
        expr = 0.50
        correct_exp_dollars = expr * correct_one_r
        wrong_exp_dollars = expr * wrong_one_r
        assert correct_exp_dollars != wrong_exp_dollars  # sanity

        # Set median_risk_points high enough that both formulas pass the gate,
        # so we can inspect the note string for the correct Exp$ value.
        passes, note = _check_dollar_gate({"expectancy_r": expr, "median_risk_points": median}, "MGC")
        assert passes is True
        # The note contains "Exp$XX.XX" — verify it matches the CORRECT formula
        assert f"Exp${correct_exp_dollars:.2f}" in note
        assert f"Exp${wrong_exp_dollars:.2f}" not in note

    def test_multiplier_constant_is_applied(self):
        """Gate threshold equals LIVE_MIN_EXPECTANCY_DOLLARS_MULT * RT cost."""
        # MGC total_friction=$5.74; boundary case: exp$ just above vs just below threshold
        from pipeline.cost_model import get_cost_spec

        spec = get_cost_spec("MGC")
        threshold = LIVE_MIN_EXPECTANCY_DOLLARS_MULT * spec.total_friction
        median = 1.0
        one_r = median * spec.point_value
        # ExpR that lands exactly at threshold (boundary should pass)
        expr_at_threshold = threshold / one_r
        passes, _ = _check_dollar_gate(self._variant(expr_at_threshold, median), "MGC")
        assert passes is True
        # ExpR just below threshold should fail
        passes_below, _ = _check_dollar_gate(self._variant(expr_at_threshold * 0.99, median), "MGC")
        assert passes_below is False


class TestLoadBestExperimentalVariant:
    def test_found(self, live_config_db):
        """Matching experimental strategy with positive ExpR returns dict."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO experimental_strategies (strategy_id, instrument, orb_label,
                entry_model, rr_target, confirm_bars, filter_type, expectancy_r,
                win_rate, sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
            VALUES (
                'exp_test', 'MGC', 'TOKYO_OPEN', 'E2', 2.0, 1, 'ORB_G5',
                0.25, 0.51, 200, 1.0, 4.0, 3.5
            )
        """)
        con.close()

        result = _load_best_experimental_variant(live_config_db, "MGC", "TOKYO_OPEN", "E2", "ORB_G5")
        assert result is not None
        assert result["expectancy_r"] == 0.25

    def test_negative_expr_filtered(self, live_config_db):
        """Negative expectancy is filtered out."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO experimental_strategies (strategy_id, instrument, orb_label,
                entry_model, rr_target, confirm_bars, filter_type, expectancy_r,
                win_rate, sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
            VALUES (
                'neg_test', 'MGC', 'TOKYO_OPEN', 'E2', 2.0, 1, 'ORB_G5',
                -0.10, 0.45, 200, -0.5, 5.0, 3.5
            )
        """)
        con.close()

        result = _load_best_experimental_variant(live_config_db, "MGC", "TOKYO_OPEN", "E2", "ORB_G5")
        assert result is None

    def test_not_found(self, live_config_db):
        result = _load_best_experimental_variant(live_config_db, "MNQ", "CME_REOPEN", "E1", "ORB_G4")
        assert result is None


def _seed_seasonal_gate_data(db_path: Path) -> None:
    """Insert minimal data so build_live_portfolio can load one strategy for MGC."""
    con = duckdb.connect(str(db_path))
    con.execute("""
        INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
            rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
            sample_size, sharpe_ratio, max_drawdown_r, status, orb_minutes,
            fdr_significant, noise_risk, oos_exp_r)
        VALUES (
            'MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4', 'MGC', 'TOKYO_OPEN', 'E2',
            2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 'active', 5, TRUE,
            FALSE, 0.50
        )
    """)
    con.execute("""
        INSERT INTO experimental_strategies (strategy_id, instrument, orb_label,
            entry_model, rr_target, confirm_bars, filter_type, expectancy_r,
            win_rate, sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
        VALUES (
            'MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4', 'MGC', 'TOKYO_OPEN', 'E2',
            2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 5.0
        )
    """)
    con.execute("""
        INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
            orb_minutes, confirm_bars, locked_rr, method,
            sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
        VALUES ('MGC', 'TOKYO_OPEN', 'ORB_G4', 'E2', 5, 1, 2.0, 'ONLY_RR',
                1.2, 3.5, 150, 0.35, 50.0)
    """)
    con.close()


_SEASONAL_SPEC = LiveStrategySpec(
    "TOKYO_OPEN_E2_ORB_G4",
    "core",
    "TOKYO_OPEN",
    "E2",
    "ORB_G4",
    None,
    active_months=frozenset({11, 12, 1, 2}),
)

_UNCONSTRAINED_SPEC = LiveStrategySpec(
    "TOKYO_OPEN_E2_ORB_G4",
    "core",
    "TOKYO_OPEN",
    "E2",
    "ORB_G4",
    None,
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSeasonalGate:
    """Seasonal gate: active_months + as_of_date parameter on build_live_portfolio."""

    def test_out_of_season_skips_strategy(self, live_config_db):
        """June with active_months={11,12,1,2} -> 0 strategies, SEASONAL in notes."""
        _seed_seasonal_gate_data(live_config_db)
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_SEASONAL_SPEC]),
            patch("trading_app.live_config.INSTRUMENT_ATR_GATE", {}),
        ):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db,
                instrument="MGC",
                as_of_date=date(2026, 6, 15),
            )
        assert len(portfolio.strategies) == 0
        seasonal_notes = [n for n in notes if "SEASONAL" in n]
        assert len(seasonal_notes) == 1
        assert "month 6" in seasonal_notes[0]

    def test_in_season_loads_strategy(self, live_config_db):
        """January with active_months={11,12,1,2} -> 1 strategy, weight=1.0."""
        _seed_seasonal_gate_data(live_config_db)
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_SEASONAL_SPEC]),
            patch("trading_app.live_config.INSTRUMENT_ATR_GATE", {}),
        ):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db,
                instrument="MGC",
                as_of_date=date(2026, 1, 15),
            )
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 1.0

    def test_as_of_date_defaults_to_today(self, live_config_db):
        """No active_months constraint, no as_of_date -> loads normally."""
        _seed_seasonal_gate_data(live_config_db)
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_UNCONSTRAINED_SPEC]),
            patch("trading_app.live_config.INSTRUMENT_ATR_GATE", {}),
        ):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db,
                instrument="MGC",
            )
        assert len(portfolio.strategies) == 1


# --- Weight override + recovery spec fixtures ---

_DEMOTED_SPEC = LiveStrategySpec(
    "TOKYO_OPEN_E2_ORB_G4",
    "core",
    "TOKYO_OPEN",
    "E2",
    "ORB_G4",
    None,
    weight_override=0.5,
)

_DEMOTED_WITH_RECOVERY_SPEC = LiveStrategySpec(
    "TOKYO_OPEN_E2_ORB_G4",
    "core",
    "TOKYO_OPEN",
    "E2",
    "ORB_G4",
    None,
    weight_override=0.5,
    recovery_expr_threshold=0.25,
)


def _mock_rolling_result(rolling_avg_expr: float) -> list[dict]:
    """Build a mock rolling validated result matching TOKYO_OPEN E2 ORB_G4 for MGC."""
    return [
        {
            "strategy_id": "MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4",
            "instrument": "MGC",
            "orb_label": "TOKYO_OPEN",
            "entry_model": "E2",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G4",
            "orb_minutes": 5,
            "expectancy_r": 0.35,
            "win_rate": 0.52,
            "sample_size": 150,
            "sharpe_ratio": 1.2,
            "max_drawdown_r": 3.5,
            "median_risk_points": 5.0,
            "stop_multiplier": 1.0,
            "fdr_significant": True,
            "noise_risk": False,
            "oos_exp_r": 0.35,
            "rolling_avg_expectancy_r": rolling_avg_expr,
            "rolling_weighted_stability": 0.85,
        }
    ]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestWeightOverrideAndRecovery:
    """Weight override + auto-recovery using rolling_avg_expectancy_r (family avg)."""

    def test_weight_override_applied(self):
        """Spec with weight_override=0.5 -> strategy weight is 0.5, DEMOTED in notes."""
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_DEMOTED_SPEC]),
            patch(
                "trading_app.live_config.load_rolling_validated_strategies",
                return_value=_mock_rolling_result(0.35),
            ),
        ):
            portfolio, notes = build_live_portfolio(instrument="MGC")

        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 0.5
        demoted_notes = [n for n in notes if "DEMOTED" in n]
        assert len(demoted_notes) == 1

    def test_recovery_promotes_back_using_rolling_avg(self):
        """Rolling avg ExpR=0.28 above threshold 0.25 -> weight=1.0, RECOVERED in notes."""
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_DEMOTED_WITH_RECOVERY_SPEC]),
            patch(
                "trading_app.live_config.load_rolling_validated_strategies",
                return_value=_mock_rolling_result(0.28),
            ),
        ):
            portfolio, notes = build_live_portfolio(instrument="MGC")

        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 1.0
        recovered_notes = [n for n in notes if "RECOVERED" in n]
        assert len(recovered_notes) == 1
        assert "rolling_avg_ExpR=+0.280" in recovered_notes[0]

    def test_recovery_does_not_fire_below_threshold(self):
        """Rolling avg ExpR=0.18 below threshold 0.25 -> weight stays 0.5, DEMOTED in notes."""
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_DEMOTED_WITH_RECOVERY_SPEC]),
            patch(
                "trading_app.live_config.load_rolling_validated_strategies",
                return_value=_mock_rolling_result(0.18),
            ),
        ):
            portfolio, notes = build_live_portfolio(instrument="MGC")

        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 0.5
        demoted_notes = [n for n in notes if "DEMOTED" in n]
        assert len(demoted_notes) == 1
        recovered_notes = [n for n in notes if "RECOVERED" in n]
        assert len(recovered_notes) == 0

    def test_recovery_only_from_rolling_source(self):
        """Empty rolling results -> baseline fallback, weight stays 0.5 even if baseline ExpR > threshold."""
        # Mock baseline variant with high ExpR — recovery must NOT fire because source is "baseline"
        baseline_match = {
            "strategy_id": "MGC_TOKYO_OPEN_E2_CB1_ORB_G4_RR2.0",
            "instrument": "MGC",
            "orb_label": "TOKYO_OPEN",
            "entry_model": "E2",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G4",
            "orb_minutes": 5,
            "expectancy_r": 0.40,  # above recovery threshold — but source is baseline
            "win_rate": 0.45,
            "sample_size": 200,
            "sharpe_ratio": 1.2,
            "max_drawdown_r": 3.0,
            "median_risk_points": 3.0,
            "stop_multiplier": 1.0,
            "fdr_significant": True,
            "noise_risk": False,
            "oos_exp_r": 0.40,
        }
        with (
            patch("trading_app.live_config.LIVE_PORTFOLIO", [_DEMOTED_WITH_RECOVERY_SPEC]),
            patch(
                "trading_app.live_config.load_rolling_validated_strategies",
                return_value=[],  # No rolling results — forces baseline fallback
            ),
            patch(
                "trading_app.live_config._load_best_regime_variant",
                return_value=(baseline_match, None),
            ),
        ):
            portfolio, notes = build_live_portfolio(instrument="MGC")

        # Baseline match loads, but recovery gate requires source=="rolling" — weight stays 0.5
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 0.5, "Recovery must not fire from baseline source"
        demoted_notes = [n for n in notes if "DEMOTED" in n]
        assert len(demoted_notes) == 1
        recovered_notes = [n for n in notes if "RECOVERED" in n]
        assert len(recovered_notes) == 0
