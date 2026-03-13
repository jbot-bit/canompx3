"""Tests for trading_app.live_config — live portfolio configuration."""

from pathlib import Path

import duckdb
import pytest

from trading_app.live_config import (
    LIVE_MIN_EXPECTANCY_DOLLARS_MULT,
    LIVE_PORTFOLIO,
    LiveStrategySpec,
    _check_dollar_gate,
    _load_best_experimental_variant,
    _load_best_regime_variant,
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
            fdr_significant BOOLEAN
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

        result = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result is not None
        assert result["strategy_id"] == "MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G4"
        assert result["sharpe_ratio"] == 1.2

    def test_not_found(self, live_config_db):
        """No matching strategy returns None."""
        result = _load_best_regime_variant(live_config_db, "MGC", "9999", "E1", "ORB_G4")
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

        result = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
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

        result = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
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

        result = _load_best_regime_variant(live_config_db, "MGC", "NYSE_OPEN", "E2", "ORB_G8")
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

        result = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
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

        result = _load_best_regime_variant(live_config_db, "MGC", "TOKYO_OPEN", "E1", "ORB_G4")
        assert result is not None
        assert result["strategy_id"] == "rr15"
        assert result["rr_target"] == 1.5


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
