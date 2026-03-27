"""
Tests for trading_app.strategy_validator module.
"""

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import pytest

from pipeline.cost_model import get_cost_spec
from trading_app.strategy_validator import _parse_orb_size_bounds, classify_regime, run_validation, validate_strategy


def _cost():
    return get_cost_spec("MGC")


def _make_row(**overrides):
    """Build a strategy row dict with sane defaults."""
    base = {
        "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER",
        "instrument": "MGC",
        "orb_label": "CME_REOPEN",
        "orb_minutes": 5,
        "rr_target": 2.0,
        "confirm_bars": 1,
        "entry_model": "E1",
        "filter_type": "NO_FILTER",
        "filter_params": "{}",
        "sample_size": 150,
        "win_rate": 0.55,
        "avg_win_r": 1.8,
        "avg_loss_r": 1.0,
        "expectancy_r": 0.54,
        "sharpe_ratio": 0.3,
        "max_drawdown_r": 5.0,
        "median_risk_points": 10.0,
        "avg_risk_points": 10.5,
        "yearly_results": json.dumps(
            {
                "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "win_rate": 0.56, "avg_r": 0.2},
                "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "win_rate": 0.54, "avg_r": 0.16},
                "2024": {"trades": 50, "wins": 28, "total_r": 9.0, "win_rate": 0.56, "avg_r": 0.18},
            }
        ),
    }
    base.update(overrides)
    return base


class TestValidateStrategy:
    """Tests for the 6-phase validation function."""

    def test_all_phases_pass(self):
        """Strategy that passes all phases."""
        status, notes, waivers = validate_strategy(_make_row(), _cost())
        assert status == "PASSED"
        assert waivers == []

    def test_reject_low_sample(self):
        """Sample < 30 -> REJECT."""
        status, notes, _ = validate_strategy(_make_row(sample_size=20), _cost())
        assert status == "REJECTED"
        assert "Phase 1" in notes
        assert "20" in notes

    def test_warn_medium_sample(self):
        """30 <= sample < 100 -> PASS with warning."""
        status, notes, _ = validate_strategy(_make_row(sample_size=50), _cost())
        assert status == "PASSED"
        assert "WARN" in notes

    def test_reject_negative_expectancy(self):
        """ExpR <= 0 -> REJECT."""
        status, notes, _ = validate_strategy(_make_row(expectancy_r=-0.1), _cost())
        assert status == "REJECTED"
        assert "Phase 2" in notes

    def test_reject_zero_expectancy(self):
        """ExpR == 0 -> REJECT."""
        status, notes, _ = validate_strategy(_make_row(expectancy_r=0.0), _cost())
        assert status == "REJECTED"
        assert "Phase 2" in notes

    def test_e2_low_expr_reaches_phase3(self):
        """E2 strategy with low but positive ExpR reaches Phase 3 (no noise floor gate).

        Phase 2b (noise floor hard gate) removed 2026-03-21. Strategies with ExpR > 0
        now proceed to Phase 3 regardless of magnitude. Noise risk is computed
        post-validation as a flag, not a pre-validation gate.
        """
        # ExpR=0.28 previously rejected by noise floor; now reaches Phase 3+
        status, notes, _ = validate_strategy(_make_row(entry_model="E2", expectancy_r=0.28), _cost())
        # Should pass (yearly data is positive in _make_row defaults)
        assert status == "PASSED"

    def test_e1_low_expr_reaches_phase3(self):
        """E1 strategy with low but positive ExpR proceeds past removed Phase 2b."""
        status, notes, _ = validate_strategy(_make_row(entry_model="E1", expectancy_r=0.10), _cost())
        assert status == "PASSED"

    def test_reject_one_year_negative(self):
        """One year with avg_r <= 0 -> REJECT."""
        yearly = json.dumps(
            {
                "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
                "2023": {"trades": 50, "wins": 20, "total_r": -5.0, "avg_r": -0.1},
                "2024": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
            }
        )
        status, notes, _ = validate_strategy(_make_row(yearly_results=yearly), _cost())
        assert status == "REJECTED"
        assert "Phase 3" in notes
        assert "2023" in notes

    def test_reject_no_yearly_data(self):
        """No yearly data -> REJECT."""
        status, notes, _ = validate_strategy(_make_row(yearly_results="{}"), _cost())
        assert status == "REJECTED"
        assert "Phase 3" in notes

    def test_exclude_years_skips_bad_year(self):
        """Excluding a bad year lets strategy pass Phase 3."""
        yearly = json.dumps(
            {
                "2021": {"trades": 30, "wins": 10, "total_r": -5.0, "avg_r": -0.17},
                "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
                "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "avg_r": 0.16},
            }
        )
        # Fails without exclusion
        status, _, _ = validate_strategy(_make_row(yearly_results=yearly), _cost())
        assert status == "REJECTED"

        # Passes with 2021 excluded
        status, _, _ = validate_strategy(_make_row(yearly_results=yearly), _cost(), exclude_years={2021})
        assert status == "PASSED"

    def test_min_years_positive_pct(self):
        """80% threshold allows 1 bad year out of 5."""
        yearly = json.dumps(
            {
                "2021": {"trades": 30, "wins": 10, "total_r": -5.0, "avg_r": -0.17},
                "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
                "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "avg_r": 0.16},
                "2024": {"trades": 50, "wins": 28, "total_r": 9.0, "avg_r": 0.18},
                "2025": {"trades": 50, "wins": 30, "total_r": 12.0, "avg_r": 0.24},
            }
        )
        # Fails at 100%
        status, _, _ = validate_strategy(_make_row(yearly_results=yearly), _cost(), min_years_positive_pct=1.0)
        assert status == "REJECTED"

        # Passes at 80% (4/5 = 80%)
        status, _, _ = validate_strategy(_make_row(yearly_results=yearly), _cost(), min_years_positive_pct=0.8)
        assert status == "PASSED"

    def test_exclude_all_years_rejects(self):
        """Excluding all years -> REJECT."""
        yearly = json.dumps(
            {
                "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
            }
        )
        status, notes, _ = validate_strategy(_make_row(yearly_results=yearly), _cost(), exclude_years={2022})
        assert status == "REJECTED"
        assert "Phase 3" in notes

    def test_reject_stress_test(self):
        """Marginal ExpR that fails stress test -> REJECT.

        Use very small risk points (0.5) so stress friction delta_r is large
        enough to push stress_exp below 0.
        """
        status, notes, _ = validate_strategy(
            _make_row(expectancy_r=0.26, median_risk_points=0.5, avg_risk_points=0.5),
            _cost(),
        )
        assert status == "REJECTED"
        assert "Phase 4" in notes

    def test_pass_stress_test_high_exp(self):
        """High ExpR survives stress test."""
        status, notes, _ = validate_strategy(_make_row(expectancy_r=0.8), _cost())
        assert status == "PASSED"

    def test_reject_low_sharpe(self):
        """Sharpe below threshold -> REJECT when threshold set."""
        status, notes, _ = validate_strategy(_make_row(sharpe_ratio=0.1), _cost(), min_sharpe=0.5)
        assert status == "REJECTED"
        assert "Phase 5" in notes

    def test_reject_high_drawdown(self):
        """Drawdown above threshold -> REJECT when threshold set."""
        status, notes, _ = validate_strategy(_make_row(max_drawdown_r=15.0), _cost(), max_drawdown=10.0)
        assert status == "REJECTED"
        assert "Phase 6" in notes

    def test_stress_test_uses_outcome_risk(self):
        """Stress test uses median_risk_points when available.

        MGC min_risk_floor_dollars=10.0 floors the denominator, so
        stress delta is ~0.287R. Need ExpR > 0.287 to pass stress.
        """
        # With large risk, stress delta is small -> passes
        status, _, _ = validate_strategy(_make_row(expectancy_r=0.30, median_risk_points=20.0), _cost())
        assert status == "PASSED"

        # ExpR 0.26 is below stress delta (0.287) -> rejects at Phase 4
        status, notes, _ = validate_strategy(_make_row(expectancy_r=0.26, median_risk_points=0.5), _cost())
        assert status == "REJECTED"
        assert "Phase 4" in notes

    def test_stress_test_falls_back_to_avg_risk(self):
        """Stress test uses avg_risk_points when median is None."""
        status, _, _ = validate_strategy(
            _make_row(expectancy_r=0.30, median_risk_points=None, avg_risk_points=20.0), _cost()
        )
        assert status == "PASSED"

    def test_stress_test_falls_back_to_tick_floor(self):
        """Stress test uses tick-based floor when both risk stats are None.

        tick floor = 10 * 0.10 = 1.0 point, risk $ = 10.0
        stress delta = (5.74 * 0.5) / 10.0 = 0.287R -> needs ExpR > 0.287
        """
        status, _, _ = validate_strategy(
            _make_row(expectancy_r=0.50, median_risk_points=None, avg_risk_points=None), _cost()
        )
        assert status == "PASSED"

        status, notes, _ = validate_strategy(
            _make_row(expectancy_r=0.26, median_risk_points=None, avg_risk_points=None), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 4" in notes

    def test_optional_phases_skipped_by_default(self):
        """Without min_sharpe/max_drawdown, phases 5/6 are skipped."""
        status, notes, _ = validate_strategy(_make_row(sharpe_ratio=0.01, max_drawdown_r=50.0), _cost())
        assert status == "PASSED"

    # --- Phase 4c/4d: DSR and FST — REMOVED (2026-03-18 adversarial review) ---
    # Both were "logged, not rejected" (fake gates). DSR/FST data columns remain
    # in experimental_strategies but are no longer checked during validation.
    # Multiple testing is now handled by FDR hard gate in Phase C.

    def test_dsr_fields_ignored(self):
        """DSR fields in row dict have no effect on validation (removed gate)."""
        status, _, _ = validate_strategy(_make_row(sharpe_haircut=-0.5), _cost())
        assert status == "PASSED"

    def test_fst_fields_ignored(self):
        """FST fields in row dict have no effect on validation (removed gate)."""
        status, _, _ = validate_strategy(_make_row(sharpe_ratio=0.05, fst_hurdle=0.15), _cost())
        assert status == "PASSED"

    def test_validation_notes_contain_reason(self):
        """Rejection notes explain which phase failed."""
        status, notes, _ = validate_strategy(_make_row(sample_size=10), _cost())
        assert "Phase 1" in notes
        assert "10" in notes


def _yearly(years_data: dict) -> str:
    """Build yearly_results JSON from {year: (trades, avg_r)} dict."""
    return json.dumps(
        {
            str(y): {"trades": t, "wins": max(1, int(t * 0.5)), "total_r": t * avg_r, "win_rate": 0.5, "avg_r": avg_r}
            for y, (t, avg_r) in years_data.items()
        }
    )


class TestBenjaminiHochberg:
    """Tests for BH FDR correction — the sole multiple-testing gate."""

    def test_all_significant(self):
        """All p-values well below threshold pass FDR."""
        from trading_app.strategy_validator import benjamini_hochberg

        p_values = [("s1", 0.001), ("s2", 0.002), ("s3", 0.003)]
        results = benjamini_hochberg(p_values, alpha=0.05)
        assert all(r["fdr_significant"] for r in results.values())

    def test_all_insignificant(self):
        """All p-values above threshold fail FDR."""
        from trading_app.strategy_validator import benjamini_hochberg

        p_values = [("s1", 0.8), ("s2", 0.9), ("s3", 0.95)]
        results = benjamini_hochberg(p_values, alpha=0.05)
        assert not any(r["fdr_significant"] for r in results.values())

    def test_mixed_significance(self):
        """BH step-up correctly separates signal from noise."""
        from trading_app.strategy_validator import benjamini_hochberg

        # 3 real signals + 7 noise = K=10
        p_values = [
            ("real1", 0.001),
            ("real2", 0.003),
            ("real3", 0.005),
            ("noise1", 0.10),
            ("noise2", 0.20),
            ("noise3", 0.30),
            ("noise4", 0.50),
            ("noise5", 0.60),
            ("noise6", 0.80),
            ("noise7", 0.90),
        ]
        results = benjamini_hochberg(p_values, alpha=0.05)
        # Real signals should pass, noise should fail
        for sid in ["real1", "real2", "real3"]:
            assert results[sid]["fdr_significant"], f"{sid} should be significant"
        for sid in ["noise1", "noise2", "noise3", "noise4"]:
            assert not results[sid]["fdr_significant"], f"{sid} should not be significant"

    def test_empty_input(self):
        """Empty list returns empty dict."""
        from trading_app.strategy_validator import benjamini_hochberg

        assert benjamini_hochberg([], alpha=0.05) == {}

    def test_global_k_stricter_than_subset(self):
        """A p-value that passes with K=10 may fail with K=1000 (more tests = higher bar)."""
        from trading_app.strategy_validator import benjamini_hochberg

        # Borderline p-value
        small_set = [("target", 0.04)] + [(f"n{i}", 0.5 + i * 0.01) for i in range(9)]
        large_set = [("target", 0.04)] + [(f"n{i}", 0.5 + i * 0.0001) for i in range(999)]

        result_small = benjamini_hochberg(small_set, alpha=0.05)
        result_large = benjamini_hochberg(large_set, alpha=0.05)

        # With K=10, rank 1 threshold = 0.05 * 1/10 = 0.005 — target at 0.04 fails both
        # Actually BH threshold at rank 1 of 10 = 0.005, at rank 1 of 1000 = 0.00005
        # But target is at rank 1 (smallest), so threshold scales with 1/K
        # This test verifies the monotonicity property
        assert result_small["target"]["adjusted_p"] <= result_large["target"]["adjusted_p"]

    def test_total_tests_stricter_than_default(self):
        """Passing total_tests > len(valid) makes BH stricter (higher adjusted p)."""
        from trading_app.strategy_validator import benjamini_hochberg

        pvals = [("a", 0.01), ("b", 0.03), ("c", 0.05)]
        result_default = benjamini_hochberg(pvals, alpha=0.05)
        result_global = benjamini_hochberg(pvals, alpha=0.05, total_tests=1000)

        # global K=1000 > n=3 -> adjusted_p must be higher (stricter)
        assert result_global["a"]["adjusted_p"] > result_default["a"]["adjusted_p"]

    def test_total_tests_equal_to_n_matches_default(self):
        """total_tests == len(valid) should match default behavior exactly."""
        from trading_app.strategy_validator import benjamini_hochberg

        pvals = [("a", 0.01), ("b", 0.03), ("c", 0.05)]
        result_default = benjamini_hochberg(pvals, alpha=0.05)
        result_explicit = benjamini_hochberg(pvals, alpha=0.05, total_tests=3)

        for sid in ["a", "b", "c"]:
            assert result_default[sid]["adjusted_p"] == result_explicit[sid]["adjusted_p"]

    def test_total_tests_less_than_n_raises(self):
        """total_tests < len(valid) violates BH assumptions -> must raise."""
        import pytest

        from trading_app.strategy_validator import benjamini_hochberg

        pvals = [("a", 0.01), ("b", 0.03), ("c", 0.05)]
        with pytest.raises(ValueError, match="BH requires m >= n"):
            benjamini_hochberg(pvals, alpha=0.05, total_tests=2)


class TestRegimeWaivers:
    """Tests for DORMANT regime waiver logic in Phase 3."""

    def test_regime_classify_dormant(self):
        """ATR < 20 -> DORMANT."""
        assert classify_regime(15.0) == "DORMANT"
        assert classify_regime(0.0) == "DORMANT"
        assert classify_regime(19.9) == "DORMANT"

    def test_regime_classify_marginal(self):
        """20 <= ATR < 30 -> MARGINAL."""
        assert classify_regime(20.0) == "MARGINAL"
        assert classify_regime(25.0) == "MARGINAL"
        assert classify_regime(29.9) == "MARGINAL"

    def test_regime_classify_active(self):
        """ATR >= 30 -> ACTIVE."""
        assert classify_regime(30.0) == "ACTIVE"
        assert classify_regime(35.0) == "ACTIVE"
        assert classify_regime(110.0) == "ACTIVE"

    def test_dormant_year_waived(self):
        """Negative DORMANT year with <= 5 trades is waived."""
        yearly = _yearly({2022: (50, 0.2), 2023: (50, 0.1), 2017: (2, -0.02)})
        atr = {2022: 25.0, 2023: 28.0, 2017: 12.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "PASSED"
        assert waivers == [2017]
        assert "DORMANT" in notes

    def test_marginal_year_not_waived(self):
        """Negative MARGINAL year is NOT waived."""
        yearly = _yearly({2022: (50, 0.2), 2023: (50, 0.1), 2021: (2, -0.02)})
        atr = {2022: 25.0, 2023: 28.0, 2021: 25.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes
        assert waivers == []

    def test_dormant_high_trades_not_waived(self):
        """DORMANT year with > 5 trades is NOT waived."""
        yearly = _yearly({2022: (50, 0.2), 2023: (50, 0.1), 2017: (8, -0.1)})
        atr = {2022: 25.0, 2023: 28.0, 2017: 15.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes
        assert waivers == []

    def test_all_years_waived_fails(self):
        """All years requiring waiver -> REJECTED (no clean positive year)."""
        yearly = _yearly({2017: (2, -0.02), 2018: (1, -0.05)})
        atr = {2017: 12.0, 2018: 11.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "REJECTED"
        assert "clean positive year" in notes
        assert waivers == []

    def test_no_regime_waivers_flag(self):
        """enable_regime_waivers=False uses strict logic (rejects)."""
        yearly = _yearly({2022: (50, 0.2), 2023: (50, 0.1), 2017: (2, -0.02)})
        atr = {2022: 25.0, 2023: 28.0, 2017: 12.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=False,
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes
        assert waivers == []

    def test_waiver_metadata_recorded(self):
        """Returned waivers list and notes contain waiver details."""
        yearly = _yearly({2022: (50, 0.2), 2023: (50, 0.1), 2017: (2, -0.02), 2018: (3, -0.01)})
        atr = {2022: 25.0, 2023: 28.0, 2017: 12.0, 2018: 14.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "PASSED"
        assert waivers == [2017, 2018]
        assert "Year 2017 waived" in notes
        assert "Year 2018 waived" in notes
        assert "mean_atr=12.0" in notes
        assert "mean_atr=14.0" in notes

    def test_mixed_years_some_waived_passes_at_threshold(self):
        """2 positive + 1 waived + 1 unwaived neg = 3/4 (75%) -> PASSED.

        FIX (2026-03-26): Previously ANY unwaived negative year caused immediate
        rejection. Now waived years count as passing and the 75% threshold applies.
        """
        yearly = _yearly(
            {
                2022: (50, 0.2),
                2023: (50, 0.1),
                2017: (2, -0.02),  # DORMANT, waivable
                2020: (30, -0.05),  # ACTIVE, not waivable — but within 75% tolerance
            }
        )
        atr = {2022: 25.0, 2023: 28.0, 2017: 12.0, 2020: 30.6}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "PASSED"
        assert 2017 in waivers  # DORMANT year waived
        assert "2020" in notes  # unwaived negative noted

    def test_mixed_years_too_many_unwaived_fails(self):
        """2 positive + 1 waived + 2 unwaived neg = 3/5 (60%) -> REJECTED."""
        yearly = _yearly(
            {
                2022: (50, 0.2),
                2023: (50, 0.1),
                2017: (2, -0.02),  # DORMANT, waivable
                2020: (30, -0.05),  # ACTIVE, not waivable
                2021: (40, -0.03),  # ACTIVE, not waivable
            }
        )
        atr = {2022: 25.0, 2023: 28.0, 2017: 12.0, 2020: 30.6, 2021: 29.0}
        status, notes, waivers = validate_strategy(
            _make_row(yearly_results=yearly),
            _cost(),
            atr_by_year=atr,
            enable_regime_waivers=True,
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes
        assert "60%" in notes or "3/5" in notes
        assert waivers == []


class TestRunValidation:
    """Integration tests with temp DB."""

    def _setup_db(self, tmp_path, strategies):
        """Create temp DB with schema + strategies."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()

        init_trading_app_schema = __import__(
            "trading_app.db_manager", fromlist=["init_trading_app_schema"]
        ).init_trading_app_schema
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))
        for s in strategies:
            cols = list(s.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_str = ", ".join(cols)
            con.execute(
                f"INSERT INTO experimental_strategies ({col_str}) VALUES ({placeholders})",
                list(s.values()),
            )
        con.commit()
        con.close()
        return db_path

    def test_promotes_passing_strategy(self, tmp_path):
        """Passing strategy appears in validated_setups."""
        db_path = self._setup_db(tmp_path, [_make_row()])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC", enable_walkforward=False)
        assert passed == 1
        assert rejected == 0

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
        con.close()
        assert count == 1

    def test_rejected_not_in_validated(self, tmp_path):
        """Rejected strategy does NOT appear in validated_setups."""
        db_path = self._setup_db(tmp_path, [_make_row(sample_size=10)])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC")
        assert passed == 0
        assert rejected == 1

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
        con.close()
        assert count == 0

    def test_already_validated_not_reprocessed(self, tmp_path):
        """Strategy with existing validation_status is skipped."""
        row = _make_row(validation_status="PASSED", validation_notes="Already done")
        db_path = self._setup_db(tmp_path, [row])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC")
        assert passed == 0
        assert rejected == 0

    def test_entry_model_in_validated_setups(self, tmp_path):
        """entry_model column is populated in validated_setups."""
        db_path = self._setup_db(tmp_path, [_make_row()])
        run_validation(db_path=db_path, instrument="MGC", enable_walkforward=False)

        con = duckdb.connect(str(db_path), read_only=True)
        em = con.execute("SELECT entry_model FROM validated_setups").fetchone()[0]
        con.close()
        assert em == "E1"


class TestParseOrbSizeBounds:
    """Tests for _parse_orb_size_bounds pure function."""

    def test_json_params_both(self):
        assert _parse_orb_size_bounds("ORB_G5", '{"min_size": 5, "max_size": 12}') == (5.0, 12.0)

    def test_json_params_min_only(self):
        assert _parse_orb_size_bounds("ORB_G5", '{"min_size": 5}') == (5.0, None)

    def test_json_params_zero_min(self):
        """Zero is a valid bound, not None (the bug we fixed)."""
        assert _parse_orb_size_bounds("ORB_G0", '{"min_size": 0, "max_size": 10}') == (0.0, 10.0)

    def test_fallback_g_filter(self):
        assert _parse_orb_size_bounds("ORB_G5", None) == (5.0, None)

    def test_fallback_gl_filter(self):
        assert _parse_orb_size_bounds("ORB_G4_L12", None) == (4.0, 12.0)

    def test_no_filter(self):
        assert _parse_orb_size_bounds("NONE", None) == (None, None)

    def test_none_filter_type(self):
        assert _parse_orb_size_bounds(None, None) == (None, None)


class TestComputeDstSplit:
    """Tests for compute_dst_split."""

    def test_clean_session_returns_clean(self):
        """Non-DST-affected sessions return verdict='CLEAN' without querying."""
        from trading_app.strategy_validator import compute_dst_split

        result = compute_dst_split(None, "test", "MGC", "TOKYO_OPEN", "E1", 2.0, 1, "NONE")
        assert result["verdict"] == "CLEAN"
        assert result["winter_n"] is None


class TestCLI:
    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["strategy_validator", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.strategy_validator import main

            main()
        assert exc_info.value.code == 0
        assert "instrument" in capsys.readouterr().out
