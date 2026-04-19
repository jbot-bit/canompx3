"""Tests for research/phase_2_5_portfolio_subset_t_sweep.py.

Covers:
- Canonical delegation guards (no re-declared research stats)
- subset_t_stat math (correct Chordia t formula, NaN on edge cases)
- classify_lane flag logic (Rule 8.3 narrow positive-lift interpretation,
  EXTREME_FIRE_{HIGH,LOW}, ZERO_FIRE, FILTER_NOT_ACTIVE, SUBSET_T_BELOW_CHORDIA,
  FILTER_REMOVES_EDGE, N_BELOW_DEPLOYABLE, SUBSET_T_UNEVALUABLE, PASS)
- SQL-injection guard (_VALID_SESSION_NAMES whitelist)
- Smoke test: real DB end-to-end produces CSV with 38 rows
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research import phase_2_5_portfolio_subset_t_sweep as mod


class TestCanonicalDelegation:
    def test_chordia_threshold_from_mode_a_revalidation(self):
        from research.mode_a_revalidation_active_setups import C4_T_WITH_THEORY
        assert mod.SUBSET_T_THRESHOLD is C4_T_WITH_THEORY
        assert mod.SUBSET_T_THRESHOLD == 3.00

    def test_no_inlined_chordia_value(self):
        src = (PROJECT_ROOT / "research" / "phase_2_5_portfolio_subset_t_sweep.py").read_text()
        # Must reference C4_T_WITH_THEORY, not bare 3.00
        assert "C4_T_WITH_THEORY" in src
        assert "SUBSET_T_THRESHOLD: float = C4_T_WITH_THEORY" in src

    def test_imports_canonical_sources(self):
        src = (PROJECT_ROOT / "research" / "phase_2_5_portfolio_subset_t_sweep.py").read_text()
        assert "from trading_app.holdout_policy import HOLDOUT_SACRED_FROM" in src
        assert "from pipeline.paths import GOLD_DB_PATH" in src
        assert "from pipeline.dst import SESSION_CATALOG" in src
        assert "from research.mode_a_revalidation_active_setups import" in src
        assert "compute_mode_a" in src  # delegated

    def test_no_inlined_holdout_date(self):
        src = (PROJECT_ROOT / "research" / "phase_2_5_portfolio_subset_t_sweep.py").read_text()
        assert "'2026-01-01'" not in src
        assert 'date(2026, 1, 1)' not in src

    def test_valid_session_names_sourced_from_catalog(self):
        """The SQL-injection whitelist must match the canonical catalog."""
        from pipeline.dst import SESSION_CATALOG
        expected = frozenset(SESSION_CATALOG.keys())
        assert mod._VALID_SESSION_NAMES == expected


class TestSubsetTStat:
    def test_standard_case(self):
        # ExpR=0.1, sd=1.0, n=100 → t = 0.1 / (1/10) = 1.0
        assert math.isclose(mod.subset_t_stat(0.1, 1.0, 100), 1.0, abs_tol=1e-9)

    def test_positive_t_on_large_n(self):
        # ExpR=0.2, sd=1.0, n=400 → t = 0.2 / (1/20) = 4.0 (above Chordia)
        assert math.isclose(mod.subset_t_stat(0.2, 1.0, 400), 4.0, abs_tol=1e-9)

    def test_none_expr_returns_nan(self):
        assert math.isnan(mod.subset_t_stat(None, 1.0, 100))

    def test_none_sd_returns_nan(self):
        assert math.isnan(mod.subset_t_stat(0.1, None, 100))

    def test_zero_sd_returns_nan(self):
        assert math.isnan(mod.subset_t_stat(0.1, 0.0, 100))

    def test_n_below_2_returns_nan(self):
        assert math.isnan(mod.subset_t_stat(0.1, 1.0, 1))


class TestClassifyLane:
    """Flag logic — exhaustive case coverage."""

    def test_pass_clean_lane(self):
        flags, primary = mod.classify_lane(
            n_on=300, n_universe=600, subset_expr=0.15, unfiltered_expr=0.05,
            subset_t=3.5, fire_rate=0.5,
        )
        assert flags == []
        assert primary == "PASS"

    def test_arithmetic_lift_narrow_positive_only(self):
        """Rule 8.3: positive lift + subset_t<3 + filter reduces sample."""
        flags, primary = mod.classify_lane(
            n_on=112, n_universe=850, subset_expr=0.046, unfiltered_expr=-0.157,
            subset_t=0.43, fire_rate=0.13,
        )
        # lift = 0.203 (>0.10), subset_t=0.43 (<3.0), n_on<n_universe
        assert "ARITHMETIC_LIFT" in flags
        assert primary == "ARITHMETIC_LIFT"

    def test_negative_lift_does_not_trigger_arithmetic_lift(self):
        """Rule 8.3 is positive-lift-only per addendum text; negative → FILTER_REMOVES_EDGE."""
        flags, primary = mod.classify_lane(
            n_on=200, n_universe=600, subset_expr=-0.05, unfiltered_expr=0.10,
            subset_t=-1.5, fire_rate=0.33,
        )
        # lift = -0.15 (< -0.10) → FILTER_REMOVES_EDGE, NOT ARITHMETIC_LIFT
        assert "ARITHMETIC_LIFT" not in flags
        assert "FILTER_REMOVES_EDGE" in flags

    def test_extreme_fire_high(self):
        flags, _ = mod.classify_lane(
            n_on=950, n_universe=1000, subset_expr=0.05, unfiltered_expr=0.04,
            subset_t=2.0, fire_rate=0.95,
        )
        assert "EXTREME_FIRE_HIGH" in flags

    def test_extreme_fire_low(self):
        flags, _ = mod.classify_lane(
            n_on=30, n_universe=1000, subset_expr=0.20, unfiltered_expr=0.01,
            subset_t=2.0, fire_rate=0.03,
        )
        assert "EXTREME_FIRE_LOW" in flags

    def test_filter_not_active_when_n_on_equals_n_universe(self):
        flags, primary = mod.classify_lane(
            n_on=500, n_universe=500, subset_expr=0.1, unfiltered_expr=0.1,
            subset_t=2.5, fire_rate=1.0,
        )
        assert "FILTER_NOT_ACTIVE" in flags
        # ARITHMETIC_LIFT is explicitly guarded against n_on < n_universe
        assert "ARITHMETIC_LIFT" not in flags

    def test_zero_fire(self):
        flags, primary = mod.classify_lane(
            n_on=0, n_universe=500, subset_expr=None, unfiltered_expr=0.1,
            subset_t=float("nan"), fire_rate=0.0,
        )
        assert "ZERO_FIRE" in flags
        assert primary == "ZERO_FIRE"

    def test_subset_t_below_chordia_on_deployable_n(self):
        flags, primary = mod.classify_lane(
            n_on=500, n_universe=1000, subset_expr=0.05, unfiltered_expr=0.04,
            subset_t=2.0, fire_rate=0.5,
        )
        # lift = 0.01 (<0.10) so no ARITHMETIC_LIFT; N>=100 + t<3 triggers
        assert "SUBSET_T_BELOW_CHORDIA" in flags
        assert primary == "SUBSET_T_BELOW_CHORDIA"

    def test_n_below_deployable(self):
        flags, _ = mod.classify_lane(
            n_on=50, n_universe=500, subset_expr=0.15, unfiltered_expr=0.05,
            subset_t=2.0, fire_rate=0.1,
        )
        # N_on=50 < 100, so N_BELOW_DEPLOYABLE fires, not SUBSET_T_BELOW_CHORDIA
        assert "N_BELOW_DEPLOYABLE" in flags
        assert "SUBSET_T_BELOW_CHORDIA" not in flags

    def test_subset_t_unevaluable_when_t_nan_but_n_ok(self):
        flags, _ = mod.classify_lane(
            n_on=5, n_universe=500, subset_expr=0.0, unfiltered_expr=0.0,
            subset_t=float("nan"), fire_rate=0.01,
        )
        # n_on<100 so N_BELOW_DEPLOYABLE; also UNEVALUABLE for t
        assert "SUBSET_T_UNEVALUABLE" in flags

    def test_primary_flag_ranking_arithmetic_lift_wins(self):
        """When multiple flags fire, ARITHMETIC_LIFT takes priority."""
        flags, primary = mod.classify_lane(
            n_on=200, n_universe=600, subset_expr=0.20, unfiltered_expr=0.05,
            subset_t=1.5, fire_rate=0.33,
        )
        # Triggers both ARITHMETIC_LIFT and SUBSET_T_BELOW_CHORDIA
        assert "ARITHMETIC_LIFT" in flags
        assert "SUBSET_T_BELOW_CHORDIA" in flags
        assert primary == "ARITHMETIC_LIFT"


class TestSqlInjectionGuard:
    def test_invalid_session_name_raises(self):
        """fetch_break_universe must reject non-catalog session names."""
        import duckdb
        con = duckdb.connect(":memory:")
        spec = {
            "instrument": "MNQ",
            "orb_label": "NOT_A_REAL_SESSION",
            "orb_minutes": 5,
            "entry_model": "E2",
            "confirm_bars": 1,
            "rr_target": 1.5,
            "execution_spec": '{"direction":"long"}',
        }
        with pytest.raises(ValueError, match="SESSION_CATALOG"):
            mod.fetch_break_universe(con, spec)
        con.close()

    def test_sql_injection_attempt_blocked(self):
        """Direct SQL injection attempt in orb_label is refused, not executed."""
        import duckdb
        con = duckdb.connect(":memory:")
        evil = "EUROPE_FLOW'; DROP TABLE orb_outcomes; --"
        spec = {
            "instrument": "MNQ",
            "orb_label": evil,
            "orb_minutes": 5,
            "entry_model": "E2",
            "confirm_bars": 1,
            "rr_target": 1.5,
            "execution_spec": '{"direction":"long"}',
        }
        with pytest.raises(ValueError):
            mod.fetch_break_universe(con, spec)
        con.close()


class TestThresholdConstants:
    def test_lift_threshold_matches_rule_8_3(self):
        """Rule 8.3 addendum text says '> 0.10 R'."""
        assert mod.LIFT_THRESHOLD == 0.10

    def test_backtesting_methodology_rule_8_3_cited(self):
        src = (PROJECT_ROOT / "research" / "phase_2_5_portfolio_subset_t_sweep.py").read_text()
        assert "RULE 8.3" in src or "Rule 8.3" in src
        assert "backtesting-methodology.md" in src


@pytest.mark.integration
class TestEndToEnd:
    def test_main_runs_and_covers_38_lanes(self, tmp_path, monkeypatch, capsys):
        if not Path(mod.GOLD_DB_PATH).exists():
            pytest.skip("gold.db not available")
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        rc = mod.main()
        assert rc == 0
        csv = tmp_path / "phase_2_5_portfolio_subset_t_sweep.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        # Currently 38 active validated_setups — this may drift as lanes retire/add.
        # Assert shape sanity (>=10 lanes, <=200 lanes) rather than exact 38.
        assert 10 <= len(df) <= 200
        required_cols = {
            "strategy_id", "instrument", "n_universe", "n_on", "fire_rate",
            "expr_unfiltered", "expr_on", "lift", "subset_t", "flags", "primary_flag",
        }
        assert required_cols.issubset(df.columns)
        captured = capsys.readouterr()
        assert "PORTFOLIO SUBSET-T" in captured.out
