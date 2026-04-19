"""Tests for research/phase_2_4_mes_composite_c1_c12_audit.py.

Covers:
- Canonical delegation (imports from mode_a_revalidation, data_era, chordia-compatible thresholds)
- t_stat math (correct formula, NaN on low N)
- walk_forward_efficiency (WFE calculation, NaN on sparse folds)
- year_break (year grouping)
- composite_fire (AND logic)
- C8 power-tier classification per backtesting-methodology.md RULE 3.2
- Threshold constants correctly sourced (no re-declared research stats)
- Smoke test that main() runs end-to-end and produces CSV + correct VERDICT shape
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research import phase_2_4_mes_composite_c1_c12_audit as mod


class TestCanonicalDelegation:
    """No re-declared research stats — all from canonical sources."""

    def test_c4_threshold_delegated_to_mode_a_revalidation(self):
        """C4_T_WITH_THEORY must be the same object as mode_a_revalidation's canonical."""
        from research.mode_a_revalidation_active_setups import C4_T_WITH_THEORY
        assert mod.C4_T_WITH_THEORY is C4_T_WITH_THEORY
        assert mod.C4_T_WITH_THEORY == 3.00

    def test_c7_threshold_delegated(self):
        from research.mode_a_revalidation_active_setups import C7_MIN_N
        assert mod.C7_MIN_N is C7_MIN_N
        assert mod.C7_MIN_N == 100

    def test_c9_thresholds_delegated(self):
        from research.mode_a_revalidation_active_setups import (
            C9_ERA_THRESHOLD,
            C9_MIN_N_PER_ERA,
        )
        assert mod.C9_ERA_THRESHOLD is C9_ERA_THRESHOLD
        assert mod.C9_MIN_N_PER_ERA is C9_MIN_N_PER_ERA
        assert mod.C9_ERA_THRESHOLD == -0.05
        assert mod.C9_MIN_N_PER_ERA == 50

    def test_micro_launch_delegated_to_data_era(self):
        """C10 MICRO-launch comes from pipeline.data_era, not inlined."""
        from pipeline.data_era import micro_launch_day
        assert callable(micro_launch_day)
        # MES and MNQ both launched 2019-05-06 per data_era
        assert micro_launch_day("MES").isoformat() == "2019-05-06"
        assert micro_launch_day("MNQ").isoformat() == "2019-05-06"

    def test_no_inlined_date_constants(self):
        """Source must NOT contain a literal date(2019, 5, 6) constant anywhere."""
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        # Delegated call is allowed; standalone literal is banned
        assert "date(2019, 5, 6)" not in src
        # Must import from canonical
        assert "from pipeline.data_era import micro_launch_day" in src

    def test_holdout_and_gold_db_delegated(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        assert "from trading_app.holdout_policy import HOLDOUT_SACRED_FROM" in src
        assert "from pipeline.paths import GOLD_DB_PATH" in src
        assert "'2026-01-01'" not in src
        assert '"gold.db"' not in src

    def test_filter_signal_delegated(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        assert "from research.filter_utils import filter_signal" in src

    def test_c6_wfe_has_research_source_annotation(self):
        """C6 WFE lacks a shared canonical constant — must carry @research-source."""
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        # Must cite pre_registered_criteria.md § Criterion 6 somewhere near the constant
        assert "@research-source" in src
        assert "pre_registered_criteria.md" in src
        assert "Criterion 6" in src

    def test_c6_threshold_value(self):
        assert mod.C6_WFE_THRESHOLD == 0.50


class TestTStat:
    def test_positive_t_on_positive_expr(self):
        # ExpR=0.5, sd=1.0, n=100 → t = 0.5 / (1/10) = 5.0
        assert math.isclose(mod.t_stat(0.5, 1.0, 100), 5.0, abs_tol=1e-9)

    def test_zero_sd_returns_nan(self):
        assert math.isnan(mod.t_stat(0.1, 0.0, 100))

    def test_n_below_2_returns_nan(self):
        assert math.isnan(mod.t_stat(0.5, 1.0, 1))
        assert math.isnan(mod.t_stat(0.5, 1.0, 0))

    def test_negative_expr(self):
        # ExpR=-0.2, sd=1.0, n=25 → t = -0.2 / (1/5) = -1.0
        assert math.isclose(mod.t_stat(-0.2, 1.0, 25), -1.0, abs_tol=1e-9)


class TestCompositeFire:
    def test_and_logic(self):
        df = pd.DataFrame({
            "trading_day": pd.date_range("2020-01-01", periods=5),
            "orb_EUROPE_FLOW_size": [10.0, 3.0, 10.0, 4.0, 10.0],  # G5 fires on rows 0,2,4
            "orb_EUROPE_FLOW_break_dir": ["long"] * 5,
            "orb_SINGAPORE_OPEN_break_dir": ["long", "long", "short", "long", "long"],
            # CrossSessionMomentumFilter fires when prior_session direction == current
            # We will not evaluate this via real filter_signal in unit tests — just assert
            # that composite_fire is `g5 & sgp` via monkeypatch below
        })

        # Monkeypatch filter_signal to return known masks
        g5_mask = [True, False, True, False, True]
        sgp_mask = [True, True, True, True, False]

        calls = []

        def fake_filter_signal(d, key, sess):
            calls.append((key, sess))
            if key == "ORB_G5":
                return np.array(g5_mask)
            if key == "CROSS_SGP_MOMENTUM":
                return np.array(sgp_mask)
            raise ValueError(f"unexpected filter {key}")

        orig = mod.filter_signal
        try:
            mod.filter_signal = fake_filter_signal
            out = mod.composite_fire(df)
        finally:
            mod.filter_signal = orig

        # Expected AND: rows 0 and 2
        np.testing.assert_array_equal(out, np.array([True, False, True, False, False]))
        # Verify both canonical filters queried
        assert ("ORB_G5", "EUROPE_FLOW") in calls
        assert ("CROSS_SGP_MOMENTUM", "EUROPE_FLOW") in calls


class TestWalkForwardEfficiency:
    def _df(self, year_counts: dict[int, int]) -> pd.DataFrame:
        rows = []
        for y, n in year_counts.items():
            for _ in range(n):
                rows.append({"trading_day": pd.Timestamp(f"{y}-06-01"), "pnl_r": 0.1})
        return pd.DataFrame(rows)

    def test_insufficient_years_returns_nan(self):
        df = self._df({2020: 50, 2021: 50})  # only 2 years → <3 years gate returns NaN
        wfe, folds = mod.walk_forward_efficiency(df, np.ones(100, dtype=bool))
        assert math.isnan(wfe)
        # folds may contain 1 entry (2021 as test, 2020 as train) but the top-level
        # guard returns early when len(years) < 3

    def test_zero_sharpe_train_returns_nan(self):
        """When training years have mean=0 pnl, train sharpe = 0 → WFE NaN."""
        # Zero-mean pnl by symmetric sign alternation ensures exact float-0 mean
        rows = []
        for y in [2020, 2021, 2022]:
            for i in range(50):
                rows.append({"trading_day": pd.Timestamp(f"{y}-06-01"),
                             "pnl_r": 1.0 if i % 2 == 0 else -1.0})
        df = pd.DataFrame(rows)
        fire = np.ones(len(df), dtype=bool)
        wfe, _ = mod.walk_forward_efficiency(df, fire)
        # Mean is zero → sharpe = 0 — but the divisor (sd>0) is also present,
        # so fold sharpes are 0 exactly → mean_train_sh = 0 → WFE = NaN
        assert math.isnan(wfe)

    def test_folds_structure_present(self):
        # Variable pnl so sharpe > 0
        rows = []
        rng = np.random.default_rng(42)
        for y in [2020, 2021, 2022, 2023]:
            for pnl in rng.normal(0.05, 1.0, 50):
                rows.append({"trading_day": pd.Timestamp(f"{y}-06-01"), "pnl_r": float(pnl)})
        df = pd.DataFrame(rows)
        fire = np.ones(len(df), dtype=bool)
        wfe, folds = mod.walk_forward_efficiency(df, fire)
        assert len(folds) == 3  # 2021, 2022, 2023 each used as test
        for f in folds:
            assert "test_year" in f
            assert "train_sharpe" in f
            assert "test_sharpe" in f


class TestYearBreak:
    def test_year_grouping(self):
        df = pd.DataFrame({
            "trading_day": pd.to_datetime([
                "2020-01-01", "2020-06-01",
                "2021-03-01", "2021-04-01", "2021-09-01",
                "2022-07-01",
            ]),
            "pnl_r": [0.1, 0.3, -0.2, 0.5, 0.1, 0.0],
        })
        fire = np.array([True, True, False, True, True, True])
        out = mod.year_break(df, fire)
        years = {y["year"]: y for y in out}
        assert years[2020]["n"] == 2
        assert math.isclose(years[2020]["expr"], 0.20, abs_tol=1e-9)
        assert years[2021]["n"] == 2  # rows 3 and 4 (row 2 not fired)
        assert math.isclose(years[2021]["expr"], 0.30, abs_tol=1e-9)
        assert years[2022]["n"] == 1
        assert math.isclose(years[2022]["expr"], 0.0, abs_tol=1e-9)


class TestC8PowerTier:
    """C8 classification per backtesting-methodology.md RULE 3.2."""

    def test_reporting_path_sets_power_tier_in_source(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        # All three tiers must be named in source
        for tier in ("UNEVALUABLE", "DIRECTIONAL", "CONFIRMATORY"):
            assert tier in src, f"C8 tier {tier} must be named in source"
        # Tier is wired into criteria_rows output
        assert "power_tier=" in src

    def test_reference_to_rule_3_2_cited(self):
        """C8 power-tier logic must cite the rule in code comments."""
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        assert "RULE 3.2" in src


class TestLockedSurface:
    def test_surface_matches_pre_reg(self):
        assert mod.INSTRUMENT == "MES"
        assert mod.SESSION == "EUROPE_FLOW"
        assert mod.ORB_MINUTES == 5
        assert mod.ENTRY_MODEL == "E2"
        assert mod.CONFIRM_BARS == 1
        assert mod.RR_TARGET == 1.5
        assert mod.DIRECTION == "long"


class TestT0Threshold:
    def test_t0_has_research_source(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_mes_composite_c1_c12_audit.py").read_text()
        # T0 constant is inlined (no shared canonical) — must cite rule
        assert "backtesting-methodology.md" in src
        assert "RULE 7" in src
        assert mod.T0_TAUT_THRESHOLD == 0.90


@pytest.mark.integration
class TestEndToEnd:
    """Smoke test: real DB, CSV produced, VERDICT string present."""

    def test_main_runs_and_produces_csv(self, tmp_path, monkeypatch, capsys):
        if not Path(mod.GOLD_DB_PATH).exists():
            pytest.skip("gold.db not available")
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        rc = mod.main()
        assert rc == 0
        out_csv = tmp_path / "phase_2_4_mes_composite_audit.csv"
        assert out_csv.exists()
        result = pd.read_csv(out_csv)
        # 13 criteria rows (C1, C2, C3, C4, C6, C7, C8, C9, C10, T0_g5, T0_sgp, T5, T8)
        assert len(result) == 13
        # Every row has boolean pass column
        assert set(result["pass"].unique()).issubset({True, False})
        # Captured stdout has VERDICT
        captured = capsys.readouterr()
        assert "VERDICT:" in captured.out
