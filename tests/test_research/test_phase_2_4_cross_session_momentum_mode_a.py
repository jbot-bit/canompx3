"""Tests for research/phase_2_4_cross_session_momentum_mode_a.py.

Verifies:
- Canonical source delegation (HOLDOUT_SACRED_FROM, filter_signal, compute_mode_a, GOLD_DB_PATH)
- overlap_stats math on synthetic fire masks
- score_options output shape + gate-blocking behavior at rho >= 0.70
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

from research import phase_2_4_cross_session_momentum_mode_a as mod


class TestCanonicalDelegation:
    def test_holdout_imported_not_inlined(self):
        from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
        src = (PROJECT_ROOT / "research" / "phase_2_4_cross_session_momentum_mode_a.py").read_text()
        assert "from trading_app.holdout_policy import HOLDOUT_SACRED_FROM" in src
        assert "date(2026, 1, 1)" not in src
        assert "datetime(2026, 1, 1" not in src
        assert "'2026-01-01'" not in src
        assert str(HOLDOUT_SACRED_FROM) == "2026-01-01"

    def test_gold_db_path_imported_not_hardcoded(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_cross_session_momentum_mode_a.py").read_text()
        assert "from pipeline.paths import GOLD_DB_PATH" in src
        assert "'gold.db'" not in src
        assert '"gold.db"' not in src
        assert "C:/db/gold.db" not in src
        assert "C:\\db\\gold.db" not in src

    def test_filter_signal_delegated(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_cross_session_momentum_mode_a.py").read_text()
        assert "from research.filter_utils import filter_signal" in src
        # Script should NOT re-implement ORB_G5 or CROSS_SGP_MOMENTUM logic inline.
        assert "orb_EUROPE_FLOW_size >= 5" not in src
        assert "prior_session=" not in src  # no inline filter construction

    def test_compute_mode_a_delegated(self):
        src = (PROJECT_ROOT / "research" / "phase_2_4_cross_session_momentum_mode_a.py").read_text()
        assert "from research.mode_a_revalidation_active_setups import compute_mode_a" in src

    def test_locked_surface_matches_pre_reg(self):
        assert mod.INSTRUMENT == "MNQ"
        assert mod.SESSION == "EUROPE_FLOW"
        assert mod.ORB_MINUTES == 5
        assert mod.ENTRY_MODEL == "E2"
        assert mod.CONFIRM_BARS == 1
        assert mod.RR_TARGETS == (1.0, 1.5, 2.0)
        # Pre-reg direction is long
        assert '"long"' in mod.EXECUTION_SPEC


class TestOverlapStats:
    """overlap_stats must compute fire cells, rho_mask, rho_canonical, composite."""

    @staticmethod
    def _build(g5: list[bool], sgp: list[bool], pnl: list[float]) -> dict:
        g5_s = pd.Series(g5)
        sgp_s = pd.Series(sgp)
        pnl_arr = np.array(pnl, dtype=float)
        pnl_g5 = pd.Series(np.where(np.asarray(g5), pnl_arr, np.nan))
        pnl_sgp = pd.Series(np.where(np.asarray(sgp), pnl_arr, np.nan))
        return mod.overlap_stats(g5_s, sgp_s, pnl_g5, pnl_sgp)

    def test_identical_fires_give_canonical_rho_one(self):
        # Both filters fire identically on a 5-day window → intersection pnl
        # sequence is identical → rho_canonical == 1.0 by construction.
        out = self._build(
            g5=[True, True, True, True, True, False],
            sgp=[True, True, True, True, True, False],
            pnl=[1.0, -0.5, 0.8, -1.0, 0.3, 0.0],
        )
        assert out["n_both"] == 5
        assert out["n_g5_only"] == 0
        assert out["n_sgp_only"] == 0
        assert out["n_neither"] == 1
        assert math.isclose(out["rho_canonical"], 1.0, abs_tol=1e-9)

    def test_disjoint_fires_give_nan_canonical_rho(self):
        # Zero intersection → rho_canonical NaN (<5 shared days).
        out = self._build(
            g5=[True, True, True, True, True, False],
            sgp=[False, False, False, False, False, True],
            pnl=[1.0, -1.0, 0.5, -0.5, 0.2, 0.3],
        )
        assert out["n_both"] == 0
        assert out["n_g5_only"] == 5
        assert out["n_sgp_only"] == 1
        assert math.isnan(out["rho_canonical"])

    def test_composite_expr_on_intersection_only(self):
        out = self._build(
            g5=[True, True, True, True, False, False],
            sgp=[True, True, False, False, True, True],
            pnl=[2.0, 0.0, 1.0, -1.0, 0.5, -0.5],
        )
        # Intersection is days 0 and 1: pnl 2.0 and 0.0 → mean = 1.0
        assert out["n_both"] == 2
        assert math.isclose(out["composite_expr"], 1.0, abs_tol=1e-9)
        assert out["composite_n"] == 2

    def test_fire_mask_rho_vs_canonical_rho_are_distinct(self):
        # Both measures tracked separately. Intersection must be >=5 for
        # canonical rho (matches lane_correlation._pearson n<5 → 0.0 guard).
        out = self._build(
            g5=[True, True, True, True, True, True, False, False],
            sgp=[True, True, True, True, True, False, True, False],
            pnl=[1.0, -0.5, 0.8, 0.3, -0.2, 2.0, -1.0, 0.1],
        )
        assert "rho_mask" in out
        assert "rho_canonical" in out
        # 5 intersection days with identical pnl → rho_canonical = 1.0
        assert out["n_both"] == 5
        assert math.isclose(out["rho_canonical"], 1.0, abs_tol=1e-9)

    def test_break_day_universe_count(self):
        out = self._build(
            g5=[True, False, True, False, True, False],
            sgp=[False, True, True, False, False, True],
            pnl=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        assert out["n_break_days"] == 6
        assert (
            out["n_both"] + out["n_g5_only"] + out["n_sgp_only"] + out["n_neither"]
            == 6
        )


class TestScoreOptions:
    def test_option_d_blocked_when_canonical_rho_ge_threshold(self):
        """Gate-block flag fires when rho_canonical >= 0.70."""
        g5_by_rr = {rr: mod.LaneModeA(strategy_id=f"G5_{rr}", filter_type="ORB_G5",
                                      rr_target=rr, n=100, expr=0.05, sharpe_ann=0.4, wr=0.55, sd=0.9)
                    for rr in mod.RR_TARGETS}
        sgp_by_rr = {rr: mod.LaneModeA(strategy_id=f"SGP_{rr}", filter_type="CROSS_SGP_MOMENTUM",
                                       rr_target=rr, n=60, expr=0.08, sharpe_ann=0.5, wr=0.58, sd=0.9)
                     for rr in mod.RR_TARGETS}
        overlap_by_rr = {rr: {
            "n_break_days": 200, "n_both": 50, "n_g5_only": 50, "n_sgp_only": 10, "n_neither": 90,
            "rho_canonical": 0.95, "rho_mask": -0.02,
            "composite_n": 50, "composite_expr": 0.10,
        } for rr in mod.RR_TARGETS}

        df = mod.score_options(g5_by_rr, sgp_by_rr, overlap_by_rr)
        assert len(df) == 3
        assert all(df["D_gate_blocked"] == True)  # noqa: E712
        assert all(df["D_rho_canonical"] == 0.95)

    def test_option_d_not_blocked_when_rho_low(self):
        g5_by_rr = {1.5: mod.LaneModeA(strategy_id="G5", filter_type="ORB_G5",
                                       rr_target=1.5, n=100, expr=0.05, sharpe_ann=0.4, wr=0.55, sd=0.9)}
        sgp_by_rr = {1.5: mod.LaneModeA(strategy_id="SGP", filter_type="CROSS_SGP_MOMENTUM",
                                        rr_target=1.5, n=60, expr=0.08, sharpe_ann=0.5, wr=0.58, sd=0.9)}
        overlap_by_rr = {1.5: {
            "n_break_days": 200, "n_both": 20, "n_g5_only": 80, "n_sgp_only": 40, "n_neither": 60,
            "rho_canonical": 0.10, "rho_mask": -0.02,
            "composite_n": 20, "composite_expr": 0.15,
        }}
        # Filter lanes dict to match
        g5_by_rr_full = {rr: g5_by_rr[1.5] if rr == 1.5 else mod.LaneModeA(
            strategy_id=f"G5_{rr}", filter_type="ORB_G5", rr_target=rr,
            n=100, expr=0.05, sharpe_ann=0.4, wr=0.55, sd=0.9)
            for rr in mod.RR_TARGETS}
        sgp_by_rr_full = {rr: sgp_by_rr[1.5] if rr == 1.5 else mod.LaneModeA(
            strategy_id=f"SGP_{rr}", filter_type="CROSS_SGP_MOMENTUM", rr_target=rr,
            n=60, expr=0.08, sharpe_ann=0.5, wr=0.58, sd=0.9)
            for rr in mod.RR_TARGETS}
        overlap_by_rr_full = {rr: overlap_by_rr[1.5] if rr == 1.5 else {
            "n_break_days": 200, "n_both": 20, "n_g5_only": 80, "n_sgp_only": 40, "n_neither": 60,
            "rho_canonical": 0.10, "rho_mask": -0.02,
            "composite_n": 20, "composite_expr": 0.15,
        } for rr in mod.RR_TARGETS}

        df = mod.score_options(g5_by_rr_full, sgp_by_rr_full, overlap_by_rr_full)
        assert all(df["D_gate_blocked"] == False)  # noqa: E712

    def test_output_has_abcd_columns(self):
        g5 = {rr: mod.LaneModeA(strategy_id=f"G{rr}", filter_type="ORB_G5", rr_target=rr,
                                n=100, expr=0.05, sharpe_ann=0.4, wr=0.5, sd=1.0)
              for rr in mod.RR_TARGETS}
        sgp = {rr: mod.LaneModeA(strategy_id=f"S{rr}", filter_type="CROSS_SGP_MOMENTUM", rr_target=rr,
                                 n=60, expr=0.08, sharpe_ann=0.5, wr=0.5, sd=1.0)
               for rr in mod.RR_TARGETS}
        ov = {rr: {"n_break_days": 200, "n_both": 50, "n_g5_only": 50,
                   "n_sgp_only": 10, "n_neither": 90,
                   "rho_canonical": 0.5, "rho_mask": 0.0,
                   "composite_n": 50, "composite_expr": 0.10}
              for rr in mod.RR_TARGETS}
        df = mod.score_options(g5, sgp, ov)
        needed = {"A_strategy", "A_N", "A_ExpR", "A_R_per_yr",
                  "B_strategy", "B_N", "B_ExpR", "B_R_per_yr",
                  "C_N", "C_ExpR", "C_R_per_yr",
                  "D_rho_canonical", "D_rho_mask", "D_gate_blocked",
                  "B_minus_A_ExpR", "B_minus_A_R_per_yr",
                  "overlap_n_both", "overlap_n_g5_only", "overlap_n_sgp_only"}
        assert needed.issubset(df.columns)


class TestSpecFor:
    def test_spec_is_mode_a_revalidation_compatible(self):
        spec = mod.spec_for("ORB_G5", 1.5)
        # compute_mode_a consumes these keys
        needed = {"instrument", "orb_label", "orb_minutes", "entry_model",
                  "confirm_bars", "rr_target", "filter_type", "execution_spec"}
        assert needed.issubset(spec.keys())
        assert spec["filter_type"] == "ORB_G5"
        assert spec["rr_target"] == 1.5
        assert spec["orb_label"] == "EUROPE_FLOW"

    def test_spec_strategy_id_format(self):
        spec = mod.spec_for("CROSS_SGP_MOMENTUM", 2.0)
        assert spec["strategy_id"] == "MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM"


class TestLaneModeA:
    def test_as_row_counts_positive_years(self):
        lane = mod.LaneModeA(
            strategy_id="test", filter_type="ORB_G5", rr_target=1.5,
            n=100, expr=0.05, sharpe_ann=0.4, wr=0.5, sd=1.0,
            year_break={
                2020: {"n": 20, "expr": 0.1, "positive": True},
                2021: {"n": 30, "expr": -0.05, "positive": False},
                2022: {"n": 25, "expr": 0.08, "positive": True},
            },
        )
        row = lane.as_row()
        assert row["years_positive"] == 2
        assert row["years_total"] == 3

    def test_as_row_empty_year_break(self):
        lane = mod.LaneModeA(strategy_id="x", filter_type="y", rr_target=1.0)
        row = lane.as_row()
        assert row["years_positive"] == 0
        assert row["years_total"] == 0


@pytest.mark.integration
class TestEndToEndAgainstDB:
    """Integration: script runs against live gold.db and returns expected shape.

    Skipped when DB absent. Not a numerical assertion — shape only — to avoid
    baking specific Mode A numbers that drift with append.
    """

    def test_main_produces_csvs(self, tmp_path, monkeypatch):
        from pipeline.paths import GOLD_DB_PATH
        if not Path(GOLD_DB_PATH).exists():
            pytest.skip("gold.db not available")
        # Redirect OUTPUT_DIR to tmp so the real CSVs aren't touched
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        rc = mod.main()
        assert rc == 0
        assert (tmp_path / "phase_2_4_cross_session_momentum_mode_a_lanes.csv").exists()
        assert (tmp_path / "phase_2_4_cross_session_momentum_mode_a_options.csv").exists()

        lanes = pd.read_csv(tmp_path / "phase_2_4_cross_session_momentum_mode_a_lanes.csv")
        assert len(lanes) == 6  # 3 RRs × 2 filters
        assert set(lanes["filter_type"].unique()) == {"ORB_G5", "CROSS_SGP_MOMENTUM"}

        options = pd.read_csv(tmp_path / "phase_2_4_cross_session_momentum_mode_a_options.csv")
        assert len(options) == 3  # one row per RR
        # D gate must be blocked across all 3 RRs on live data (rho=1.0 by construction)
        assert all(options["D_gate_blocked"] == True)  # noqa: E712
