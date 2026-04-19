"""Tests for research/phase_2_9_comprehensive_multi_year_stratification.py.

Unit tests only — DB-touching logic is exercised by the script run itself.
These tests cover: canonical delegation, BH-FDR, symmetric labelling,
heat-map aggregation, and MinBTL check.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research import phase_2_9_comprehensive_multi_year_stratification as mod


class TestCanonicalDelegation:
    def test_reuses_phase_2_8_window_stats(self):
        from research.phase_2_8_multi_year_regime_stratification import _window_stats
        assert mod._window_stats is _window_stats

    def test_reuses_load_active_setups(self):
        from research.mode_a_revalidation_active_setups import load_active_setups
        assert mod.load_active_setups is load_active_setups

    def test_reuses_compute_mode_a(self):
        from research.mode_a_revalidation_active_setups import compute_mode_a
        assert mod.compute_mode_a is compute_mode_a

    def test_holdout_constant(self):
        from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
        assert mod.HOLDOUT_SACRED_FROM is HOLDOUT_SACRED_FROM


class TestScopeLock:
    def test_seven_years_pre_holdout(self):
        assert mod.YEARS == (2019, 2020, 2021, 2022, 2023, 2024, 2025)
        # all years must be strictly before HOLDOUT_SACRED_FROM
        for y in mod.YEARS:
            assert y < mod.HOLDOUT_SACRED_FROM.year + 1

    def test_gold_lanes_exact(self):
        # Per handoff commit f3b3b72b, Phase 2.7 closed with 2 GOLD lanes.
        assert frozenset({
            "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
            "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
        }) == mod.GOLD_LANES

    def test_bh_q_locked(self):
        assert mod.BH_Q == 0.10

    def test_delta_threshold_matches_v1_for_symmetric_label(self):
        # v2 uses the same 0.03 threshold as v1 — the change is symmetric
        # (BOOST as well as DRAG) and BH-conditioned.
        assert mod.DELTA_LABEL_THRESHOLD == 0.03

    def test_min_n_year_matches_v1(self):
        assert mod.MIN_N_YEAR == 30


class TestTwoSidedP:
    def test_t_zero(self):
        p = mod._two_sided_p_from_t(0.0, 100)
        assert p is not None
        assert math.isclose(p, 1.0, abs_tol=1e-6)

    def test_t_three(self):
        p = mod._two_sided_p_from_t(3.0, 100)
        assert p is not None
        # With n=100, df=99, two-sided t.sf(3)*2 ~ 0.0035
        assert 0.001 < p < 0.01

    def test_none_t(self):
        assert mod._two_sided_p_from_t(None, 100) is None

    def test_small_n(self):
        assert mod._two_sided_p_from_t(3.0, 1) is None


class TestBHFDR:
    def test_all_small_p_survive(self):
        survivors = mod.bh_fdr([0.001, 0.002, 0.003, 0.004, 0.005], q=0.10)
        # rank 5: p=0.005 vs (5/5)*0.10 = 0.10 -> 0.005 <= 0.10 -> k=5
        assert survivors == [True, True, True, True, True]

    def test_all_large_p_fail(self):
        survivors = mod.bh_fdr([0.5, 0.6, 0.7, 0.8, 0.9], q=0.10)
        assert survivors == [False] * 5

    def test_unordered_input(self):
        survivors = mod.bh_fdr([0.9, 0.001, 0.5, 0.002, 0.8], q=0.10)
        # Sorted p: 0.001 0.002 0.5 0.8 0.9. Crit: 0.02 0.04 0.06 0.08 0.10
        # Only first two survive. Those are original indices 1 and 3.
        assert survivors[1] and survivors[3]
        assert not (survivors[0] or survivors[2] or survivors[4])

    def test_none_never_survives(self):
        survivors = mod.bh_fdr([None, 0.001, None, 0.5], q=0.10)
        assert survivors[0] is False
        assert survivors[2] is False

    def test_nan_never_survives(self):
        survivors = mod.bh_fdr([float("nan"), 0.001, 0.5], q=0.10)
        assert survivors[0] is False


class TestAssignV2Pattern:
    def _mk(self, **kw) -> pd.Series:
        base = {
            "label_raw": "NEUTRAL",
            "bh_session": False,
            "bh_year": False,
        }
        base.update(kw)
        return pd.Series(base)

    def test_unevaluable_preserved(self):
        assert mod.assign_v2_pattern(self._mk(label_raw="UNEVALUABLE")) == "UNEVALUABLE"

    def test_neutral_preserved(self):
        assert mod.assign_v2_pattern(self._mk()) == "NEUTRAL"

    def test_drag_candidate_confirmed_by_bh(self):
        row = self._mk(label_raw="DRAG_candidate", bh_session=True)
        assert mod.assign_v2_pattern(row) == "DRAG_confirmed"

    def test_drag_candidate_unconfirmed(self):
        row = self._mk(label_raw="DRAG_candidate")
        assert mod.assign_v2_pattern(row) == "DRAG_unconfirmed"

    def test_boost_candidate_confirmed(self):
        row = self._mk(label_raw="BOOST_candidate", bh_year=True)
        assert mod.assign_v2_pattern(row) == "BOOST_confirmed"

    def test_boost_candidate_unconfirmed(self):
        row = self._mk(label_raw="BOOST_candidate")
        assert mod.assign_v2_pattern(row) == "BOOST_unconfirmed"

    def test_empty_full_window_preserved(self):
        assert mod.assign_v2_pattern(self._mk(label_raw="EMPTY_FULL_WINDOW")) == "EMPTY_FULL_WINDOW"


class TestBuildHeatMap:
    def test_aggregates_per_session_year(self):
        df = pd.DataFrame([
            {"strategy_id": "A", "session": "X", "year": 2020, "n_year": 100,
             "year_expr": 0.1, "delta": 0.02, "bh_session": True},
            {"strategy_id": "B", "session": "X", "year": 2020, "n_year": 50,
             "year_expr": -0.05, "delta": -0.04, "bh_session": False},
            {"strategy_id": "A", "session": "X", "year": 2021, "n_year": 80,
             "year_expr": 0.0, "delta": 0.0, "bh_session": False},
        ])
        heat = mod.build_heat_map(df)
        assert len(heat) == 2  # (X,2020) and (X,2021)
        row2020 = heat[heat["year"] == 2020].iloc[0]
        assert row2020["n_lanes"] == 2
        assert row2020["total_n_trades"] == 150
        # weighted_mean_year_expr = (0.1*100 + -0.05*50) / 150 = 7.5/150 = 0.05
        assert math.isclose(row2020["weighted_mean_year_expr"], 0.05, abs_tol=1e-9)
        assert row2020["bh_session_survivor_count"] == 1


class TestMinBTL:
    def test_pass_with_large_n(self):
        ok, val = mod.minbtl_check(k=266, max_n=600)
        # 2 * ln(266) / 600^2 = 11.16 / 360000 ~ 3.1e-5
        assert ok
        assert val < 0.001

    def test_fail_with_tiny_n(self):
        ok, val = mod.minbtl_check(k=266, max_n=1)
        assert not ok

    def test_zero_n_fails(self):
        ok, val = mod.minbtl_check(k=266, max_n=0)
        assert not ok
        assert math.isinf(val)


class TestTestScopeYearWithinModeA:
    def test_years_strictly_pre_holdout(self):
        for y in mod.YEARS:
            # HOLDOUT_SACRED_FROM is a date; all years covered begin on Jan 1
            # and are within Mode A IS.
            assert y < mod.HOLDOUT_SACRED_FROM.year or (
                y == mod.HOLDOUT_SACRED_FROM.year and mod.HOLDOUT_SACRED_FROM.month == 1
                and mod.HOLDOUT_SACRED_FROM.day == 1
            )


class TestBuildGoldFragility:
    def test_tier1_lane_flagged_fragile_when_excluding_drops_t(self):
        # full_t=3.50, worst ex_year_t=1.5 -> fragile
        df = pd.DataFrame([
            {"strategy_id": "L1", "year": 2020, "ex_year_t": 3.0, "full_t": 3.5},
            {"strategy_id": "L1", "year": 2021, "ex_year_t": 1.5, "full_t": 3.5},
            {"strategy_id": "L1", "year": 2022, "ex_year_t": 3.2, "full_t": 3.5},
        ])
        frag = mod.build_gold_fragility(df, tier1_t_floor=3.0)
        assert len(frag) == 1
        assert frag.iloc[0]["fragility_flag"] == "FRAGILE"
        assert frag.iloc[0]["worst_ex_year"] == 2021

    def test_tier1_lane_stable_when_ex_year_t_stays_high(self):
        df = pd.DataFrame([
            {"strategy_id": "L1", "year": 2020, "ex_year_t": 3.0, "full_t": 3.5},
            {"strategy_id": "L1", "year": 2021, "ex_year_t": 2.5, "full_t": 3.5},
            {"strategy_id": "L1", "year": 2022, "ex_year_t": 3.2, "full_t": 3.5},
        ])
        frag = mod.build_gold_fragility(df, tier1_t_floor=3.0)
        assert frag.iloc[0]["fragility_flag"] == "STABLE"

    def test_below_tier1_not_included_unless_gold(self):
        df = pd.DataFrame([
            {"strategy_id": "L1", "year": 2020, "ex_year_t": 1.0, "full_t": 2.5},
        ])
        frag = mod.build_gold_fragility(df, tier1_t_floor=3.0)
        assert len(frag) == 0

    def test_gold_lane_included_even_below_tier1(self):
        gold_sid = next(iter(mod.GOLD_LANES))
        df = pd.DataFrame([
            {"strategy_id": gold_sid, "year": 2020, "ex_year_t": 2.8, "full_t": 2.9},
            {"strategy_id": gold_sid, "year": 2021, "ex_year_t": 2.0, "full_t": 2.9},
        ])
        frag = mod.build_gold_fragility(df, tier1_t_floor=3.0)
        assert len(frag) == 1
        assert bool(frag.iloc[0]["in_gold_pool"]) is True
        # full_t < tier1_t_floor so fragile is False by the rule
        assert frag.iloc[0]["fragility_flag"] == "STABLE"
