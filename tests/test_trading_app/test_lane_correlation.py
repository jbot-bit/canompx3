"""Tests for the trade-level correlation pre-deploy gate."""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import patch

import pytest

from trading_app.lane_correlation import (
    CorrelationReport,
    PairResult,
    RHO_REJECT_THRESHOLD,
    SUBSET_REJECT_THRESHOLD,
    _daily_pnl,
    _pearson,
    check_candidate_correlation,
)


def _outcome(day: date, pnl_r: float) -> dict:
    return {"trading_day": day, "pnl_r": pnl_r, "outcome": "win" if pnl_r > 0 else "loss"}


def _outcomes_from_list(pnls: list[float], start_ordinal: int = 738000) -> list[dict]:
    return [_outcome(date.fromordinal(start_ordinal + i), p) for i, p in enumerate(pnls)]


class TestDailyPnl:
    def test_groups_by_day(self):
        outcomes = [
            _outcome(date(2025, 1, 1), 0.5),
            _outcome(date(2025, 1, 1), 0.3),
            _outcome(date(2025, 1, 2), -1.0),
        ]
        d = _daily_pnl(outcomes)
        assert len(d) == 2
        assert abs(d[date(2025, 1, 1)] - 0.8) < 1e-9
        assert abs(d[date(2025, 1, 2)] - (-1.0)) < 1e-9

    def test_skips_none(self):
        outcomes = [{"trading_day": date(2025, 1, 1), "pnl_r": None}]
        assert _daily_pnl(outcomes) == {}


class TestPearson:
    def test_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_pearson(xs, xs) - 1.0) < 1e-9

    def test_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [-x for x in xs]
        assert abs(_pearson(xs, ys) - (-1.0)) < 1e-9

    def test_zero_variance(self):
        xs = [1.0, 1.0, 1.0, 1.0, 1.0]
        ys = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _pearson(xs, ys) == 0.0

    def test_too_few(self):
        assert _pearson([1.0, 2.0], [3.0, 4.0]) == 0.0

    def test_uncorrelated(self):
        xs = [1.0, -1.0, 1.0, -1.0, 1.0]
        ys = [1.0, 1.0, -1.0, -1.0, 1.0]
        rho = _pearson(xs, ys)
        assert abs(rho) < 0.5


class TestCheckCandidateCorrelation:
    """Integration tests using mocked canonical loader."""

    def _make_lane(self, sid: str, orb_label: str, filter_type: str, rr: float = 1.5) -> dict:
        return {
            "strategy_id": sid,
            "instrument": "MNQ",
            "orb_label": orb_label,
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": rr,
            "confirm_bars": 1,
            "filter_type": filter_type,
        }

    def _patch_and_run(self, candidate_outcomes, deployed_outcomes_map, candidate_lane, deployed_lanes):
        def mock_load(con, *, instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type):
            key = f"{instrument}_{orb_label}_{entry_model}_RR{rr_target}_CB{confirm_bars}_{filter_type}"
            if key in deployed_outcomes_map:
                return deployed_outcomes_map[key]
            return candidate_outcomes

        with (
            patch("trading_app.lane_correlation._load_strategy_outcomes", side_effect=mock_load),
            patch("trading_app.lane_correlation.get_profile_lane_definitions", return_value=deployed_lanes),
            patch("trading_app.lane_correlation.duckdb") as mock_db,
        ):
            mock_con = mock_db.connect.return_value
            return check_candidate_correlation(candidate_lane, "test_profile", con=mock_con)

    def test_strict_subset_rejected(self):
        base = _outcomes_from_list([0.5, -1.0, 0.8, -1.0, 0.3, 0.7, -1.0, 0.6, 0.4, -1.0])
        subset = base[1:9]  # 8 of 10 — strict subset

        deployed_lane = self._make_lane("DEPLOYED_G5", "EUROPE_FLOW", "ORB_G5")
        candidate_lane = self._make_lane("CANDIDATE_COST", "EUROPE_FLOW", "COST_LT12")
        key = "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5"

        report = self._patch_and_run(subset, {key: base}, candidate_lane, [deployed_lane])
        assert not report.gate_pass
        assert report.worst_rho > RHO_REJECT_THRESHOLD
        assert len(report.reject_reasons) == 1

    def test_orthogonal_session_passes(self):
        cand = _outcomes_from_list([0.5, -1.0, 0.3, -1.0, 0.8], start_ordinal=738000)
        dep = _outcomes_from_list([0.2, -1.0, 0.6, -1.0, 0.1], start_ordinal=738100)

        deployed_lane = self._make_lane("DEPLOYED_NYSE", "NYSE_OPEN", "ORB_G5")
        candidate_lane = self._make_lane("CANDIDATE_COMEX", "COMEX_SETTLE", "OVNRNG_100")
        key = "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5"

        report = self._patch_and_run(cand, {key: dep}, candidate_lane, [deployed_lane])
        assert report.gate_pass
        assert report.worst_rho == 0.0
        assert report.worst_subset == 0.0

    def test_same_session_different_filter_partial_overlap_passes(self):
        cand = _outcomes_from_list([0.5, -1.0, 0.3, -1.0, 0.8, 0.2, -1.0, 0.4, -1.0, 0.6], start_ordinal=738010)
        dep = _outcomes_from_list([-1.0, 0.5, -1.0, 0.3, -1.0, 0.8, 0.2, 0.1, -1.0, 0.9], start_ordinal=738014)

        deployed_lane = self._make_lane("DEPLOYED", "COMEX_SETTLE", "OVNRNG_100")
        candidate_lane = self._make_lane("CANDIDATE", "COMEX_SETTLE", "X_MES_ATR60")
        key = "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100"

        report = self._patch_and_run(cand, {key: dep}, candidate_lane, [deployed_lane])
        assert report.gate_pass
        assert report.worst_subset <= SUBSET_REJECT_THRESHOLD

    def test_high_subset_coverage_rejected(self):
        base = _outcomes_from_list([0.5, -1.0, 0.3, -1.0, 0.8, 0.2, -1.0, 0.6, 0.4, -1.0])
        subset = base[:9]

        deployed_lane = self._make_lane("DEPLOYED", "EUROPE_FLOW", "ORB_G5")
        candidate_lane = self._make_lane("CANDIDATE", "EUROPE_FLOW", "COST_LT12")
        key = "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5"

        report = self._patch_and_run(subset, {key: base}, candidate_lane, [deployed_lane])
        assert not report.gate_pass
        assert report.worst_subset > SUBSET_REJECT_THRESHOLD

    def test_empty_candidate_passes(self):
        dep = _outcomes_from_list([0.5, -1.0, 0.3])
        deployed_lane = self._make_lane("DEPLOYED", "NYSE_OPEN", "ORB_G5")
        candidate_lane = self._make_lane("CANDIDATE", "COMEX_SETTLE", "COST_LT12")
        key = "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5"

        report = self._patch_and_run([], {key: dep}, candidate_lane, [deployed_lane])
        assert report.gate_pass

    def test_multiple_deployed_worst_wins(self):
        cand = _outcomes_from_list([0.5, -1.0, 0.3, -1.0, 0.8, 0.2, -1.0])
        dep_ok = _outcomes_from_list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], start_ordinal=738100)
        dep_bad = list(cand)  # identical

        d1 = self._make_lane("DEPLOYED_OK", "NYSE_OPEN", "ORB_G5")
        d2 = self._make_lane("DEPLOYED_BAD", "EUROPE_FLOW", "ORB_G5")
        candidate_lane = self._make_lane("CANDIDATE", "EUROPE_FLOW", "COST_LT12")

        def mock_load(con, *, instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type):
            if orb_label == "NYSE_OPEN":
                return dep_ok
            if filter_type == "ORB_G5" and orb_label == "EUROPE_FLOW":
                return dep_bad
            return cand

        with (
            patch("trading_app.lane_correlation._load_strategy_outcomes", side_effect=mock_load),
            patch("trading_app.lane_correlation.get_profile_lane_definitions", return_value=[d1, d2]),
            patch("trading_app.lane_correlation.duckdb") as mock_db,
        ):
            mock_con = mock_db.connect.return_value
            report = check_candidate_correlation(candidate_lane, "test", con=mock_con)

        assert not report.gate_pass
        assert len(report.reject_reasons) >= 1
        assert any("DEPLOYED_BAD" in r for r in report.reject_reasons)

    def test_report_structure(self):
        cand = _outcomes_from_list([0.5, -1.0])
        dep = _outcomes_from_list([0.3, -1.0], start_ordinal=738100)

        deployed_lane = self._make_lane("DEP", "NYSE_OPEN", "ORB_G5")
        candidate_lane = self._make_lane("CAND", "COMEX_SETTLE", "COST_LT12")
        key = "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5"

        report = self._patch_and_run(cand, {key: dep}, candidate_lane, [deployed_lane])
        assert isinstance(report, CorrelationReport)
        assert isinstance(report.pairs, tuple)
        assert all(isinstance(p, PairResult) for p in report.pairs)
        assert isinstance(report.reject_reasons, tuple)
