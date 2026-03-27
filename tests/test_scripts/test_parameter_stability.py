"""Tests for parameter stability heatmap generator pure functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.reports.parameter_stability_heatmap import (
    _cell_color,
    _stability_score,
    build_heatmap_data,
)


class TestCellColor:
    """Color assignment based on stability vs baseline."""

    def test_stable_cell(self):
        assert _cell_color(0.10, 0.09, 100) == "green"

    def test_ok_cell(self):
        assert _cell_color(0.10, 0.06, 100) == "yellow"

    def test_weak_cell(self):
        assert _cell_color(0.10, 0.03, 100) == "orange"

    def test_unstable_cell(self):
        assert _cell_color(0.10, 0.01, 100) == "red"

    def test_sign_flip(self):
        assert _cell_color(0.10, -0.05, 100) == "red"

    def test_insufficient_n(self):
        assert _cell_color(0.10, 0.09, 15) == "gray"

    def test_none_expr(self):
        assert _cell_color(0.10, None, 100) == "gray"

    def test_negative_baseline_positive_cell(self):
        assert _cell_color(-0.05, 0.02, 100) == "green"

    def test_negative_baseline_negative_cell(self):
        assert _cell_color(-0.05, -0.03, 100) == "red"


class TestStabilityScore:
    """Overall stability verdict from neighbors."""

    def _make_neighbor(self, expr, n=100):
        return {"ExpR": expr, "N": n}

    def test_stable(self):
        baseline = 0.10
        neighbors = [self._make_neighbor(0.09), self._make_neighbor(0.11)]
        assert _stability_score(baseline, neighbors) == "STABLE"

    def test_ok(self):
        baseline = 0.10
        neighbors = [self._make_neighbor(0.06), self._make_neighbor(0.09)]
        assert _stability_score(baseline, neighbors) == "OK"

    def test_isolated_peak(self):
        baseline = 0.10
        neighbors = [self._make_neighbor(0.01), self._make_neighbor(0.02)]
        assert _stability_score(baseline, neighbors) == "ISOLATED_PEAK"

    def test_no_neighbors(self):
        assert _stability_score(0.10, []) == "NO_NEIGHBORS"

    def test_insufficient_data(self):
        neighbors = [self._make_neighbor(0.09, n=5)]
        assert _stability_score(0.10, neighbors) == "INSUFFICIENT_DATA"

    def test_negative_baseline(self):
        neighbors = [self._make_neighbor(0.02)]
        assert _stability_score(-0.05, neighbors) == "NEGATIVE_BASELINE"


class TestBuildHeatmapData:
    """Heatmap matrix construction from grid data."""

    def _make_grid_cell(self, rr, cb, om, expr=0.10, n=100):
        return {
            "rr": rr,
            "cb": cb,
            "om": om,
            "ExpR": expr,
            "WR": 0.45,
            "N": n,
            "Sharpe": 0.5,
            "SharpeH": 0.3,
            "strategy_id": f"TEST_RR{rr}_CB{cb}_O{om}",
        }

    def test_e2_grid_rr_x_om_matrix(self):
        """E2 always CB1 -> should produce RR×aperture matrix."""
        grid = [
            self._make_grid_cell(1.0, 1, 5, 0.08),
            self._make_grid_cell(1.5, 1, 5, 0.10),
            self._make_grid_cell(2.0, 1, 5, 0.09),
            self._make_grid_cell(1.5, 1, 15, 0.07),
            self._make_grid_cell(1.5, 1, 30, 0.06),
        ]
        result = build_heatmap_data(grid, baseline_rr=1.5, baseline_cb=1, baseline_om=5)
        assert result["baseline"]["rr"] == 1.5
        assert result["cb_varies"] is False
        assert "rr_x_om" in result["matrices"]
        assert len(result["neighbors"]) >= 2  # RR±1 + OM neighbors

    def test_e1_grid_cb_varies(self):
        """E1 with CB 1-3 -> should produce per-aperture RR×CB matrices."""
        grid = [
            self._make_grid_cell(1.5, 1, 5, 0.08),
            self._make_grid_cell(1.5, 2, 5, 0.10),
            self._make_grid_cell(1.5, 3, 5, 0.07),
            self._make_grid_cell(2.0, 1, 5, 0.06),
            self._make_grid_cell(2.0, 2, 5, 0.09),
            self._make_grid_cell(2.0, 3, 5, 0.05),
        ]
        result = build_heatmap_data(grid, baseline_rr=1.5, baseline_cb=2, baseline_om=5)
        assert result["cb_varies"] is True
        assert 5 in result["matrices"]

    def test_no_baseline_found(self):
        """Grid doesn't contain the specified baseline params."""
        grid = [self._make_grid_cell(1.0, 1, 5)]
        result = build_heatmap_data(grid, baseline_rr=3.0, baseline_cb=1, baseline_om=5)
        assert result["baseline"] is None
        assert result["verdict"] == "NO_BASELINE"

    def test_neighbor_count(self):
        """Baseline at RR1.5 CB1 O5 should find RR1.0 and RR2.0 as neighbors."""
        grid = [
            self._make_grid_cell(1.0, 1, 5, 0.09),
            self._make_grid_cell(1.5, 1, 5, 0.10),
            self._make_grid_cell(2.0, 1, 5, 0.08),
        ]
        result = build_heatmap_data(grid, baseline_rr=1.5, baseline_cb=1, baseline_om=5)
        assert len(result["neighbors"]) == 2
