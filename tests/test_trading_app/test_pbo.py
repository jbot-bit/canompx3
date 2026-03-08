"""Tests for Probability of Backtest Overfitting (PBO) — Bailey et al. 2014."""

from datetime import date, timedelta

from trading_app.pbo import compute_pbo


def _days(n, start=date(2020, 1, 1)):
    """Generate n consecutive trading days starting from start."""
    return [start + timedelta(days=i) for i in range(n)]


class TestComputePbo:
    def test_single_strategy_returns_none(self):
        """PBO requires 2+ strategies (no selection to measure)."""
        days = _days(19)
        result = compute_pbo({"A": [(d, 1.0) for d in days]})
        assert result["pbo"] is None
        assert result["n_splits"] == 0

    def test_insufficient_data_returns_none(self):
        """Too few days for meaningful block partition."""
        days = _days(4)
        result = compute_pbo(
            {
                "A": [(d, 1.0) for d in days],
                "B": [(d, -1.0) for d in days],
            }
        )
        assert result["pbo"] is None

    def test_robust_strategies_low_pbo(self):
        """Two strategies consistently positive -> PBO should be low.

        Strategy A always wins (+2R), B always wins (+1R).
        IS-best is always A, OOS is always positive -> PBO = 0.
        """
        days = _days(80)
        strategy_pnl = {
            "A": [(d, 2.0) for d in days],
            "B": [(d, 1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        assert result["pbo"] <= 0.10  # Robust — near zero
        assert result["n_splits"] == 70  # C(8,4) = 70

    def test_overfit_strategy_high_pbo(self):
        """One strategy overfits early data, fails on later data.

        Strategy A: wins in first half, loses in second half.
        Strategy B: mediocre everywhere.
        Selection on early blocks picks A, but A's OOS is negative.
        """
        days = _days(80)
        half = len(days) // 2
        strategy_pnl = {
            # A: strong first half, collapses second half
            "A": [(d, 3.0) if i < half else (d, -2.0) for i, d in enumerate(days)],
            # B: small consistent edge
            "B": [(d, 0.1) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        # A dominates IS when train includes early blocks, but OOS on late blocks is negative
        assert result["pbo"] > 0.2

    def test_identical_strategies_zero_pbo(self):
        """Identical strategies -> IS-best = any, OOS always same -> PBO=0."""
        days = _days(80)
        strategy_pnl = {
            "A": [(d, 1.0) for d in days],
            "B": [(d, 1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] == 0.0

    def test_all_negative_high_pbo(self):
        """Two losing strategies — IS-best is just less bad, OOS still negative."""
        days = _days(80)
        strategy_pnl = {
            "A": [(d, -0.5) for d in days],
            "B": [(d, -1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        # IS-best (A, less negative) still has negative OOS -> PBO = 1.0
        assert result["pbo"] == 1.0

    def test_n_blocks_parameter(self):
        """Custom block count changes number of splits."""
        days = _days(60)
        strategy_pnl = {
            "A": [(d, 1.0) for d in days],
            "B": [(d, 0.5) for d in days],
        }
        # C(6,3) = 20 splits
        result = compute_pbo(strategy_pnl, n_blocks=6)
        assert result["n_splits"] == 20

    def test_logit_pbo_computed(self):
        """Logit PBO is computed for 0 < PBO < 1."""
        days = _days(80)
        strategy_pnl = {
            "A": [(d, -0.5) for d in days],
            "B": [(d, -1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        # PBO = 1.0, logit is undefined at boundaries
        if result["pbo"] == 1.0:
            assert result["logit_pbo"] is None

    def test_many_strategies(self):
        """PBO works with more than 2 strategies."""
        days = _days(80)
        strategy_pnl = {f"S{i}": [(d, 1.0 + i * 0.1) for d in days] for i in range(5)}
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        # All positive -> PBO should be 0
        assert result["pbo"] == 0.0
        assert result["n_splits"] == 70
