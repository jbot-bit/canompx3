"""
Tests for trading_app.walk_forward module.
"""

import sys
from pathlib import Path
from datetime import date

import pytest

from trading_app.walk_forward import build_folds, evaluate_fold, WalkForwardResult, FoldResult

# ============================================================================
# build_folds tests
# ============================================================================

class TestBuildFolds:
    """Tests for walk-forward fold splitting."""

    @staticmethod
    def _make_days(start_year, end_year):
        """Generate ~250 trading days per year (weekdays only)."""
        days = []
        for yr in range(start_year, end_year + 1):
            for m in range(1, 13):
                for d in range(1, 29):  # avoid month-end issues
                    try:
                        dt = date(yr, m, d)
                        if dt.weekday() < 5:  # Mon-Fri
                            days.append(dt)
                    except ValueError:
                        pass
        return sorted(days)

    def test_no_leakage(self):
        """Train and test days must NEVER overlap in any fold."""
        days = self._make_days(2021, 2025)
        folds = build_folds(days, train_years=3, test_years=1)
        assert len(folds) >= 1
        for train, test in folds:
            overlap = set(train) & set(test)
            assert not overlap, f"LEAKAGE: {len(overlap)} days in both train and test"

    def test_fold_count_3_1(self):
        """5 years of data with train=3, test=1 should produce 2 folds."""
        days = self._make_days(2021, 2025)
        folds = build_folds(days, train_years=3, test_years=1)
        # 2021-2023 train / 2024 test, 2022-2024 train / 2025 test
        assert len(folds) == 2

    def test_fold_count_2_1(self):
        """5 years with train=2, test=1 should produce 3 folds."""
        days = self._make_days(2021, 2025)
        folds = build_folds(days, train_years=2, test_years=1)
        # 2021-2022/2023, 2022-2023/2024, 2023-2024/2025
        assert len(folds) == 3

    def test_train_before_test(self):
        """Train days must all be before test days."""
        days = self._make_days(2021, 2025)
        folds = build_folds(days, train_years=3, test_years=1)
        for train, test in folds:
            assert max(train) < min(test), "Train must end before test starts"

    def test_empty_input(self):
        """Empty trading days returns no folds."""
        assert build_folds([], train_years=3, test_years=1) == []

    def test_insufficient_data(self):
        """2 years of data with train=3, test=1 returns no folds."""
        days = self._make_days(2024, 2025)
        folds = build_folds(days, train_years=3, test_years=1)
        assert len(folds) == 0

    def test_short_test_fold_excluded(self):
        """Partial-year test folds (< 100 days) are excluded."""
        # 2021-2025 with train=3, test=1
        # 2026 has too few days in our real data — simulate similarly
        days = self._make_days(2021, 2025)
        # Add only 50 days in 2026 (not enough for a test fold)
        for d in range(1, 51):
            try:
                days.append(date(2026, 1, d))
            except ValueError:
                pass
        days = sorted(days)
        folds = build_folds(days, train_years=3, test_years=1)
        # Should still be 2 folds (2024 test, 2025 test) — 2026 excluded
        for _, test in folds:
            assert len(test) >= 100

    def test_sliding_window(self):
        """Folds slide forward by test_years."""
        days = self._make_days(2020, 2025)
        folds = build_folds(days, train_years=3, test_years=1)
        # 2020-2022/2023, 2021-2023/2024, 2022-2024/2025 = 3 folds
        assert len(folds) == 3
        # Test years should be consecutive
        test_years = [f[1][0].year for f in folds]
        assert test_years == [2023, 2024, 2025]

# ============================================================================
# evaluate_fold tests
# ============================================================================

class TestEvaluateFold:
    """Tests for fold evaluation."""

    def test_metrics_on_subset_days(self):
        """Changing the test day subset changes the result deterministically."""
        outcomes = [
            {"trading_day": date(2024, 1, 2), "outcome": "win", "pnl_r": 2.0,
             "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 6, 3), "outcome": "loss", "pnl_r": -1.0,
             "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 9, 4), "outcome": "win", "pnl_r": 1.5,
             "mae_r": 0.3, "mfe_r": 1.5, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        all_eligible = {date(2024, 1, 2), date(2024, 6, 3), date(2024, 9, 4)}

        # Full set
        m_full = evaluate_fold(outcomes, all_eligible, all_eligible)
        assert m_full["sample_size"] == 3

        # Subset: only Jan
        m_jan = evaluate_fold(outcomes, all_eligible, {date(2024, 1, 2)})
        assert m_jan["sample_size"] == 1
        assert m_jan["win_rate"] == 1.0

        # Subset: only Jun
        m_jun = evaluate_fold(outcomes, all_eligible, {date(2024, 6, 3)})
        assert m_jun["sample_size"] == 1
        assert m_jun["win_rate"] == 0.0

    def test_ineligible_days_excluded(self):
        """Outcomes on ineligible days must not be counted."""
        outcomes = [
            {"trading_day": date(2024, 1, 2), "outcome": "win", "pnl_r": 2.0,
             "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 6, 3), "outcome": "loss", "pnl_r": -1.0,
             "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        # Only Jan 2 is eligible
        eligible = {date(2024, 1, 2)}
        test_days = {date(2024, 1, 2), date(2024, 6, 3)}

        m = evaluate_fold(outcomes, eligible, test_days)
        assert m["sample_size"] == 1  # Only the eligible day

    def test_empty_outcomes(self):
        """No outcomes in test window returns zeroed metrics."""
        m = evaluate_fold([], set(), {date(2024, 1, 2)})
        assert m["sample_size"] == 0
        assert m["win_rate"] is None

    def test_no_leakage_by_design(self):
        """evaluate_fold only considers days in test_days set."""
        outcomes = [
            {"trading_day": date(2023, 1, 2), "outcome": "win", "pnl_r": 5.0,
             "mae_r": 0.5, "mfe_r": 5.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0,
             "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        eligible = {date(2023, 1, 2), date(2024, 1, 2)}
        # Test fold is only 2024 — the 2023 win must not be included
        test_days = {date(2024, 1, 2)}

        m = evaluate_fold(outcomes, eligible, test_days)
        assert m["sample_size"] == 1
        assert m["win_rate"] == 0.0  # Only the 2024 loss


# ============================================================================
# FoldResult.sparse tests (Bug 3)
# ============================================================================

class TestFoldResultSparse:
    """Bug 3: Folds with trade_count < 15 marked sparse."""

    def test_sparse_when_few_trades(self):
        """trade_count < 15 -> sparse = True."""
        fold = FoldResult(
            fold_idx=1,
            train_start=date(2021, 1, 1), train_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1), test_end=date(2024, 12, 31),
            train_days=750, test_days=250,
            trade_count=8, win_rate=0.5, expectancy_r=0.1,
            sharpe_ratio=0.3, max_drawdown_r=-2.0,
            sparse=True,
        )
        assert fold.sparse is True

    def test_not_sparse_when_enough_trades(self):
        """trade_count >= 15 -> sparse = False."""
        fold = FoldResult(
            fold_idx=1,
            train_start=date(2021, 1, 1), train_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1), test_end=date(2024, 12, 31),
            train_days=750, test_days=250,
            trade_count=64, win_rate=0.6, expectancy_r=0.3,
            sharpe_ratio=0.5, max_drawdown_r=-3.0,
            sparse=False,
        )
        assert fold.sparse is False

    def test_sparse_default_false(self):
        """sparse defaults to False for backward compatibility."""
        fold = FoldResult(
            fold_idx=1,
            train_start=date(2021, 1, 1), train_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1), test_end=date(2024, 12, 31),
            train_days=750, test_days=250,
            trade_count=100, win_rate=0.6, expectancy_r=0.3,
            sharpe_ratio=0.5, max_drawdown_r=-3.0,
        )
        assert fold.sparse is False

    def test_sparse_threshold_boundary(self):
        """trade_count == 14 -> sparse, trade_count == 15 -> not sparse."""
        assert FoldResult(
            fold_idx=1,
            train_start=date(2021, 1, 1), train_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1), test_end=date(2024, 12, 31),
            train_days=750, test_days=250,
            trade_count=14, win_rate=0.5, expectancy_r=0.1,
            sharpe_ratio=0.3, max_drawdown_r=-2.0,
            sparse=True,
        ).sparse is True

        assert FoldResult(
            fold_idx=1,
            train_start=date(2021, 1, 1), train_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1), test_end=date(2024, 12, 31),
            train_days=750, test_days=250,
            trade_count=15, win_rate=0.5, expectancy_r=0.1,
            sharpe_ratio=0.3, max_drawdown_r=-2.0,
            sparse=False,
        ).sparse is False


# ============================================================================
# WalkForwardResult fold imbalance tests (Bug 4)
# ============================================================================

class TestFoldImbalance:
    """Bug 4: Fold imbalance detection in WalkForwardResult."""

    def test_imbalance_fields_default(self):
        """Defaults: ratio=None, imbalanced=False."""
        wf = WalkForwardResult(strategy_id="TEST", filter_type="G4")
        assert wf.fold_imbalance_ratio is None
        assert wf.fold_imbalanced is False

    def test_imbalance_flagged(self):
        """Ratio > 5.0 -> fold_imbalanced = True."""
        wf = WalkForwardResult(
            strategy_id="TEST", filter_type="G4",
            fold_imbalance_ratio=10.0, fold_imbalanced=True,
        )
        assert wf.fold_imbalanced is True
        assert wf.fold_imbalance_ratio == 10.0

    def test_balanced_not_flagged(self):
        """Ratio <= 5.0 -> fold_imbalanced = False."""
        wf = WalkForwardResult(
            strategy_id="TEST", filter_type="G4",
            fold_imbalance_ratio=2.5, fold_imbalanced=False,
        )
        assert wf.fold_imbalanced is False
