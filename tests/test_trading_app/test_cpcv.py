"""Tests for trading_app.cpcv.

Companion to ``docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml``
and ``docs/institutional/pre_registered_criteria.md`` Amendment 3.2.

Integrity checks (from the pre-reg § implementation_integrity_checks):
- IS/OOS discipline: tests operate on synthetic indices only — no real
  orb_outcomes access here.
- Purge: (not index-based in this module; trading_day purge is caller's
  responsibility per cpcv.py docstring).
- Embargo: tested directly for index-disjointness and post-test-chunk
  exclusion.
- Fold invariance: deterministic partitioning verified.
- Aggregation: per-fold df = n_test - 1 verified indirectly via t-stat
  sanity on a known-null draw.
"""

from __future__ import annotations

import random

import pytest

from trading_app.cpcv import (
    _contiguous_chunks,
    cpcv_evaluate,
    cpcv_fold_t_statistic,
    cpcv_splits,
)


class TestContiguousChunks:
    def test_even_partition(self):
        chunks = _contiguous_chunks(12, 4)
        assert [len(c) for c in chunks] == [3, 3, 3, 3]
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9, 10, 11]

    def test_uneven_partition_first_chunks_get_extra(self):
        # 10 / 3 → base=3, remainder=1 → first chunk gets +1
        chunks = _contiguous_chunks(10, 3)
        assert [len(c) for c in chunks] == [4, 3, 3]
        # Contiguous cover
        flat = [i for c in chunks for i in c]
        assert flat == list(range(10))

    def test_covers_range_exactly(self):
        chunks = _contiguous_chunks(100, 7)
        flat = [i for c in chunks for i in c]
        assert flat == list(range(100))

    def test_deterministic(self):
        a = _contiguous_chunks(50, 5)
        b = _contiguous_chunks(50, 5)
        assert a == b


class TestCpcvSplits:
    def test_produces_correct_fold_count(self):
        # C(6, 2) = 15 folds
        folds = list(cpcv_splits(n_trades=60, n_splits=6, n_test_splits=2, embargo_trades=0))
        assert len(folds) == 15
        # C(4, 1) = 4 folds
        folds = list(cpcv_splits(n_trades=40, n_splits=4, n_test_splits=1, embargo_trades=0))
        assert len(folds) == 4

    def test_train_and_test_disjoint(self):
        for train_idx, test_idx in cpcv_splits(n_trades=60, n_splits=6, n_test_splits=2, embargo_trades=0):
            assert set(train_idx).isdisjoint(test_idx)

    def test_embargo_zero_covers_full_range(self):
        # With embargo=0, train + test should cover exactly range(n)
        for train_idx, test_idx in cpcv_splits(n_trades=60, n_splits=6, n_test_splits=2, embargo_trades=0):
            covered = set(train_idx) | set(test_idx)
            assert covered == set(range(60)), "train ∪ test must equal full range at embargo=0"

    def test_embargo_excludes_indices_after_test_chunk(self):
        # n=30, n_splits=3 → chunks [0-9], [10-19], [20-29]
        # Pick test_chunk_ids = (0,) → test_idx = [0..9]
        # With embargo=3, indices 10,11,12 must NOT be in train.
        folds = list(cpcv_splits(n_trades=30, n_splits=3, n_test_splits=1, embargo_trades=3))
        # The first fold is test=chunk[0]
        train0, test0 = folds[0]
        assert test0 == list(range(0, 10))
        for embargoed in (10, 11, 12):
            assert embargoed not in train0, f"embargoed idx {embargoed} leaked into train"
        # 13 onwards should be in train (except test indices themselves)
        for kept in (13, 14, 19, 20, 29):
            assert kept in train0

    def test_embargo_at_last_chunk_no_overflow(self):
        # Test chunk is the LAST chunk — embargo window would run past n_trades.
        # Must not error and must not wrap around.
        folds = list(cpcv_splits(n_trades=30, n_splits=3, n_test_splits=1, embargo_trades=5))
        # Fold with test = chunk[2] = [20..29] has no indices after 29 to embargo.
        train_last, test_last = folds[-1]
        assert test_last == list(range(20, 30))
        # Indices 0..19 are all in train (the other two chunks).
        for idx in range(0, 20):
            assert idx in train_last

    def test_deterministic_same_inputs_same_folds(self):
        a = list(cpcv_splits(60, 6, 2, 5))
        b = list(cpcv_splits(60, 6, 2, 5))
        assert a == b

    @pytest.mark.parametrize(
        "n_trades,n_splits,n_test_splits,embargo,expected_error",
        [
            (10, 1, 1, 0, "n_splits=1"),
            (10, 3, 0, 0, "n_test_splits=0"),
            (10, 3, 3, 0, "n_test_splits=3"),
            (10, 3, 2, -1, "embargo_trades=-1"),
            (2, 3, 1, 0, "n_trades=2 must be >= n_splits=3"),
        ],
    )
    def test_invalid_params_raise(self, n_trades, n_splits, n_test_splits, embargo, expected_error):
        with pytest.raises(ValueError, match=expected_error):
            list(cpcv_splits(n_trades, n_splits, n_test_splits, embargo))


class TestCpcvFoldTStatistic:
    def test_returns_none_for_empty_or_single(self):
        assert cpcv_fold_t_statistic([]) is None
        assert cpcv_fold_t_statistic([0.5]) is None

    def test_returns_none_for_zero_variance(self):
        # Same fp-dust guard as _estimate_oos_power
        assert cpcv_fold_t_statistic([0.1] * 30) is None

    def test_positive_mean_gives_positive_t(self):
        rng = random.Random(42)
        pnls = [rng.gauss(0.3, 1.0) for _ in range(100)]
        t = cpcv_fold_t_statistic(pnls)
        assert t is not None
        assert t > 0  # positive mean → positive t

    def test_negative_mean_gives_negative_t(self):
        rng = random.Random(42)
        pnls = [rng.gauss(-0.3, 1.0) for _ in range(100)]
        t = cpcv_fold_t_statistic(pnls)
        assert t is not None
        assert t < 0


class TestCpcvEvaluate:
    def test_null_data_reject_rate_near_alpha(self):
        """Known-null sanity check (rough H1 preview with 1 seed)."""
        rng = random.Random(123)
        returns = [rng.gauss(0.0, 1.0) for _ in range(2000)]
        result = cpcv_evaluate(returns, n_splits=6, n_test_splits=2, embargo_trades=5)
        # Not yet the full H1 10-seed calibration — that runs in
        # research/cpcv_calibration_v1.py. Here we just sanity check
        # that reject_fraction is in a plausible band on a single seed.
        assert 0.0 <= result["reject_fraction"] <= 0.30, (
            f"single-seed reject_fraction implausible: {result['reject_fraction']}"
        )
        assert result["n_folds"] == 15  # C(6, 2)
        assert result["params"]["embargo_trades"] == 5

    def test_strong_edge_data_reject_rate_high(self):
        """Known-edge sanity check (rough H2 preview with 1 seed)."""
        rng = random.Random(123)
        returns = [rng.gauss(0.5, 1.0) for _ in range(2000)]
        result = cpcv_evaluate(returns, n_splits=6, n_test_splits=2, embargo_trades=5)
        # With effect 0.5 sd, N per fold ~666 → power should be near 1.
        assert result["reject_fraction"] > 0.80, (
            f"strong-effect reject_fraction should be high: {result['reject_fraction']}"
        )

    def test_result_structure_has_required_keys(self):
        rng = random.Random(1)
        returns = [rng.gauss(0.1, 1.0) for _ in range(200)]
        result = cpcv_evaluate(returns, n_splits=5, n_test_splits=2, embargo_trades=0)
        assert set(result.keys()) >= {
            "n_folds",
            "reject_fraction",
            "mean_r_across_folds",
            "folds",
            "params",
        }
        for fold in result["folds"]:
            assert set(fold.keys()) >= {
                "n_test",
                "mean_r",
                "t",
                "p_two_tailed",
                "rejects_h0",
            }
