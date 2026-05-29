"""Tests for research/mgc_cpcv_audit.py — CPCV methodology-correct audit.

Covers the load-bearing logic that the verdict depends on:
  - phi formula (AFML Sec 12.4.1) and group-bounds partition (Sec 12.4.1)
  - the FIXED-OUTCOME degeneracy: phi full-coverage paths are identical, so the
    informative distribution is the per-split test-fold set (the core finding)
  - LOCKED kill criteria K1-K4 routing (no post-hoc change)
  - NaN / zero-variance degenerate-input guard (institutional-rigor Sec 6)
  - purge/embargo removes train (not test) observations
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from research import mgc_cpcv_audit as mod


# ---- phi formula + partition ------------------------------------------------
def test_phi_matches_afml_worked_example():
    # AFML Fig 12.1: N=6, k=2 -> C(6,4)=15 splits -> phi=5 paths.
    assert math.comb(6, 4) == 15
    assert mod.cpcv_phi(6, 2) == 5


def test_group_bounds_partition_is_complete_and_contiguous():
    bounds = mod._group_bounds(100, 6)
    assert len(bounds) == 6
    # Contiguous, no gaps/overlaps, covers [0, 100).
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 100
    # Pair each group with its successor (lengths differ by 1 by design).
    for cur, nxt in zip(bounds[:-1], bounds[1:], strict=True):
        assert cur[1] == nxt[0]  # this group's end == next group's start
    # Groups 1..N-1 are floor(T/N)=16; Nth absorbs the remainder.
    sizes = [e - s for s, e in bounds]
    assert sizes[:5] == [16, 16, 16, 16, 16]
    assert sizes[5] == 100 - 16 * 5  # = 20


# ---- the core finding: fixed-outcome path degeneracy ------------------------
def _toy_df(pnl: list[float]) -> pd.DataFrame:
    days = pd.Timestamp("2024-01-01") + pd.to_timedelta(range(len(pnl)), "D")
    return pd.DataFrame({"trading_day": days, "pnl_r": np.array(pnl, dtype=float)})


def test_fixed_outcome_phi_paths_are_structurally_identical():
    """The phi full-coverage paths reconstruct the identical full series.

    This is the load-bearing methodological finding: for a backtest with no
    model refit, every phi path is a union of ALL N groups, so they cannot
    differ. The script reports the per-split fold distribution instead.
    """
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.2, 1.0, size=120).tolist()
    df = _toy_df(pnl)
    c = mod.Candidate(1, "TOY", 30, 1.0, "long", "none", "none", 0)
    r = mod.run_cpcv(c, df)
    # phi disclosed as 5 but the fold distribution (15) is what is reported.
    assert r.phi == 5
    assert r.n_splits == 15
    assert len(r.folds) == 15
    # The full-coverage path = full series: its t equals the pooled t exactly.
    full_t = mod._t_stat(np.array(pnl))[3]
    assert r.pooled_t == pytest.approx(full_t, rel=1e-9)


def test_fold_distribution_has_real_dispersion():
    """The 15 test-fold means must vary (unlike the degenerate phi paths)."""
    rng = np.random.default_rng(2)
    pnl = rng.normal(0.15, 1.0, size=180).tolist()
    df = _toy_df(pnl)
    c = mod.Candidate(1, "TOY", 30, 1.0, "long", "none", "none", 0)
    r = mod.run_cpcv(c, df)
    fold_exprs = [fr.expr for fr in r.folds]
    # Genuine dispersion: not all identical (the whole point of the fix).
    assert len(set(round(e, 6) for e in fold_exprs)) > 1
    assert r.iqr_expr > 0.0


def test_each_fold_holds_out_exactly_k_groups():
    rng = np.random.default_rng(3)
    df = _toy_df(rng.normal(0.1, 1.0, size=120).tolist())
    c = mod.Candidate(1, "TOY", 30, 1.0, "long", "none", "none", 0)
    r = mod.run_cpcv(c, df)
    expected_combos = set(combinations(range(mod.N_GROUPS), mod.K_TEST))
    seen = {fr.test_groups for fr in r.folds}
    assert seen == expected_combos
    for fr in r.folds:
        assert len(fr.test_groups) == mod.K_TEST


# ---- LOCKED kill criteria K1-K4 (no post-hoc change) ------------------------
def test_k3_pbo_over_half_is_wrong():
    v, reason = mod._classify(
        median_expr=0.2,
        median_t=5.0,
        worst_expr=0.1,
        frac_positive=0.9,
        pbo=0.60,
        pooled_power=0.9,
    )
    assert v == "WRONG"
    assert "K3" in reason


def test_k2_catastrophic_worst_fold_is_wrong():
    v, reason = mod._classify(
        median_expr=0.1,
        median_t=4.0,
        worst_expr=-0.10,
        frac_positive=0.55,
        pbo=0.1,
        pooled_power=0.9,  # 45% folds negative
    )
    assert v == "WRONG"
    assert "K2" in reason


def test_k1_nonpositive_median_is_unverified():
    v, reason = mod._classify(
        median_expr=-0.01,
        median_t=4.0,
        worst_expr=-0.02,
        frac_positive=0.5,
        pbo=0.0,
        pooled_power=0.9,
    )
    assert v == "UNVERIFIED"
    assert "K1" in reason


def test_k4_subchordia_underpowered_is_unverified():
    # full-path t < 3.79 AND pooled power < 0.50 -> UNVERIFIED.
    v, reason = mod._classify(
        median_expr=0.2,
        median_t=2.5,
        worst_expr=0.05,
        frac_positive=1.0,
        pbo=0.0,
        pooled_power=0.40,
    )
    assert v == "UNVERIFIED"
    assert "K4" in reason


def test_conditional_when_positive_but_underpowered_but_not_useless():
    # Mirrors the real MGC result: positive, low PBO, t<3.79, power 0.5-0.8.
    v, _ = mod._classify(
        median_expr=0.2,
        median_t=2.6,
        worst_expr=0.05,
        frac_positive=1.0,
        pbo=0.0,
        pooled_power=0.70,
    )
    assert v == "CONDITIONAL"


def test_valid_only_when_all_gates_pass():
    v, _ = mod._classify(
        median_expr=0.2,
        median_t=4.0,
        worst_expr=0.05,
        frac_positive=1.0,
        pbo=0.0,
        pooled_power=0.85,
    )
    assert v == "VALID"


def test_chordia_threshold_is_strict_no_theory():
    # The locked no-theory Chordia bound is 3.79, not 3.0.
    assert mod.CHORDIA_T_STRICT == 3.79
    # t just below 3.79 with high power is NOT valid (gate held strict).
    v, _ = mod._classify(
        median_expr=0.2,
        median_t=3.78,
        worst_expr=0.05,
        frac_positive=1.0,
        pbo=0.0,
        pooled_power=0.95,
    )
    assert v != "VALID"


# ---- degenerate input guard (institutional-rigor Sec 6) ---------------------
def test_nan_median_routes_to_unverified_not_conditional():
    v, reason = mod._classify(
        median_expr=float("nan"),
        median_t=float("nan"),
        worst_expr=float("nan"),
        frac_positive=0.0,
        pbo=0.0,
        pooled_power=0.0,
    )
    assert v == "UNVERIFIED"
    assert "degenerate" in reason


def test_zero_variance_constant_series_is_not_valid():
    # A constant-positive series (zero variance) must NOT be blessed VALID.
    df = _toy_df([0.5] * 120)
    c = mod.Candidate(1, "TOY", 30, 1.0, "long", "none", "none", 0)
    r = mod.run_cpcv(c, df)
    assert r.verdict != "VALID"


# ---- purge/embargo removes train, never test --------------------------------
def test_purge_embargo_drops_train_in_test_span():
    n = 60
    days = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
        dtype="datetime64[D]",
    )
    bounds = mod._group_bounds(n, 6)  # 6 groups of 10
    # Test on groups (2, 3); train is (0,1,4,5).
    train_groups = (0, 1, 4, 5)
    test_groups = (2, 3)
    keep = mod._purge_embargo_train_idx(train_groups, test_groups, bounds, days, embargo_days=1)
    # No kept index may fall inside the test groups' positional span.
    test_positions = set()
    for tg in test_groups:
        s, e = bounds[tg]
        test_positions.update(range(s, e))
    assert not (set(keep.tolist()) & test_positions)
    # Embargo removes the 1 day immediately after the test span (group 4 start).
    g4_start = bounds[4][0]
    assert g4_start not in set(keep.tolist())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
