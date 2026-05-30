"""Tests for research.oos_holdout — powered trade-fraction OOS carve.

The helper is the single canonical source for the fraction-split policy
declared in feedback_powered_oos_holdout_at_discovery_no_calendar_wait_2026_05_29.
These tests pin:

  - default-fraction is kept when it already clears the target tier
  - OOS GROWS (never shrinks) when the default block is underpowered
  - the STATISTICALLY_USELESS fallback (doctrine § 3 — report truth, no "wait")
  - degenerate/thin data never raises
  - index blocks are contiguous, disjoint, temporal-order-preserving
  - delegation: power numbers match research.oos_power exactly (no re-encoding)
"""

from __future__ import annotations

import numpy as np
import pytest

from research.oos_holdout import (
    DEFAULT_OOS_FRACTION,
    PoweredOOSSplit,
    powered_oos_split,
)
from research.oos_power import one_sample_power, power_verdict

# ─────────────────────────────────────────────────────────────────────────
# Synthetic series builders. A strong, low-variance positive edge gives a
# high IS t (large Cohen's d), so even a modest OOS block reaches power. A
# weak edge needs a large OOS to reach any tier — used to exercise GROW and
# the underpowered fallback.
# ─────────────────────────────────────────────────────────────────────────


def _edge_series(n: int, mean: float, std: float, seed: int = 0) -> np.ndarray:
    """Deterministic gaussian-ish return block (temporal order = array order)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(mean, std, size=n)
    # Center to the exact requested mean so the t-stat is reproducible.
    x = x - x.mean() + mean
    return x


# ─────────────────────────────────────────────────────────────────────────
# Structural invariants
# ─────────────────────────────────────────────────────────────────────────


def test_index_blocks_are_contiguous_disjoint_and_cover_full():
    r = _edge_series(200, 0.20, 1.0, seed=1)
    res = powered_oos_split(r)
    # Disjoint, contiguous, cover [0, n).
    assert res.is_idx[0] == 0
    assert res.oos_idx[-1] == len(r) - 1
    assert res.is_idx[-1] + 1 == res.oos_idx[0]
    combined = np.concatenate([res.is_idx, res.oos_idx])
    np.testing.assert_array_equal(combined, np.arange(len(r)))


def test_oos_is_the_trailing_block_temporal_order_preserved():
    # OOS must be the MOST-RECENT trades (highest indices), never random.
    r = _edge_series(120, 0.15, 1.0, seed=2)
    res = powered_oos_split(r)
    assert res.oos_idx.min() > res.is_idx.max()


def test_returns_dataclass_with_all_diagnostics():
    res = powered_oos_split(_edge_series(150, 0.2, 1.0, seed=3))
    assert isinstance(res, PoweredOOSSplit)
    assert 0.0 <= res.achieved_power <= 1.0
    assert res.tier in {"CAN_REFUTE", "DIRECTIONAL_ONLY", "STATISTICALLY_USELESS"}
    assert res.is_cohen_d >= 0.0


# ─────────────────────────────────────────────────────────────────────────
# Sizing policy: keep default when it clears, grow when underpowered
# ─────────────────────────────────────────────────────────────────────────


def test_keeps_default_fraction_when_it_already_clears_target():
    # Strong edge: 30% OOS on N=200 = 60 trades. With a large Cohen's d the
    # default block clears DIRECTIONAL_ONLY, so the fraction must stay at 0.30.
    r = _edge_series(200, 0.30, 1.0, seed=4)
    res = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    assert res.reaches_target
    # Achieved fraction equals the default (within the rounding of N*0.30).
    expected_oos = round(200 * DEFAULT_OOS_FRACTION)
    assert len(res.oos_idx) == expected_oos
    assert res.achieved_fraction == pytest.approx(expected_oos / 200)


def test_grows_oos_when_default_fraction_is_underpowered():
    # Weak edge: the default 30% block is underpowered, so the helper must
    # GROW the OOS block beyond 30% (achieved fraction > default).
    r = _edge_series(300, 0.05, 1.0, seed=5)
    res = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    if res.reaches_target:
        # If it reached target, it must have done so by growing past default.
        assert res.achieved_fraction >= DEFAULT_OOS_FRACTION
        assert len(res.oos_idx) > round(300 * DEFAULT_OOS_FRACTION)
    else:
        # Or it exhausted the admissible OOS without clearing — that is the
        # underpowered fallback, verified separately below.
        assert res.tier == "STATISTICALLY_USELESS"


def test_never_shrinks_below_default_when_default_clears():
    # Even a very strong edge keeps the full default block (policy: keep, don't
    # win back IS trades).
    r = _edge_series(200, 0.50, 1.0, seed=6)
    res = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    assert len(res.oos_idx) == round(200 * DEFAULT_OOS_FRACTION)


# ─────────────────────────────────────────────────────────────────────────
# Doctrine § 3: underpowered → report truth, never "wait"
# ─────────────────────────────────────────────────────────────────────────


def test_underpowered_full_history_returns_useless_not_raise():
    # Tiny, noisy edge across the whole series: even the maximal admissible OOS
    # cannot reach DIRECTIONAL_ONLY → STATISTICALLY_USELESS, reaches_target False.
    r = _edge_series(40, 0.01, 1.0, seed=7)
    res = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    assert not res.reaches_target
    assert res.tier == "STATISTICALLY_USELESS"
    # Still returns a usable (best-achievable) split, never an exception.
    assert len(res.oos_idx) >= 2


def test_can_refute_target_is_harder_than_directional():
    # The same series should be at least as easy to clear at the lower tier.
    r = _edge_series(250, 0.12, 1.0, seed=8)
    res_dir = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    res_ref = powered_oos_split(r, target_tier="CAN_REFUTE")
    # CAN_REFUTE needs >= as much OOS as DIRECTIONAL_ONLY to clear.
    if res_ref.reaches_target:
        assert res_dir.reaches_target
    assert len(res_ref.oos_idx) >= len(res_dir.oos_idx)


# ─────────────────────────────────────────────────────────────────────────
# Degenerate / thin data — never raises
# ─────────────────────────────────────────────────────────────────────────


def test_too_thin_returns_all_is_no_oos():
    res = powered_oos_split([0.1, 0.2, 0.3])  # below min_is+min_oos default (4)
    assert len(res.oos_idx) == 0
    assert len(res.is_idx) == 3
    assert not res.reaches_target


def test_empty_series_does_not_raise():
    res = powered_oos_split([])
    assert len(res.is_idx) == 0
    assert len(res.oos_idx) == 0
    assert not res.reaches_target


def test_zero_variance_series_is_degenerate_not_crash():
    # All identical returns → std 0 → d 0 → useless, but no divide-by-zero.
    res = powered_oos_split([0.5] * 50)
    assert res.is_cohen_d == 0.0
    assert res.tier == "STATISTICALLY_USELESS"
    assert not res.reaches_target


def test_unknown_target_tier_raises_valueerror():
    with pytest.raises(ValueError, match="unknown target_tier"):
        powered_oos_split([0.1] * 20, target_tier="MAGIC")


# ─────────────────────────────────────────────────────────────────────────
# Canonical delegation — power numbers must match research.oos_power exactly
# ─────────────────────────────────────────────────────────────────────────


def test_achieved_power_matches_oos_power_module_exactly():
    # The helper must NOT re-encode the power calc. Reproduce its number from
    # the public oos_power API given the same (d, n_oos).
    r = _edge_series(200, 0.20, 1.0, seed=9)
    res = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    n_oos = len(res.oos_idx)
    expected = one_sample_power(res.is_cohen_d, n_oos, alpha=0.05)
    assert res.achieved_power == pytest.approx(expected)
    assert res.tier == power_verdict(expected)


@pytest.mark.parametrize(
    "n_full,expected_oos",
    [
        (201, 61),  # graveyard H1_MES_LM_ovn80 — int(201*0.7)=140 → 61
        (219, 66),  # graveyard H2_MNQ_COMEX_garch70 — int(219*0.7)=153 → 66
        (639, 192),  # graveyard H5 — int(639*0.7)=447 → 192
        (179, 54),  # graveyard H3 — int(179*0.7)=125 → 54
        (200, 60),  # round-number control
    ],
)
def test_default_boundary_matches_resweep_convention(n_full, expected_oos):
    # Locks the IS/OOS boundary to int(n*(1-frac)) so the canonical helper
    # reproduces powered_oos_graveyard_resweep's long-standing cut exactly.
    # Use a strong edge so the default block is KEPT (not grown).
    r = _edge_series(n_full, 0.40, 1.0, seed=11)
    res = powered_oos_split(r, target_tier="DIRECTIONAL_ONLY")
    assert len(res.oos_idx) == expected_oos


def test_cohen_d_recovered_from_full_history_t():
    # d = |t_full| / sqrt(N_full). Verify against a hand-computed t.
    r = _edge_series(100, 0.25, 1.0, seed=10)
    m, sd, n = r.mean(), r.std(ddof=1), r.size
    t_full = m / (sd / np.sqrt(n))
    expected_d = abs(t_full) / np.sqrt(n)
    res = powered_oos_split(r)
    assert res.is_cohen_d == pytest.approx(expected_d, rel=1e-9)
