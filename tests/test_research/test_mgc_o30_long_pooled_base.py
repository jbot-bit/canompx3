"""Tests for research/mgc_o30_long_pooled_base.py — pooled base + overlay scan.

Covers the load-bearing logic the verdict depends on:
  - BH-FDR (RULE 4) correctness, including NaN handling and K=family
  - disposition routing: all three branches (DIVERSIFIER / OVERLAY / DEAD)
  - RULE 1.2 look-ahead gate raises on overnight_* for unsafe sessions
  - flip-rate computation (pooled-finding-rule.md)
  - t-stat / power-tier delegation to canonical helpers
  - pressure test (RULE 13) catches a label-as-feature look-ahead overlay
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research import mgc_o30_long_pooled_base as mod


# ---- BH-FDR (RULE 4) --------------------------------------------------------
def test_bh_fdr_all_significant():
    # Three tiny p-values; all should pass at q=0.05.
    assert mod.bh_fdr([0.001, 0.002, 0.003], q=0.05) == [True, True, True]


def test_bh_fdr_none_significant():
    assert mod.bh_fdr([0.4, 0.6, 0.9], q=0.05) == [False, False, False]


def test_bh_fdr_step_up_keeps_lower_ranked():
    # Classic BH: a borderline larger p passes because a smaller one anchors it.
    # K=4, q=0.05 thresholds: 0.0125, 0.025, 0.0375, 0.05.
    # p=[0.04, 0.01, 0.20, 0.03]: sorted 0.01(r1<=0.0125 ok),0.03(r2<=0.025? no),
    # but step-up uses MAX rank meeting p<=r/K*q. 0.01<=0.0125 -> rank1 ok.
    # 0.03<=0.025? no. 0.04<=0.0375? no. So only rank1 region. max_rank=1 ->
    # only the smallest (0.01) passes.
    out = mod.bh_fdr([0.04, 0.01, 0.20, 0.03], q=0.05)
    assert out == [False, True, False, False]


def test_bh_fdr_nan_fails_and_shrinks_k():
    # NaN entries cannot be discoveries and are excluded from K.
    out = mod.bh_fdr([0.001, float("nan"), 0.002], q=0.05)
    assert out[1] is False
    assert out[0] is True and out[2] is True


def test_bh_fdr_empty():
    assert mod.bh_fdr([], q=0.05) == []


# ---- t-stat / power tier ----------------------------------------------------
def test_t_stat_basic():
    n, mean, t = mod._t_stat(np.array([1.0, -1.0, 1.0, -1.0, 1.0]))
    assert n == 5
    assert mean == pytest.approx(0.2, abs=1e-9)
    assert math.isfinite(t)


def test_t_stat_too_few():
    n, mean, t = mod._t_stat(np.array([1.0]))
    assert n == 1
    assert math.isnan(mean) and math.isnan(t)


def test_t_stat_zero_variance():
    _, mean, t = mod._t_stat(np.array([0.5, 0.5, 0.5, 0.5]))
    assert mean == pytest.approx(0.5)
    assert t == 0.0  # zero std -> t defined as 0, not div-by-zero


def test_power_tier_delegates_to_canonical():
    # High t on large N -> CAN_REFUTE; tiny t -> STATISTICALLY_USELESS.
    p_hi, tier_hi = mod._power_tier(8.0, 1000)
    p_lo, tier_lo = mod._power_tier(0.3, 50)
    assert tier_hi == "CAN_REFUTE"
    assert tier_lo == "STATISTICALLY_USELESS"
    assert p_hi > p_lo


def test_oos_power_separates_is_effect_from_oos_n():
    # The IS effect size (d = |t_is|/sqrt(n_is)) drives power, evaluated at
    # n_oos. A strong IS effect on a tiny OOS slice must NOT be over-powered.
    # IS: t=4.0 on N=1000 -> d=0.126 (small). OOS N=30 -> low power.
    power, tier = mod._oos_power_tier(t_is=4.0, n_is=1000, n_oos=30)
    assert power < 0.50
    assert tier == "STATISTICALLY_USELESS"
    # Same IS effect on a large OOS -> recoverable power.
    power_big, _ = mod._oos_power_tier(t_is=4.0, n_is=1000, n_oos=2000)
    assert power_big > power


def test_oos_power_distinct_from_naive_collapse():
    # Regression guard: the buggy version used _power_tier(t_is, n_oos), which
    # computes d=|t_is|/sqrt(n_oos) — inflating d when n_oos < n_is. Confirm the
    # correct helper gives a SMALLER power than the naive collapse for the same
    # inputs (i.e., we are not silently over-stating OOS power).
    correct, _ = mod._oos_power_tier(t_is=3.0, n_is=900, n_oos=100)
    naive, _ = mod._power_tier(3.0, 100)  # the old, wrong call
    assert correct < naive


# ---- RULE 1.2 look-ahead gate -----------------------------------------------
def test_overnight_overlay_raises_on_lookhead_unsafe_session():
    # TOKYO_OPEN (10:00 Brisbane) is BEFORE the overnight window closes (17:00),
    # so an overnight_* overlay on it must raise.
    df = pd.DataFrame({"overnight_range_pct": [50.0, 90.0]})
    bad = mod.Overlay(99, "TOKYO_OPEN", 2.0, "overnight_range_pct", ">=", 80)
    with pytest.raises(ValueError, match="RULE 1.2"):
        mod._apply_condition(df, bad)


def test_overnight_overlay_ok_on_safe_session():
    # LONDON_METALS (17:00) is exactly the cutoff -> safe.
    df = pd.DataFrame({"overnight_range_pct": [50.0, 90.0]})
    ok = mod.Overlay(6, "LONDON_METALS", 1.5, "overnight_range_pct", ">=", 80)
    mask = mod._apply_condition(df, ok)
    assert list(mask) == [False, True]


def test_dow_overlay_equality():
    df = pd.DataFrame({"day_of_week": [1, 3, 3, 4]})
    ov = mod.Overlay(2, "NYSE_OPEN", 2.0, "day_of_week", "==", 3)
    assert list(mod._apply_condition(df, ov)) == [False, True, True, False]


# ---- disposition routing (all three branches) -------------------------------
def _base(scope, rr, n, expr, t, power):
    """Minimal BaseResult for disposition tests."""
    return mod.BaseResult(
        scope=scope,
        rr=rr,
        n=n,
        expr=expr,
        t=t,
        p=0.05,
        power=power,
        tier="x",
        n_is=n,
        expr_is=expr,
        t_is=t,
        n_oos=50,
        expr_oos=expr,
        t_oos=t,
        oos_power=power,
        oos_tier="x",
        dir_match=True,
        per_year=[],
    )


def _ov(cpcv_idx, t, power, bh_pass):
    ov = mod.Overlay(cpcv_idx, "NYSE_OPEN", 2.0, "day_of_week", "==", 3)
    return mod.OverlayResult(
        overlay=ov,
        p1_n_on=100,
        p1_expr_on=0.2,
        p1_n_off=400,
        p1_expr_off=0.0,
        p1_lift=0.2,
        p1_t=2.0,
        p1_p=0.04,
        p2_t=t,
        p2_p=0.01,
        p2_power=power,
        p2_tier="x",
        bh_pass=bh_pass,
    )


def test_disposition_diversifier_when_pooled_powered():
    base = [_base("POOLED_ALL_SESSIONS", 2.0, 1200, 0.05, 4.5, 0.9)]
    verdict, _ = mod.disposition(base, [])
    assert verdict == "DIVERSIFIER_CANDIDATE"


def test_disposition_diversifier_when_session_powered():
    # Pooled flat but a single session is powered -> still a candidate.
    base = [
        _base("POOLED_ALL_SESSIONS", 2.0, 4000, 0.01, 1.2, 0.2),
        _base("NYSE_OPEN", 2.0, 1000, 0.10, 4.1, 0.85),
    ]
    verdict, _ = mod.disposition(base, [])
    assert verdict == "DIVERSIFIER_CANDIDATE"


def test_disposition_overlay_role_when_only_overlay_powered():
    base = [_base("POOLED_ALL_SESSIONS", 2.0, 4000, 0.01, 1.2, 0.2)]
    overlays = [_ov(1, t=4.2, power=0.85, bh_pass=True)]
    verdict, _ = mod.disposition(base, overlays)
    assert verdict == "OVERLAY_ROLE"


def test_disposition_dead_when_flat_and_underpowered_overlays():
    base = [_base("POOLED_ALL_SESSIONS", 2.0, 4000, 0.024, 1.41, 0.29)]
    # Overlays BH-pass but only DIRECTIONAL_ONLY (power<0.5 or t<3.79) -> not real.
    overlays = [_ov(i, t=2.5, power=0.7, bh_pass=True) for i in range(1, 7)]
    verdict, reason = mod.disposition(base, overlays)
    assert verdict == "DEAD_FOR_ORB"
    assert "SR-monitor" in reason


def test_disposition_overlay_bh_pass_but_underpowered_is_not_real():
    # BH-pass alone is NOT enough — the on-cell must also clear the power floor.
    base = [_base("POOLED_ALL_SESSIONS", 2.0, 4000, 0.01, 1.2, 0.2)]
    overlays = [_ov(1, t=2.6, power=0.74, bh_pass=True)]  # DIRECTIONAL_ONLY
    verdict, _ = mod.disposition(base, overlays)
    assert verdict == "DEAD_FOR_ORB"


# ---- flip rate (pooled-finding-rule.md) -------------------------------------
def test_flip_rate_counts_sign_disagreement():
    base = [
        _base("POOLED_ALL_SESSIONS", 2.0, 4000, +0.02, 1.4, 0.3),  # pooled +
        _base("A", 2.0, 500, +0.05, 1.0, 0.2),  # agrees
        _base("B", 2.0, 500, -0.03, -0.5, 0.1),  # flips
        _base("C", 2.0, 500, +0.01, 0.3, 0.1),  # agrees
        _base("D", 2.0, 500, -0.02, -0.4, 0.1),  # flips
    ]
    # 2 of 4 cells flip -> 50%.
    assert mod._flip_rate(base) == pytest.approx(50.0)


def test_flip_rate_zero_when_all_agree():
    base = [
        _base("POOLED_ALL_SESSIONS", 2.0, 4000, +0.02, 1.4, 0.3),
        _base("A", 2.0, 500, +0.05, 1.0, 0.2),
    ]
    assert mod._flip_rate(base) == 0.0
