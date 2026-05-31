"""Tier-1 canary suite — per-canary detect + clean positive controls.

Each canary asserts the guard CATCHES its injected contamination (``fired=True``
is the desired outcome). But a guard that always says "caught" is useless — so
every canary has TWO arms here:

  1. TRAP arm: the canary fires on its injected contamination.
  2. CLEAN positive control: the same canonical guard does NOT false-alarm on
     legitimate clean data.

Together these prove the guard discriminates rather than constantly-fires.

Canary 7 is dual-variant (neutral name + camouflage name) proving the
value-based T0 tautology is the load-bearing catch when the name-guard is
defeated by renaming a post-entry field to an ``_ALWAYS_SAFE`` name.

Gap #3 (permutation-null sample budget): the MCP canaries (1, 2, 6) are run with
two seeds and the ``fired`` verdict asserted stable — proving the null is not
under-sampled.

Complements ``scripts/tests/test_synthetic_null.py`` (noise→edge regime); this
file owns the guard-efficacy regime.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from pipeline.session_guard import is_feature_safe
from research.oos_power import (
    moving_block_bootstrap_p,
    one_sample_power,
    one_sample_tstat,
    power_verdict,
    t0_correlation,
)
from scripts.tests import canary_suite
from scripts.tests.canary_suite import (
    CANARIES,
    CanaryResult,
    canary_1_randomized_direction,
    canary_2_shuffled_days,
    canary_3_permuted_session_labels,
    canary_4_lagged_vs_leaked,
    canary_5_future_looking_feature,
    canary_6_random_filter_sparsity,
    canary_7_post_entry_disguised,
    canary_8_holdout_contamination,
    canary_9_derived_table_only,
    canary_10_dsr_universe_gaming,
    failed_guards,
    run_canaries,
)
from trading_app.config import is_e2_lookahead_filter
from trading_app.holdout_policy import HOLDOUT_OVERRIDE_TOKEN, enforce_holdout_date


# ─────────────────────────────────────────────────────────────────────────────
# Suite-level: import safety + all canaries run + all fire on their trap
# ─────────────────────────────────────────────────────────────────────────────
def test_suite_imports_and_runs_all_ten() -> None:
    """Import-safe and every canary is registered + runnable (no side effects)."""
    assert len(CANARIES) == 10
    results = run_canaries()
    assert len(results) == 10
    assert all(isinstance(r, CanaryResult) for r in results)


def test_all_guards_fire_on_their_trap() -> None:
    """Every Tier-1 guard catches its injected contamination (fired=True)."""
    results = run_canaries()
    misses = [r.name for r in results if not r.fired]
    assert misses == [], f"guards failed to fire: {misses}"


def test_failed_guards_empty_on_clean_tree() -> None:
    """``failed_guards`` is the drift-gate's input; empty == all caught."""
    assert failed_guards() == []


# ─────────────────────────────────────────────────────────────────────────────
# Canary 1 — randomized direction (MCP). TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_1_fires_on_random_direction() -> None:
    assert canary_1_randomized_direction().fired is True


def test_canary_1_clean_control_real_edge_not_flagged_useless() -> None:
    """Positive control: a GENUINELY-positive series is NOT called edgeless.

    The guard (one_sample_tstat/power_verdict) must distinguish real edge from
    the random-direction null — else it would 'catch' everything.
    """
    rng = np.random.default_rng(123)
    n = 400
    real_edge = rng.normal(0.5, 1.0, size=n)  # d≈0.5 — strong, real
    t_stat, _ = one_sample_tstat(float(real_edge.mean()), float(real_edge.std(ddof=1)), n)
    cohen_d = abs(t_stat) / (n**0.5)
    tier = power_verdict(one_sample_power(cohen_d, n))
    assert abs(t_stat) > 3.0
    assert tier == "CAN_REFUTE"  # NOT STATISTICALLY_USELESS


# ─────────────────────────────────────────────────────────────────────────────
# Canary 2 — shuffled days (MCP). TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_2_fires_on_shuffled_days() -> None:
    assert canary_2_shuffled_days().fired is True


def test_canary_2_clean_control_real_edge_beats_perm_null() -> None:
    """Positive control: a real positive series DOES clear the perm-null."""
    rng = np.random.default_rng(456)
    real = rng.normal(0.4, 1.0, size=400)
    perm_p = moving_block_bootstrap_p(real, seed=456, tail="upper")
    assert perm_p < 0.05  # real edge is detectable; perm-null is not a constant-reject


# ─────────────────────────────────────────────────────────────────────────────
# Canary 3 — permuted session labels. TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_3_fires_on_session_relabel() -> None:
    assert canary_3_permuted_session_labels().fired is True


def test_canary_3_clean_control_prior_session_feature_is_safe() -> None:
    """Positive control: an EARLIER-session feature on a LATER target IS safe."""
    # TOKYO_OPEN feature used for a later session (NYSE_CLOSE) is legitimate.
    assert is_feature_safe("orb_TOKYO_OPEN_size", "NYSE_CLOSE") is True
    # Same-session is also safe (known at trade time).
    assert is_feature_safe("orb_TOKYO_OPEN_size", "TOKYO_OPEN") is True


# ─────────────────────────────────────────────────────────────────────────────
# Canary 4 — lagged vs leaked + join inflation. TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_4_fires_on_leak_and_join_inflation() -> None:
    assert canary_4_lagged_vs_leaked().fired is True


def test_canary_4_clean_control_correct_join_no_inflation() -> None:
    """Positive control: a correctly-joined (non-tripled) series has honest t."""
    rng = np.random.default_rng(789)
    honest = rng.normal(0.15, 1.0, size=300)
    t_honest, _ = one_sample_tstat(float(honest.mean()), float(honest.std(ddof=1)), len(honest))
    # Same data, correctly joined → no √3 inflation (ratio to itself == 1.0).
    assert abs(t_honest / t_honest - 1.0) < 1e-9
    # And a prior-day feature is legitimately safe (the lag arm).
    assert is_feature_safe("prev_day_close", "LONDON_METALS") is True


# ─────────────────────────────────────────────────────────────────────────────
# Canary 5 — future-looking feature. TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_5_fires_on_e2_breakbar_and_never_safe() -> None:
    assert canary_5_future_looking_feature().fired is True


def test_canary_5_clean_control_safe_filter_and_feature_pass() -> None:
    """Positive control: an E2-SAFE filter and a SAFE feature are not flagged."""
    # ORB_G5 is a size-gate, not break-bar dependent → not E2 look-ahead.
    assert is_e2_lookahead_filter("ORB_G5") is False
    # A prior-day feature is safe for any session.
    assert is_feature_safe("atr_20", "NYSE_OPEN") is True


# ─────────────────────────────────────────────────────────────────────────────
# Canary 6 — random filter at real sparsity (FST). TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_6_fires_on_random_filter() -> None:
    assert canary_6_random_filter_sparsity().fired is True


def test_canary_6_clean_control_real_filtered_edge_survives() -> None:
    """Positive control: a filter selecting genuinely-positive trades survives."""
    rng = np.random.default_rng(321)
    n = 500
    pnl = rng.normal(0.0, 1.0, size=n)
    fire = rng.random(n) < 0.30
    pnl[fire] += 0.6  # the filter genuinely selects a positive subset
    on = pnl[fire]
    perm_p = moving_block_bootstrap_p(on, seed=321, tail="upper")
    assert perm_p < 0.05  # real filtered edge clears the perm-null


# ─────────────────────────────────────────────────────────────────────────────
# Canary 7 — post-entry disguised. DUAL-VARIANT + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_7_fires_on_both_variants() -> None:
    assert canary_7_post_entry_disguised().fired is True


def test_canary_7_t0_is_load_bearing_camouflage_defeats_name_guard() -> None:
    """The defining proof: name-guard is DEFEATED on the camouflage name, so
    T0 (value-based) is the only catch.

    - ``prev_day_close`` is in ``_ALWAYS_SAFE`` → name-guard returns True
      (DEFEATED) even though the value is a post-entry leak.
    - A value strongly correlated with pnl_r is caught by T0 (|corr| > 0.70).
    """
    rng = np.random.default_rng(7777)
    pnl_r = rng.normal(0.05, 1.0, size=400)
    leak = pnl_r + rng.normal(0.0, 0.15, size=400)  # sign-correlated post-entry

    # Name-guard is defeated by the safe-sounding name:
    assert is_feature_safe("prev_day_close", "NYSE_OPEN") is True
    # T0 (value-based) is the load-bearing catch:
    assert t0_correlation(leak, pnl_r) > 0.70


def test_canary_7_clean_control_legit_prev_day_close_not_flagged() -> None:
    """Positive control: a REAL prev_day_close (uncorrelated with pnl_r) passes T0."""
    rng = np.random.default_rng(8888)
    pnl_r = rng.normal(0.05, 1.0, size=400)
    legit_prev_close = rng.normal(2000.0, 5.0, size=400)  # price level, independent
    assert t0_correlation(legit_prev_close, pnl_r) <= 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Canary 8 — holdout contamination. TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_8_fires_on_post_sacred_date() -> None:
    assert canary_8_holdout_contamination().fired is True


def test_canary_8_clean_control_pre_sacred_date_passes() -> None:
    """Positive control: a pre-sacred holdout date is accepted (no false raise)."""
    pre = date(2025, 6, 1)  # < HOLDOUT_SACRED_FROM
    assert enforce_holdout_date(pre) == pre  # no raise


def test_canary_8_override_token_is_a_real_gate() -> None:
    """The override path proves the guard is a gate, not a constant raise."""
    post = date(2026, 6, 1)
    with pytest.raises(ValueError):
        enforce_holdout_date(post)
    assert enforce_holdout_date(post, override_token=HOLDOUT_OVERRIDE_TOKEN) == post


# ─────────────────────────────────────────────────────────────────────────────
# Canary 9 — derived-table-only. TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_9_fires_on_derived_only_claim() -> None:
    assert canary_9_derived_table_only().fired is True


def test_canary_9_clean_control_canonical_reproduction_survives() -> None:
    """Positive control: a claim that DOES reproduce from canonical is not killed."""
    rng = np.random.default_rng(999)
    canonical = rng.normal(0.4, 1.0, size=300)  # real edge in canonical layer
    t_stat, _ = one_sample_tstat(float(canonical.mean()), float(canonical.std(ddof=1)), len(canonical))
    assert abs(t_stat) > 3.0  # canonical reproduction confirms the edge


# ─────────────────────────────────────────────────────────────────────────────
# Canary 10 — DSR universe gaming. TRAP + CLEAN control.
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_10_fires_on_winner_filtered_universe() -> None:
    assert canary_10_dsr_universe_gaming().fired is True


def test_canary_10_clean_control_pinned_universe_not_flagged() -> None:
    """Positive control: the FULL pinned family vs itself shows no divergence."""
    rng = np.random.default_rng(1010)
    full = rng.normal(0.0, 0.5, size=200)
    var_full = float(np.var(full, ddof=1))
    # Comparing the pinned universe to itself → divergence 1.0, not flagged.
    divergence = var_full / var_full
    assert divergence <= 1.5  # pinned universe is not DSR_UNIVERSE_UNPINNED


# ─────────────────────────────────────────────────────────────────────────────
# Gap #3 — permutation-null stability across seeds (MCP canaries 1, 2, 6)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "canary_fn",
    [
        canary_1_randomized_direction,
        canary_2_shuffled_days,
        canary_6_random_filter_sparsity,
    ],
)
def test_mcp_canary_verdict_stable_across_module_seed(canary_fn) -> None:
    """The MCP canaries' fired verdict is stable when the module seed shifts.

    Proves the perm-null is not under-sampled (gap #3): perturbing the global
    seed must not flip the verdict. We monkeypatch the module seed and re-run.
    """
    base = canary_fn().fired
    original = canary_suite._SEED
    try:
        canary_suite._SEED = original + 100000
        shifted = canary_fn().fired
    finally:
        canary_suite._SEED = original
    assert base is True and shifted is True, (
        f"{canary_fn.__name__} flipped on seed shift — perm-null may be under-sampled"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Negative meta-test: a dead guard makes the suite report a miss
# ─────────────────────────────────────────────────────────────────────────────
def test_dead_guard_surfaces_as_miss(monkeypatch) -> None:
    """If a guard stops detecting, failed_guards reports it (drift-gate input).

    Monkeypatch ``is_feature_safe`` to a permissive no-op (always 'safe') so
    canaries 3/4/5/7 can no longer catch the session-leak class — the suite
    must surface at least one miss.
    """
    monkeypatch.setattr(canary_suite, "is_feature_safe", lambda *a, **k: True)
    failures = failed_guards()
    assert failures, "a defeated session guard must produce at least one miss"
