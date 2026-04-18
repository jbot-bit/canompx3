"""Self-tests for research.oneshot_utils.

Guards against the two latent bugs caught in the F5_BELOW_PDL one-shot audit
(2026-04-18):
  A. block-bootstrap H0:mean=0 p-value must differentiate positive / null /
     negative signal. The pre-fix implementation returned p ~ 0.5 on all three.
  B. decide_verdict must evaluate kills UNCONDITIONALLY on power; PARK only
     fires when no KILL fires and N is below the PASS floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.oneshot_utils import (
    POWER_FLOOR_PASS,
    OneShotGates,
    classify_power,
    decide_verdict,
    moving_block_bootstrap_p,
)


# --------------------------------------------------------------------------- #
# A. Bootstrap self-test                                                       #
# --------------------------------------------------------------------------- #


class TestMovingBlockBootstrap:
    """The failure caught in the F5 audit: p ~ 0.5 regardless of signal."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_strong_positive_upper_tail_rejects_null(self, rng):
        data = rng.normal(loc=+0.5, scale=1.0, size=200)
        p = moving_block_bootstrap_p(data, B=3000, block=5, seed=42, tail="upper")
        assert p < 0.05, f"strong-positive upper-tail p must be < 0.05, got {p}"

    def test_true_null_p_values_are_calibrated(self):
        """Under H0 true, p-values should be approximately uniform on [0,1].

        A single realization is noisy; the mean over many realizations must be
        near 0.5. This is the correct null calibration test. The pre-fix
        implementation would have a mean p ≈ 0.5 only by construction (broken
        invariance), so this test alone does not catch the old bug — the
        strong-positive and strong-negative tests do.
        """
        master = np.random.default_rng(12345)
        ps = []
        for trial in range(30):
            data = master.normal(loc=0.0, scale=1.0, size=200)
            ps.append(
                moving_block_bootstrap_p(
                    data, B=2000, block=5, seed=1000 + trial, tail="upper"
                )
            )
        mean_p = float(np.mean(ps))
        assert 0.40 < mean_p < 0.60, f"null calibration failed, mean p = {mean_p}"

    def test_strong_negative_upper_tail_does_not_reject(self, rng):
        data = rng.normal(loc=-0.5, scale=1.0, size=200)
        p = moving_block_bootstrap_p(data, B=3000, block=5, seed=42, tail="upper")
        assert p > 0.95, f"strong-negative upper-tail p must be > 0.95, got {p}"

    def test_strong_positive_lower_tail_does_not_reject(self, rng):
        data = rng.normal(loc=+0.5, scale=1.0, size=200)
        p = moving_block_bootstrap_p(data, B=3000, block=5, seed=42, tail="lower")
        assert p > 0.95, f"strong-positive lower-tail p must be > 0.95, got {p}"

    def test_returns_nan_when_insufficient_data(self):
        data = np.array([0.1, 0.2, 0.3], dtype=float)
        p = moving_block_bootstrap_p(data, B=1000, block=5, seed=0, tail="upper")
        assert np.isnan(p), "p must be NaN when n < block*2"

    def test_two_tail_symmetric(self, rng):
        data = rng.normal(loc=+0.5, scale=1.0, size=200)
        p_two = moving_block_bootstrap_p(data, B=3000, block=5, seed=42, tail="two")
        assert p_two < 0.10, f"strong-positive two-tail p must be < 0.10, got {p_two}"


# --------------------------------------------------------------------------- #
# B. Verdict ordering self-test                                                #
# --------------------------------------------------------------------------- #


class TestVerdictOrdering:
    """Kills are UNCONDITIONAL on power. F5-era short-circuit is gone."""

    def _gates(self) -> OneShotGates:
        return OneShotGates(expR_on_is=+0.3258, eff_ratio_min=0.40)

    def test_negative_expR_kills_even_with_low_N(self):
        res = decide_verdict(expR_on_oos=-0.50, n_on_oos=3, gates=self._gates())
        assert res.verdict == "KILL", res
        assert "K1 ExpR_on_OOS<0" in res.kill_reasons
        assert "K3 sign flip" in res.kill_reasons
        assert res.park_reason is None

    def test_sign_flip_kills_even_with_low_N(self):
        res = decide_verdict(expR_on_oos=-0.01, n_on_oos=4, gates=self._gates())
        assert res.verdict == "KILL"
        assert "K3 sign flip" in res.kill_reasons

    def test_low_eff_ratio_kills_even_with_powered_N(self):
        # expR positive but only 20% of IS effect
        res = decide_verdict(expR_on_oos=+0.05, n_on_oos=50, gates=self._gates())
        assert res.verdict == "KILL"
        assert any("eff_ratio" in r for r in res.kill_reasons)

    def test_underpowered_parks_when_no_kill(self):
        # Positive and above eff floor but only 8 fires
        res = decide_verdict(expR_on_oos=+0.20, n_on_oos=8, gates=self._gates())
        assert res.verdict == "PARK"
        assert res.power_band == "INSUFFICIENT"

    def test_underpowered_parks_at_mid_range_N(self):
        res = decide_verdict(expR_on_oos=+0.20, n_on_oos=20, gates=self._gates())
        assert res.verdict == "PARK"
        assert res.power_band == "UNDERPOWERED"

    def test_pass_requires_power_floor(self):
        res = decide_verdict(
            expR_on_oos=+0.20, n_on_oos=POWER_FLOOR_PASS, gates=self._gates()
        )
        assert res.verdict == "PASS"
        assert res.power_band == "POWERED"
        assert not res.kill_reasons

    def test_pass_requires_sign_match(self):
        # IS is positive; OOS negative, strongly below eff — classic flip
        res = decide_verdict(expR_on_oos=-0.20, n_on_oos=100, gates=self._gates())
        assert res.verdict == "KILL"
        assert not res.sign_match

    def test_park_when_no_fires(self):
        res = decide_verdict(expR_on_oos=float("nan"), n_on_oos=0, gates=self._gates())
        assert res.verdict == "PARK"

    def test_f5_actual_verdict_unchanged(self):
        """Regression: re-apply the F5 numbers under corrected logic.

        Historical F5 one-shot observed: ExpR_on_OOS=-0.0243, N_on_OOS=8.
        Pre-fix script reported KILL. Corrected logic must still report KILL
        (K1 + K2 + K3 all fire). F5 remains DEAD post-audit.
        """
        res = decide_verdict(
            expR_on_oos=-0.0243, n_on_oos=8, gates=self._gates()
        )
        assert res.verdict == "KILL", f"F5 verdict changed under fix: {res}"
        assert "K1 ExpR_on_OOS<0" in res.kill_reasons
        assert "K3 sign flip" in res.kill_reasons


class TestPowerBands:
    def test_below_10_is_insufficient(self):
        assert classify_power(0) == "INSUFFICIENT"
        assert classify_power(9) == "INSUFFICIENT"

    def test_10_to_29_is_underpowered(self):
        assert classify_power(10) == "UNDERPOWERED"
        assert classify_power(29) == "UNDERPOWERED"

    def test_30_plus_is_powered(self):
        assert classify_power(30) == "POWERED"
        assert classify_power(500) == "POWERED"
