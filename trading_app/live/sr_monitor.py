"""Score-based Shiryaev-Roberts drift monitor.

Implements the nonparametric score-function SR recursion from
Pepelyshev & Polunchenko (2015), Eq. 13-18:

    R_n = (1 + R_{n-1}) * exp(S_n)

where S_n is a change-sensitive linear-quadratic score applied to the
standardized trade outcome stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import random


def _score_coefficients(delta: float, variance_ratio: float) -> tuple[float, float, float]:
    """Return Eq. 18 coefficients for the Eq. 17 score function.

    `delta` is the standardized post-change mean shift:
        delta = (mu_post - mu_pre) / sigma_pre

    `variance_ratio` is q = sigma_pre / sigma_post.
    For mean-shift-only monitoring we use q=1.0, which collapses the
    quadratic term to zero.
    """
    q = variance_ratio
    if q <= 0:
        raise ValueError("variance_ratio must be > 0")
    c1 = delta * q * q
    c2 = (1 - q * q) / 2
    c3 = (delta * delta * q * q) / 2 - math.log(q)
    return c1, c2, c3


def _estimate_arl(
    threshold: float,
    *,
    delta: float,
    variance_ratio: float,
    n_paths: int,
    max_steps: int,
    seed: int,
) -> float:
    """Estimate pre-change ARL by Monte Carlo on standardized iid samples."""
    c1, c2, c3 = _score_coefficients(delta, variance_ratio)
    rng = random.Random(seed)
    total_steps = 0

    for _ in range(n_paths):
        stat = 0.0
        for step in range(1, max_steps + 1):
            x = rng.gauss(0.0, 1.0)
            score = c1 * x + c2 * (x * x) - c3
            stat = (1.0 + stat) * math.exp(score)
            if stat >= threshold:
                total_steps += step
                break
        else:
            total_steps += max_steps

    return total_steps / n_paths


@lru_cache(maxsize=32)
def calibrate_sr_threshold(
    target_arl: int = 60,
    *,
    delta: float = -1.0,
    variance_ratio: float = 1.0,
    n_paths: int = 2000,
    max_steps: int = 5000,
    seed: int = 0,
) -> float:
    """Calibrate an SR threshold to an approximate pre-change ARL target.

    The calibration is done on the standardized pre-change stream, so the
    same threshold can be reused across strategies provided their trade
    outcomes are standardized consistently.
    """
    if target_arl <= 1:
        raise ValueError("target_arl must be > 1")

    low = 1.0
    high = 10.0
    while _estimate_arl(
        high,
        delta=delta,
        variance_ratio=variance_ratio,
        n_paths=n_paths,
        max_steps=max_steps,
        seed=seed,
    ) < target_arl:
        low = high
        high *= 2.0
        if high > 1_000_000:
            break

    for _ in range(12):
        mid = (low + high) / 2.0
        arl = _estimate_arl(
            mid,
            delta=delta,
            variance_ratio=variance_ratio,
            n_paths=n_paths,
            max_steps=max_steps,
            seed=seed,
        )
        if arl < target_arl:
            low = mid
        else:
            high = mid

    return round(high, 2)


@dataclass
class ShiryaevRobertsMonitor:
    """Score-function SR monitor for a single trade-result stream."""

    expected_r: float
    std_r: float
    threshold: float
    delta: float = -1.0
    variance_ratio: float = 1.0

    sr_stat: float = field(default=0.0, init=False)
    alarm_triggered: bool = field(default=False, init=False)
    n_trades: int = field(default=0, init=False)

    def score(self, actual_r: float) -> float:
        """Compute the Eq. 17 score on the standardized trade outcome."""
        if self.std_r <= 0:
            return 0.0
        x = (actual_r - self.expected_r) / self.std_r
        c1, c2, c3 = _score_coefficients(self.delta, self.variance_ratio)
        return c1 * x + c2 * (x * x) - c3

    def update(self, actual_r: float) -> bool:
        """Process one trade. Return True only when the alarm first triggers."""
        self.n_trades += 1
        if self.std_r <= 0 or self.threshold <= 0:
            return False

        self.sr_stat = (1.0 + self.sr_stat) * math.exp(self.score(actual_r))
        if self.sr_stat >= self.threshold and not self.alarm_triggered:
            self.alarm_triggered = True
            return True
        return False

    def clear(self) -> None:
        """Reset the SR statistic after investigation/restart."""
        self.sr_stat = 0.0
        self.alarm_triggered = False

    @property
    def alarm_ratio(self) -> float:
        """Current SR statistic as a fraction of the alarm threshold."""
        if self.threshold <= 0:
            return 0.0
        return self.sr_stat / self.threshold

