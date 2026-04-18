"""Shared helpers for Pathway-B one-shot validators.

Created 2026-04-18 after the F5_BELOW_PDL one-shot audit identified two
latent HIGH findings in per-script copies of this logic:

1. block-bootstrap H0:mean=0 test was centered incorrectly (resampled from
   uncentered data, counted sampled.mean() >= observed.mean() → p ≈ 0.5
   regardless of signal).
2. kill/park ordering short-circuited kills when N was below power floor.

This module is deliberately minimal. It is NOT a general-purpose statistics
library. It exists so the NEXT Pathway-B one-shot (not F5; F5 is DEAD) does
not re-derive the same two bugs.

Authority:
- .claude/rules/backtesting-methodology.md Rule 4 (moving-block bootstrap)
- Lahiri 2003 / Politis-Romano 1994 (moving-block bootstrap under the null)
- docs/institutional/pre_registered_criteria.md Amendment 2.7 (Mode A gates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Power bands used by Pathway-B one-shot gates. PASS requires >= CLT heuristic.
POWER_FLOOR_PASS: int = 30       # CLT heuristic (strategy_validator.py:1052)
POWER_FLOOR_UNDERPOWERED: int = 10  # 10-29: PARK with stronger wording
# N < 10 is INSUFFICIENT (bootstrap cannot run; gate reports PARK — unusable).

PowerBand = Literal["INSUFFICIENT", "UNDERPOWERED", "POWERED"]
Verdict = Literal["PASS", "KILL", "PARK", "INDETERMINATE"]


def classify_power(n_on: int) -> PowerBand:
    """Bucket N_on into one of three power bands.

    - n_on < 10              -> INSUFFICIENT (bootstrap undefined, verdict PARK)
    - 10 <= n_on < 30        -> UNDERPOWERED (PARK with stronger wording)
    - n_on >= 30             -> POWERED (PASS eligible)
    """
    if n_on < POWER_FLOOR_UNDERPOWERED:
        return "INSUFFICIENT"
    if n_on < POWER_FLOOR_PASS:
        return "UNDERPOWERED"
    return "POWERED"


def moving_block_bootstrap_p(
    pnl_on: np.ndarray,
    B: int = 10_000,
    block: int = 5,
    seed: int = 20260418,
    tail: Literal["upper", "lower", "two"] = "upper",
) -> float:
    """Moving-block bootstrap p-value for H0: E[pnl_on] = 0.

    Correctly centers the DATA (not the resample) so the bootstrap distribution
    is generated under the null. Phipson-Smyth adjustment is applied.

    Parameters
    ----------
    pnl_on
        1-D array of per-trade pnl values on the on-signal subset.
    B
        Number of bootstrap replicates.
    block
        Block length (preserves intra-day autocorrelation).
    seed
        RNG seed for reproducibility.
    tail
        - "upper": p = P(boot_mean >= observed | H0).  Use when H1: mean > 0.
        - "lower": p = P(boot_mean <= observed | H0).  Use when H1: mean < 0.
        - "two":   p = P(|boot_mean| >= |observed| | H0).

    Returns
    -------
    float
        Bootstrap p-value. NaN if n < block*2 (insufficient data to form even
        two disjoint blocks — bootstrap is not defined).

    Correctness
    -----------
    Strong-positive synthetic data must give p << 0.05 (upper tail).
    True-null synthetic data must give p ≈ 0.5 (upper or lower tail).
    Strong-negative synthetic data must give p ≈ 1.0 (upper tail).
    See tests/research/test_oneshot_utils.py for the numerical self-test.
    """
    n = len(pnl_on)
    if n < block * 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl_on, dtype=float)
    observed_mean = float(pnl.mean())

    # Center the DATA so H0 holds. Bootstrap from centered distribution.
    centered_data = pnl - observed_mean
    n_blocks = int(np.ceil(n / block))

    boot_means = np.empty(B, dtype=float)
    for b in range(B):
        starts = rng.integers(low=0, high=n - block + 1, size=n_blocks)
        sampled = np.concatenate(
            [centered_data[s : s + block] for s in starts]
        )[:n]
        boot_means[b] = sampled.mean()

    if tail == "upper":
        count = int(np.sum(boot_means >= observed_mean))
    elif tail == "lower":
        count = int(np.sum(boot_means <= observed_mean))
    elif tail == "two":
        count = int(np.sum(np.abs(boot_means) >= abs(observed_mean)))
    else:
        raise ValueError(f"tail must be upper|lower|two, got {tail!r}")

    return (count + 1) / (B + 1)


@dataclass(frozen=True)
class OneShotGates:
    """Locked gate thresholds for a Pathway-B one-shot verdict."""

    expR_on_is: float
    eff_ratio_min: float = 0.40
    power_floor_pass: int = POWER_FLOOR_PASS


@dataclass(frozen=True)
class OneShotResult:
    verdict: Verdict
    power_band: PowerBand
    eff_ratio: float
    sign_match: bool
    kill_reasons: tuple[str, ...]
    park_reason: str | None


def decide_verdict(
    expR_on_oos: float,
    n_on_oos: int,
    gates: OneShotGates,
) -> OneShotResult:
    """Apply the Pathway-B one-shot decision rule.

    Kill evaluation is UNCONDITIONAL on power. Kills take precedence over park.
    Park only fires when no kill fires and N is below the PASS floor. PASS
    requires N_on_OOS >= gates.power_floor_pass AND all primary gates clear.

    This resolves the F5-era ambiguity where K4 (N<5 PARK) and kill_if
    (ExpR<0 KILL) could both fire — kills win.
    """
    sign_is = 1 if gates.expR_on_is > 0 else (-1 if gates.expR_on_is < 0 else 0)
    if n_on_oos > 0 and not np.isnan(expR_on_oos):
        sign_oos = 1 if expR_on_oos > 0 else (-1 if expR_on_oos < 0 else 0)
    else:
        sign_oos = 0
    sign_match = sign_is == sign_oos and sign_oos != 0

    eff_ratio = (
        expR_on_oos / gates.expR_on_is
        if gates.expR_on_is != 0 and n_on_oos > 0 and not np.isnan(expR_on_oos)
        else float("nan")
    )

    kill_reasons: list[str] = []
    # Unconditional on N. Kills only evaluated when there is any OOS observation.
    if n_on_oos > 0 and not np.isnan(expR_on_oos):
        if expR_on_oos < 0:
            kill_reasons.append("K1 ExpR_on_OOS<0")
        if not np.isnan(eff_ratio) and eff_ratio < gates.eff_ratio_min:
            kill_reasons.append(f"K2 eff_ratio<{gates.eff_ratio_min}")
        if not sign_match:
            kill_reasons.append("K3 sign flip")

    band = classify_power(n_on_oos)

    if kill_reasons:
        verdict: Verdict = "KILL"
        park_reason = None
    elif band != "POWERED":
        verdict = "PARK"
        park_reason = (
            "N_on_OOS<10 (INSUFFICIENT — bootstrap undefined)"
            if band == "INSUFFICIENT"
            else f"N_on_OOS<{POWER_FLOOR_PASS} (UNDERPOWERED vs CLT heuristic)"
        )
    elif (
        expR_on_oos >= 0
        and not np.isnan(eff_ratio)
        and eff_ratio >= gates.eff_ratio_min
        and sign_match
    ):
        verdict = "PASS"
        park_reason = None
    else:
        verdict = "INDETERMINATE"
        park_reason = None

    return OneShotResult(
        verdict=verdict,
        power_band=band,
        eff_ratio=eff_ratio,
        sign_match=sign_match,
        kill_reasons=tuple(kill_reasons),
        park_reason=park_reason,
    )


__all__ = [
    "POWER_FLOOR_PASS",
    "POWER_FLOOR_UNDERPOWERED",
    "PowerBand",
    "Verdict",
    "OneShotGates",
    "OneShotResult",
    "classify_power",
    "decide_verdict",
    "moving_block_bootstrap_p",
]
