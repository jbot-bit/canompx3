"""OOS statistical power helper — canonical implementation for RULE 3.3.

Enforces `.claude/rules/backtesting-methodology.md` RULE 3.3: before any
binary OOS gate (dir_match, sign-flip, p_oos threshold) can hard-kill a
signal, the OOS sample must have adequate power to detect the IS effect.
Underpowered OOS produces noise-consistent results that cannot distinguish
"signal alive", "signal dead", or "signal reversed" — treating any of them
as refutation is a methodological error.

Literature grounding (extracts in `docs/institutional/literature/`):
    - Harvey & Liu 2015 — OOS is a Sharpe-ratio *haircut*, not a binary veto
    - LdP 2020 ML for Asset Managers — CPCV for short-OOS data; binary
      IS/OOS split is misspecified when OOS < ~20% of total sample

Origin: 2026-04-20 incident. `bull_short_avoidance_deployed_lane_verify.py`
applied dir_match=FALSE as hard kill on N_OOS_per_group=19/20 with power 7.9%
to detect the IS effect (Cohen's d = 0.165). Corrected verdict was
CONDITIONAL — UNVERIFIED, not REJECTED. Canonical helper added so future
verify scripts enforce the power floor in one line.

Usage
-----
    from research.oos_power import oos_ttest_power, power_verdict

    report = oos_ttest_power(
        is_delta=0.157,
        is_pooled_std=0.955,
        n_oos_a=19,
        n_oos_b=20,
        alpha=0.05,
    )
    # report is a dict with cohen_d, power, n_for_80pct, se, ci

    verdict = power_verdict(report["power"])
    # verdict ∈ {"CAN_REFUTE", "DIRECTIONAL_ONLY", "STATISTICALLY_USELESS"}

Integration pattern in a verify script (RULE 3.3 enforcement)::

    pwr = oos_ttest_power(is_delta, is_std, n_oos_bear, n_oos_bull)
    print(f"OOS power to detect IS effect: {pwr['power']*100:.1f}%")
    print(f"dir_match verdict tier: {power_verdict(pwr['power'])}")
    if dir_match is False and power_verdict(pwr["power"]) != "CAN_REFUTE":
        verdict = "UNVERIFIED"   # NOT "REJECTED"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def oos_ttest_power(
    is_delta: float,
    is_pooled_std: float,
    n_oos_a: int,
    n_oos_b: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Power of a two-sample Welch t-test on OOS to detect the IS effect.

    Computes Cohen's d from (is_delta, is_pooled_std), then two-sided
    non-central t power at the given OOS per-group sample sizes. Also
    returns the N per group needed for 80% power at the same effect size,
    so result docs can state how much more OOS must accumulate before
    the dir_match gate is legitimately applicable.

    Parameters
    ----------
    is_delta
        Observed IS mean-difference (group_a - group_b), in the same
        units as pnl_r.
    is_pooled_std
        Pooled IS standard deviation of pnl_r across both groups.
    n_oos_a, n_oos_b
        OOS per-group sample sizes.
    alpha
        Two-sided significance level. Defaults to 0.05.

    Returns
    -------
    dict with keys:
        cohen_d         — |is_delta| / is_pooled_std
        power           — probability of |T_oos| > t_crit under the IS effect
        n_for_80pct     — per-group N needed for 80% power at this d
        se_if_oos       — expected SE of OOS delta given the pooled-std assumption
        ci95_half_width — half-width of 95% CI on OOS delta

    Notes
    -----
    Assumes OOS population standard deviation equals the IS pooled std.
    For deployed lanes this is typically close; large divergence is itself
    informative (regime change).
    """
    if is_pooled_std <= 0:
        raise ValueError("is_pooled_std must be positive")
    if n_oos_a < 2 or n_oos_b < 2:
        raise ValueError("each OOS group needs at least 2 observations")
    cohen_d = abs(is_delta) / is_pooled_std
    n1, n2 = int(n_oos_a), int(n_oos_b)
    df = n1 + n2 - 2
    # Non-centrality parameter for a two-sample t-test with pooled variance
    ncp = cohen_d * np.sqrt((n1 * n2) / (n1 + n2))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Two-sided power: P(|T| > t_crit | ncp)
    power = float(
        1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    )
    # N needed for 80% power (per group; bisection)
    n_for_80 = _n_for_power(cohen_d, 0.80, alpha)
    se_if_oos = float(is_pooled_std * np.sqrt(1 / n1 + 1 / n2))
    ci95_half_width = float(stats.norm.ppf(1 - alpha / 2) * se_if_oos)
    return {
        "cohen_d": float(cohen_d),
        "power": power,
        "n_for_80pct": float(n_for_80),
        "se_if_oos": se_if_oos,
        "ci95_half_width": ci95_half_width,
    }


def _power_at_n(d: float, n: int, alpha: float) -> float:
    """Two-sample Welch t-test power for effect size d at per-group N.

    Uses non-central t for moderate N; falls back to normal approximation
    when the ncp is large enough that `scipy.stats.nct.cdf` returns NaN
    from floating-point overflow (typically ncp > ~30). The normal
    approximation is standard: as df -> inf the non-central t converges
    to a normal(ncp, 1), so power = Phi(ncp - z_crit) + Phi(-ncp - z_crit).
    """
    df = 2 * n - 2
    ncp = d * np.sqrt(n / 2)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    t_crit = stats.t.ppf(1 - alpha / 2, df) if df < 10_000 else z_crit
    power_nct = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    if np.isnan(power_nct):
        # Normal approximation — valid for large df / large ncp
        return float(
            stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        )
    return float(power_nct)


def _n_for_power(d: float, target_power: float, alpha: float) -> int:
    """Smallest per-group N achieving `target_power` for effect size d.

    Uses bisection with `_power_at_n` which falls back to the normal
    approximation at large ncp (avoiding a NaN plateau in the underlying
    non-central t CDF). Search bounded by a practical ceiling of 10M per
    group — sufficient for d > 0.0005.
    """
    if d <= 0:
        return 10_000_000
    lo, hi = 3, 10_000_000
    # Quick sanity: if even hi doesn't reach target, signal with ceiling.
    if _power_at_n(d, hi, alpha) < target_power:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        power_mid = _power_at_n(d, mid, alpha)
        if power_mid >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


@dataclass(frozen=True)
class PowerTier:
    name: str
    min_power: float
    description: str


POWER_TIERS: tuple[PowerTier, ...] = (
    PowerTier(
        "CAN_REFUTE", 0.80,
        "OOS sample is adequately powered; binary OOS gate (dir_match) "
        "is legitimately applicable.",
    ),
    PowerTier(
        "DIRECTIONAL_ONLY", 0.50,
        "OOS can contribute directional evidence but cannot confirm/refute "
        "at alpha=0.05. dir_match is informational, NOT a hard kill.",
    ),
    PowerTier(
        "STATISTICALLY_USELESS", 0.0,
        "OOS cannot distinguish signal from noise at the IS effect size. "
        "Any dir_match outcome is noise-consistent. Verdict must be "
        "UNVERIFIED, never DEAD.",
    ),
)


def power_verdict(power: float) -> str:
    """Return the RULE 3.3 tier name for a given OOS power.

    Thresholds (canonical, per methodology failure-log 2026-04-20):
        >= 0.80 → CAN_REFUTE
        >= 0.50 → DIRECTIONAL_ONLY
        <  0.50 → STATISTICALLY_USELESS
    """
    for tier in POWER_TIERS:
        if power >= tier.min_power:
            return tier.name
    return POWER_TIERS[-1].name


def format_power_report(
    report: dict[str, float],
    *,
    label: str = "OOS power",
    indent: str = "  ",
) -> str:
    """Return a plain-text block suitable for verify-script stdout.

    Used by verify scripts to satisfy the RULE 3.3 implementation
    requirement: "every verify script with an OOS gate must print the
    power number explicitly next to the dir_match status."
    """
    tier = power_verdict(report["power"])
    lines = [
        f"{indent}{label}:",
        f"{indent}  Cohen's d (IS effect): {report['cohen_d']:.3f}",
        f"{indent}  Expected OOS SE:       {report['se_if_oos']:.4f}",
        f"{indent}  Expected 95% CI half-width: {report['ci95_half_width']:.4f}",
        f"{indent}  Power at alpha=0.05 two-sided: {report['power'] * 100:.1f}%",
        f"{indent}  N per group for 80% power: {report['n_for_80pct']:.0f}",
        f"{indent}  RULE 3.3 tier: {tier}",
    ]
    return "\n".join(lines)


__all__ = [
    "oos_ttest_power",
    "power_verdict",
    "format_power_report",
    "POWER_TIERS",
    "PowerTier",
]
