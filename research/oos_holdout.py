"""Powered out-of-sample holdout — carve OOS by trade-fraction, sized for power.

Doctrine: ``feedback_powered_oos_holdout_at_discovery_no_calendar_wait_2026_05_29``.

At discovery time we hold out a POWERED out-of-sample block by *trade fraction*
(most-recent trades, temporal order), NOT by calendar date. A low-frequency
session's calendar-OOS slice is permanently underpowered; the only way to grow
it is waiting real calendar time — a banned posture. A fraction holdout gives a
usable OOS immediately and still tests forward-generalization / decay because
the reserved block is the most-recent trades.

This module is the SINGLE canonical source for the fraction-split policy. The
power math is delegated to ``research.oos_power`` (one-sample framing, matching
runners that emit a one-sample t on pooled ``pnl_r``). Re-encoding either the
split arithmetic or the power calc in another script is a canonical-delegation
violation per ``.claude/rules/institutional-rigor.md`` § 4 — import this
instead.

Sizing policy (operator-confirmed 2026-05-31):
  - The reserved OOS block is the most-recent ``default_fraction`` of trades.
  - If that default block already reaches ``target_tier``, keep it (do NOT
    shrink OOS to win back IS trades — the default fraction is the contract).
  - Only GROW the OOS block beyond ``default_fraction`` when the default is
    underpowered, and only up to the point where IS still retains
    ``min_is`` trades.
  - If even the maximal admissible OOS cannot reach ``target_tier``, return the
    best achievable split flagged ``STATISTICALLY_USELESS``. The helper reports
    truth; it does NOT decide deployability. The caller emits the
    ``UNVERIFIED_INSUFFICIENT_POWER`` verdict (doctrine § 3) — never "wait".

The 2026-01-01 SACRED calendar holdout (``trading_app.holdout_policy``) still
stands for leakage protection; this fraction slice sits *beside* it inside the
pre-2026 IS region. It does not override Amendment 2.7.

Literature grounding and SCOPE LIMIT (read before claiming this is "the"
institutional OOS gate):
  - Harvey & Liu 2015 (``literature/harvey_liu_2015_backtesting.md`` p.17): an
    underpowered OOS must NOT binary-veto IS multiple-testing evidence — OOS is
    corroboration, not a kill. That is what the power floor (RULE 3.3) and this
    helper's "report truth, never wait" posture honor. GROUNDED.
  - López de Prado 2018 AFML § 12.2 (``literature/lopez_de_prado_2018_afml_ch_3_7_8.md``):
    a single train-then-trailing-OOS split — EXACTLY what this helper produces —
    is named WF pitfall #1: "a single scenario is tested (the historical path),
    which can be easily overfit." This helper fixes WF pitfall #3 (the small
    initial OOS sample) by sizing the cut for power; it does NOT fix pitfall #1.
  - The literature's remedy for short / single-path OOS is **CPCV** (AFML
    § 12.4 — multiple combinatorial purged paths), NOT a different single-path
    cut. ``powered_oos_split`` is therefore a BETTER single-path OOS, not a
    substitute for CPCV, and not a substitute for Criterion 8 forward-OOS, which
    ``docs/institutional/pre_registered_criteria.md`` (Criterion 8 / Amendment
    3.5) keeps REQUIRED for deployment and explicitly does NOT demote. A
    ``reaches_target`` result is research-validation evidence on clean history,
    never a deployment clearance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from research.oos_power import (
    one_sample_n_for_power,
    one_sample_power,
    power_verdict,
)

# Default OOS reservation per the doctrine. The achieved power is what gates a
# verdict, not the fraction — but 0.30 is the declared default block size.
DEFAULT_OOS_FRACTION = 0.30

# Power tiers, lowest-acceptable first, for resolving a target-tier name to its
# floor. Mirrors research.oos_power.POWER_TIERS thresholds (delegated, not
# re-encoded — we read the floor via one_sample power_verdict semantics).
_TIER_FLOORS = {
    "CAN_REFUTE": 0.80,
    "DIRECTIONAL_ONLY": 0.50,
    "STATISTICALLY_USELESS": 0.0,
}


@dataclass(frozen=True)
class PoweredOOSSplit:
    """Result of a powered trade-fraction OOS carve.

    ``is_idx`` / ``oos_idx`` are positional indices into the temporally-ordered
    input series (0-based, contiguous: IS is the leading block, OOS the trailing
    block). The caller is responsible for having sorted the series by
    ``trading_day`` BEFORE calling — this helper does not reorder.
    """

    is_idx: np.ndarray
    oos_idx: np.ndarray
    requested_fraction: float
    achieved_fraction: float
    achieved_power: float
    tier: str
    is_cohen_d: float
    n_for_target: int | None
    reaches_target: bool


def _one_sample_t(x: np.ndarray) -> tuple[int, float, float]:
    """Return (n, mean, one-sample t) for a return block.

    t = mean / (std/sqrt(n)); std is sample std (ddof=1). Returns t=nan when
    N<2 and t=0.0 when the block has zero variance (degenerate, no signal).
    """
    n = int(x.size)
    if n < 2:
        return n, (float(x.mean()) if n else float("nan")), float("nan")
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return n, m, t


def _cohen_d(t_is: float, n_is: int) -> float:
    """One-sample Cohen's d from the IS t-stat: d = |t_is| / sqrt(N_is).

    The IS std cancels out of the t/d ratio, so d is recoverable from t and N
    alone — matching research.oos_power's RULE 3.3 one-sample convention.
    """
    if n_is < 2 or not np.isfinite(t_is):
        return 0.0
    return abs(t_is) / np.sqrt(n_is)


def powered_oos_split(
    returns: Sequence[float],
    *,
    target_tier: str = "DIRECTIONAL_ONLY",
    default_fraction: float = DEFAULT_OOS_FRACTION,
    min_is: int = 2,
    min_oos: int = 2,
    alpha: float = 0.05,
) -> PoweredOOSSplit:
    """Carve a powered trade-fraction OOS block from a temporally-ordered series.

    Parameters
    ----------
    returns:
        Per-trade returns (``pnl_r``) in TEMPORAL ORDER (oldest first). Caller
        must sort by ``trading_day`` before passing — NaNs must be dropped.
    target_tier:
        Power tier the OOS block should reach: ``CAN_REFUTE`` (>=0.80),
        ``DIRECTIONAL_ONLY`` (>=0.50, default), or ``STATISTICALLY_USELESS``.
    default_fraction:
        Default trailing fraction reserved as OOS. Kept as-is when it already
        reaches ``target_tier``; grown only when underpowered.
    min_is, min_oos:
        Floors — IS retains >= ``min_is`` trades; OOS holds >= ``min_oos``.
    alpha:
        Two-sided significance for the power computation.

    Returns
    -------
    PoweredOOSSplit
        Index arrays plus achieved power/tier/effect-size diagnostics. Never
        raises on ordinary thin/degenerate data — returns a degenerate split
        flagged ``STATISTICALLY_USELESS`` with ``reaches_target=False`` instead.
    """
    if target_tier not in _TIER_FLOORS:
        raise ValueError(
            f"unknown target_tier {target_tier!r}; "
            f"expected one of {sorted(_TIER_FLOORS)}"
        )
    target_floor = _TIER_FLOORS[target_tier]

    r = np.asarray(returns, dtype=float)
    n_full = int(r.size)

    # Helper to build a degenerate (all-IS-fallback) result without raising.
    def _degenerate(reason_idx_oos: np.ndarray, reason_idx_is: np.ndarray) -> PoweredOOSSplit:
        no = int(reason_idx_oos.size)
        return PoweredOOSSplit(
            is_idx=reason_idx_is,
            oos_idx=reason_idx_oos,
            requested_fraction=default_fraction,
            achieved_fraction=(no / n_full) if n_full else 0.0,
            achieved_power=0.0,
            tier=power_verdict(0.0),
            is_cohen_d=0.0,
            n_for_target=None,
            reaches_target=False,
        )

    # Too thin to form both blocks at the configured floors → no OOS.
    if n_full < (min_is + min_oos):
        return _degenerate(
            np.array([], dtype=int), np.arange(n_full, dtype=int)
        )

    # IS effect size from the FULL-history one-sample t. Using the full series
    # for the d estimate (rather than a moving IS block) keeps d stable as we
    # search OOS sizes — d is the property of the edge, the split only changes
    # how many trades land in OOS.
    _, _, t_full = _one_sample_t(r)
    d = _cohen_d(t_full, n_full)
    n_for_target = (
        one_sample_n_for_power(d, target=max(target_floor, 1e-9), alpha=alpha)
        if d > 0 and target_floor > 0
        else None
    )

    def _split_at(n_oos: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Build (is_idx, oos_idx, oos_power) reserving the last n_oos trades."""
        k = n_full - n_oos  # IS count
        is_idx = np.arange(k, dtype=int)
        oos_idx = np.arange(k, n_full, dtype=int)
        pw = one_sample_power(d, n_oos, alpha=alpha) if n_oos >= 2 else 0.0
        return is_idx, oos_idx, pw

    # Maximal admissible OOS: leave at least min_is in IS.
    max_oos = n_full - min_is
    # Default block size. Convention: the IS block is the LEADING
    # int(n_full * (1 - fraction)) trades, so OOS = n_full - that. This is the
    # canonical boundary (matches the long-standing graveyard-resweep cut) and
    # is equivalent to ceil(n_full * fraction) when n_full*fraction is
    # fractional. Floored at min_oos, capped at max_oos.
    default_oos = n_full - int(n_full * (1.0 - default_fraction))
    default_oos = max(min_oos, default_oos)
    default_oos = min(default_oos, max_oos)

    is_idx, oos_idx, pw = _split_at(default_oos)
    chosen_oos = default_oos

    # Sizing policy: keep the default block if it already clears the target.
    # Only grow (never shrink) when underpowered, up to max_oos.
    if pw < target_floor and target_floor > 0:
        grown_oos = default_oos
        for n_oos in range(default_oos + 1, max_oos + 1):
            gi, go, gp = _split_at(n_oos)
            if gp >= target_floor:
                is_idx, oos_idx, pw, grown_oos = gi, go, gp, n_oos
                break
            # Track the best (largest-power) as fallback even if target unmet.
            is_idx, oos_idx, pw, grown_oos = gi, go, gp, n_oos
        chosen_oos = grown_oos

    return PoweredOOSSplit(
        is_idx=is_idx,
        oos_idx=oos_idx,
        requested_fraction=default_fraction,
        achieved_fraction=chosen_oos / n_full,
        achieved_power=float(pw),
        tier=power_verdict(float(pw)),
        is_cohen_d=float(d),
        n_for_target=n_for_target,
        reaches_target=bool(pw >= target_floor and target_floor > 0),
    )
