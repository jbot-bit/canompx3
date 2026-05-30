"""Powered trade-fraction out-of-sample holdout split.

DISCOVERY SAFETY: SAFE — pure function on an in-memory trade list. No DB
access, no writes, no scan-on-import. Safe to import anywhere.

WHY THIS EXISTS
---------------
The legacy Criterion-8 OOS gate (``trading_app.strategy_validator.
_evaluate_criterion_8_oos``) carves its holdout by CALENDAR date
(``trading_day >= HOLDOUT_SACRED_FROM`` = 2026-01-01). For any strategy
whose validated history ends near that boundary the calendar OOS slice is
tiny (often N < 30) and therefore statistically useless per RULE 3.3
(``.claude/rules/backtesting-methodology.md`` § 3.3) — it cannot distinguish
signal from noise, so it can neither confirm nor refute. Worse, a brand-new
strategy has ~zero calendar OOS until months of wall-clock time pass: an
artificial *deployment wait*.

This module replaces the deploy-blocking calendar wait with a **powered
temporal trade-fraction split** that exists at discovery/validation time:
hold out the most-recent fraction of a candidate's OWN pre-holdout trades,
in strict temporal order, sized so the OOS slice carries usable statistical
power. No calendar wait; the holdout exists the moment the strategy is
discovered.

MANDATORY RULES (operator-commissioned 2026-05-31)
--------------------------------------------------
- Temporal split, never random. Trades are sorted by trading_day ascending;
  the OOS slice is the most-recent contiguous tail.
- 2026+ (>= ``HOLDOUT_SACRED_FROM``) is EXCLUDED from the selection/deploy
  split. The sacred calendar holdout stays a separate read-only monitor /
  leakage sentinel; it is never the powered-OOS deploy slice and is never
  tuned against.
- Fail-closed. If the powered-OOS verdict cannot be computed (too few trades,
  degenerate variance, sub-power tail), the verdict is BLOCKED — never a
  silent NULL/PASS.
- The split fraction is NOT a free parameter to manufacture a pass. The
  fraction is fixed (default 0.30); power *informs the verdict tier* (RULE
  3.3), it does not move the boundary to rescue a candidate. (cf. the DSR
  reference-universe lesson — a free parameter swings the verdict.)
- Power math is DELEGATED to ``research.oos_power`` (canonical single source);
  this module never re-encodes a t-test or power calculation.

VERDICTS
--------
- ``PASS``     — OOS ExpR > 0, dir_match True, OOS power >= DIRECTIONAL_ONLY
                 (>= 0.50), and OOS/IS ratio >= ``MIN_OOS_IS_RATIO``.
- ``FAIL``     — computable but refuted: OOS ExpR <= 0, or dir_match False at
                 CAN_REFUTE power, or OOS/IS ratio below floor at adequate power.
- ``BLOCKED``  — cannot be computed to a deploy-grade conclusion: too few
                 trades, degenerate IS variance, or OOS power below
                 DIRECTIONAL_ONLY (statistically useless tail). Fail-closed:
                 NOT deployable, but NOT a merit refutation either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from research.oos_power import (
    one_sample_n_for_power,
    one_sample_power,
    one_sample_tstat,
    power_verdict,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# Default fraction of (temporally ordered) eligible trades held out as OOS.
# Fixed, not tuned per candidate. Operator spec: "~last 30% of trades".
DEFAULT_OOS_FRACTION = 0.30

# Minimum total eligible (pre-holdout) trades to attempt a powered split.
# Below this, neither train nor OOS can be meaningful — fail closed.
MIN_TOTAL_TRADES = 30

# OOS expectancy must retain at least this fraction of IS expectancy to PASS,
# mirroring the legacy Criterion-8 OOS/IS ratio floor (>= 0.40 * IS). Kept
# identical so the powered gate is no more permissive than the calendar gate
# it replaces on the ratio dimension.
MIN_OOS_IS_RATIO = 0.40

# Verdict labels (deploy gate). These map onto the existing validated_setups
# c8_oos_status column per the no-schema-change decision: PASS -> "PASSED",
# FAIL/BLOCKED -> a non-PASSED label the allocator's apply_c8_gate already
# demotes. Mapping is applied by the validator wiring (Stage 2), not here.
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class PoweredOOSResult:
    """Full reporting record for one powered-OOS split. All fields populated."""

    verdict: str  # PASS / FAIL / BLOCKED
    reason: str
    n_total: int
    n_train: int
    n_powered_oos: int
    split_start: date | None  # first trading_day of the OOS slice
    split_end: date | None  # last trading_day of the OOS slice
    oos_fraction: float
    estimated_power: float
    power_tier: str
    is_exp_r: float | None
    powered_oos_exp_r: float | None
    oos_is_ratio: float | None
    dir_match: bool | None
    provenance_clean: bool
    provenance_warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 for n < 2."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return var**0.5


def powered_oos_split(
    trades: list[tuple[date, float]],
    *,
    oos_fraction: float = DEFAULT_OOS_FRACTION,
    min_total_trades: int = MIN_TOTAL_TRADES,
    min_oos_is_ratio: float = MIN_OOS_IS_RATIO,
    discovery_touched_recent_fraction: bool = False,
) -> PoweredOOSResult:
    """Carve a powered temporal OOS holdout from one candidate's trades.

    Parameters
    ----------
    trades
        List of ``(trading_day, pnl_r)`` for ONE candidate/lane. Order does
        not matter on input — this function sorts strictly by ``trading_day``
        ascending before splitting. ``pnl_r`` is per-trade R-multiple.
    oos_fraction
        Fraction of eligible (pre-holdout) trades to hold out as the most
        recent temporal tail. Fixed default 0.30; exposed only for tests.
        NOT to be tuned per candidate to manufacture a pass.
    min_total_trades
        Below this many eligible trades, fail closed (BLOCKED).
    min_oos_is_ratio
        OOS ExpR must be >= this fraction of IS ExpR to PASS.
    discovery_touched_recent_fraction
        Honest provenance flag. Set True when the candidate's ORIGINAL
        discovery/selection consumed trades that now fall in the OOS tail
        (i.e. the OOS slice is not clean for THIS candidate). When True the
        verdict can still be computed but ``provenance_clean`` is False and a
        warning is attached — the split is contaminated for this candidate and
        downstream must not call it "clean OOS".

    Returns
    -------
    PoweredOOSResult
        Fully-populated record. ``verdict`` is one of PASS / FAIL / BLOCKED.
        Fail-closed: any inability to reach a deploy-grade conclusion is
        BLOCKED, never a silent pass.
    """
    n_input = len(trades)

    # --- Exclude 2026+ from the selection/deploy split (sacred holdout) -----
    # The calendar holdout is a separate read-only monitor; it is never the
    # powered-OOS deploy slice. Drop any trade on/after HOLDOUT_SACRED_FROM.
    eligible = [(d, r) for (d, r) in trades if d < HOLDOUT_SACRED_FROM]
    n_excluded_2026 = n_input - len(eligible)

    # Strict temporal order (ascending). Split is the most-recent tail.
    eligible.sort(key=lambda t: t[0])
    n_total = len(eligible)

    base_warning: str | None = None
    if n_excluded_2026 > 0:
        base_warning = (
            f"{n_excluded_2026} trade(s) on/after {HOLDOUT_SACRED_FROM} "
            f"excluded from powered split (sacred calendar holdout is "
            f"monitor-only)"
        )

    provenance_clean = not discovery_touched_recent_fraction
    if discovery_touched_recent_fraction:
        prov_warn = (
            "discovery/selection consumed trades now in the OOS tail — split "
            "is CONTAMINATED for this candidate; not clean OOS"
        )
        provenance_warning = (
            prov_warn if base_warning is None else f"{base_warning}; {prov_warn}"
        )
    else:
        provenance_warning = base_warning

    def _blocked(reason: str, **over: Any) -> PoweredOOSResult:
        fields: dict[str, Any] = dict(
            verdict=BLOCKED,
            reason=reason,
            n_total=n_total,
            n_train=0,
            n_powered_oos=0,
            split_start=None,
            split_end=None,
            oos_fraction=oos_fraction,
            estimated_power=0.0,
            power_tier="STATISTICALLY_USELESS",
            is_exp_r=None,
            powered_oos_exp_r=None,
            oos_is_ratio=None,
            dir_match=None,
            provenance_clean=provenance_clean,
            provenance_warning=provenance_warning,
        )
        fields.update(over)
        return PoweredOOSResult(**fields)

    if n_total < min_total_trades:
        return _blocked(
            f"insufficient eligible trades: n_total={n_total} < "
            f"min_total_trades={min_total_trades}"
        )

    # --- Temporal split -----------------------------------------------------
    n_oos = max(1, round(n_total * oos_fraction))
    n_train = n_total - n_oos
    if n_train < 2 or n_oos < 2:
        return _blocked(
            f"split degenerate: n_train={n_train}, n_oos={n_oos} "
            f"(need >=2 each)",
            n_train=max(n_train, 0),
            n_powered_oos=max(n_oos, 0),
        )

    train = eligible[:n_train]
    oos = eligible[n_train:]
    train_r = [r for (_, r) in train]
    oos_r = [r for (_, r) in oos]
    split_start = oos[0][0]
    split_end = oos[-1][0]

    is_exp_r = _mean(train_r)
    oos_exp_r = _mean(oos_r)
    is_std = _std(train_r)

    # --- Power of the OOS slice at the IS effect size (RULE 3.3) ------------
    # One-sample framing: pooled per-trade pnl_r tested via t = mean*sqrt(N)/std.
    # Cohen's d = |t_IS| / sqrt(N_IS) = |IS_mean| / IS_std. Delegated power calc.
    if is_std <= 0:
        return _blocked(
            f"degenerate IS variance (std={is_std}); cannot size OOS power",
            n_train=n_train,
            n_powered_oos=n_oos,
            split_start=split_start,
            split_end=split_end,
            is_exp_r=is_exp_r,
            powered_oos_exp_r=oos_exp_r,
        )

    cohen_d = abs(is_exp_r) / is_std
    estimated_power = one_sample_power(cohen_d, n_oos)
    tier = power_verdict(estimated_power)

    oos_is_ratio = (oos_exp_r / is_exp_r) if is_exp_r not in (0, None) else None
    # dir_match: OOS expectancy points the same direction as IS expectancy.
    dir_match = (oos_exp_r > 0) == (is_exp_r > 0)

    common: dict[str, Any] = dict(
        n_total=n_total,
        n_train=n_train,
        n_powered_oos=n_oos,
        split_start=split_start,
        split_end=split_end,
        oos_fraction=oos_fraction,
        estimated_power=estimated_power,
        power_tier=tier,
        is_exp_r=is_exp_r,
        powered_oos_exp_r=oos_exp_r,
        oos_is_ratio=oos_is_ratio,
        dir_match=dir_match,
        provenance_clean=provenance_clean,
        provenance_warning=provenance_warning,
    )

    # --- Verdict (fail-closed) ----------------------------------------------
    # STATISTICALLY_USELESS tail (power < 0.50): cannot reach a deploy-grade
    # conclusion. BLOCKED, never PASS. Suggest the N needed for usable power.
    if tier == "STATISTICALLY_USELESS":
        n_needed = one_sample_n_for_power(cohen_d, target=0.50)
        return PoweredOOSResult(
            verdict=BLOCKED,
            reason=(
                f"OOS power {estimated_power:.2f} < 0.50 (STATISTICALLY_USELESS); "
                f"need ~{n_needed} OOS trades at d={cohen_d:.3f}. Not deployable, "
                f"not refuted."
            ),
            **common,
        )

    # Refutations (computable at adequate power):
    if oos_exp_r <= 0:
        return PoweredOOSResult(
            verdict=FAIL,
            reason=f"OOS ExpR {oos_exp_r:.4f} <= 0 (sign collapse out of sample)",
            **common,
        )
    if not dir_match and tier == "CAN_REFUTE":
        return PoweredOOSResult(
            verdict=FAIL,
            reason=(
                f"dir_match FALSE at CAN_REFUTE power {estimated_power:.2f} "
                f"(IS {is_exp_r:.4f} vs OOS {oos_exp_r:.4f})"
            ),
            **common,
        )
    if oos_is_ratio is not None and oos_is_ratio < min_oos_is_ratio:
        return PoweredOOSResult(
            verdict=FAIL,
            reason=(
                f"OOS/IS ratio {oos_is_ratio:.2f} < floor {min_oos_is_ratio} "
                f"(OOS {oos_exp_r:.4f} / IS {is_exp_r:.4f})"
            ),
            **common,
        )

    # PASS: positive OOS ExpR, dir_match, ratio floor cleared, power >= 0.50.
    return PoweredOOSResult(
        verdict=PASS,
        reason=(
            f"powered-OOS PASS: OOS ExpR {oos_exp_r:.4f}, ratio "
            f"{oos_is_ratio:.2f}, power {estimated_power:.2f} ({tier})"
        ),
        **common,
    )
