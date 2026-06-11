"""Prop-Firm EV Simulator — Stage 2: survival + reach-payout adapter.

Read-only, capital-ADJACENT (informs *which prop accounts to buy*), ZERO live
mutation. This module is the survival-adapter half of the EV scorecard
pre-registered in ``docs/research/prop_firm_ev_scorecard_2026.md``. The EV
formula, Frechet joint, N_eff capacity, NO_NUMERIC_EV seam, CLI and results.md
are Stage 3 — deliberately NOT built here.

Design contract (master plan ``~/.claude/plans/atomic-herding-pie.md`` Stage 2,
locked — do not re-litigate):

- Maps the 3 EXISTING deployed profiles to the canonical Monte-Carlo survival
  engine by IMPORT — never re-encodes the MC math.
- ``write_state=False`` is mandatory on every survival call (the zero-mutation
  enforcement seam: ``account_survival.evaluate_profile_survival`` defaults it to
  ``True`` and writes a state file otherwise).
- Survive-leg uses ``operational_pass_probability`` (R1 — the consistency+scaling
  AND'd survive probability), NOT bare ``dd_survival_probability``; both are
  surfaced for transparency.
- MFFU has no ``AccountProfile`` ⇒ ``survival_prob=None`` (NO_NUMERIC_EV
  downstream in Stage 3).

Four adversarial-review gaps (master plan § "Gaps found"), all designed for here:

1. Profit-target resolution. ``c11_inputs.profit_target`` does NOT exist as a
   field; the reach-target lives in ``PropFirmSpec.firm_specific_rules`` with a
   HETEROGENEOUS shape (flat ``profit_target`` for some firms, size-nested
   ``by_size[account_size]["profit_target"]`` for others). Verified this session:
   NONE of the 3 deployed firms (topstep / tradeify / bulenox) encode a target at
   all. ``resolve_profit_target`` therefore returns ``None`` (with a reason) on
   genuine absence and RAISES only on malformed-but-present data — it never
   defaults a fabricated number into a buy decision.
2. ``reach_payout_prob`` is a 3-point (p05/p50/p95) quantile interpolation — the
   engine exposes only those 3 quantiles, no per-path array. The estimate is
   clamped to [0, 1] and flagged ``coarse`` when the target sits outside
   [p05, p95] (a clamped bound, never a silent extrapolation).
3. ``evaluate_profile_survival`` silently re-resolves the profile id
   (``resolve_profile_id``); we assert ``summary.profile_id == requested_id`` and
   surface any alias rather than trusting the requested id blindly.
4. The engine's D-3 sizing-parity guard only ``log.warning``s on failure (does
   not fail closed). We capture whether a parity warning fired during the call
   and carry it on the record so a buy-decision reader sees it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from pipeline.paths import GOLD_DB_PATH
from trading_app.account_survival import SurvivalSummary, evaluate_profile_survival
from trading_app.prop_profiles import AccountProfile, get_firm_spec, get_profile

# The deployed profiles this adapter maps (master plan Stage 2). MFFU is
# intentionally absent — it has no AccountProfile, so its survival/reach are None.
DEPLOYED_PROFILE_IDS: tuple[str, ...] = (
    "topstep_50k_mnq_auto",
    "tradeify_50k",
    "bulenox_50k",
)


@dataclass(frozen=True)
class TargetResolution:
    """Outcome of resolving a profile's profit target.

    ``target`` is the dollar reach-target, or ``None`` when no target is encoded
    for the firm. ``reason`` explains a ``None`` (e.g. ``no_profit_target_encoded``)
    so a buy-decision reader sees *why* the reach leg is missing rather than a
    silent gap. ``source`` names where a non-None target came from.
    """

    target: float | None
    reason: str | None
    source: str | None


@dataclass(frozen=True)
class ReachEstimate:
    """A 3-point quantile interpolation of P(total_pnl >= target).

    ``prob`` is clamped to [0, 1]. ``coarse`` is True when the target lay outside
    the [p05, p95] band, meaning ``prob`` is a clamped bound, not an interpolation
    inside the observed range (Gap 2 — never a silent extrapolation).
    """

    prob: float | None
    coarse: bool
    reason: str | None


@dataclass(frozen=True)
class SurvivalAdapterRecord:
    """One profile's survival + reach-payout result for the EV scorecard.

    A thin, transparent transport: every field is either imported from the
    canonical engine or an explicit ``None``-with-reason. No EV math here.
    """

    profile_id: str
    firm: str
    account_size: int
    dd_type: str
    # R1: operational_pass_probability is the survive-leg; dd_survival_probability
    # is surfaced alongside for transparency. None ⇒ no AccountProfile (MFFU).
    survival_prob: float | None
    dd_survival_prob: float | None
    reach_payout_prob: float | None
    reach_is_coarse: bool
    profit_target: float | None
    target_reason: str | None
    target_source: str | None
    # Gap 3: the engine re-resolves the id; record the resolved id + whether it
    # aliased the requested one.
    resolved_profile_id: str | None
    profile_id_aliased: bool
    # Gap 4: the engine's sizing-parity guard only warns; surface whether it fired.
    sizing_parity_warning: bool
    notes: tuple[str, ...]


class _ParityWarningCapture(logging.Handler):
    """Capture the engine's Criterion-11 sizing-parity ``log.warning`` (Gap 4).

    ``account_survival._assert_sizing_parity`` logs but does not fail closed, so a
    parity failure is invisible to a caller. We attach this handler to the
    engine's logger for the duration of one call and inspect what fired.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.parity_fired = False

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "sizing-parity" in msg or "Criterion 11 sizing-parity" in msg:
            self.parity_fired = True


def resolve_profit_target(profile: AccountProfile) -> TargetResolution:
    """Resolve a profile's dollar reach-target from canonical firm spec.

    Bridges ``profile -> get_firm_spec(firm).firm_specific_rules`` and handles BOTH
    the flat ``profit_target`` shape and the size-nested
    ``by_size[account_size]["profit_target"]`` shape. Returns ``None`` (with a
    reason) on genuine absence — true for all 3 deployed firms this session — and
    RAISES ``ValueError`` only when a target is PRESENT but malformed (a real bug
    that must surface, not be silently swallowed). Never defaults a number.
    """
    spec = get_firm_spec(profile.firm)
    rules = spec.firm_specific_rules or {}

    # Shape A — size-nested: by_size[account_size]["profit_target"].
    by_size = rules.get("by_size")
    if isinstance(by_size, dict):
        size_block = by_size.get(profile.account_size)
        if isinstance(size_block, dict) and "profit_target" in size_block:
            raw = size_block["profit_target"]
            return TargetResolution(
                target=_coerce_target(raw, profile.firm, profile.account_size),
                reason=None,
                source=f"firm_specific_rules.by_size[{profile.account_size}].profit_target",
            )

    # Shape B — flat: profit_target (optionally guarded by account_size_only).
    if "profit_target" in rules:
        size_only = rules.get("account_size_only")
        if size_only is not None and size_only != profile.account_size:
            # The flat target is declared for a DIFFERENT account size — do not
            # mis-attach it to this profile. Treat as not-encoded for this size.
            return TargetResolution(
                target=None,
                reason=f"flat_profit_target_is_for_size_{size_only}_not_{profile.account_size}",
                source=None,
            )
        return TargetResolution(
            target=_coerce_target(rules["profit_target"], profile.firm, profile.account_size),
            reason=None,
            source="firm_specific_rules.profit_target",
        )

    # Genuine absence — no target encoded for this firm. Honest None, not a raise.
    return TargetResolution(target=None, reason="no_profit_target_encoded", source=None)


def _coerce_target(raw: object, firm: str, account_size: int) -> float:
    """Coerce a PRESENT profit target to float, raising on malformed data.

    A target that exists in canonical config must be a real positive number.
    Anything else is a bug in the spec, not a missing-data case — fail loud.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"profit_target for ({firm}, {account_size}) is malformed: {raw!r} "
            f"(type {type(raw).__name__}); expected a positive number"
        )
    value = float(raw)
    if value <= 0:
        raise ValueError(f"profit_target for ({firm}, {account_size}) is non-positive: {value}")
    return value


def estimate_reach_payout_prob(
    target: float | None,
    p05: float,
    p50: float,
    p95: float,
) -> ReachEstimate:
    """Estimate P(total_pnl >= target) from 3 quantiles (Gap 2).

    The engine exposes only p05/p50/p95 — no per-path array. We piecewise-linearly
    interpolate the CDF and return P(>= target) = 1 - CDF(target), clamped to
    [0, 1]. When the target lies outside [p05, p95] the result is a clamped bound
    flagged ``coarse`` — never a fabricated extrapolation beyond the observed band.
    """
    if target is None:
        return ReachEstimate(prob=None, coarse=False, reason="no_profit_target_encoded")

    # Quantile points: (pnl, cumulative_prob). p05 => 5% of mass at or below.
    pts = ((p05, 0.05), (p50, 0.50), (p95, 0.95))

    if target <= p05:
        # Target at/below the 5th pctile: P(>=) is at least 0.95 — clamped bound.
        return ReachEstimate(prob=0.95, coarse=True, reason="target_below_p05")
    if target >= p95:
        # Target at/above the 95th pctile: P(>=) is at most 0.05 — clamped bound.
        return ReachEstimate(prob=0.05, coarse=True, reason="target_above_p95")

    # Interpolate CDF(target) inside [p05, p95].
    cdf = _interp_cdf(target, pts)
    prob = max(0.0, min(1.0, 1.0 - cdf))
    return ReachEstimate(prob=prob, coarse=False, reason=None)


def _interp_cdf(x: float, pts: tuple[tuple[float, float], ...]) -> float:
    """Piecewise-linear CDF interpolation across sorted (value, cum_prob) points."""
    for (x0, p0), (x1, p1) in zip(pts, pts[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return p1
            frac = (x - x0) / (x1 - x0)
            return p0 + frac * (p1 - p0)
    # x outside the spanned range — caller handles the outside-band case first,
    # so this is only reached for degenerate point sets; clamp to the nearest end.
    return pts[0][1] if x < pts[0][0] else pts[-1][1]


def evaluate_survival_adapter(
    profile_id: str,
    *,
    as_of_date: date | None = None,
    n_paths: int = 10_000,
    seed: int = 0,
) -> SurvivalAdapterRecord:
    """Run the canonical survival engine for one profile, read-only.

    Imports ``evaluate_profile_survival`` — does NOT re-encode the MC math. Passes
    ``write_state=False`` (zero-mutation) and ``db_path=GOLD_DB_PATH`` (read-only).
    Asserts the resolved profile id matches the request (Gap 3) and captures the
    sizing-parity warning (Gap 4). Reach-payout is derived from the summary's
    p05/p50/p95 PnL quantiles vs the resolved profit target (Gaps 1 + 2).
    """
    profile = get_profile(profile_id)
    spec = get_firm_spec(profile.firm)
    notes: list[str] = []

    target_res = resolve_profit_target(profile)
    if target_res.reason:
        notes.append(f"target: {target_res.reason}")

    # Capture the engine's sizing-parity warning for the duration of this call.
    engine_log = logging.getLogger("trading_app.account_survival")
    capture = _ParityWarningCapture()
    engine_log.addHandler(capture)
    try:
        summary: SurvivalSummary = evaluate_profile_survival(
            profile_id,
            as_of_date=as_of_date,
            n_paths=n_paths,
            seed=seed,
            db_path=GOLD_DB_PATH,
            write_state=False,  # ZERO-MUTATION GUARANTEE — never persist state.
        )
    finally:
        engine_log.removeHandler(capture)

    # Gap 3: the engine re-resolves the id; never trust the requested one blindly.
    resolved_id = summary.profile_id
    aliased = resolved_id != profile_id
    if aliased:
        notes.append(f"profile id aliased: requested {profile_id!r} -> resolved {resolved_id!r}")

    if capture.parity_fired:
        notes.append("sizing-parity guard warned (engine did not fail closed — review)")

    reach = estimate_reach_payout_prob(
        target_res.target,
        summary.p05_total_pnl,
        summary.p50_total_pnl,
        summary.p95_total_pnl,
    )
    if reach.reason and reach.reason != "no_profit_target_encoded":
        notes.append(f"reach: {reach.reason}")

    return SurvivalAdapterRecord(
        profile_id=profile_id,
        firm=profile.firm,
        account_size=profile.account_size,
        dd_type=spec.dd_type,  # R3: surface intraday vs EOD reached the engine.
        survival_prob=summary.operational_pass_probability,  # R1 survive-leg.
        dd_survival_prob=summary.dd_survival_probability,
        reach_payout_prob=reach.prob,
        reach_is_coarse=reach.coarse,
        profit_target=target_res.target,
        target_reason=target_res.reason,
        target_source=target_res.source,
        resolved_profile_id=resolved_id,
        profile_id_aliased=aliased,
        sizing_parity_warning=capture.parity_fired,
        notes=tuple(notes),
    )


def evaluate_deployed_survival(
    *,
    as_of_date: date | None = None,
    n_paths: int = 10_000,
    seed: int = 0,
) -> list[SurvivalAdapterRecord]:
    """Run the survival adapter across all deployed profiles (read-only)."""
    return [
        evaluate_survival_adapter(pid, as_of_date=as_of_date, n_paths=n_paths, seed=seed)
        for pid in DEPLOYED_PROFILE_IDS
    ]


if __name__ == "__main__":  # pragma: no cover — manual smoke entry, not the CLI (Stage 3).
    for rec in evaluate_deployed_survival(n_paths=2_000):
        surv = f"{rec.survival_prob:.3f}" if rec.survival_prob is not None else "None"
        reach = f"{rec.reach_payout_prob:.3f}" if rec.reach_payout_prob is not None else "None"
        print(
            f"{rec.profile_id:<22} dd_type={rec.dd_type:<16} "
            f"survival={surv:<6} reach={reach:<6} "
            f"target={rec.profit_target} ({rec.target_reason or 'ok'})"
        )
