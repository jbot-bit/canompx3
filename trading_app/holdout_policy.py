"""Sacred holdout policy — single source of truth for Mode A (Amendment 2.7).

This module is the CANONICAL source for the project's holdout policy constants
and enforcement helpers. All downstream consumers (``pipeline.check_drift``,
``trading_app.strategy_discovery``, ``trading_app.strategy_validator``, and any
future drift checks) MUST import from here rather than inlining the constants.

Authority chain
---------------

- **Project policy:** ``docs/institutional/pre_registered_criteria.md``
  Amendment 2.7 (2026-04-08) — Mode A (holdout-clean) operative.
- **Decision doc:** ``docs/plans/2026-04-07-holdout-policy-decision.md``
  (top-of-file RESCINDED header marks Mode B as corrected).
- **Research rules:** ``RESEARCH_RULES.md`` § "2026 holdout is sacred" rule.
- **Original discipline:** ``docs/plans/2026-04-02-16yr-pipeline-rebuild.md``
  L79-83 — *"CRITICAL: --holdout-date 2026-01-01 protects sacred holdout"*.

Semantics
---------

Two distinct dates are needed, and the pre-refactor ``HOLDOUT_DECLARATIONS``
dict conflated them:

- **HOLDOUT_SACRED_FROM** — the start of the sacred window. Data with
  ``trading_day >= HOLDOUT_SACRED_FROM`` is protected OOS. Discovery must NOT
  use it. Live trading and paper-forward monitoring MAY use it. Under Mode A
  this is ``2026-01-01``.
- **HOLDOUT_GRANDFATHER_CUTOFF** — the moment Amendment 2.7 was committed.
  Any ``experimental_strategies`` row with ``created_at`` at or before this
  moment is grandfathered (known pre-correction contamination from the Apr 3
  Mode B deviation). Rows created AFTER this moment are subject to
  enforcement. Under Amendment 2.7 this is ``2026-04-08 00:00:00+00:00``.

Conflating these produces wrong enforcement: if you use
``sacred_from`` as the grandfather cutoff, every existing experimental row
matches (they were all created in April, after January) and the drift check
flags them as NEW violations even though they are known historical
contamination. If you use ``grandfather_cutoff`` as the sacred boundary, you
permit 2026 Q1 to leak into new discoveries.

The ``HOLDOUT_DECLARATIONS`` dict in ``pipeline.check_drift`` used
``grandfather_cutoff`` (Apr 8) for both purposes, which happened to produce the
correct behavior for the current corpus because ``sacred_from`` was
``2026-01-01`` and all experimental rows were created in April. The refactor
here makes the two concepts explicit so future policies (e.g., a new 2027
holdout with a different grandfather moment) do not inherit the conflation.

Enforcement helper
------------------

``enforce_holdout_date(holdout_date)`` is the CLI-side gate for the
``--holdout-date`` argument on ``strategy_discovery.py``. It implements
Amendment 2.7 Rule: any ``holdout_date`` later than ``HOLDOUT_SACRED_FROM``
means the discovery would touch sacred data, which is banned. ``None`` is
silently upgraded to ``HOLDOUT_SACRED_FROM`` so that running discovery with
no explicit flag still respects Mode A.

EARLY_HOLDOUT_REDISCOVERY (EHR) probe-mode constants
----------------------------------------------------

``EARLY_HOLDOUT_BOUNDARY`` is a SEPARATE probe-mode constant for the
``EARLY_HOLDOUT_REDISCOVERY`` validation mode (PASS 2, 2026-05-17). It is NOT
an alias, replacement, or weakening of ``HOLDOUT_SACRED_FROM`` — Mode A
sacredness is invariant under this module. EHR is a research/probe mode that
runs discovery on ``trading_day < EARLY_HOLDOUT_BOUNDARY`` and labels
``EARLY_HOLDOUT_BOUNDARY <= trading_day < HOLDOUT_SACRED_FROM`` as
``PSEUDO_OOS_ROBUSTNESS`` evidence only. EHR survivors are NEVER deployable
(verdict ceiling ``RESEARCH_PROVISIONAL``); the true forward OOS window
(``trading_day >= HOLDOUT_SACRED_FROM``, still accumulating) remains the only
path to ``CLEAN_OOS`` and capital deployment.

Authority: ``EARLY_HOLDOUT_REDISCOVERY — Narrow Guarded PASS 2 Plan`` dated
2026-05-17, invariants 1-6 (top-of-plan). The plan is grounded in
``docs/institutional/literature/`` extracts that explicitly oppose relabeling
previously-experienced data as ``CLEAN_OOS`` (Chan 2013 Ch1 p.4; Harvey-Liu
2015 pp.15-17; Bailey-Lopez de Prado 2014 pp.2-3; Harris 2002 pp.471-472).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from pipeline.log import get_logger

logger = get_logger(__name__)

# Sacred holdout window start (Amendment 2.7 / Mode A).
# Discovery may not use trading_day >= this value.
HOLDOUT_SACRED_FROM: date = date(2026, 1, 1)

# Amendment 2.7 commit moment. Rows in ``experimental_strategies`` with
# ``created_at`` at or before this moment are grandfathered (known
# pre-correction contamination from the Apr 3 Mode B deviation). Rows created
# after this moment are subject to enforcement.
HOLDOUT_GRANDFATHER_CUTOFF: datetime = datetime(2026, 4, 8, 0, 0, 0, tzinfo=UTC)

# Phase 4 Stage 4.1 ship moment — the instant the discovery-side hypothesis
# file integration becomes active. Used by drift check #94
# (``check_phase_4_sha_integrity``) as the cutoff for SHA integrity
# enforcement: rows written with ``created_at >= PHASE_4_1_SHIP_DATE`` that
# carry a non-null ``hypothesis_file_sha`` must reference a real file in
# ``docs/audit/hypotheses/``.
#
# Deliberately set to the day AFTER ``HOLDOUT_GRANDFATHER_CUTOFF`` (2026-04-09
# 00:00 UTC = 2026-04-09 10:00 Brisbane) so that rows created during Stage 4.1
# implementation and testing on 2026-04-08 itself are NOT retroactively
# subject to the integrity check. This is the operational grace period for
# the stage to land cleanly.
#
# Distinct from ``HOLDOUT_GRANDFATHER_CUTOFF`` (which gates the VALIDATOR-side
# Phase 4 pre-flight gates per Stage 4.0): this constant gates the DRIFT
# CHECK integrity assertion on the DISCOVERY-side SHA stamping added in Stage
# 4.1. The two cutoffs are one day apart to give Stage 4.1 a grace window.
PHASE_4_1_SHIP_DATE: datetime = datetime(2026, 4, 9, 0, 0, 0, tzinfo=UTC)

# EARLY_HOLDOUT_REDISCOVERY (EHR) probe-mode boundary — PASS 2 plan, 2026-05-17.
#
# NOT an alias of HOLDOUT_SACRED_FROM. EHR is a research/probe validation mode
# that runs discovery on data ending before EARLY_HOLDOUT_BOUNDARY (so the
# 2025 calendar year becomes a PSEUDO-OOS measurement window) WITHOUT
# touching the sacred 2026+ window. Mode A discovery and capital deployment
# continue to be gated by HOLDOUT_SACRED_FROM exclusively.
#
# The plan locks this constant at 2025-01-01 because: (a) a 5-year EHR IS
# window (2020-01-01 → 2024-12-31) gives MinBTL ≤ 45 trials for MNQ/MES per
# Bailey 2013 Fig 2, which is 3.75× the 12-trial Mode A family; (b) the
# resulting 2025 PSEUDO-OOS spans ~250 trading days, sufficient for OOS power
# computation per backtesting-methodology.md § RULE 3.3; and (c) the
# alternative 2024-01-01 boundary would shrink MinBTL to ~30 trials while
# doubling regime overlap with original Mode A discovery — worse on both
# axes. Changing this date requires a NEW pre-reg file (it cannot be tuned
# after results are seen — anti-Chan 2013 Ch1 p.4).
EARLY_HOLDOUT_BOUNDARY: date = date(2025, 1, 1)

# String literal identifying the EHR validation mode. Used as the
# ``validation_mode`` value written to ``validated_setups`` rows discovered
# under EHR, and as the input to ``is_ehr_mode()``. Deliberately verbose:
# short names ("EHR") get casually misapplied; the literal embeds the
# semantics directly in the data layer.
EHR_MODE_LABEL: str = "EARLY_HOLDOUT_REDISCOVERY"

# Default validation_mode for all non-EHR discovery (Mode A and earlier
# research-provisional). Schema column default in Stage 2 of the PASS 2 plan.
STANDARD_MODE_LABEL: str = "STANDARD"

# Hard-gate override token (added 2026-04-08 per explicit user instruction
# "we need to ensure strictly x 100000 that discovery NEVER RUNS 2026 EVER
# UNLESS WE FUCKING TYPE THE PASSWORD NUMBER 3656"). The token is a SPEED BUMP
# against accidental contamination, not cryptographic security — anyone reading
# this source can find it. The audit trail (loud warning logging + drift checks)
# is the real defense. Override invocations destroy OOS validity for any
# experimental_strategies row they produce; treat any strategy created with the
# override as research-provisional, not deployment-eligible.
HOLDOUT_OVERRIDE_TOKEN: str = "3656"


def enforce_holdout_date(
    holdout_date: date | None,
    override_token: str | None = None,
) -> date:
    """Validate a ``--holdout-date`` argument against Mode A — function-level gate.

    This is the SINGLE chokepoint for holdout enforcement across the project.
    Every discovery / validator path that takes a ``holdout_date`` parameter
    MUST call this function at its top to prevent accidental sacred-window
    contamination. The function is called from:

    - ``trading_app.strategy_discovery.run_discovery`` (function-level gate)
    - ``trading_app.strategy_discovery.main`` (CLI gate, defense in depth)
    - ``trading_app.nested.discovery.run_nested_discovery``
    - ``trading_app.regime.discovery.run_regime_discovery``
    - ``trading_app.strategy_validator`` (Mode A integrity gate at promotion)

    Parameters
    ----------
    holdout_date
        The holdout cutoff supplied by the caller, or ``None`` if omitted.
    override_token
        Optional token for explicit override. If equal to
        ``HOLDOUT_OVERRIDE_TOKEN`` ("3656"), post-sacred holdout dates are
        accepted with a LOUD WARNING log. Any other value (including ``None``)
        causes post-sacred dates to raise. Default ``None`` = strict Mode A.

    Returns
    -------
    date
        The effective holdout cutoff to use for discovery. Always a real
        ``date`` (never ``None``). Guaranteed to be ``<= HOLDOUT_SACRED_FROM``
        UNLESS a valid override token was supplied.

    Raises
    ------
    ValueError
        If ``holdout_date`` is strictly greater than ``HOLDOUT_SACRED_FROM``
        AND no valid override token was supplied. The error message cites
        Amendment 2.7 and the canonical source.

    Notes
    -----
    ``holdout_date == HOLDOUT_SACRED_FROM`` is the expected common case under
    Mode A. Earlier holdout dates are also permitted (e.g., for auditing an
    older era).

    The override mechanism exists for legitimate research that needs to peek
    past the sacred boundary (e.g., post-validation OOS sanity checks). Any
    use of the override:

    1. Is logged loudly via ``logger.warning`` with the override date and token
    2. Should be documented in ``docs/audit/`` with justification
    3. Destroys the OOS-clean property of any strategy it produces — those
       strategies must be marked research-provisional, not deployable

    The token value is NOT a secret. It is a speed bump to prevent accidental
    contamination. The defense is the audit trail, not cryptography.
    """
    if holdout_date is None:
        return HOLDOUT_SACRED_FROM
    if holdout_date > HOLDOUT_SACRED_FROM:
        if override_token == HOLDOUT_OVERRIDE_TOKEN:
            logger.warning(
                "HOLDOUT OVERRIDE INVOKED: holdout_date=%s exceeds sacred boundary "
                "%s. Override token verified. ANY STRATEGY DISCOVERED WITH THIS "
                "OVERRIDE IS RESEARCH-PROVISIONAL — its OOS-clean property is "
                "destroyed and it MUST NOT be promoted to deployment without "
                "explicit re-validation against a fresh, untouched holdout window. "
                "Authority: docs/institutional/pre_registered_criteria.md "
                "Amendment 2.7. Canonical source: trading_app.holdout_policy",
                holdout_date.isoformat(),
                HOLDOUT_SACRED_FROM.isoformat(),
            )
            return holdout_date
        raise ValueError(
            f"--holdout-date {holdout_date.isoformat()} violates Mode A "
            f"(pre_registered_criteria.md Amendment 2.7, 2026-04-08). "
            f"The sacred holdout window begins {HOLDOUT_SACRED_FROM.isoformat()}; "
            f"discovery must not use data from this window or later. "
            f"Use --holdout-date {HOLDOUT_SACRED_FROM.isoformat()} (or earlier) "
            f"or omit the flag to accept the sacred-from default. "
            f"To explicitly override (research only, destroys OOS validity), "
            f"pass override_token=HOLDOUT_OVERRIDE_TOKEN (CLI: --unlock-holdout TOKEN). "
            f"Canonical source: trading_app.holdout_policy"
        )
    return holdout_date


def is_ehr_mode(mode: str | None) -> bool:
    """Return True iff ``mode`` is the exact EHR validation-mode literal.

    Case-sensitive and strict — no normalization, no aliases. EHR survivors
    are non-deployable per plan invariant #2; callers that branch on
    ``is_ehr_mode(row["validation_mode"])`` to block lane writes (Stage 5)
    or cap verdicts at ``RESEARCH_PROVISIONAL`` (Stage 3) MUST get a stable,
    exact predicate. Soft matching (``.upper()``, ``in`` lookup) would let a
    typo like ``"early_holdout_rediscovery"`` silently fall through to the
    STANDARD path and bypass the EHR guards.

    Parameters
    ----------
    mode
        Value from ``validated_setups.validation_mode`` (or any caller
        passing a mode label). ``None`` is the legitimate pre-Stage-2
        default for rows that predate the schema migration.

    Returns
    -------
    bool
        ``True`` only when ``mode == EHR_MODE_LABEL`` exactly. All other
        inputs (including ``None``, empty string, lower-case variants,
        STANDARD, future labels) return ``False``.
    """
    return mode == EHR_MODE_LABEL


def enforce_early_holdout_date(holdout_date: date | None) -> date:
    """Validate a ``--early-holdout-date`` argument for EHR discovery.

    Parallel to ``enforce_holdout_date()`` but gated on
    ``EARLY_HOLDOUT_BOUNDARY`` instead of ``HOLDOUT_SACRED_FROM``. Used by
    EHR discovery paths added in Stage 4. Mode A discovery continues to
    call ``enforce_holdout_date()`` unchanged.

    There is deliberately NO override token here. The plan's invariant #1
    forbids any softening of the boundary; if a future need to peek past
    2025-01-01 inside EHR mode emerges, it requires a new amendment to
    the PASS 2 plan and a separate pre-reg, not a runtime escape hatch.

    Parameters
    ----------
    holdout_date
        The EHR holdout cutoff supplied by the caller, or ``None`` if
        omitted. ``None`` is silently upgraded to ``EARLY_HOLDOUT_BOUNDARY``
        so an EHR discovery run with no explicit flag still respects the
        2025-01-01 boundary.

    Returns
    -------
    date
        The effective EHR holdout cutoff. Always a real ``date``,
        guaranteed ``<= EARLY_HOLDOUT_BOUNDARY``.

    Raises
    ------
    ValueError
        If ``holdout_date > EARLY_HOLDOUT_BOUNDARY``. The error cites the
        PASS 2 plan and reminds the caller that EHR cannot be tuned by
        moving the boundary after results are seen.
    """
    if holdout_date is None:
        return EARLY_HOLDOUT_BOUNDARY
    if holdout_date > EARLY_HOLDOUT_BOUNDARY:
        raise ValueError(
            f"--early-holdout-date {holdout_date.isoformat()} violates the "
            f"EARLY_HOLDOUT_REDISCOVERY boundary "
            f"({EARLY_HOLDOUT_BOUNDARY.isoformat()}). EHR discovery must use "
            f"trading_day < {EARLY_HOLDOUT_BOUNDARY.isoformat()}; the 2025 "
            f"calendar year is reserved for PSEUDO-OOS measurement. "
            f"Moving the boundary after seeing results is forbidden "
            f"(plan invariant #1, Chan 2013 Ch1 p.4). A new boundary "
            f"requires a separate pre-reg and a new amendment to the "
            f"PASS 2 plan. Canonical source: trading_app.holdout_policy"
        )
    return holdout_date


__all__ = [
    "HOLDOUT_SACRED_FROM",
    "HOLDOUT_GRANDFATHER_CUTOFF",
    "HOLDOUT_OVERRIDE_TOKEN",
    "PHASE_4_1_SHIP_DATE",
    "EARLY_HOLDOUT_BOUNDARY",
    "EHR_MODE_LABEL",
    "STANDARD_MODE_LABEL",
    "enforce_holdout_date",
    "enforce_early_holdout_date",
    "is_ehr_mode",
]
