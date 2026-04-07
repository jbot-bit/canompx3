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
"""

from __future__ import annotations

from datetime import UTC, date, datetime

# Sacred holdout window start (Amendment 2.7 / Mode A).
# Discovery may not use trading_day >= this value.
HOLDOUT_SACRED_FROM: date = date(2026, 1, 1)

# Amendment 2.7 commit moment. Rows in ``experimental_strategies`` with
# ``created_at`` at or before this moment are grandfathered (known
# pre-correction contamination from the Apr 3 Mode B deviation). Rows created
# after this moment are subject to enforcement.
HOLDOUT_GRANDFATHER_CUTOFF: datetime = datetime(2026, 4, 8, 0, 0, 0, tzinfo=UTC)


def enforce_holdout_date(holdout_date: date | None) -> date:
    """Validate a ``--holdout-date`` CLI argument against Mode A.

    Parameters
    ----------
    holdout_date
        The ``--holdout-date`` value supplied by the caller, or ``None`` if
        the flag was omitted.

    Returns
    -------
    date
        The effective holdout cutoff to use for discovery. Always a real
        ``date`` (never ``None``). Guaranteed to be ``<= HOLDOUT_SACRED_FROM``.

    Raises
    ------
    ValueError
        If ``holdout_date`` is strictly greater than ``HOLDOUT_SACRED_FROM``.
        The error message cites Amendment 2.7 and the canonical source so the
        caller can quickly locate the policy.

    Notes
    -----
    ``holdout_date == HOLDOUT_SACRED_FROM`` is the expected common case under
    Mode A. Earlier holdout dates are also permitted (e.g., for auditing an
    older era) — only post-sacred dates are rejected.
    """
    if holdout_date is None:
        return HOLDOUT_SACRED_FROM
    if holdout_date > HOLDOUT_SACRED_FROM:
        raise ValueError(
            f"--holdout-date {holdout_date.isoformat()} violates Mode A "
            f"(pre_registered_criteria.md Amendment 2.7, 2026-04-08). "
            f"The sacred holdout window begins {HOLDOUT_SACRED_FROM.isoformat()}; "
            f"discovery must not use data from this window or later. "
            f"Use --holdout-date {HOLDOUT_SACRED_FROM.isoformat()} (or earlier) "
            f"or omit the flag to accept the sacred-from default. "
            f"Canonical source: trading_app.holdout_policy"
        )
    return holdout_date


__all__ = [
    "HOLDOUT_SACRED_FROM",
    "HOLDOUT_GRANDFATHER_CUTOFF",
    "enforce_holdout_date",
]
