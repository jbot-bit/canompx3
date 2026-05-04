"""Immutable data types for filter eligibility reports.

Nine explicit `ConditionStatus` values replace the naive PASS/FAIL/PENDING
trichotomy. Each status represents a distinct way a filter can be in a
non-determinate state — conflating any of them with PASS or FAIL is exactly
what creates silent failures in the v1 design.

Design rationale: docs/plans/2026-04-07-eligibility-context-design.md §
"Silent Failure Audits" and "Nine explicit statuses, never silent defaults"

Grounding: Aronson Evidence-Based Technical Analysis Ch.6 warns that hiding
the test universe inflates apparent edge (confirmation bias). The
NOT_APPLICABLE_* variants ensure filters validated NOT-to-work for a given
instrument/session/direction remain visible rather than quietly absent.
Cited from training memory; PDF at resources/Evidence_Based_Technical_Analysis_Aronson.pdf
not extracted in this session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class ConditionStatus(str, Enum):
    """Explicit status for an atomic filter/overlay condition.

    Never use a boolean — every condition must resolve to one of these nine
    values. "I don't know yet" (PENDING) is distinct from "I can't know because
    the data is missing" (DATA_MISSING) is distinct from "this filter was
    validated to not apply here" (NOT_APPLICABLE_*).
    """

    PASS = "PASS"
    """Pre-session or resolvable check: condition met today."""

    FAIL = "FAIL"
    """Pre-session or resolvable check: condition not met today."""

    PENDING = "PENDING"
    """Intra-session condition that cannot resolve until the session runs
    (e.g., ORB size threshold — ORB must form first; break delay — break must
    occur first; cost ratio — needs actual ORB size)."""

    DATA_MISSING = "DATA_MISSING"
    """Required feature column is NULL or the daily_features row is absent for
    the trading day. Distinct from FAIL — surfaces ingestion/infrastructure
    problems rather than hiding them as false negatives."""

    NOT_APPLICABLE_INSTRUMENT = "NOT_APPLICABLE_INSTRUMENT"
    """Filter was tested on this instrument and failed validation (e.g., FAST5
    is MNQ-specific; for MGC and MES, the signal did not survive T1-T8).
    Distinct from FAIL — the filter is not a gate on this instrument at all."""

    NOT_APPLICABLE_ENTRY_MODEL = "NOT_APPLICABLE_ENTRY_MODEL"
    """Filter is look-ahead unsafe for this entry model (e.g., break bar
    continuation is known only after the bar closes, so it cannot gate an
    E2 stop-market entry that fires on first touch). Distinct from
    NOT_APPLICABLE_INSTRUMENT — the reason is execution mechanics, not
    per-instrument research validation. Canonical exclusion list is in
    trading_app.config.E2_EXCLUDED_FILTER_PREFIXES / _SUBSTRINGS."""

    NOT_APPLICABLE_DIRECTION = "NOT_APPLICABLE_DIRECTION"
    """Direction-conditional filter (e.g., bull-day short avoidance) where the
    trade direction is not yet determined. The condition is relevant but
    cannot be evaluated until a direction is chosen at break time."""

    RULES_NOT_LOADED = "RULES_NOT_LOADED"
    """Overlay depends on an external rules file that is missing (e.g.,
    calendar_cascade_rules.json). Distinct from NEUTRAL — the rule system is
    unavailable, not reporting NEUTRAL."""

    STALE_VALIDATION = "STALE_VALIDATION"
    """Filter has not been revalidated within the freshness threshold
    (default: 180 days). Shown as a warning marker, not a hard fail — the
    filter still applies but the trader should know its provenance is aging.
    Implements the Research Provenance Rule visibly."""


class ConditionCategory(str, Enum):
    """When does this condition become relevant in the trade lifecycle?"""

    PRE_SESSION = "PRE_SESSION"
    """Resolvable before the session starts (ATR pct, overnight range, PDR,
    GAP, PIT_MIN, calendar, DOW). Uses prior-day or pre-session feature data."""

    INTRA_SESSION = "INTRA_SESSION"
    """Requires data that only exists during/after the session runs
    (ORB size, break delay, break bar continuation, cost ratio)."""

    OVERLAY = "OVERLAY"
    """Portfolio-level overlay applied at execution time (calendar sizing,
    ATR velocity skip, bull-day short avoidance). Not part of the strategy's
    filter_type but still gates trades."""

    DIRECTIONAL = "DIRECTIONAL"
    """Direction-conditional filter that applies only to long or short
    entries. Resolves at break detection when direction is determined."""


class ResolvesAt(str, Enum):
    """Event in the session lifecycle at which this condition's status can
    transition from PENDING to PASS/FAIL.

    Used by the dashboard to know when to re-compute a specific condition
    rather than polling per bar.
    """

    STARTUP = "STARTUP"
    """Resolved at session orchestrator startup (or daily_features refresh)."""

    ORB_FORMATION = "ORB_FORMATION"
    """Resolved when the ORB window closes (size, volume conditions)."""

    BREAK_DETECTED = "BREAK_DETECTED"
    """Resolved when first break bar closes outside ORB (direction, break delay)."""

    CONFIRM_COMPLETE = "CONFIRM_COMPLETE"
    """Resolved when confirm bars finish (break bar continuation for E1)."""

    TRADE_ENTERED = "TRADE_ENTERED"
    """Resolved when ActiveTrade.entry_price is set (cost ratio, stop distance)."""


class FreshnessStatus(str, Enum):
    """Freshness of the daily_features snapshot the report was built from."""

    FRESH = "FRESH"
    """daily_features as_of matches current trading day."""

    PRIOR_DAY = "PRIOR_DAY"
    """daily_features as_of is one trading day old. Acceptable for pre-session
    brief run before the daily pipeline has completed, but flag it."""

    STALE = "STALE"
    """daily_features as_of is more than one trading day old. Hard warning —
    report may misrepresent today's conditions."""

    NO_DATA = "NO_DATA"
    """No daily_features row found for this strategy's instrument at all.
    Report is meaningless; block trading decisions."""


class OverallStatus(str, Enum):
    """Derived summary status for the entire report.

    NOT a green/red semaphore for trading — eligibility is necessary-but-not-
    sufficient. Actual trade decisions also require fitness (compute_fitness())
    to be FIT. The dashboard shows BOTH icons per lane.
    """

    ELIGIBLE = "ELIGIBLE"
    """All PRE_SESSION and OVERLAY conditions are PASS. INTRA_SESSION conditions
    remain PENDING until the session runs."""

    INELIGIBLE = "INELIGIBLE"
    """At least one PRE_SESSION or OVERLAY condition is FAIL. The strategy will
    not trade today (or will trade with reduced sizing if overlay is HALF)."""

    DATA_MISSING = "DATA_MISSING"
    """At least one required feature is unavailable. Report is incomplete —
    cannot determine eligibility."""

    NEEDS_LIVE_DATA = "NEEDS_LIVE_DATA"
    """All PRE_SESSION checks PASS, but INTRA_SESSION conditions must resolve
    during the session (normal pre-session state)."""


class ConfidenceTier(str, Enum):
    """Validation confidence for this filter/overlay.

    Shown in the eligibility view so the trader sees filter provenance at a
    glance. Addresses v1 bias B4: "all filters are equal" false equivalence.

    Boundaries (proposed defaults):
    - PROVEN: DSR < 0.05, BH FDR survivor at honest K, >=10 of 16 years positive,
      revalidated within 180 days
    - PLAUSIBLE: permutation p < 0.01 but not DSR-deflated, or year stability
      < 10/16, or validation older than 180 days
    - LEGACY: validated under prior schema/methodology, retained for DB compat
    - UNKNOWN: filter lacks research provenance annotations
    """

    PROVEN = "PROVEN"
    PLAUSIBLE = "PLAUSIBLE"
    LEGACY = "LEGACY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ConditionRecord:
    """An atomic condition in a filter/overlay decomposition.

    Composite filters (e.g., ORB_G5_FAST5_CONT) decompose into multiple
    ConditionRecord instances — one per logical check. This ensures the user
    can see WHICH component of a composite failed, not just that "the filter"
    failed.

    Immutable by design: once built, the report is a snapshot. Refreshing
    requires building a new report.
    """

    name: str
    """Human-readable atomic check (e.g., "ORB size ≥ 5 pts", "cost ratio < 10%")."""

    category: ConditionCategory
    """When in the trade lifecycle this condition becomes relevant."""

    status: ConditionStatus
    """One of nine explicit status values — never a boolean, never silent."""

    resolves_at: ResolvesAt
    """Event at which a PENDING condition will be re-evaluated."""

    observed_value: float | int | str | bool | None = None
    """Today's actual value, or None if pending/missing."""

    threshold: float | int | str | bool | None = None
    """The comparison value from the filter definition."""

    comparator: str = ""
    """Plain-English comparator (e.g., '≥', '<', '==', 'in set')."""

    source_filter: str = ""
    """The filter_type key or overlay name this atom came from (e.g., 'FAST5',
    'PIT_MIN', 'calendar', 'atr_velocity')."""

    validated_for: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Sequence of (instrument, session) tuples where this atom is validated.
    Used to resolve NOT_APPLICABLE_INSTRUMENT when the current (instrument,
    session) pair is outside this set."""

    last_revalidated: date | None = None
    """Date of last revalidation from @revalidated-for annotation in config.py."""

    confidence_tier: ConfidenceTier = ConfidenceTier.UNKNOWN
    """PROVEN / PLAUSIBLE / LEGACY / UNKNOWN — anti-bias disclosure."""

    explanation: str = ""
    """One-sentence plain-English description of what this condition means."""

    size_multiplier: float = 1.0
    """Trade size multiplier to apply IF this condition passes. 1.0 = full
    position, 0.5 = half size, 0.0 = skip. Default 1.0 for ordinary
    pass/fail gates.

    This field lets overlay conditions (like calendar HALF_SIZE) pass the
    eligibility gate while still communicating "trade at reduced size" to
    the execution engine. Without this separation, HALF_SIZE would be
    forced to either PASS (losing the sizing information) or FAIL (blocking
    a profitable trade entirely — a critical behavioral bug).

    The overall size multiplier for a lane is the product of all individual
    condition multipliers whose status is PASS. See
    EligibilityReport.effective_size_multiplier."""

    @property
    def is_blocking(self) -> bool:
        """Does this condition currently block trade execution?

        True if status is FAIL (pre-session check failed) OR DATA_MISSING (can't
        verify). Does NOT block on PENDING (normal pre-session state) or any
        NOT_APPLICABLE_* variant (filter doesn't gate this lane).

        STALE_VALIDATION is a warning, not a block — the filter still applies
        but provenance is aging. Trader must decide whether to trust it.
        """
        return self.status in (ConditionStatus.FAIL, ConditionStatus.DATA_MISSING)

    @property
    def is_resolved(self) -> bool:
        """Has this condition finished its lifecycle?

        True if status is anything other than PENDING. NOT_APPLICABLE_DIRECTION
        counts as unresolved because it will resolve when direction is known.
        """
        return self.status not in (
            ConditionStatus.PENDING,
            ConditionStatus.NOT_APPLICABLE_DIRECTION,
        )


@dataclass(frozen=True)
class EligibilityReport:
    """Immutable per-strategy eligibility report for a trading day.

    Built once per data refresh (cold build) and updated event-driven by the
    execution engine (Phase 3). Consumed by both trade sheet (View A) and
    dashboard signal strip (View B).

    Fields:
    - Identity: strategy_id, instrument, session, trading_day
    - Freshness: as_of_timestamp, freshness_status
    - Content: conditions (list of ConditionRecord, one per atomic check)
    - Summary: overall_status (derived), counts per status
    - Provenance: data_provenance dict for traceability
    """

    strategy_id: str
    instrument: str
    session: str
    entry_model: str
    trading_day: date
    as_of_timestamp: datetime | None
    freshness_status: FreshnessStatus
    conditions: tuple[ConditionRecord, ...]
    overall_status: OverallStatus
    data_provenance: dict[str, str] = field(default_factory=dict)
    build_errors: tuple[str, ...] = field(default_factory=tuple)
    """Non-fatal errors during build (e.g., 'calendar rules file missing').
    Build errors do not raise — they are surfaced as DATA_MISSING or
    RULES_NOT_LOADED conditions. This field provides additional context."""

    @property
    def summary(self) -> dict[ConditionStatus, int]:
        """Count of conditions per status. Never silent — every condition is
        in exactly one status, so counts always sum to len(conditions)."""
        counts: dict[ConditionStatus, int] = {s: 0 for s in ConditionStatus}
        for c in self.conditions:
            counts[c.status] += 1
        return counts

    @property
    def blocking_conditions(self) -> tuple[ConditionRecord, ...]:
        """Conditions that currently block trade execution (FAIL + DATA_MISSING)."""
        return tuple(c for c in self.conditions if c.is_blocking)

    @property
    def pending_conditions(self) -> tuple[ConditionRecord, ...]:
        """Conditions awaiting intra-session resolution."""
        return tuple(c for c in self.conditions if c.status == ConditionStatus.PENDING)

    def conditions_by_category(self, category: ConditionCategory) -> tuple[ConditionRecord, ...]:
        """Filter conditions to a single category (PRE_SESSION, INTRA_SESSION, OVERLAY, DIRECTIONAL)."""
        return tuple(c for c in self.conditions if c.category == category)

    @property
    def effective_size_multiplier(self) -> float:
        """Product of all PASSING conditions' size_multiplier values.

        Returns 1.0 when no condition reduces size. Returns 0.5 when exactly
        one HALF_SIZE calendar rule fires. Returns 0.25 if two independent
        HALF_SIZE rules compound (unlikely but supported).

        Only PASS conditions contribute — FAIL/DATA_MISSING conditions are
        excluded from the product because they do not trade at all.
        """
        mult = 1.0
        for c in self.conditions:
            if c.status == ConditionStatus.PASS:
                mult *= c.size_multiplier
        return mult
