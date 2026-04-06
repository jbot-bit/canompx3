"""Filter decomposition registry.

Explodes composite filter_type strings into atomic condition specs. A
composite like `ORB_G5_FAST5_CONT` produces three atoms:
1. ORB size >= 5pts (INTRA_SESSION, resolves at ORB_FORMATION)
2. break delay <= 5min (INTRA_SESSION, resolves at BREAK_DETECTED)
3. break bar continues in break direction (INTRA_SESSION, resolves at CONFIRM_COMPLETE)

The registry is the SINGLE SOURCE OF TRUTH for "what atomic checks does
filter_type X perform." Adding a new filter to trading_app.config.ALL_FILTERS
requires adding its decomposition here, or the eligibility builder will
fall back to an UNKNOWN atom.

Design rationale: docs/plans/2026-04-07-eligibility-context-design.md §
"Conditions are atomic, not composite"

Grounding: Aronson Ch.6 argues composite filters hide which component
failed. Decomposition surfaces each atomic check independently so the trader
can see exactly which gate a day failed. Cited from training memory; PDF at
resources/Evidence_Based_Technical_Analysis_Aronson.pdf not extracted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from trading_app.eligibility.types import (
    ConditionCategory,
    ConfidenceTier,
    ResolvesAt,
)


@dataclass(frozen=True)
class AtomSpec:
    """Specification for an atomic condition — how to build a ConditionRecord
    from a daily_features row when this atom applies.

    The builder reads this spec and produces a ConditionRecord by looking up
    the feature_column value, comparing to threshold with comparator_fn, and
    assigning the appropriate status.

    Fields:
    - name: human-readable check ("ORB size >= 5 pts")
    - category: PRE_SESSION | INTRA_SESSION | OVERLAY | DIRECTIONAL
    - resolves_at: lifecycle event at which PENDING -> resolved
    - feature_column: daily_features column name, None for overlay-only atoms
    - feature_column_template: for session-templated columns (e.g., orb_{label}_size)
    - threshold: comparison value
    - comparator: plain-English ("≥", "<", "==", etc.)
    - source_filter: filter_type key or overlay name
    - validated_for: which (instrument, session) pairs this atom has been validated on
    - last_revalidated: date of last revalidation annotation
    - confidence_tier: PROVEN / PLAUSIBLE / LEGACY / UNKNOWN
    - explanation: one-sentence plain English
    """

    name: str
    category: ConditionCategory
    resolves_at: ResolvesAt
    threshold: float | int | str | bool | None
    comparator: str
    source_filter: str
    feature_column: str | None = None
    feature_column_template: str | None = None
    validated_for: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    last_revalidated: date | None = None
    confidence_tier: ConfidenceTier = ConfidenceTier.UNKNOWN
    explanation: str = ""

    def resolve_feature_column(self, orb_label: str) -> str | None:
        """Resolve the session-templated column name if applicable."""
        if self.feature_column_template is not None:
            return self.feature_column_template.format(label=orb_label)
        return self.feature_column


# ==========================================================================
# Atomic filter decomposition functions
#
# Each function takes a filter_type string and returns a tuple of AtomSpec
# if the filter_type matches a known pattern, else None.
# ==========================================================================


# Patterns match substrings inside composite filter_types (e.g., ORB_G5_FAST5_CONT).
# ORB_G and ORB_G*_L use `(?=_|$)` lookahead so ORB_G5 matches but ORB_G50 doesn't
# accidentally match ORB_G5. COST_LT and OVNRNG use the same discipline.
_ORB_G_PATTERN = re.compile(r"ORB_G(\d+)(?=_|$)")
_ORB_GN_LN_PATTERN = re.compile(r"ORB_G(\d+)_L(\d+)(?=_|$)")
_COST_LT_PATTERN = re.compile(r"COST_LT(\d+)(?=_|$)")
_OVNRNG_PATTERN = re.compile(r"OVNRNG_(\d+)(?=_|$)")
_PDR_PATTERN = re.compile(r"PDR_R(\d+)(?=_|$)")
_GAP_PATTERN = re.compile(r"GAP_R(\d+)(?=_|$)")
_ATR_P_PATTERN = re.compile(r"ATR_P(\d+)(?=_|$)")
_X_ATR_PATTERN = re.compile(r"X_(\w+?)_ATR(\d+)(?=_|$)")
_FAST_PATTERN = re.compile(r"FAST(\d+)(?=_|$)")
_NOMON_PATTERN = re.compile(r"NOMON")
_NOFRI_PATTERN = re.compile(r"NOFRI")
_NOTUE_PATTERN = re.compile(r"NOTUE")


def _decompose_orb_size(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """ORB_G{N} or ORB_G{N}_L{M}: ORB size gate."""
    m = _ORB_GN_LN_PATTERN.search(filter_type)
    if m:
        min_pts, max_pts = int(m.group(1)), int(m.group(2))
        return (
            AtomSpec(
                name=f"ORB size >= {min_pts} pts",
                category=ConditionCategory.INTRA_SESSION,
                resolves_at=ResolvesAt.ORB_FORMATION,
                feature_column_template="orb_{label}_size",
                threshold=float(min_pts),
                comparator=">=",
                source_filter=filter_type,
                explanation=f"Skip ORBs smaller than {min_pts} points — too thin to trade.",
            ),
            AtomSpec(
                name=f"ORB size < {max_pts} pts",
                category=ConditionCategory.INTRA_SESSION,
                resolves_at=ResolvesAt.ORB_FORMATION,
                feature_column_template="orb_{label}_size",
                threshold=float(max_pts),
                comparator="<",
                source_filter=filter_type,
                explanation=f"Skip ORBs larger than {max_pts} points — band filter.",
            ),
        )
    m = _ORB_G_PATTERN.search(filter_type)
    if m:
        min_pts = int(m.group(1))
        return (
            AtomSpec(
                name=f"ORB size >= {min_pts} pts",
                category=ConditionCategory.INTRA_SESSION,
                resolves_at=ResolvesAt.ORB_FORMATION,
                feature_column_template="orb_{label}_size",
                threshold=float(min_pts),
                comparator=">=",
                source_filter=filter_type,
                explanation=f"Skip ORBs smaller than {min_pts} points — friction-to-edge gate.",
            ),
        )
    return None


def _decompose_cost_lt(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """COST_LT{N}: round-trip friction < N% of raw ORB risk."""
    m = _COST_LT_PATTERN.search(filter_type)
    if m is None:
        return None
    pct = int(m.group(1))
    return (
        AtomSpec(
            name=f"cost ratio < {pct}%",
            category=ConditionCategory.INTRA_SESSION,
            resolves_at=ResolvesAt.ORB_FORMATION,
            feature_column_template="orb_{label}_size",
            threshold=float(pct),
            comparator="<",
            source_filter=filter_type,
            explanation=(
                f"Skip days where round-trip commission + spread exceeds {pct}% of the raw ORB risk "
                "(derived from instrument cost spec, not a daily_features column)."
            ),
        ),
    )


def _decompose_ovnrng(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """OVNRNG_{N}: absolute overnight range >= N points.

    LOOK-AHEAD SAFETY: only routed to sessions starting after 17:00 Brisbane
    (LONDON_METALS through NYSE_CLOSE). Routing is enforced by
    get_filters_for_grid(), not here.
    """
    m = _OVNRNG_PATTERN.search(filter_type)
    if m is None:
        return None
    n_pts = int(m.group(1))
    return (
        AtomSpec(
            name=f"overnight range >= {n_pts} pts",
            category=ConditionCategory.PRE_SESSION,
            resolves_at=ResolvesAt.STARTUP,
            feature_column="overnight_range",
            threshold=float(n_pts),
            comparator=">=",
            source_filter=filter_type,
            explanation=(
                f"Skip days where Asia-session range (09:00-17:00 Brisbane) was less than {n_pts} points. "
                "Look-ahead safe only for sessions starting after 17:00 Brisbane."
            ),
        ),
    )


def _decompose_pdr(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """PDR_R{NNN}: prev_day_range / atr_20 >= N/100."""
    m = _PDR_PATTERN.search(filter_type)
    if m is None:
        return None
    ratio = int(m.group(1)) / 100.0
    return (
        AtomSpec(
            name=f"prev day range / ATR >= {ratio:.2f}",
            category=ConditionCategory.PRE_SESSION,
            resolves_at=ResolvesAt.STARTUP,
            feature_column="prev_day_range",  # requires atr_20 too; builder handles the ratio
            threshold=ratio,
            comparator=">=",
            source_filter=filter_type,
            last_revalidated=date(2026, 4, 2),
            confidence_tier=ConfidenceTier.PROVEN,
            explanation=(
                f"Skip days where prior-day range / ATR20 was below {ratio:.2f} — "
                "prior-day volatility predicts breakout follow-through."
            ),
        ),
    )


def _decompose_gap(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """GAP_R{NNN}: abs(gap_open_points) / atr_20 >= N/1000 (annotation uses 3-digit ratio)."""
    m = _GAP_PATTERN.search(filter_type)
    if m is None:
        return None
    # Annotations are R005 = 0.005, R015 = 0.015
    ratio = int(m.group(1)) / 1000.0
    return (
        AtomSpec(
            name=f"abs(gap) / ATR >= {ratio:.3f}",
            category=ConditionCategory.PRE_SESSION,
            resolves_at=ResolvesAt.STARTUP,
            feature_column="gap_open_points",  # requires atr_20 too
            threshold=ratio,
            comparator=">=",
            source_filter=filter_type,
            last_revalidated=date(2026, 4, 2),
            confidence_tier=ConfidenceTier.PROVEN,
            validated_for=(("MGC", "CME_REOPEN"),),
            explanation=(
                f"Skip days where abs(overnight gap) / ATR20 was below {ratio:.3f} — "
                "larger gaps correlate with CME_REOPEN breakout quality on MGC."
            ),
        ),
    )


def _decompose_atr_p(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """ATR_P{NN}: own ATR20 percentile >= NN."""
    m = _ATR_P_PATTERN.search(filter_type)
    if m is None:
        return None
    pct = int(m.group(1))
    return (
        AtomSpec(
            name=f"own ATR20 pct >= {pct}",
            category=ConditionCategory.PRE_SESSION,
            resolves_at=ResolvesAt.STARTUP,
            feature_column="atr_20_pct",
            threshold=float(pct),
            comparator=">=",
            source_filter=filter_type,
            confidence_tier=ConfidenceTier.PLAUSIBLE,
            explanation=f"Skip days where own ATR20 percentile was below {pct} (rolling 252d rank).",
        ),
    )


def _decompose_x_atr(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """X_{INST}_ATR{NN}: cross-asset ATR pct >= NN."""
    m = _X_ATR_PATTERN.search(filter_type)
    if m is None:
        return None
    source_inst = m.group(1)
    pct = int(m.group(2))
    # Cross-asset ATR is only validated for MNQ at US sessions
    validated = tuple(
        ("MNQ", s)
        for s in ("CME_PRECLOSE", "COMEX_SETTLE", "US_DATA_1000", "NYSE_OPEN", "NYSE_CLOSE", "US_DATA_830")
    )
    return (
        AtomSpec(
            name=f"{source_inst} ATR20 pct >= {pct}",
            category=ConditionCategory.PRE_SESSION,
            resolves_at=ResolvesAt.STARTUP,
            feature_column=f"cross_atr_{source_inst}_pct",
            threshold=float(pct),
            comparator=">=",
            source_filter=filter_type,
            validated_for=validated,
            confidence_tier=ConfidenceTier.PROVEN,
            explanation=(
                f"Skip days where {source_inst} ATR20 percentile was below {pct} — "
                "cross-asset vol regime gate for MNQ US sessions only."
            ),
        ),
    )


def _pit_min_atom() -> AtomSpec:
    """PIT_MIN: pit_range_atr >= 0.10."""
    return AtomSpec(
        name="pit range / ATR >= 0.10",
        category=ConditionCategory.PRE_SESSION,
        resolves_at=ResolvesAt.STARTUP,
        feature_column="pit_range_atr",
        threshold=0.10,
        comparator=">=",
        source_filter="PIT_MIN",
        validated_for=(
            ("MGC", "CME_REOPEN"),
            ("MES", "CME_REOPEN"),
            ("MNQ", "CME_REOPEN"),
        ),
        last_revalidated=date(2026, 4, 4),
        confidence_tier=ConfidenceTier.PROVEN,
        explanation=(
            "Skip dead-pit days (pit_range/ATR20 < 0.10, bottom ~20% of days) — "
            "weak pit sessions predict CME_REOPEN breakout failure. "
            "Zero look-ahead: pit closes 21:00 UTC, CME_REOPEN starts 23:00 UTC."
        ),
    )


def _decompose_pit_min(filter_type: str) -> tuple[AtomSpec, ...] | None:
    if filter_type == "PIT_MIN":
        return (_pit_min_atom(),)
    return None


def _decompose_fast(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """FAST{N}: break_delay_min <= N. Only present as part of composites.

    Per-instrument validity: validated for MNQ only at CME_REOPEN, NYSE_OPEN,
    NYSE_CLOSE, TOKYO_OPEN, LONDON_METALS (and MGC at CME_REOPEN). Not validated
    for MGC/MES at other sessions — builder marks as NOT_APPLICABLE_INSTRUMENT.
    """
    m = _FAST_PATTERN.search(filter_type)
    if m is None:
        return None
    max_delay = int(m.group(1))
    # Validated combos from break_speed_signal_retest.md (Apr 2026)
    validated = (
        ("MNQ", "NYSE_CLOSE"),
        ("MNQ", "NYSE_OPEN"),
        ("MNQ", "TOKYO_OPEN"),
        ("MNQ", "LONDON_METALS"),
        ("MNQ", "CME_REOPEN"),
        ("MGC", "CME_REOPEN"),
    )
    return (
        AtomSpec(
            name=f"break delay <= {max_delay} min",
            category=ConditionCategory.INTRA_SESSION,
            resolves_at=ResolvesAt.BREAK_DETECTED,
            feature_column_template="orb_{label}_break_delay_min",
            threshold=float(max_delay),
            comparator="<=",
            source_filter=f"FAST{max_delay}",
            validated_for=validated,
            last_revalidated=date(2026, 4, 1),
            confidence_tier=ConfidenceTier.PROVEN,
            explanation=(
                f"Skip slow-break days (break delay > {max_delay} min from ORB close). "
                "Fast breaks select momentum days; slow breaks are grinding / mean-reversion. "
                "Per-instrument validity: MNQ + specific sessions only."
            ),
        ),
    )


def _decompose_cont(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """CONT: break bar continues in break direction. E1-only (E2 is look-ahead blocked)."""
    if "_CONT" not in filter_type and filter_type != "CONT":
        return None
    return (
        AtomSpec(
            name="break bar closes in break direction",
            category=ConditionCategory.INTRA_SESSION,
            resolves_at=ResolvesAt.CONFIRM_COMPLETE,
            feature_column_template="orb_{label}_break_bar_continues",
            threshold=True,
            comparator="==",
            source_filter="CONT",
            confidence_tier=ConfidenceTier.PLAUSIBLE,
            explanation=(
                "Skip reversal candles at break point — require the break bar to close in "
                "the break direction. E1-only (look-ahead unsafe for E2 stop-market entries)."
            ),
        ),
    )


def _decompose_dir(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """DIR_LONG or DIR_SHORT direction filters."""
    if "DIR_LONG" in filter_type:
        return (
            AtomSpec(
                name="direction == LONG",
                category=ConditionCategory.DIRECTIONAL,
                resolves_at=ResolvesAt.BREAK_DETECTED,
                feature_column_template="orb_{label}_break_dir",
                threshold="long",
                comparator="==",
                source_filter="DIR_LONG",
                explanation="Long breakouts only — shorts were proven unprofitable on this session.",
            ),
        )
    if "DIR_SHORT" in filter_type:
        return (
            AtomSpec(
                name="direction == SHORT",
                category=ConditionCategory.DIRECTIONAL,
                resolves_at=ResolvesAt.BREAK_DETECTED,
                feature_column_template="orb_{label}_break_dir",
                threshold="short",
                comparator="==",
                source_filter="DIR_SHORT",
                explanation="Short breakouts only.",
            ),
        )
    return None


def _decompose_dow(filter_type: str) -> tuple[AtomSpec, ...] | None:
    """NOMON / NOFRI / NOTUE day-of-week skip atoms."""
    atoms: list[AtomSpec] = []
    if _NOMON_PATTERN.search(filter_type):
        atoms.append(
            AtomSpec(
                name="day of week != Monday",
                category=ConditionCategory.PRE_SESSION,
                resolves_at=ResolvesAt.STARTUP,
                feature_column=None,  # computed from trading_day
                threshold="Monday",
                comparator="!=",
                source_filter="NOMON",
                confidence_tier=ConfidenceTier.PLAUSIBLE,
                explanation="Skip Monday entries — weekend gap noise reduces edge.",
            )
        )
    if _NOFRI_PATTERN.search(filter_type):
        atoms.append(
            AtomSpec(
                name="day of week != Friday",
                category=ConditionCategory.PRE_SESSION,
                resolves_at=ResolvesAt.STARTUP,
                feature_column=None,
                threshold="Friday",
                comparator="!=",
                source_filter="NOFRI",
                confidence_tier=ConfidenceTier.LEGACY,
                explanation="Skip Friday entries (removed from grid Mar 2026 — retained for DB compat).",
            )
        )
    if _NOTUE_PATTERN.search(filter_type):
        atoms.append(
            AtomSpec(
                name="day of week != Tuesday",
                category=ConditionCategory.PRE_SESSION,
                resolves_at=ResolvesAt.STARTUP,
                feature_column=None,
                threshold="Tuesday",
                comparator="!=",
                source_filter="NOTUE",
                confidence_tier=ConfidenceTier.LEGACY,
                explanation="Skip Tuesday entries (removed from grid Mar 2026 — retained for DB compat).",
            )
        )
    return tuple(atoms) if atoms else None


# ==========================================================================
# Top-level decompose() function
# ==========================================================================


_DECOMPOSERS = (
    _decompose_orb_size,
    _decompose_cost_lt,
    _decompose_ovnrng,
    _decompose_pdr,
    _decompose_gap,
    _decompose_atr_p,
    _decompose_x_atr,
    _decompose_pit_min,
    _decompose_fast,
    _decompose_cont,
    _decompose_dir,
    _decompose_dow,
)


def decompose(filter_type: str) -> tuple[AtomSpec, ...]:
    """Explode a filter_type string into atomic condition specs.

    A composite like `ORB_G5_FAST5_CONT` returns 3 atoms because it calls
    multiple decomposers that each match a substring. Order of returned atoms
    reflects the order decomposers are called.

    Special cases:
    - NO_FILTER returns an empty tuple (no atoms — always ELIGIBLE)
    - Unknown filter_type returns an UNKNOWN placeholder atom (explicit,
      not silent)
    """
    if filter_type == "NO_FILTER":
        return ()

    atoms: list[AtomSpec] = []
    for decomposer in _DECOMPOSERS:
        result = decomposer(filter_type)
        if result is not None:
            atoms.extend(result)

    if not atoms:
        # Unknown filter — return explicit UNKNOWN placeholder atom so the
        # builder surfaces it, never silently treats it as PASS.
        atoms.append(
            AtomSpec(
                name=f"UNKNOWN filter: {filter_type}",
                category=ConditionCategory.INTRA_SESSION,
                resolves_at=ResolvesAt.STARTUP,
                feature_column=None,
                threshold=None,
                comparator="",
                source_filter=filter_type,
                confidence_tier=ConfidenceTier.UNKNOWN,
                explanation=(
                    f"Filter type '{filter_type}' has no decomposition registered. "
                    "Add a decomposer to trading_app/eligibility/decomposition.py."
                ),
            )
        )

    return tuple(atoms)


# ==========================================================================
# Overlay atom builders (calendar, ATR velocity, bull-day short avoidance)
#
# These are not driven by filter_type — they apply as portfolio overlays.
# ==========================================================================


def calendar_atom_template() -> AtomSpec:
    """Template atom for the calendar overlay. Observed value and status are
    filled in by the builder at report-build time."""
    return AtomSpec(
        name="calendar action",
        category=ConditionCategory.OVERLAY,
        resolves_at=ResolvesAt.STARTUP,
        feature_column=None,
        threshold="NEUTRAL",
        comparator="==",
        source_filter="calendar",
        explanation=(
            "Calendar overlay from research/output/calendar_cascade_rules.json: NEUTRAL = trade, "
            "HALF_SIZE = half size, SKIP = no trade. RULES_NOT_LOADED if file is missing."
        ),
    )


def atr_velocity_atom_template() -> AtomSpec:
    """Template atom for the ATR velocity overlay (MGC-only, CME_REOPEN + TOKYO_OPEN)."""
    return AtomSpec(
        name="ATR velocity regime not contracting+compressed",
        category=ConditionCategory.OVERLAY,
        resolves_at=ResolvesAt.STARTUP,
        feature_column="atr_vel_regime",
        threshold="Contracting+Compressed",
        comparator="!=",
        source_filter="atr_velocity",
        validated_for=(
            ("MGC", "CME_REOPEN"),
            ("MGC", "TOKYO_OPEN"),
        ),
        confidence_tier=ConfidenceTier.PROVEN,
        explanation=(
            "Skip sessions when ATR is contracting AND ORB compression tier is Neutral or "
            "Compressed. MGC-specific overlay (MNQ/MES not validated)."
        ),
    )


def bull_day_short_atom_template() -> AtomSpec:
    """Template atom for bull-day short avoidance (deferred — requires NYSE_OPEN lanes)."""
    return AtomSpec(
        name="not half-size-short on post-bull-day",
        category=ConditionCategory.DIRECTIONAL,
        resolves_at=ResolvesAt.BREAK_DETECTED,
        feature_column=None,
        threshold="bear_day_or_long",
        comparator="==",
        source_filter="bull_day_short_avoidance",
        validated_for=(("MNQ", "NYSE_OPEN"),),
        last_revalidated=date(2026, 4, 4),
        confidence_tier=ConfidenceTier.PROVEN,
        explanation=(
            "Half-size shorts after bull days — p=0.0007, 14/17 years. "
            "NYSE_OPEN strongest. Not yet deployed (no NYSE_OPEN lanes active)."
        ),
    )
