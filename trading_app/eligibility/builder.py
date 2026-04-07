"""Eligibility report builder — THIN ADAPTER over canonical filters.

Post-canonical-filter-self-description refactor (2026-04-07): this module
is a pure mechanical translation layer. ALL filter semantics live in
trading_app/config.py via StrategyFilter.describe(). The adapter's job:

1. parse_strategy_id → extract dimensions
2. ALL_FILTERS lookup → instantiate canonical filter (raise on unknown)
3. Walk composite tree → call describe() on each leaf filter
4. Translate AtomDescription → ConditionRecord with status mapping
5. Apply validated_for + last_revalidated overrides mechanically
6. Aggregate per-atom error_messages into report.build_errors
7. Add calendar overlay (RULES_NOT_LOADED + HALF_SIZE handling)
8. Add ATR velocity overlay (canonical delegation, monitored sessions)
9. Derive overall status

ZERO re-encoded filter logic. Parallel-model drift is mechanically
impossible: there is one and only one source of truth for "what does
filter X check" — the filter class's describe() method.

Design rationale: docs/plans/2026-04-07-canonical-filter-self-description-design.md
Stage:           docs/runtime/stages/canonical-filter-self-description.md
Rule:            .claude/rules/institutional-rigor.md (no re-encoded logic)

Grounding: Pardo Ch.4 (look-ahead) and Aronson Ch.6 (confirmation bias)
remain the load-bearing references for the eligibility report design.
Both PDFs at resources/ — citations from training memory unless extracted.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from trading_app.config import (
    ALL_FILTERS,
    ATR_VELOCITY_OVERLAY,
    AtomDescription,
    CompositeFilter,
    StrategyFilter,
)
from trading_app.eligibility.types import (
    ConditionCategory,
    ConditionRecord,
    ConditionStatus,
    ConfidenceTier,
    EligibilityReport,
    FreshnessStatus,
    OverallStatus,
    ResolvesAt,
)

# Freshness threshold for STALE_VALIDATION status (days).
VALIDATION_FRESHNESS_DAYS = 180

# Path to calendar cascade rules file (for RULES_NOT_LOADED detection).
_CALENDAR_RULES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "research"
    / "output"
    / "calendar_cascade_rules.json"
)


# ==========================================================================
# Strategy ID parsing — preserved from prior version (not filter logic)
# ==========================================================================


def parse_strategy_id(strategy_id: str) -> dict[str, Any]:
    """Extract dimensions from a strategy_id string.

    Format examples:
      MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6
      MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12
      MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10_O15  (with aperture suffix)

    Returns dict with: instrument, orb_label, entry_model, rr_target,
    confirm_bars, filter_type, orb_minutes (default 5).

    Raises ValueError on unparseable input — no silent failure.
    """
    parts = strategy_id.split("_")
    if len(parts) < 6:
        raise ValueError(f"strategy_id too short to parse: {strategy_id!r}")

    instrument = parts[0]
    if instrument not in ("MGC", "MNQ", "MES"):
        raise ValueError(
            f"strategy_id has unknown instrument {instrument!r}: {strategy_id!r}"
        )

    em_idx = None
    for i, p in enumerate(parts):
        if p in ("E1", "E2", "E3"):
            em_idx = i
            break
    if em_idx is None:
        raise ValueError(
            f"strategy_id has no entry model (E1/E2/E3): {strategy_id!r}"
        )

    orb_label = "_".join(parts[1:em_idx])
    entry_model = parts[em_idx]

    if em_idx + 1 >= len(parts) or not parts[em_idx + 1].startswith("RR"):
        raise ValueError(f"strategy_id missing RR after entry model: {strategy_id!r}")
    rr_target = float(parts[em_idx + 1][2:])

    if em_idx + 2 >= len(parts) or not parts[em_idx + 2].startswith("CB"):
        raise ValueError(f"strategy_id missing CB after RR: {strategy_id!r}")
    confirm_bars = int(parts[em_idx + 2][2:])

    remaining = parts[em_idx + 3 :]
    orb_minutes = 5
    if remaining and remaining[-1].startswith("O") and remaining[-1][1:].isdigit():
        orb_minutes = int(remaining[-1][1:])
        remaining = remaining[:-1]

    filter_type = "_".join(remaining) if remaining else "NO_FILTER"

    return {
        "instrument": instrument,
        "orb_label": orb_label,
        "entry_model": entry_model,
        "rr_target": rr_target,
        "confirm_bars": confirm_bars,
        "filter_type": filter_type,
        "orb_minutes": orb_minutes,
    }


# ==========================================================================
# Recursive composite walk
# ==========================================================================


def _synthetic_failed_atom(
    filter_type: str,
    error_msg: str,
    violation_subtype: str = "contract violation",
) -> AtomDescription:
    """Fail-closed synthetic atom for filter describe() contract violations.

    Single source of truth for "the filter did not honour its describe()
    contract" — covers both exception-raising and malformed-return paths in
    _walk_filter_atoms. The returned atom is translated to a DATA_MISSING
    ConditionRecord by _status_from_atom via the passes=None +
    is_data_missing=True + PRE_SESSION + STARTUP path. The error_msg flows
    into report.build_errors via the main loop's atom.error_message
    aggregation.

    Args:
        filter_type: The filter_type tag for the broken filter (used for
            provenance in the condition name and report traceability).
        error_msg: Full diagnostic message — exposed in both error_message
            and explanation fields, and surfaced in report.build_errors
            via the main loop aggregation.
        violation_subtype: Short name label distinguishing the violation
            class. Default "contract violation" preserves a generic label
            when the caller doesn't know or care. Specific sites pass
            "raised exception", "returned non-list", or "yielded non-atom"
            so operators scanning a condition list can diagnose at a
            glance without reading error_msg.

    Institutional-rigor rule #4: one helper, not parallel constructions.
    """
    return AtomDescription(
        name=f"{filter_type}: describe() {violation_subtype}",
        category="PRE_SESSION",
        resolves_at="STARTUP",
        passes=None,
        is_data_missing=True,
        error_message=error_msg,
        explanation=error_msg,
    )


def _walk_filter_atoms(
    filt: StrategyFilter,
    row: dict[str, Any],
    orb_label: str,
    entry_model: str,
) -> list[tuple[str, AtomDescription]]:
    """Descend into a composite filter, yielding (leaf_filter_type, atom) pairs.

    Top-level non-composite → calls describe() once. CompositeFilter →
    recurses into base + overlay, preserving order. Each leaf's filter_type
    becomes the source_filter tag for its atoms.

    Fail-closed discipline: if a leaf's describe() raises OR returns a
    malformed value (not list/tuple, or contains non-AtomDescription
    elements), a synthetic DATA_MISSING AtomDescription is emitted via
    _synthetic_failed_atom with the contract-violation message captured
    in `error_message`. This flows into:
      1. report.conditions — a visible DATA_MISSING record tagged with the
         broken leaf's filter_type, so _derive_overall_status returns
         DATA_MISSING instead of silently reporting ELIGIBLE
      2. report.build_errors — via the main loop's atom.error_message
         aggregation, preserving diagnostic detail for debugging

    Sibling leaves in a composite are unaffected: if ORB_G5 succeeds but
    an overlay leaf raises, the composite's report contains ORB_G5's real
    atoms AND the overlay's synthetic DATA_MISSING atom. Within a single
    leaf's return list, valid AtomDescription elements are preserved and
    only the bad elements are replaced with synthetic atoms — partial
    success is honoured.

    Exception policy: we catch a broad Exception here because describe()
    is an adapter boundary — infrastructure failures must surface as
    explicit conditions, not bubble up as unhandled exceptions from
    build_eligibility_report. The synthetic atom is the single-source-of-
    truth for the error; the main loop reads atom.error_message and flows
    it into build_errors, so there is no duplicate append here.

    Return-shape validation: drift check #85 catches describe() contract
    violations at check time. This runtime validation is defense-in-depth
    per .claude/rules/institutional-rigor.md rule #6 ("no silent failures").
    """
    if isinstance(filt, CompositeFilter):
        return _walk_filter_atoms(
            filt.base, row, orb_label, entry_model
        ) + _walk_filter_atoms(
            filt.overlay, row, orb_label, entry_model
        )
    try:
        atoms = filt.describe(row, orb_label, entry_model)
    except Exception as exc:  # noqa: BLE001 — adapter boundary, fail-closed
        error_msg = (
            f"{filt.filter_type}.describe raised: "
            f"{type(exc).__name__}: {exc}"
        )
        return [
            (
                filt.filter_type,
                _synthetic_failed_atom(
                    filt.filter_type, error_msg, violation_subtype="raised exception"
                ),
            )
        ]

    # Validate return shape — fail-closed on any contract violation. The
    # contract is list[AtomDescription]; anything else (None, dict, str,
    # generator, set) is a filter bug we refuse to paper over.
    if not isinstance(atoms, (list, tuple)):
        error_msg = (
            f"{filt.filter_type}.describe returned {type(atoms).__name__}, "
            f"expected list[AtomDescription]"
        )
        return [
            (
                filt.filter_type,
                _synthetic_failed_atom(
                    filt.filter_type, error_msg, violation_subtype="returned non-list"
                ),
            )
        ]

    # Validate each element — bad elements become synthetic DATA_MISSING
    # atoms while valid siblings in the same list are preserved.
    result: list[tuple[str, AtomDescription]] = []
    for atom in atoms:
        if not isinstance(atom, AtomDescription):
            error_msg = (
                f"{filt.filter_type}.describe yielded {type(atom).__name__}, "
                f"expected AtomDescription"
            )
            result.append(
                (
                    filt.filter_type,
                    _synthetic_failed_atom(
                        filt.filter_type,
                        error_msg,
                        violation_subtype="yielded non-atom",
                    ),
                )
            )
        else:
            result.append((filt.filter_type, atom))
    return result


# ==========================================================================
# AtomDescription → ConditionRecord translation
# ==========================================================================

# Map AtomDescription string fields to enum types. AtomDescription uses
# str-based fields (category, resolves_at) for adapter-agnostic
# serialisation; the adapter normalises them here.
_CATEGORY_MAP: dict[str, ConditionCategory] = {
    "PRE_SESSION": ConditionCategory.PRE_SESSION,
    "INTRA_SESSION": ConditionCategory.INTRA_SESSION,
    "OVERLAY": ConditionCategory.OVERLAY,
    "DIRECTIONAL": ConditionCategory.DIRECTIONAL,
}

_RESOLVES_AT_MAP: dict[str, ResolvesAt] = {
    "STARTUP": ResolvesAt.STARTUP,
    "ORB_FORMATION": ResolvesAt.ORB_FORMATION,
    "BREAK_DETECTED": ResolvesAt.BREAK_DETECTED,
    "CONFIRM_COMPLETE": ResolvesAt.CONFIRM_COMPLETE,
    "TRADE_ENTERED": ResolvesAt.TRADE_ENTERED,
}

_CONFIDENCE_TIER_MAP: dict[str, ConfidenceTier] = {
    "PROVEN": ConfidenceTier.PROVEN,
    "PLAUSIBLE": ConfidenceTier.PLAUSIBLE,
    "LEGACY": ConfidenceTier.LEGACY,
    "UNKNOWN": ConfidenceTier.UNKNOWN,
}


def _status_from_atom(
    atom: AtomDescription,
    instrument: str,
    session: str,
    trading_day: date,
) -> ConditionStatus:
    """Mechanical translation of AtomDescription state → ConditionStatus.

    Precedence (each rule short-circuits):
      1. is_not_applicable=True → NOT_APPLICABLE_ENTRY_MODEL
         (current sole producer is _e2_look_ahead_reason in config.py)
      2. validated_for non-empty AND (instrument, session) NOT in tuple
         → NOT_APPLICABLE_INSTRUMENT
      3. last_revalidated set AND (trading_day - last_revalidated) > 180 days
         → STALE_VALIDATION
      4. passes is True → PASS
      5. passes is False → FAIL
      6. passes is None:
         - INTRA_SESSION or DIRECTIONAL category → PENDING
           (these atoms resolve later in the trade lifecycle)
         - PRE_SESSION or OVERLAY with is_data_missing → DATA_MISSING
           (real infrastructure problem — feature should be there)
         - PRE_SESSION or OVERLAY without is_data_missing → PENDING
           (unusual — atom is genuinely waiting on something)
    """
    # 1. Look-ahead exclusion (canonical _e2_look_ahead_reason path)
    if atom.is_not_applicable:
        return ConditionStatus.NOT_APPLICABLE_ENTRY_MODEL

    # 2. Per-lane validation restriction (canonical VALIDATED_FOR ClassVar)
    if atom.validated_for and (instrument, session) not in atom.validated_for:
        return ConditionStatus.NOT_APPLICABLE_INSTRUMENT

    # 3. Stale research provenance
    if atom.last_revalidated is not None:
        age_days = (trading_day - atom.last_revalidated).days
        if age_days > VALIDATION_FRESHNESS_DAYS:
            return ConditionStatus.STALE_VALIDATION

    # 4-6. passes-based mapping
    if atom.passes is True:
        return ConditionStatus.PASS
    if atom.passes is False:
        return ConditionStatus.FAIL

    # passes is None — distinguish PENDING from DATA_MISSING by category
    if atom.category in ("INTRA_SESSION", "DIRECTIONAL"):
        return ConditionStatus.PENDING
    if atom.is_data_missing:
        return ConditionStatus.DATA_MISSING
    return ConditionStatus.PENDING  # PRE_SESSION/OVERLAY genuinely pending


def _contract_violation_record(
    source_filter: str,
    error_msg: str,
    violation_subtype: str = "atom contract violation",
) -> ConditionRecord:
    """Fail-closed ConditionRecord for atom enum-string contract violations.

    Single source of truth for "the atom carries a typo'd category,
    resolves_at, or confidence_tier string". Returns a visible
    DATA_MISSING condition tagged with the source filter so
    _derive_overall_status refuses to report ELIGIBLE and the user can
    trace which filter emitted the bad atom. Paired with build_errors
    accumulation in _atom_to_condition's caller, this closes the runtime
    defense-in-depth gap left by drift check #85 (which only catches at
    check time, not runtime).

    Args:
        source_filter: The filter_type tag for the broken filter
            (carried through to the condition for traceability).
        error_msg: Full diagnostic message — exposed in the
            explanation field and surfaced in report.build_errors.
        violation_subtype: Short label distinguishing the violation
            class. Default "atom contract violation" preserves a
            generic label. Specific sites pass "bad category",
            "bad resolves_at", or "bad confidence_tier" so operators
            can diagnose at a glance without reading error_msg.

    Institutional-rigor rule #4: one helper, not parallel constructions.
    """
    return ConditionRecord(
        name=f"{source_filter}: {violation_subtype}",
        category=ConditionCategory.PRE_SESSION,
        status=ConditionStatus.DATA_MISSING,
        resolves_at=ResolvesAt.STARTUP,
        source_filter=source_filter,
        confidence_tier=ConfidenceTier.UNKNOWN,
        explanation=error_msg,
    )


def _atom_to_condition(
    atom: AtomDescription,
    source_filter: str,
    instrument: str,
    session: str,
    trading_day: date,
    build_errors: list[str],
) -> ConditionRecord:
    """Translate one AtomDescription into one ConditionRecord.

    Mechanical mapping only — no semantic logic. Category / resolves_at /
    confidence_tier string→enum lookups are validated explicitly:
    fail-closed on typos via _contract_violation_record rather than
    silently coercing to a default. The previous `.get(default)` pattern
    would flip a typo'd-category atom to PRE_SESSION and let passes=True
    escape as PASS, hiding a real filter bug from the live eligibility
    gate.

    Drift check #85 catches these at check time; this runtime validation
    is defense-in-depth per .claude/rules/institutional-rigor.md rule #6
    ("no silent failures").

    Status derivation via _status_from_atom (only reached when all enum
    strings are valid — direct indexing after the membership guards).
    The atom's error_message is collected separately by the caller
    (build_eligibility_report) into report.build_errors. Contract
    violations detected here also append to build_errors directly, so
    both channels surface the same information.
    """
    # Validate enum strings — fail-closed on typos. Membership check
    # instead of `.get(default)` because a silent default would let a
    # typo'd atom resolve to PASS (see prior PASS-instead-of-DATA_MISSING
    # regression fixed by this hardening).
    if atom.category not in _CATEGORY_MAP:
        error_msg = (
            f"{source_filter}: atom.category={atom.category!r} not a valid "
            f"ConditionCategory (expected one of {sorted(_CATEGORY_MAP)})"
        )
        build_errors.append(error_msg)
        return _contract_violation_record(
            source_filter, error_msg, violation_subtype="bad category"
        )
    if atom.resolves_at not in _RESOLVES_AT_MAP:
        error_msg = (
            f"{source_filter}: atom.resolves_at={atom.resolves_at!r} not a "
            f"valid ResolvesAt (expected one of {sorted(_RESOLVES_AT_MAP)})"
        )
        build_errors.append(error_msg)
        return _contract_violation_record(
            source_filter, error_msg, violation_subtype="bad resolves_at"
        )
    if atom.confidence_tier not in _CONFIDENCE_TIER_MAP:
        error_msg = (
            f"{source_filter}: atom.confidence_tier={atom.confidence_tier!r} "
            f"not a valid ConfidenceTier "
            f"(expected one of {sorted(_CONFIDENCE_TIER_MAP)})"
        )
        build_errors.append(error_msg)
        return _contract_violation_record(
            source_filter, error_msg, violation_subtype="bad confidence_tier"
        )

    # All enum strings valid — direct indexing, no silent defaults.
    category = _CATEGORY_MAP[atom.category]
    resolves_at = _RESOLVES_AT_MAP[atom.resolves_at]
    confidence_tier = _CONFIDENCE_TIER_MAP[atom.confidence_tier]
    status = _status_from_atom(atom, instrument, session, trading_day)

    return ConditionRecord(
        name=atom.name,
        category=category,
        status=status,
        resolves_at=resolves_at,
        observed_value=atom.observed_value,
        threshold=atom.threshold,
        comparator=atom.comparator,
        source_filter=source_filter,
        validated_for=atom.validated_for,
        last_revalidated=atom.last_revalidated,
        confidence_tier=confidence_tier,
        explanation=atom.explanation,
        size_multiplier=atom.size_multiplier,
    )


# ==========================================================================
# Calendar overlay (NOT a filter — external rules file integration)
# ==========================================================================


def _calendar_rules_loaded() -> bool:
    """True iff the calendar cascade rules file exists and is non-empty."""
    if not _CALENDAR_RULES_PATH.exists():
        return False
    try:
        raw = json.loads(_CALENDAR_RULES_PATH.read_text(encoding="utf-8"))
        return bool(raw.get("rules"))
    except (json.JSONDecodeError, OSError):
        return False


def _build_calendar_condition(
    instrument: str,
    session: str,
    trading_day: date,
    build_errors: list[str],
) -> ConditionRecord:
    """Build the calendar overlay condition.

    Returns RULES_NOT_LOADED if the rules file is missing or empty —
    NOT silent NEUTRAL. NEUTRAL action → PASS with size_multiplier=1.0.
    HALF_SIZE action → PASS with size_multiplier=0.5 (overlay sizing
    is separate from the pass/fail gate). SKIP → FAIL.

    The calendar overlay is intentionally NOT a filter.describe() path
    because it depends on an external JSON rules file, not on
    daily_features columns. Keeping it adapter-side preserves the
    rule "filters describe row-derived gates; overlays integrate
    external systems."
    """
    if not _calendar_rules_loaded():
        return ConditionRecord(
            name="calendar action",
            category=ConditionCategory.OVERLAY,
            status=ConditionStatus.RULES_NOT_LOADED,
            resolves_at=ResolvesAt.STARTUP,
            source_filter="calendar",
            confidence_tier=ConfidenceTier.UNKNOWN,
            explanation=(
                f"calendar_cascade_rules.json not found at "
                f"{_CALENDAR_RULES_PATH} — rules system unavailable."
            ),
        )

    try:
        from trading_app.calendar_overlay import CalendarAction, get_calendar_action

        action = get_calendar_action(instrument, session, trading_day)
        if action == CalendarAction.NEUTRAL:
            status = ConditionStatus.PASS
            size_multiplier = 1.0
        elif action == CalendarAction.HALF_SIZE:
            status = ConditionStatus.PASS
            size_multiplier = 0.5
        else:  # SKIP
            status = ConditionStatus.FAIL
            size_multiplier = 0.0
        observed = action.name
    except Exception as exc:  # noqa: BLE001
        build_errors.append(
            f"calendar lookup raised: {type(exc).__name__}: {exc}"
        )
        return ConditionRecord(
            name="calendar action",
            category=ConditionCategory.OVERLAY,
            status=ConditionStatus.DATA_MISSING,
            resolves_at=ResolvesAt.STARTUP,
            source_filter="calendar",
            confidence_tier=ConfidenceTier.UNKNOWN,
            explanation=(
                f"calendar lookup raised: {type(exc).__name__}: {exc}"
            ),
        )

    return ConditionRecord(
        name="calendar action",
        category=ConditionCategory.OVERLAY,
        status=status,
        resolves_at=ResolvesAt.STARTUP,
        observed_value=observed,
        source_filter="calendar",
        confidence_tier=ConfidenceTier.UNKNOWN,
        explanation=(
            "Calendar overlay (NEUTRAL=trade, HALF_SIZE=half size, "
            "SKIP=no trade)."
        ),
        size_multiplier=size_multiplier,
    )


# ==========================================================================
# ATR velocity overlay (canonical delegation)
# ==========================================================================


def _build_atr_velocity_condition(
    feature_row: dict[str, Any] | None,
    instrument: str,
    session: str,
    trading_day: date,
    build_errors: list[str],
) -> ConditionRecord | None:
    """Build the ATR velocity overlay via canonical describe() delegation.

    Returns None if the overlay does not apply to this (instrument, session)
    pair. Returns a synthetic DATA_MISSING ConditionRecord if the canonical
    describe() raises — fail-closed, not fail-open.

    Flow:
      1. Pre-check ATR_VELOCITY_OVERLAY.VALIDATED_FOR (reading the
         canonical ClassVar, not re-encoding). If the lane is outside
         the validated tuple, skip entirely — no describe() call, no
         condition added. This is the "overlay doesn't apply here"
         signal and matches pre-refactor behavior.
      2. Call ATR_VELOCITY_OVERLAY.describe() inside a try/except. On
         exception, append to build_errors and return a synthetic
         DATA_MISSING ConditionRecord so the overlay failure surfaces
         in the report rather than propagating uncaught (which would
         break the "never raise on bad data" contract).
      3. If describe() returns an atom with is_not_applicable=True,
         respect it (canonical said it doesn't apply — may be a future
         config change where apply_to_sessions diverges from VALIDATED_FOR).
      4. Otherwise, mechanically translate to ConditionRecord.

    Pre-refactor parallel model re-encoded vel_regime/compression
    comparisons and diverged on warm-up edge cases. Now: pure delegation,
    inheriting canonical fail-open warm-up semantics for free.

    Why pre-check VALIDATED_FOR before calling describe():
      - Avoids surfacing a spurious DATA_MISSING overlay on non-applicable
        lanes (e.g., MNQ NYSE_CLOSE) if describe() fails. The overlay
        genuinely doesn't apply, so a failure there is moot.
      - Reading ATR_VELOCITY_OVERLAY.VALIDATED_FOR is canonical lookup,
        not re-encoding. The ClassVar IS the single source of truth; we
        simply read it directly to decide whether to invoke describe().
    """
    # Pre-check: if this (instrument, session) is outside the canonical
    # VALIDATED_FOR tuple, the overlay genuinely doesn't apply — skip
    # entirely. Reading the ClassVar is canonical delegation.
    if ATR_VELOCITY_OVERLAY.VALIDATED_FOR and (
        instrument, session
    ) not in ATR_VELOCITY_OVERLAY.VALIDATED_FOR:
        return None

    row_safe = feature_row if feature_row is not None else {}
    # entry_model is irrelevant for the ATR velocity overlay (it applies
    # to all entry models on the validated MGC sessions).
    try:
        atoms = ATR_VELOCITY_OVERLAY.describe(row_safe, session, "E2")
    except Exception as exc:  # noqa: BLE001 — adapter boundary, fail-closed
        error_msg = (
            f"ATR_VELOCITY_OVERLAY.describe raised: "
            f"{type(exc).__name__}: {exc}"
        )
        build_errors.append(error_msg)
        # Fail-closed: surface the failure as a visible DATA_MISSING
        # condition on the (already-verified-applicable) lane rather
        # than returning None (which means "doesn't apply").
        return ConditionRecord(
            name="ATR velocity overlay",
            category=ConditionCategory.OVERLAY,
            status=ConditionStatus.DATA_MISSING,
            resolves_at=ResolvesAt.ORB_FORMATION,
            source_filter="atr_velocity",
            confidence_tier=ConfidenceTier.UNKNOWN,
            explanation=error_msg,
        )

    if not atoms:
        return None
    atom = atoms[0]
    if atom.is_not_applicable:
        # Defensive: canonical said doesn't apply despite our pre-check
        # passing. Respect it (possible future divergence between
        # apply_to_sessions and VALIDATED_FOR).
        return None
    return _atom_to_condition(
        atom,
        source_filter="atr_velocity",
        instrument=instrument,
        session=session,
        trading_day=trading_day,
        build_errors=build_errors,
    )


# ==========================================================================
# Freshness resolution
# ==========================================================================


def _resolve_freshness(
    feature_row: dict[str, Any] | None,
    current_trading_day: date,
) -> tuple[FreshnessStatus, datetime | None]:
    """Determine freshness status from the feature row's trading_day."""
    if feature_row is None:
        return FreshnessStatus.NO_DATA, None

    row_day = feature_row.get("trading_day")
    if row_day is None:
        return FreshnessStatus.NO_DATA, None

    if isinstance(row_day, datetime):
        ts = row_day
        row_day = row_day.date()
    elif isinstance(row_day, str):
        try:
            row_day = date.fromisoformat(row_day)
            ts = datetime.combine(row_day, datetime.min.time())
        except ValueError:
            return FreshnessStatus.NO_DATA, None
    else:
        ts = datetime.combine(row_day, datetime.min.time())

    diff = (current_trading_day - row_day).days
    if diff == 0:
        return FreshnessStatus.FRESH, ts
    if diff == 1:
        return FreshnessStatus.PRIOR_DAY, ts
    if diff > 1:
        return FreshnessStatus.STALE, ts
    return FreshnessStatus.FRESH, ts


# ==========================================================================
# Overall status derivation
# ==========================================================================


def _derive_overall_status(
    conditions: tuple[ConditionRecord, ...],
    freshness: FreshnessStatus,
) -> OverallStatus:
    """Derive the overall report status from individual conditions + freshness."""
    if freshness == FreshnessStatus.NO_DATA:
        return OverallStatus.DATA_MISSING

    has_data_missing = False
    has_pre_session_fail = False
    has_pending = False

    for c in conditions:
        if c.status == ConditionStatus.DATA_MISSING:
            has_data_missing = True
        elif c.status == ConditionStatus.FAIL and c.category in (
            ConditionCategory.PRE_SESSION,
            ConditionCategory.OVERLAY,
        ):
            has_pre_session_fail = True
        elif c.status == ConditionStatus.PENDING:
            has_pending = True

    if has_pre_session_fail:
        return OverallStatus.INELIGIBLE
    if has_data_missing:
        return OverallStatus.DATA_MISSING
    if has_pending:
        return OverallStatus.NEEDS_LIVE_DATA
    return OverallStatus.ELIGIBLE


# ==========================================================================
# Main entry point
# ==========================================================================


def build_eligibility_report(
    strategy_id: str,
    trading_day: date,
    feature_row: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> EligibilityReport:
    """Build an eligibility report for a strategy on a trading day.

    Thin adapter over canonical filter self-description. The flow is:

      1. parse_strategy_id → dimensions
      2. ALL_FILTERS[filter_type] → filter instance (raise on unknown)
      3. _walk_filter_atoms → list[(leaf_filter_type, AtomDescription)]
      4. _atom_to_condition for each pair → list[ConditionRecord]
      5. Aggregate atom error_messages into build_errors
      6. _build_calendar_condition → calendar overlay
      7. _build_atr_velocity_condition → ATR velocity overlay (if applies)
      8. _derive_overall_status → summary

    Args:
        strategy_id: Fully-qualified strategy ID (e.g.
            MNQ_NYSE_CLOSE_E2_RR2.0_CB1_COST_LT10).
        trading_day: The trading day this report is for (Brisbane TZ).
        feature_row: Pre-loaded daily_features row. If None, db_path
            must be provided. Fixture-friendly: tests pass a dict directly.
        db_path: Path to gold.db for fetching feature_row. Ignored if
            feature_row is provided.

    Returns:
        Immutable EligibilityReport.

    Raises:
        ValueError: If strategy_id cannot be parsed OR filter_type is
            not in ALL_FILTERS. Both are caller bugs, not data problems.
    """
    dims = parse_strategy_id(strategy_id)
    instrument = dims["instrument"]
    orb_label = dims["orb_label"]
    filter_type = dims["filter_type"]
    entry_model = dims["entry_model"]

    build_errors: list[str] = []

    if feature_row is None and db_path is not None:
        feature_row = _fetch_feature_row(
            db_path, instrument, trading_day, dims["orb_minutes"], build_errors
        )

    freshness, as_of_ts = _resolve_freshness(feature_row, trading_day)

    # Look up canonical filter — raise on unknown (caller bug)
    if filter_type not in ALL_FILTERS:
        raise ValueError(
            f"Unknown filter_type {filter_type!r} in strategy_id "
            f"{strategy_id!r} — not registered in trading_app.config.ALL_FILTERS"
        )
    filter_instance = ALL_FILTERS[filter_type]

    # Walk the filter tree, producing (source_filter, atom) pairs.
    # Filters expect a dict; pass {} when feature_row is None so they
    # can decide for themselves whether to surface DATA_MISSING.
    # On describe() exception, _walk_filter_atoms returns a synthetic
    # DATA_MISSING atom whose error_message flows into build_errors via
    # the main loop below (single source of truth for error aggregation).
    row_for_describe = feature_row if feature_row is not None else {}
    atom_pairs = _walk_filter_atoms(
        filter_instance, row_for_describe, orb_label, entry_model
    )

    # Translate atoms to ConditionRecords AND collect error_messages.
    # _atom_to_condition also writes into build_errors on enum contract
    # violations (defense-in-depth vs. drift check #85, which runs at
    # check time not runtime).
    conditions: list[ConditionRecord] = []
    for source_filter, atom in atom_pairs:
        if atom.error_message is not None:
            build_errors.append(atom.error_message)
        conditions.append(
            _atom_to_condition(
                atom,
                source_filter,
                instrument,
                orb_label,
                trading_day,
                build_errors,
            )
        )

    # Add overlay conditions
    conditions.append(
        _build_calendar_condition(instrument, orb_label, trading_day, build_errors)
    )
    atr_vel = _build_atr_velocity_condition(
        feature_row, instrument, orb_label, trading_day, build_errors
    )
    if atr_vel is not None:
        conditions.append(atr_vel)

    overall = _derive_overall_status(tuple(conditions), freshness)

    return EligibilityReport(
        strategy_id=strategy_id,
        instrument=instrument,
        session=orb_label,
        entry_model=entry_model,
        trading_day=trading_day,
        as_of_timestamp=as_of_ts,
        freshness_status=freshness,
        conditions=tuple(conditions),
        overall_status=overall,
        data_provenance={
            "feature_row": "daily_features" if feature_row is not None else "none"
        },
        build_errors=tuple(build_errors),
    )


# ==========================================================================
# DB feature row fetch (infrastructure, not filter logic)
# ==========================================================================


def _fetch_feature_row(
    db_path: Path,
    instrument: str,
    trading_day: date,
    orb_minutes: int,
    build_errors: list[str],
) -> dict[str, Any] | None:
    """Fetch one daily_features row for (instrument, trading_day, orb_minutes).

    Returns None if no row found OR an exception was caught. Mutates
    build_errors to surface infrastructure failures (DB connect, schema
    drift, permission denied) rather than swallowing them silently.
    """
    try:
        import duckdb
    except ImportError:
        build_errors.append("duckdb package not importable — cannot fetch feature row")
        return None

    con = None
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception as exc:  # noqa: BLE001
        build_errors.append(
            f"duckdb.connect({db_path}) raised: {type(exc).__name__}: {exc}"
        )
        return None

    try:
        row = con.execute(
            """
            SELECT *
            FROM daily_features
            WHERE symbol = ?
              AND trading_day = ?
              AND orb_minutes = ?
            LIMIT 1
            """,
            [instrument, trading_day, orb_minutes],
        ).fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, row, strict=True))
    except Exception as exc:  # noqa: BLE001
        build_errors.append(
            f"daily_features query raised: {type(exc).__name__}: {exc}"
        )
        return None
    finally:
        if con is not None:
            con.close()
