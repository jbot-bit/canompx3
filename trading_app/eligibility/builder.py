"""Eligibility report builder.

Given a strategy, a trading day, and a database path (or a pre-loaded feature
row), produces an immutable `EligibilityReport` with explicit status for every
atomic condition that gates that strategy's trades.

Design principles (from docs/plans/2026-04-07-eligibility-context-design.md):
- Nine explicit statuses, never silent defaults
- Atomic decomposition — composites explode into per-component conditions
- Fixture-friendly — accepts either a DB path or a pre-loaded feature row
- Read-only — never mutates data or schema
- Fail-loud — missing data surfaces as DATA_MISSING, not silent FAIL

Grounding: Pardo Ch.4 (look-ahead) and Aronson Ch.6 (confirmation bias) are
the two load-bearing references for this design. Both PDFs are at
resources/; citations are from training memory unless the builder explicitly
reads them.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from trading_app.eligibility.decomposition import (
    AtomSpec,
    atr_velocity_atom_template,
    calendar_atom_template,
    decompose,
)
from trading_app.eligibility.types import (
    ConditionCategory,
    ConditionRecord,
    ConditionStatus,
    EligibilityReport,
    FreshnessStatus,
    OverallStatus,
)

# Freshness threshold for STALE_VALIDATION status (days).
VALIDATION_FRESHNESS_DAYS = 180

# Path to calendar cascade rules file (for RULES_NOT_LOADED detection).
_CALENDAR_RULES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "research" / "output" / "calendar_cascade_rules.json"
)


# ==========================================================================
# Strategy ID parsing
# ==========================================================================


def parse_strategy_id(strategy_id: str) -> dict[str, Any]:
    """Extract dimensions from a strategy_id string.

    Format examples:
      MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6
      MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12
      MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT10
      MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100
      MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10_O15  (with aperture suffix)

    Returns dict with keys: instrument, orb_label, entry_model, rr_target,
    confirm_bars, filter_type, orb_minutes (default 5).

    Raises ValueError on unparseable input — no silent failure.
    """
    parts = strategy_id.split("_")
    if len(parts) < 6:
        raise ValueError(f"strategy_id too short to parse: {strategy_id!r}")

    instrument = parts[0]
    if instrument not in ("MGC", "MNQ", "MES"):
        raise ValueError(f"strategy_id has unknown instrument {instrument!r}: {strategy_id!r}")

    # Find entry model (E1, E2, E3)
    em_idx = None
    for i, p in enumerate(parts):
        if p in ("E1", "E2", "E3"):
            em_idx = i
            break
    if em_idx is None:
        raise ValueError(f"strategy_id has no entry model (E1/E2/E3): {strategy_id!r}")

    orb_label = "_".join(parts[1:em_idx])
    entry_model = parts[em_idx]

    # RR target: RR{float}
    if em_idx + 1 >= len(parts) or not parts[em_idx + 1].startswith("RR"):
        raise ValueError(f"strategy_id missing RR after entry model: {strategy_id!r}")
    rr_target = float(parts[em_idx + 1][2:])

    # Confirm bars: CB{int}
    if em_idx + 2 >= len(parts) or not parts[em_idx + 2].startswith("CB"):
        raise ValueError(f"strategy_id missing CB after RR: {strategy_id!r}")
    confirm_bars = int(parts[em_idx + 2][2:])

    # Filter type is everything remaining, minus optional aperture suffix _O{N}
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
# Feature row helpers
# ==========================================================================


def _resolve_observed(
    atom: AtomSpec,
    feature_row: dict[str, Any] | None,
    orb_label: str,
) -> tuple[Any, bool]:
    """Resolve the observed value for an atom from a feature row.

    Returns (value, is_missing). `is_missing=True` means the required feature
    is absent or NULL — the builder will produce DATA_MISSING status (not FAIL).
    """
    col = atom.resolve_feature_column(orb_label)
    if col is None:
        return None, False  # overlay atoms have no direct column
    if feature_row is None:
        return None, True
    if col not in feature_row:
        return None, True
    val = feature_row[col]
    if val is None:
        return None, True
    return val, False


def _compare(observed: Any, threshold: Any, comparator: str) -> bool:
    """Evaluate observed vs threshold using a comparator string."""
    try:
        if comparator == ">=":
            return observed >= threshold
        if comparator == "<=":
            return observed <= threshold
        if comparator == ">":
            return observed > threshold
        if comparator == "<":
            return observed < threshold
        if comparator == "==":
            return observed == threshold
        if comparator == "!=":
            return observed != threshold
        if comparator == "in_set":
            return observed in threshold
    except TypeError:
        return False
    return False


# ==========================================================================
# Special atom resolvers (PDR, GAP, DOW, cost ratio, pit range)
#
# Some atoms need more than a simple feature_column lookup. These resolvers
# compute the derived value (e.g., prev_day_range / atr_20) before comparing.
# ==========================================================================


def _resolve_pdr(feature_row: dict[str, Any]) -> tuple[float | None, bool]:
    """prev_day_range / atr_20 ratio."""
    pdr = feature_row.get("prev_day_range")
    atr = feature_row.get("atr_20")
    if pdr is None or atr is None or atr <= 0:
        return None, True
    return pdr / atr, False


def _resolve_gap(feature_row: dict[str, Any]) -> tuple[float | None, bool]:
    """abs(gap_open_points) / atr_20 ratio."""
    gap = feature_row.get("gap_open_points")
    atr = feature_row.get("atr_20")
    if gap is None or atr is None or atr <= 0:
        return None, True
    return abs(gap) / atr, False


def _resolve_dow(trading_day: date) -> tuple[str, bool]:
    """Day of week name for the trading day."""
    dow_names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    return dow_names[trading_day.weekday()], False


# ==========================================================================
# Atom -> ConditionRecord
# ==========================================================================


def _is_validated_for(atom: AtomSpec, instrument: str, session: str) -> bool:
    """Check if this atom has been validated for (instrument, session).

    Returns True when:
    - atom has no validated_for constraint (applies everywhere by default), OR
    - the (instrument, session) tuple is in the validated_for list
    """
    if not atom.validated_for:
        return True
    return (instrument, session) in atom.validated_for


def _atom_to_condition(
    atom: AtomSpec,
    feature_row: dict[str, Any] | None,
    instrument: str,
    session: str,
    trading_day: date,
    freshness: FreshnessStatus,
) -> ConditionRecord:
    """Build a ConditionRecord from an AtomSpec and the current feature snapshot."""
    # Check per-instrument validity first — applies even before data
    if not _is_validated_for(atom, instrument, session):
        return ConditionRecord(
            name=atom.name,
            category=atom.category,
            status=ConditionStatus.NOT_APPLICABLE_INSTRUMENT,
            resolves_at=atom.resolves_at,
            observed_value=None,
            threshold=atom.threshold,
            comparator=atom.comparator,
            source_filter=atom.source_filter,
            validated_for=atom.validated_for,
            last_revalidated=atom.last_revalidated,
            confidence_tier=atom.confidence_tier,
            explanation=atom.explanation,
        )

    # Check validation freshness
    if atom.last_revalidated is not None:
        age_days = (trading_day - atom.last_revalidated).days
        if age_days > VALIDATION_FRESHNESS_DAYS:
            return ConditionRecord(
                name=atom.name,
                category=atom.category,
                status=ConditionStatus.STALE_VALIDATION,
                resolves_at=atom.resolves_at,
                observed_value=None,
                threshold=atom.threshold,
                comparator=atom.comparator,
                source_filter=atom.source_filter,
                validated_for=atom.validated_for,
                last_revalidated=atom.last_revalidated,
                confidence_tier=atom.confidence_tier,
                explanation=f"{atom.explanation} (validation {age_days}d old)",
            )

    # Intra-session atoms are PENDING at startup regardless of data state
    if atom.category == ConditionCategory.INTRA_SESSION:
        status = ConditionStatus.PENDING
        observed = None
    elif atom.category == ConditionCategory.DIRECTIONAL:
        # Direction unknown until break — report as NOT_APPLICABLE_DIRECTION
        status = ConditionStatus.NOT_APPLICABLE_DIRECTION
        observed = None
    else:
        # PRE_SESSION or OVERLAY — resolve from feature row
        # Special resolvers for derived values
        source = atom.source_filter
        if source.startswith("PDR_R") and feature_row is not None:
            observed, missing = _resolve_pdr(feature_row)
        elif source.startswith("GAP_R") and feature_row is not None:
            observed, missing = _resolve_gap(feature_row)
        elif source in ("NOMON", "NOFRI", "NOTUE"):
            observed, missing = _resolve_dow(trading_day)
        else:
            observed, missing = _resolve_observed(atom, feature_row, session)

        if missing:
            status = ConditionStatus.DATA_MISSING
        elif freshness == FreshnessStatus.STALE:
            # Still show the observed value but mark stale
            status = ConditionStatus.FAIL if not _compare(observed, atom.threshold, atom.comparator) else ConditionStatus.PASS
            # Note: freshness is reported at report level; per-atom stays PASS/FAIL
        else:
            passed = _compare(observed, atom.threshold, atom.comparator)
            status = ConditionStatus.PASS if passed else ConditionStatus.FAIL

    return ConditionRecord(
        name=atom.name,
        category=atom.category,
        status=status,
        resolves_at=atom.resolves_at,
        observed_value=observed,
        threshold=atom.threshold,
        comparator=atom.comparator,
        source_filter=atom.source_filter,
        validated_for=atom.validated_for,
        last_revalidated=atom.last_revalidated,
        confidence_tier=atom.confidence_tier,
        explanation=atom.explanation,
    )


# ==========================================================================
# Overlay builders
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
) -> ConditionRecord:
    """Build the calendar overlay condition.

    Returns RULES_NOT_LOADED if the rules file is missing or empty —
    NOT silent NEUTRAL. This is the v2 gap fix.

    Args:
        instrument: strategy instrument for rule lookup.
        session: orb_label for rule lookup.
        trading_day: the trading day for signal detection.
    """
    _ = instrument  # currently unused; kept for rule lookup when file present
    atom = calendar_atom_template()
    if not _calendar_rules_loaded():
        return ConditionRecord(
            name=atom.name,
            category=atom.category,
            status=ConditionStatus.RULES_NOT_LOADED,
            resolves_at=atom.resolves_at,
            observed_value=None,
            threshold=atom.threshold,
            comparator=atom.comparator,
            source_filter=atom.source_filter,
            validated_for=atom.validated_for,
            last_revalidated=atom.last_revalidated,
            confidence_tier=atom.confidence_tier,
            explanation=(
                f"{atom.explanation} calendar_cascade_rules.json not found at "
                f"{_CALENDAR_RULES_PATH} — rules system is unavailable."
            ),
        )

    # Rules loaded — look up the action
    try:
        from trading_app.calendar_overlay import CalendarAction, get_calendar_action

        action = get_calendar_action(instrument, session, trading_day)
        if action == CalendarAction.NEUTRAL:
            status = ConditionStatus.PASS
        elif action == CalendarAction.HALF_SIZE:
            status = ConditionStatus.FAIL  # non-neutral = partial block
        else:  # SKIP
            status = ConditionStatus.FAIL
        observed = action.name
    except Exception as exc:  # noqa: BLE001 — surface as DATA_MISSING, not silent
        return ConditionRecord(
            name=atom.name,
            category=atom.category,
            status=ConditionStatus.DATA_MISSING,
            resolves_at=atom.resolves_at,
            observed_value=None,
            threshold=atom.threshold,
            comparator=atom.comparator,
            source_filter=atom.source_filter,
            validated_for=atom.validated_for,
            last_revalidated=atom.last_revalidated,
            confidence_tier=atom.confidence_tier,
            explanation=f"{atom.explanation} calendar lookup raised: {type(exc).__name__}: {exc}",
        )

    return ConditionRecord(
        name=atom.name,
        category=atom.category,
        status=status,
        resolves_at=atom.resolves_at,
        observed_value=observed,
        threshold=atom.threshold,
        comparator=atom.comparator,
        source_filter=atom.source_filter,
        validated_for=atom.validated_for,
        last_revalidated=atom.last_revalidated,
        confidence_tier=atom.confidence_tier,
        explanation=atom.explanation,
    )


def _build_atr_velocity_condition(
    instrument: str,
    session: str,
    feature_row: dict[str, Any] | None,
    trading_day: date,
) -> ConditionRecord | None:
    """Build the ATR velocity overlay condition.

    Returns None if this overlay does not apply to (instrument, session).
    Otherwise returns a ConditionRecord with explicit status.
    """
    _ = trading_day  # reserved for future freshness checks on regime column
    atom = atr_velocity_atom_template()
    if not _is_validated_for(atom, instrument, session):
        return None

    if feature_row is None:
        return ConditionRecord(
            name=atom.name,
            category=atom.category,
            status=ConditionStatus.DATA_MISSING,
            resolves_at=atom.resolves_at,
            observed_value=None,
            threshold=atom.threshold,
            comparator=atom.comparator,
            source_filter=atom.source_filter,
            validated_for=atom.validated_for,
            last_revalidated=atom.last_revalidated,
            confidence_tier=atom.confidence_tier,
            explanation=atom.explanation,
        )

    atr_vel = feature_row.get("atr_vel_regime")
    compression = feature_row.get(f"orb_{session}_compression_tier")
    if atr_vel is None or compression is None:
        return ConditionRecord(
            name=atom.name,
            category=atom.category,
            status=ConditionStatus.DATA_MISSING,
            resolves_at=atom.resolves_at,
            observed_value=None,
            threshold=atom.threshold,
            comparator=atom.comparator,
            source_filter=atom.source_filter,
            validated_for=atom.validated_for,
            last_revalidated=atom.last_revalidated,
            confidence_tier=atom.confidence_tier,
            explanation=atom.explanation,
        )

    # Contracting + {Neutral or Compressed} = FAIL (skip)
    is_skip = atr_vel == "Contracting" and compression in ("Neutral", "Compressed")
    status = ConditionStatus.FAIL if is_skip else ConditionStatus.PASS
    return ConditionRecord(
        name=atom.name,
        category=atom.category,
        status=status,
        resolves_at=atom.resolves_at,
        observed_value=f"{atr_vel}/{compression}",
        threshold=atom.threshold,
        comparator=atom.comparator,
        source_filter=atom.source_filter,
        validated_for=atom.validated_for,
        last_revalidated=atom.last_revalidated,
        confidence_tier=atom.confidence_tier,
        explanation=atom.explanation,
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

    # Convert to date if it's a datetime
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
    # Future date — treat as fresh (shouldn't happen in practice)
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

    Args:
        strategy_id: Fully-qualified strategy ID (e.g., MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10)
        trading_day: The trading day this report is for (Brisbane timezone).
            Callers should use `pipeline.build_daily_features.compute_trading_day`
            to resolve this from a UTC timestamp — never `date.today()` directly.
        feature_row: Pre-loaded daily_features row. If None, db_path must be
            provided and the builder will fetch the row. Fixture-friendly:
            tests pass a dict directly, no DB needed.
        db_path: Path to gold.db for fetching feature_row. Ignored if
            feature_row is provided.

    Returns:
        An immutable EligibilityReport. Never raises on missing data — all
        problems surface as DATA_MISSING / STALE / RULES_NOT_LOADED statuses.

    Raises:
        ValueError: If strategy_id cannot be parsed. Parsing errors are the
            only hard failure — a malformed strategy_id is a caller bug, not
            a data problem.
    """
    dims = parse_strategy_id(strategy_id)
    instrument = dims["instrument"]
    orb_label = dims["orb_label"]
    filter_type = dims["filter_type"]
    entry_model = dims["entry_model"]

    # Fetch feature row from DB if not provided
    if feature_row is None and db_path is not None:
        feature_row = _fetch_feature_row(db_path, instrument, trading_day, dims["orb_minutes"])

    freshness, as_of_ts = _resolve_freshness(feature_row, trading_day)

    # Decompose the filter_type into atomic specs
    atoms = decompose(filter_type)

    # Build a condition for each atom
    conditions: list[ConditionRecord] = []
    build_errors: list[str] = []

    for atom in atoms:
        # CONT atoms are E2-excluded (look-ahead for stop-market entries)
        if atom.source_filter == "CONT" and entry_model == "E2":
            conditions.append(
                ConditionRecord(
                    name=atom.name,
                    category=atom.category,
                    status=ConditionStatus.NOT_APPLICABLE_INSTRUMENT,  # close-enough semantically
                    resolves_at=atom.resolves_at,
                    observed_value=None,
                    threshold=atom.threshold,
                    comparator=atom.comparator,
                    source_filter=atom.source_filter,
                    validated_for=atom.validated_for,
                    last_revalidated=atom.last_revalidated,
                    confidence_tier=atom.confidence_tier,
                    explanation=(
                        f"{atom.explanation} Not applicable for E2 entry model "
                        "(break bar closes after stop-market touches — look-ahead)."
                    ),
                )
            )
            continue
        conditions.append(
            _atom_to_condition(atom, feature_row, instrument, orb_label, trading_day, freshness)
        )

    # Add overlay conditions (calendar always, ATR velocity conditionally)
    conditions.append(_build_calendar_condition(instrument, orb_label, trading_day))
    atr_vel_condition = _build_atr_velocity_condition(instrument, orb_label, feature_row, trading_day)
    if atr_vel_condition is not None:
        conditions.append(atr_vel_condition)

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
        data_provenance={"feature_row": "daily_features" if feature_row is not None else "none"},
        build_errors=tuple(build_errors),
    )


def _fetch_feature_row(
    db_path: Path,
    instrument: str,
    trading_day: date,
    orb_minutes: int,
) -> dict[str, Any] | None:
    """Fetch one daily_features row for (instrument, trading_day, orb_minutes).

    Returns None if no row found. Does not raise — missing data is normal
    and surfaces as NO_DATA freshness.
    """
    try:
        import duckdb
    except ImportError:
        return None

    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception:  # noqa: BLE001 — return None on any connection failure
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
    except Exception:  # noqa: BLE001 — DB errors surface as NO_DATA
        return None
    finally:
        con.close()
