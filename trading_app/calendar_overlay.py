"""
Calendar overlay for per-instrument x session calendar rules.

Replaces the blanket CALENDAR_SKIP_NFP_OPEX with data-driven rules from
the cascade scanner (research/research_calendar_cascade.py).

Rules are loaded from research/output/calendar_cascade_rules.json at import time.
If the JSON is missing or empty, all actions default to NEUTRAL (trade normally).
This is fail-open-to-trade — missing rules never prevent trading.

@research-source research/research_calendar_cascade.py
@revalidated-for E1/E2 event-based sessions (Mar 2026)
"""

import json
from datetime import date
from enum import Enum
from pathlib import Path

from pipeline.calendar_filters import (
    day_of_week,
    is_nfp_day,
    is_opex_day,
)
from pipeline.log import get_logger
from research.research_calendar_effects import (
    _build_cpi_set,
    _build_fomc_set,
    _is_month_end,
    _is_month_start,
    _is_quarter_end,
    _opex_week_dates,
)

logger = get_logger(__name__)


class CalendarAction(Enum):
    """Calendar overlay action for a given instrument x session x day."""

    SKIP = 0.0
    HALF_SIZE = 0.5
    NEUTRAL = 1.0


# =========================================================================
# Rule loading
# =========================================================================

_RULES_PATH = Path(__file__).resolve().parent.parent / "research" / "output" / "calendar_cascade_rules.json"

_ACTION_MAP = {
    "SKIP": CalendarAction.SKIP,
    "HALF_SIZE": CalendarAction.HALF_SIZE,
    "NEUTRAL": CalendarAction.NEUTRAL,
}


def _load_rules() -> dict[tuple[str, str, str], CalendarAction]:
    """Load cascade rules from JSON. Returns empty dict on any error."""
    if not _RULES_PATH.exists():
        logger.warning("Calendar rules file not found: %s — defaulting to NEUTRAL", _RULES_PATH)
        return {}
    try:
        raw = json.loads(_RULES_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse calendar rules JSON: %s — defaulting to NEUTRAL", exc)
        return {}

    rules: dict[tuple[str, str, str], CalendarAction] = {}
    for entry in raw.get("rules", []):
        key = (entry["instrument"], entry["session"], entry["signal"])
        action_str = entry.get("action", "NEUTRAL")
        action = _ACTION_MAP.get(action_str, CalendarAction.NEUTRAL)
        rules[key] = action
    return rules


CALENDAR_RULES: dict[tuple[str, str, str], CalendarAction] = _load_rules()


# =========================================================================
# Calendar signal detection
# =========================================================================

# Pre-built date sets (computed once at import time)
_FOMC_DATES: set[date] = _build_fomc_set()
_CPI_DATES: set[date] = _build_cpi_set()
_OPEX_WEEK_DATES: set[date] = _opex_week_dates()

# Signal name → column mapping (matches cascade scanner names)
# Lazy-built set of all known trading days for month-end/start/quarter-end checks.
# These functions need a set of trading days to count backwards/forwards. In
# production, this doesn't matter much — if a day is close to month boundary and
# happens to be a non-trading day, the window calculation degrades gracefully
# (may be off by 1 day). For perfect accuracy, the cascade scanner uses the full
# trading day set from the database. Here we pass a minimal set.
# TODO: _EMPTY_TRADING_DAYS causes _is_month_end/_is_month_start/_is_quarter_end
# to always return True (count=0 <= window). No impact while CALENDAR_RULES is
# empty, but must be fixed before any MONTH_END/MONTH_START rules go live.
_EMPTY_TRADING_DAYS: set[date] = set()


def _get_active_signals(trading_day: date) -> list[str]:
    """Return list of signal names that fire on this date."""
    signals: list[str] = []

    if is_nfp_day(trading_day):
        signals.append("NFP")
    if is_opex_day(trading_day):
        signals.append("OPEX")
    if trading_day in _FOMC_DATES:
        signals.append("FOMC")
    if trading_day in _CPI_DATES:
        signals.append("CPI")
    if _is_month_end(trading_day, _EMPTY_TRADING_DAYS):
        signals.append("MONTH_END")
    if _is_month_start(trading_day, _EMPTY_TRADING_DAYS):
        signals.append("MONTH_START")
    if _is_quarter_end(trading_day, _EMPTY_TRADING_DAYS):
        signals.append("QUARTER_END")
    if trading_day in _OPEX_WEEK_DATES:
        signals.append("OPEX_WEEK")

    dow = day_of_week(trading_day)
    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    if dow in dow_names:
        signals.append(dow_names[dow])

    return signals


# =========================================================================
# Main lookup
# =========================================================================


def get_calendar_action(
    instrument: str,
    session: str,
    trading_day: date,
) -> CalendarAction:
    """
    Look up the calendar action for this instrument x session x day.

    Checks all calendar signals that fire on trading_day (NFP, OPEX, FOMC, etc.),
    looks up each in CALENDAR_RULES, returns the most restrictive action.

    Returns NEUTRAL if no rules match or if CALENDAR_RULES is empty.
    On ANY exception, logs a warning and returns NEUTRAL (never skip on error).
    """
    try:
        if not CALENDAR_RULES:
            return CalendarAction.NEUTRAL

        signals = _get_active_signals(trading_day)
        if not signals:
            return CalendarAction.NEUTRAL

        most_restrictive = CalendarAction.NEUTRAL
        for sig in signals:
            action = CALENDAR_RULES.get((instrument, session, sig))
            if action is not None and action.value < most_restrictive.value:
                most_restrictive = action

        return most_restrictive
    except Exception:
        logger.warning(
            "Error in get_calendar_action(%s, %s, %s) — defaulting to NEUTRAL",
            instrument,
            session,
            trading_day,
            exc_info=True,
        )
        return CalendarAction.NEUTRAL
