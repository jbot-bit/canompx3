"""CME market calendar — holidays, early closes, per-minute open checks.

Uses the ``exchange_calendars`` library for authoritative CME schedule.
CMES (equity index: ES/MES, NQ/MNQ) and COMEX (metals: GC/MGC) share
identical holiday calendars for our instruments.

All functions are deterministic from the date alone — no DB or network calls.

Early close = 12:00 PM CT (= 1:00 PM ET) on ~8 days/year:
    MLK Day, Presidents Day, Memorial Day, July 4th, Labor Day,
    Thanksgiving, Black Friday, Christmas Eve.

Full holidays = ~3/year: New Year's Day, Good Friday, Christmas Day.

Library coverage: through ~April 2027. Beyond that, functions fail-open
(return False for is_cme_holiday, True for is_market_open_at) with a
WARNING log. Fail-open protects against missed trading days;
the alternative (fail-closed) would block trading on unknown dates.
"""

import logging
from datetime import date, datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def _get_cmes():
    """Return the cached CMES exchange_calendars handle.

    Lazy-loaded because `import exchange_calendars` + `get_calendar('CMES')`
    is ~1.9s of work paid by every importer of this module — even importers
    that never call any function below. PEP 8 endorses delayed imports for
    performance; @lru_cache(maxsize=1) memoizes the singleton.
    """
    import exchange_calendars as xcals

    return xcals.get_calendar("CMES")


def is_cme_holiday(trading_day: date) -> bool:
    """True if CME is fully closed (no sessions trade). Includes weekends.

    Returns False (fail-open) if the date is beyond library coverage.
    """
    ts = pd.Timestamp(trading_day)
    try:
        sessions = _get_cmes().sessions_in_range(ts, ts)
        return len(sessions) == 0
    except (ValueError, KeyError, IndexError):
        log.warning(
            "Calendar data unavailable for %s — assuming NOT holiday (fail-open)",
            trading_day,
        )
        return False


def is_early_close(trading_day: date) -> bool:
    """True if CME closes early (12:00 PM CT / 1:00 PM ET) on this date.

    A full holiday returns False (it's closed entirely, not early-close).
    """
    ts = pd.Timestamp(trading_day)
    try:
        sessions = _get_cmes().sessions_in_range(ts, ts)
        if len(sessions) == 0:
            return False  # Full holiday or weekend
        close_utc = _get_cmes().session_close(ts)
        # Normal close: 22:00 UTC (EDT) or 23:00 UTC (EST) = 5:00 PM CT
        # Early close: 17:00 UTC (EDT) or 18:00 UTC (EST) = 12:00 PM CT
        # Threshold: if close hour < 20 UTC, it's an early close
        return close_utc.hour < 20
    except (ValueError, KeyError, IndexError):
        log.warning(
            "Cannot determine early close for %s — assuming normal close",
            trading_day,
        )
        return False


def session_close_utc(trading_day: date) -> datetime | None:
    """Actual exchange close time in UTC. None if holiday/weekend."""
    ts = pd.Timestamp(trading_day)
    try:
        sessions = _get_cmes().sessions_in_range(ts, ts)
        if len(sessions) == 0:
            return None
        return _get_cmes().session_close(ts).to_pydatetime()
    except (ValueError, KeyError, IndexError):
        log.warning("Cannot get close time for %s", trading_day)
        return None


def session_close_et(trading_day: date) -> time | None:
    """Exchange close time in ET. None if holiday/weekend."""
    close = session_close_utc(trading_day)
    if close is None:
        return None
    return close.astimezone(_ET).time()


def is_market_open_at(utc_time: datetime) -> bool:
    """True if the CME is open at this specific UTC timestamp.

    Handles overnight sessions, early closes, holidays, weekends, DST.
    Fail-open if library data unavailable.
    """
    ts = pd.Timestamp(utc_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    try:
        return _get_cmes().is_open_on_minute(ts)
    except (ValueError, KeyError, IndexError):
        # ValueError/KeyError: date outside library coverage (post-Apr 2027)
        # IndexError: edge case in library internals
        log.warning("Calendar check failed for %s — assuming OPEN (fail-open)", utc_time)
        return True  # INTENTIONAL fail-open: trade rather than miss opportunities


def effective_close_et(trading_day: date, *, firm_close_et: time) -> time | None:
    """The earlier of firm close time and exchange close time, in ET.

    Returns None on full holidays (no trading at all).
    On early close days, the exchange close (1:00 PM ET) beats most firm
    close times (TopStep 4:10 PM, Tradeify 4:59 PM).
    """
    exchange_close = session_close_et(trading_day)
    if exchange_close is None:
        return None  # Holiday — no trading
    return min(firm_close_et, exchange_close)
