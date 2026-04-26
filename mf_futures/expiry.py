"""Source-backed expiry-date surface for supported `mf_futures` contracts.

This module is intentionally narrow. It supports only the contract families
whose expiry rules are verified from official CME/COMEX sources and needed for
the current research surface. Unsupported symbols fail closed by returning
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True)
class ExpiryRule:
    """Deterministic expiry rule for one contract family."""

    calendar_name: str
    rule_name: str


_RULES: dict[str, ExpiryRule] = {
    "MES": ExpiryRule(calendar_name="CMES", rule_name="third_friday_quarterly"),
    "MNQ": ExpiryRule(calendar_name="CMES", rule_name="third_friday_quarterly"),
    "GC": ExpiryRule(calendar_name="COMEX", rule_name="third_last_business_day"),
    "MGC": ExpiryRule(calendar_name="COMEX", rule_name="third_last_business_day"),
}


def supported_expiry_rule(symbol: str) -> ExpiryRule | None:
    """Return the supported expiry rule for a stats symbol, if any."""
    return _RULES.get(symbol)


def compute_expiry_date(symbol: str, *, contract_year: int, contract_month: int) -> date | None:
    """Return the source-backed expiry date for one supported contract."""
    rule = supported_expiry_rule(symbol)
    if rule is None:
        return None

    try:
        calendar = xcals.get_calendar(rule.calendar_name)
    except Exception:
        return None

    if rule.rule_name == "third_friday_quarterly":
        if contract_month not in (3, 6, 9, 12):
            return None
        return _third_friday_session_date(calendar, contract_year, contract_month)

    if rule.rule_name == "third_last_business_day":
        return _nth_last_session_date(calendar, contract_year, contract_month, n=3)

    return None


def _third_friday_session_date(calendar: xcals.ExchangeCalendar, year: int, month: int) -> date | None:
    month_start = date(year, month, 1)
    first_of_next_month = date(year + (month // 12), (month % 12) + 1, 1)
    month_end = first_of_next_month - timedelta(days=1)
    sessions = _session_dates(calendar, month_start, month_end)
    if sessions is None:
        return None

    current = month_start
    friday_count = 0
    while current.month == month:
        if current.weekday() == 4:
            friday_count += 1
            if friday_count == 3:
                return current if current in sessions else None
        current += timedelta(days=1)
    return None


def _nth_last_session_date(
    calendar: xcals.ExchangeCalendar,
    year: int,
    month: int,
    *,
    n: int,
) -> date | None:
    if n <= 0:
        raise ValueError("n must be positive")

    month_start = date(year, month, 1)
    first_of_next_month = date(year + (month // 12), (month % 12) + 1, 1)
    month_end = first_of_next_month - timedelta(days=1)
    sessions = _session_dates(calendar, month_start, month_end)
    if sessions is None or len(sessions) < n:
        return None
    return sessions[-n]


def _session_dates(
    calendar: xcals.ExchangeCalendar,
    start: date,
    end: date,
) -> list[date] | None:
    try:
        sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    except Exception:
        return None
    return [session.date() for session in sessions]
