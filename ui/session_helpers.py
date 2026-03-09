# ui/session_helpers.py
"""
Session scheduling, filter translation, and briefing card builder.

DST-SAFE: All session time logic uses chronological datetime comparison.
No `hour < 9` boundary checks. No trading-day logic for "what's next."
Trading-day concept used ONLY for database queries.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.build_daily_features import compute_trading_day
from pipeline.dst import SESSION_CATALOG, is_uk_dst, is_us_dst

BRISBANE = ZoneInfo("Australia/Brisbane")

# Awake hours (Brisbane). Sessions outside this range are "overnight."
AWAKE_START = int(os.getenv("DASHBOARD_AWAKE_START", "7"))
AWAKE_END = int(os.getenv("DASHBOARD_AWAKE_END", "21"))


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class AppState:
    """Current state of the co-pilot UI."""

    name: str  # WEEKEND, IDLE, APPROACHING, ALERT, LIVE, POST, OVERNIGHT
    next_session: str | None = None
    next_session_dt: datetime | None = None
    minutes_to_next: float | None = None
    then_session: str | None = None  # back-to-back session within 30 min
    then_session_dt: datetime | None = None
    next_monday: date | None = None  # only set for WEEKEND state
    trading_day: date | None = None


@dataclass
class SessionBriefing:
    """Merged briefing card for one instrument at one session."""

    session: str
    instrument: str
    conditions: list[str]  # Human-readable filter conditions
    rr_target: float
    entry_instruction: str  # e.g. "Place stop-market at ORB edge"
    direction_note: str | None  # e.g. "Long breakouts only"
    session_hour: int  # Hour in Brisbane (for display)
    session_minute: int  # Minute in Brisbane
    orb_minutes: int = 5  # ORB aperture
    strategy_count: int = 1  # How many underlying strategies merged


# ── Session time resolution (DST-safe) ──────────────────────────────────────


def get_upcoming_sessions(now: datetime) -> list[tuple[str, datetime]]:
    """Get all sessions in the next ~36 hours, sorted chronologically.

    DST-safe: computes session datetimes for today AND tomorrow using each
    session's per-day resolver. No `hour < 9` logic. No trading-day boundaries.
    Pure chronological comparison.
    """
    today = now.date()
    tomorrow = today + timedelta(days=1)

    # Cutoff: sessions beyond 9:00 AM tomorrow are next cycle's sessions,
    # not "upcoming" from the current perspective.
    cutoff = datetime.combine(tomorrow, time(9, 0), tzinfo=BRISBANE)

    all_sessions: list[tuple[str, datetime]] = []
    for cal_day in [today, tomorrow]:
        for name, entry in SESSION_CATALOG.items():
            h, m = entry["resolver"](cal_day)
            dt = datetime.combine(cal_day, time(h, m), tzinfo=BRISBANE)
            if dt < cutoff:
                all_sessions.append((name, dt))

    # Sort chronologically
    all_sessions.sort(key=lambda x: x[1])

    # Keep only future, deduplicate by name (first occurrence wins)
    seen: set[str] = set()
    result: list[tuple[str, datetime]] = []
    for name, dt in all_sessions:
        if dt > now and name not in seen:
            seen.add(name)
            result.append((name, dt))

    return result


def current_trading_day(now: datetime) -> date:
    """Get the current trading day using the pipeline's logic.

    Uses compute_trading_day from build_daily_features — the same function
    that assigns bars to trading days. Guarantees consistency with DB data.
    """
    ts = pd.Timestamp(now)
    return compute_trading_day(ts)


def is_weekend(now: datetime) -> bool:
    """Check if markets are closed (Saturday or Sunday calendar day).

    Saturday/Sunday calendar days have no trading sessions.
    Friday's overnight sessions (NYSE_CLOSE at 7 AM Sat) are part of
    Friday's trading day — by Saturday morning they're complete.
    CME futures close Friday ~5 PM CT and reopen Sunday ~5 PM CT.
    In Brisbane: closed from Saturday ~9 AM to Monday ~9 AM.
    """
    return now.weekday() >= 5  # 5=Saturday, 6=Sunday


def _next_monday(from_date: date) -> date:
    """Find the next Monday on or after from_date."""
    days_ahead = (7 - from_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_date + timedelta(days=days_ahead)


def _is_in_awake_hours(dt: datetime) -> bool:
    """Check if a datetime falls within configured awake hours."""
    return AWAKE_START <= dt.hour < AWAKE_END


# ── Filter translation ──────────────────────────────────────────────────────

_G_PATTERN = re.compile(r"^ORB_G(\d+)(.*)$")

_MODIFIERS = {
    "_CONT": " + continuation bar",
    "_FAST5": " + fast break (5 bars)",
    "_FAST10": " + fast break (10 bars)",
    "_NOMON": " + not Monday",
    "_NOTUE": " + not Tuesday",
    "_NOFRI": " + not Friday",
}

_SPECIAL_FILTERS = {
    "NO_FILTER": "All days (no filter)",
    "VOL_RV12_N20": "Realized vol in top 20%",
    "DIR_LONG": "Long breakouts only",
    "DIR_SHORT": "Short breakouts only",
}


def filter_to_english(filter_type: str) -> str:
    """Translate a filter machine name to human-readable English.

    Auto-parses G-filter naming convention (ORB_G{N}{_MODIFIER}).
    Falls back to the raw name for unknown filters.
    """
    # Check special filters first
    if filter_type in _SPECIAL_FILTERS:
        return _SPECIAL_FILTERS[filter_type]

    # Parse G-filter pattern
    match = _G_PATTERN.match(filter_type)
    if match:
        threshold = match.group(1)
        modifier_key = match.group(2)
        base = f"ORB >= {threshold}pts"
        modifier_text = _MODIFIERS.get(modifier_key, modifier_key)
        return base + modifier_text

    # Unknown — return as-is
    return filter_type


# ── Briefing card builder ────────────────────────────────────────────────────


def build_session_briefings() -> list[SessionBriefing]:
    """Build merged briefing cards from LIVE_PORTFOLIO.

    Groups all strategies by (session, instrument) and merges their filter
    conditions into a single human-readable instruction card.
    """
    from datetime import date as date_type

    from pipeline.asset_configs import get_active_instruments
    from trading_app.live_config import build_live_portfolio

    today = date_type.today()

    # Collect all strategies across all instruments
    all_strategies: list = []
    for instrument in get_active_instruments():
        try:
            portfolio, _ = build_live_portfolio(instrument=instrument)
            all_strategies.extend(portfolio.strategies)
        except Exception:
            logging.warning("Failed to build briefings for %s", instrument, exc_info=True)
            continue

    # Group by (session, instrument)
    groups: dict[tuple[str, str], list] = {}
    for s in all_strategies:
        key = (s.orb_label, s.instrument)
        groups.setdefault(key, []).append(s)

    briefings: list[SessionBriefing] = []
    for (session, instrument), strats in groups.items():
        # Merge filter conditions
        conditions: list[str] = []
        direction_note = None
        rr_target = strats[0].rr_target  # All same session share RR

        for s in strats:
            english = filter_to_english(s.filter_type)
            if s.filter_type.startswith("DIR_"):
                direction_note = english
            else:
                if english not in conditions:
                    conditions.append(english)

        # Resolve session time for today
        if session in SESSION_CATALOG:
            h, m = SESSION_CATALOG[session]["resolver"](today)
        else:
            h, m = 0, 0

        # Entry instruction — all current strategies are E2
        entry_model = strats[0].entry_model
        if entry_model == "E2":
            entry_instruction = "Place stop-market at ORB edge"
        elif entry_model == "E1":
            entry_instruction = "Market order after confirm bar closes"
        else:
            entry_instruction = f"Entry model: {entry_model}"

        briefings.append(
            SessionBriefing(
                session=session,
                instrument=instrument,
                conditions=conditions,
                rr_target=rr_target,
                entry_instruction=entry_instruction,
                direction_note=direction_note,
                session_hour=h,
                session_minute=m,
                orb_minutes=strats[0].orb_minutes,
                strategy_count=len(strats),
            )
        )

    return briefings


# ── App state machine ────────────────────────────────────────────────────────


def get_app_state(now: datetime) -> AppState:
    """Determine the current UI state based on time.

    States:
      WEEKEND     — Saturday/Sunday, no sessions
      OVERNIGHT   — Next session is outside awake hours
      IDLE        — >60 min to next session
      APPROACHING — 15-60 min to next session
      ALERT       — <15 min to next session
    """
    trading_day = current_trading_day(now)

    # Weekend check
    if is_weekend(now):
        monday = _next_monday(now.date())
        return AppState(
            name="WEEKEND",
            next_monday=monday,
            trading_day=trading_day,
        )

    # Find upcoming sessions
    upcoming = get_upcoming_sessions(now)
    if not upcoming:
        # No sessions found (shouldn't happen on weekdays)
        return AppState(name="IDLE", trading_day=trading_day)

    next_name, next_dt = upcoming[0]
    minutes_to_next = (next_dt - now).total_seconds() / 60

    # Check for back-to-back session (within 30 min of next)
    then_name = None
    then_dt = None
    if len(upcoming) >= 2:
        second_name, second_dt = upcoming[1]
        gap = (second_dt - next_dt).total_seconds() / 60
        if gap <= 30:
            then_name = second_name
            then_dt = second_dt

    base = AppState(
        next_session=next_name,
        next_session_dt=next_dt,
        minutes_to_next=minutes_to_next,
        then_session=then_name,
        then_session_dt=then_dt,
        trading_day=trading_day,
        name="",  # set below
    )

    # Determine state
    if not _is_in_awake_hours(next_dt) and minutes_to_next > 60:
        base.name = "OVERNIGHT"
    elif minutes_to_next <= 15:
        base.name = "ALERT"
    elif minutes_to_next <= 60:
        base.name = "APPROACHING"
    else:
        base.name = "IDLE"

    return base


def get_refresh_seconds(minutes_to_next: float, is_weekend: bool = False) -> int:
    """Adaptive refresh rate based on time to next session."""
    if is_weekend:
        return 120
    if minutes_to_next <= 15:
        return 5
    if minutes_to_next <= 60:
        return 15
    return 30


# ── DST transition awareness ────────────────────────────────────────────────


@dataclass
class DSTChange:
    """A session whose Brisbane time shifted due to a DST transition."""

    session: str
    old_hour: int
    old_minute: int
    new_hour: int
    new_minute: int
    shift_minutes: int  # negative = earlier, positive = later


@dataclass
class UpcomingDSTTransition:
    """An upcoming DST transition date within the lookahead window."""

    region: str  # "US" or "UK"
    transition_date: date
    direction: str  # "start" (clocks forward) or "end" (clocks back)
    days_away: int


def get_dst_session_changes(today: date) -> list[DSTChange]:
    """Compare today's session times vs yesterday's. Return any that shifted.

    Skips weekends (Saturday/Sunday) — compares against the most recent
    weekday to avoid false positives from weekend gaps.
    """
    # Find the most recent prior weekday
    yesterday = today - timedelta(days=1)
    while yesterday.weekday() >= 5:  # skip Sat/Sun
        yesterday -= timedelta(days=1)

    changes: list[DSTChange] = []
    for name, entry in SESSION_CATALOG.items():
        h_old, m_old = entry["resolver"](yesterday)
        h_new, m_new = entry["resolver"](today)
        shift = (h_new * 60 + m_new) - (h_old * 60 + m_old)
        # Handle midnight wrap (e.g. 00:30 -> 23:30 is -60min, not +1380)
        if shift > 720:
            shift -= 1440
        elif shift < -720:
            shift += 1440
        if shift != 0:
            changes.append(
                DSTChange(
                    session=name,
                    old_hour=h_old,
                    old_minute=m_old,
                    new_hour=h_new,
                    new_minute=m_new,
                    shift_minutes=shift,
                )
            )
    return changes


def get_recent_dst_changes(today: date, lookback_days: int = 3) -> list[DSTChange]:
    """Return DST changes from the last N days (deduplicated by session)."""
    seen: set[str] = set()
    result: list[DSTChange] = []
    for offset in range(lookback_days):
        check_date = today - timedelta(days=offset)
        if check_date.weekday() >= 5:
            continue
        for change in get_dst_session_changes(check_date):
            if change.session not in seen:
                seen.add(change.session)
                result.append(change)
    return result


def get_upcoming_dst_transitions(today: date, lookahead_days: int = 30) -> list[UpcomingDSTTransition]:
    """Find upcoming DST transitions within the lookahead window.

    Scans day-by-day for US and UK DST state changes.
    """
    transitions: list[UpcomingDSTTransition] = []

    us_dst_today = is_us_dst(today)
    uk_dst_today = is_uk_dst(today)

    for offset in range(1, lookahead_days + 1):
        check = today + timedelta(days=offset)

        us_dst_check = is_us_dst(check)
        if us_dst_check != us_dst_today and not any(t.region == "US" for t in transitions):
            transitions.append(
                UpcomingDSTTransition(
                    region="US",
                    transition_date=check,
                    direction="start" if us_dst_check else "end",
                    days_away=offset,
                )
            )

        uk_dst_check = is_uk_dst(check)
        if uk_dst_check != uk_dst_today and not any(t.region == "UK" for t in transitions):
            transitions.append(
                UpcomingDSTTransition(
                    region="UK",
                    transition_date=check,
                    direction="start" if uk_dst_check else "end",
                    days_away=offset,
                )
            )

        if len(transitions) >= 2:
            break

    return transitions
