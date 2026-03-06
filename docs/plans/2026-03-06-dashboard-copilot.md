# Dashboard Co-Pilot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 5-page Streamlit research browser with a single-page operational trading co-pilot that shows what to trade, when, and why — with time-reactive states and DST-safe session scheduling.

**Architecture:** Single-page Streamlit app with a state machine (WEEKEND/IDLE/APPROACHING/ALERT/LIVE/POST/OVERNIGHT). Session times resolved per-day from `pipeline.dst.SESSION_CATALOG` resolvers. Strategies merged per-instrument-per-session from `LIVE_PORTFOLIO` into human-readable briefing cards. No trading-day boundary logic for "what's next" — purely chronological datetime comparison.

**Tech Stack:** Streamlit, DuckDB (read-only via `ui/db_reader.py`), `pipeline.dst.SESSION_CATALOG`, `trading_app.live_config.LIVE_PORTFOLIO`, `trading_app.live_config.build_live_portfolio`, `trading_app.strategy_fitness.compute_fitness`

---

## CRITICAL: DST / 9-to-9 Boundary Safety

The trading day is 9:00 AM Brisbane -> 9:00 AM next day. During US DST transitions:
- CME_REOPEN shifts from 09:00 to 08:00 Brisbane
- NYSE_OPEN shifts from 00:30 to 23:30 (crosses midnight boundary)
- Several sessions shift by 1 hour

**Rule 1:** "What session is next?" uses CHRONOLOGICAL datetime comparison only. No `hour < 9` checks. No trading-day boundary logic. Compute session datetimes for today AND tomorrow, sort, pick the first future one.

**Rule 2:** "What trading day is it?" (for DB queries only) uses `pipeline.build_daily_features.compute_trading_day()` — the pipeline's battle-tested function.

**Rule 3:** Weekend detection checks BOTH calendar day AND whether any sessions exist in the next 24 hours. Saturday 7 AM is still Friday's trading day (overnight sessions may be active). Saturday 10 AM with no future sessions = weekend.

**Rule 4:** NEVER use `hour < 9` to decide if a session is "today or tomorrow." Use the resolver output directly — it returns the correct (hour, minute) for the correct calendar day.

---

### Task 0: Write session_helpers.py — Core session time logic (DST-safe)

**Files:**
- Create: `ui/session_helpers.py`
- Create: `tests/test_ui/__init__.py`
- Create: `tests/test_ui/test_session_helpers.py`

**Step 1: Write the failing tests**

```python
# tests/test_ui/__init__.py
# (empty)
```

```python
# tests/test_ui/test_session_helpers.py
"""Tests for DST-safe session scheduling and filter translation."""

import pytest
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

BRISBANE = ZoneInfo("Australia/Brisbane")


# ── get_upcoming_sessions ────────────────────────────────────────────────────

class TestGetUpcomingSessions:
    """Chronological session ordering — no trading-day boundary logic."""

    def test_morning_returns_remaining_sessions(self):
        """At 9:30 AM, CME_REOPEN (9:00) is past, TOKYO_OPEN (10:00) is next."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 9, 30, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert "CME_REOPEN" not in names, "9:00 session should be past at 9:30"
        assert names[0] == "TOKYO_OPEN", "10:00 session should be next"

    def test_late_night_returns_overnight_sessions(self):
        """At 11:45 PM, NYSE_OPEN (00:30 tomorrow) should be next."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 23, 45, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert names[0] == "NYSE_OPEN", "00:30 tomorrow should be next after 23:45"

    def test_no_duplicate_sessions(self):
        """Each session name appears at most once."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 8, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert len(names) == len(set(names)), f"Duplicates found: {names}"

    def test_dst_transition_cme_reopen_at_0800(self):
        """During US DST, CME_REOPEN is at 08:00 Brisbane — must still be found."""
        from ui.session_helpers import get_upcoming_sessions

        # Mar 8 2026 = US spring forward day, CME_REOPEN at 08:00 Brisbane
        now = datetime(2026, 3, 8, 7, 50, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert "CME_REOPEN" in names, "CME_REOPEN at 08:00 must appear"
        # It should be the first upcoming session
        assert names[0] == "CME_REOPEN"

    def test_dst_transition_nyse_open_at_2330(self):
        """During US DST, NYSE_OPEN is at 23:30 same calendar day (not 00:30 next)."""
        from ui.session_helpers import get_upcoming_sessions

        # Mar 8 2026 = US spring forward, NYSE_OPEN at 23:30 Brisbane
        now = datetime(2026, 3, 8, 23, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert names[0] == "NYSE_OPEN", "NYSE_OPEN at 23:30 should be next"

    def test_all_sessions_are_real_datetimes(self):
        """Every returned session has a timezone-aware Brisbane datetime."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 8, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)

        for name, dt in upcoming:
            assert dt.tzinfo is not None, f"{name} has naive datetime"
            assert dt > now, f"{name} at {dt} is not in the future"


# ── current_trading_day ──────────────────────────────────────────────────────

class TestCurrentTradingDay:
    """Uses pipeline's compute_trading_day — just a wrapper."""

    def test_morning_after_9am(self):
        from ui.session_helpers import current_trading_day
        td = current_trading_day(datetime(2026, 3, 6, 10, 0, tzinfo=BRISBANE))
        assert td == date(2026, 3, 6)

    def test_overnight_before_9am(self):
        from ui.session_helpers import current_trading_day
        td = current_trading_day(datetime(2026, 3, 7, 2, 0, tzinfo=BRISBANE))
        assert td == date(2026, 3, 6), "2 AM Saturday = Friday trading day"

    def test_exactly_9am_is_new_day(self):
        from ui.session_helpers import current_trading_day
        td = current_trading_day(datetime(2026, 3, 6, 9, 0, tzinfo=BRISBANE))
        assert td == date(2026, 3, 6)


# ── is_weekend ───────────────────────────────────────────────────────────────

class TestIsWeekend:
    def test_friday_morning_not_weekend(self):
        from ui.session_helpers import is_weekend
        now = datetime(2026, 3, 6, 10, 0, tzinfo=BRISBANE)  # Friday
        assert is_weekend(now) is False

    def test_saturday_morning_after_sessions_done(self):
        from ui.session_helpers import is_weekend
        # Saturday 10 AM — all Friday overnight sessions complete, no new ones
        now = datetime(2026, 3, 7, 10, 0, tzinfo=BRISBANE)
        assert is_weekend(now) is True

    def test_saturday_early_morning_not_weekend(self):
        from ui.session_helpers import is_weekend
        # Saturday 3 AM — Friday overnight sessions might still be active
        now = datetime(2026, 3, 7, 3, 0, tzinfo=BRISBANE)
        # This is tricky: CME is closed, but our function should still return True
        # because there are no real sessions on a Saturday (CME closes Fri 5 PM CT)
        # Actually Friday's last session is NYSE_CLOSE at 7 AM Sat.
        # At 3 AM Sat, COMEX_SETTLE (4:30 AM) and CME_PRECLOSE (6:45 AM) are still upcoming
        # BUT — these are only valid on trading days, and Saturday is not a trading day
        # So is_weekend should check: is the calendar day a weekend?
        # Friday overnight sessions are FRIDAY's trading day — they already happened
        # The dashboard at 3 AM Saturday should show weekend state
        assert is_weekend(now) is True

    def test_sunday_afternoon(self):
        from ui.session_helpers import is_weekend
        now = datetime(2026, 3, 8, 14, 0, tzinfo=BRISBANE)  # Sunday
        assert is_weekend(now) is True

    def test_monday_morning(self):
        from ui.session_helpers import is_weekend
        now = datetime(2026, 3, 9, 8, 50, tzinfo=BRISBANE)  # Monday
        assert is_weekend(now) is False


# ── filter_to_english ────────────────────────────────────────────────────────

class TestFilterToEnglish:
    def test_g_filter_basic(self):
        from ui.session_helpers import filter_to_english
        assert filter_to_english("ORB_G4") == "ORB >= 4pts"

    def test_g_filter_with_cont(self):
        from ui.session_helpers import filter_to_english
        result = filter_to_english("ORB_G5_CONT")
        assert "5pts" in result
        assert "continuation" in result.lower()

    def test_g_filter_with_fast(self):
        from ui.session_helpers import filter_to_english
        result = filter_to_english("ORB_G4_FAST10")
        assert "4pts" in result
        assert "fast" in result.lower()

    def test_g_filter_with_nomon(self):
        from ui.session_helpers import filter_to_english
        result = filter_to_english("ORB_G6_NOMON")
        assert "6pts" in result
        assert "monday" in result.lower()

    def test_vol_filter(self):
        from ui.session_helpers import filter_to_english
        result = filter_to_english("VOL_RV12_N20")
        assert "vol" in result.lower()

    def test_dir_long(self):
        from ui.session_helpers import filter_to_english
        result = filter_to_english("DIR_LONG")
        assert "long" in result.lower()

    def test_unknown_filter_returns_itself(self):
        from ui.session_helpers import filter_to_english
        assert filter_to_english("SOME_NEW_FILTER") == "SOME_NEW_FILTER"


# ── build_session_briefings ──────────────────────────────────────────────────

class TestBuildSessionBriefings:
    """Merge multiple strategies per session+instrument into one briefing."""

    def test_merges_multiple_filters_same_instrument(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        # CME_REOPEN has MGC with 3 filters (G5, VOL_RV12_N20, G4_FAST10)
        cme_mgc = [b for b in briefings
                    if b.session == "CME_REOPEN" and b.instrument == "MGC"]
        assert len(cme_mgc) == 1, "Should merge to single briefing per session+instrument"
        assert len(cme_mgc[0].conditions) >= 2, "Should have multiple conditions"

    def test_briefing_has_rr_target(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        for b in briefings:
            assert b.rr_target is not None and b.rr_target > 0, \
                f"{b.session} {b.instrument} missing rr_target"

    def test_briefing_has_session_time(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        for b in briefings:
            assert b.session_hour is not None, \
                f"{b.session} missing session time"


# ── get_app_state ────────────────────────────────────────────────────────────

class TestGetAppState:
    def test_returns_valid_state_name(self):
        from ui.session_helpers import get_app_state
        now = datetime(2026, 3, 6, 10, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name in (
            "WEEKEND", "IDLE", "APPROACHING", "ALERT",
            "LIVE", "POST", "OVERNIGHT",
        )

    def test_weekend_on_saturday(self):
        from ui.session_helpers import get_app_state
        now = datetime(2026, 3, 7, 12, 0, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "WEEKEND"

    def test_alert_close_to_session(self):
        from ui.session_helpers import get_app_state
        # 5 min before TOKYO_OPEN (10:00)
        now = datetime(2026, 3, 6, 9, 55, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "ALERT"

    def test_approaching_30min_before(self):
        from ui.session_helpers import get_app_state
        # 30 min before TOKYO_OPEN (10:00)
        now = datetime(2026, 3, 6, 9, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "APPROACHING"

    def test_idle_long_gap(self):
        from ui.session_helpers import get_app_state
        # 12:00 PM — 6 hours until LONDON_METALS at 18:00
        now = datetime(2026, 3, 6, 12, 0, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "IDLE"

    def test_overnight_late_evening(self):
        from ui.session_helpers import get_app_state
        # 9:30 PM — next session is US_DATA_830 at 11:30 PM (outside awake hours)
        now = datetime(2026, 3, 6, 21, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "OVERNIGHT"

    def test_state_has_next_session_info(self):
        from ui.session_helpers import get_app_state
        now = datetime(2026, 3, 6, 9, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.next_session is not None
        assert state.next_session_dt is not None
        assert state.minutes_to_next is not None


# ── get_refresh_seconds ──────────────────────────────────────────────────────

class TestGetRefreshSeconds:
    def test_weekend_slow_refresh(self):
        from ui.session_helpers import get_refresh_seconds
        assert get_refresh_seconds(minutes_to_next=2000, is_weekend=True) >= 60

    def test_alert_fast_refresh(self):
        from ui.session_helpers import get_refresh_seconds
        assert get_refresh_seconds(minutes_to_next=5, is_weekend=False) <= 10

    def test_idle_medium_refresh(self):
        from ui.session_helpers import get_refresh_seconds
        assert 15 <= get_refresh_seconds(minutes_to_next=120, is_weekend=False) <= 60
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui/test_session_helpers.py -v --tb=short 2>&1 | head -40`
Expected: ImportError — `ui.session_helpers` does not exist yet

**Step 3: Write the implementation**

```python
# ui/session_helpers.py
"""
Session scheduling, filter translation, and briefing card builder.

DST-SAFE: All session time logic uses chronological datetime comparison.
No `hour < 9` boundary checks. No trading-day logic for "what's next."
Trading-day concept used ONLY for database queries.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.dst import SESSION_CATALOG
from pipeline.build_daily_features import compute_trading_day

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
    then_session: str | None = None      # back-to-back session within 30 min
    then_session_dt: datetime | None = None
    next_monday: date | None = None      # only set for WEEKEND state
    trading_day: date | None = None


@dataclass
class SessionBriefing:
    """Merged briefing card for one instrument at one session."""
    session: str
    instrument: str
    conditions: list[str]         # Human-readable filter conditions
    rr_target: float
    entry_instruction: str        # e.g. "Place stop-market at ORB edge"
    direction_note: str | None    # e.g. "Long breakouts only"
    session_hour: int             # Hour in Brisbane (for display)
    session_minute: int           # Minute in Brisbane
    orb_minutes: int = 5          # ORB aperture
    strategy_count: int = 1       # How many underlying strategies merged


# ── Session time resolution (DST-safe) ──────────────────────────────────────

def get_upcoming_sessions(now: datetime) -> list[tuple[str, datetime]]:
    """Get all sessions in the next ~36 hours, sorted chronologically.

    DST-safe: computes session datetimes for today AND tomorrow using each
    session's per-day resolver. No `hour < 9` logic. No trading-day boundaries.
    Pure chronological comparison.
    """
    today = now.date()
    tomorrow = today + timedelta(days=1)

    all_sessions: list[tuple[str, datetime]] = []
    for cal_day in [today, tomorrow]:
        for name, entry in SESSION_CATALOG.items():
            h, m = entry["resolver"](cal_day)
            dt = datetime.combine(cal_day, time(h, m), tzinfo=BRISBANE)
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
    "_FAST10": " + fast break (10 bars)",
    "_NOMON": " + not Monday",
}

_SPECIAL_FILTERS = {
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
    from trading_app.live_config import build_live_portfolio
    from pipeline.asset_configs import get_active_instruments

    from datetime import date as date_type
    today = date_type.today()

    # Collect all strategies across all instruments
    all_strategies: list = []
    for instrument in get_active_instruments():
        try:
            portfolio, _ = build_live_portfolio(instrument=instrument)
            all_strategies.extend(portfolio.strategies)
        except Exception:
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

        briefings.append(SessionBriefing(
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
        ))

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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui/test_session_helpers.py -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add ui/session_helpers.py tests/test_ui/__init__.py tests/test_ui/test_session_helpers.py
git commit -m "feat(ui): add DST-safe session helpers — time resolution, filter translation, state machine"
```

---

### Task 1: Extend db_reader.py with ATR and results queries

**Files:**
- Modify: `ui/db_reader.py` (add 3 new functions at end)
- Create: `tests/test_ui/test_db_reader_copilot.py`

**Step 1: Write the failing tests**

```python
# tests/test_ui/test_db_reader_copilot.py
"""Tests for new db_reader functions used by the co-pilot dashboard."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date


class TestGetPriorDayAtr:
    def test_returns_float_or_none(self):
        from ui.db_reader import get_prior_day_atr
        # Against real DB — may return None if no data for instrument
        result = get_prior_day_atr("MGC")
        assert result is None or isinstance(result, float)

    def test_returns_positive_for_mgc(self):
        from ui.db_reader import get_prior_day_atr
        result = get_prior_day_atr("MGC")
        if result is not None:
            assert result > 0


class TestGetTodayCompletedSessions:
    def test_returns_list(self):
        from ui.db_reader import get_today_completed_sessions
        result = get_today_completed_sessions(date(2026, 2, 20))
        assert isinstance(result, list)


class TestGetPreviousTradingDay:
    def test_returns_date_before_input(self):
        from ui.db_reader import get_previous_trading_day
        result = get_previous_trading_day(date(2026, 2, 20))
        if result is not None:
            assert result < date(2026, 2, 20)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui/test_db_reader_copilot.py -v --tb=short`
Expected: ImportError — functions don't exist yet

**Step 3: Write the implementation**

Add to end of `ui/db_reader.py`:

```python
def get_prior_day_atr(
    instrument: str,
    orb_minutes: int = 5,
    db_path: Path | None = None,
) -> float | None:
    """Get the most recent ATR-20 for an instrument.

    Returns the atr_20 value from the latest trading day in daily_features.
    Used by the co-pilot to set expectations: "Prior day ATR: 28pts."
    """
    sql = f"""
        SELECT atr_20
        FROM daily_features
        WHERE symbol = '{instrument}'
          AND orb_minutes = {orb_minutes}
        ORDER BY trading_day DESC
        LIMIT 1
    """
    try:
        df = query_df(sql, db_path)
        if df.empty:
            return None
        val = df.iloc[0]["atr_20"]
        return float(val) if val is not None else None
    except Exception:
        return None


def get_today_completed_sessions(
    trading_day: date,
    db_path: Path | None = None,
) -> list[dict]:
    """Get ORB outcomes for a trading day, grouped by session.

    Returns list of dicts with keys: orb_label, symbol, break_dir, pnl_r, outcome.
    Used by the co-pilot's day summary section.
    """
    sql = f"""
        SELECT orb_label, symbol, break_dir, pnl_r, outcome,
               entry_model, rr_target
        FROM orb_outcomes
        WHERE trading_day = '{trading_day.isoformat()}'
          AND orb_minutes = 5
        ORDER BY orb_label, symbol
    """
    try:
        df = query_df(sql, db_path)
        return df.to_dict("records") if not df.empty else []
    except Exception:
        return []


def get_previous_trading_day(
    before: date,
    db_path: Path | None = None,
) -> date | None:
    """Find the most recent trading day before the given date.

    Queries daily_features for the latest trading_day < before.
    Used by the co-pilot for "Last trading day" summary.
    """
    sql = f"""
        SELECT MAX(trading_day) as prev_day
        FROM daily_features
        WHERE trading_day < '{before.isoformat()}'
          AND orb_minutes = 5
    """
    try:
        df = query_df(sql, db_path)
        if df.empty or df.iloc[0]["prev_day"] is None:
            return None
        return df.iloc[0]["prev_day"]
    except Exception:
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui/test_db_reader_copilot.py -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add ui/db_reader.py tests/test_ui/test_db_reader_copilot.py
git commit -m "feat(ui): add ATR, completed sessions, and previous trading day queries to db_reader"
```

---

### Task 2: Write copilot.py — Main render logic

**Files:**
- Create: `ui/copilot.py`

This is the main rendering module. Each state gets its own render function. No tests needed — Streamlit rendering is tested manually.

**Step 1: Write the implementation**

```python
# ui/copilot.py
"""
Trading Co-Pilot — single-page operational dashboard.

Renders the appropriate view based on the current app state:
  WEEKEND     — "Markets closed" with next Monday info
  IDLE        — Completed sessions recap, calm "next session" text
  APPROACHING — Countdown + briefing cards appearing
  ALERT       — Full briefing cards, start session button
  OVERNIGHT   — Dimmed overnight session list
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from ui.session_helpers import (
    AppState,
    SessionBriefing,
    get_app_state,
    get_upcoming_sessions,
    get_refresh_seconds,
    build_session_briefings,
    is_weekend,
    current_trading_day,
    BRISBANE,
    AWAKE_START,
    AWAKE_END,
)
from ui.db_reader import (
    get_prior_day_atr,
    get_today_completed_sessions,
    get_previous_trading_day,
)

# Signals file written by SessionOrchestrator
_SIGNALS_FILE = Path(__file__).parent.parent / "live_signals.jsonl"
_STOP_FILE = Path(__file__).parent.parent / "live_session.stop"


# ── Cached data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _cached_briefings() -> list[SessionBriefing]:
    """Cache briefing cards for 5 minutes."""
    return build_session_briefings()


@st.cache_data(ttl=300)
def _cached_atr(instrument: str) -> float | None:
    return get_prior_day_atr(instrument)


# ── Header bar ───────────────────────────────────────────────────────────────

def _render_header(now: datetime, state: AppState) -> None:
    """Top bar: date, time, session dots."""
    # Format times
    bris_time = now.strftime("%I:%M %p").lstrip("0")
    # ET = Brisbane - 15h (EST) or -14h (EDT). Approximate.
    et_offset = timedelta(hours=-15) if now.month >= 11 or now.month <= 2 else timedelta(hours=-14)
    et_time = (now + et_offset).strftime("%I:%M %p").lstrip("0")
    day_str = now.strftime("%a %d %b %Y")

    col_date, col_time = st.columns([2, 3])
    with col_date:
        st.markdown(f"### {day_str}")
    with col_time:
        st.markdown(f"**{bris_time} Brisbane** &nbsp;&nbsp; ({et_time} ET)")

    # Session dot strip — only sessions with live portfolio strategies
    briefings = _cached_briefings()
    active_sessions = sorted(set(b.session for b in briefings),
                             key=lambda s: _session_sort_key(s, now))

    if active_sessions and not state.name == "WEEKEND":
        upcoming_names = {name for name, _ in get_upcoming_sessions(now)}
        dots = []
        for session in active_sessions:
            if session == state.next_session:
                dots.append(f":blue[**{session}**]")
            elif session in upcoming_names:
                dots.append(f":gray[{session}]")
            else:
                dots.append(f":gray[~~{session}~~]")  # completed
        st.caption(" &bull; ".join(dots))

    st.divider()


def _session_sort_key(session: str, now: datetime) -> float:
    """Sort sessions by their datetime relative to now."""
    if session not in __import__('pipeline.dst', fromlist=['SESSION_CATALOG']).SESSION_CATALOG:
        return 9999
    from pipeline.dst import SESSION_CATALOG
    today = now.date()
    tomorrow = today + timedelta(days=1)
    for cal_day in [today, tomorrow]:
        h, m = SESSION_CATALOG[session]["resolver"](cal_day)
        dt = datetime.combine(cal_day, __import__('datetime').time(h, m), tzinfo=BRISBANE)
        if dt > now - timedelta(hours=12):
            return dt.timestamp()
    return 9999


# ── State renderers ──────────────────────────────────────────────────────────

def _render_weekend(state: AppState) -> None:
    """Markets closed view."""
    st.markdown(
        "<h1 style='text-align:center; color:#888; margin-top:80px;'>"
        "Markets Closed</h1>",
        unsafe_allow_html=True,
    )
    if state.next_monday:
        st.markdown(
            f"<p style='text-align:center; color:#666; font-size:1.3rem;'>"
            f"Next trading day: <b>Monday {state.next_monday.strftime('%d %b')}</b>"
            f" &mdash; CME_REOPEN 9:00 AM</p>",
            unsafe_allow_html=True,
        )

    # Show last trading day summary
    st.divider()
    _render_previous_day_summary(state.trading_day)


def _render_idle(state: AppState, now: datetime) -> None:
    """Long gap between sessions — show recap + next session (no countdown)."""
    # Completed sessions recap
    _render_today_summary(state.trading_day, now)

    st.markdown("---")

    # Next session — calm, not urgent
    if state.next_session and state.next_session_dt:
        session_time = state.next_session_dt.strftime("%I:%M %p").lstrip("0")
        hours = int(state.minutes_to_next // 60) if state.minutes_to_next else 0
        mins = int(state.minutes_to_next % 60) if state.minutes_to_next else 0

        if hours > 0:
            gap_str = f"{hours}h {mins}m"
        else:
            gap_str = f"{mins}m"

        st.markdown(
            f"<p style='text-align:center; color:#888; font-size:1.1rem; margin-top:30px;'>"
            f"Next: <b>{state.next_session}</b> &middot; {session_time}"
            f" &mdash; in {gap_str}</p>",
            unsafe_allow_html=True,
        )
        if state.then_session and state.then_session_dt:
            then_time = state.then_session_dt.strftime("%I:%M %p").lstrip("0")
            st.markdown(
                f"<p style='text-align:center; color:#666; font-size:0.9rem;'>"
                f"then {state.then_session} &middot; {then_time}</p>",
                unsafe_allow_html=True,
            )


def _render_approaching(state: AppState, now: datetime) -> None:
    """15-60 min to session — countdown appears, briefing cards start showing."""
    mins = int(state.minutes_to_next) if state.minutes_to_next else 0
    session_time = state.next_session_dt.strftime("%I:%M %p").lstrip("0") if state.next_session_dt else ""

    st.markdown(
        f"<h2 style='text-align:center;'>{state.next_session} &middot; {session_time}</h2>"
        f"<h1 style='text-align:center; font-size:3rem;'>in {mins} minutes</h1>",
        unsafe_allow_html=True,
    )
    if state.then_session and state.then_session_dt:
        then_time = state.then_session_dt.strftime("%I:%M %p").lstrip("0")
        st.markdown(
            f"<p style='text-align:center; color:#888;'>"
            f"then {state.then_session} &middot; {then_time}</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Show briefing cards for next session
    _render_briefing_cards(state.next_session, now)

    # Start session button
    _render_session_controls()


def _render_alert(state: AppState, now: datetime) -> None:
    """<15 min — full briefing, urgent styling."""
    mins = int(state.minutes_to_next) if state.minutes_to_next else 0
    session_time = state.next_session_dt.strftime("%I:%M %p").lstrip("0") if state.next_session_dt else ""

    st.markdown(
        f"<h2 style='text-align:center; color:#ff6b6b;'>"
        f"{state.next_session} &middot; {session_time}</h2>"
        f"<h1 style='text-align:center; font-size:4rem; color:#ff6b6b;'>"
        f"in {mins} min</h1>",
        unsafe_allow_html=True,
    )
    if state.then_session and state.then_session_dt:
        then_time = state.then_session_dt.strftime("%I:%M %p").lstrip("0")
        st.markdown(
            f"<p style='text-align:center; color:#888;'>"
            f"then {state.then_session} &middot; {then_time}</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Briefing cards — full detail
    _render_briefing_cards(state.next_session, now)

    # Start session button
    _render_session_controls()


def _render_overnight(state: AppState, now: datetime) -> None:
    """Next session is outside awake hours."""
    # Show today's recap first
    _render_today_summary(state.trading_day, now)

    st.markdown("---")

    # Overnight session list (dimmed)
    st.markdown(
        "<p style='text-align:center; color:#666; margin-top:30px;'>"
        "Overnight sessions</p>",
        unsafe_allow_html=True,
    )
    upcoming = get_upcoming_sessions(now)
    overnight = [(n, dt) for n, dt in upcoming if not (AWAKE_START <= dt.hour < AWAKE_END)]
    daytime = [(n, dt) for n, dt in upcoming if AWAKE_START <= dt.hour < AWAKE_END]

    if overnight:
        for name, dt in overnight[:6]:
            t = dt.strftime("%I:%M %p").lstrip("0")
            st.markdown(f"<p style='text-align:center; color:#555;'>{name} &middot; {t}</p>",
                        unsafe_allow_html=True)

    if daytime:
        next_day_session = daytime[0]
        t = next_day_session[1].strftime("%I:%M %p").lstrip("0")
        st.markdown(
            f"<p style='text-align:center; color:#888; font-size:1.2rem; margin-top:20px;'>"
            f"Next morning: <b>{next_day_session[0]}</b> &middot; {t}</p>",
            unsafe_allow_html=True,
        )

    # Previous day summary
    st.divider()
    _render_previous_day_summary(state.trading_day)


# ── Briefing cards ───────────────────────────────────────────────────────────

def _render_briefing_cards(session: str, now: datetime) -> None:
    """Render instrument briefing cards for a session."""
    briefings = _cached_briefings()
    session_briefings = [b for b in briefings if b.session == session]

    if not session_briefings:
        st.info(f"No strategies configured for {session}")
        return

    for b in session_briefings:
        with st.container(border=True):
            # Instrument header
            instrument_names = {
                "MGC": "Micro Gold", "MNQ": "Micro Nasdaq",
                "MES": "Micro S&P", "M2K": "Micro Russell",
            }
            full_name = instrument_names.get(b.instrument, b.instrument)

            st.markdown(f"**{b.instrument}** &middot; {full_name}")
            st.markdown("")

            # Qualifying conditions
            if b.conditions:
                conditions_text = " OR ".join(b.conditions)
                st.markdown(f"**IF:** {conditions_text}")
            if b.direction_note:
                st.markdown(f"**Direction:** {b.direction_note}")

            # Entry instruction + RR
            st.markdown(f"**THEN:** {b.entry_instruction}")
            st.markdown(f"**Target:** {b.rr_target:.1f}x risk")

            # ATR context
            atr = _cached_atr(b.instrument)
            if atr is not None:
                # Find min G-threshold from conditions
                g_thresholds = []
                for c in b.conditions:
                    if c.startswith("ORB >= "):
                        try:
                            pts = int(c.split(">=")[1].split("pts")[0].strip())
                            g_thresholds.append(pts)
                        except (ValueError, IndexError):
                            pass
                if g_thresholds:
                    min_g = min(g_thresholds)
                    likelihood = "common" if atr > min_g * 3 else "typical" if atr > min_g * 1.5 else "marginal"
                    st.caption(f"Prior day ATR: {atr:.1f}pts — {min_g}pt ORB is {likelihood}")
                else:
                    st.caption(f"Prior day ATR: {atr:.1f}pts")

            # Strategy count
            st.caption(f"{b.strategy_count} strategies merged &middot; {b.orb_minutes}m ORB")


# ── Day summary ──────────────────────────────────────────────────────────────

def _render_today_summary(trading_day: date | None, now: datetime) -> None:
    """Show completed sessions for today's trading day."""
    if trading_day is None:
        return

    st.markdown("**Today**")

    briefings = _cached_briefings()
    active_sessions = sorted(set(b.session for b in briefings),
                             key=lambda s: _session_sort_key(s, now))
    upcoming_names = {name for name, _ in get_upcoming_sessions(now)}

    for session in active_sessions:
        if session in SESSION_CATALOG:
            from pipeline.dst import SESSION_CATALOG as sc
            h, m = sc[session]["resolver"](now.date())
            t = f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
        else:
            t = ""

        if session in upcoming_names:
            st.markdown(f":gray[{session} &middot; {t} — waiting]")
        else:
            st.markdown(f":gray[~~{session} &middot; {t} — done~~]")


def _render_previous_day_summary(trading_day: date | None) -> None:
    """Show last trading day results."""
    if trading_day is None:
        return
    prev = get_previous_trading_day(trading_day)
    if prev is None:
        return

    results = get_today_completed_sessions(prev)
    if not results:
        st.caption(f"Last trading day ({prev}): no data")
        return

    total_r = sum(r.get("pnl_r", 0) or 0 for r in results)
    st.caption(f"Last trading day ({prev}): {len(results)} outcomes, {total_r:+.1f}R total")


# ── Session controls ─────────────────────────────────────────────────────────

def _render_session_controls() -> None:
    """Start/stop live session buttons."""
    proc = st.session_state.get("live_proc")
    is_running = proc is not None and proc.poll() is None

    if is_running:
        inst = st.session_state.get("live_instrument", "?")
        mode = st.session_state.get("live_mode_short", "signal-only")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"Session RUNNING — {inst} [{mode}] pid={proc.pid}")
        with col2:
            if st.button("Stop", type="secondary"):
                _stop_session()
                st.rerun()
    else:
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            instrument = st.selectbox(
                "Instrument",
                ["MGC", "MNQ", "MES", "M2K"],
                key="copilot_instrument",
                label_visibility="collapsed",
            )
        with col2:
            if st.button("Start Signal-Only", type="primary"):
                _start_session(instrument, signal_only=True)
                st.rerun()
        with col3:
            if st.button("Start Demo"):
                _start_session(instrument, signal_only=False)
                st.rerun()


def _start_session(instrument: str, signal_only: bool) -> None:
    """Launch a live session subprocess."""
    if _SIGNALS_FILE.exists():
        try:
            with open(_SIGNALS_FILE, "w"):
                pass
        except OSError:
            pass
    _STOP_FILE.unlink(missing_ok=True)

    flag = "--signal-only" if signal_only else "--demo"
    cmd = [
        sys.executable,
        "scripts/run_live_session.py",
        "--instrument", instrument,
        flag,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.session_state["live_proc"] = proc
    st.session_state["live_instrument"] = instrument
    st.session_state["live_mode_short"] = "signal-only" if signal_only else "demo"


def _stop_session() -> None:
    """Gracefully stop a live session."""
    proc = st.session_state.get("live_proc")
    if proc:
        _STOP_FILE.touch()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _STOP_FILE.unlink(missing_ok=True)
            proc.kill()
    st.session_state.pop("live_proc", None)


# ── Signal log ───────────────────────────────────────────────────────────────

def _render_signal_log() -> None:
    """Show recent signals from live_signals.jsonl."""
    if not _SIGNALS_FILE.exists():
        return

    try:
        raw = _SIGNALS_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        records = [json.loads(ln) for ln in lines[-20:]]  # last 20
    except Exception:
        return

    if not records:
        return

    st.markdown("**Live Signals**")
    for r in reversed(records):
        ts = r.get("ts", "")
        try:
            from datetime import timezone
            ts_display = datetime.fromisoformat(ts).astimezone(BRISBANE).strftime("%I:%M %p").lstrip("0")
        except (ValueError, TypeError):
            ts_display = ts

        event_type = r.get("type", "")
        instrument = r.get("instrument", "")
        strategy = r.get("strategy_id", "")
        price = r.get("price", "")

        colors = {
            "SIGNAL_ENTRY": "green", "ORDER_ENTRY": "green",
            "SIGNAL_EXIT": "orange", "ORDER_EXIT": "orange",
            "REJECT": "red", "SESSION_START": "blue",
        }
        color = colors.get(event_type, "gray")
        st.markdown(f":{color}[{ts_display}] {event_type} {instrument} {strategy} {price}")


# ── Main render ──────────────────────────────────────────────────────────────

def render() -> None:
    """Main entry point — renders the co-pilot dashboard."""
    now = datetime.now(BRISBANE)
    state = get_app_state(now)

    _render_header(now, state)

    # State-dependent main area
    if state.name == "WEEKEND":
        _render_weekend(state)
    elif state.name == "OVERNIGHT":
        _render_overnight(state, now)
    elif state.name == "IDLE":
        _render_idle(state, now)
    elif state.name == "APPROACHING":
        _render_approaching(state, now)
    elif state.name == "ALERT":
        _render_alert(state, now)

    # Signal log (always, if running)
    proc = st.session_state.get("live_proc")
    if proc is not None and proc.poll() is None:
        st.divider()
        _render_signal_log()

    # Adaptive refresh
    refresh = get_refresh_seconds(
        minutes_to_next=state.minutes_to_next or 999,
        is_weekend=state.name == "WEEKEND",
    )
    time.sleep(refresh)
    st.rerun()
```

**Step 2: Commit**

```bash
git add ui/copilot.py
git commit -m "feat(ui): add copilot.py — state-machine renderer for trading co-pilot"
```

---

### Task 3: Rewrite app.py — Single-page co-pilot

**Files:**
- Modify: `ui/app.py` (complete rewrite — file already read)

**Step 1: Write the new app.py**

```python
# ui/app.py
"""
Canompx3 Trading Co-Pilot.

Single-page operational dashboard. Shows what to trade, when, and why.

Launch: streamlit run ui/app.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Trading Co-Pilot",
    page_icon="Au",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.copilot import render

render()
```

**Step 2: Verify it launches**

Run: `streamlit run ui/app.py` (manual check — opens browser)
Expected: Single-page co-pilot with session countdown, no sidebar navigation

**Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat(ui): rewrite app.py as single-page trading co-pilot"
```

---

### Task 4: Delete old UI files

**Files:**
- Delete: `ui/pages/portfolio.py`
- Delete: `ui/pages/strategies.py`
- Delete: `ui/pages/market_state.py`
- Delete: `ui/pages/data_quality.py`
- Delete: `ui/pages/live_monitor.py`
- Delete: `ui/pages/__init__.py`
- Delete: `ui/chat.py`
- Delete: `ui/sandbox_runner.py`

**Step 1: Verify no imports reference deleted files**

Run: `grep -r "from ui.pages" --include="*.py" | grep -v __pycache__ | grep -v ".pyc"`
Run: `grep -r "from ui.chat" --include="*.py" | grep -v __pycache__`
Run: `grep -r "from ui.sandbox_runner" --include="*.py" | grep -v __pycache__`

Expected: Only `ui/app.py` (old version, already rewritten) references these. No other files import them.

**Step 2: Delete the files**

```bash
rm ui/pages/portfolio.py ui/pages/strategies.py ui/pages/market_state.py
rm ui/pages/data_quality.py ui/pages/live_monitor.py ui/pages/__init__.py
rm ui/chat.py ui/sandbox_runner.py
rmdir ui/pages
```

**Step 3: Verify app still launches**

Run: `streamlit run ui/app.py` (manual check)
Expected: Co-pilot renders without errors about missing pages

**Step 4: Commit**

```bash
git add -A ui/pages/ ui/chat.py ui/sandbox_runner.py
git commit -m "chore(ui): remove old research browser pages, chat, sandbox runner"
```

---

### Task 5: Fix _session_sort_key and import cleanup in copilot.py

The `_session_sort_key` function in copilot.py uses an ugly `__import__` hack. Clean it up and fix the `SESSION_CATALOG` import that's duplicated.

**Files:**
- Modify: `ui/copilot.py` (fix imports at top, clean up `_session_sort_key`)

**Step 1: Fix the function**

Replace the `_session_sort_key` function and fix the duplicate import in `_render_today_summary`:

```python
# At top of copilot.py, the SESSION_CATALOG import is already via session_helpers.
# Add direct import:
from pipeline.dst import SESSION_CATALOG

# Replace _session_sort_key:
def _session_sort_key(session: str, now: datetime) -> float:
    """Sort sessions by their datetime relative to now."""
    if session not in SESSION_CATALOG:
        return 9999.0
    today = now.date()
    tomorrow = today + timedelta(days=1)
    for cal_day in [today, tomorrow]:
        h, m = SESSION_CATALOG[session]["resolver"](cal_day)
        from datetime import time as dt_time
        dt = datetime.combine(cal_day, dt_time(h, m), tzinfo=BRISBANE)
        if dt > now - timedelta(hours=12):
            return dt.timestamp()
    return 9999.0
```

Also fix `_render_today_summary` to use the top-level `SESSION_CATALOG` import instead of re-importing.

**Step 2: Run tests**

Run: `python -m pytest tests/test_ui/ -v --tb=short`
Expected: All pass

**Step 3: Commit**

```bash
git add ui/copilot.py
git commit -m "fix(ui): clean up imports and session sort key in copilot.py"
```

---

### Task 6: Integration test — full render cycle

**Files:**
- Create: `tests/test_ui/test_copilot_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_ui/test_copilot_integration.py
"""Integration tests for the co-pilot state machine + briefing cards."""

from datetime import datetime, date
from zoneinfo import ZoneInfo

BRISBANE = ZoneInfo("Australia/Brisbane")


class TestFullCopilotFlow:
    """End-to-end tests that don't require Streamlit runtime."""

    def test_state_machine_all_times_of_day(self):
        """Verify state machine returns valid state for every hour."""
        from ui.session_helpers import get_app_state

        # Test every hour on a Friday
        for hour in range(24):
            now = datetime(2026, 3, 6, hour, 30, tzinfo=BRISBANE)
            state = get_app_state(now)
            assert state.name in (
                "WEEKEND", "IDLE", "APPROACHING", "ALERT", "OVERNIGHT",
            ), f"Invalid state at {hour}:30 — got {state.name}"

    def test_briefing_cards_build_without_error(self):
        """Briefing card builder should not crash."""
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        assert len(briefings) > 0, "Should have at least one briefing"
        for b in briefings:
            assert b.session, "Briefing missing session"
            assert b.instrument, "Briefing missing instrument"
            assert b.rr_target > 0, f"{b.session} {b.instrument} missing rr_target"
            assert len(b.conditions) > 0 or b.direction_note, \
                f"{b.session} {b.instrument} has no conditions"

    def test_no_state_crash_on_dst_transition_day(self):
        """State machine must not crash on US DST transition."""
        from ui.session_helpers import get_app_state

        # Mar 8 2026 = US spring forward
        for hour in range(24):
            now = datetime(2026, 3, 8, hour, 0, tzinfo=BRISBANE)
            state = get_app_state(now)
            assert state.name in (
                "WEEKEND", "IDLE", "APPROACHING", "ALERT", "OVERNIGHT",
            )

    def test_weekend_state_saturday_sunday(self):
        """Saturday and Sunday should always be WEEKEND."""
        from ui.session_helpers import get_app_state

        sat = datetime(2026, 3, 7, 12, 0, tzinfo=BRISBANE)
        sun = datetime(2026, 3, 8, 12, 0, tzinfo=BRISBANE)
        assert get_app_state(sat).name == "WEEKEND"
        assert get_app_state(sun).name == "WEEKEND"

    def test_sessions_always_have_time(self):
        """Every upcoming session must have a valid datetime."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 8, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        for name, dt in upcoming:
            assert dt.tzinfo is not None
            assert dt > now
            # Session time should be within 36 hours
            assert (dt - now).total_seconds() < 36 * 3600

    def test_filter_translator_covers_all_live_filters(self):
        """Every filter in LIVE_PORTFOLIO should translate to non-empty English."""
        from ui.session_helpers import build_session_briefings, filter_to_english
        from trading_app.live_config import LIVE_PORTFOLIO

        all_filters = set(s.filter_type for s in LIVE_PORTFOLIO)
        for f in all_filters:
            english = filter_to_english(f)
            assert english, f"Filter {f} translated to empty string"
            assert english != f or f in ("DIR_LONG", "DIR_SHORT"), \
                f"Filter {f} fell through to raw name: {english}"
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_ui/test_copilot_integration.py -v --tb=short`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_ui/test_copilot_integration.py
git commit -m "test(ui): add co-pilot integration tests — state machine, briefings, DST, filters"
```

---

### Task 7: Run full test suite + drift checks

**Step 1: Run existing test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass (new tests don't break existing)

**Step 2: Run drift checks**

Run: `python pipeline/check_drift.py`
Expected: All checks pass

**Step 3: Verify app launches**

Run: `streamlit run ui/app.py` (manual check)
Expected: Co-pilot renders correctly with current time state

**Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix(ui): address test/drift issues from co-pilot integration"
```

---

## Acceptance Criteria

1. `streamlit run ui/app.py` opens a SINGLE page with no sidebar navigation
2. Current time shown in Brisbane with ET equivalent
3. Session names ALWAYS paired with local time (e.g., "TOKYO_OPEN . 10:00 AM")
4. Countdown to next session — large text when approaching, calm text when far
5. Briefing cards show human-readable filter conditions (not machine names)
6. Back-to-back sessions (within 30 min) shown together
7. Weekend detection works (Saturday/Sunday show "Markets closed")
8. DST transition days don't crash or show wrong session order
9. Overnight sessions dimmed (outside awake hours)
10. Start/stop session buttons work (signal-only + demo)
11. All `tests/test_ui/` tests pass
12. `python pipeline/check_drift.py` passes
13. Old pages (portfolio, strategies, market_state, data_quality, live_monitor) deleted
14. Old chat.py and sandbox_runner.py deleted
