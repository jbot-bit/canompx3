"""
DST detection and dynamic session resolvers.

Determines whether the US or UK is in Daylight Saving Time on a given
trading day, and resolves dynamic ORB session times to Brisbane local
hours accordingly.

All sessions are dynamic (DST-aware resolver per-day):
  CME_REOPEN      - CME Globex electronic reopen at 5:00 PM CT
  TOKYO_OPEN      - Tokyo Stock Exchange open at 9:00 AM JST
  SINGAPORE_OPEN  - SGX/HKEX open at 9:00 AM SGT
  LONDON_METALS   - London metals AM session at 8:00 AM London
  US_DATA_830     - US economic data release at 8:30 AM ET
  NYSE_OPEN       - NYSE cash open at 9:30 AM ET
  US_DATA_1000    - US 10:00 AM data (ISM/CC) + post-equity-open flow
  COMEX_SETTLE    - COMEX gold settlement at 1:30 PM ET
  CME_PRECLOSE    - CME equity futures pre-settlement at 2:45 PM CT
  NYSE_CLOSE      - NYSE closing bell at 4:00 PM ET

Uses zoneinfo (stdlib) for all timezone math -- correct for all years,
handles edge cases automatically.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_US_EASTERN = ZoneInfo("America/New_York")
_US_CHICAGO = ZoneInfo("America/Chicago")
_UK_LONDON = ZoneInfo("Europe/London")
_BRISBANE = ZoneInfo("Australia/Brisbane")
_UTC = ZoneInfo("UTC")

# Public aliases — canonical timezone constants for use across pipeline/ and trading_app/.
# Promoted from build_daily_features.py during the E2 canonical-window refactor (2026-04-07)
# to give the codebase a single source of truth for time boundaries.
#
# Reference: Chan, "Algorithmic Trading: Winning Strategies and Their Rationale" (Wiley, 2013),
# Chapter 1 ("Backtesting and Automated Execution"), p4:
#   "If your backtesting and live trading programs are one and the same, and the only
#    difference between backtesting versus live trading is what kind of data you are
#    feeding into the program (historical data in the former, and live market data in the
#    latter), then there can be no look-ahead bias in the program."
#
# This module enforces that invariant for ORB window timing: every consumer of "ORB window
# end UTC" — backtest (outcome_builder), live engine (execution_engine), feature builder
# (build_daily_features) — derives from the SAME canonical function (orb_utc_window below).
BRISBANE_TZ = _BRISBANE
UTC_TZ = _UTC

# Trading day boundary: 09:00 Brisbane local. A bar at 23:00:00 UTC starts the next
# trading day (because 09:00 Brisbane = 23:00 UTC previous day, no DST in Brisbane).
TRADING_DAY_START_HOUR_LOCAL = 9

# Sanity bounds on ORB aperture minutes (matches VALID_ORB_MINUTES in build_daily_features).
# Bounds chosen to reject obviously wrong inputs (zero, negative, full-day) while allowing
# all currently-supported apertures (5, 15, 30) plus reasonable future expansion.
_ORB_MINUTES_MIN = 1
_ORB_MINUTES_MAX = 60


def is_us_dst(trading_day: date) -> bool:
    """True if US Eastern is in Daylight Saving Time (EDT, UTC-4) on this date.

    During DST (roughly Mar second Sunday -> Nov first Sunday):
      EDT = UTC-4, so 08:30 ET = 12:30 UTC
    Standard time (EST = UTC-5):
      08:30 ET = 13:30 UTC
    """
    # Use noon to avoid any ambiguity on transition days
    dt = datetime(trading_day.year, trading_day.month, trading_day.day, 12, 0, 0, tzinfo=_US_EASTERN)
    return dt.utcoffset().total_seconds() == -4 * 3600


def is_uk_dst(trading_day: date) -> bool:
    """True if UK is in British Summer Time (BST, UTC+1) on this date.

    During BST (roughly Mar last Sunday -> Oct last Sunday):
      BST = UTC+1, so 08:00 London = 07:00 UTC
    Standard time (GMT = UTC+0):
      08:00 London = 08:00 UTC
    """
    dt = datetime(trading_day.year, trading_day.month, trading_day.day, 12, 0, 0, tzinfo=_UK_LONDON)
    return dt.utcoffset().total_seconds() == 1 * 3600


# =========================================================================
# CANONICAL TRADING DAY / ORB WINDOW FUNCTIONS
# =========================================================================
#
# Promoted from pipeline/build_daily_features.py during the E2 canonical-window
# refactor (2026-04-07). These are the single source of truth for:
#   - The [start, end) UTC range of a trading day
#   - The [start, end) UTC range of an ORB window within a trading day
#
# Every consumer (backtest via outcome_builder, live engine via execution_engine,
# feature builder via build_daily_features) must import from here — not re-encode.
# See the module docstring and BRISBANE_TZ/UTC_TZ constants above for the Chan
# Ch 1 p4 invariant that motivates this consolidation.


def compute_trading_day_utc_range(trading_day: date) -> tuple[datetime, datetime]:
    """
    Return the [start, end) UTC range for a given trading day.

    A trading day begins at 09:00 Brisbane local (no DST in Brisbane — stable
    year-round), so:

      trading_day 2024-01-05:
        start = 2024-01-04 23:00:00 UTC (09:00 Brisbane on 2024-01-05)
        end   = 2024-01-05 23:00:00 UTC (09:00 Brisbane on 2024-01-06)

    The range is a 24-hour window: [start, end) — start inclusive, end exclusive.

    This function is CANONICAL. Do not re-encode this calendar math anywhere
    else in the codebase. Import from here.
    """
    # 09:00 Brisbane on trading_day = 23:00 UTC on (trading_day - 1).
    # Brisbane is UTC+10 with no DST — this formula is stable across all dates.
    start_utc = datetime(
        trading_day.year,
        trading_day.month,
        trading_day.day,
        TRADING_DAY_START_HOUR_LOCAL,
        0,
        0,
        tzinfo=BRISBANE_TZ,
    ).astimezone(UTC_TZ)

    end_utc = start_utc + timedelta(hours=24)
    return start_utc, end_utc


def compute_trading_day_from_timestamp(ts: datetime) -> date:
    """
    Assign an aware timestamp to its Brisbane trading day.

    Trading day boundary is 09:00 Brisbane. Callers must pass timezone-aware
    timestamps so UTC/local ambiguity cannot silently change the trading day.
    """
    if ts.tzinfo is None:
        raise ValueError("compute_trading_day_from_timestamp requires a timezone-aware timestamp")
    return (ts.astimezone(BRISBANE_TZ) - timedelta(hours=TRADING_DAY_START_HOUR_LOCAL)).date()


# Note: `orb_utc_window` (the canonical ORB window function) is defined
# further down in this module, AFTER `DYNAMIC_ORB_RESOLVERS` is populated
# from SESSION_CATALOG. This ordering is deliberate: placing the function
# next to its data dependency avoids forward-reference code smell.


# =========================================================================
# DST REGIME HELPERS — for winter/summer split analysis
# =========================================================================

# Sessions affected by DST and which timezone governs them.
# "US" = winter/summer split uses is_us_dst(). "UK" = uses is_uk_dst().
#
# All sessions are now dynamic and handle DST internally via resolver functions.
# No fixed sessions remain, so DST_AFFECTED_SESSIONS is empty.
# Consumers (strategy_validator.compute_dst_split, strategy_fitness) always return
# verdict='CLEAN' / None because this dict is empty. This is correct — DST splits
# are meaningless when all sessions use dynamic resolvers.
DST_AFFECTED_SESSIONS: dict = {}

# All sessions are dynamic (DST-aware resolvers) — all are "clean" by definition.
DST_CLEAN_SESSIONS = {
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
    "BRISBANE_1025",
}


# =========================================================================
# DOW ALIGNMENT — Brisbane DOW vs Exchange-Timezone DOW
# =========================================================================
# For each session, whether the Brisbane calendar day matches the
# exchange's calendar day.  Determines if a DayOfWeekSkipFilter targeting
# e.g. "Friday" actually skips the exchange's Friday session.
#
# Investigation: research/research_dow_alignment.py (Feb 2026)
#
# Result: all sessions EXCEPT NYSE_OPEN are aligned.
#   CME_REOPEN: Brisbane-Fri = CME Friday (5PM Thu CT = start of CME Fri session) ✓
#   TOKYO_OPEN: Brisbane DOW = Tokyo DOW (no DST, same calendar day) ✓
#   SINGAPORE_OPEN: Brisbane DOW = Singapore DOW (no DST) ✓
#   LONDON_METALS: Brisbane DOW = London DOW (both morning same calendar day) ✓
#   EUROPE_FLOW: Brisbane DOW = London DOW (same morning calendar day) ✓
#   US_DATA_830: Brisbane DOW = US DOW (13:00 UTC = US morning same day) ✓
#   NYSE_OPEN: Brisbane DOW = US DOW + 1 (00:30 Bris = 14:30 UTC PREV day) ✗
#   US_DATA_1000: Brisbane DOW = US DOW (01:00 Bris = 15:00 UTC same US day) ✓
#   COMEX_SETTLE: Brisbane DOW = US DOW (next cal day, but same US trading day) ✓
#   CME_PRECLOSE: Brisbane DOW = US DOW (same logic as COMEX_SETTLE) ✓
#   NYSE_CLOSE: Brisbane DOW = US DOW (same logic) ✓
#
# The NYSE_OPEN mismatch: Brisbane 00:30 crosses midnight from the previous UTC
# day.  Brisbane-Friday 00:30 = UTC Thursday 14:30 = US Thursday 9:30 AM.
# So Brisbane-Friday at NYSE_OPEN is the US THURSDAY equity open.
# Any DOW filter for NYSE_OPEN must account for this -1 day offset.

DOW_ALIGNED_SESSIONS = {
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
    "BRISBANE_1025",
}
DOW_MISALIGNED_SESSIONS = {
    "NYSE_OPEN": -1,  # Brisbane DOW = exchange DOW + 1 (i.e. exchange is 1 day behind)
}


def validate_dow_filter_alignment(session: str, skip_days: tuple[int, ...]) -> None:
    """Fail-closed guard: prevent DOW filters on sessions with known misalignment.

    Raises ValueError if a DOW skip filter is applied to a session where
    Brisbane DOW != exchange DOW, unless the caller has explicitly handled
    the offset (not yet implemented — NYSE_OPEN has no DOW filters currently).
    """
    offset = DOW_MISALIGNED_SESSIONS.get(session)
    if offset is not None and skip_days:
        raise ValueError(
            f"DOW filter with skip_days={skip_days} applied to session '{session}' "
            f"which has a Brisbane→Exchange DOW offset of {offset} day(s). "
            f"Brisbane-Friday at {session} = US Thursday (not US Friday). "
            f"Either use exchange-adjusted DOW or remove the filter."
        )


def is_winter_for_session(trading_day: date, orb_label: str) -> bool | None:
    """Classify a trading day as winter (True) or summer (False) for a given session.

    Returns None if the session is not affected by DST (clean sessions).
    Uses US Eastern for CME_REOPEN/NYSE_OPEN/US_DATA_830, UK London for LONDON_METALS.
    """
    dst_type = DST_AFFECTED_SESSIONS.get(orb_label)
    if dst_type is None:
        return None  # Clean session
    if dst_type == "US":
        return not is_us_dst(trading_day)  # winter = NOT DST
    else:  # UK
        return not is_uk_dst(trading_day)  # winter = NOT BST


def classify_dst_verdict(winter_avg_r: float | None, summer_avg_r: float | None, winter_n: int, summer_n: int) -> str:
    """Classify DST stability verdict for a strategy.

    Verdicts:
      STABLE:       |winter - summer| <= 0.10R AND both N >= 15
      WINTER-DOM:   winter > summer + 0.10R AND winter N >= 15
      SUMMER-DOM:   summer > winter + 0.10R AND summer N >= 15
      WINTER-ONLY:  winter > 0 AND summer <= 0 AND both N >= 10
      SUMMER-ONLY:  summer > 0 AND winter <= 0 AND both N >= 10
      BOTH-POS:     both halves positive but N too small for DOM/STABLE (e.g. summer N 10-14)
      LOW-N:        either regime < 10 trades
      UNSTABLE:     one or both halves negative and no cleaner label applies
    """
    if winter_n < 10 or summer_n < 10:
        return "LOW-N"

    if winter_avg_r is None or summer_avg_r is None:
        return "LOW-N"

    diff = abs(winter_avg_r - summer_avg_r)

    # Check for edge dying in one regime
    if winter_avg_r > 0 and summer_avg_r <= 0 and winter_n >= 10 and summer_n >= 10:
        return "WINTER-ONLY"
    if summer_avg_r > 0 and winter_avg_r <= 0 and winter_n >= 10 and summer_n >= 10:
        return "SUMMER-ONLY"

    # Stable
    if diff <= 0.10 and winter_n >= 15 and summer_n >= 15:
        return "STABLE"

    # Dominant
    if winter_avg_r > summer_avg_r + 0.10 and winter_n >= 15:
        return "WINTER-DOM"
    if summer_avg_r > winter_avg_r + 0.10 and summer_n >= 15:
        return "SUMMER-DOM"

    # Both halves profitable but couldn't reach STABLE/DOM thresholds (small N in one regime)
    if winter_avg_r > 0 and summer_avg_r > 0:
        return "BOTH-POS"

    return "UNSTABLE"


def cme_open_brisbane(trading_day: date) -> tuple[int, int]:
    """CME Globex electronic open (5:00 PM CT) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (CDT): 5PM CT = 22:00 UTC = 08:00 AEST
      Winter (CST): 5PM CT = 23:00 UTC = 09:00 AEST
    """
    ct_open = datetime(trading_day.year, trading_day.month, trading_day.day, 17, 0, 0, tzinfo=_US_CHICAGO)
    bris = ct_open.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def us_equity_open_brisbane(trading_day: date) -> tuple[int, int]:
    """NYSE cash open (09:30 ET) expressed in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 09:30 ET = 13:30 UTC = 23:30 AEST
      Winter (EST): 09:30 ET = 14:30 UTC = 00:30 AEST (next cal day)
    """
    et_open = datetime(trading_day.year, trading_day.month, trading_day.day, 9, 30, 0, tzinfo=_US_EASTERN)
    bris = et_open.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def us_data_open_brisbane(trading_day: date) -> tuple[int, int]:
    """US economic data release time (08:30 ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 08:30 ET = 12:30 UTC = 22:30 AEST
      Winter (EST): 08:30 ET = 13:30 UTC = 23:30 AEST
    """
    et_data = datetime(trading_day.year, trading_day.month, trading_day.day, 8, 30, 0, tzinfo=_US_EASTERN)
    bris = et_data.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def london_open_brisbane(trading_day: date) -> tuple[int, int]:
    """London metals open (08:00 London time) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (BST): 08:00 London = 07:00 UTC = 17:00 AEST
      Winter (GMT): 08:00 London = 08:00 UTC = 18:00 AEST
    """
    ldn_open = datetime(trading_day.year, trading_day.month, trading_day.day, 8, 0, 0, tzinfo=_UK_LONDON)
    bris = ldn_open.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def europe_flow_brisbane(trading_day: date) -> tuple[int, int]:
    """Hour adjacent to London metals open, opposite side of DST switch.

    Returns (hour, minute) in Australia/Brisbane.
      Winter (GMT): 07:00 London = 07:00 UTC = 17:00 AEST (pre-London-open flow)
      Summer (BST): 09:00 London = 08:00 UTC = 18:00 AEST (post-metals equity flow)
    """
    h_lm, m_lm = london_open_brisbane(trading_day)
    if is_uk_dst(trading_day):
        return (h_lm + 1, m_lm)  # Summer: LM=17:00, adjacent=18:00
    else:
        return (h_lm - 1, m_lm)  # Winter: LM=18:00, adjacent=17:00


def us_post_equity_brisbane(trading_day: date) -> tuple[int, int]:
    """US post-equity-open (10:00 AM ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 10:00 ET = 14:00 UTC = 00:00 AEST (next cal day)
      Winter (EST): 10:00 ET = 15:00 UTC = 01:00 AEST (next cal day)
    """
    et_time = datetime(trading_day.year, trading_day.month, trading_day.day, 10, 0, 0, tzinfo=_US_EASTERN)
    bris = et_time.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def cme_close_brisbane(trading_day: date) -> tuple[int, int]:
    """CME equity futures pre-close (2:45 PM CT) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (CDT): 2:45 PM CT = 19:45 UTC = 05:45 AEST
      Winter (CST): 2:45 PM CT = 20:45 UTC = 06:45 AEST
    """
    ct_close = datetime(trading_day.year, trading_day.month, trading_day.day, 14, 45, 0, tzinfo=_US_CHICAGO)
    bris = ct_close.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def tokyo_open_brisbane(trading_day: date) -> tuple[int, int]:
    """Tokyo Stock Exchange open (9:00 AM JST) in Brisbane local time.

    JST = UTC+9, Brisbane = UTC+10. Always 10:00 Brisbane. No DST.
    """
    return (10, 0)


def singapore_open_brisbane(trading_day: date) -> tuple[int, int]:
    """SGX/HKEX open (9:00 AM SGT) in Brisbane local time.

    SGT = UTC+8, Brisbane = UTC+10. Always 11:00 Brisbane. No DST.
    """
    return (11, 0)


def comex_settle_brisbane(trading_day: date) -> tuple[int, int]:
    """COMEX gold settlement (01:30 PM ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 01:30 PM ET = 17:30 UTC = 03:30 AEST (next cal day)
      Winter (EST): 01:30 PM ET = 18:30 UTC = 04:30 AEST (next cal day)
    """
    et_settle = datetime(trading_day.year, trading_day.month, trading_day.day, 13, 30, 0, tzinfo=_US_EASTERN)
    bris = et_settle.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def nyse_close_brisbane(trading_day: date) -> tuple[int, int]:
    """NYSE closing bell (04:00 PM ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 04:00 PM ET = 20:00 UTC = 06:00 AEST (next cal day)
      Winter (EST): 04:00 PM ET = 21:00 UTC = 07:00 AEST (next cal day)
    """
    et_close = datetime(trading_day.year, trading_day.month, trading_day.day, 16, 0, 0, tzinfo=_US_EASTERN)
    bris = et_close.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def fixed_1025_brisbane(trading_day: date) -> tuple[int, int]:
    """Fixed 10:25 AM Brisbane session. No market event anchor.

    Session discovery scan (2026-03-01): FDR survivor for MNQ.
    N=1,272-1,289, avgR=+0.221 to +0.247 (RR2.5-3.0), Sharpe_ann=2.09-2.21.
    Positive 6/6 years. Both DST seasons positive (Rw=+0.305, Rs=+0.189).
    Independent from 09:25 cluster — 1 hour later, inverted season bias.
    M2K: all strategies negative in pipeline (entry-model artifact — scan used
    close-based break detection; E2 stop-market gets stopped out before bar close
    on M2K's small ORBs at this time). Kept in M2K enabled_sessions for monitoring.
    No near existing session.
    """
    return (10, 25)


# =========================================================================
# SESSION CATALOG: master registry of all ORB sessions
# =========================================================================
# All sessions are dynamic (DST-aware, resolver function per-day).
# No fixed sessions or aliases remain after the Feb 2026 event-based rename.

SESSION_CATALOG = {
    "CME_REOPEN": {
        "type": "dynamic",
        "resolver": cme_open_brisbane,
        "break_group": "cme",
        "event": "CME Globex electronic reopen 5:00 PM CT",
    },
    "TOKYO_OPEN": {
        "type": "dynamic",
        "resolver": tokyo_open_brisbane,
        "break_group": "asia",
        "event": "Tokyo Stock Exchange open 9:00 AM JST",
    },
    "SINGAPORE_OPEN": {
        "type": "dynamic",
        "resolver": singapore_open_brisbane,
        "break_group": "asia",
        "event": "SGX/HKEX open 9:00 AM SGT",
    },
    "LONDON_METALS": {
        "type": "dynamic",
        "resolver": london_open_brisbane,
        "break_group": "london",
        "event": "London metals AM session 8:00 AM London",
    },
    "EUROPE_FLOW": {
        "type": "dynamic",
        "resolver": europe_flow_brisbane,
        "break_group": "london",
        "event": "European flow adjacent to London metals (7AM London winter / 9AM London summer)",
    },
    "US_DATA_830": {
        "type": "dynamic",
        "resolver": us_data_open_brisbane,
        "break_group": "us",
        "event": "US economic data release 8:30 AM ET",
    },
    "NYSE_OPEN": {
        "type": "dynamic",
        "resolver": us_equity_open_brisbane,
        "break_group": "us",
        "event": "NYSE cash open 9:30 AM ET",
    },
    "US_DATA_1000": {
        "type": "dynamic",
        "resolver": us_post_equity_brisbane,
        "break_group": "us",
        "event": "US 10:00 AM data (ISM/CC) + post-equity-open flow",
    },
    "COMEX_SETTLE": {
        "type": "dynamic",
        "resolver": comex_settle_brisbane,
        "break_group": "us",
        "event": "COMEX gold settlement 1:30 PM ET",
    },
    "CME_PRECLOSE": {
        "type": "dynamic",
        "resolver": cme_close_brisbane,
        "break_group": "us",
        "event": "CME equity futures pre-settlement 2:45 PM CT",
    },
    "NYSE_CLOSE": {
        "type": "dynamic",
        "resolver": nyse_close_brisbane,
        "break_group": "us",
        "event": "NYSE closing bell 4:00 PM ET",
    },
    "BRISBANE_1025": {
        "type": "dynamic",
        "resolver": fixed_1025_brisbane,
        "break_group": "asia",
        "event": "Fixed 10:25 AM Brisbane (not event-relative)",
    },
}


def get_break_group(orb_label: str) -> str | None:
    """Return the break_group for an ORB label.

    Returns None if label not in catalog or is an alias.
    """
    entry = SESSION_CATALOG.get(orb_label)
    if entry is None or entry["type"] == "alias":
        return None
    return entry.get("break_group")


# Only non-alias dynamic sessions get resolvers
DYNAMIC_ORB_RESOLVERS = {
    label: entry["resolver"] for label, entry in SESSION_CATALOG.items() if entry["type"] == "dynamic"
}


def orb_utc_window(trading_day: date, orb_label: str, orb_minutes: int) -> tuple[datetime, datetime]:
    """
    Compute the [start, end) UTC window for an ORB on a given trading day.

    The ORB starts at the session's Brisbane local time (dynamic, DST-aware
    per DYNAMIC_ORB_RESOLVERS) and lasts `orb_minutes`. The returned range
    is [start, end) — start inclusive, end exclusive.

    Example: CME_REOPEN ORB with 5-minute aperture on 2024-01-05
      local start = 2024-01-05 09:00 Brisbane
      local end   = 2024-01-05 09:05 Brisbane
      UTC start   = 2024-01-04 23:00 UTC
      UTC end     = 2024-01-04 23:05 UTC

    Special case: NYSE_OPEN ORB belongs to the SAME trading day but falls at
    00:30 the NEXT calendar day in Brisbane. The `hour < 9` branch below
    handles this — the local calendar date is bumped forward by one day so
    the Brisbane timestamp is constructed correctly.

      trading_day 2024-01-05, NYSE_OPEN ORB:
        local start = 2024-01-06 00:30 Brisbane (next calendar day)
        UTC start   = 2024-01-05 14:30 UTC

    Canonical contract (per E2 canonical-window refactor, 2026-04-07):
      - Single source of truth: every E2 producer (backtest + live + feature
        builder) calls this function. No parallel re-encoding allowed.
      - Fail-closed on `orb_label` not registered in DYNAMIC_ORB_RESOLVERS
        (caller bug — unknown session).
      - Fail-closed on `orb_minutes` outside [_ORB_MINUTES_MIN, _ORB_MINUTES_MAX]
        (caller bug — invalid aperture). Note: the predecessor function
        pipeline/build_daily_features._orb_utc_window did NOT validate this
        input; the canonical version tightens the contract. Production
        callers use VALID_ORB_MINUTES = {5, 15, 30} which are all within
        bounds, so the tightening is zero-impact on real pipeline runs.
      - Fail-closed if the resolved UTC start falls outside the trading day's
        [start, end) window (caller bug or session-catalog drift).
      - Idempotent: same inputs always return the same outputs.

    Raises:
        ValueError: If orb_label is not in DYNAMIC_ORB_RESOLVERS, if
            orb_minutes is outside valid bounds, or if the resolved window
            falls outside the trading day's UTC range.
    """
    # Validate orb_minutes explicitly — canonical tightening over predecessor
    # in build_daily_features._orb_utc_window (which accepted any int). The
    # bounds catch zero, negative, and implausibly-large apertures.
    if not (_ORB_MINUTES_MIN <= orb_minutes <= _ORB_MINUTES_MAX):
        raise ValueError(f"orb_minutes={orb_minutes} outside valid range [{_ORB_MINUTES_MIN}, {_ORB_MINUTES_MAX}]")

    # Resolve the session's Brisbane (hour, minute) for this trading day.
    # DYNAMIC_ORB_RESOLVERS holds the per-session DST-aware resolver callables.
    if orb_label not in DYNAMIC_ORB_RESOLVERS:
        raise ValueError(f"Unknown ORB label '{orb_label}' — not in DYNAMIC_ORB_RESOLVERS")
    hour, minute = DYNAMIC_ORB_RESOLVERS[orb_label](trading_day)

    # Determine the Brisbane calendar date for this ORB time. The trading day
    # starts at 09:00 Brisbane, so:
    #   - hour in [09, 23] → ORB is on the same calendar date as trading_day
    #   - hour in [00, 08] → ORB is on the NEXT calendar date (e.g., NYSE_OPEN)
    if hour < TRADING_DAY_START_HOUR_LOCAL:
        cal_date = trading_day + timedelta(days=1)
    else:
        cal_date = trading_day

    local_start = datetime(
        cal_date.year,
        cal_date.month,
        cal_date.day,
        hour,
        minute,
        0,
        tzinfo=BRISBANE_TZ,
    )
    local_end = local_start + timedelta(minutes=orb_minutes)

    utc_start = local_start.astimezone(UTC_TZ)
    utc_end = local_end.astimezone(UTC_TZ)

    # Fail-closed: the resolved ORB start MUST fall within the trading day's
    # UTC window. If it doesn't, the session catalog or resolver has drifted
    # and we should surface that immediately rather than silently producing
    # the wrong window.
    td_start, td_end = compute_trading_day_utc_range(trading_day)
    if not (td_start <= utc_start < td_end):
        raise ValueError(
            f"ORB {orb_label} on {trading_day} resolved to {utc_start} UTC, "
            f"outside trading day window [{td_start}, {td_end})"
        )

    return utc_start, utc_end


def validate_catalog():
    """Fail-closed: verify no two non-alias sessions PERMANENTLY collide.

    Checks both a winter and summer date. A collision is only an error if
    two sessions resolve to the same time on ALL test dates. Seasonal
    overlaps (e.g., CME_REOPEN = 09:00 Brisbane in winter but 08:00 in summer) are
    expected and acceptable -- that's why dynamic sessions exist.

    Raises ValueError if any permanent collision is found.
    """
    test_dates = [date(2025, 1, 15), date(2025, 7, 15)]  # winter + summer

    # Collect collisions per date
    collisions_per_date = []
    for td in test_dates:
        times = {}
        date_collisions = set()
        for label, entry in SESSION_CATALOG.items():
            if entry["type"] == "alias":
                continue
            if entry["type"] == "dynamic":
                h, m = entry["resolver"](td)
            else:
                h, m = entry["brisbane"]
            key = (h, m)
            if key in times:
                pair = tuple(sorted([label, times[key]]))
                date_collisions.add(pair)
            times[key] = label
        collisions_per_date.append(date_collisions)

    # Permanent collisions: pairs that collide on ALL test dates
    if collisions_per_date:
        permanent = collisions_per_date[0]
        for dc in collisions_per_date[1:]:
            permanent = permanent & dc

        if permanent:
            pairs = ", ".join(f"{a}+{b}" for a, b in sorted(permanent))
            raise ValueError(f"Permanent collision (same time on all test dates): {pairs}")
