"""
DST detection and dynamic session resolvers.

Determines whether the US or UK is in Daylight Saving Time on a given
trading day, and resolves dynamic ORB session times to Brisbane local
hours accordingly.

Dynamic sessions track specific market events regardless of DST:
  CME_OPEN        - CME Globex electronic open at 5:00 PM CT
  US_EQUITY_OPEN  - NYSE cash open at 09:30 ET (MES, MNQ)
  US_DATA_OPEN    - Economic data releases at 08:30 ET (MGC)
  LONDON_OPEN     - London metals open at 08:00 London time (MGC)
  CME_CLOSE       - CME equity futures pre-close at 2:45 PM CT (MNQ, MES)

Uses zoneinfo (stdlib) for all timezone math -- correct for all years,
handles edge cases automatically.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

_US_EASTERN = ZoneInfo("America/New_York")
_US_CHICAGO = ZoneInfo("America/Chicago")
_UK_LONDON = ZoneInfo("Europe/London")
_BRISBANE = ZoneInfo("Australia/Brisbane")
_UTC = ZoneInfo("UTC")


def is_us_dst(trading_day: date) -> bool:
    """True if US Eastern is in Daylight Saving Time (EDT, UTC-4) on this date.

    During DST (roughly Mar second Sunday -> Nov first Sunday):
      EDT = UTC-4, so 08:30 ET = 12:30 UTC
    Standard time (EST = UTC-5):
      08:30 ET = 13:30 UTC
    """
    # Use noon to avoid any ambiguity on transition days
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_US_EASTERN)
    return dt.utcoffset().total_seconds() == -4 * 3600


def is_uk_dst(trading_day: date) -> bool:
    """True if UK is in British Summer Time (BST, UTC+1) on this date.

    During BST (roughly Mar last Sunday -> Oct last Sunday):
      BST = UTC+1, so 08:00 London = 07:00 UTC
    Standard time (GMT = UTC+0):
      08:00 London = 08:00 UTC
    """
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_UK_LONDON)
    return dt.utcoffset().total_seconds() == 1 * 3600


# =========================================================================
# DST REGIME HELPERS — for winter/summer split analysis
# =========================================================================

# Sessions affected by DST and which timezone governs them.
# "US" = winter/summer split uses is_us_dst(). "UK" = uses is_uk_dst().
#
# Alignment notes (Brisbane = UTC+10, no DST):
#   0900: Winter = CME open exactly. Summer = 1hr AFTER CME open.
#   1800: Winter = London open exactly. Summer = 1hr AFTER London open.
#   0030: Winter = US equity open exactly. Summer = 1hr AFTER equity open.
#   2300: NEVER aligned with US data release (8:30 ET = 23:30 winter / 22:30 summer).
#         Winter: 2300 = 30min BEFORE data release (pre-positioning).
#         Summer: 2300 = 30min AFTER data release (reaction window).
#         Volume confirms: summer +76-90% (data already out, market reacting).
#         DST classification still correct — US DST flips which side of data event 2300 sits on.
DST_AFFECTED_SESSIONS = {
    "0900": "US",   # CME open shifts with US DST
    "0030": "US",   # US equity open shifts with US DST
    "2300": "US",   # US morning activity context shifts; winter=pre-data, summer=post-data
    "1800": "UK",   # London open shifts with UK DST
}

# Sessions NOT affected (Asia has no DST; dynamic sessions self-adjust)
DST_CLEAN_SESSIONS = {"1000", "1100", "1130",
                       "CME_OPEN", "LONDON_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN",
                       "US_POST_EQUITY", "CME_CLOSE"}


# =========================================================================
# DOW ALIGNMENT — Brisbane DOW vs Exchange-Timezone DOW
# =========================================================================
# For each fixed session, whether the Brisbane calendar day matches the
# exchange's calendar day.  Determines if a DayOfWeekSkipFilter targeting
# e.g. "Friday" actually skips the exchange's Friday session.
#
# Investigation: research/research_dow_alignment.py (Feb 2026)
#
# Result: all sessions EXCEPT 0030 are aligned.
#   0900: Brisbane-Fri = CME Friday (5PM Thu CT = start of CME Fri session) ✓
#   1000: Brisbane DOW = Tokyo DOW (no DST, same calendar day) ✓
#   1100: Brisbane DOW = Singapore DOW (no DST) ✓
#   1130: Brisbane DOW = HK DOW (no DST) ✓
#   1800: Brisbane DOW = London DOW (both morning same calendar day) ✓
#   2300: Brisbane DOW = US DOW (13:00 UTC = US morning same day) ✓
#   0030: Brisbane DOW = US DOW + 1 (00:30 Bris = 14:30 UTC PREV day) ✗
#
# The 0030 mismatch: Brisbane 00:30 crosses midnight from the previous UTC
# day.  Brisbane-Friday 00:30 = UTC Thursday 14:30 = US Thursday 9:30 AM.
# So Brisbane-Friday at 0030 is the US THURSDAY equity open.
# Any DOW filter for 0030 must account for this -1 day offset.

DOW_ALIGNED_SESSIONS = {"0900", "1000", "1100", "1130", "1800", "2300"}
DOW_MISALIGNED_SESSIONS = {
    "0030": -1,  # Brisbane DOW = exchange DOW + 1 (i.e. exchange is 1 day behind)
}


def validate_dow_filter_alignment(session: str, skip_days: tuple[int, ...]) -> None:
    """Fail-closed guard: prevent DOW filters on sessions with known misalignment.

    Raises ValueError if a DOW skip filter is applied to a session where
    Brisbane DOW != exchange DOW, unless the caller has explicitly handled
    the offset (not yet implemented — 0030 has no DOW filters currently).
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
    Uses US Eastern for 0900/0030/2300, UK London for 1800.
    """
    dst_type = DST_AFFECTED_SESSIONS.get(orb_label)
    if dst_type is None:
        return None  # Clean session
    if dst_type == "US":
        return not is_us_dst(trading_day)  # winter = NOT DST
    else:  # UK
        return not is_uk_dst(trading_day)  # winter = NOT BST


def classify_dst_verdict(winter_avg_r: float | None, summer_avg_r: float | None,
                         winter_n: int, summer_n: int) -> str:
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
    ct_open = datetime(trading_day.year, trading_day.month, trading_day.day,
                       17, 0, 0, tzinfo=_US_CHICAGO)
    bris = ct_open.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def us_equity_open_brisbane(trading_day: date) -> tuple[int, int]:
    """NYSE cash open (09:30 ET) expressed in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 09:30 ET = 13:30 UTC = 23:30 AEST
      Winter (EST): 09:30 ET = 14:30 UTC = 00:30 AEST (next cal day)
    """
    et_open = datetime(trading_day.year, trading_day.month, trading_day.day,
                       9, 30, 0, tzinfo=_US_EASTERN)
    bris = et_open.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def us_data_open_brisbane(trading_day: date) -> tuple[int, int]:
    """US economic data release time (08:30 ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 08:30 ET = 12:30 UTC = 22:30 AEST
      Winter (EST): 08:30 ET = 13:30 UTC = 23:30 AEST
    """
    et_data = datetime(trading_day.year, trading_day.month, trading_day.day,
                       8, 30, 0, tzinfo=_US_EASTERN)
    bris = et_data.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def london_open_brisbane(trading_day: date) -> tuple[int, int]:
    """London metals open (08:00 London time) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (BST): 08:00 London = 07:00 UTC = 17:00 AEST
      Winter (GMT): 08:00 London = 08:00 UTC = 18:00 AEST
    """
    ldn_open = datetime(trading_day.year, trading_day.month, trading_day.day,
                        8, 0, 0, tzinfo=_UK_LONDON)
    bris = ldn_open.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def us_post_equity_brisbane(trading_day: date) -> tuple[int, int]:
    """US post-equity-open (10:00 AM ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 10:00 ET = 14:00 UTC = 00:00 AEST (next cal day)
      Winter (EST): 10:00 ET = 15:00 UTC = 01:00 AEST (next cal day)
    """
    et_time = datetime(trading_day.year, trading_day.month, trading_day.day,
                       10, 0, 0, tzinfo=_US_EASTERN)
    bris = et_time.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def cme_close_brisbane(trading_day: date) -> tuple[int, int]:
    """CME equity futures pre-close (2:45 PM CT) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (CDT): 2:45 PM CT = 19:45 UTC = 05:45 AEST
      Winter (CST): 2:45 PM CT = 20:45 UTC = 06:45 AEST
    """
    ct_close = datetime(trading_day.year, trading_day.month, trading_day.day,
                        14, 45, 0, tzinfo=_US_CHICAGO)
    bris = ct_close.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


# =========================================================================
# SESSION CATALOG: master registry of all ORB sessions
# =========================================================================
# Three entry types:
#   "dynamic" - DST-aware, resolver function per-day
#   "fixed"   - constant Brisbane time, no resolver needed
#   "alias"   - maps to existing ORB label, NO separate column

SESSION_CATALOG = {
    # Dynamic sessions (DST-aware, resolver per-day)
    "CME_OPEN": {
        "type": "dynamic",
        "resolver": cme_open_brisbane,
        "break_group": "cme",
        "event": "CME Globex electronic open 5:00 PM CT",
    },
    "US_EQUITY_OPEN": {
        "type": "dynamic",
        "resolver": us_equity_open_brisbane,
        "break_group": "us",
        "event": "NYSE cash open 9:30 AM ET",
    },
    "US_DATA_OPEN": {
        "type": "dynamic",
        "resolver": us_data_open_brisbane,
        "break_group": "us",
        "event": "US economic data release 8:30 AM ET",
    },
    "LONDON_OPEN": {
        "type": "dynamic",
        "resolver": london_open_brisbane,
        "break_group": "london",
        "event": "London metals open 8:00 AM London",
    },
    "US_POST_EQUITY": {
        "type": "dynamic",
        "resolver": us_post_equity_brisbane,
        "break_group": "us",
        "event": "US post-equity-open 10:00 AM ET (~30min after NYSE cash open)",
    },
    "CME_CLOSE": {
        "type": "dynamic",
        "resolver": cme_close_brisbane,
        "break_group": "us",
        "event": "CME equity futures pre-close 2:45 PM CT",
    },
    # Fixed sessions (constant UTC, no DST)
    #
    # break_group: sessions in the same group share a break-window boundary.
    # Break detection extends to the start of the next GROUP, not the next
    # label. This prevents adding a nearby session (e.g., 1130) from silently
    # shrinking an existing session's break window (e.g., 1100).
    "0900": {
        "type": "fixed",
        "brisbane": (9, 0),
        "break_group": "cme",
        "event": "23:00 UTC -- CME open in winter / 1hr after in summer",
    },
    "1000": {
        "type": "fixed",
        "brisbane": (10, 0),
        "break_group": "asia",
        "event": "00:00 UTC -- Tokyo 9AM JST (no DST)",
    },
    "1100": {
        "type": "fixed",
        "brisbane": (11, 0),
        "break_group": "asia",
        "event": "01:00 UTC -- ~Singapore/Shanghai open (no DST)",
    },
    "1130": {
        "type": "fixed",
        "brisbane": (11, 30),
        "break_group": "asia",
        "event": "01:30 UTC -- HK/SG equity open 9:30 AM HKT (no DST)",
    },
    "1800": {
        "type": "fixed",
        "brisbane": (18, 0),
        "break_group": "london",
        "event": "08:00 UTC -- London metals in winter / 1hr after summer",
    },
    "2300": {
        "type": "fixed",
        "brisbane": (23, 0),
        "break_group": "us",
        "event": "13:00 UTC -- 30min pre-data(W) / 30min post-data(S), never aligned",
    },
    "0030": {
        "type": "fixed",
        "brisbane": (0, 30),
        "break_group": "us",
        "event": "14:30 UTC -- NYSE 9:30 ET in winter / 1hr after summer",
    },
    # Aliases (map to existing ORB label -- NO separate column)
    "TOKYO_OPEN": {
        "type": "alias",
        "maps_to": "1000",
        "event": "Tokyo Stock Exchange 9:00 AM JST = 00:00 UTC = 10:00 AEST",
    },
    "HK_SG_OPEN": {
        "type": "alias",
        "maps_to": "1130",
        "event": "HKEX/SGX 9:30 AM HKT = 01:30 UTC = 11:30 AEST",
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
    label: entry["resolver"]
    for label, entry in SESSION_CATALOG.items()
    if entry["type"] == "dynamic"
}


def validate_catalog():
    """Fail-closed: verify no two non-alias sessions PERMANENTLY collide.

    Checks both a winter and summer date. A collision is only an error if
    two sessions resolve to the same time on ALL test dates. Seasonal
    overlaps (e.g., CME_OPEN = 0900 in winter but diverges in summer) are
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
            raise ValueError(
                f"Permanent collision (same time on all test dates): {pairs}"
            )
