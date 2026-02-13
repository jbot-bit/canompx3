"""
DST detection and dynamic session resolvers.

Determines whether the US or UK is in Daylight Saving Time on a given
trading day, and resolves dynamic ORB session times to Brisbane local
hours accordingly.

Dynamic sessions track specific market events regardless of DST:
  US_EQUITY_OPEN  - NYSE cash open at 09:30 ET (MES, MNQ)
  US_DATA_OPEN    - Economic data releases at 08:30 ET (MGC)
  LONDON_OPEN     - London metals open at 08:00 London time (MGC)

Uses zoneinfo (stdlib) for all timezone math -- correct for all years,
handles edge cases automatically.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

_US_EASTERN = ZoneInfo("America/New_York")
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


# Registry for dynamic ORB resolvers
DYNAMIC_ORB_RESOLVERS = {
    "US_EQUITY_OPEN": us_equity_open_brisbane,
    "US_DATA_OPEN": us_data_open_brisbane,
    "LONDON_OPEN": london_open_brisbane,
}
