"""
Calendar-based date flags: NFP, OPEX, FOMC, CPI, month-end, OPEX-week, day-of-week.

Pure date functions — no timezone logic, no DB access.
These flags are deterministic from the trading day date alone.

IMPORTANT: Calendar effects are INSTRUMENT x SESSION SPECIFIC, not universal.
See research/research_calendar_effects.py (Mar 2026) for the comprehensive
BH FDR analysis. The blanket "skip all NFP/OPEX" approach was proven wrong —
some instrument×session combos are BETTER on NFP/OPEX days.

NFP (Non-Farm Payrolls): First Friday of month.
  Effect is mixed — WORSE for MGC TOKYO_OPEN, MNQ CME_PRECLOSE, M2K NYSE_OPEN.
  BETTER for MES US_DATA_1000, MNQ NYSE_OPEN, MGC US_DATA_1000.
  NOT universally toxic.

OPEX (Options Expiration): Third Friday of month.
  WORSE for MGC NYSE_OPEN (9/11 years consistent).
  BETTER for MNQ NYSE_OPEN (6/6 years), MNQ SINGAPORE_OPEN (5/6 years),
  MES NYSE_CLOSE (+0.46R), MES LONDON_METALS, MES NYSE_OPEN.
  NOT universally negative.

Day-of-week: Session-specific, varies by instrument.
"""

from datetime import date, timedelta


def is_nfp_day(d: date) -> bool:
    """True if d is the first Friday of its month (NFP release day)."""
    if d.weekday() != 4:  # Not Friday
        return False
    return d.day <= 7


def is_opex_day(d: date) -> bool:
    """True if d is the third Friday of its month (monthly options expiration)."""
    if d.weekday() != 4:  # Not Friday
        return False
    return 15 <= d.day <= 21


def is_friday(d: date) -> bool:
    """True if d is a Friday."""
    return d.weekday() == 4


def is_monday(d: date) -> bool:
    """True if d is a Monday."""
    return d.weekday() == 0


def is_tuesday(d: date) -> bool:
    """True if d is a Tuesday."""
    return d.weekday() == 1


def day_of_week(d: date) -> int:
    """Return Python weekday: 0=Monday, 1=Tuesday, ..., 4=Friday, 5=Saturday, 6=Sunday."""
    return d.weekday()


# =========================================================================
# FOMC / CPI date sets (hardcoded from official sources)
# =========================================================================

# FOMC announcement dates
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Last updated: 2026-03-08 (covers through 2026-03-18)
_FOMC_DATES_RAW = [
    # 2016
    "2016-01-27",
    "2016-03-16",
    "2016-04-27",
    "2016-06-15",
    "2016-07-27",
    "2016-09-21",
    "2016-11-02",
    "2016-12-14",
    # 2017
    "2017-02-01",
    "2017-03-15",
    "2017-05-03",
    "2017-06-14",
    "2017-07-26",
    "2017-09-20",
    "2017-11-01",
    "2017-12-13",
    # 2018
    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",
    # 2019
    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-30",
    "2019-12-11",
    # 2020
    "2020-01-29",
    "2020-03-03",
    "2020-03-15",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    # 2022
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    # 2023
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-17",
    # 2026
    "2026-01-28",
    "2026-03-18",
]

# CPI release dates (typically around 10th-13th of month)
# Source: https://www.bls.gov/schedule/news_release/cpi.htm
# Last updated: 2026-03-08 (covers through 2026-02-11)
_CPI_DATES_RAW = [
    # 2020
    "2020-01-14",
    "2020-02-13",
    "2020-03-11",
    "2020-04-10",
    "2020-05-12",
    "2020-06-10",
    "2020-07-14",
    "2020-08-12",
    "2020-09-11",
    "2020-10-13",
    "2020-11-12",
    "2020-12-10",
    # 2021
    "2021-01-13",
    "2021-02-10",
    "2021-03-10",
    "2021-04-13",
    "2021-05-12",
    "2021-06-10",
    "2021-07-13",
    "2021-08-11",
    "2021-09-14",
    "2021-10-13",
    "2021-11-10",
    "2021-12-10",
    # 2022
    "2022-01-12",
    "2022-02-10",
    "2022-03-10",
    "2022-04-12",
    "2022-05-11",
    "2022-06-10",
    "2022-07-13",
    "2022-08-10",
    "2022-09-13",
    "2022-10-13",
    "2022-11-10",
    "2022-12-13",
    # 2023
    "2023-01-12",
    "2023-02-14",
    "2023-03-14",
    "2023-04-12",
    "2023-05-10",
    "2023-06-13",
    "2023-07-12",
    "2023-08-10",
    "2023-09-13",
    "2023-10-12",
    "2023-11-14",
    "2023-12-12",
    # 2024
    "2024-01-11",
    "2024-02-13",
    "2024-03-12",
    "2024-04-10",
    "2024-05-15",
    "2024-06-12",
    "2024-07-11",
    "2024-08-14",
    "2024-09-11",
    "2024-10-10",
    "2024-11-13",
    "2024-12-11",
    # 2025
    "2025-01-15",
    "2025-02-12",
    "2025-03-12",
    "2025-04-10",
    "2025-05-13",
    "2025-06-11",
    "2025-07-15",
    "2025-08-12",
    "2025-09-10",
    "2025-10-14",
    "2025-11-12",
    "2025-12-10",
    # 2026
    "2026-01-13",
    "2026-02-11",
]


def build_fomc_set() -> set[date]:
    """FOMC announcement day + day after."""
    dates = set()
    for d_str in _FOMC_DATES_RAW:
        d = date.fromisoformat(d_str)
        dates.add(d)
        dates.add(d + timedelta(days=1))
    return dates


def build_cpi_set() -> set[date]:
    """CPI release day."""
    return {date.fromisoformat(d) for d in _CPI_DATES_RAW}


def is_month_end(td: date, all_trading_days: set[date], window: int = 2) -> bool:
    """True if td is within the last `window` trading days of its month."""
    if td.month == 12:
        next_month_first = date(td.year + 1, 1, 1)
    else:
        next_month_first = date(td.year, td.month + 1, 1)
    count = 0
    check = td
    while check < next_month_first:
        if check in all_trading_days:
            count += 1
        check += timedelta(days=1)
    return count <= window


def is_month_start(td: date, all_trading_days: set[date], window: int = 2) -> bool:
    """True if td is within the first `window` trading days of its month."""
    first_of_month = date(td.year, td.month, 1)
    count = 0
    check = first_of_month
    while check <= td:
        if check in all_trading_days:
            count += 1
        check += timedelta(days=1)
    return count <= window


def is_quarter_end(td: date, all_trading_days: set[date], window: int = 2) -> bool:
    """True if td is within the last `window` trading days of a quarter."""
    if td.month not in (3, 6, 9, 12):
        return False
    return is_month_end(td, all_trading_days, window)


def opex_week_dates(yr_start: int = 2016, yr_end: int = 2027) -> set[date]:
    """All Mon-Fri of OPEX week (the week containing third Friday)."""
    dates = set()
    for year in range(yr_start, yr_end):
        for month in range(1, 13):
            d = date(year, month, 1)
            friday_count = 0
            while True:
                if d.weekday() == 4:
                    friday_count += 1
                    if friday_count == 3:
                        break
                d += timedelta(days=1)
            monday = d - timedelta(days=4)
            for i in range(5):
                dates.add(monday + timedelta(days=i))
    return dates
