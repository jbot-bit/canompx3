"""
Calendar-based skip filters: NFP, OPEX, Friday, day-of-week.

Pure date functions — no timezone logic, no DB access.
These flags are deterministic from the trading day date alone.

NFP (Non-Farm Payrolls): Released on the first Friday of each month.
  Random spike destroys ORB signal. Universally toxic across instruments.

OPEX (Options Expiration): Third Friday of each month.
  Options pinning kills follow-through. Negative expectancy on MNQ.

Friday: Position-squaring mechanism at 0900 session specifically.
  Not a universal skip — only applies to session 0900.

Day-of-week (DOW): Session-specific skip rules from DOW research (Feb 2026).
  0900: Skip Friday (position-squaring kills follow-through).
  1800: Skip Monday (thin London open, no follow-through).
  1000: Skip Tuesday (consistently weakest day at Tokyo session).
"""

from datetime import date


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
