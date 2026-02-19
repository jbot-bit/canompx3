"""
Calendar-based skip filters: NFP, OPEX, Friday.

Pure date functions — no timezone logic, no DB access.
These flags are deterministic from the trading day date alone.

NFP (Non-Farm Payrolls): Released on the first Friday of each month.
  Random spike destroys ORB signal. Universally toxic across instruments.

OPEX (Options Expiration): Third Friday of each month.
  Options pinning kills follow-through. Negative expectancy on MNQ.

Friday: Position-squaring mechanism at 0900 session specifically.
  Not a universal skip — only applies to session 0900.
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
