"""
Calendar-based date flags: NFP, OPEX, Friday, day-of-week.

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
