"""
Builds ORB ranges incrementally from live 1-minute bars.

CRITICAL: DYNAMIC_ORB_RESOLVERS[label](date) returns (hour, minute) in
Brisbane local time (UTC+10, no DST). Must convert to UTC for bar comparisons.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from pipeline.dst import DYNAMIC_ORB_RESOLVERS
from trading_app.live.bar_aggregator import Bar

_BRISBANE = ZoneInfo("Australia/Brisbane")
_UTC = UTC


def _session_start_utc(session_label: str, trading_day: date) -> datetime | None:
    """
    Get session start as UTC datetime for a given Brisbane trading_day.

    DYNAMIC_ORB_RESOLVERS[label](date) returns (hour, minute) in Brisbane local time.
    Sessions before 09:00 Brisbane (e.g. NYSE_OPEN at 00:30 in winter) fall on
    trading_day + 1 in calendar terms — same logic as execution_engine.py:276-279.
    """
    resolver = DYNAMIC_ORB_RESOLVERS.get(session_label)
    if resolver is None:
        return None
    try:
        bris_h, bris_m = resolver(trading_day)
    except Exception:
        return None

    # Before 09:00 Brisbane = next calendar day (midnight-crossing sessions)
    cal_date = trading_day + timedelta(days=1) if bris_h < 9 else trading_day
    bris_dt = datetime(cal_date.year, cal_date.month, cal_date.day, bris_h, bris_m, 0, tzinfo=_BRISBANE)
    return bris_dt.astimezone(_UTC)


@dataclass
class LiveORB:
    high: float
    low: float
    size: float
    bars_count: int
    complete: bool


class LiveORBBuilder:
    """
    Accumulates 1-minute live bars and computes ORB ranges on demand.

    Call on_bar() for each completed bar from DataFeed.
    Call get_orb(session_label, orb_minutes) to check if the ORB is ready.
    """

    def __init__(self, instrument: str, trading_day: date):
        self.instrument = instrument
        self.trading_day = trading_day
        self._bars: list[Bar] = []

    def on_bar(self, bar: Bar) -> None:
        """Append a completed 1-minute bar."""
        self._bars.append(bar)

    def get_orb(self, session_label: str, orb_minutes: int) -> LiveORB | None:
        """
        Return the completed ORB for this session/aperture, or None if not enough bars yet.
        """
        start_utc = _session_start_utc(session_label, self.trading_day)
        if start_utc is None:
            return None

        end_utc = start_utc + timedelta(minutes=orb_minutes)
        orb_bars = [b for b in self._bars if start_utc <= b.ts_utc < end_utc]

        if len(orb_bars) < orb_minutes:
            return None

        high = max(b.high for b in orb_bars)
        low = min(b.low for b in orb_bars)
        return LiveORB(
            high=high,
            low=low,
            size=round(high - low, 4),
            bars_count=len(orb_bars),
            complete=True,
        )

    def get_bars_since(self, session_label: str) -> list[Bar]:
        """Return all bars at or after session start (for live engine replay)."""
        start = _session_start_utc(session_label, self.trading_day)
        if start is None:
            return []
        return [b for b in self._bars if b.ts_utc >= start]
