"""ORB state tracker — tracks live ORB aperture and filter qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OrbState:
    """Snapshot of a live ORB aperture being formed."""

    session: str
    instrument: str
    orb_minutes: int
    high: float
    low: float
    size: float
    bars_elapsed: int
    bars_total: int
    started_at: datetime | None = None
    qualifications: dict[str, bool] = field(default_factory=dict)


class OrbTracker:
    """Track live ORB formation across (session, instrument) pairs."""

    def __init__(self) -> None:
        self._active: dict[tuple[str, str], OrbState] = {}

    def start_orb(
        self,
        session: str,
        instrument: str,
        orb_minutes: int,
        first_bar: dict,
    ) -> OrbState:
        """Begin tracking a new ORB aperture.

        Args:
            session: Session label (e.g. 'CME_REOPEN').
            instrument: Symbol (e.g. 'MGC').
            orb_minutes: ORB aperture in minutes (5, 15, 30).
            first_bar: Dict with at least 'high', 'low', and optionally 'ts'.

        Returns:
            The newly created OrbState.
        """
        high = first_bar["high"]
        low = first_bar["low"]
        state = OrbState(
            session=session,
            instrument=instrument,
            orb_minutes=orb_minutes,
            high=high,
            low=low,
            size=high - low,
            bars_elapsed=1,
            bars_total=orb_minutes,
            started_at=first_bar.get("ts"),
        )
        self._active[(session, instrument)] = state
        return state

    def update_bar(
        self,
        session: str,
        instrument: str,
        bar_high: float,
        bar_low: float,
    ) -> OrbState | None:
        """Update an active ORB with a new bar.

        Returns the updated state, or None if not currently tracking
        this (session, instrument) pair.  Auto-completes when
        bars_elapsed reaches bars_total.
        """
        key = (session, instrument)
        state = self._active.get(key)
        if state is None:
            return None

        state.high = max(state.high, bar_high)
        state.low = min(state.low, bar_low)
        state.size = state.high - state.low
        state.bars_elapsed += 1

        if state.bars_elapsed >= state.bars_total:
            return self.complete_orb(session, instrument)

        return state

    def complete_orb(self, session: str, instrument: str) -> OrbState | None:
        """Mark an ORB as complete and remove from active tracking.

        Returns the final OrbState, or None if not tracking.
        """
        return self._active.pop((session, instrument), None)

    def get_state(self, session: str, instrument: str) -> OrbState | None:
        """Return the current OrbState, or None if not tracking."""
        return self._active.get((session, instrument))

    @staticmethod
    def to_dict(state: OrbState) -> dict:
        """Serialize an OrbState for JSON / SSE broadcast."""
        return {
            "session": state.session,
            "instrument": state.instrument,
            "orb_minutes": state.orb_minutes,
            "high": state.high,
            "low": state.low,
            "size": state.size,
            "bars_elapsed": state.bars_elapsed,
            "bars_total": state.bars_total,
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "qualifications": state.qualifications,
        }
