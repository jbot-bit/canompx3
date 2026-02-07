"""
Entry detection logic with confirm_bars support.

Confirm bars require N consecutive 1m closes outside the ORB range
before confirming an entry signal. This filters fakeout breaks.

Rules (CANONICAL_LOGIC.txt section 8):
  - confirm_bars=1: Enter on first close outside ORB (same as current pipeline)
  - confirm_bars=2: Wait for 2nd consecutive close outside ORB
  - confirm_bars=3: Wait for 3rd consecutive close outside ORB
  - Reset: If any bar closes back INSIDE ORB, count resets to 0
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EntrySignal:
    """Result of entry detection with confirm_bars."""

    triggered: bool
    entry_ts: datetime | None
    entry_price: float | None
    stop_price: float | None


def detect_entry_with_confirm_bars(
    bars_df: pd.DataFrame,
    orb_break_ts: datetime,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    confirm_bars: int,
    detection_window_end: datetime,
) -> EntrySignal:
    """
    Detect entry after N consecutive closes outside ORB.

    Args:
        bars_df: DataFrame with ts_utc, open, high, low, close columns
        orb_break_ts: Timestamp of first break (from detect_break())
        orb_high: ORB high price
        orb_low: ORB low price
        break_dir: 'long' or 'short'
        confirm_bars: Number of consecutive closes required (1, 2, or 3)
        detection_window_end: End of window for confirmation

    Returns:
        EntrySignal with triggered=True if confirmed, False otherwise.
        entry_price is the ORB breakout level (orb_high for long, orb_low for short).
        entry_ts is the timestamp of the confirming bar (the Nth consecutive close).
    """
    if confirm_bars < 1 or confirm_bars > 10:
        raise ValueError(f"confirm_bars must be 1-10, got {confirm_bars}")

    if break_dir not in ("long", "short"):
        raise ValueError(f"break_dir must be 'long' or 'short', got {break_dir}")

    # Get bars from break_ts onwards within window
    candidate_bars = bars_df[
        (bars_df["ts_utc"] >= orb_break_ts)
        & (bars_df["ts_utc"] < detection_window_end)
    ].sort_values("ts_utc")

    if candidate_bars.empty:
        return EntrySignal(
            triggered=False, entry_ts=None, entry_price=None, stop_price=None
        )

    closes = candidate_bars["close"].values

    # Boolean mask: close outside ORB in break direction
    if break_dir == "long":
        outside = closes > orb_high
    else:
        outside = closes < orb_low

    # Consecutive run detection via cumsum of resets
    # Each time outside is False, cumsum increments creating a new group
    groups = (~outside).cumsum()
    # Running count within each True-group
    consecutive = np.zeros(len(outside), dtype=int)
    for i in range(len(outside)):
        if outside[i]:
            consecutive[i] = (consecutive[i - 1] if i > 0 else 0) + 1
        # else stays 0

    # Find first bar where consecutive >= confirm_bars
    mask = consecutive >= confirm_bars
    if mask.any():
        idx = np.argmax(mask)
        entry_price = orb_high if break_dir == "long" else orb_low
        stop_price = orb_low if break_dir == "long" else orb_high
        ts = candidate_bars.iloc[idx]["ts_utc"]
        return EntrySignal(
            triggered=True,
            entry_ts=ts.to_pydatetime(),
            entry_price=entry_price,
            stop_price=stop_price,
        )

    # Confirm not satisfied within window
    return EntrySignal(
        triggered=False, entry_ts=None, entry_price=None, stop_price=None
    )
