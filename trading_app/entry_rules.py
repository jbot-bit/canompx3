"""
Entry detection logic with confirm_bars support and multiple entry models.

Confirm bars require N consecutive 1m closes outside the ORB range
before confirming an entry signal. This filters fakeout breaks.

Entry Models:
  - E0 (Limit-On-Confirm): Enter at ORB level ON the confirm bar itself
  - E1 (Market-On-Confirm): Enter at next bar OPEN after confirm bar
  - E3 (Limit-At-ORB): Enter at ORB level if price retraces after confirm

Rules (CANONICAL_LOGIC.txt section 8):
  - confirm_bars=1: Enter on first close outside ORB (same as current pipeline)
  - confirm_bars=2: Wait for 2nd consecutive close outside ORB
  - confirm_bars=3+: Wait for Nth consecutive close outside ORB
  - Reset: If any bar closes back INSIDE ORB, count resets to 0

E0 vs E3 distinction:
  E0 fills on the CONFIRM bar (the Nth closing bar outside ORB).
  E3 waits for a RETRACE bar AFTER the confirm bar.
  For CB1, E0 essentially always fills (break bar must cross ORB edge).
  For CB2+, E0 may not fill if the confirm bar gapped fully past ORB edge.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from trading_app.config import E3_RETRACE_WINDOW_MINUTES


@dataclass(frozen=True)
class ConfirmResult:
    """Result of confirmation detection (separated from entry logic)."""

    confirmed: bool
    confirm_bar_idx: int | None  # Index into candidate_bars
    confirm_bar_ts: datetime | None
    confirm_bar_close: float | None
    orb_high: float
    orb_low: float
    break_dir: str


@dataclass(frozen=True)
class BreakTouchResult:
    """Result of break-touch detection (range crosses ORB, no close requirement).

    Used by E2 (stop-market) where a resting stop order triggers on any
    intra-bar touch of the ORB level, regardless of where the bar closes.
    """
    touched: bool
    touch_bar_ts: datetime | None
    touch_bar_idx: int | None
    orb_high: float
    orb_low: float
    break_dir: str


@dataclass(frozen=True)
class EntrySignal:
    """Result of entry detection with confirm_bars."""

    triggered: bool
    entry_ts: datetime | None
    entry_price: float | None
    stop_price: float | None
    entry_model: str | None = None
    confirm_bar_ts: datetime | None = None


def detect_confirm(
    bars_df: pd.DataFrame,
    orb_break_ts: datetime,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    confirm_bars: int,
    detection_window_end: datetime,
) -> ConfirmResult:
    """
    Detect N consecutive closes outside ORB range.

    Returns ConfirmResult with the confirm bar's index, timestamp, and close.
    Does NOT determine entry price — that depends on the entry model.
    """
    if confirm_bars < 1 or confirm_bars > 10:
        raise ValueError(f"confirm_bars must be 1-10, got {confirm_bars}")

    if break_dir not in ("long", "short"):
        raise ValueError(f"break_dir must be 'long' or 'short', got {break_dir}")

    no_confirm = ConfirmResult(
        confirmed=False, confirm_bar_idx=None, confirm_bar_ts=None,
        confirm_bar_close=None, orb_high=orb_high, orb_low=orb_low,
        break_dir=break_dir,
    )

    # Get bars from break_ts onwards within window
    candidate_bars = bars_df[
        (bars_df["ts_utc"] >= orb_break_ts)
        & (bars_df["ts_utc"] < detection_window_end)
    ].sort_values("ts_utc")

    if candidate_bars.empty:
        return no_confirm

    closes = candidate_bars["close"].values

    # Boolean mask: close outside ORB in break direction
    if break_dir == "long":
        outside = closes > orb_high
    else:
        outside = closes < orb_low

    # Consecutive run detection
    consecutive = np.zeros(len(outside), dtype=int)
    for i in range(len(outside)):
        if outside[i]:
            consecutive[i] = (consecutive[i - 1] if i > 0 else 0) + 1

    # Find first bar where consecutive >= confirm_bars
    mask = consecutive >= confirm_bars
    if mask.any():
        idx = int(np.argmax(mask))
        ts = candidate_bars.iloc[idx]["ts_utc"].to_pydatetime()
        close = float(candidate_bars.iloc[idx]["close"])
        return ConfirmResult(
            confirmed=True, confirm_bar_idx=idx, confirm_bar_ts=ts,
            confirm_bar_close=close, orb_high=orb_high, orb_low=orb_low,
            break_dir=break_dir,
        )

    return no_confirm


def detect_break_touch(
    bars_df: pd.DataFrame,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    detection_window_start: datetime,
    detection_window_end: datetime,
) -> BreakTouchResult:
    """
    Detect first bar whose range crosses the ORB level.

    Unlike detect_confirm(), this does NOT require the bar to close outside
    the ORB. A bar whose high > orb_high (long) or low < orb_low (short)
    counts as a touch, even if it closes back inside (fakeout).

    This is the detection path for E2 (stop-market), where a resting stop
    order triggers on any intra-bar touch of the level.
    """
    no_touch = BreakTouchResult(
        touched=False, touch_bar_ts=None, touch_bar_idx=None,
        orb_high=orb_high, orb_low=orb_low, break_dir=break_dir,
    )

    if break_dir not in ("long", "short"):
        raise ValueError(f"break_dir must be 'long' or 'short', got {break_dir}")

    candidate_bars = bars_df[
        (bars_df["ts_utc"] >= pd.Timestamp(detection_window_start))
        & (bars_df["ts_utc"] < pd.Timestamp(detection_window_end))
    ].sort_values("ts_utc")

    if candidate_bars.empty:
        return no_touch

    if break_dir == "long":
        touch_mask = candidate_bars["high"].values > orb_high
    else:
        touch_mask = candidate_bars["low"].values < orb_low

    if not touch_mask.any():
        return no_touch

    idx = int(np.argmax(touch_mask))
    ts = candidate_bars.iloc[idx]["ts_utc"].to_pydatetime()

    return BreakTouchResult(
        touched=True, touch_bar_ts=ts, touch_bar_idx=idx,
        orb_high=orb_high, orb_low=orb_low, break_dir=break_dir,
    )


def resolve_entry(
    bars_df: pd.DataFrame,
    confirm: ConfirmResult,
    entry_model: str,
    scan_window_end: datetime,
) -> EntrySignal:
    """
    Given a confirmed signal, resolve the entry price based on the entry model.

    Args:
        bars_df: Full bars DataFrame (same one passed to detect_confirm)
        confirm: ConfirmResult from detect_confirm()
        entry_model: "E1" or "E3"
        scan_window_end: End of trading day / scan window

    Returns:
        EntrySignal with model-specific entry_price and entry_ts.
    """
    if not confirm.confirmed:
        return EntrySignal(
            triggered=False, entry_ts=None, entry_price=None, stop_price=None,
            entry_model=entry_model, confirm_bar_ts=None,
        )

    stop_price = confirm.orb_low if confirm.break_dir == "long" else confirm.orb_high

    if entry_model == "E0":
        return _resolve_e0(bars_df, confirm, stop_price)
    elif entry_model == "E1":
        return _resolve_e1(bars_df, confirm, stop_price, scan_window_end)
    elif entry_model == "E3":
        # Cap retrace scan window if configured (stale fill prevention).
        # See research/research_e3_fill_timing.py for the audit backing this.
        effective_end = scan_window_end
        if E3_RETRACE_WINDOW_MINUTES is not None and confirm.confirm_bar_ts is not None:
            capped = confirm.confirm_bar_ts + timedelta(minutes=E3_RETRACE_WINDOW_MINUTES)
            effective_end = min(scan_window_end, capped)
        return _resolve_e3(bars_df, confirm, stop_price, effective_end)
    else:
        raise ValueError(f"Unknown entry_model: {entry_model}")


def _resolve_e1(
    bars_df: pd.DataFrame,
    confirm: ConfirmResult,
    stop_price: float,
    scan_window_end: datetime,
) -> EntrySignal:
    """E1: Market-On-Confirm. Entry = next bar OPEN after confirm bar."""
    # Find bars strictly after confirm bar timestamp
    next_bars = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(confirm.confirm_bar_ts))
        & (bars_df["ts_utc"] < pd.Timestamp(scan_window_end))
    ].sort_values("ts_utc")

    if next_bars.empty:
        return EntrySignal(
            triggered=False, entry_ts=None, entry_price=None, stop_price=None,
            entry_model="E1", confirm_bar_ts=confirm.confirm_bar_ts,
        )

    entry_bar = next_bars.iloc[0]
    entry_ts = entry_bar["ts_utc"].to_pydatetime()
    entry_price = float(entry_bar["open"])

    return EntrySignal(
        triggered=True, entry_ts=entry_ts, entry_price=entry_price,
        stop_price=stop_price, entry_model="E1",
        confirm_bar_ts=confirm.confirm_bar_ts,
    )


def _resolve_e0(
    bars_df: pd.DataFrame,
    confirm: ConfirmResult,
    stop_price: float,
) -> EntrySignal:
    """E0: Limit-On-Confirm. Entry at ORB edge ON the confirm bar itself.

    A limit order sits at the ORB edge before the break. It fills as soon as
    the confirm bar's range touches the ORB edge. For CB1, this essentially
    always fills (the break bar must have crossed the ORB edge). For CB2+,
    the confirm bar may have gapped fully past the ORB edge → no fill.
    """
    no_fill = EntrySignal(
        triggered=False, entry_ts=None, entry_price=None, stop_price=None,
        entry_model="E0", confirm_bar_ts=confirm.confirm_bar_ts,
    )

    confirm_bar = bars_df[bars_df["ts_utc"] == pd.Timestamp(confirm.confirm_bar_ts)]
    if confirm_bar.empty:
        return no_fill

    bar = confirm_bar.iloc[0]
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])

    if confirm.break_dir == "long":
        # Long: limit buy at orb_high — bar must have touched orb_high from below
        if bar_low > confirm.orb_high:
            return no_fill
        entry_price = confirm.orb_high
    else:
        # Short: limit sell at orb_low — bar must have touched orb_low from above
        if bar_high < confirm.orb_low:
            return no_fill
        entry_price = confirm.orb_low

    return EntrySignal(
        triggered=True,
        entry_ts=confirm.confirm_bar_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        entry_model="E0",
        confirm_bar_ts=confirm.confirm_bar_ts,
    )


def _resolve_e3(
    bars_df: pd.DataFrame,
    confirm: ConfirmResult,
    stop_price: float,
    scan_window_end: datetime,
) -> EntrySignal:
    """E3: Limit-At-ORB. Entry at ORB level if price retraces after confirm.

    IMPORTANT: Before filling the limit order, we verify that the stop level
    has not been breached on any bar between confirm and the retrace bar
    (inclusive). If the stop is breached before or on the retrace bar, the
    fill is invalid — you cannot enter a trade that is already stopped out.
    """
    no_fill = EntrySignal(
        triggered=False, entry_ts=None, entry_price=None, stop_price=None,
        entry_model="E3", confirm_bar_ts=confirm.confirm_bar_ts,
    )

    # Scan bars strictly after confirm bar
    post_confirm = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(confirm.confirm_bar_ts))
        & (bars_df["ts_utc"] < pd.Timestamp(scan_window_end))
    ].sort_values("ts_utc")

    if post_confirm.empty:
        return no_fill

    highs = post_confirm["high"].values
    lows = post_confirm["low"].values

    # Check for retrace to ORB level AND stop breach
    if confirm.break_dir == "long":
        # Long: limit buy at orb_high, stop at orb_low
        retrace_mask = lows <= confirm.orb_high
        stop_hit_mask = lows <= stop_price
        entry_price = confirm.orb_high
    else:
        # Short: limit sell at orb_low, stop at orb_high
        retrace_mask = highs >= confirm.orb_low
        stop_hit_mask = highs >= stop_price
        entry_price = confirm.orb_low

    if not retrace_mask.any():
        return no_fill

    retrace_idx = int(np.argmax(retrace_mask))

    # CRITICAL: Check if stop was breached on or before the retrace bar.
    # If ANY bar from confirm to retrace (inclusive) hit the stop, the
    # limit fill is invalid — the trade was dead before it could fill.
    if stop_hit_mask[:retrace_idx + 1].any():
        return no_fill

    entry_ts = post_confirm.iloc[retrace_idx]["ts_utc"].to_pydatetime()

    return EntrySignal(
        triggered=True, entry_ts=entry_ts, entry_price=entry_price,
        stop_price=stop_price, entry_model="E3",
        confirm_bar_ts=confirm.confirm_bar_ts,
    )


def _resolve_e2(
    touch: BreakTouchResult,
    slippage_ticks: int,
    tick_size: float,
) -> EntrySignal:
    """E2: Stop-Market. Entry at ORB level + N ticks slippage.

    A stop order sits at the ORB boundary before the break. It fills
    the moment the bar's range crosses the level. Fill price includes
    slippage (fill-through, not fill-on-touch).

    Fakeout bars (close back inside ORB) ARE valid fills — the stop
    triggered intra-bar regardless of where the bar closes.
    """
    if not touch.touched:
        return EntrySignal(
            triggered=False, entry_ts=None, entry_price=None,
            stop_price=None, entry_model="E2", confirm_bar_ts=None,
        )

    slippage = slippage_ticks * tick_size

    if touch.break_dir == "long":
        entry_price = touch.orb_high + slippage
        stop_price = touch.orb_low
    else:
        entry_price = touch.orb_low - slippage
        stop_price = touch.orb_high

    return EntrySignal(
        triggered=True,
        entry_ts=touch.touch_bar_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        entry_model="E2",
        confirm_bar_ts=touch.touch_bar_ts,
    )


def detect_entry_with_confirm_bars(
    bars_df: pd.DataFrame,
    orb_break_ts: datetime,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    confirm_bars: int,
    detection_window_end: datetime,
    entry_model: str = "E1",
) -> EntrySignal:
    """
    Detect entry after N consecutive closes outside ORB.

    Wrapper that calls detect_confirm() + resolve_entry() for E1/E3.
    """
    # E0 + CB>1 is look-ahead: by the time CB2+ confirms, the confirm bar has
    # already closed — you can't retroactively fill a limit on it.
    if entry_model == "E0" and confirm_bars > 1:
        return EntrySignal(
            triggered=False, entry_ts=None, entry_price=None,
            stop_price=None, entry_model="E0", confirm_bar_ts=None,
        )

    confirm = detect_confirm(
        bars_df, orb_break_ts, orb_high, orb_low,
        break_dir, confirm_bars, detection_window_end,
    )
    return resolve_entry(bars_df, confirm, entry_model, detection_window_end)
