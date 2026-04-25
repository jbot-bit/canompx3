"""
Entry detection logic with confirm_bars support and multiple entry models.

Confirm bars require N consecutive 1m closes outside the ORB range
before confirming an entry signal. This filters fakeout breaks.

Entry Models:
  - E1 (Market-On-Confirm): Enter at next bar OPEN after confirm bar
  - E2 (Stop-Market): Stop order at ORB level + slippage. Triggers on first
    bar whose range crosses the ORB level. Uses detect_break_touch(), not this
    confirm-based path. Fakeouts included as trades.
  - E3 (Limit-At-ORB): Enter at ORB level if price retraces after confirm

Rules (CANONICAL_LOGIC.txt section 8):
  - confirm_bars=1: Enter on first close outside ORB (same as current pipeline)
  - confirm_bars=2: Wait for 2nd consecutive close outside ORB
  - confirm_bars=3+: Wait for Nth consecutive close outside ORB
  - Reset: If any bar closes back INSIDE ORB, count resets to 0

E2 vs E1/E3 distinction:
  E2 uses detect_break_touch() + _resolve_e2() — completely separate path.
  E1/E3 use detect_confirm() + resolve_entry() — the confirm-based path.
  E0 (limit-on-confirm) was purged Feb 2026 — 3 structural biases made it an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

# numpy / pandas lazy-loaded inside the functions that use them (PEP 8).
# Dataclass fields below use only builtins + datetime, so PEP 563 string
# annotations are safe. pd.DataFrame annotations on function signatures
# resolve via TYPE_CHECKING for static checkers.
if TYPE_CHECKING:
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
    import numpy as np

    if confirm_bars < 1 or confirm_bars > 10:
        raise ValueError(f"confirm_bars must be 1-10, got {confirm_bars}")

    if break_dir not in ("long", "short"):
        raise ValueError(f"break_dir must be 'long' or 'short', got {break_dir}")

    no_confirm = ConfirmResult(
        confirmed=False,
        confirm_bar_idx=None,
        confirm_bar_ts=None,
        confirm_bar_close=None,
        orb_high=orb_high,
        orb_low=orb_low,
        break_dir=break_dir,
    )

    # Get bars from break_ts onwards within window
    candidate_bars = bars_df[
        (bars_df["ts_utc"] >= orb_break_ts) & (bars_df["ts_utc"] < detection_window_end)
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
            confirmed=True,
            confirm_bar_idx=idx,
            confirm_bar_ts=ts,
            confirm_bar_close=close,
            orb_high=orb_high,
            orb_low=orb_low,
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
    import numpy as np
    import pandas as pd

    no_touch = BreakTouchResult(
        touched=False,
        touch_bar_ts=None,
        touch_bar_idx=None,
        orb_high=orb_high,
        orb_low=orb_low,
        break_dir=break_dir,
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
        touched=True,
        touch_bar_ts=ts,
        touch_bar_idx=idx,
        orb_high=orb_high,
        orb_low=orb_low,
        break_dir=break_dir,
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
            triggered=False,
            entry_ts=None,
            entry_price=None,
            stop_price=None,
            entry_model=entry_model,
            confirm_bar_ts=None,
        )

    stop_price = confirm.orb_low if confirm.break_dir == "long" else confirm.orb_high

    if entry_model == "E1":
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
    import pandas as pd

    # Find bars strictly after confirm bar timestamp
    next_bars = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(confirm.confirm_bar_ts)) & (bars_df["ts_utc"] < pd.Timestamp(scan_window_end))
    ].sort_values("ts_utc")

    if next_bars.empty:
        return EntrySignal(
            triggered=False,
            entry_ts=None,
            entry_price=None,
            stop_price=None,
            entry_model="E1",
            confirm_bar_ts=confirm.confirm_bar_ts,
        )

    entry_bar = next_bars.iloc[0]
    entry_ts = entry_bar["ts_utc"].to_pydatetime()
    entry_price = float(entry_bar["open"])

    return EntrySignal(
        triggered=True,
        entry_ts=entry_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        entry_model="E1",
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
    import numpy as np
    import pandas as pd

    no_fill = EntrySignal(
        triggered=False,
        entry_ts=None,
        entry_price=None,
        stop_price=None,
        entry_model="E3",
        confirm_bar_ts=confirm.confirm_bar_ts,
    )

    # Scan bars strictly after confirm bar
    post_confirm = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(confirm.confirm_bar_ts)) & (bars_df["ts_utc"] < pd.Timestamp(scan_window_end))
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
    if stop_hit_mask[: retrace_idx + 1].any():
        return no_fill

    entry_ts = post_confirm.iloc[retrace_idx]["ts_utc"].to_pydatetime()

    return EntrySignal(
        triggered=True,
        entry_ts=entry_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        entry_model="E3",
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
            triggered=False,
            entry_ts=None,
            entry_price=None,
            stop_price=None,
            entry_model="E2",
            confirm_bar_ts=None,
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
    # E2 uses detect_break_touch() + _resolve_e2(), not the confirm-based path.
    if entry_model == "E2":
        raise ValueError("E2 uses detect_break_touch(), not detect_entry_with_confirm_bars().")

    confirm = detect_confirm(
        bars_df,
        orb_break_ts,
        orb_high,
        orb_low,
        break_dir,
        confirm_bars,
        detection_window_end,
    )
    return resolve_entry(bars_df, confirm, entry_model, detection_window_end)


# =========================================================================
# 5-minute resampling helpers
# =========================================================================
#
# Extracted from trading_app/nested/builder.py during the E2 canonical-window
# refactor (2026-04-07, Stage 4). nested/builder.py is slated for deletion in
# Stage 7 (536 lines of dead code targeting a never-created nested_outcomes
# table), but these two helpers are still used by
# trading_app/nested/audit_outcomes.py. Extracting them here so they survive
# the deletion and live in a canonical home alongside the other entry-rule
# primitives (detect_confirm, resolve_entry, detect_break_touch).
#
# Logic is verbatim from nested/builder.py — no semantic changes.


def resample_to_5m(bars_1m_df: pd.DataFrame, after_ts: datetime) -> pd.DataFrame:
    """Resample post-ORB 1m bars to 5m OHLCV.

    Takes 1m bars with ts_utc > after_ts, groups into 5-minute buckets
    (floor to nearest 5-minute boundary), and aggregates OHLCV.

    Args:
        bars_1m_df: DataFrame with columns [ts_utc, open, high, low, close, volume]
        after_ts: Only include bars strictly after this timestamp

    Returns:
        DataFrame with same columns, timestamps floored to 5m boundaries.
        Empty DataFrame if no bars after after_ts.
    """
    import pandas as pd

    post_orb = bars_1m_df[bars_1m_df["ts_utc"] > pd.Timestamp(after_ts)].copy()

    if post_orb.empty:
        return pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])

    # Floor to 5-minute boundaries (robust for any datetime64 resolution)
    post_orb["bucket"] = post_orb["ts_utc"].dt.floor("5min")

    # Aggregate per bucket
    grouped = post_orb.groupby("bucket", sort=True)
    bars_5m = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    bars_5m = bars_5m.rename(columns={"bucket": "ts_utc"})
    return bars_5m


def _verify_e3_sub_bar_fill(
    bars_1m_df: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    break_dir: str,
) -> bool:
    """Verify that 1m bars within the 5m entry candle actually touched the limit price.

    For E3 (limit-at-ORB), the 5m candle may show the price touched the ORB level,
    but the underlying 1m data might not actually reach it. This post-processing
    check ensures fill accuracy.

    Args:
        bars_1m_df: Original 1m bars
        entry_ts: The 5m bar timestamp where E3 fill was detected
        entry_price: The limit order price (ORB level)
        break_dir: "long" or "short"

    Returns:
        True if 1m data confirms the fill, False otherwise.
    """
    import pandas as pd

    # Find 1m bars within the 5m candle: [entry_ts, entry_ts + 5min)
    bucket_start = pd.Timestamp(entry_ts)
    bucket_end = bucket_start + pd.Timedelta(minutes=5)

    bars_in_candle = bars_1m_df[(bars_1m_df["ts_utc"] >= bucket_start) & (bars_1m_df["ts_utc"] < bucket_end)]

    if bars_in_candle.empty:
        return False

    if break_dir == "long":
        # Long E3: limit buy at orb_high. Need low <= entry_price
        return bool((bars_in_candle["low"].values <= entry_price).any())
    else:
        # Short E3: limit sell at orb_low. Need high >= entry_price
        return bool((bars_in_candle["high"].values >= entry_price).any())
