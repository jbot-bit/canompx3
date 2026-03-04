#!/usr/bin/env python3
"""
Research: Dalton acceptance setup monetization (mid vs far vs split)

Read-only research script. No pipeline/live logic changes.

Trigger (default): close_A_B
- Open outside prior-day Value Area
- A and B 30m bracket closes are inside prior-day VA
- Enter at next 1m bar OPEN after B close

Targets compared:
- MID: VA midpoint
- FAR: opposite VA boundary
- SPLIT70: 70% at MID + 30% runner to FAR (same stop)
- SPLIT70_BE: same, but runner stop moves to breakeven after MID fills

Stop:
- If open above VA (short): stop = VAH + stop_buffer * VA_width
- If open below VA (long):  stop = VAL - stop_buffer * VA_width

Outputs:
- research/output/dalton_mid_monetization_summary.csv
- research/output/dalton_mid_monetization_yearly.csv
- research/output/dalton_mid_monetization_notes.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range, _orb_utc_window
from pipeline.cost_model import get_cost_spec, to_r_multiple
from research.archive.analyze_value_area import compute_volume_profile


@dataclass
class TradeResult:
    symbol: str
    anchor: str
    variant: str
    target_mode: str
    trading_day: pd.Timestamp
    open_side: str
    pnl_r: float


def _slice_day(bars: pd.DataFrame, ts_all, trading_day):
    s, e = compute_trading_day_utc_range(trading_day)
    i0 = int(np.searchsorted(ts_all, pd.Timestamp(s).asm8, side="left"))
    i1 = int(np.searchsorted(ts_all, pd.Timestamp(e).asm8, side="left"))
    return bars.iloc[i0:i1]


def _close_in_va(bar: pd.Series, val: float, vah: float) -> bool:
    c = float(bar["close"])
    return val <= c <= vah


def _to_points(direction: str, entry: float, exit_px: float) -> float:
    return float(exit_px - entry) if direction == "long" else float(entry - exit_px)


def _simulate_single_target(
    bars: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target: float,
) -> float:
    """Single-leg strategy: MID or FAR target."""
    if bars.empty:
        return _to_points(direction, entry, entry)

    for _, b in bars.iterrows():
        hi = float(b["high"])
        lo = float(b["low"])

        if direction == "long":
            hit_t = hi >= target
            hit_s = lo <= stop
        else:
            hit_t = lo <= target
            hit_s = hi >= stop

        if hit_t and hit_s:
            # Conservative ambiguity handling
            return _to_points(direction, entry, stop)
        if hit_t:
            return _to_points(direction, entry, target)
        if hit_s:
            return _to_points(direction, entry, stop)

    # End-of-day mark-to-close
    close_px = float(bars.iloc[-1]["close"])
    return _to_points(direction, entry, close_px)


def _simulate_split(
    bars: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    mid: float,
    far: float,
    be_after_mid: bool,
) -> float:
    """70% at MID + 30% runner to FAR."""
    w1, w2 = 0.7, 0.3
    pnl1 = None
    pnl2 = None
    stop2 = stop

    if bars.empty:
        return 0.0

    for _, b in bars.iterrows():
        hi = float(b["high"])
        lo = float(b["low"])

        if direction == "long":
            hit_mid = hi >= mid
            hit_far = hi >= far
            hit_stop1 = lo <= stop if pnl1 is None else False
            hit_stop2 = lo <= stop2 if pnl2 is None else False
        else:
            hit_mid = lo <= mid
            hit_far = lo <= far
            hit_stop1 = hi >= stop if pnl1 is None else False
            hit_stop2 = hi >= stop2 if pnl2 is None else False

        # Leg1 handling (to MID)
        if pnl1 is None:
            if hit_mid and hit_stop1:
                # Ambiguous: conservative full stop on both open legs
                loss = _to_points(direction, entry, stop)
                pnl1 = loss
                if pnl2 is None:
                    pnl2 = loss
                break
            elif hit_stop1:
                loss = _to_points(direction, entry, stop)
                pnl1 = loss
                if pnl2 is None:
                    pnl2 = loss
                break
            elif hit_mid:
                pnl1 = _to_points(direction, entry, mid)
                if be_after_mid:
                    stop2 = entry
                # If far also hit same bar, allow runner to fill too (no stop ambiguity above)
                if pnl2 is None and hit_far:
                    pnl2 = _to_points(direction, entry, far)
                    break

        # Leg2 handling (runner)
        if pnl2 is None:
            # Recompute stop/far ambiguity for runner state
            if direction == "long":
                hit_far_now = hi >= far
                hit_stop_now = lo <= stop2
            else:
                hit_far_now = lo <= far
                hit_stop_now = hi >= stop2

            if hit_far_now and hit_stop_now:
                # Conservative: stop first
                pnl2 = _to_points(direction, entry, stop2)
                break
            elif hit_far_now:
                pnl2 = _to_points(direction, entry, far)
                break
            elif hit_stop_now:
                pnl2 = _to_points(direction, entry, stop2)
                break

    if pnl1 is None:
        close_px = float(bars.iloc[-1]["close"])
        pnl1 = _to_points(direction, entry, close_px)
    if pnl2 is None:
        close_px = float(bars.iloc[-1]["close"])
        pnl2 = _to_points(direction, entry, close_px)

    return w1 * pnl1 + w2 * pnl2


def _strategy_points(
    mode: str,
    bars_after_entry: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    val: float,
    vah: float,
) -> float:
    mid = 0.5 * (val + vah)
    far = val if direction == "short" else vah

    if mode == "MID":
        return _simulate_single_target(bars_after_entry, direction, entry, stop, mid)
    if mode == "FAR":
        return _simulate_single_target(bars_after_entry, direction, entry, stop, far)
    if mode == "SPLIT70":
        return _simulate_split(bars_after_entry, direction, entry, stop, mid, far, be_after_mid=False)
    if mode == "SPLIT70_BE":
        return _simulate_split(bars_after_entry, direction, entry, stop, mid, far, be_after_mid=True)

    raise ValueError(f"Unknown mode: {mode}")


def run_symbol(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    anchors: list[str],
    bin_size: float,
    stop_buffer: float,
    variant: str,
) -> list[TradeResult]:
    tdays = [r[0] for r in con.execute(
        "SELECT DISTINCT trading_day FROM daily_features WHERE symbol=? AND orb_minutes=5 ORDER BY trading_day",
        [symbol],
    ).fetchall()]
    if len(tdays) < 3:
        return []

    gs, _ = compute_trading_day_utc_range(tdays[0])
    _, ge = compute_trading_day_utc_range(tdays[-1])
    bars = con.execute(
        """
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol=?
          AND ts_utc>=?::TIMESTAMPTZ
          AND ts_utc<?::TIMESTAMPTZ
        ORDER BY ts_utc
        """,
        [symbol, gs.isoformat(), ge.isoformat()],
    ).fetchdf()
    if bars.empty:
        return []

    bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True)
    ts_all = bars["ts_utc"].values

    spec = get_cost_spec(symbol)
    out: list[TradeResult] = []

    for anchor in anchors:
        for i in range(1, len(tdays)):
            prev_td = tdays[i - 1]
            td = tdays[i]

            prev_bars = _slice_day(bars, ts_all, prev_td)
            day_bars = _slice_day(bars, ts_all, td)
            if prev_bars.empty or day_bars.empty:
                continue

            prof = compute_volume_profile(prev_bars, bin_size=bin_size)
            if prof is None:
                continue

            val = float(prof["val"])
            vah = float(prof["vah"])
            va_width = max(vah - val, 1e-9)

            start, _ = _orb_utc_window(td, anchor, 1)
            first = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < start + pd.Timedelta(minutes=1))]
            if first.empty:
                continue

            open_px = float(first.iloc[0]["open"])
            if val <= open_px <= vah:
                continue

            side = "above" if open_px > vah else "below"
            direction = "short" if side == "above" else "long"

            a_end = start + pd.Timedelta(minutes=30)
            b_end = start + pd.Timedelta(minutes=60)
            bars_a = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < a_end)]
            bars_b = day_bars[(day_bars["ts_utc"] >= a_end) & (day_bars["ts_utc"] < b_end)]

            trigger_ok = False
            trigger_ts = pd.Timestamp(b_end)

            if variant == "close_A_B":
                trigger_ok = (
                    not bars_a.empty and not bars_b.empty
                    and _close_in_va(bars_a.iloc[-1], val, vah)
                    and _close_in_va(bars_b.iloc[-1], val, vah)
                )
            else:
                raise ValueError(f"Unsupported variant: {variant}")

            if not trigger_ok:
                continue

            # Entry at next 1m bar open after trigger_ts (avoid lookahead)
            post = day_bars[day_bars["ts_utc"] > trigger_ts]
            if post.empty:
                continue

            entry_bar = post.iloc[0]
            entry_px = float(entry_bar["open"])
            entry_ts = pd.Timestamp(entry_bar["ts_utc"])

            if direction == "short":
                stop_px = vah + stop_buffer * va_width
            else:
                stop_px = val - stop_buffer * va_width

            risk_points = abs(entry_px - stop_px)
            if risk_points < spec.min_risk_floor_points:
                continue

            bars_after_entry = day_bars[day_bars["ts_utc"] >= entry_ts]
            for mode in ("MID", "FAR", "SPLIT70", "SPLIT70_BE"):
                pnl_points = _strategy_points(
                    mode, bars_after_entry, direction, entry_px, stop_px, val, vah
                )
                pnl_r = float(to_r_multiple(spec, entry=entry_px, stop=stop_px, pnl_points=pnl_points))
                out.append(
                    TradeResult(
                        symbol=symbol,
                        anchor=anchor,
                        variant=variant,
                        target_mode=mode,
                        trading_day=pd.Timestamp(td),
                        open_side=side,
                        pnl_r=pnl_r,
                    )
                )

    return out


def _max_dd(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    c = np.cumsum(np.array(pnls, dtype=float))
    peak = np.maximum.accumulate(c)
    dd = peak - c
    return float(np.max(dd))


def summarize(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for (symbol, anchor, mode), g in df.groupby(["symbol", "anchor", "target_mode"]):
        pnls = g["pnl_r"].tolist()
        rows.append(
            {
                "symbol": symbol,
                "anchor": anchor,
                "target_mode": mode,
                "n": len(pnls),
                "wr": float(np.mean(np.array(pnls) > 0)),
                "avg_r": float(np.mean(pnls)),
                "total_r": float(np.sum(pnls)),
                "max_dd_r": _max_dd(pnls),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["anchor", "symbol", "target_mode"])

    y = df.copy()
    y["year"] = pd.to_datetime(y["trading_day"]).dt.year
    yearly = (
        y.groupby(["symbol", "anchor", "target_mode", "year"], as_index=False)
        .agg(n=("pnl_r", "count"), avg_r=("pnl_r", "mean"), total_r=("pnl_r", "sum"), wr=("pnl_r", lambda s: (s > 0).mean()))
        .sort_values(["anchor", "symbol", "target_mode", "year"])
    )

    return summary, yearly


def main() -> int:
    p = argparse.ArgumentParser(description="Dalton setup monetization: mid vs far vs split")
    p.add_argument("--db-path", default="gold.db")
    p.add_argument("--symbols", nargs="+", default=["MGC", "MES", "MNQ"])
    p.add_argument("--anchors", nargs="+", default=["0900", "1000", "1100"])
    p.add_argument("--variant", default="close_A_B", choices=["close_A_B"])
    p.add_argument("--bin-size", type=float, default=0.5)
    p.add_argument("--stop-buffer", type=float, default=0.10, help="Stop offset as fraction of VA width")
    args = p.parse_args()

    con = duckdb.connect(args.db_path, read_only=True)
    all_rows: list[TradeResult] = []

    for sym in args.symbols:
        all_rows.extend(
            run_symbol(
                con=con,
                symbol=sym,
                anchors=args.anchors,
                bin_size=args.bin_size,
                stop_buffer=args.stop_buffer,
                variant=args.variant,
            )
        )
    con.close()

    if not all_rows:
        print("No trades generated.")
        return 0

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    summary, yearly = summarize(df)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_sum = out_dir / "dalton_mid_monetization_summary.csv"
    p_year = out_dir / "dalton_mid_monetization_yearly.csv"
    p_notes = out_dir / "dalton_mid_monetization_notes.md"

    summary.to_csv(p_sum, index=False)
    yearly.to_csv(p_year, index=False)

    lines = []
    lines.append("# Dalton Monetization: Mid vs Far vs Split")
    lines.append("")
    lines.append(f"- Variant: {args.variant}")
    lines.append(f"- Anchors: {', '.join(args.anchors)}")
    lines.append(f"- Symbols: {', '.join(args.symbols)}")
    lines.append(f"- Bin size: {args.bin_size}")
    lines.append(f"- Stop buffer: {args.stop_buffer} x VA width")
    lines.append("")
    lines.append("## Summary")
    for r in summary.itertuples(index=False):
        lines.append(
            f"- {r.symbol} {r.anchor} {r.target_mode}: N={r.n}, WR={r.wr:.1%}, "
            f"avgR={r.avg_r:+.4f}, totalR={r.total_r:+.1f}, maxDD={r.max_dd_r:.2f}"
        )

    p_notes.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_year}")
    print(f"Saved: {p_notes}")
    print("\nTop summary:")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
