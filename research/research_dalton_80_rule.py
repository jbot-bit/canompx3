#!/usr/bin/env python3
"""
Research: Dalton-style 80% Value Area Rule (read-only)

Rule idea (common Market Profile framing):
- If market opens outside prior day's Value Area (VA)
- And re-enters VA and is accepted (often phrased as 2 consecutive 30m periods)
- Then price has high probability of traversing to the opposite side of VA.

This script tests probability-style outcomes (not PnL-first):
- Success = opposite VA boundary hit first
- Failure = same-side VA boundary reclaimed first
- Ambiguous (both in same bar) = failure (conservative)

Outputs:
- research/output/dalton_80_rule_summary.csv
- research/output/dalton_80_rule_yearly.csv
- research/output/dalton_80_rule_notes.md
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

from pipeline.paths import GOLD_DB_PATH
from pipeline.build_daily_features import compute_trading_day_utc_range, _orb_utc_window
from research.archive.analyze_value_area import compute_volume_profile


@dataclass
class SetupResult:
    symbol: str
    trading_day: pd.Timestamp
    variant: str
    open_side: str
    success: int
    resolved: int


def _slice_day(bars: pd.DataFrame, ts_all, trading_day):
    td_start, td_end = compute_trading_day_utc_range(trading_day)
    i0 = int(np.searchsorted(ts_all, pd.Timestamp(td_start).asm8, side="left"))
    i1 = int(np.searchsorted(ts_all, pd.Timestamp(td_end).asm8, side="left"))
    return bars.iloc[i0:i1]


def _bar_overlap_va(bar: pd.Series, val: float, vah: float) -> bool:
    return (float(bar["high"]) >= val) and (float(bar["low"]) <= vah)


def _bar_close_inside_va(bar: pd.Series, val: float, vah: float) -> bool:
    c = float(bar["close"])
    return val <= c <= vah


def _evaluate_path(after_entry: pd.DataFrame, side: str, val: float, vah: float) -> tuple[int, int]:
    """Return (success, resolved)."""
    if after_entry.empty:
        return 0, 0

    for _, b in after_entry.iterrows():
        hi = float(b["high"])
        lo = float(b["low"])

        if side == "above":
            # short bias from VAH toward VAL
            hit_target = lo <= val
            # failure only if price re-accepts ABOVE VAH (strictly outside)
            hit_fail = hi > vah
        else:
            # long bias from VAL toward VAH
            hit_target = hi >= vah
            # failure only if price re-accepts BELOW VAL (strictly outside)
            hit_fail = lo < val

        if hit_target and hit_fail:
            return 0, 1  # conservative fail
        if hit_target:
            return 1, 1
        if hit_fail:
            return 0, 1

    return 0, 0


def run_for_symbol(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    session_label: str,
    bin_size: float,
) -> list[SetupResult]:
    tdays = con.execute(
        """
        SELECT DISTINCT trading_day
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
        ORDER BY trading_day
        """,
        [symbol],
    ).fetchall()

    trading_days = [r[0] for r in tdays]
    if len(trading_days) < 3:
        return []

    # Load full bars range once
    global_start, _ = compute_trading_day_utc_range(trading_days[0])
    _, global_end = compute_trading_day_utc_range(trading_days[-1])
    bars = con.execute(
        """
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?::TIMESTAMPTZ
          AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc
        """,
        [symbol, global_start.isoformat(), global_end.isoformat()],
    ).fetchdf()

    if bars.empty:
        return []

    bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True)
    ts_all = bars["ts_utc"].values

    results: list[SetupResult] = []

    for i in range(1, len(trading_days)):
        prev_td = trading_days[i - 1]
        td = trading_days[i]

        prev_bars = _slice_day(bars, ts_all, prev_td)
        day_bars = _slice_day(bars, ts_all, td)
        if prev_bars.empty or day_bars.empty:
            continue

        prof = compute_volume_profile(prev_bars, bin_size=bin_size)
        if prof is None:
            continue

        val = float(prof["val"])
        vah = float(prof["vah"])

        session_start, _ = _orb_utc_window(td, session_label, 1)
        first_min = day_bars[(day_bars["ts_utc"] >= session_start) & (day_bars["ts_utc"] < session_start + pd.Timedelta(minutes=1))]
        if first_min.empty:
            continue

        open_px = float(first_min.iloc[0]["open"])
        if val <= open_px <= vah:
            continue  # opens inside VA; not this rule trigger

        side = "above" if open_px > vah else "below"

        a_end = session_start + pd.Timedelta(minutes=30)
        b_end = session_start + pd.Timedelta(minutes=60)

        bars_a = day_bars[(day_bars["ts_utc"] >= session_start) & (day_bars["ts_utc"] < a_end)]
        bars_b = day_bars[(day_bars["ts_utc"] >= a_end) & (day_bars["ts_utc"] < b_end)]

        # Variant 1: A and B both overlap/touch VA
        v1_ok = (not bars_a.empty and not bars_b.empty and
                 any(_bar_overlap_va(r, val, vah) for _, r in bars_a.iterrows()) and
                 any(_bar_overlap_va(r, val, vah) for _, r in bars_b.iterrows()))
        if v1_ok:
            after = day_bars[day_bars["ts_utc"] >= b_end]
            s, r = _evaluate_path(after, side, val, vah)
            results.append(SetupResult(symbol, pd.Timestamp(td), "touch_A_B", side, s, r))

        # Variant 2: A and B closes both inside VA (stricter)
        v2_ok = (not bars_a.empty and not bars_b.empty and
                 _bar_close_inside_va(bars_a.iloc[-1], val, vah) and
                 _bar_close_inside_va(bars_b.iloc[-1], val, vah))
        if v2_ok:
            after = day_bars[day_bars["ts_utc"] >= b_end]
            s, r = _evaluate_path(after, side, val, vah)
            results.append(SetupResult(symbol, pd.Timestamp(td), "close_A_B", side, s, r))

        # Variant 3: first re-entry touch within first hour; enter immediately
        first_hour = day_bars[(day_bars["ts_utc"] >= session_start) & (day_bars["ts_utc"] < b_end)]
        touch = first_hour[(first_hour["high"] >= val) & (first_hour["low"] <= vah)]
        if not touch.empty:
            entry_ts = pd.Timestamp(touch.iloc[0]["ts_utc"])
            after = day_bars[day_bars["ts_utc"] > entry_ts]
            s, r = _evaluate_path(after, side, val, vah)
            results.append(SetupResult(symbol, pd.Timestamp(td), "first_touch_1h", side, s, r))

    return results


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sdf = (
        df.groupby(["symbol", "variant", "open_side"], as_index=False)
        .agg(setups=("resolved", "count"), resolved=("resolved", "sum"), wins=("success", "sum"))
    )
    sdf["hit_rate"] = np.where(sdf["resolved"] > 0, sdf["wins"] / sdf["resolved"], np.nan)

    ydf = df.copy()
    ydf["year"] = pd.to_datetime(ydf["trading_day"]).dt.year
    ydf = (
        ydf.groupby(["symbol", "variant", "open_side", "year"], as_index=False)
        .agg(setups=("resolved", "count"), resolved=("resolved", "sum"), wins=("success", "sum"))
    )
    ydf["hit_rate"] = np.where(ydf["resolved"] > 0, ydf["wins"] / ydf["resolved"], np.nan)

    return sdf, ydf


def main() -> int:
    p = argparse.ArgumentParser(description="Research Dalton-style 80% rule")
    p.add_argument("--db-path", type=str, default=None)
    p.add_argument("--symbols", nargs="+", default=["MGC", "MES", "MNQ"])
    p.add_argument("--session-label", default="0900", help="Session anchor for A/B 30m brackets")
    p.add_argument("--bin-size", type=float, default=0.5, help="Volume profile bin size")
    args = p.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    rows: list[SetupResult] = []
    for sym in args.symbols:
        rows.extend(run_for_symbol(con, sym, args.session_label, args.bin_size))
    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No setups found.")
        return 0

    df = pd.DataFrame([r.__dict__ for r in rows])
    sdf, ydf = summarize(df)

    summary_path = out_dir / "dalton_80_rule_summary.csv"
    yearly_path = out_dir / "dalton_80_rule_yearly.csv"
    notes_path = out_dir / "dalton_80_rule_notes.md"

    sdf.to_csv(summary_path, index=False)
    ydf.to_csv(yearly_path, index=False)

    lines = []
    lines.append("# Dalton 80% Rule Test Notes")
    lines.append("")
    lines.append(f"- Session anchor: {args.session_label}")
    lines.append(f"- Symbols: {', '.join(args.symbols)}")
    lines.append(f"- Profile bin size: {args.bin_size}")
    lines.append("")
    lines.append("## Variants")
    lines.append("- touch_A_B: both first two 30m brackets overlap prior VA")
    lines.append("- close_A_B: both first two 30m bracket closes are inside prior VA")
    lines.append("- first_touch_1h: first touch/re-entry within first hour")
    lines.append("")
    lines.append("## Rule outcome")
    lines.append("Success = opposite VA boundary hit before same-side boundary reclaim.")
    lines.append("Ambiguous same-bar hit = failure (conservative).")
    lines.append("")
    lines.append("## Summary")
    for r in sdf.itertuples(index=False):
        lines.append(
            f"- {r.symbol} {r.variant} {r.open_side}: setups={r.setups}, resolved={r.resolved}, "
            f"wins={r.wins}, hit_rate={r.hit_rate:.1%}"
        )

    notes_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {yearly_path}")
    print(f"Saved: {notes_path}")
    print("\nTop summary:")
    print(sdf.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
