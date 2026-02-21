#!/usr/bin/env python3
"""Deep-dive scan for Dalton 80% rule variants.

Compares two outcome definitions:
- strict_first_pass: opposite VA side hit before rejection beyond entry side.
- loose_eventual_touch: opposite VA side touched at any point after trigger.

Scans session anchors 0900/1000/1100 and trigger variants.

Outputs:
- research/output/dalton_80_deepdive_summary.csv
- research/output/dalton_80_deepdive_notes.md
"""

from __future__ import annotations

from pathlib import Path
import sys
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range, _orb_utc_window
from research.archive.analyze_value_area import compute_volume_profile

DB_PATH = "gold.db"
SYMBOLS = ["MGC", "MES", "MNQ"]
ANCHORS = ["0900", "1000", "1100"]


def _slice_day(bars: pd.DataFrame, ts_all, trading_day):
    s, e = compute_trading_day_utc_range(trading_day)
    i0 = int(np.searchsorted(ts_all, pd.Timestamp(s).asm8, side="left"))
    i1 = int(np.searchsorted(ts_all, pd.Timestamp(e).asm8, side="left"))
    return bars.iloc[i0:i1]


def _overlap_va(bar: pd.Series, val: float, vah: float) -> bool:
    return float(bar["high"]) >= val and float(bar["low"]) <= vah


def _close_in_va(bar: pd.Series, val: float, vah: float) -> bool:
    c = float(bar["close"])
    return val <= c <= vah


def _evaluate(after: pd.DataFrame, side: str, val: float, vah: float):
    if after.empty:
        return 0, 0, 0, 0

    # loose outcome
    if side == "above":
        loose_hit = (after["low"] <= val).any()
    else:
        loose_hit = (after["high"] >= vah).any()
    loose_win = 1 if loose_hit else 0
    loose_resolved = 1

    # strict first-pass outcome
    strict_win = 0
    strict_resolved = 0
    for _, b in after.iterrows():
        hi = float(b["high"])
        lo = float(b["low"])
        if side == "above":
            hit_target = lo <= val
            hit_fail = hi > vah
        else:
            hit_target = hi >= vah
            hit_fail = lo < val

        if hit_target and hit_fail:
            strict_resolved = 1
            strict_win = 0
            break
        if hit_target:
            strict_resolved = 1
            strict_win = 1
            break
        if hit_fail:
            strict_resolved = 1
            strict_win = 0
            break

    return strict_win, strict_resolved, loose_win, loose_resolved


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = []

    for sym in SYMBOLS:
        tdays = [r[0] for r in con.execute(
            "SELECT DISTINCT trading_day FROM daily_features WHERE symbol=? AND orb_minutes=5 ORDER BY trading_day",
            [sym],
        ).fetchall()]
        if len(tdays) < 3:
            continue

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
            [sym, gs.isoformat(), ge.isoformat()],
        ).fetchdf()
        if bars.empty:
            continue
        bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True)
        ts_all = bars["ts_utc"].values

        for anchor in ANCHORS:
            for i in range(1, len(tdays)):
                prev_td = tdays[i - 1]
                td = tdays[i]
                prev_bars = _slice_day(bars, ts_all, prev_td)
                day_bars = _slice_day(bars, ts_all, td)
                if prev_bars.empty or day_bars.empty:
                    continue

                prof = compute_volume_profile(prev_bars, bin_size=0.5)
                if prof is None:
                    continue

                val = float(prof["val"])
                vah = float(prof["vah"])
                start, _ = _orb_utc_window(td, anchor, 1)

                first = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < start + pd.Timedelta(minutes=1))]
                if first.empty:
                    continue
                op = float(first.iloc[0]["open"])
                if val <= op <= vah:
                    continue
                side = "above" if op > vah else "below"

                a_end = start + pd.Timedelta(minutes=30)
                b_end = start + pd.Timedelta(minutes=60)
                bars_a = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < a_end)]
                bars_b = day_bars[(day_bars["ts_utc"] >= a_end) & (day_bars["ts_utc"] < b_end)]

                checks: list[tuple[str, pd.Timestamp]] = []

                v_touch = (
                    not bars_a.empty and not bars_b.empty
                    and any(_overlap_va(r, val, vah) for _, r in bars_a.iterrows())
                    and any(_overlap_va(r, val, vah) for _, r in bars_b.iterrows())
                )
                if v_touch:
                    checks.append(("touch_A_B", pd.Timestamp(b_end)))

                v_close = (
                    not bars_a.empty and not bars_b.empty
                    and _close_in_va(bars_a.iloc[-1], val, vah)
                    and _close_in_va(bars_b.iloc[-1], val, vah)
                )
                if v_close:
                    checks.append(("close_A_B", pd.Timestamp(b_end)))

                first_hour = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < b_end)]
                touch = first_hour[(first_hour["high"] >= val) & (first_hour["low"] <= vah)]
                if not touch.empty:
                    checks.append(("first_touch_1h", pd.Timestamp(touch.iloc[0]["ts_utc"]) + pd.Timedelta(minutes=1)))

                for variant, entry_ts in checks:
                    after = day_bars[day_bars["ts_utc"] >= entry_ts]
                    sw, sr, lw, lr = _evaluate(after, side, val, vah)
                    rows.append({
                        "symbol": sym,
                        "anchor": anchor,
                        "variant": variant,
                        "open_side": side,
                        "strict_win": sw,
                        "strict_resolved": sr,
                        "loose_win": lw,
                        "loose_resolved": lr,
                    })

    con.close()

    if not rows:
        print("No setups found.")
        return 0

    df = pd.DataFrame(rows)

    out = []
    for mode in ("strict", "loose"):
        w = f"{mode}_win"
        r = f"{mode}_resolved"
        g = df.groupby(["anchor", "variant", "symbol", "open_side"], as_index=False).agg(
            setups=(r, "count"),
            resolved=(r, "sum"),
            wins=(w, "sum"),
        )
        g["hit_rate"] = np.where(g["resolved"] > 0, g["wins"] / g["resolved"], np.nan)
        g["mode"] = mode
        out.append(g)

    summary = pd.concat(out, ignore_index=True)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dalton_80_deepdive_summary.csv"
    summary.to_csv(summary_path, index=False)

    notes = []
    notes.append("# Dalton 80% Rule Deep-Dive")
    notes.append("")
    notes.append("Outcome modes:")
    notes.append("- strict: opposite VA side hit before rejection beyond entry side")
    notes.append("- loose: opposite VA side touched at any point after trigger")
    notes.append("")

    for mode in ("strict", "loose"):
        notes.append(f"## {mode.upper()} top lines")
        m = summary[summary["mode"] == mode].sort_values(["anchor", "variant", "symbol", "open_side"])
        for r in m.itertuples(index=False):
            notes.append(
                f"- {r.anchor} {r.variant} {r.symbol} {r.open_side}: "
                f"N={r.setups}, hit_rate={r.hit_rate:.1%}"
            )
        notes.append("")

    notes_path = out_dir / "dalton_80_deepdive_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
