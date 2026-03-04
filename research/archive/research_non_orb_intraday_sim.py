#!/usr/bin/env python3
"""
Non-ORB Intraday Simulation
Properly simulates two candidate edges with real ATR-based stop/target and intraday path.

Strategy 1: Monday MES long — enter at US equity open (9:30am NY) on Mondays, long, ATR stop
Strategy 2: London direction -> NY follow — enter at US equity open in London's direction

Entry: open of first 5m bar at/after US equity open (handles DST automatically)
Stop:  0.5 * ATR20 (in points)
Target: RR * stop
Time-stop: 6 hours (no hit = scratch at 0)
"""

from __future__ import annotations
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"
OUT_DIR = ROOT / "research" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RR = 1.5
MAX_BARS = 72  # 6 hours of 5m bars
US_OPEN_UTC_START = 13  # 13:30 UTC (EDT) or 14:00+ UTC (EST) — grab window 13:25-14:35 UTC

# ─── Data loading ────────────────────────────────────────────────────────────

def load_features(sym: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = f"""
    SELECT trading_day,
           atr_20,
           day_of_week, is_monday, is_friday,
           orb_LONDON_OPEN_break_dir    AS london_dir,
           orb_LONDON_OPEN_outcome      AS london_outcome,
           orb_US_EQUITY_OPEN_break_dir AS us_dir,
           us_dst
    FROM daily_features
    WHERE symbol='{sym}' AND orb_minutes=5
    ORDER BY trading_day
    """
    df = con.execute(q).fetchdf()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    return df[df["year"] >= 2019].copy()


def load_bars(sym: str) -> pd.DataFrame:
    """Load all 5m bars, indexed by ts_utc."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = f"SELECT ts_utc, open, high, low, close FROM bars_5m WHERE symbol='{sym}' ORDER BY ts_utc"
    df = con.execute(q).fetchdf()
    con.close()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.sort_values("ts_utc").reset_index(drop=True)
    return df


def find_entry_bar_idx(bars: pd.DataFrame, trading_day_date, us_dst: bool) -> int | None:
    """
    Find the index of the first 5m bar at/after the US equity open on trading_day.
    US equity open = 9:30am ET:
      EST (us_dst=False): 14:30 UTC
      EDT (us_dst=True):  13:30 UTC
    Tight 10-minute window to avoid catching pre-market bars.
    """
    day = pd.Timestamp(trading_day_date, tz="UTC")
    if us_dst:
        # EDT: 9:30am = 13:30 UTC
        window_start = day + pd.Timedelta(hours=13, minutes=28)
        window_end   = day + pd.Timedelta(hours=13, minutes=45)
    else:
        # EST: 9:30am = 14:30 UTC
        window_start = day + pd.Timedelta(hours=14, minutes=28)
        window_end   = day + pd.Timedelta(hours=14, minutes=45)
    mask = (bars["ts_utc"] >= window_start) & (bars["ts_utc"] <= window_end)
    idxs = bars.index[mask].tolist()
    if not idxs:
        return None
    return idxs[0]


def simulate_trade(bars: pd.DataFrame, entry_idx: int, direction: str,
                   atr: float, rr: float = RR, max_bars: int = MAX_BARS):
    """
    Enter at open of bars[entry_idx], direction = 'long'|'short'.
    Stop  = 0.5 * ATR20
    Target = rr * stop
    Returns pnl_r: +rr (win), -1.0 (loss), 0.0 (time-stop)
    """
    if entry_idx is None or entry_idx >= len(bars):
        return None
    if np.isnan(atr) or atr <= 0:
        return None

    stop_dist = 0.5 * atr
    entry_price = bars.iloc[entry_idx]["open"]
    if direction == "long":
        stop   = entry_price - stop_dist
        target = entry_price + stop_dist * rr
    else:
        stop   = entry_price + stop_dist
        target = entry_price - stop_dist * rr

    end_idx = min(entry_idx + max_bars, len(bars))
    for i in range(entry_idx, end_idx):
        bar = bars.iloc[i]
        if direction == "long":
            if bar["low"] <= stop:
                return -1.0
            if bar["high"] >= target:
                return rr
        else:
            if bar["high"] >= stop:
                return -1.0
            if bar["low"] <= target:
                return rr
    return 0.0  # time-stop: scratch


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def yr_summary(df: pd.DataFrame, label: str) -> dict:
    df = df[df["pnl_r"].notna()].copy()
    if len(df) < 20:
        return None
    yr = df.groupby("year")["pnl_r"].agg(["mean", "count"])
    pos_yrs = (yr["mean"] > 0).sum()
    total_yrs = len(yr)
    return {
        "label": label,
        "n": len(df),
        "wr_%": round((df["pnl_r"] > 0).mean() * 100, 1),
        "avg_r": round(df["pnl_r"].mean(), 3),
        "total_r": round(df["pnl_r"].sum(), 1),
        "yr_pos": f"{pos_yrs}/{total_yrs}",
        "yr_detail": yr["mean"].round(3).to_dict(),
    }


def print_result(r: dict):
    if r is None:
        print("  (insufficient data)")
        return
    print(f"  n={r['n']}  WR={r['wr_%']}%  avg_r={r['avg_r']}  total_r={r['total_r']}R  years={r['yr_pos']}")
    for yr, val in sorted(r["yr_detail"].items()):
        bar = "+" * max(0, int(val * 10)) if val > 0 else "-" * max(0, int(abs(val) * 10))
        print(f"    {yr}: {val:+.3f}  {bar}")


# ─── Main ────────────────────────────────────────────────────────────────────

def run(sym: str):
    print(f"\n{'='*60}")
    print(f"SYMBOL: {sym}")
    print(f"{'='*60}")

    feats = load_features(sym)
    bars  = load_bars(sym)

    results_mon_long    = []
    results_london_follow = []
    results_london_long  = []
    results_london_short = []

    for _, row in feats.iterrows():
        atr = row["atr_20"]
        us_dst = bool(row["us_dst"]) if pd.notna(row["us_dst"]) else False
        entry_idx = find_entry_bar_idx(bars, row["trading_day"], us_dst)

        # ── S1: Monday long ────────────────────────────────────────────
        if row["is_monday"]:
            pnl = simulate_trade(bars, entry_idx, "long", atr)
            if pnl is not None:
                results_mon_long.append({"year": row["year"], "pnl_r": pnl,
                                         "trading_day": row["trading_day"]})

        # ── S2: London direction -> NY follow ──────────────────────────
        ld = row["london_dir"]
        if ld in ("long", "short"):
            direction = "long" if ld == "long" else "short"
            pnl = simulate_trade(bars, entry_idx, direction, atr)
            if pnl is not None:
                results_london_follow.append({"year": row["year"], "pnl_r": pnl,
                                              "direction": direction,
                                              "trading_day": row["trading_day"]})
                if ld == "long":
                    results_london_long.append({"year": row["year"], "pnl_r": pnl})
                else:
                    results_london_short.append({"year": row["year"], "pnl_r": pnl})

    # ── Print results ─────────────────────────────────────────────────
    print("\n[S1] Monday LONG — enter at US equity open")
    r = yr_summary(pd.DataFrame(results_mon_long), "Monday Long")
    print_result(r)

    print("\n[S2a] London direction -> NY follow (all)")
    r = yr_summary(pd.DataFrame(results_london_follow), "London->NY all")
    print_result(r)

    print("\n[S2b] London LONG -> NY long")
    r = yr_summary(pd.DataFrame(results_london_long), "London long->NY")
    print_result(r)

    print("\n[S2c] London SHORT -> NY short")
    r = yr_summary(pd.DataFrame(results_london_short), "London short->NY")
    print_result(r)

    return {
        "sym": sym,
        "mon_long": yr_summary(pd.DataFrame(results_mon_long), "Monday Long"),
        "london_follow": yr_summary(pd.DataFrame(results_london_follow), "London->NY"),
    }


if __name__ == "__main__":
    print("Non-ORB Intraday Simulation")
    print(f"RR={RR}  Stop=0.5*ATR20  Max bars={MAX_BARS} (6h)")
    print("Entry: open of first 5m bar at/after 9:30am NY")

    all_res = []
    for sym in ["MES", "MGC", "M2K", "M6E"]:
        r = run(sym)
        all_res.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symbol':<8} {'Strategy':<22} {'n':>5} {'WR%':>6} {'avg_r':>7} {'years':>8}")
    print("-" * 60)
    for r in all_res:
        sym = r["sym"]
        for key, label in [("mon_long", "Mon long"), ("london_follow", "London->NY follow")]:
            v = r[key]
            if v:
                print(f"{sym:<8} {label:<22} {v['n']:>5} {v['wr_%']:>5.1f}% {v['avg_r']:>7.3f} {v['yr_pos']:>8}")
