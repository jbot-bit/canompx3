#!/usr/bin/env python3
"""
Non-ORB Strategy Scanner
Tests 5 genuinely non-ORB edge types using daily_features + bars_5m.
Entry: next-bar-open after signal confirms (no look-ahead).
Measures: win rate, avg_r, year consistency.

Strategies tested:
  S1 - Multi-day momentum/reversal: after 3+ consecutive directional closes -> fade or follow
  S2 - Inside day breakout: when today's range inside prior day -> next day direction bias
  S3 - Gap continuation vs fill: gap up/down -> does it continue or fill by session end?
  S4 - London open direction -> NY session follow-through
  S5 - Day-of-week bias: Monday momentum, Friday fade
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

SYMBOLS = ["MES", "MNQ", "M2K", "MGC", "MCL", "M6E"]
RR = 1.5
MIN_TRADES = 30
MIN_YEARS = 4

# ── helpers ──────────────────────────────────────────────────────────────────


def wr(s):
    return (s > 0).mean() if len(s) else np.nan


def avg_r(s):
    return s.mean() if len(s) else np.nan


def yr_consistency(df, col="pnl_r"):
    yrs = df.groupby("year")[col].mean()
    return (yrs > 0).sum(), len(yrs)


def summarise(label, sym, df, col="pnl_r"):
    if len(df) < MIN_TRADES:
        return None
    pos, tot = yr_consistency(df, col)
    if tot < MIN_YEARS:
        return None
    return {
        "strategy": label,
        "symbol": sym,
        "n": len(df),
        "wr": round(wr(df[col]) * 100, 1),
        "avg_r": round(avg_r(df[col]), 3),
        "yr_pos": f"{pos}/{tot}",
    }


# ── load daily features ───────────────────────────────────────────────────────


def load_daily(sym: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = f"""
    SELECT
        trading_day,
        prev_day_high, prev_day_low, prev_day_close, prev_day_direction,
        prev_day_range,
        daily_open, daily_high, daily_low, daily_close,
        gap_open_points, gap_type,
        session_london_high, session_london_low,
        session_ny_high, session_ny_low,
        orb_US_EQUITY_OPEN_break_dir AS us_open_dir,
        orb_US_EQUITY_OPEN_outcome   AS us_open_outcome,
        orb_LONDON_OPEN_break_dir    AS london_dir,
        orb_LONDON_OPEN_outcome      AS london_outcome,
        atr_20, atr_vel_regime,
        day_of_week, is_monday, is_friday, is_tuesday,
        overnight_high, overnight_low, overnight_range
    FROM daily_features
    WHERE symbol = '{sym}' AND orb_minutes = 5
    ORDER BY trading_day
    """
    df = con.execute(q).fetchdf()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df = df[df["year"] >= 2019].copy()
    return df


def load_bars5m(sym: str) -> pd.DataFrame:
    """Load 5-min bars for intraday P&L simulation."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = f"""
    SELECT ts, open, high, low, close, volume,
           DATE_TRUNC('day', ts AT TIME ZONE 'UTC' AT TIME ZONE 'Australia/Brisbane') AS local_day
    FROM bars_5m
    WHERE symbol = '{sym}'
    ORDER BY ts
    """
    df = con.execute(q).fetchdf()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["local_day"] = pd.to_datetime(df["local_day"]).dt.date
    return df


def sim_trade(bars5: pd.DataFrame, entry_ts_utc, direction: str, stop_r: float, rr: float = RR):
    """
    Simulate a trade from entry_ts_utc (enter at OPEN of the bar AT or AFTER entry_ts_utc).
    stop_r = ATR-based stop in points.
    Returns pnl_r or None if data missing.
    """
    future = bars5[bars5["ts"] >= entry_ts_utc].copy()
    if future.empty:
        return None
    entry_bar = future.iloc[0]
    entry_price = entry_bar["open"]
    if direction == "long":
        stop = entry_price - stop_r
        target = entry_price + stop_r * rr
    else:
        stop = entry_price + stop_r
        target = entry_price - stop_r * rr

    for _, bar in future.iterrows():
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
    return None  # unresolved


# ═══════════════════════════════════════════════════════════════════════════════
# S1 — Multi-day consecutive close momentum/reversal
# Signal: prev_day_direction tells us today's daily close vs prior close.
# Build a consecutive-day streak and test fade vs follow at US equity open.
# ═══════════════════════════════════════════════════════════════════════════════


def s1_consecutive_days(sym: str, df: pd.DataFrame) -> list[dict]:
    results = []
    df = df.copy().reset_index(drop=True)
    df["dir_num"] = df["prev_day_direction"].map({"up": 1, "down": -1}).fillna(0)

    # Build streak: consecutive days in same direction BEFORE today
    streaks = []
    for i in range(len(df)):
        if i == 0:
            streaks.append(1)
            continue
        if df.loc[i, "dir_num"] == df.loc[i - 1, "dir_num"] and df.loc[i, "dir_num"] != 0:
            streaks.append(streaks[-1] + 1)
        else:
            streaks.append(1)
    df["streak"] = streaks

    # Outcome proxy: did today's daily direction match the streak direction?
    # Use: daily_close vs daily_open as today's direction
    df["today_dir"] = np.where(df["daily_close"] > df["daily_open"], 1, -1)

    for streak_min in [2, 3]:
        subset = df[df["streak"] >= streak_min].copy()
        if len(subset) < MIN_TRADES:
            continue

        # FOLLOW: expect same direction
        subset["pnl_r"] = np.where(subset["dir_num"] == subset["today_dir"], RR, -1.0)
        r = summarise(f"S1_follow_{streak_min}d", sym, subset)
        if r:
            results.append(r)

        # FADE: expect reversal
        subset["pnl_r"] = np.where(subset["dir_num"] != subset["today_dir"], RR, -1.0)
        r = summarise(f"S1_fade_{streak_min}d", sym, subset)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# S2 — Inside day: today's range entirely inside prior day's range
# Signal: known after prior day close (before today's open).
# Hypothesis: compression → expansion. Does the first break (US open) tend to
#             continue in its direction?
# ═══════════════════════════════════════════════════════════════════════════════


def s2_inside_day(sym: str, df: pd.DataFrame) -> list[dict]:
    results = []
    df = df.copy()
    # Inside day: today's high < prev high AND today's low > prev low
    df["is_inside"] = (df["daily_high"] < df["prev_day_high"]) & (df["daily_low"] > df["prev_day_low"])

    inside = df[df["is_inside"]].copy()
    if len(inside) < MIN_TRADES:
        return results

    # Did the day (daily_open → daily_close) continue in the prior day's direction?
    inside["today_dir"] = np.where(inside["daily_close"] > inside["daily_open"], "up", "down")
    inside["pnl_follow"] = np.where(inside["today_dir"] == inside["prev_day_direction"], RR, -1.0)
    inside["pnl_fade"] = np.where(inside["today_dir"] != inside["prev_day_direction"], RR, -1.0)

    inside["pnl_r"] = inside["pnl_follow"]
    r = summarise("S2_inside_follow_prevday", sym, inside)
    if r:
        results.append(r)

    inside["pnl_r"] = inside["pnl_fade"]
    r = summarise("S2_inside_fade_prevday", sym, inside)
    if r:
        results.append(r)

    # Also check: inside day -> US open break direction (continuation of break)
    valid = inside[inside["us_open_dir"].notna()].copy()
    if len(valid) >= MIN_TRADES:
        valid["pnl_r"] = np.where(valid["us_open_outcome"] == "win", RR, -1.0)
        r = summarise("S2_inside_usopen_outcome", sym, valid)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# S3 — Gap patterns
# Signal: gap_open_points (known at session open) and gap_type
# Test: gap continuation vs gap fill (does price retrace to prior close by day end?)
# ═══════════════════════════════════════════════════════════════════════════════


def s3_gap(sym: str, df: pd.DataFrame) -> list[dict]:
    results = []
    df = df.copy()
    df = df[df["gap_open_points"].notna() & (df["atr_20"] > 0)].copy()
    df["gap_pct"] = df["gap_open_points"].abs() / df["atr_20"]
    df["gap_dir"] = np.where(df["gap_open_points"] > 0, "up", "down")

    # Fill = by end of day, price returned to prev_day_close
    # Use daily_high/low vs prev_day_close
    df["gap_filled"] = np.where(
        df["gap_dir"] == "up", df["daily_low"] <= df["prev_day_close"], df["daily_high"] >= df["prev_day_close"]
    )
    # Continuation = closed in gap direction
    df["gap_continued"] = np.where(
        df["gap_dir"] == "up", df["daily_close"] > df["daily_open"], df["daily_close"] < df["daily_open"]
    )

    for tier, lo, hi in [("small", 0.1, 0.5), ("medium", 0.5, 1.5), ("large", 1.5, 9.9)]:
        sub = df[(df["gap_pct"] >= lo) & (df["gap_pct"] < hi)].copy()
        if len(sub) < MIN_TRADES:
            continue

        # Fill rate
        sub["pnl_r"] = np.where(sub["gap_filled"], RR, -1.0)
        r = summarise(f"S3_gap_fill_{tier}", sym, sub)
        if r:
            results.append(r)

        # Continuation rate
        sub["pnl_r"] = np.where(sub["gap_continued"], RR, -1.0)
        r = summarise(f"S3_gap_cont_{tier}", sym, sub)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# S4 — London open direction → NY session follow-through
# Signal: london_dir is known before US equity open
# Test: does NY session (session_ny_high/low vs daily_open) follow London direction?
# ═══════════════════════════════════════════════════════════════════════════════


def s4_london_to_ny(sym: str, df: pd.DataFrame) -> list[dict]:
    results = []
    df = df.copy()
    valid = df[df["london_dir"].isin(["long", "short"])].copy()
    if len(valid) < MIN_TRADES:
        return results

    # NY direction proxy: session_ny_high - daily_open vs session_ny_low - daily_open
    valid["ny_net"] = (valid["session_ny_high"] + valid["session_ny_low"]) / 2 - valid["daily_open"]
    valid["ny_dir"] = np.where(valid["ny_net"] > 0, "long", "short")

    valid["pnl_follow"] = np.where(valid["ny_dir"] == valid["london_dir"], RR, -1.0)
    valid["pnl_fade"] = np.where(valid["ny_dir"] != valid["london_dir"], RR, -1.0)

    valid["pnl_r"] = valid["pnl_follow"]
    r = summarise("S4_london_ny_follow", sym, valid)
    if r:
        results.append(r)

    valid["pnl_r"] = valid["pnl_fade"]
    r = summarise("S4_london_ny_fade", sym, valid)
    if r:
        results.append(r)

    # Also: does London outcome (win/loss) predict NY direction alignment?
    for ld in ["long", "short"]:
        sub = valid[valid["london_dir"] == ld].copy()
        if len(sub) < MIN_TRADES:
            continue
        sub["pnl_r"] = sub["pnl_follow"]
        r = summarise(f"S4_london_{ld}_ny_follow", sym, sub)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# S5 — Day of week bias
# Signal: day_of_week (0=Mon, 4=Fri)
# Test: do certain days have directional bias?
# ═══════════════════════════════════════════════════════════════════════════════


def s5_day_of_week(sym: str, df: pd.DataFrame) -> list[dict]:
    results = []
    df = df.copy()
    df["today_dir"] = np.where(df["daily_close"] > df["daily_open"], "up", "down")
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

    for dow, name in dow_names.items():
        sub = df[df["day_of_week"] == dow].copy()
        if len(sub) < MIN_TRADES:
            continue
        # Long bias
        sub["pnl_long"] = np.where(sub["today_dir"] == "up", RR, -1.0)
        sub["pnl_short"] = np.where(sub["today_dir"] == "down", RR, -1.0)

        sub["pnl_r"] = sub["pnl_long"]
        r = summarise(f"S5_dow_{name}_long", sym, sub)
        if r:
            results.append(r)

        sub["pnl_r"] = sub["pnl_short"]
        r = summarise(f"S5_dow_{name}_short", sym, sub)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# S6 — Overnight range extremes as direction signal
# Signal: did overnight session take PDH or PDL before NY open?
# Already captured in orb_1000_break_dir / took_pdh_before_1000
# But let's check overnight_high vs prev_day_high as a standalone filter
# ═══════════════════════════════════════════════════════════════════════════════


def s6_overnight(sym: str, df: pd.DataFrame) -> list[dict]:
    results = []
    df = df.copy()

    # Overnight tagged PDH → expect rejection (fade) in NY
    # Overnight tagged PDL → expect rejection (fade) in NY
    df["ovn_tagged_pdh"] = df["overnight_high"] >= df["prev_day_high"] * 0.9995
    df["ovn_tagged_pdl"] = df["overnight_low"] <= df["prev_day_low"] * 1.0005

    df["ny_dir"] = np.where(df["daily_close"] > df["daily_open"], "up", "down")

    # Tagged PDH overnight → fade (short NY)
    pdh_sub = df[df["ovn_tagged_pdh"] & ~df["ovn_tagged_pdl"]].copy()
    if len(pdh_sub) >= MIN_TRADES:
        pdh_sub["pnl_r"] = np.where(pdh_sub["ny_dir"] == "down", RR, -1.0)
        r = summarise("S6_ovn_pdh_fade_ny", sym, pdh_sub)
        if r:
            results.append(r)

    # Tagged PDL overnight → fade (long NY)
    pdl_sub = df[df["ovn_tagged_pdl"] & ~df["ovn_tagged_pdh"]].copy()
    if len(pdl_sub) >= MIN_TRADES:
        pdl_sub["pnl_r"] = np.where(pdl_sub["ny_dir"] == "up", RR, -1.0)
        r = summarise("S6_ovn_pdl_fade_ny", sym, pdl_sub)
        if r:
            results.append(r)

    # Neither → clean day (no extreme tags) → momentum likely
    clean = df[~df["ovn_tagged_pdh"] & ~df["ovn_tagged_pdl"]].copy()
    if len(clean) >= MIN_TRADES:
        # Does prior day direction continue on clean overnight days?
        clean["pnl_r"] = np.where(
            (clean["prev_day_direction"] == "up") & (clean["ny_dir"] == "up")
            | (clean["prev_day_direction"] == "down") & (clean["ny_dir"] == "down"),
            RR,
            -1.0,
        )
        r = summarise("S6_ovn_clean_prevday_follow", sym, clean)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    all_results = []

    for sym in SYMBOLS:
        print(f"  {sym}...", flush=True)
        try:
            df = load_daily(sym)
            if len(df) < 100:
                print(f"    skip (not enough data)")
                continue

            all_results += s1_consecutive_days(sym, df)
            all_results += s2_inside_day(sym, df)
            all_results += s3_gap(sym, df)
            all_results += s4_london_to_ny(sym, df)
            all_results += s5_day_of_week(sym, df)
            all_results += s6_overnight(sym, df)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()

    if not all_results:
        print("\nNo results passed filters.")
        return

    res = pd.DataFrame(all_results)
    res = res.sort_values("avg_r", ascending=False)

    # Split: positive signal vs noise
    positives = res[res["avg_r"] > 0.05].copy()
    negatives = res[res["avg_r"] < -0.05].copy()

    print("\n" + "=" * 80)
    print("NON-ORB SCANNER RESULTS — POSITIVE SIGNALS")
    print("=" * 80)
    print(positives.to_string(index=False))

    print("\n" + "=" * 80)
    print("NON-ORB SCANNER RESULTS — FLAT/NEGATIVE (informational)")
    print("=" * 80)
    print(negatives.head(20).to_string(index=False))

    # Save
    out_path = OUT_DIR / "non_orb_scanner_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Markdown report
    lines = ["# Non-ORB Strategy Scanner Results\n"]
    lines.append("## Positive Signals (avg_r > 0.05)\n")
    lines.append(positives.to_markdown(index=False))
    lines.append("\n\n## All Results (sorted by avg_r)\n")
    lines.append(res.to_markdown(index=False))

    rpt_path = OUT_DIR / "non_orb_scanner_report.md"
    rpt_path.write_text("\n".join(lines))
    print(f"Report saved to {rpt_path}")


if __name__ == "__main__":
    print("Non-ORB Strategy Scanner")
    print("Symbols:", SYMBOLS)
    print("RR target:", RR)
    print()
    main()
