"""
HONEST VERIFICATION: MNQ 1100 direction asymmetry from raw bars_1m only.

Does NOT use:
  - orb_outcomes (pre-computed)
  - daily_features (pre-computed)
  - break_quality_conditions.csv (pre-computed)
  - Any pipeline output tables

Uses ONLY:
  - bars_1m (raw OHLCV data)
  - pipeline.cost_model (arithmetic functions, no DB)

Methodology:
  For each trading day in MNQ bars_1m range:
  1. Pull 5-min ORB window (11:00-11:04 Brisbane = 01:00-01:04 UTC)
  2. Compute ORB high/low from raw bars
  3. Apply G4+ filter (ORB size >= 4 NQ points)
  4. Find first bar AFTER ORB with close outside range (break detection)
  5. Classify break_dir: long if close > orb_high, short if close < orb_low
  6. E1 entry: next bar after confirm bar (CB1)
  7. Scan forward to RR2.5 target or stop, or trading day end
  8. Compute pnl_r with costs
  9. Report LONG vs SHORT outcomes independently
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, to_r_multiple, risk_in_dollars

# =============================================================================
# CONFIG
# =============================================================================
INSTRUMENT = "MNQ"
SESSION_BRISBANE_HOUR = 11   # 1100 session = 11:00 Brisbane = 01:00 UTC
ORB_MINUTES = 5              # 5-min ORB window
RR_TARGET = 2.5
CONFIRM_BARS = 1             # CB1: first close outside ORB
MIN_ORB_G = 4.0              # G4+ filter

# Brisbane = UTC+10 always (no DST)
BRISBANE_UTC_OFFSET_HOURS = 10

# Trading day: 09:00 Brisbane → next 09:00 Brisbane
# Session 1100 bars are from 11:00 Brisbane onward
# Trading day END: next calendar day 09:00 Brisbane
# In UTC: calendar_date 23:00 UTC = next calendar day 09:00 Brisbane


def brisbane_to_utc(cal_date: date, hour: int, minute: int = 0) -> datetime:
    """Convert Brisbane calendar date + time to UTC datetime."""
    bris = datetime(cal_date.year, cal_date.month, cal_date.day, hour, minute, 0)
    return bris - timedelta(hours=BRISBANE_UTC_OFFSET_HOURS)


def trading_day_from_brisbane_dt(dt: datetime) -> date:
    """Return Brisbane trading day for a Brisbane datetime.
    Trading day starts at 09:00 Brisbane, so bars before 09:00 belong to prev day."""
    if dt.hour < 9:
        return (dt - timedelta(days=1)).date()
    return dt.date()


def run_verification():
    print(f"VERIFICATION: MNQ 1100 direction asymmetry from raw bars_1m")
    print(f"Config: CB{CONFIRM_BARS}/RR{RR_TARGET}/G{MIN_ORB_G}+ | DB: {GOLD_DB_PATH}")
    print()

    cost = get_cost_spec(INSTRUMENT)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Pull ALL MNQ bars_1m as a single DataFrame
    # Timestamps are stored as Brisbane time (UTC+10)
    print("Loading bars_1m... ", end="", flush=True)
    df_all = con.execute("""
        SELECT ts_utc, open, high, low, close
        FROM bars_1m
        WHERE symbol = 'MNQ'
        ORDER BY ts_utc
    """).df()
    con.close()
    print(f"{len(df_all):,} bars loaded.")

    # Convert timestamps: they come back as tz-aware Brisbane or naive Brisbane?
    # From the spot check we saw they return as Brisbane-tz datetimes.
    # Normalize to naive for simplicity (they're all Brisbane time).
    if hasattr(df_all["ts_utc"].iloc[0], "tzinfo") and df_all["ts_utc"].iloc[0].tzinfo is not None:
        df_all["ts_bris"] = df_all["ts_utc"].dt.tz_localize(None)
    else:
        df_all["ts_bris"] = df_all["ts_utc"]

    df_all["cal_date"] = df_all["ts_bris"].dt.date
    df_all["hour"] = df_all["ts_bris"].dt.hour
    df_all["minute"] = df_all["ts_bris"].dt.minute

    # Get all calendar dates that have 1100-session bars
    session_dates = df_all[df_all["hour"] == SESSION_BRISBANE_HOUR]["cal_date"].unique()
    session_dates.sort()
    print(f"Calendar dates with 1100 bars: {len(session_dates)}")

    results = []
    skipped_no_orb = 0
    skipped_small_orb = 0
    skipped_no_break = 0
    skipped_no_entry = 0

    for cal_date in session_dates:
        # ----- Step 1: ORB window (11:00-11:04 Brisbane) -----
        orb_bars = df_all[
            (df_all["cal_date"] == cal_date)
            & (df_all["hour"] == SESSION_BRISBANE_HOUR)
            & (df_all["minute"] < ORB_MINUTES)
        ].sort_values("ts_bris")

        if len(orb_bars) < ORB_MINUTES:
            skipped_no_orb += 1
            continue

        orb_high = float(orb_bars["high"].max())
        orb_low = float(orb_bars["low"].min())
        orb_size = orb_high - orb_low

        # ----- Step 2: G4+ filter -----
        if orb_size < MIN_ORB_G:
            skipped_small_orb += 1
            continue

        # ----- Step 3: Post-ORB bars (from 11:05 onwards, same trading day) -----
        # Trading day ends at next day 09:00 Brisbane = current cal_date 23:00 Brisbane + 10h
        # i.e., bars up to 08:59 Brisbane on (cal_date + 1 day)
        td_end_bris = datetime(cal_date.year, cal_date.month, cal_date.day,
                               9, 0, 0) + timedelta(days=1)

        post_orb = df_all[
            (df_all["ts_bris"] >= datetime(cal_date.year, cal_date.month, cal_date.day,
                                           SESSION_BRISBANE_HOUR, ORB_MINUTES, 0))
            & (df_all["ts_bris"] < td_end_bris)
        ].sort_values("ts_bris").reset_index(drop=True)

        if post_orb.empty:
            skipped_no_break += 1
            continue

        # ----- Step 4: Break detection (CB1: first close outside ORB) -----
        closes = post_orb["close"].values
        long_outside = closes > orb_high
        short_outside = closes < orb_low

        # Find first bar where close breaks out in either direction
        long_idx = int(np.argmax(long_outside)) if long_outside.any() else None
        short_idx = int(np.argmax(short_outside)) if short_outside.any() else None

        if long_idx is None and short_idx is None:
            skipped_no_break += 1
            continue

        # Determine which break came first (or the only one)
        if long_idx is None:
            break_idx = short_idx
            break_dir = "short"
        elif short_idx is None:
            break_idx = long_idx
            break_dir = "long"
        elif long_idx <= short_idx:
            break_idx = long_idx
            break_dir = "long"
        else:
            break_idx = short_idx
            break_dir = "short"

        # ----- Step 5: E1 entry (next bar after confirm bar) -----
        entry_bar_idx = break_idx + 1
        if entry_bar_idx >= len(post_orb):
            skipped_no_entry += 1
            continue

        entry_bar = post_orb.iloc[entry_bar_idx]
        entry_price = float(entry_bar["open"])
        stop_price = orb_low if break_dir == "long" else orb_high

        risk_points = abs(entry_price - stop_price)
        if risk_points <= 0:
            skipped_no_entry += 1
            continue

        # ----- Step 6: Compute target price -----
        if break_dir == "long":
            target_price = entry_price + risk_points * RR_TARGET
        else:
            target_price = entry_price - risk_points * RR_TARGET

        # ----- Step 7: Scan forward for exit -----
        scan_bars = post_orb.iloc[entry_bar_idx:]  # includes entry bar itself
        outcome = None
        pnl_r = None

        for idx, bar in scan_bars.iterrows():
            bh = float(bar["high"])
            bl = float(bar["low"])

            if break_dir == "long":
                hit_target = bh >= target_price
                hit_stop = bl <= stop_price
            else:
                hit_target = bl <= target_price
                hit_stop = bh >= stop_price

            # Fill bar itself (entry at open, then check rest of bar)
            if idx == entry_bar_idx:
                # For E1: entered at open; check if rest of bar hits stop or target
                # For safety, check both — if ambiguous, conservative loss
                if hit_target and hit_stop:
                    outcome = "loss"; pnl_r = -1.0; break
                elif hit_target:
                    outcome = "win"
                    win_pts = risk_points * RR_TARGET
                    pnl_r = to_r_multiple(cost, entry_price, stop_price, win_pts)
                    break
                elif hit_stop:
                    outcome = "loss"; pnl_r = -1.0; break
                # else: continue to next bar
                continue

            # Post-entry bars
            if hit_target and hit_stop:
                outcome = "loss"; pnl_r = -1.0; break
            elif hit_target:
                outcome = "win"
                win_pts = risk_points * RR_TARGET
                pnl_r = to_r_multiple(cost, entry_price, stop_price, win_pts)
                break
            elif hit_stop:
                outcome = "loss"; pnl_r = -1.0; break

        if outcome is None:
            # Session ended without hit
            outcome = "scratch"
            pnl_r = to_r_multiple(cost, entry_price, stop_price, 0.0)

        results.append({
            "cal_date": cal_date,
            "break_dir": break_dir,
            "orb_size": round(orb_size, 2),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": round(target_price, 2),
            "risk_points": round(risk_points, 2),
            "outcome": outcome,
            "pnl_r": round(pnl_r, 4),
        })

    print(f"Skipped — insufficient ORB bars: {skipped_no_orb}")
    print(f"Skipped — ORB too small (<G4): {skipped_small_orb}")
    print(f"Skipped — no break detected: {skipped_no_break}")
    print(f"Skipped — no entry bar: {skipped_no_entry}")
    print(f"Processed trades: {len(results)}")
    print()

    if not results:
        print("No results — something is wrong.")
        return

    df = pd.DataFrame(results)
    df["year"] = pd.to_datetime(df["cal_date"]).dt.year

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print("=" * 60)
    print("HONEST VERIFICATION: MNQ 1100 LONG vs SHORT")
    print(f"Source: bars_1m ONLY. Pre-computed tables: NOT USED.")
    print("=" * 60)

    for dirn in ("long", "short"):
        sub = df[df["break_dir"] == dirn]["pnl_r"].values
        n = len(sub)
        avg_r = float(np.mean(sub))
        wr = float(np.mean(sub > 0))
        t, p = stats.ttest_1samp(sub, 0.0)
        print(f"\n  {dirn.upper():5s}: N={n:4d}  avgR={avg_r:+.4f}  WR={wr:.1%}  t={t:.3f}  p={p:.5f}")

    print(f"\n  Asymmetry (LONG - SHORT): {df[df['break_dir']=='long']['pnl_r'].mean() - df[df['break_dir']=='short']['pnl_r'].mean():+.4f}")

    # Year-by-year
    print("\n--- Year-by-year ---")
    for yr, grp in df.groupby("year"):
        for dirn in ("long", "short"):
            sub = grp[grp["break_dir"] == dirn]["pnl_r"].values
            if len(sub) < 5:
                continue
            print(f"  {yr} {dirn:5s}: N={len(sub):4d}  avgR={np.mean(sub):+.4f}  WR={np.mean(sub>0):.1%}")

    # Outcome breakdown
    print("\n--- Outcome breakdown ---")
    print(df.groupby(["break_dir", "outcome"]).size().unstack(fill_value=0).to_string())

    # Scratch analysis — what's dragging the average?
    print("\n--- Non-scratch only (win/loss) ---")
    df_ns = df[df["outcome"] != "scratch"]
    for dirn in ("long", "short"):
        sub = df_ns[df_ns["break_dir"] == dirn]["pnl_r"].values
        if len(sub) == 0:
            continue
        print(f"  {dirn:5s}: N={len(sub):4d}  avgR={np.mean(sub):+.4f}  WR={np.mean(sub>0):.1%}")

    # Save raw results
    out_path = PROJECT_ROOT / "research" / "output" / "verify_direction_asymmetry_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRaw results saved: {out_path}")


if __name__ == "__main__":
    run_verification()
