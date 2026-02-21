"""
research_inside_entry.py -- Inside ORB entry test for MGC 0900.

CORE QUESTION: if you set a limit order 1/5 ORB inside the edge (after ORB forms
but before a confirmed break), is it profitable — INCLUDING the false fills?

HONEST SIMULATION RULES:
  1. E0 BASELINE  : fills when bar CLOSES above/below ORB edge (match real CB1 logic)
  2. INSIDE ENTRY : fills on intra-bar TOUCH of inside level (limit order = touch fills)
  3. No direction signal — first side touched = trade direction (OCO model)
  4. Stop = opposite ORB edge. Target = entry +/- risk * RR.
  5. Prev outcome via LAG (zero look-ahead).
  6. Skip ORB formation bar. Only scan post-ORB bars.

DIRECTION AUDIT:
  Also checks whether prev=LOSS has a directional bias (long vs short fill rate)
  so we know if a one-sided limit (not OCO) is warranted.

Usage:
    python research/research_inside_entry.py
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.paths import GOLD_DB_PATH

INSTRUMENT  = "MGC"
SESSION     = "0900"
ORB_MINUTES = 5
ORB_MIN_PTS = 4.0
RR_TARGETS  = [1.5, 2.0, 2.5, 3.0]
OFFSETS     = [0.0, 0.10, 0.15, 0.20, 0.25]


def load_orb_days(con) -> pd.DataFrame:
    return con.execute("""
        WITH prev AS (
            SELECT trading_day, symbol,
                   LAG(orb_0900_outcome) OVER (
                       PARTITION BY symbol ORDER BY trading_day
                   ) AS prev_outcome
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
        )
        SELECT
            d.trading_day,
            d.orb_0900_high                         AS orb_high,
            d.orb_0900_low                          AS orb_low,
            (d.orb_0900_high - d.orb_0900_low)      AS orb_size,
            p.prev_outcome
        FROM daily_features d
        JOIN prev p ON p.trading_day = d.trading_day AND p.symbol = d.symbol
        WHERE d.symbol = ?
          AND d.orb_minutes = 5
          AND d.orb_0900_high IS NOT NULL AND d.orb_0900_low IS NOT NULL
          AND (d.orb_0900_high - d.orb_0900_low) >= ?
        ORDER BY d.trading_day
    """, [INSTRUMENT, INSTRUMENT, ORB_MIN_PTS]).df()


def load_post_orb_bars(con, trading_days: list) -> dict:
    days_str = ", ".join(f"DATE '{d}'" for d in trading_days)
    bars = con.execute(f"""
        SELECT b.ts_utc, d.trading_day, b.open, b.high, b.low, b.close
        FROM bars_1m b
        JOIN (
            SELECT DISTINCT
                trading_day,
                trading_day::TIMESTAMP AT TIME ZONE 'UTC'
                    - INTERVAL 1 HOUR
                    + INTERVAL '{ORB_MINUTES} minutes'  AS post_orb_start,
                trading_day::TIMESTAMP AT TIME ZONE 'UTC' + INTERVAL 9 HOUR AS session_end
            FROM daily_features
            WHERE symbol = '{INSTRUMENT}' AND orb_minutes = 5
              AND trading_day IN ({days_str})
        ) d ON b.ts_utc >= d.post_orb_start AND b.ts_utc <= d.session_end
        WHERE b.symbol = '{INSTRUMENT}'
        ORDER BY b.ts_utc
    """).df()
    bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True)
    return {day: grp.reset_index(drop=True) for day, grp in bars.groupby("trading_day")}


def scan_outcome(bars: pd.DataFrame, fill_bar_i: int, fill_dir: str,
                 fill_price: float, stop: float, target: float) -> dict:
    """Scan bars from fill onwards for target/stop hit. Returns outcome dict with pnl in R-multiples."""
    risk = abs(fill_price - stop)
    rr   = abs(target - fill_price) / risk  # e.g. 2.0

    # Check fill bar itself
    fb = bars.iloc[fill_bar_i]
    if fill_dir == "long":
        if fb["high"] >= target and fb["low"] <= stop:
            return {"outcome": "loss", "pnl_r": -1.0}  # ambiguous -> conservative
        if fb["high"] >= target:
            return {"outcome": "win",  "pnl_r": +rr}
        if fb["low"]  <= stop:
            return {"outcome": "loss", "pnl_r": -1.0}
    else:
        if fb["low"] <= target and fb["high"] >= stop:
            return {"outcome": "loss", "pnl_r": -1.0}
        if fb["low"]  <= target:
            return {"outcome": "win",  "pnl_r": +rr}
        if fb["high"] >= stop:
            return {"outcome": "loss", "pnl_r": -1.0}

    # Post-fill bars
    for _, bar in bars.iloc[fill_bar_i + 1:].iterrows():
        if fill_dir == "long":
            ht = bar["high"] >= target
            hs = bar["low"]  <= stop
        else:
            ht = bar["low"]  <= target
            hs = bar["high"] >= stop
        if ht and hs:
            return {"outcome": "loss", "pnl_r": -1.0}
        elif ht:
            return {"outcome": "win",  "pnl_r": +rr}
        elif hs:
            return {"outcome": "loss", "pnl_r": -1.0}

    return {"outcome": "scratch", "pnl_r": 0.0}


def simulate_e0_baseline(bars: pd.DataFrame, orb_high: float, orb_low: float,
                          orb_size: float, rr: float) -> dict | None:
    """
    E0 CB1 baseline: fill when bar CLOSES above/below ORB edge (matching real CB1 logic).
    First confirmed close = fill direction.
    """
    for i, bar in bars.iterrows():
        closed_above = bar["close"] > orb_high
        closed_below = bar["close"] < orb_low
        if closed_above:
            fill_dir   = "long"
            fill_price = orb_high
            stop       = orb_low
            risk       = orb_size
            target     = fill_price + risk * rr
            result     = scan_outcome(bars, i, fill_dir, fill_price, stop, target)
            result["fill_dir"] = fill_dir
            return result
        elif closed_below:
            fill_dir   = "short"
            fill_price = orb_low
            stop       = orb_high
            risk       = orb_size
            target     = fill_price - risk * rr
            result     = scan_outcome(bars, i, fill_dir, fill_price, stop, target)
            result["fill_dir"] = fill_dir
            return result
    return None


def simulate_inside_entry(bars: pd.DataFrame, orb_high: float, orb_low: float,
                           orb_size: float, offset: float, rr: float) -> dict | None:
    """
    Inside entry: fill on intra-bar TOUCH of inside level (resting limit order).
    First side touched = fill direction (OCO model).
    """
    delta            = orb_size * offset
    long_fill_level  = orb_high - delta
    short_fill_level = orb_low  + delta

    for i, bar in bars.iterrows():
        tl = bar["high"] >= long_fill_level
        ts = bar["low"]  <= short_fill_level

        if tl and ts:
            # Same bar: use close direction
            if bar["close"] >= (orb_high + orb_low) / 2:
                fill_dir, fill_price = "long",  long_fill_level
            else:
                fill_dir, fill_price = "short", short_fill_level
        elif tl:
            fill_dir, fill_price = "long",  long_fill_level
        elif ts:
            fill_dir, fill_price = "short", short_fill_level
        else:
            continue

        if fill_dir == "long":
            stop   = orb_low
            risk   = fill_price - stop
            target = fill_price + risk * rr
        else:
            stop   = orb_high
            risk   = stop - fill_price
            target = fill_price - risk * rr

        if risk <= 0:
            return None

        result = scan_outcome(bars, i, fill_dir, fill_price, stop, target)
        result["fill_dir"] = fill_dir
        return result

    return None


def summarise(records: list, label: str, min_n: int = 10) -> None:
    if len(records) < min_n:
        print(f"  {label:58s}  N<{min_n}")
        return
    arr   = np.array([r["pnl_r"] for r in records])
    wins  = sum(1 for r in records if r["outcome"] == "win")
    n     = len(arr)
    wr    = wins / n
    avg_r = arr.mean()
    t, p  = stats.ttest_1samp(arr, 0)
    star  = "***" if p < 0.01 else "** " if p < 0.05 else "   "
    print(f"  {label:58s}  N={n:4d}  WR={wr:.1%}  avgR={avg_r:+.4f}  p={p:.4f} {star}")


def run_all_days(days_df, bars_by_day, offset, rr, use_e0_baseline=False):
    """Run simulation for all days, return list of result dicts."""
    results = []
    for _, day in days_df.iterrows():
        day_bars = bars_by_day.get(day["trading_day"], pd.DataFrame())
        if use_e0_baseline:
            result = simulate_e0_baseline(
                day_bars, day["orb_high"], day["orb_low"], day["orb_size"], rr
            )
        else:
            result = simulate_inside_entry(
                day_bars, day["orb_high"], day["orb_low"], day["orb_size"], offset, rr
            )
        if result is None:
            continue
        result["prev_outcome"] = day["prev_outcome"]
        results.append(result)
    return results


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("Loading ORB days (G4+)...")
    days_df = load_orb_days(con)
    n_days  = len(days_df)
    print(f"  Total G4+ trading days: {n_days}")

    print("Loading post-ORB bars (5m, after ORB formation)...")
    bars_by_day = load_post_orb_bars(con, days_df["trading_day"].tolist())
    con.close()
    print(f"  Days with bars: {len(bars_by_day)}")

    print()
    print("=" * 100)
    print(f"INSIDE ENTRY TEST  --  {INSTRUMENT} {SESSION}  G4+  [{n_days} trading days]")
    print("=" * 100)
    print("E0 BASELINE  : fills when bar CLOSES above/below ORB edge  (matches real CB1)")
    print("INSIDE ENTRY : resting limit order — fills on intra-bar touch of inside level")
    print("Direction    : first side touched (OCO model, no look-ahead)")
    print()

    for rr in RR_TARGETS:
        print(f"--- RR = {rr} {'':->70}")

        # E0 baseline (correct: close confirmation)
        e0_results = run_all_days(days_df, bars_by_day, 0, rr, use_e0_baseline=True)
        summarise(e0_results, f"E0 BASELINE (close-confirmed, {len(e0_results)} fills)")

        for offset in OFFSETS:
            if offset == 0.0:
                continue  # shown as E0 baseline above

            inside_results = run_all_days(days_df, bars_by_day, offset, rr)
            n_extra        = len(inside_results) - len(e0_results)
            label = f"inside {offset:.0%}  ({len(inside_results)} fills, {'+' if n_extra >= 0 else ''}{n_extra} vs E0)"
            summarise(inside_results, label)

            # prev=LOSS subset
            pl = [r for r in inside_results if r["prev_outcome"] == "loss"]
            if len(pl) >= 10:
                summarise(pl, f"  +- prev=LOSS  ({len(pl)} fills)")

        print()

    # ── Direction audit ──────────────────────────────────────────────────────
    print("=" * 100)
    print("DIRECTION AUDIT -- does prev=LOSS predict which side probes first?")
    print("If long fills and short fills are 50/50, there's NO directional edge;")
    print("OCO is the only way to trade it (random direction).")
    print("=" * 100)

    # At RR=2.0 inside 20%
    all20 = run_all_days(days_df, bars_by_day, 0.20, 2.0)
    pl20  = [r for r in all20 if r["prev_outcome"] == "loss"]
    pw20  = [r for r in all20 if r["prev_outcome"] == "win"]

    for label, group in [("ALL days", all20), ("prev=LOSS days", pl20), ("prev=WIN days", pw20)]:
        if len(group) < 10:
            continue
        longs  = [r for r in group if r["fill_dir"] == "long"]
        shorts = [r for r in group if r["fill_dir"] == "short"]
        n      = len(group)
        pct_l  = len(longs) / n
        # t-test for direction bias (is long fill rate != 50%?)
        arr    = np.array([1 if r["fill_dir"] == "long" else 0 for r in group])
        t, p   = stats.ttest_1samp(arr, 0.5)

        print(f"\n  {label}  (N={n})")
        print(f"    Long fills : {len(longs):3d}  ({pct_l:.1%})  avgR={np.mean([r['pnl_r'] for r in longs]):+.3f}")
        print(f"    Short fills: {len(shorts):3d}  ({1-pct_l:.1%})  avgR={np.mean([r['pnl_r'] for r in shorts]):+.3f}")
        print(f"    Direction bias test: p={p:.4f}  {'(BIASED - one side is more likely)' if p < 0.10 else '(NO BIAS - both sides equally likely)'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("FILL COUNT SUMMARY (RR=2.0) -- how many EXTRA trades does inside entry create?")
    print("=" * 100)
    e0_r = run_all_days(days_df, bars_by_day, 0, 2.0, use_e0_baseline=True)
    print(f"  {'Config':40s}  {'Fills':>6s}  {'WR':>6s}  {'avgR':>8s}  {'p':>8s}")
    arr = np.array([r["pnl_r"] for r in e0_r])
    wr  = sum(1 for r in e0_r if r["outcome"] == "win") / len(e0_r)
    _, p = stats.ttest_1samp(arr, 0)
    print(f"  {'E0 BASELINE (close-confirmed)':40s}  {len(e0_r):6d}  {wr:6.1%}  {arr.mean():+8.4f}  {p:8.4f}")
    for offset in [0.10, 0.15, 0.20, 0.25]:
        res = run_all_days(days_df, bars_by_day, offset, 2.0)
        if not res: continue
        arr = np.array([r["pnl_r"] for r in res])
        wr  = sum(1 for r in res if r["outcome"] == "win") / len(res)
        _, p = stats.ttest_1samp(arr, 0)
        print(f"  {'inside ' + f'{offset:.0%}':40s}  {len(res):6d}  {wr:6.1%}  {arr.mean():+8.4f}  {p:8.4f}")


if __name__ == "__main__":
    main()
