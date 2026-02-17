#!/usr/bin/env python3
"""
AUDIT: IB Single Break script -- strip back, verify, rebuild honestly.

This script does 3 things:
  1. VERIFY the original classification is mechanically correct (spot-check trades)
  2. EXPOSE the look-ahead bias by testing REAL-TIME checkpoints
  3. REBUILD with honest, no-look-ahead classification

The honest version: at each checkpoint AFTER IB forms, classify based on
what you know SO FAR (not the final session state). Make the hold/exit
decision with ONLY information available at that moment.

Read-only. No DB writes.

Usage:
    python scripts/audit_ib_single_break.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple
from research._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SESSION_START_UTC_HOUR = 0
IB_MINUTES = 60
MIN_ORB_SIZE = 4.0
RR_TARGET = 2.0
CB = 2
HOLD_HOURS = 7
VOL_MULTIPLIER = 1.30
VOL_LOOKBACK = 10

# Checkpoints AFTER IB: minutes after entry to check state
# Entry is ~7 min into session, IB ends at 60 min.
# So first useful checkpoint is IB_MINUTES from session start = ~53 min after entry.
# We check at these minutes-after-entry marks:
CHECKPOINTS_AFTER_ENTRY = [60, 90, 120, 180, 240]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_ib(bars: pd.DataFrame) -> dict | None:
    """Compute IB from bars. Returns ib_high, ib_low, ib_start, ib_end."""
    ib_start = None
    for _, bar in bars.iterrows():
        ts = bar["ts_utc"]
        if ts.hour == SESSION_START_UTC_HOUR:
            ib_start = ts
            break
    if ib_start is None:
        return None

    ib_end = ib_start + timedelta(minutes=IB_MINUTES)
    ib_bars = bars[(bars["ts_utc"] >= ib_start) & (bars["ts_utc"] < ib_end)]
    if len(ib_bars) < 5:
        return None

    return {
        "ib_high": float(ib_bars["high"].max()),
        "ib_low": float(ib_bars["low"].min()),
        "ib_start": ib_start,
        "ib_end": ib_end,
    }

def classify_at_checkpoint(
    bars: pd.DataFrame,
    ib: dict,
    checkpoint_ts: pd.Timestamp,
) -> str:
    """Classify day state using ONLY bars up to checkpoint_ts.

    Returns: 'single_bull', 'single_bear', 'double_break', 'no_break'
    This is the HONEST version -- no future information.
    """
    ib_high = ib["ib_high"]
    ib_low = ib["ib_low"]
    ib_end = ib["ib_end"]

    # Only look at bars from IB end up to checkpoint
    mask = (bars["ts_utc"] >= ib_end) & (bars["ts_utc"] <= checkpoint_ts)
    scan_bars = bars[mask]

    broke_high = False
    broke_low = False
    for _, bar in scan_bars.iterrows():
        if bar["high"] > ib_high:
            broke_high = True
        if bar["low"] < ib_low:
            broke_low = True
        if broke_high and broke_low:
            break

    if broke_high and not broke_low:
        return "single_bull"
    elif broke_low and not broke_high:
        return "single_bear"
    elif broke_high and broke_low:
        return "double_break"
    else:
        return "no_break"

def classify_final(bars: pd.DataFrame, ib: dict) -> str:
    """Classify using ALL post-IB bars (the look-ahead version)."""
    ib_high = ib["ib_high"]
    ib_low = ib["ib_low"]
    ib_end = ib["ib_end"]

    post_ib = bars[bars["ts_utc"] >= ib_end]
    broke_high = (post_ib["high"] > ib_high).any()
    broke_low = (post_ib["low"] < ib_low).any()

    if broke_high and not broke_low:
        return "single_bull"
    elif broke_low and not broke_high:
        return "single_bear"
    elif broke_high and broke_low:
        return "double_break"
    else:
        return "no_break"

def compute_hold_pnl(
    bars: pd.DataFrame, entry_ts, entry_price: float,
    stop_price: float, is_long: bool, hold_hours: int,
) -> float | None:
    """7h hold with stop. Returns pnl_r."""
    spec = get_cost_spec("MGC")
    cutoff = entry_ts + timedelta(hours=hold_hours)
    entry_mask = bars["ts_utc"] >= entry_ts
    if not entry_mask.any():
        return None
    entry_idx = entry_mask.idxmax()
    last_close = entry_price

    for i in range(entry_idx, len(bars)):
        bar = bars.iloc[i]
        if is_long and bar["low"] <= stop_price:
            return to_r_multiple(spec, entry_price, stop_price, stop_price - entry_price)
        if not is_long and bar["high"] >= stop_price:
            return to_r_multiple(spec, entry_price, stop_price, entry_price - stop_price)
        last_close = bar["close"]
        if bar["ts_utc"] >= cutoff:
            pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
            return to_r_multiple(spec, entry_price, stop_price, pnl)

    pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
    return to_r_multiple(spec, entry_price, stop_price, pnl)

def compute_fixed_exit_pnl(
    bars: pd.DataFrame, entry_ts, entry_price: float,
    stop_price: float, target_price: float, is_long: bool,
) -> float | None:
    """Fixed stop/target exit. Returns pnl_r. Independent recomputation."""
    spec = get_cost_spec("MGC")
    entry_mask = bars["ts_utc"] >= entry_ts
    if not entry_mask.any():
        return None
    entry_idx = entry_mask.idxmax()

    for i in range(entry_idx, len(bars)):
        bar = bars.iloc[i]
        if is_long:
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
        else:
            stop_hit = bar["high"] >= stop_price
            target_hit = bar["low"] <= target_price

        if stop_hit and target_hit:
            pnl = stop_price - entry_price if is_long else entry_price - stop_price
            return to_r_multiple(spec, entry_price, stop_price, pnl)
        if stop_hit:
            pnl = stop_price - entry_price if is_long else entry_price - stop_price
            return to_r_multiple(spec, entry_price, stop_price, pnl)
        if target_hit:
            pnl = target_price - entry_price if is_long else entry_price - target_price
            return to_r_multiple(spec, entry_price, stop_price, pnl)

    return None  # no resolution

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: Path, start: date, end: date):
    con = duckdb.connect(str(db_path), read_only=True)

    print(f"Loading 1000 E1 CB{CB} RR{RR_TARGET} G{MIN_ORB_SIZE}+ trades...")
    df = con.execute("""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.outcome, o.pnl_r,
               d.orb_1000_size, d.orb_1000_break_dir
        FROM orb_outcomes o
        JOIN daily_features d ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = '1000' AND o.entry_model = 'E1'
          AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_1000_size >= ?
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [RR_TARGET, CB, MIN_ORB_SIZE, start, end]).fetchdf()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    print(f"  {len(df)} trades")

    # Load bars
    unique_days = sorted(df["trading_day"].unique())
    bars_cache = {}
    for td in unique_days:
        s, e = compute_trading_day_utc_range(td)
        b = con.execute(
            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
            "WHERE symbol='MGC' AND ts_utc>=? AND ts_utc<? ORDER BY ts_utc",
            [s, e],
        ).fetchdf()
        if not b.empty:
            b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
        bars_cache[td] = b
    con.close()
    print(f"  Bars loaded for {len(bars_cache)} days\n")

    # ===================================================================
    # AUDIT 1: Verify fixed PnL matches orb_outcomes
    # ===================================================================
    print("=" * 90)
    print("AUDIT 1: VERIFY FIXED RR PNL MATCHES ORB_OUTCOMES")
    print("=" * 90)

    mismatch_count = 0
    verified_count = 0
    for _, row in df.head(30).iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue
        is_long = row["orb_1000_break_dir"] == "long"
        recomputed = compute_fixed_exit_pnl(
            bars, row["entry_ts"], row["entry_price"],
            row["stop_price"], row["target_price"], is_long,
        )
        if recomputed is None:
            continue
        stored = row["pnl_r"]
        if abs(recomputed - stored) > 0.05:
            mismatch_count += 1
            print(f"  MISMATCH {td}: stored={stored:+.4f} recomputed={recomputed:+.4f} "
                  f"delta={recomputed-stored:+.4f}")
        verified_count += 1

    print(f"  Verified {verified_count} trades, {mismatch_count} mismatches")
    if mismatch_count == 0:
        print("  PASS: Fixed RR PnL recomputation matches stored values")
    else:
        print("  WARNING: PnL mismatches detected -- investigate before trusting results")

    # ===================================================================
    # AUDIT 2: LOOK-AHEAD BIAS EXPOSURE
    # ===================================================================
    print("\n" + "=" * 90)
    print("AUDIT 2: LOOK-AHEAD BIAS -- CLASSIFICATION DRIFT OVER TIME")
    print("=" * 90)
    print("How often does the classification CHANGE between checkpoints?")
    print("If it changes a lot, the original 'final' classification was look-ahead.\n")

    results = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue

        ib = compute_ib(bars)
        if ib is None:
            continue

        entry_ts = row["entry_ts"]
        entry_p = row["entry_price"]
        stop_p = row["stop_price"]
        is_long = row["orb_1000_break_dir"] == "long"

        # Final classification (look-ahead)
        final_class = classify_final(bars, ib)

        # Checkpoint classifications (honest, real-time)
        cp_classes = {}
        for cp_min in CHECKPOINTS_AFTER_ENTRY:
            cp_ts = entry_ts + timedelta(minutes=cp_min)
            cp_classes[cp_min] = classify_at_checkpoint(bars, ib, cp_ts)

        # 7h hold PnL
        hold_pnl = compute_hold_pnl(bars, entry_ts, entry_p, stop_p, is_long, HOLD_HOURS)
        if hold_pnl is None:
            continue

        year = str(td.year) if hasattr(td, "year") else str(td)[:4]

        rec = {
            "td": td,
            "year": year,
            "is_long": is_long,
            "final_class": final_class,
            "fixed_pnl_r": row["pnl_r"],
            "hold_pnl_r": hold_pnl,
        }
        for cp_min in CHECKPOINTS_AFTER_ENTRY:
            rec[f"class_{cp_min}m"] = cp_classes[cp_min]
        results.append(rec)

    pdf = pd.DataFrame(results)
    print(f"Classified {len(pdf)} trades\n")

    # Show classification drift
    for cp_min in CHECKPOINTS_AFTER_ENTRY:
        col = f"class_{cp_min}m"
        matches_final = (pdf[col] == pdf["final_class"]).sum()
        pct = matches_final / len(pdf) * 100
        # Breakdown
        single_at_cp = pdf[col].isin(["single_bull", "single_bear"]).sum()
        double_at_cp = (pdf[col] == "double_break").sum()
        no_break_at_cp = (pdf[col] == "no_break").sum()
        print(f"  At {cp_min:>3d}m after entry: "
              f"single={single_at_cp:>3d} double={double_at_cp:>3d} no_break={no_break_at_cp:>3d} | "
              f"matches final={matches_final}/{len(pdf)} ({pct:.0f}%)")

    # Final distribution for reference
    print(f"\n  Final (look-ahead): "
          f"single={pdf['final_class'].isin(['single_bull','single_bear']).sum()} "
          f"double={(pdf['final_class']=='double_break').sum()} "
          f"no_break={(pdf['final_class']=='no_break').sum()}")

    # ===================================================================
    # AUDIT 3: HONEST BLENDED STRATEGY AT EACH CHECKPOINT
    # ===================================================================
    print("\n" + "=" * 90)
    print("AUDIT 3: HONEST BLENDED STRATEGY (NO LOOK-AHEAD)")
    print("=" * 90)
    print("At checkpoint N, use classification known SO FAR.")
    print("Single at checkpoint -> hold 7h. Double/no_break -> fixed RR.\n")

    print(f"  {'Checkpoint':12s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Control: fixed RR
    mf = compute_strategy_metrics(pdf["fixed_pnl_r"].values)
    if mf:
        print(f"  {'Fixed RR':12s} {mf['n']:>5d} {mf['wr']:>7.3f} {mf['expr']:>8.4f} "
              f"{mf['sharpe']:>8.4f} {mf['maxdd']:>8.2f} {mf['total']:>8.1f}")

    # Control: pure 7h
    m7 = compute_strategy_metrics(pdf["hold_pnl_r"].values)
    if m7:
        print(f"  {'7h hold':12s} {m7['n']:>5d} {m7['wr']:>7.3f} {m7['expr']:>8.4f} "
              f"{m7['sharpe']:>8.4f} {m7['maxdd']:>8.2f} {m7['total']:>8.1f}")

    # Look-ahead blended (for comparison -- the DISHONEST number)
    la_single = pdf["final_class"].isin(["single_bull", "single_bear"])
    la_blended = np.where(la_single, pdf["hold_pnl_r"], pdf["fixed_pnl_r"])
    mla = compute_strategy_metrics(la_blended)
    if mla:
        print(f"  {'LookAhead*':12s} {mla['n']:>5d} {mla['wr']:>7.3f} {mla['expr']:>8.4f} "
              f"{mla['sharpe']:>8.4f} {mla['maxdd']:>8.2f} {mla['total']:>8.1f}")

    print()

    # Honest checkpoint blends
    for cp_min in CHECKPOINTS_AFTER_ENTRY:
        col = f"class_{cp_min}m"
        is_single = pdf[col].isin(["single_bull", "single_bear"])
        blended = np.where(is_single, pdf["hold_pnl_r"], pdf["fixed_pnl_r"])
        m = compute_strategy_metrics(blended)
        if m:
            n_single = is_single.sum()
            print(f"  {f'{cp_min}m honest':12s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
                  f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}  "
                  f"(single={n_single})")

    # ===================================================================
    # AUDIT 4: DIRECTION ALIGNMENT DEEP DIVE
    # ===================================================================
    print("\n" + "=" * 90)
    print("AUDIT 4: DIRECTION ALIGNMENT (is 100% loss on opposed REAL?)")
    print("=" * 90)

    for cp_min in [120, 180]:
        col = f"class_{cp_min}m"
        print(f"\n  --- At {cp_min}m checkpoint ---")

        for dt in ["single_bull", "single_bear"]:
            sub = pdf[pdf[col] == dt]
            if len(sub) < 2:
                continue

            if dt == "single_bull":
                aligned = sub[sub["is_long"]]
                opposed = sub[~sub["is_long"]]
            else:
                aligned = sub[~sub["is_long"]]
                opposed = sub[sub["is_long"]]

            for label, ss in [("Aligned", aligned), ("Opposed", opposed)]:
                if len(ss) < 2:
                    print(f"    {dt} {label:8s}: N={len(ss)} (too few)")
                    continue
                m = compute_strategy_metrics(ss["hold_pnl_r"].values)
                if m:
                    print(f"    {dt} {label:8s}: N={m['n']:>3d}, "
                          f"ExpR={m['expr']:+.4f}, Sharpe={m['sharpe']:.4f}, "
                          f"WR={m['wr']:.3f}")

    # ===================================================================
    # AUDIT 5: YEARLY STABILITY OF HONEST BLENDED
    # ===================================================================
    print("\n" + "=" * 90)
    print("AUDIT 5: YEARLY STABILITY OF HONEST 120m BLENDED")
    print("=" * 90)

    best_cp = 120
    col = f"class_{best_cp}m"
    is_single = pdf[col].isin(["single_bull", "single_bear"])
    pdf["honest_blended"] = np.where(is_single, pdf["hold_pnl_r"], pdf["fixed_pnl_r"])

    print(f"  {'Year':5s} {'N':>4s} {'Single':>7s} {'Double':>7s} {'Fixed':>8s} {'7h':>8s} {'Honest':>8s} {'Best':>8s}")
    print(f"  {'-'*5} {'-'*4} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for year in sorted(pdf["year"].unique()):
        ydf = pdf[pdf["year"] == year]
        if len(ydf) < 3:
            continue
        n_single = is_single[ydf.index].sum()
        n_double = len(ydf) - n_single

        mf = compute_strategy_metrics(ydf["fixed_pnl_r"].values)
        m7 = compute_strategy_metrics(ydf["hold_pnl_r"].values)
        mh = compute_strategy_metrics(ydf["honest_blended"].values)

        if mf and m7 and mh:
            sharpes = {"fixed": mf["sharpe"], "7h": m7["sharpe"], "honest": mh["sharpe"]}
            best = max(sharpes, key=sharpes.get)
            print(f"  {year:5s} {len(ydf):>4d} {n_single:>7d} {n_double:>7d} "
                  f"{mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
                  f"{mh['sharpe']:>8.3f} {best:>8s}")

    mf = compute_strategy_metrics(pdf["fixed_pnl_r"].values)
    m7 = compute_strategy_metrics(pdf["hold_pnl_r"].values)
    mh = compute_strategy_metrics(pdf["honest_blended"].values)
    if mf and m7 and mh:
        n_s = is_single.sum()
        n_d = len(pdf) - n_s
        print(f"  {'TOTAL':5s} {len(pdf):>4d} {n_s:>7d} {n_d:>7d} "
              f"{mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
              f"{mh['sharpe']:>8.3f}")

    # ===================================================================
    # AUDIT 6: SPOT-CHECK INDIVIDUAL TRADES
    # ===================================================================
    print("\n" + "=" * 90)
    print("AUDIT 6: SPOT-CHECK -- 5 SINGLE BREAK TRADES (hold 7h)")
    print("=" * 90)

    single_trades = pdf[pdf[f"class_{best_cp}m"].isin(["single_bull", "single_bear"])].head(5)
    for _, t in single_trades.iterrows():
        print(f"  {t['td']} | {t[f'class_{best_cp}m']:12s} | "
              f"final={t['final_class']:12s} | "
              f"long={t['is_long']} | "
              f"fixed={t['fixed_pnl_r']:+.3f} | "
              f"7h={t['hold_pnl_r']:+.3f}")

    print("\n  5 DOUBLE BREAK TRADES (fixed RR):")
    double_trades = pdf[pdf[f"class_{best_cp}m"] == "double_break"].head(5)
    for _, t in double_trades.iterrows():
        print(f"  {t['td']} | {t[f'class_{best_cp}m']:12s} | "
              f"final={t['final_class']:12s} | "
              f"long={t['is_long']} | "
              f"fixed={t['fixed_pnl_r']:+.3f} | "
              f"7h={t['hold_pnl_r']:+.3f}")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print("\n" + "=" * 90)
    print("AUDIT VERDICT")
    print("=" * 90)
    if mla and mh:
        delta_sharpe = mla["sharpe"] - mh["sharpe"]
        print(f"  Look-ahead Sharpe:  {mla['sharpe']:.4f}")
        print(f"  Honest 120m Sharpe: {mh['sharpe']:.4f}")
        print(f"  Look-ahead inflation: {delta_sharpe:+.4f} ({delta_sharpe/mh['sharpe']*100:+.0f}%)" if mh["sharpe"] != 0 else "")
        if mh["sharpe"] > mf["sharpe"]:
            print(f"  Honest blended STILL beats fixed RR ({mh['sharpe']:.4f} vs {mf['sharpe']:.4f})")
            print(f"  SIGNAL SURVIVES AUDIT (with reduced magnitude)")
        else:
            print(f"  Honest blended does NOT beat fixed RR")
            print(f"  SIGNAL IS LOOK-AHEAD ARTIFACT -- KILL IT")

    print()

def main():
    parser = argparse.ArgumentParser(description="Audit IB Single Break")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()
    run(args.db_path, args.start, args.end)

if __name__ == "__main__":
    main()
