#!/usr/bin/env python
"""
MFE/MAE analysis of MGC Asian session fade strategy.

Hypothesis: After trending from CME_REOPEN through Asian sessions,
MGC reverts. Correlation is proven (r=-0.118, p<0.001). Question:
what's the TAIL structure? Is the R:R distribution favorable even
at 50% win rate?

Measures:
  - MFE (max favorable excursion) of fade trades in points and R
  - MAE (max adverse excursion) of fade trades
  - Strategy simulation at various stop/target combos
  - Year-by-year breakdown

@research-source research_mgc_asian_fade_mfe.py
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.cost_model import get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH


def main():
    cost = get_cost_spec("MGC")
    friction_pts = cost.total_friction / cost.point_value
    print(f"MGC: tick={cost.tick_size}, pt_val={cost.point_value}, friction={cost.total_friction:.2f} (={friction_pts:.3f} pts)")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    trading_days = (
        con.execute("""
            SELECT DISTINCT trading_day FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
            ORDER BY trading_day
        """)
        .fetchdf()["trading_day"]
        .tolist()
    )

    results = []
    skipped = 0

    for td in trading_days:
        td_date = td.date() if hasattr(td, "date") else td
        if td_date.weekday() >= 5:
            continue

        cme_h, cme_m = SESSION_CATALOG["CME_REOPEN"]["resolver"](td_date)
        sing_h, sing_m = SESSION_CATALOG["SINGAPORE_OPEN"]["resolver"](td_date)

        cme_bris = datetime.combine(td_date, time(cme_h, cme_m))
        sing_bris = datetime.combine(td_date, time(sing_h, sing_m))
        peak_bris = sing_bris + timedelta(minutes=30)

        # Full window: through London for reversion tracking
        london_h, london_m = SESSION_CATALOG["LONDON_METALS"]["resolver"](td_date)
        london_bris = datetime.combine(td_date, time(london_h, london_m))
        session_end = london_bris + timedelta(hours=4)

        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m WHERE symbol = 'MGC'
            AND ts_utc >= ? AND ts_utc <= ?
            ORDER BY ts_utc
            """,
            [cme_bris, session_end],
        ).fetchdf()

        if len(bars) < 30:
            skipped += 1
            continue

        bars["ts_bris"] = pd.to_datetime(bars["ts_utc"]).dt.tz_localize(None)

        cme_open = float(bars.iloc[0]["open"])
        peak_bars = bars[bars["ts_bris"] <= peak_bris]
        if len(peak_bars) < 10:
            skipped += 1
            continue

        peak_close = float(peak_bars.iloc[-1]["close"])
        trend_return = (peak_close - cme_open) / cme_open

        # Only trending days (>0.1% move)
        if abs(trend_return) < 0.001:
            continue

        fade_dir = "short" if trend_return > 0 else "long"
        entry_price = peak_close
        trend_points = abs(peak_close - cme_open)

        post_peak = bars[bars["ts_bris"] > peak_bris].copy()
        if len(post_peak) < 5:
            continue

        highs = post_peak["high"].values
        lows = post_peak["low"].values

        # Bar-by-bar MFE/MAE tracking for proper stop/target ordering
        if fade_dir == "short":
            running_mfe = np.maximum.accumulate(entry_price - lows)
            running_mae = np.maximum.accumulate(highs - entry_price)
        else:
            running_mfe = np.maximum.accumulate(highs - entry_price)
            running_mae = np.maximum.accumulate(entry_price - lows)

        mfe_points = float(running_mfe[-1])
        mae_points = float(running_mae[-1])

        # Asian range as risk reference
        asian_bars = bars[bars["ts_bris"] <= peak_bris]
        asian_range = float(asian_bars["high"].max() - asian_bars["low"].min())

        # ORB size at CME_REOPEN
        orb_row = con.execute(
            """
            SELECT orb_CME_REOPEN_size FROM daily_features
            WHERE symbol = 'MGC' AND trading_day = ? AND orb_minutes = 5
            """,
            [td_date],
        ).fetchone()
        orb_size = float(orb_row[0]) if orb_row and orb_row[0] and orb_row[0] > 0 else None

        # Time to peak MFE
        mfe_bar_idx = int(np.argmax(running_mfe))

        # Store bar-by-bar data for stop/target simulation
        results.append({
            "trading_day": td_date,
            "fade_dir": fade_dir,
            "trend_return": trend_return,
            "trend_points": trend_points,
            "entry_price": entry_price,
            "mfe_points": mfe_points,
            "mae_points": mae_points,
            "asian_range": asian_range,
            "orb_size": orb_size,
            "mfe_minutes": mfe_bar_idx,
            "n_post_bars": len(post_peak),
            "running_mfe": running_mfe,
            "running_mae": running_mae,
            # R-multiples
            "mfe_r_asian": mfe_points / asian_range if asian_range > 0 else None,
            "mae_r_asian": mae_points / asian_range if asian_range > 0 else None,
            "mfe_r_trend": mfe_points / trend_points if trend_points > 0 else None,
            "mae_r_trend": mae_points / trend_points if trend_points > 0 else None,
        })

    con.close()

    df = pd.DataFrame(results)
    print(f"\nTotal trending days: {len(df)} (skipped {skipped})")
    print(f"Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")
    print(f"Fade long: {(df['fade_dir'] == 'long').sum()}, short: {(df['fade_dir'] == 'short').sum()}")

    # ── MFE/MAE distributions ──
    print(f"\n{'=' * 70}")
    print("MFE / MAE DISTRIBUTION (points)")
    print(f"{'=' * 70}")
    for col, label in [("mfe_points", "MFE (favorable)"), ("mae_points", "MAE (adverse)")]:
        v = df[col].dropna()
        print(f"\n  {label}:")
        for pct, name in [(0.10, "P10"), (0.25, "P25"), (0.50, "Median"), (0.75, "P75"), (0.90, "P90"), (0.95, "P95")]:
            print(f"    {name:6s}: {v.quantile(pct):7.2f} pts")
        print(f"    Mean:   {v.mean():7.2f} pts")
        print(f"    Max:    {v.max():7.2f} pts")

    print(f"\n{'=' * 70}")
    print("R-MULTIPLES (risk = Asian range)")
    print(f"{'=' * 70}")
    for col, label in [("mfe_r_asian", "MFE-R"), ("mae_r_asian", "MAE-R")]:
        v = df[col].dropna()
        print(f"\n  {label}:")
        for pct, name in [(0.25, "P25"), (0.50, "Median"), (0.75, "P75"), (0.90, "P90"), (0.95, "P95")]:
            print(f"    {name:6s}: {v.quantile(pct):6.3f}R")
        print(f"    Mean:   {v.mean():6.3f}R")

    print(f"\n{'=' * 70}")
    print("R-MULTIPLES (risk = trend size)")
    print(f"{'=' * 70}")
    for col, label in [("mfe_r_trend", "MFE-R"), ("mae_r_trend", "MAE-R")]:
        v = df[col].dropna()
        print(f"\n  {label}:")
        for pct, name in [(0.25, "P25"), (0.50, "Median"), (0.75, "P75"), (0.90, "P90"), (0.95, "P95")]:
            print(f"    {name:6s}: {v.quantile(pct):6.3f}R")
        print(f"    Mean:   {v.mean():6.3f}R")

    # ── MFE vs MAE ratio ──
    print(f"\n{'=' * 70}")
    print("MFE:MAE RATIO (the key question)")
    print(f"{'=' * 70}")
    df["mfe_mae_ratio"] = df["mfe_points"] / df["mae_points"].replace(0, np.nan)
    v = df["mfe_mae_ratio"].dropna()
    print(f"  Median MFE:MAE = {v.median():.2f}x")
    print(f"  Mean MFE:MAE   = {v.mean():.2f}x")
    print(f"  % where MFE > MAE (favorable): {(v > 1).mean():.1%}")
    print(f"  % where MFE > 2x MAE:          {(v > 2).mean():.1%}")
    print(f"  % where MFE > 5x MAE:          {(v > 5).mean():.1%}")
    print(f"  % where MFE > 10x MAE:         {(v > 10).mean():.1%}")

    # ── Strategy simulation with bar-by-bar stop/target ──
    print(f"\n{'=' * 70}")
    print("STRATEGY SIMULATION (bar-by-bar stop/target, risk=Asian range)")
    print(f"Friction: {friction_pts:.3f} pts per trade")
    print(f"{'=' * 70}")

    for stop_r in [0.5, 0.75, 1.0, 1.5]:
        print(f"\n  STOP = {stop_r}R")
        for target_r in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
            wins = 0
            losses = 0
            scratches = 0
            pnl_list = []

            for _, row in df.iterrows():
                ar = row["asian_range"]
                if ar <= 0 or pd.isna(ar):
                    continue

                stop_pts = stop_r * ar
                target_pts = target_r * ar
                rmfe = row["running_mfe"]
                rmae = row["running_mae"]

                # Find first bar where stop or target is hit
                stop_hit = np.where(rmae >= stop_pts)[0]
                target_hit = np.where(rmfe >= target_pts)[0]

                stop_bar = stop_hit[0] if len(stop_hit) > 0 else 9999
                target_bar = target_hit[0] if len(target_hit) > 0 else 9999

                if target_bar < stop_bar:
                    wins += 1
                    pnl_list.append(target_r - friction_pts / ar)
                elif stop_bar < target_bar:
                    losses += 1
                    pnl_list.append(-stop_r - friction_pts / ar)
                else:
                    # Neither hit — scratch. Use last bar close vs entry
                    scratches += 1
                    final_mfe = float(rmfe[-1])
                    final_mae = float(rmae[-1])
                    # Approximate: net = mfe - mae (very rough)
                    pnl_list.append((final_mfe - final_mae) / ar - friction_pts / ar)

            if len(pnl_list) < 30:
                continue

            arr = np.array(pnl_list)
            total = wins + losses
            wr = wins / total if total > 0 else 0
            exp_r = arr.mean()
            sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0

            marker = " ***" if exp_r > 0.05 else ""
            print(
                f"    T={target_r:4.1f}R | N={len(pnl_list):4d} "
                f"W={wins:4d} L={losses:4d} S={scratches:3d} | "
                f"WR={wr:.1%} ExpR={exp_r:+.4f} Sh={sharpe:+.2f}{marker}"
            )

    # ── Year-by-year for promising combos ──
    df["year"] = df["trading_day"].apply(lambda d: d.year)

    for stop_r, target_r in [(1.0, 3.0), (0.75, 5.0), (1.0, 5.0)]:
        print(f"\n{'=' * 70}")
        print(f"YEAR-BY-YEAR: Stop={stop_r}R, Target={target_r}R")
        print(f"{'=' * 70}")

        for yr in sorted(df["year"].unique()):
            yr_df = df[df["year"] == yr]
            wins = 0
            losses = 0
            pnl = []

            for _, row in yr_df.iterrows():
                ar = row["asian_range"]
                if ar <= 0 or pd.isna(ar):
                    continue
                rmfe = row["running_mfe"]
                rmae = row["running_mae"]
                stop_pts = stop_r * ar
                target_pts = target_r * ar

                stop_bar = np.where(rmae >= stop_pts)[0]
                target_bar = np.where(rmfe >= target_pts)[0]
                sb = stop_bar[0] if len(stop_bar) > 0 else 9999
                tb = target_bar[0] if len(target_bar) > 0 else 9999

                if tb < sb:
                    wins += 1
                    pnl.append(target_r - friction_pts / ar)
                elif sb < tb:
                    losses += 1
                    pnl.append(-stop_r - friction_pts / ar)
                else:
                    final_mfe = float(rmfe[-1])
                    final_mae = float(rmae[-1])
                    pnl.append((final_mfe - final_mae) / ar - friction_pts / ar)

            if len(pnl) > 0:
                arr = np.array(pnl)
                total = wins + losses
                wr = wins / total if total > 0 else 0
                marker = "+" if arr.mean() > 0 else "-"
                print(
                    f"  {yr}: N={len(pnl):3d} W={wins:3d} L={losses:3d} "
                    f"WR={wr:.0%} ExpR={arr.mean():+.4f} cumR={arr.sum():+.2f} {marker}"
                )

    # ── VWAP conditional ──
    print(f"\n{'=' * 70}")
    print("VWAP DEVIATION CONDITIONAL (high-dev only)")
    print(f"{'=' * 70}")

    # Recompute VWAP deviation if not stored
    if "vwap_dev" not in df.columns:
        print("  (VWAP deviation not in data — skipped)")
    else:
        med_dev = df["vwap_dev"].abs().median()
        high_dev = df[df["vwap_dev"].abs() > med_dev]
        print(f"  High VWAP deviation days: {len(high_dev)}")
        # TODO: re-run simulation on high_dev subset

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
