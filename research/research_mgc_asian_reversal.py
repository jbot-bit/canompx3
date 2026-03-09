#!/usr/bin/env python
"""
Test mean-reversion hypothesis: MGC trends during Asian morning,
peaks around SINGAPORE_OPEN+30min, then reverts toward CME_REOPEN open / VWAP.

Hypothesis: After trending from CME_REOPEN through TOKYO/BRISBANE/SINGAPORE,
MGC price reverses. Measured as negative correlation between the
trend return (CME_REOPEN → peak) and subsequent return (peak → exit).

Statistical tests:
  - Pearson correlation between trend and post-peak returns
  - One-sample t-test on "fade" strategy returns (short if up, long if down)
  - Win rate with binomial exact test vs 50%

@research-source research_mgc_asian_reversal.py
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from zoneinfo import ZoneInfo

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH

_BRISBANE = ZoneInfo("Australia/Brisbane")


def brisbane_to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=_BRISBANE).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def compute_vwap(bars: pd.DataFrame) -> float:
    """Volume-weighted average price from 1m bars."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"].replace(0, np.nan)
    if vol.sum() == 0:
        return bars["close"].mean()
    return float((typical * vol).sum() / vol.sum())


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # All MGC trading days
    trading_days = (
        con.execute(
            """
        SELECT DISTINCT trading_day
        FROM daily_features
        WHERE symbol = 'MGC' AND orb_minutes = 5
        ORDER BY trading_day
    """
        )
        .fetchdf()["trading_day"]
        .tolist()
    )

    results = []
    skipped = 0

    for td in trading_days:
        td_date = td.date() if hasattr(td, "date") else td

        if td_date.weekday() >= 5:
            continue

        # Session times (Brisbane) for this specific day
        cme_h, cme_m = SESSION_CATALOG["CME_REOPEN"]["resolver"](td_date)
        sing_h, sing_m = SESSION_CATALOG["SINGAPORE_OPEN"]["resolver"](td_date)

        cme_brisbane = datetime.combine(td_date, time(cme_h, cme_m))
        sing_brisbane = datetime.combine(td_date, time(sing_h, sing_m))
        peak_brisbane = sing_brisbane + timedelta(minutes=30)

        # Multiple peak windows to avoid anchoring on one time
        peak_windows = {
            "sing_open": sing_brisbane,
            "sing_plus30": peak_brisbane,
            "sing_plus60": sing_brisbane + timedelta(minutes=60),
        }

        # Exit horizons after peak
        exit_offsets = {
            "exit_60m": timedelta(minutes=60),
            "exit_120m": timedelta(minutes=120),
            "exit_180m": timedelta(minutes=180),
        }

        # Query window: CME_REOPEN to latest possible exit
        latest_exit = max(
            pw + max(exit_offsets.values())
            for pw in peak_windows.values()
        )
        cme_utc = brisbane_to_utc(cme_brisbane)
        end_utc = brisbane_to_utc(latest_exit)

        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ? AND ts_utc <= ?
            ORDER BY ts_utc
        """,
            [cme_utc, end_utc],
        ).fetchdf()

        if len(bars) < 30:
            skipped += 1
            continue

        bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True).dt.tz_localize(None)
        cme_open = float(bars.iloc[0]["open"])

        row: dict = {"trading_day": td_date, "cme_open": cme_open}

        for pw_name, pw_brisbane in peak_windows.items():
            pw_utc = brisbane_to_utc(pw_brisbane)
            pw_bars = bars[bars["ts_utc"] <= pw_utc]
            if len(pw_bars) < 10:
                continue

            peak_close = float(pw_bars.iloc[-1]["close"])
            trend_return = (peak_close - cme_open) / cme_open
            vwap = compute_vwap(pw_bars)
            vwap_dev = (peak_close - vwap) / vwap  # How far from VWAP at peak

            row[f"{pw_name}_trend"] = trend_return
            row[f"{pw_name}_peak"] = peak_close
            row[f"{pw_name}_vwap"] = vwap
            row[f"{pw_name}_vwap_dev"] = vwap_dev

            for ex_name, ex_offset in exit_offsets.items():
                ex_utc = brisbane_to_utc(pw_brisbane + ex_offset)
                ex_bars = bars[bars["ts_utc"] <= ex_utc]
                if len(ex_bars) == 0:
                    continue
                exit_close = float(ex_bars.iloc[-1]["close"])
                reversion = (exit_close - peak_close) / peak_close
                # Also measure reversion toward VWAP
                vwap_reversion = (exit_close - peak_close) / (vwap - peak_close) if abs(vwap - peak_close) > 0.01 else np.nan
                row[f"{pw_name}_{ex_name}_ret"] = reversion
                row[f"{pw_name}_{ex_name}_vwap_rev"] = vwap_reversion

        results.append(row)

    con.close()

    df = pd.DataFrame(results)
    print(f"Total trading days: {len(df)} (skipped {skipped} for thin data)")
    print(f"Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")
    print()

    # ── Test each peak window × exit horizon ─────────────────────────────
    for pw_name in ["sing_open", "sing_plus30", "sing_plus60"]:
        trend_col = f"{pw_name}_trend"
        if trend_col not in df.columns:
            continue

        print(f"{'=' * 60}")
        print(f"PEAK WINDOW: {pw_name}")
        print(f"{'=' * 60}")

        valid_trend = df.dropna(subset=[trend_col])
        trending = valid_trend[valid_trend[trend_col].abs() > 0.001]
        print(f"  Days with data: {len(valid_trend)}, trending (>0.1%): {len(trending)}")
        print(f"  Mean trend: {valid_trend[trend_col].mean():.5f}")
        print(f"  Trend std:  {valid_trend[trend_col].std():.5f}")
        print()

        for ex_name in ["exit_60m", "exit_120m", "exit_180m"]:
            ret_col = f"{pw_name}_{ex_name}_ret"
            if ret_col not in df.columns:
                continue

            valid = trending.dropna(subset=[ret_col])
            if len(valid) < 30:
                print(f"  {ex_name}: insufficient data ({len(valid)} days)")
                continue

            trend_vals = valid[trend_col].values
            ret_vals = valid[ret_col].values

            # 1. Correlation test
            if np.std(ret_vals) < 1e-12 or np.std(trend_vals) < 1e-12:
                print(f"  {ex_name}: constant input, skipping")
                continue
            corr, p_corr = stats.pearsonr(trend_vals, ret_vals)

            # 2. Fade strategy: take opposite position to trend
            fade_rets = -np.sign(trend_vals) * ret_vals
            mean_fade = fade_rets.mean()
            t_stat, p_t = stats.ttest_1samp(fade_rets, 0)
            win_rate = (fade_rets > 0).mean()
            n_wins = int((fade_rets > 0).sum())
            n_total = len(fade_rets)

            # 3. Binomial test on win rate
            binom_p = stats.binomtest(n_wins, n_total, 0.5).pvalue

            # 4. Year-by-year breakdown
            valid_copy = valid.copy()
            valid_copy["year"] = valid_copy["trading_day"].apply(
                lambda d: d.year if isinstance(d, date) else d.year
            )
            valid_copy["fade_ret"] = fade_rets

            print(f"  --- {ex_name} (N={n_total}) ---")
            print(f"    Corr(trend, post-peak): {corr:+.4f}  p={p_corr:.4f}")
            print(f"    Fade mean return:       {mean_fade:+.6f}  t={t_stat:.3f}  p={p_t:.4f}")
            print(f"    Win rate:               {win_rate:.1%} ({n_wins}/{n_total})  binom p={binom_p:.4f}")

            # Year breakdown
            yr_summary = (
                valid_copy.groupby("year")
                .agg(
                    n=("fade_ret", "count"),
                    mean_ret=("fade_ret", "mean"),
                    win_rate=("fade_ret", lambda x: (x > 0).mean()),
                )
                .reset_index()
            )
            print(f"    Year breakdown:")
            for _, yr in yr_summary.iterrows():
                star = "+" if yr["mean_ret"] > 0 else "-"
                print(
                    f"      {int(yr['year'])}: N={int(yr['n']):3d}  "
                    f"mean={yr['mean_ret']:+.6f}  WR={yr['win_rate']:.0%} {star}"
                )
            print()

    # ── VWAP deviation analysis ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VWAP DEVIATION AT PEAK (sing_plus30)")
    print(f"{'=' * 60}")
    vwap_col = "sing_plus30_vwap_dev"
    if vwap_col in df.columns:
        valid_vwap = df.dropna(subset=[vwap_col])
        print(f"  Mean VWAP deviation at peak: {valid_vwap[vwap_col].mean():+.5f}")
        print(f"  Std VWAP deviation:          {valid_vwap[vwap_col].std():.5f}")

        # Do bigger deviations predict stronger reversions?
        ret_col = "sing_plus30_exit_120m_ret"
        if ret_col in valid_vwap.columns:
            v2 = valid_vwap.dropna(subset=[ret_col])
            # Split into high/low deviation
            med_dev = v2[vwap_col].abs().median()
            high_dev = v2[v2[vwap_col].abs() > med_dev]
            low_dev = v2[v2[vwap_col].abs() <= med_dev]

            if len(high_dev) > 20 and len(low_dev) > 20:
                high_fade = (-np.sign(high_dev["sing_plus30_trend"]) * high_dev[ret_col]).values
                low_fade = (-np.sign(low_dev["sing_plus30_trend"]) * low_dev[ret_col]).values

                print(f"\n  High VWAP deviation (>{med_dev:.5f}):")
                print(f"    N={len(high_fade)}, fade mean={high_fade.mean():+.6f}, WR={(high_fade > 0).mean():.1%}")
                print(f"  Low VWAP deviation (<={med_dev:.5f}):")
                print(f"    N={len(low_fade)}, fade mean={low_fade.mean():+.6f}, WR={(low_fade > 0).mean():.1%}")

                # Mann-Whitney test for difference
                u_stat, p_mw = stats.mannwhitneyu(high_fade, low_fade, alternative="greater")
                print(f"  Mann-Whitney (high > low): U={u_stat:.0f}, p={p_mw:.4f}")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
