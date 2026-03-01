#!/usr/bin/env python3
"""
Exhaustive Session Discovery — full-grid 5-minute ORB time scan.

Tests 5-minute ORBs starting at every 5-minute mark (288 candidates) across
the 24-hour CME futures trading day for all 4 active instruments.

Grid per candidate time:
  - RR targets: 1.0, 1.5, 2.0, 2.5, 3.0
  - G-filters: G4 (>=4pt), G5 (>=5pt), G6 (>=6pt)
  - Entry: break-close (first 1m close outside ORB — matches E2 mechanics)
  - Cost-adjusted: COST_SPECS friction subtracted from each R-multiple

Statistical framework:
  - One-sample t-test (H0: mean pnl_r <= 0) per combo
  - BH FDR at q=0.05 across ALL tested combos simultaneously
  - Year-by-year stability for any FDR survivor
  - DST winter/summer split
  - Comparison to existing 10 sessions

Total combos: 4 instruments x 288 times x 5 RR x 3 G = 17,280 raw
After N>=30 filtering: ~5,000-10,000 effective tests

Predecessor: research/research_orb_time_scan.py (96 times, 1 RR, no costs, no FDR)

Usage:
  python research/research_session_discovery.py
"""
import sys
import time as _time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH

# =========================================================================
# Constants
# =========================================================================

# 288 candidate times: every 5 minutes from 00:00 to 23:55 Brisbane
CANDIDATE_TIMES = [(h, m) for h in range(24) for m in range(0, 60, 5)]

RR_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0]
G_THRESHOLDS = {"G4": 4.0, "G5": 5.0, "G6": 6.0}
G_LIST = sorted(G_THRESHOLDS.items(), key=lambda x: x[1])  # (name, min_size) ascending

ORB_BARS = 5          # 5-minute ORB
BREAK_WINDOW = 240    # 4 hours to detect break (minutes)
OUTCOME_WINDOW = 480  # 8 hours to resolve target/stop (minutes)
MIN_TRADES = 30       # Minimum for statistical testing
FDR_Q = 0.05          # BH FDR threshold

_US_EASTERN = ZoneInfo("America/New_York")

# Known session approximate Brisbane times for labeling.
# Each entry: list of (hour, minute) — includes both winter and summer DST variants.
KNOWN_SESSIONS_BRIS = {
    "CME_REOPEN":     [(9, 0), (8, 0)],
    "TOKYO_OPEN":     [(10, 0)],
    "SINGAPORE_OPEN": [(11, 0)],
    "LONDON_METALS":  [(18, 0), (17, 0)],
    "US_DATA_830":    [(23, 30), (22, 30)],
    "NYSE_OPEN":      [(0, 30)],
    "US_DATA_1000":   [(1, 0), (0, 0)],
    "COMEX_SETTLE":   [(4, 30), (3, 30)],
    "CME_PRECLOSE":   [(6, 45), (5, 45)],
    "NYSE_CLOSE":     [(7, 0), (6, 0)],
}


def nearest_session(bris_h, bris_m, max_dist=10):
    """Find nearest known session within max_dist minutes, or empty string."""
    cand = bris_h * 60 + bris_m
    best_name, best_dist = "", 999
    for name, times in KNOWN_SESSIONS_BRIS.items():
        for sh, sm in times:
            sess = sh * 60 + sm
            dist = min(abs(cand - sess), abs(cand - sess + 1440), abs(cand - sess - 1440))
            if dist < best_dist:
                best_dist = dist
                best_name = name
    if best_dist <= max_dist:
        return best_name
    return ""


def session_distance_min(bris_h, bris_m):
    """Distance in minutes to the nearest known session."""
    cand = bris_h * 60 + bris_m
    best = 999
    for times in KNOWN_SESSIONS_BRIS.values():
        for sh, sm in times:
            sess = sh * 60 + sm
            dist = min(abs(cand - sess), abs(cand - sess + 1440), abs(cand - sess - 1440))
            if dist < best:
                best = dist
    return best


# =========================================================================
# Data Loading (proven pattern from research_orb_time_scan.py)
# =========================================================================

def load_bars(con, instrument):
    """Load all 1m bars for an instrument."""
    return con.execute("""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
    """, [instrument]).fetchdf()


def build_day_arrays(bars_df):
    """Convert bars into 2D numpy arrays indexed by (day, minute_offset).

    minute_offset: 0 = 09:00 Brisbane, 1439 = 08:59 next day.
    Returns (all_days, opens, highs, lows, closes, volumes).
    """
    df = bars_df.copy()
    df["bris_dt"] = df["ts_utc"] + pd.Timedelta(hours=10)
    df["bris_hour"] = df["bris_dt"].dt.hour
    df["bris_minute"] = df["bris_dt"].dt.minute

    # Trading day boundary: 09:00 Brisbane
    df["trading_day"] = df["bris_dt"].dt.normalize()
    mask = df["bris_hour"] < 9
    df.loc[mask, "trading_day"] -= pd.Timedelta(days=1)
    df["trading_day"] = df["trading_day"].dt.date

    # Minute offset from 09:00 Brisbane (0=09:00 .. 1439=08:59 next day)
    df["min_offset"] = ((df["bris_hour"] - 9) % 24) * 60 + df["bris_minute"]

    all_days = sorted(df["trading_day"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_days)}
    n_days = len(all_days)

    opens = np.full((n_days, 1440), np.nan)
    highs = np.full((n_days, 1440), np.nan)
    lows = np.full((n_days, 1440), np.nan)
    closes = np.full((n_days, 1440), np.nan)
    volumes = np.full((n_days, 1440), np.nan)

    idx_d = df["trading_day"].map(day_to_idx).values
    idx_m = df["min_offset"].values

    opens[idx_d, idx_m] = df["open"].values
    highs[idx_d, idx_m] = df["high"].values
    lows[idx_d, idx_m] = df["low"].values
    closes[idx_d, idx_m] = df["close"].values
    volumes[idx_d, idx_m] = df["volume"].values

    return all_days, opens, highs, lows, closes, volumes


def build_dst_mask(trading_days):
    """Boolean array: True if US is in DST (EDT) on that trading day."""
    mask = np.zeros(len(trading_days), dtype=bool)
    for i, td in enumerate(trading_days):
        dt = datetime(td.year, td.month, td.day, 12, 0, 0, tzinfo=_US_EASTERN)
        mask[i] = dt.utcoffset().total_seconds() == -4 * 3600
    return mask


# =========================================================================
# Core Scan
# =========================================================================

def scan_time(bris_h, bris_m, highs, lows, closes, volumes,
              all_days, dst_mask, cost_spec):
    """Scan one candidate ORB start time across all trading days.

    Returns list of trade dicts. Each trade contains outcomes for ALL RR
    targets. G-filter is applied during aggregation (we record orb_size).
    """
    start_min = ((bris_h - 9) % 24) * 60 + bris_m
    orb_end_min = start_min + ORB_BARS

    if orb_end_min >= 1440:
        return []  # ORB wraps past trading day boundary

    orb_mins = list(range(start_min, orb_end_min))

    # Vectorized ORB computation across all days
    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m] for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])
    orb_v = np.column_stack([volumes[:, m] for m in orb_mins])

    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high = np.nanmax(orb_h, axis=1)
    orb_low = np.nanmin(orb_l, axis=1)
    orb_size = orb_high - orb_low
    avg_vol = np.nanmean(orb_v, axis=1)  # avg volume during ORB window

    # Filter: valid ORB + minimum size >= smallest G-filter + nonzero range
    min_g = G_LIST[0][1]  # smallest threshold (G4=4.0)
    active_mask = valid_orb & (orb_size >= min_g) & (orb_size > 0)
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        return []

    break_start = orb_end_min
    max_break_min = min(break_start + BREAK_WINDOW, 1440)

    friction = cost_spec.total_friction
    point_value = cost_spec.point_value

    trades = []

    for day_idx in active_indices:
        oh = orb_high[day_idx]
        ol = orb_low[day_idx]
        os_val = orb_size[day_idx]

        # --- Break detection: first 1m close outside ORB ---
        break_dir = None
        entry = np.nan
        break_at = -1

        for m in range(break_start, max_break_min):
            c = closes[day_idx, m]
            if np.isnan(c):
                continue
            if c > oh:
                break_dir = "long"
                entry = c
                break_at = m
                break
            elif c < ol:
                break_dir = "short"
                entry = c
                break_at = m
                break

        if break_dir is None:
            continue

        # --- Pre-compute targets ---
        stop = ol if break_dir == "long" else oh
        targets = {}
        for rr in RR_TARGETS:
            if break_dir == "long":
                targets[rr] = entry + rr * os_val
            else:
                targets[rr] = entry - rr * os_val

        # --- Single-pass outcome resolution for all RR targets ---
        max_outcome_min = min(break_at + 1 + OUTCOME_WINDOW, 1440)

        stop_hit_at = None    # scan index where stop was first hit
        target_hit_at = {rr: None for rr in RR_TARGETS}
        last_close = entry

        for scan_idx, m in enumerate(range(break_at + 1, max_outcome_min)):
            h_val = highs[day_idx, m]
            l_val = lows[day_idx, m]
            c_val = closes[day_idx, m]

            if np.isnan(c_val):
                continue
            last_close = c_val

            # Check stop (once — earliest hit)
            if stop_hit_at is None:
                if break_dir == "long" and l_val <= stop:
                    stop_hit_at = scan_idx
                elif break_dir == "short" and h_val >= stop:
                    stop_hit_at = scan_idx

            # Check each target (earliest hit per RR)
            for rr in RR_TARGETS:
                if target_hit_at[rr] is None:
                    if break_dir == "long" and h_val >= targets[rr]:
                        target_hit_at[rr] = scan_idx
                    elif break_dir == "short" and l_val <= targets[rr]:
                        target_hit_at[rr] = scan_idx

            # Early exit if everything resolved
            if stop_hit_at is not None and all(
                t is not None for t in target_hit_at.values()
            ):
                break

        # --- Determine outcome per RR ---
        cost_r = friction / (os_val * point_value) if (os_val * point_value) > 0 else 0

        outcomes = {}
        for rr in RR_TARGETS:
            sb = stop_hit_at
            tb = target_hit_at[rr]

            if sb is not None and tb is not None:
                if sb <= tb:
                    # Stop hit first or same bar (Gate C: ambiguous = loss)
                    raw_r = -1.0
                else:
                    raw_r = rr
            elif sb is not None:
                raw_r = -1.0
            elif tb is not None:
                raw_r = rr
            else:
                # Timeout: mark-to-market
                if break_dir == "long":
                    raw_r = (last_close - entry) / os_val
                else:
                    raw_r = (entry - last_close) / os_val

            outcomes[rr] = raw_r - cost_r  # cost-adjusted

        trades.append({
            "day_idx": day_idx,
            "year": all_days[day_idx].year,
            "orb_size": os_val,
            "break_dir": break_dir,
            "is_dst": bool(dst_mask[day_idx]),
            "avg_vol": float(avg_vol[day_idx]),
            "outcomes": outcomes,
        })

    return trades


# =========================================================================
# Statistical Analysis
# =========================================================================

def compute_combo_stats(trades, rr, g_min, all_days):
    """Compute stats for one (rr, g_filter) combo from trade list."""
    filtered = [t for t in trades if t["orb_size"] >= g_min]
    n = len(filtered)
    if n < MIN_TRADES:
        return None

    pnl = np.array([t["outcomes"][rr] for t in filtered])
    mean_r = float(pnl.mean())
    std_r = float(pnl.std(ddof=1))

    # One-sample t-test (H0: mean <= 0), one-sided
    if std_r > 0 and n > 1:
        t_stat = mean_r / (std_r / np.sqrt(n))
        p_value = float(scipy_stats.t.sf(t_stat, df=n - 1))
    else:
        t_stat = 0.0
        p_value = 1.0

    win_rate = float((pnl > 0).mean())
    total_r = float(pnl.sum())

    # Year-by-year
    by_year = defaultdict(list)
    for t in filtered:
        by_year[t["year"]].append(t["outcomes"][rr])
    years_positive = sum(1 for vals in by_year.values() if np.mean(vals) > 0)
    years_total = len(by_year)

    # Sharpe annualized by actual trade frequency
    day_indices = [t["day_idx"] for t in filtered]
    first_day = all_days[min(day_indices)]
    last_day = all_days[max(day_indices)]
    span_days = max((last_day - first_day).days, 1)
    years_span = max(span_days / 365.25, 0.25)
    trades_per_year = n / years_span
    sharpe_per_trade = mean_r / std_r if std_r > 0 else 0
    sharpe_ann = sharpe_per_trade * np.sqrt(trades_per_year)

    # DST split
    winter_pnl = [t["outcomes"][rr] for t in filtered if not t["is_dst"]]
    summer_pnl = [t["outcomes"][rr] for t in filtered if t["is_dst"]]

    avg_orb = float(np.mean([t["orb_size"] for t in filtered]))
    avg_vol = float(np.mean([t["avg_vol"] for t in filtered]))

    return {
        "n": n,
        "mean_r": round(mean_r, 6),
        "std_r": round(std_r, 6),
        "total_r": round(total_r, 2),
        "win_rate": round(win_rate, 4),
        "t_stat": round(t_stat, 4),
        "p_value": p_value,
        "sharpe_ann": round(sharpe_ann, 4),
        "trades_per_year": round(trades_per_year, 1),
        "years_pos": years_positive,
        "years_total": years_total,
        "avg_orb_size": round(avg_orb, 2),
        "avg_vol": round(avg_vol, 1),
        "n_winter": len(winter_pnl),
        "avg_r_winter": round(float(np.mean(winter_pnl)), 6) if winter_pnl else np.nan,
        "n_summer": len(summer_pnl),
        "avg_r_summer": round(float(np.mean(summer_pnl)), 6) if summer_pnl else np.nan,
    }


def apply_bh_fdr(results, q=FDR_Q):
    """Apply Benjamini-Hochberg FDR correction to results list.

    Modifies dicts in-place: adds 'p_bh' and 'fdr_significant'.
    Returns count of significant results.
    """
    # Collect (index, p_value) for non-None results
    valid = [(i, r["p_value"]) for i, r in enumerate(results) if r is not None]
    if not valid:
        return 0

    valid.sort(key=lambda x: x[1])
    m = len(valid)

    # Compute BH-adjusted p-values (step-up with monotonicity enforcement)
    adjusted = [0.0] * m
    adjusted[m - 1] = valid[m - 1][1]
    for k in range(m - 2, -1, -1):
        _, p = valid[k]
        rank = k + 1
        raw_adj = p * m / rank
        adjusted[k] = min(raw_adj, adjusted[k + 1])

    n_sig = 0
    for k in range(m):
        idx = valid[k][0]
        p_bh = min(adjusted[k], 1.0)
        sig = p_bh <= q
        results[idx]["p_bh"] = round(p_bh, 8)
        results[idx]["fdr_significant"] = sig
        if sig:
            n_sig += 1

    # Mark non-tested (None) entries
    for r in results:
        if r is not None and "p_bh" not in r:
            r["p_bh"] = 1.0
            r["fdr_significant"] = False

    return n_sig


# =========================================================================
# Output
# =========================================================================

def write_summary(survivors, all_results, n_tested, output_dir):
    """Write markdown summary of FDR survivors."""
    path = output_dir / "session_discovery_summary.md"

    with open(path, "w") as f:
        f.write("# Exhaustive Session Discovery — Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- **Candidate times:** {len(CANDIDATE_TIMES)} (every 5 min, 24 hours)\n")
        f.write(f"- **RR targets:** {RR_TARGETS}\n")
        f.write(f"- **G-filters:** {list(G_THRESHOLDS.keys())}\n")
        f.write(f"- **ORB aperture:** {ORB_BARS} minutes\n")
        f.write(f"- **Break window:** {BREAK_WINDOW} min | Outcome window: {OUTCOME_WINDOW} min\n")
        f.write(f"- **Min trades:** {MIN_TRADES}\n")
        f.write(f"- **BH FDR:** q={FDR_Q}\n")
        f.write(f"- **Cost model:** Applied (per-instrument COST_SPECS friction)\n\n")

        f.write("## Results Overview\n\n")
        f.write(f"- **Total raw combos:** {len(CANDIDATE_TIMES) * len(ACTIVE_ORB_INSTRUMENTS) * len(RR_TARGETS) * len(G_THRESHOLDS):,}\n")
        f.write(f"- **Combos with N>={MIN_TRADES}:** {n_tested:,}\n")
        f.write(f"- **BH FDR survivors (q={FDR_Q}):** {len(survivors)}\n\n")

        if not survivors:
            f.write("**NO FDR SURVIVORS.** No candidate time × parameter combination\n")
            f.write("produced statistically significant positive expectancy after\n")
            f.write("BH FDR correction across all tests.\n\n")
            f.write("**Conclusion:** The existing 10 sessions capture all discoverable\n")
            f.write("ORB breakout edge in this dataset.\n")
        else:
            # Group by near/far from existing sessions
            novel = [s for s in survivors if s["session_dist"] > 15]
            near = [s for s in survivors if s["session_dist"] <= 15]

            f.write(f"### Novel times (>15 min from any known session): {len(novel)}\n\n")
            if novel:
                f.write("| Instrument | Time | RR | G | N | ExpR | Sharpe | p_bh | Yrs+ | Dist | AvgVol |\n")
                f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
                for s in sorted(novel, key=lambda x: x["p_bh"]):
                    f.write(f"| {s['instrument']} | {s['time']} | {s['rr']} | {s['g_name']} "
                            f"| {s['n']} | {s['mean_r']:+.4f} | {s['sharpe_ann']:.2f} "
                            f"| {s['p_bh']:.6f} | {s['years_pos']}/{s['years_total']} "
                            f"| {s['session_dist']}min | {s['avg_vol']:.0f} |\n")
            else:
                f.write("None.\n")

            f.write(f"\n### Near existing sessions (<=15 min): {len(near)}\n\n")
            if near:
                f.write("| Instrument | Time | RR | G | N | ExpR | Sharpe | p_bh | Near |\n")
                f.write("|---|---|---|---|---|---|---|---|---|\n")
                for s in sorted(near, key=lambda x: x["p_bh"]):
                    f.write(f"| {s['instrument']} | {s['time']} | {s['rr']} | {s['g_name']} "
                            f"| {s['n']} | {s['mean_r']:+.4f} | {s['sharpe_ann']:.2f} "
                            f"| {s['p_bh']:.6f} | {s['near_session']} |\n")

            # Year-by-year for novel survivors
            if novel:
                f.write("\n### Year-by-Year Stability (novel survivors)\n\n")
                for s in novel:
                    f.write(f"\n**{s['instrument']} {s['time']} RR{s['rr']} {s['g_name']}** "
                            f"(N={s['n']}, p_bh={s['p_bh']:.6f})\n\n")
                    if "yearly" in s:
                        for yr in sorted(s["yearly"]):
                            yd = s["yearly"][yr]
                            f.write(f"- {yr}: N={yd['n']}, mean={yd['mean_r']:+.4f}\n")

        # Top 20 best non-FDR combos for context
        f.write("\n## Top 20 Combos by Raw p-value (regardless of FDR)\n\n")
        f.write("| Instrument | Time | RR | G | N | ExpR | p_raw | p_bh | FDR? | Near |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        sorted_all = sorted(
            [r for r in all_results if r is not None],
            key=lambda x: x["p_value"],
        )
        for r in sorted_all[:20]:
            sig_mark = "YES" if r.get("fdr_significant", False) else "no"
            f.write(f"| {r['instrument']} | {r['time']} | {r['rr']} | {r['g_name']} "
                    f"| {r['n']} | {r['mean_r']:+.4f} | {r['p_value']:.6f} "
                    f"| {r.get('p_bh', 1.0):.6f} | {sig_mark} | {r.get('near_session', '')} |\n")

    print(f"  Summary: {path}")
    return path


# =========================================================================
# Main
# =========================================================================

def main():
    t_start = _time.time()

    print(f"\n{'=' * 90}")
    print("  EXHAUSTIVE SESSION DISCOVERY")
    print(f"  Database: {GOLD_DB_PATH}")
    print(f"  Grid: {len(CANDIDATE_TIMES)} times x {len(RR_TARGETS)} RR x "
          f"{len(G_THRESHOLDS)} G = "
          f"{len(CANDIDATE_TIMES) * len(RR_TARGETS) * len(G_THRESHOLDS)} combos/instrument")
    print(f"  Instruments: {ACTIVE_ORB_INSTRUMENTS}")
    print(f"  ORB: {ORB_BARS}m | Break: {BREAK_WINDOW}m | Outcome: {OUTCOME_WINDOW}m")
    print(f"  FDR: BH q={FDR_Q} | Min trades: {MIN_TRADES}")
    print(f"{'=' * 90}")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    all_results = []  # flat list of stat dicts (one per tested combo)

    try:
        for instrument in ACTIVE_ORB_INSTRUMENTS:
            cost_spec = get_cost_spec(instrument)
            print(f"\n  [{instrument}] Loading bars (cost: ${cost_spec.total_friction:.2f} RT)...")
            t_load = _time.time()
            bars_df = load_bars(con, instrument)

            if len(bars_df) == 0:
                print(f"    No data for {instrument}, skipping.")
                continue

            print(f"    {len(bars_df):,} bars loaded in {_time.time() - t_load:.1f}s")

            t_build = _time.time()
            all_days, opens, highs, lows, closes, volumes = build_day_arrays(bars_df)
            n_days = len(all_days)
            print(f"    {n_days} trading days ({all_days[0]} to {all_days[-1]}) "
                  f"built in {_time.time() - t_build:.1f}s")

            del bars_df, opens  # free memory (opens not used in scan)

            dst_mask = build_dst_mask(all_days)
            n_summer = int(dst_mask.sum())
            print(f"    DST: {n_days - n_summer} winter / {n_summer} summer")

            print(f"    Scanning {len(CANDIDATE_TIMES)} candidate times...")
            t_scan = _time.time()
            inst_combos_tested = 0

            for ti, (bris_h, bris_m) in enumerate(CANDIDATE_TIMES):
                # Scan this time across all days
                trades = scan_time(
                    bris_h, bris_m, highs, lows, closes, volumes,
                    all_days, dst_mask, cost_spec,
                )

                # Compute stats for each (RR, G-filter) combo
                time_str = f"{bris_h:02d}:{bris_m:02d}"
                near_sess = nearest_session(bris_h, bris_m)
                sess_dist = session_distance_min(bris_h, bris_m)

                for g_name, g_min in G_LIST:
                    for rr in RR_TARGETS:
                        stat = compute_combo_stats(trades, rr, g_min, all_days)
                        if stat is not None:
                            stat["instrument"] = instrument
                            stat["time"] = time_str
                            stat["bris_h"] = bris_h
                            stat["bris_m"] = bris_m
                            stat["rr"] = rr
                            stat["g_name"] = g_name
                            stat["g_min"] = g_min
                            stat["near_session"] = near_sess
                            stat["session_dist"] = sess_dist
                            inst_combos_tested += 1

                            # Store yearly breakdown for potential survivors
                            filtered = [t for t in trades if t["orb_size"] >= g_min]
                            by_year = defaultdict(list)
                            for t in filtered:
                                by_year[t["year"]].append(t["outcomes"][rr])
                            stat["yearly"] = {
                                y: {"n": len(v), "mean_r": round(float(np.mean(v)), 4)}
                                for y, v in by_year.items()
                            }

                        all_results.append(stat)

                # Progress
                if (ti + 1) % 48 == 0:
                    elapsed = _time.time() - t_scan
                    pct = (ti + 1) / len(CANDIDATE_TIMES) * 100
                    print(f"      {instrument}: {ti + 1}/{len(CANDIDATE_TIMES)} times "
                          f"({pct:.0f}%) [{elapsed:.0f}s]")

            print(f"    {instrument}: {inst_combos_tested} combos with N>={MIN_TRADES} "
                  f"(scan: {_time.time() - t_scan:.0f}s)")

            del highs, lows, closes, volumes  # free before next instrument

    finally:
        con.close()

    # =====================================================================
    # BH FDR across ALL combos simultaneously
    # =====================================================================
    n_tested = sum(1 for r in all_results if r is not None)
    print(f"\n  Applying BH FDR (q={FDR_Q}) across {n_tested:,} tested combos...")
    n_sig = apply_bh_fdr(all_results, q=FDR_Q)
    print(f"  FDR survivors: {n_sig}")

    # Collect survivors
    survivors = [r for r in all_results if r is not None and r.get("fdr_significant", False)]
    survivors.sort(key=lambda x: x["p_bh"])

    # =====================================================================
    # Console output
    # =====================================================================
    print(f"\n{'=' * 90}")
    print(f"  RESULTS: {n_tested:,} combos tested | {n_sig} FDR survivors (q={FDR_Q})")
    print(f"{'=' * 90}")

    if survivors:
        # Group by novelty
        novel = [s for s in survivors if s["session_dist"] > 15]
        near = [s for s in survivors if s["session_dist"] <= 15]

        if novel:
            print(f"\n  NOVEL DISCOVERIES (>15 min from known sessions): {len(novel)}")
            print(f"  {'Instrument':10s} {'Time':6s} {'RR':4s} {'G':3s} {'N':>5s} "
                  f"{'ExpR':>8s} {'Sharpe':>7s} {'p_bh':>10s} {'Yrs+':>5s} "
                  f"{'Dist':>5s} {'AvgVol':>7s}")
            print(f"  {'-' * 80}")
            for s in novel:
                print(f"  {s['instrument']:10s} {s['time']:6s} {s['rr']:4.1f} {s['g_name']:3s} "
                      f"{s['n']:5d} {s['mean_r']:+8.4f} {s['sharpe_ann']:7.2f} "
                      f"{s['p_bh']:10.6f} {s['years_pos']}/{s['years_total']:1d} "
                      f"{s['session_dist']:4d}m {s['avg_vol']:7.0f}")

        if near:
            print(f"\n  NEAR EXISTING SESSIONS (<=15 min): {len(near)}")
            print(f"  {'Instrument':10s} {'Time':6s} {'RR':4s} {'G':3s} {'N':>5s} "
                  f"{'ExpR':>8s} {'p_bh':>10s} {'Near Session':20s}")
            print(f"  {'-' * 80}")
            for s in near[:30]:  # cap display at 30
                print(f"  {s['instrument']:10s} {s['time']:6s} {s['rr']:4.1f} {s['g_name']:3s} "
                      f"{s['n']:5d} {s['mean_r']:+8.4f} {s['p_bh']:10.6f} "
                      f"{s['near_session']:20s}")
            if len(near) > 30:
                print(f"  ... and {len(near) - 30} more")

        # Year-by-year for novel survivors
        if novel:
            print(f"\n  YEAR-BY-YEAR for novel survivors:")
            for s in novel[:10]:
                print(f"\n  {s['instrument']} {s['time']} RR{s['rr']} {s['g_name']} "
                      f"(N={s['n']}, p_bh={s['p_bh']:.6f}):")
                for yr in sorted(s["yearly"]):
                    yd = s["yearly"][yr]
                    bar = "+" * max(int(yd["mean_r"] * 20), 0) if yd["mean_r"] > 0 else "-" * max(int(-yd["mean_r"] * 20), 0)
                    print(f"    {yr}: N={yd['n']:3d} mean={yd['mean_r']:+.4f} {bar}")

    else:
        print("\n  NO FDR SURVIVORS.")
        print("  The existing 10 sessions appear to capture all discoverable")
        print("  ORB breakout edge at the tested parameters.\n")

        # Show the best near-misses
        tested = [r for r in all_results if r is not None]
        if tested:
            tested.sort(key=lambda x: x["p_value"])
            print(f"  Top 10 by raw p-value (did NOT survive FDR):")
            print(f"  {'Instrument':10s} {'Time':6s} {'RR':4s} {'G':3s} {'N':>5s} "
                  f"{'ExpR':>8s} {'p_raw':>10s} {'p_bh':>10s} {'Near':15s} {'Dist':>5s}")
            print(f"  {'-' * 90}")
            for r in tested[:10]:
                print(f"  {r['instrument']:10s} {r['time']:6s} {r['rr']:4.1f} {r['g_name']:3s} "
                      f"{r['n']:5d} {r['mean_r']:+8.4f} {r['p_value']:10.6f} "
                      f"{r.get('p_bh', 1.0):10.6f} {r.get('near_session', ''):15s} "
                      f"{r.get('session_dist', 0):4d}m")

    # =====================================================================
    # Save outputs
    # =====================================================================
    output_dir = Path("research/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV with all tested combos
    csv_rows = []
    for r in all_results:
        if r is None:
            continue
        row = {k: v for k, v in r.items() if k != "yearly"}
        csv_rows.append(row)

    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "session_discovery_full.csv"
    csv_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV: {csv_path} ({len(csv_df)} rows)")

    # Summary markdown
    write_summary(survivors, all_results, n_tested, output_dir)

    elapsed_total = _time.time() - t_start
    print(f"\n  Total runtime: {elapsed_total / 60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
