#!/usr/bin/env python3
"""
E3 Retrace Timing Research — 6-question analysis of retrace entry mechanics.

Tests how long after an ORB break price retraces, how long to hold after entry,
and how these interact with ORB size and volatility regime.

Uses daily_features.orb_{label}_break_ts as the anchor (pre-computed by
pipeline), then scans bars_1m forward to simulate E3 retrace logic exactly
mirroring _resolve_e3() in trading_app/entry_rules.py.

Read-only: no writes to gold.db.

Output:
  research/output/retrace_delay_sweep.csv          <- Q1
  research/output/hold_duration_decay.csv          <- Q2
  research/output/delay_hold_heatmap.csv           <- Q3
  research/output/retrace_size_interaction.csv     <- Q4
  research/output/retrace_regime_conditioning.csv  <- Q6
  research/output/retrace_timing_summary.md        <- narrative (Q1-Q6)

Usage:
  python research/research_retrace_timing.py
  python research/research_retrace_timing.py --instruments MGC --sessions 1000
  python research/research_retrace_timing.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Path bootstrap (canonical pattern — PROJECT_ROOT on sys.path)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Timezone helpers (standalone — no imports from pipeline)
# ---------------------------------------------------------------------------
_US_EASTERN = ZoneInfo("America/New_York")
_UK_LONDON = ZoneInfo("Europe/London")


def is_us_dst(trading_day: date) -> bool:
    """True if US Eastern is in DST (EDT, UTC-4) on this date."""
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_US_EASTERN)
    return dt.utcoffset().total_seconds() == -4 * 3600


def is_uk_dst(trading_day: date) -> bool:
    """True if UK is in BST (UTC+1) on this date."""
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_UK_LONDON)
    return dt.utcoffset().total_seconds() == 1 * 3600


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sessions to test. Skip 1100 (74% double-break, confirmed NO-GO).
# Keys are Brisbane clock times. 0900/0030/2300 use US DST; 1800 uses UK DST.
SESSIONS = {
    "0900": {"bris_h": 9,  "bris_m": 0,  "orb_min": 5,  "dst_type": "US"},
    "1000": {"bris_h": 10, "bris_m": 0,  "orb_min": 15, "dst_type": "CLEAN"},
    "1800": {"bris_h": 18, "bris_m": 0,  "orb_min": 5,  "dst_type": "UK"},
    "2300": {"bris_h": 23, "bris_m": 0,  "orb_min": 5,  "dst_type": "US"},
    "0030": {"bris_h": 0,  "bris_m": 30, "orb_min": 5,  "dst_type": "US"},
}

DEFAULT_INSTRUMENTS = ["MGC", "MES"]

# G-filter thresholds (from config.py)
G_FILTERS = {"G4": 4.0, "G5": 5.0, "G6": 6.0, "G8": 8.0}
G4_MIN = 4.0  # Base filter for all analysis

# P&L checkpoints: minutes after retrace fill
CHECKPOINTS = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]

# Retrace scan window: minutes after break to look for retrace fill
SCAN_WINDOW = 240  # 4 hours baseline; sensitivity checks at 192 and 288

# Max hold minutes after fill (also the primary Q1 outcome checkpoint)
MAX_HOLD = 180
PRIMARY_CHECKPOINT = 180

# RR target (production E3 default)
RR_TARGET = 2.0

# Delay buckets for Q1/Q4/Q6 (minutes from break to fill)
DELAY_BUCKETS = [
    ("0-2",   0,   2),
    ("2-5",   2,   5),
    ("5-10",  5,  10),
    ("10-20", 10, 20),
    ("20-30", 20, 30),
    ("30+",   30, 99999),
]
DELAY_BUCKET_ORDER = [b[0] for b in DELAY_BUCKETS]


def classify_sample(n: int) -> str:
    """Per RESEARCH_RULES.md thresholds."""
    if n < 30:
        return "INVALID"
    if n < 100:
        return "REGIME"
    if n < 200:
        return "PRELIMINARY"
    return "CORE"


def delay_bucket_label(delay_min: int) -> str:
    for label, lo, hi in DELAY_BUCKETS:
        if lo <= delay_min < hi:
            return label
    return "30+"


def dst_regime_for(session: str, us_dst: bool, uk_dst: bool) -> str:
    """Compute DST regime label for a session + day combination."""
    dst_type = SESSIONS[session]["dst_type"]
    if dst_type == "CLEAN":
        return "ALL"
    elif dst_type == "US":
        return "SUMMER" if us_dst else "WINTER"
    else:  # UK
        return "SUMMER" if uk_dst else "WINTER"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(con, instrument: str) -> pd.DataFrame:
    """Load all 1m bars for an instrument."""
    return con.execute("""
        SELECT ts_utc, open, high, low, close
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
    """, [instrument]).fetchdf()


def build_day_arrays(bars_df: pd.DataFrame):
    """Convert bars DataFrame into 2D numpy arrays indexed by (day, minute_offset).

    Returns (trading_days_list, day_to_idx, opens, highs, lows, closes) where
    each array has shape (n_days, 1440) with NaN for missing bars.

    Minute offset 0 = 09:00 Brisbane; 1439 = 08:59 next day Brisbane.
    """
    df = bars_df.copy()
    df["bris_dt"] = df["ts_utc"] + pd.Timedelta(hours=10)
    df["bris_hour"] = df["bris_dt"].dt.hour
    df["bris_minute"] = df["bris_dt"].dt.minute

    df["trading_day"] = df["bris_dt"].dt.normalize()
    mask = df["bris_hour"] < 9
    df.loc[mask, "trading_day"] -= pd.Timedelta(days=1)
    df["trading_day"] = df["trading_day"].dt.date

    df["min_offset"] = ((df["bris_hour"] - 9) % 24) * 60 + df["bris_minute"]

    all_days = sorted(df["trading_day"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_days)}
    n_days = len(all_days)

    opens  = np.full((n_days, 1440), np.nan)
    highs  = np.full((n_days, 1440), np.nan)
    lows   = np.full((n_days, 1440), np.nan)
    closes = np.full((n_days, 1440), np.nan)

    day_idx = df["trading_day"].map(day_to_idx).values
    min_idx = df["min_offset"].values

    opens[day_idx, min_idx]  = df["open"].values
    highs[day_idx, min_idx]  = df["high"].values
    lows[day_idx, min_idx]   = df["low"].values
    closes[day_idx, min_idx] = df["close"].values

    return all_days, day_to_idx, opens, highs, lows, closes


def build_dst_masks(trading_days: list) -> tuple[np.ndarray, np.ndarray]:
    """Build US and UK DST boolean masks. True = summer/DST."""
    n = len(trading_days)
    us_mask = np.zeros(n, dtype=bool)
    uk_mask = np.zeros(n, dtype=bool)
    for i, td in enumerate(trading_days):
        us_mask[i] = is_us_dst(td)
        uk_mask[i] = is_uk_dst(td)
    return us_mask, uk_mask


def load_daily_features(con, instrument: str, session: str) -> pd.DataFrame:
    """Load daily_features for one (instrument, session) pair.

    Filters to G4+ days with a confirmed break only.
    """
    label = session
    return con.execute(f"""
        SELECT
            trading_day,
            orb_{label}_high    AS orb_high,
            orb_{label}_low     AS orb_low,
            orb_{label}_size    AS orb_size,
            orb_{label}_break_ts  AS break_ts,
            orb_{label}_break_dir AS break_dir,
            atr_20
        FROM daily_features
        WHERE symbol = ?
          AND orb_{label}_break_ts IS NOT NULL
          AND orb_{label}_break_dir IS NOT NULL
          AND orb_{label}_size >= {G4_MIN}
        ORDER BY trading_day
    """, [instrument]).fetchdf()


def break_ts_to_min_offset(break_ts) -> int:
    """Convert a UTC break timestamp to Brisbane trading-day minute offset.

    Handles both tz-aware (TIMESTAMPTZ) and naive pandas Timestamps.
    Brisbane = UTC+10, no DST.
    """
    if hasattr(break_ts, "timestamp"):
        # Use .timestamp() for correct UTC epoch regardless of tz-awareness
        import datetime as _dt
        utc_dt = _dt.datetime.fromtimestamp(break_ts.timestamp(), _dt.timezone.utc)
        bris_hour = (utc_dt.hour + 10) % 24
        bris_minute = utc_dt.minute
    else:
        # Fallback: treat as naive UTC
        bris_hour = (int(break_ts) // 3600 + 10) % 24
        bris_minute = 0
    return ((bris_hour - 9) % 24) * 60 + bris_minute


# ---------------------------------------------------------------------------
# Core engine — mirrors _resolve_e3() in trading_app/entry_rules.py
# ---------------------------------------------------------------------------

def scan_one_day(
    day_idx: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    break_min_offset: int,
    break_dir: str,
    orb_high: float,
    orb_low: float,
    orb_size: float,
    scan_window: int = SCAN_WINDOW,
) -> tuple | None:
    """Simulate E3 retrace entry for one trading day.

    Mirrors _resolve_e3() logic: stop hit before retrace → no fill.

    Returns (delay_minutes, pnl_at_dict, terminal_minute_or_None)
    or None if no valid fill within scan_window.

    pnl_at_dict keys are CHECKPOINTS (minutes after fill).
    terminal_minute is minutes-after-fill when T/S was hit, or None for timeout.
    """
    # Scan window: start after break bar, end at scan_window or day boundary
    scan_start = break_min_offset + 1
    scan_end = min(break_min_offset + scan_window + 1, 1440)

    if break_dir == "long":
        entry_price = orb_high
        stop_level  = orb_low
        target_price = orb_high + RR_TARGET * orb_size
    else:
        entry_price  = orb_low
        stop_level   = orb_high
        target_price = orb_low - RR_TARGET * orb_size

    # --- Phase 1: find retrace fill (mirrors _resolve_e3 stop-before-retrace) ---
    fill_min = None
    for m in range(scan_start, scan_end):
        h = highs[day_idx, m]
        l = lows[day_idx, m]
        if np.isnan(l):
            continue

        if break_dir == "long":
            # Stop: price falls all the way to orb_low
            # Retrace: price pulls back to orb_high (limit buy level)
            # Note: orb_low < orb_high, so stop_hit ⊂ retrace_hit
            stop_hit    = l <= stop_level    # l <= orb_low
            retrace_hit = l <= entry_price   # l <= orb_high
        else:
            # Stop: price rises all the way to orb_high
            # Retrace: price rises back to orb_low (limit sell level)
            stop_hit    = h >= stop_level    # h >= orb_high
            retrace_hit = h >= entry_price   # h >= orb_low

        # Stop takes priority (includes same-bar stop+retrace case)
        if stop_hit:
            return None

        if retrace_hit:
            fill_min = m
            break

    if fill_min is None:
        return None

    delay_minutes = fill_min - break_min_offset

    # --- Phase 2: simulate hold from fill_min forward ---
    # Track terminal event and record P&L at each CHECKPOINT
    terminal_r = None
    terminal_minute = None  # minutes after fill
    last_close = entry_price
    pnl_at = {}
    checkpoint_set = set(CHECKPOINTS)

    for m_offset in range(1, MAX_HOLD + 1):
        abs_m = fill_min + m_offset
        if abs_m >= 1440:
            break

        h = highs[day_idx, abs_m]
        l = lows[day_idx, abs_m]
        c = closes[day_idx, abs_m]

        if not np.isnan(c):
            last_close = c

        # Check for terminal exit (only once)
        if terminal_r is None and not np.isnan(h) and not np.isnan(l):
            if break_dir == "long":
                # Ambiguous same-bar: both stop and target hit → loss (conservative)
                if l <= stop_level and h >= target_price:
                    terminal_r = -1.0
                    terminal_minute = m_offset
                elif l <= stop_level:
                    terminal_r = -1.0
                    terminal_minute = m_offset
                elif h >= target_price:
                    terminal_r = RR_TARGET
                    terminal_minute = m_offset
            else:
                if h >= stop_level and l <= target_price:
                    terminal_r = -1.0
                    terminal_minute = m_offset
                elif h >= stop_level:
                    terminal_r = -1.0
                    terminal_minute = m_offset
                elif l <= target_price:
                    terminal_r = RR_TARGET
                    terminal_minute = m_offset

        # Record checkpoint value
        if m_offset in checkpoint_set:
            if terminal_r is not None:
                pnl_at[m_offset] = terminal_r
            else:
                # Mark-to-market at last known close
                if break_dir == "long":
                    pnl_at[m_offset] = (last_close - entry_price) / orb_size
                else:
                    pnl_at[m_offset] = (entry_price - last_close) / orb_size

    # Fill any checkpoints beyond end of day (use last known pnl)
    last_pnl = pnl_at.get(max(pnl_at.keys(), default=0), np.nan) if pnl_at else np.nan
    for cp in CHECKPOINTS:
        if cp not in pnl_at:
            if terminal_r is not None:
                pnl_at[cp] = terminal_r
            else:
                pnl_at[cp] = last_pnl

    return delay_minutes, pnl_at, terminal_minute


# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------

def collect_results(
    instrument: str,
    session: str,
    all_days: list,
    day_to_idx: dict,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    us_mask: np.ndarray,
    uk_mask: np.ndarray,
    df_features: pd.DataFrame,
    scan_window: int = SCAN_WINDOW,
) -> list[dict]:
    """Scan all days for one (instrument, session) and return flat records."""
    records = []

    for _, row in df_features.iterrows():
        td = row["trading_day"]
        if hasattr(td, "date"):
            td = td.date()
        elif hasattr(td, "item"):
            td = pd.Timestamp(td).date()

        if td not in day_to_idx:
            continue

        break_ts = row["break_ts"]
        if break_ts is None or (isinstance(break_ts, float) and np.isnan(break_ts)):
            continue
        try:
            if pd.isna(break_ts):
                continue
        except (TypeError, ValueError):
            pass

        try:
            break_min_offset = break_ts_to_min_offset(break_ts)
        except Exception:
            continue

        day_idx = day_to_idx[td]
        orb_high = float(row["orb_high"])
        orb_low  = float(row["orb_low"])
        orb_size = float(row["orb_size"])
        atr_20   = float(row["atr_20"]) if row["atr_20"] is not None and not pd.isna(row["atr_20"]) else np.nan
        break_dir = str(row["break_dir"])

        result = scan_one_day(
            day_idx, highs, lows, closes,
            break_min_offset, break_dir,
            orb_high, orb_low, orb_size,
            scan_window=scan_window,
        )
        if result is None:
            continue

        delay_minutes, pnl_at, terminal_minute = result

        us_dst = bool(us_mask[day_idx])
        uk_dst = bool(uk_mask[day_idx])
        dst_regime = dst_regime_for(session, us_dst, uk_dst)

        record = {
            "instrument": instrument,
            "session": session,
            "trading_day": td,
            "orb_size": orb_size,
            "atr_20": atr_20,
            "us_dst": us_dst,
            "uk_dst": uk_dst,
            "dst_regime": dst_regime,
            "delay_minutes": delay_minutes,
            "delay_bucket": delay_bucket_label(delay_minutes),
            "terminal_minute": terminal_minute,
        }
        for cp in CHECKPOINTS:
            record[f"pnl_{cp}"] = pnl_at.get(cp, np.nan)

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_pvalue(arr: np.ndarray) -> float:
    """One-sample t-test: is mean significantly different from 0?"""
    if not HAS_SCIPY:
        return np.nan
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan
    _, p = ttest_1samp(arr, 0.0)
    return float(p)


def bh_fdr_correction(p_values: list) -> list:
    """Benjamini-Hochberg FDR correction. NaN-safe."""
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    adjusted = np.full(n, np.nan)

    valid_mask = ~np.isnan(p_arr)
    valid_idx = np.where(valid_mask)[0]
    valid_p = p_arr[valid_idx]

    if len(valid_p) == 0:
        return adjusted.tolist()

    m = len(valid_p)
    sorted_order = np.argsort(valid_p)
    sorted_p = valid_p[sorted_order]

    bh_adjusted = np.zeros(m)
    bh_adjusted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        rank = i + 1
        bh_adjusted[i] = min(bh_adjusted[i + 1], sorted_p[i] * m / rank)

    bh_adjusted = np.clip(bh_adjusted, 0.0, 1.0)
    unsorted = np.zeros(m)
    unsorted[sorted_order] = bh_adjusted
    adjusted[valid_idx] = unsorted
    return adjusted.tolist()


def agg_stats(pnl_arr: np.ndarray) -> dict:
    """Compute avg_r, win_rate, total_r, sharpe from pnl array."""
    arr = pnl_arr[~np.isnan(pnl_arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0, "avg_r": np.nan, "win_rate": np.nan,
                "total_r": np.nan, "sharpe": np.nan, "pvalue": np.nan}
    avg_r = float(arr.mean())
    win_rate = float((arr > 0).sum() / n)
    total_r = float(arr.sum())
    std = float(arr.std())
    sharpe = avg_r / std if std > 1e-10 else np.nan
    pvalue = compute_pvalue(arr)
    return {"n": n, "avg_r": avg_r, "win_rate": win_rate,
            "total_r": total_r, "sharpe": sharpe, "pvalue": pvalue}


# ---------------------------------------------------------------------------
# Q1: Retrace Delay Sweep
# ---------------------------------------------------------------------------

def compute_q1(df: pd.DataFrame) -> list[dict]:
    """Group by (instrument, session, dst_regime, delay_bucket) → pnl_180."""
    rows = []
    pnl_col = f"pnl_{PRIMARY_CHECKPOINT}"

    for (inst, sess, dst_regime), grp in df.groupby(
            ["instrument", "session", "dst_regime"], sort=True):
        total_fills = len(grp)
        for delay_bucket in DELAY_BUCKET_ORDER:
            sub = grp[grp["delay_bucket"] == delay_bucket]
            pnl = sub[pnl_col].values
            stats = agg_stats(pnl)
            pct_of_fills = round(stats["n"] / total_fills * 100, 1) if total_fills > 0 else 0.0
            rows.append({
                "instrument": inst,
                "session": sess,
                "dst_regime": dst_regime,
                "delay_bucket": delay_bucket,
                "n": stats["n"],
                "sample_class": classify_sample(stats["n"]),
                "win_rate": round(stats["win_rate"], 4) if not np.isnan(stats["win_rate"]) else np.nan,
                "avg_r": round(stats["avg_r"], 4) if not np.isnan(stats["avg_r"]) else np.nan,
                "total_r": round(stats["total_r"], 2) if not np.isnan(stats["total_r"]) else np.nan,
                "sharpe": round(stats["sharpe"], 4) if not np.isnan(stats["sharpe"]) else np.nan,
                "pvalue": round(stats["pvalue"], 6) if not np.isnan(stats["pvalue"]) else np.nan,
                "pvalue_bh": np.nan,
                "pct_of_fills": pct_of_fills,
            })

    # BH correction across all Q1 rows
    p_vals = [r["pvalue"] for r in rows]
    adjusted = bh_fdr_correction(p_vals)
    for i, r in enumerate(rows):
        r["pvalue_bh"] = round(adjusted[i], 6) if not np.isnan(adjusted[i]) else np.nan

    return rows


# ---------------------------------------------------------------------------
# Q2: Hold Duration Decay
# ---------------------------------------------------------------------------

def compute_q2(df: pd.DataFrame) -> list[dict]:
    """Group by (instrument, session, dst_regime) × checkpoint → pnl at N min."""
    rows = []

    for (inst, sess, dst_regime), grp in df.groupby(
            ["instrument", "session", "dst_regime"], sort=True):
        n_total = len(grp)
        for cp in CHECKPOINTS:
            pnl_col = f"pnl_{cp}"
            pnl = grp[pnl_col].values
            stats = agg_stats(pnl)
            # terminal_rate: fraction where trade was already terminal by this checkpoint
            terminal_rate = float(
                grp["terminal_minute"].apply(
                    lambda tm: tm is not None and not (isinstance(tm, float) and np.isnan(tm)) and tm <= cp
                ).sum()
            ) / n_total if n_total > 0 else 0.0
            rows.append({
                "instrument": inst,
                "session": sess,
                "dst_regime": dst_regime,
                "hold_minutes": cp,
                "n": stats["n"],
                "avg_r": round(stats["avg_r"], 4) if not np.isnan(stats["avg_r"]) else np.nan,
                "win_rate": round(stats["win_rate"], 4) if not np.isnan(stats["win_rate"]) else np.nan,
                "sharpe": round(stats["sharpe"], 4) if not np.isnan(stats["sharpe"]) else np.nan,
                "pvalue": round(stats["pvalue"], 6) if not np.isnan(stats["pvalue"]) else np.nan,
                "pvalue_bh": np.nan,
                "terminal_rate": round(terminal_rate, 4),
            })

    p_vals = [r["pvalue"] for r in rows]
    adjusted = bh_fdr_correction(p_vals)
    for i, r in enumerate(rows):
        r["pvalue_bh"] = round(adjusted[i], 6) if not np.isnan(adjusted[i]) else np.nan

    return rows


# ---------------------------------------------------------------------------
# Q3: 2D Heatmap (delay_bucket × hold_minutes)
# ---------------------------------------------------------------------------

def compute_q3(df: pd.DataFrame) -> list[dict]:
    """Cross-tabulation of delay_bucket × hold_minutes. No new scan."""
    rows = []

    for (inst, sess, dst_regime), grp in df.groupby(
            ["instrument", "session", "dst_regime"], sort=True):
        for delay_bucket in DELAY_BUCKET_ORDER:
            sub_delay = grp[grp["delay_bucket"] == delay_bucket]
            for cp in CHECKPOINTS:
                pnl_col = f"pnl_{cp}"
                pnl = sub_delay[pnl_col].values
                pnl_valid = pnl[~np.isnan(pnl)]
                n = len(pnl_valid)
                avg_r = float(pnl_valid.mean()) if n > 0 else np.nan
                win_rate = float((pnl_valid > 0).sum() / n) if n > 0 else np.nan
                rows.append({
                    "instrument": inst,
                    "session": sess,
                    "dst_regime": dst_regime,
                    "delay_bucket": delay_bucket,
                    "hold_minutes": cp,
                    "n": n,
                    "avg_r": round(avg_r, 4) if not np.isnan(avg_r) else np.nan,
                    "win_rate": round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
                    "sample_class": classify_sample(n),
                })

    return rows


# ---------------------------------------------------------------------------
# Q4: ORB Size Interaction
# ---------------------------------------------------------------------------

def compute_q4(df: pd.DataFrame) -> list[dict]:
    """Re-run Q1 delay sweep segmented by G-filter level."""
    rows = []
    pnl_col = f"pnl_{PRIMARY_CHECKPOINT}"

    for (inst, sess, dst_regime), grp in df.groupby(
            ["instrument", "session", "dst_regime"], sort=True):
        for filter_name, min_size in G_FILTERS.items():
            filtered = grp[grp["orb_size"] >= min_size]
            if len(filtered) == 0:
                continue
            for delay_bucket in DELAY_BUCKET_ORDER:
                sub = filtered[filtered["delay_bucket"] == delay_bucket]
                pnl = sub[pnl_col].values
                stats = agg_stats(pnl)
                rows.append({
                    "instrument": inst,
                    "session": sess,
                    "dst_regime": dst_regime,
                    "orb_filter": filter_name,
                    "delay_bucket": delay_bucket,
                    "n": stats["n"],
                    "sample_class": classify_sample(stats["n"]),
                    "avg_r": round(stats["avg_r"], 4) if not np.isnan(stats["avg_r"]) else np.nan,
                    "win_rate": round(stats["win_rate"], 4) if not np.isnan(stats["win_rate"]) else np.nan,
                    "pvalue": round(stats["pvalue"], 6) if not np.isnan(stats["pvalue"]) else np.nan,
                    "pvalue_bh": np.nan,
                })

    p_vals = [r["pvalue"] for r in rows]
    adjusted = bh_fdr_correction(p_vals)
    for i, r in enumerate(rows):
        r["pvalue_bh"] = round(adjusted[i], 6) if not np.isnan(adjusted[i]) else np.nan

    return rows


# ---------------------------------------------------------------------------
# Q6: Regime Conditioning (ATR split at 50th percentile)
# ---------------------------------------------------------------------------

def compute_q6(df: pd.DataFrame) -> list[dict]:
    """Q1 delay sweep segmented by ATR regime (EXPANSION vs CONTRACTION)."""
    rows = []
    pnl_col = f"pnl_{PRIMARY_CHECKPOINT}"

    for (inst, sess, dst_regime), grp in df.groupby(
            ["instrument", "session", "dst_regime"], sort=True):
        atr_vals = grp["atr_20"].dropna()
        if len(atr_vals) < 10:
            continue  # Not enough ATR data for a meaningful split
        atr_threshold = float(atr_vals.median())

        for atr_regime, atr_mask in [
            ("EXPANSION",   grp["atr_20"] > atr_threshold),
            ("CONTRACTION", grp["atr_20"] <= atr_threshold),
        ]:
            atr_grp = grp[atr_mask]
            for delay_bucket in DELAY_BUCKET_ORDER:
                sub = atr_grp[atr_grp["delay_bucket"] == delay_bucket]
                pnl = sub[pnl_col].values
                stats = agg_stats(pnl)
                rows.append({
                    "instrument": inst,
                    "session": sess,
                    "dst_regime": dst_regime,
                    "atr_regime": atr_regime,
                    "atr_threshold": round(atr_threshold, 4),
                    "delay_bucket": delay_bucket,
                    "n": stats["n"],
                    "sample_class": classify_sample(stats["n"]),
                    "avg_r": round(stats["avg_r"], 4) if not np.isnan(stats["avg_r"]) else np.nan,
                    "win_rate": round(stats["win_rate"], 4) if not np.isnan(stats["win_rate"]) else np.nan,
                    "pvalue": round(stats["pvalue"], 6) if not np.isnan(stats["pvalue"]) else np.nan,
                    "pvalue_bh": np.nan,
                })

    p_vals = [r["pvalue"] for r in rows]
    adjusted = bh_fdr_correction(p_vals)
    for i, r in enumerate(rows):
        r["pvalue_bh"] = round(adjusted[i], 6) if not np.isnan(adjusted[i]) else np.nan

    return rows


# ---------------------------------------------------------------------------
# Sensitivity check (Q1 at alternate scan windows)
# ---------------------------------------------------------------------------

def run_sensitivity(
    instrument: str,
    session: str,
    all_days: list,
    day_to_idx: dict,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    us_mask: np.ndarray,
    uk_mask: np.ndarray,
    df_features: pd.DataFrame,
) -> dict:
    """Re-run Q1 at scan_window = 192 and 288 vs 240 baseline.

    Returns dict: {window: {(dst_regime, delay_bucket): avg_r}}.
    """
    results = {}
    for window in [192, SCAN_WINDOW, 288]:
        recs = collect_results(
            instrument, session, all_days, day_to_idx,
            highs, lows, closes, us_mask, uk_mask,
            df_features, scan_window=window,
        )
        if not recs:
            results[window] = {}
            continue
        tmp_df = pd.DataFrame(recs)
        pnl_col = f"pnl_{PRIMARY_CHECKPOINT}"
        bucket_avgs = {}
        for (dst_regime, delay_bucket), grp in tmp_df.groupby(["dst_regime", "delay_bucket"]):
            pnl = grp[pnl_col].dropna().values
            if len(pnl) >= 10:
                bucket_avgs[(dst_regime, delay_bucket)] = float(pnl.mean())
        results[window] = bucket_avgs
    return results


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def fmt_r(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A"
    return f"{v:+.4f}"


def fmt_p(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A  "
    return f"{v:.4f}"


def generate_markdown(
    q1_rows: list[dict],
    q2_rows: list[dict],
    q3_rows: list[dict],
    q4_rows: list[dict],
    q6_rows: list[dict],
    instruments: list[str],
    sessions: list[str],
    total_fills: int,
    date_range: str,
    sensitivity_notes: list[str],
) -> str:
    lines = []

    lines += [
        "# Retrace Timing Research — Honest Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Instruments:** {', '.join(instruments)}",
        f"**Sessions:** {', '.join(sessions)}",
        f"**RR Target:** {RR_TARGET:.1f}",
        f"**G-filter baseline:** G4+ (ORB size >= 4.0 pts)",
        f"**Scan window:** {SCAN_WINDOW} min after break (sensitivity: 192/288)",
        f"**Primary checkpoint:** {PRIMARY_CHECKPOINT} min after fill",
        f"**Data:** {date_range}",
        f"**Total fills analyzed:** {total_fills:,}",
        "**Analysis type:** IN-SAMPLE only (not walk-forward validated)",
        "",
    ]

    # SURVIVED / DID NOT SURVIVE
    core_positive = [r for r in q1_rows
                     if r["sample_class"] in ("CORE", "PRELIMINARY")
                     and not np.isnan(r.get("avg_r", np.nan))
                     and r["avg_r"] > 0
                     and not np.isnan(r.get("pvalue_bh", np.nan))
                     and r["pvalue_bh"] < 0.05]
    core_negative = [r for r in q1_rows
                     if r["sample_class"] in ("CORE", "PRELIMINARY")
                     and not np.isnan(r.get("avg_r", np.nan))
                     and r["avg_r"] <= 0
                     and not np.isnan(r.get("pvalue_bh", np.nan))
                     and r["pvalue_bh"] < 0.05]
    inconclusive = [r for r in q1_rows
                    if r["sample_class"] == "INVALID"
                    or np.isnan(r.get("pvalue_bh", np.nan))
                    or r["pvalue_bh"] >= 0.05]

    lines += [
        "## SURVIVED SCRUTINY",
        "",
        "Cells with CORE/PRELIMINARY N, positive avg_r at 180 min, BH p < 0.05:",
        "",
    ]
    if not core_positive:
        lines.append("*None — no delay bucket demonstrated statistically significant positive edge at p_bh < 0.05.*")
    else:
        lines.append("| Instrument | Session | DST | Delay | N | avg_r | p_bh | Class |")
        lines.append("|------------|---------|-----|-------|---|-------|------|-------|")
        for r in sorted(core_positive, key=lambda x: x.get("avg_r", 0), reverse=True):
            lines.append(
                f"| {r['instrument']} | {r['session']} | {r['dst_regime']} | "
                f"{r['delay_bucket']} | {r['n']} | {fmt_r(r['avg_r'])} | "
                f"{fmt_p(r['pvalue_bh'])} | {r['sample_class']} |"
            )
    lines.append("")

    lines += ["## DID NOT SURVIVE", ""]
    if core_negative:
        lines.append("Statistically significant *negative* results:")
        lines.append("| Instrument | Session | DST | Delay | N | avg_r | p_bh |")
        lines.append("|------------|---------|-----|-------|---|-------|------|")
        for r in core_negative:
            lines.append(
                f"| {r['instrument']} | {r['session']} | {r['dst_regime']} | "
                f"{r['delay_bucket']} | {r['n']} | {fmt_r(r['avg_r'])} | "
                f"{fmt_p(r['pvalue_bh'])} |"
            )
        lines.append("")
    n_invalid = sum(1 for r in q1_rows if r["sample_class"] == "INVALID")
    lines.append(f"{n_invalid} of {len(q1_rows)} rows are INVALID (N < 30).")
    lines.append("")

    lines += [
        "## CAVEATS",
        "",
        "- **IN-SAMPLE only** — all dates used in analysis, no held-out OOS period.",
        "- G4+ base filter; G5/G6/G8 sub-analysis in Q4.",
        f"- RR{RR_TARGET:.1f} target only (production tests 1.0–4.0).",
        f"- Scan window {SCAN_WINDOW} min (production cap is 60 min from E3_RETRACE_WINDOW_MINUTES).",
        "- MTM at 180 min is mark-to-market, not terminal: overstates open winner exposure.",
        "- Walk-forward validation required before any production use.",
        "",
    ]

    lines += ["## NEXT STEPS", ""]
    if core_positive:
        lines += [
            "- For SURVIVED cells: run sensitivity at 60 min scan window (production cap)",
            "  and compare avg_r — if materially different, retrace timing matters.",
            "- Consider tightening E3_RETRACE_WINDOW_MINUTES to the bucket cutoff",
            "  if early-delay cells outperform late-delay cells consistently.",
            "- Validate with walk-forward before changing production rules.",
        ]
    else:
        lines += [
            "- No statistically significant delay-bucket advantage found.",
            "- Current E3 approach (any retrace within scan window) appears consistent.",
            "- Consider testing whether a tighter retrace window improves edge",
            "  by comparing pnl_60 vs pnl_180 from Q2 hold duration data.",
        ]
    lines.append("")

    # Q1 detail
    lines += [
        "---",
        "",
        "## Q1: Retrace Delay Sweep",
        "",
        "P&L at 180-min checkpoint (terminal if earlier) by delay bucket.",
        "",
        "| Instrument | Session | DST | Delay | N | Class | WR | avg_r | p_raw | p_bh | pct% |",
        "|------------|---------|-----|-------|---|-------|----|-------|-------|------|------|",
    ]
    for r in q1_rows:
        wr_str = f"{r['win_rate']:.1%}" if not np.isnan(r.get("win_rate", np.nan)) else "N/A"
        lines.append(
            f"| {r['instrument']} | {r['session']} | {r['dst_regime']} | "
            f"{r['delay_bucket']} | {r['n']} | {r['sample_class']} | {wr_str} | "
            f"{fmt_r(r['avg_r'])} | {fmt_p(r['pvalue'])} | {fmt_p(r['pvalue_bh'])} | "
            f"{r['pct_of_fills']:.1f}% |"
        )
    lines.append("")

    # Q2 summary (compact — key sessions)
    lines += [
        "---",
        "",
        "## Q2: Hold Duration Decay",
        "",
        "Average R at each hold checkpoint. `terminal_rate` = fraction of trades",
        "already resolved (T/S hit) by that minute.",
        "",
    ]
    # Show a compact pivot per (instrument, session, dst_regime)
    q2_df = pd.DataFrame(q2_rows)
    if not q2_df.empty:
        for (inst, sess, dst_regime), grp in q2_df.groupby(
                ["instrument", "session", "dst_regime"]):
            lines.append(f"**{inst} {sess} ({dst_regime})**")
            lines.append("")
            lines.append("| hold_min | N | avg_r | win_rate | sharpe | terminal% |")
            lines.append("|----------|---|-------|----------|--------|-----------|")
            for _, row in grp.sort_values("hold_minutes").iterrows():
                wr_str = f"{row['win_rate']:.1%}" if not np.isnan(row.get("win_rate", np.nan)) else "N/A"
                sh_str = f"{row['sharpe']:+.3f}" if not np.isnan(row.get("sharpe", np.nan)) else "N/A"
                tr_str = f"{row['terminal_rate']:.1%}"
                lines.append(
                    f"| {int(row['hold_minutes'])} | {int(row['n'])} | "
                    f"{fmt_r(row['avg_r'])} | {wr_str} | {sh_str} | {tr_str} |"
                )
            lines.append("")

    # Q3 key findings
    lines += [
        "---",
        "",
        "## Q3: 2D Heatmap Key Findings",
        "",
        "Top cells by avg_r where N >= 30 (REGIME+). Full data in delay_hold_heatmap.csv.",
        "",
    ]
    q3_df = pd.DataFrame(q3_rows)
    if not q3_df.empty:
        top_q3 = q3_df[q3_df["n"] >= 30].sort_values("avg_r", ascending=False).head(15)
        if not top_q3.empty:
            lines.append("| Instrument | Session | DST | Delay | Hold | N | avg_r | Class |")
            lines.append("|------------|---------|-----|-------|------|---|-------|-------|")
            for _, row in top_q3.iterrows():
                lines.append(
                    f"| {row['instrument']} | {row['session']} | {row['dst_regime']} | "
                    f"{row['delay_bucket']} | {int(row['hold_minutes'])} | {int(row['n'])} | "
                    f"{fmt_r(row['avg_r'])} | {row['sample_class']} |"
                )
            lines.append("")

    # Q4 summary
    lines += [
        "---",
        "",
        "## Q4: ORB Size Interaction",
        "",
        "Delay sweep by G-filter. Full data in retrace_size_interaction.csv.",
        "Shows whether early-retrace advantage concentrates in large ORBs.",
        "",
    ]
    q4_df = pd.DataFrame(q4_rows)
    if not q4_df.empty:
        # Show CORE/PRELIMINARY rows with positive avg_r
        notable = q4_df[
            (q4_df["sample_class"].isin(["CORE", "PRELIMINARY"]))
            & (q4_df["avg_r"].notna())
            & (q4_df["avg_r"] > 0)
        ].sort_values("avg_r", ascending=False).head(20)
        if not notable.empty:
            lines.append("Top positive cells (N>=100):")
            lines.append("")
            lines.append("| Instrument | Session | DST | Filter | Delay | N | avg_r | p_bh |")
            lines.append("|------------|---------|-----|--------|-------|---|-------|------|")
            for _, row in notable.iterrows():
                lines.append(
                    f"| {row['instrument']} | {row['session']} | {row['dst_regime']} | "
                    f"{row['orb_filter']} | {row['delay_bucket']} | {int(row['n'])} | "
                    f"{fmt_r(row['avg_r'])} | {fmt_p(row['pvalue_bh'])} |"
                )
            lines.append("")

    # Q5: Session comparison pivot from Q2 data
    lines += [
        "---",
        "",
        "## Q5: Session Comparison",
        "",
        "avg_r at each hold checkpoint by session (rows with N >= 30).",
        "",
    ]
    if not q2_df.empty:
        q2_pivot = q2_df[q2_df["n"] >= 30].copy()
        q2_pivot["sess_dst"] = q2_pivot["session"] + "/" + q2_pivot["dst_regime"]
        for inst in instruments:
            inst_q2 = q2_pivot[q2_pivot["instrument"] == inst]
            if inst_q2.empty:
                continue
            lines.append(f"**{inst}**")
            lines.append("")
            sess_cols = sorted(inst_q2["sess_dst"].unique())
            header = "| hold_min |" + "".join(f" {s} |" for s in sess_cols)
            sep = "|----------|" + "".join("--------|" for _ in sess_cols)
            lines.append(header)
            lines.append(sep)
            for cp in CHECKPOINTS:
                row_vals = []
                for sd in sess_cols:
                    match = inst_q2[(inst_q2["sess_dst"] == sd) & (inst_q2["hold_minutes"] == cp)]
                    if not match.empty and not np.isnan(match.iloc[0]["avg_r"]):
                        row_vals.append(f"{match.iloc[0]['avg_r']:+.4f}")
                    else:
                        row_vals.append("   N/A")
                lines.append("| " + str(cp) + " |" + "".join(f" {v} |" for v in row_vals))
            lines.append("")

    # Q6 summary
    lines += [
        "---",
        "",
        "## Q6: Regime Conditioning",
        "",
        "ATR split at session-specific 50th percentile per (instrument, session, dst_regime).",
        "EXPANSION = atr_20 > median; CONTRACTION = atr_20 <= median.",
        "Full data in retrace_regime_conditioning.csv.",
        "",
    ]
    q6_df = pd.DataFrame(q6_rows)
    if not q6_df.empty:
        notable_q6 = q6_df[
            (q6_df["sample_class"].isin(["CORE", "PRELIMINARY"]))
            & (q6_df["avg_r"].notna())
            & (q6_df["avg_r"] > 0)
        ].sort_values("avg_r", ascending=False).head(20)
        if not notable_q6.empty:
            lines.append("Positive CORE/PRELIMINARY cells by ATR regime:")
            lines.append("")
            lines.append("| Instrument | Session | DST | ATR | Delay | N | avg_r | p_bh |")
            lines.append("|------------|---------|-----|-----|-------|---|-------|------|")
            for _, row in notable_q6.iterrows():
                lines.append(
                    f"| {row['instrument']} | {row['session']} | {row['dst_regime']} | "
                    f"{row['atr_regime']} | {row['delay_bucket']} | {int(row['n'])} | "
                    f"{fmt_r(row['avg_r'])} | {fmt_p(row['pvalue_bh'])} |"
                )
            lines.append("")

    # Sensitivity
    lines += [
        "---",
        "",
        "## Sensitivity Check (±20% scan window)",
        "",
        "Q1 re-run at scan_window = 192 min and 288 min vs 240 min baseline.",
        "ROBUST = top delay bucket unchanged; FRAGILE = top bucket changes.",
        "",
    ]
    for note in sensitivity_notes:
        lines.append(note)
    if not sensitivity_notes:
        lines.append("*(Sensitivity check not computed — insufficient data.)*")
    lines.append("")

    # Mandatory disclosures
    lines += [
        "---",
        "",
        "## Mandatory Disclosures",
        "",
        f"- **N trades:** {total_fills:,} total fills across all instruments/sessions",
        f"- **Time period:** {date_range}",
        "- **IS/OOS:** IN-SAMPLE only. No walk-forward, no held-out test set.",
        f"- **Variations tested:** scan_window {{192, {SCAN_WINDOW}, 288}}, G-filters {{G4, G5, G6, G8}},",
        f"  hold checkpoints {CHECKPOINTS}",
        "- **Mechanism required for any SURVIVED finding:**",
        "  Early retraces capture mean-reversion in the immediate post-break period",
        "  before momentum carries price away. Late retraces may represent failed breaks",
        "  where price hasn't committed directionally, increasing adverse selection risk.",
        "- **What kills this:**",
        "  Changes to E3_RETRACE_WINDOW_MINUTES in production would change fill counts.",
        "  Low-volatility regimes reduce orb_size → G4+ filter excludes more days.",
        "  Any regime change in post-break price behavior would invalidate timing priors.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E3 retrace timing research — 6-question analysis"
    )
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to gold.db (default: auto-resolve via pipeline.paths)")
    parser.add_argument("--instruments", nargs="+", default=DEFAULT_INSTRUMENTS,
                        help=f"Instruments to scan (default: {DEFAULT_INSTRUMENTS})")
    parser.add_argument("--sessions", nargs="+", default=list(SESSIONS.keys()),
                        choices=list(SESSIONS.keys()),
                        help="Sessions to scan (default: all)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    instruments = args.instruments
    sessions_to_run = args.sessions

    print(f"\n{'=' * 90}")
    print(f"  E3 RETRACE TIMING RESEARCH")
    print(f"  Database:    {db_path}")
    print(f"  Instruments: {instruments}")
    print(f"  Sessions:    {sessions_to_run}")
    print(f"  Scan window: {SCAN_WINDOW} min | RR: {RR_TARGET:.1f} | G4+ base filter")
    print(f"  scipy:       {HAS_SCIPY}")
    print(f"{'=' * 90}\n")

    con = duckdb.connect(str(db_path), read_only=True)
    all_records = []
    date_range = "unknown"
    sensitivity_notes = []
    t_total = time.time()

    try:
        for instrument in instruments:
            print(f"  [{instrument}] Loading 1m bars...")
            t0 = time.time()
            bars_df = load_bars(con, instrument)
            if bars_df.empty:
                print(f"    No bars found for {instrument}, skipping.")
                continue
            print(f"    {len(bars_df):,} bars in {time.time() - t0:.1f}s")

            t0 = time.time()
            all_days, day_to_idx, opens, highs, lows, closes = build_day_arrays(bars_df)
            del bars_df, opens  # opens not needed for retrace scan
            n_days = len(all_days)

            if n_days == 0:
                continue

            print(f"    {n_days} trading days ({all_days[0]} to {all_days[-1]}) "
                  f"in {time.time() - t0:.1f}s")
            date_range = f"{all_days[0]} to {all_days[-1]}"

            us_mask, uk_mask = build_dst_masks(all_days)

            for session in sessions_to_run:
                if session not in SESSIONS:
                    print(f"    [{session}] Unknown session, skipping.")
                    continue

                print(f"  [{instrument}/{session}] Loading daily_features...")
                t0 = time.time()
                df_features = load_daily_features(con, instrument, session)
                if df_features.empty:
                    print(f"    No daily_features rows for {instrument}/{session}.")
                    continue
                print(f"    {len(df_features)} break-days loaded")

                t0 = time.time()
                records = collect_results(
                    instrument, session,
                    all_days, day_to_idx,
                    highs, lows, closes,
                    us_mask, uk_mask,
                    df_features,
                    scan_window=SCAN_WINDOW,
                )
                n_fills = len(records)
                n_break_days = len(df_features)
                fill_rate = n_fills / n_break_days * 100 if n_break_days > 0 else 0
                print(f"    {n_fills} fills / {n_break_days} break-days "
                      f"({fill_rate:.1f}% retrace rate) in {time.time() - t0:.1f}s")

                all_records.extend(records)

                # Sensitivity check (only for sessions with enough data)
                if n_fills >= 100:
                    sens = run_sensitivity(
                        instrument, session,
                        all_days, day_to_idx,
                        highs, lows, closes,
                        us_mask, uk_mask,
                        df_features,
                    )
                    # Find top delay bucket at baseline window
                    base_buckets = sens.get(SCAN_WINDOW, {})
                    if base_buckets:
                        top_key = max(base_buckets, key=base_buckets.get)
                        top_base = base_buckets[top_key]
                        top_192 = sens.get(192, {}).get(top_key)
                        top_288 = sens.get(288, {}).get(top_key)

                        if top_192 is not None and top_288 is not None:
                            robust = top_192 > 0 and top_288 > 0 and top_base > 0
                            verdict = "ROBUST" if robust else "FRAGILE"
                            note = (
                                f"- **{instrument}/{session}** top bucket "
                                f"`{top_key[1]}` ({top_key[0]}): "
                                f"base={top_base:+.4f}, "
                                f"192={top_192:+.4f}, "
                                f"288={top_288:+.4f} → **{verdict}**"
                            )
                            sensitivity_notes.append(note)

            del highs, lows, closes

    finally:
        con.close()

    if not all_records:
        print("\n  No fills collected. Check data and session settings.")
        return

    total_fills = len(all_records)
    print(f"\n  Total fills collected: {total_fills:,}")
    print(f"  Running aggregations...")

    df = pd.DataFrame(all_records)

    # Filter to requested sessions
    df = df[df["session"].isin(sessions_to_run)]

    t0 = time.time()
    q1_rows = compute_q1(df)
    q2_rows = compute_q2(df)
    q3_rows = compute_q3(df)
    q4_rows = compute_q4(df)
    q6_rows = compute_q6(df)
    print(f"  Aggregations done in {time.time() - t0:.1f}s")

    # Save CSVs
    output_dir = PROJECT_ROOT / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_specs = [
        ("retrace_delay_sweep.csv",         q1_rows),
        ("hold_duration_decay.csv",          q2_rows),
        ("delay_hold_heatmap.csv",           q3_rows),
        ("retrace_size_interaction.csv",     q4_rows),
        ("retrace_regime_conditioning.csv",  q6_rows),
    ]
    for fname, rows in csv_specs:
        path = output_dir / fname
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
        print(f"  CSV saved: {path} ({len(rows)} rows)")

    # Generate markdown
    md = generate_markdown(
        q1_rows, q2_rows, q3_rows, q4_rows, q6_rows,
        instruments=instruments,
        sessions=sessions_to_run,
        total_fills=total_fills,
        date_range=date_range,
        sensitivity_notes=sensitivity_notes,
    )
    md_path = output_dir / "retrace_timing_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  MD  saved: {md_path}")

    # Console summary
    print(f"\n{'=' * 90}")
    print(f"  QUICK SUMMARY - Q1 (pnl@{PRIMARY_CHECKPOINT}min by delay bucket)")
    print(f"{'=' * 90}")
    print(f"  {'Inst':>4}  {'Sess':>4}  {'DST':>6}  {'Delay':>6}  "
          f"{'N':>5}  {'Class':>11}  {'avg_r':>8}  {'p_bh':>7}  {'pct%':>5}")
    print(f"  {'-' * 85}")
    for r in q1_rows:
        avg_str = f"{r['avg_r']:+8.4f}" if not np.isnan(r.get("avg_r", np.nan)) else "      --"
        p_str   = f"{r['pvalue_bh']:7.4f}" if not np.isnan(r.get("pvalue_bh", np.nan)) else "    --"
        print(f"  {r['instrument']:>4}  {r['session']:>4}  {r['dst_regime']:>6}  "
              f"{r['delay_bucket']:>6}  {r['n']:>5}  {r['sample_class']:>11}  "
              f"{avg_str}  {p_str}  {r['pct_of_fills']:>5.1f}%")

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print(f"  Outputs: {output_dir}\n")


if __name__ == "__main__":
    main()
