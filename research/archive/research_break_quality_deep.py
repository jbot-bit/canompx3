#!/usr/bin/env python3
"""
Break Quality Deep Dive -- cross-instrument, DST-split, filter-interaction,
early-exit simulation, and statistical robustness (Parts 1-10).

Expands research_break_quality.py to:
  - 4 instruments (MGC, MES, MNQ, MCL)
  - DST regime splits for affected sessions (0900, 1800)
  - ORB filter interaction (G4/G6/G8)
  - Combo conditions (C5 AND C6 together)
  - Optimal C6 window sweep (1-10 bars)
  - Year-over-year stability
  - Early-exit P&L simulation (C5 and C6 rules)
  - Benjamini-Hochberg FDR correction throughout

CRITICAL FIX vs original:
  The JOIN in load_break_context() now includes:
    AND oo.orb_minutes = df.orb_minutes
  Without this, daily_features (3 rows per trading_day per symbol) triples
  the row count and creates entirely spurious correlations.

Usage:
  python research/research_break_quality_deep.py --audit
  python research/research_break_quality_deep.py --instruments MGC --sessions 1000
  python research/research_break_quality_deep.py
  python research/research_break_quality_deep.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from scipy.stats import ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, risk_in_dollars
from pipeline.dst import DST_AFFECTED_SESSIONS

BRISBANE_TZ = ZoneInfo("Australia/Brisbane")
UTC_TZ = ZoneInfo("UTC")

# =========================================================================
# Constants
# =========================================================================

DEFAULT_INSTRUMENTS = ["MGC", "MES", "MNQ", "MCL"]
DEFAULT_SESSIONS = ["0900", "1000", "1800"]
MCL_NOTE = "MCL: context only -- permanently NO-GO per TRADING_RULES.md"

# Entry params (fixed -- not grid-searched)
ENTRY_MODEL = "E1"
CONFIRM_BARS = 2
RR_TARGET = 2.0
ORB_MINUTES = 5

# C6 windows to test
C6_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 10]

# Filter thresholds
G_FILTERS = {"G4": 4.0, "G6": 6.0, "G8": 8.0}

# Minimum N per group for reporting
MIN_N = 30


# =========================================================================
# Statistical utilities (sourced from research_signal_stacking.py)
# =========================================================================

def classify_sample(n: int) -> str:
    """Per RESEARCH_RULES.md thresholds."""
    if n < 30:
        return "INVALID"
    elif n < 100:
        return "REGIME"
    elif n < 200:
        return "PRELIMINARY"
    else:
        return "CORE"


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

    bh = np.zeros(m)
    bh[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        bh[i] = min(bh[i + 1], sorted_p[i] * m / (i + 1))

    bh = np.clip(bh, 0.0, 1.0)
    unsorted = np.zeros(m)
    unsorted[sorted_order] = bh
    adjusted[valid_idx] = unsorted
    return adjusted.tolist()


def compute_pvalue(pnl: pd.Series) -> float:
    """One-sample t-test against H0: mean = 0."""
    if not HAS_SCIPY:
        return np.nan
    valid = pnl.dropna()
    if len(valid) < 2:
        return np.nan
    _, p = ttest_1samp(valid.values, 0.0)
    return float(p)


def compute_stats(pnl: pd.Series) -> dict:
    """Compute n, avg_r, win_rate, tot_r, sharpe, p_value, sample_class."""
    valid = pnl.dropna()
    n = len(valid)
    if n == 0:
        return dict(n=0, avg_r=np.nan, win_rate=np.nan, tot_r=np.nan,
                    sharpe=np.nan, p_value=np.nan, sample_class="INVALID")
    avg = float(valid.mean())
    win_rate = float((valid > 0).sum() / n)
    tot_r = float(valid.sum())
    std = float(valid.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(avg / std * np.sqrt(252)) if std > 0 else np.nan
    p_value = compute_pvalue(valid)
    return dict(n=n, avg_r=avg, win_rate=win_rate, tot_r=tot_r,
                sharpe=sharpe, p_value=p_value, sample_class=classify_sample(n))


def bootstrap_ci(pnl: pd.Series, n_boot: int = 1000) -> tuple[float, float]:
    """95% CI on mean via bootstrap resampling."""
    valid = pnl.dropna().values
    if len(valid) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(valid, size=len(valid), replace=True).mean()
                  for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (lo, hi)


def permutation_test(group_a: pd.Series, group_b: pd.Series, n_perm: int = 1000) -> float:
    """Two-sample permutation test: p-value for observed mean delta."""
    a = group_a.dropna().values
    b = group_b.dropna().values
    if len(a) < 5 or len(b) < 5:
        return np.nan
    obs_delta = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_delta = combined[:na].mean() - combined[na:].mean()
        if abs(perm_delta) >= abs(obs_delta):
            count += 1
    return float(count / n_perm)


# =========================================================================
# Data loading
# =========================================================================

def load_break_context(con, instrument: str, session: str) -> pd.DataFrame:
    """Load outcomes joined with ORB context from daily_features.

    CRITICAL: JOIN includes AND oo.orb_minutes = df.orb_minutes
    Without it, daily_features (3 rows per day: orb_minutes=5/15/30)
    triples the row count, creating entirely spurious statistics.

    Note: day_of_week and risk_dollars are not stored in the DB --
    they are computed in Python after loading.
    """
    query = f"""
        SELECT
            oo.trading_day,
            df.orb_{session}_high      AS orb_high,
            df.orb_{session}_low       AS orb_low,
            df.orb_{session}_size      AS orb_size,
            df.orb_{session}_break_dir AS break_dir,
            df.orb_{session}_break_ts  AS break_ts,
            df.us_dst,
            df.uk_dst,
            EXTRACT(YEAR FROM oo.trading_day) AS year,
            oo.entry_ts,
            oo.entry_price,
            oo.stop_price,
            oo.target_price,
            oo.pnl_r,
            oo.outcome
        FROM orb_outcomes oo
        JOIN daily_features df
            ON oo.symbol = df.symbol
            AND oo.trading_day = df.trading_day
            AND oo.orb_minutes = df.orb_minutes   -- CRITICAL: prevents 3x row inflation
        WHERE oo.symbol = ?
          AND oo.orb_label = ?
          AND oo.entry_model = ?
          AND oo.rr_target = ?
          AND oo.confirm_bars = ?
          AND oo.pnl_r IS NOT NULL
          AND oo.orb_minutes = {ORB_MINUTES}
          AND df.orb_minutes = {ORB_MINUTES}
          AND df.orb_{session}_break_dir IS NOT NULL
        ORDER BY oo.trading_day
    """
    df = con.execute(query, [instrument, session, ENTRY_MODEL, RR_TARGET, CONFIRM_BARS]).fetchdf()
    if not df.empty:
        # Compute day_of_week from trading_day (0=Mon, 4=Fri)
        df["day_of_week"] = pd.to_datetime(df["trading_day"]).dt.dayofweek
        # risk_dollars not in DB -- will be computed per-trade in Part 7 via risk_in_dollars()
        df["risk_dollars"] = np.nan
    return df


def load_bars_for_instrument(con, instrument: str,
                              date_min, date_max) -> dict:
    """Bulk-load all 1m bars for an instrument into a dict keyed by trading day.

    Single query per instrument (~160MB) vs per-trade queries (~0.3s vs 19s).

    Trading day assignment:
      Brisbane trading day = UTC date shifted forward 1h.
      ts_utc + 1h -> calendar date = Brisbane trading day.
      This correctly maps 23:00 UTC -> 00:00 next day = Brisbane 09:00 = session open.
    """
    # Extend range by 1 day on each side for boundary safety
    td_min = date_min if isinstance(date_min, date) else date_min.date()
    td_max = date_max if isinstance(date_max, date) else date_max.date()
    bris_start = datetime(td_min.year, td_min.month, td_min.day, 9, 0, 0,
                          tzinfo=BRISBANE_TZ) - timedelta(days=1)
    bris_end = datetime(td_max.year, td_max.month, td_max.day, 9, 0, 0,
                        tzinfo=BRISBANE_TZ) + timedelta(days=2)
    utc_start = bris_start.astimezone(UTC_TZ).replace(tzinfo=None)
    utc_end = bris_end.astimezone(UTC_TZ).replace(tzinfo=None)

    bars_all = con.execute("""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
    """, [instrument, utc_start, utc_end]).fetchdf()

    if bars_all.empty:
        return {}

    # Shift UTC by 1h to compute Brisbane trading day date
    bars_all["_td"] = (bars_all["ts_utc"] + pd.Timedelta(hours=1)).dt.date
    result = {
        td: grp.drop(columns="_td").reset_index(drop=True)
        for td, grp in bars_all.groupby("_td")
    }
    return result


# =========================================================================
# DST regime helper
# =========================================================================

def build_dst_regime_column(df: pd.DataFrame, session: str) -> pd.Series:
    """Map rows to 'winter' / 'summer' / 'clean' based on session and DST columns.

    NaN DST values are left as NaN (not silently coerced to 'winter') so
    that callers can detect and warn about missing DST data rather than
    contaminating the winter group with unknown-regime rows.
    """
    dst_type = DST_AFFECTED_SESSIONS.get(session)
    if dst_type == "US":
        result = df["us_dst"].map({True: "summer", False: "winter"})
        n_nan = result.isna().sum()
        if n_nan > 0:
            print(f"  [WARN] {session}: {n_nan} rows have NaN us_dst -- excluded from DST splits")
        return result
    if dst_type == "UK":
        result = df["uk_dst"].map({True: "summer", False: "winter"})
        n_nan = result.isna().sum()
        if n_nan > 0:
            print(f"  [WARN] {session}: {n_nan} rows have NaN uk_dst -- excluded from DST splits")
        return result
    return pd.Series("clean", index=df.index)


# =========================================================================
# Condition extraction (single-pass)
# =========================================================================

def extract_conditions_v2(bars_day: pd.DataFrame, row: dict,
                           c6_windows: list) -> dict | None:
    """Extract all break-quality conditions for a single trade.

    Single pass over bars_day for all C6 windows simultaneously.
    Extends original C1-C6 with per-window c6_exit_close for Part 7.

    Returns dict or None if data insufficient.
    """
    orb_high = row["orb_high"]
    orb_low = row["orb_low"]
    orb_size = orb_high - orb_low
    break_dir = row["break_dir"]
    entry_ts = row["entry_ts"]

    if orb_size <= 0 or pd.isna(entry_ts):
        return None

    entry_ts_pd = pd.Timestamp(entry_ts)

    # Confirm bar = last bar before entry
    pre_entry = bars_day[bars_day["ts_utc"] < entry_ts_pd].sort_values("ts_utc")
    if pre_entry.empty:
        return None
    cb = pre_entry.iloc[-1]
    confirm_ts_pd = cb["ts_utc"]

    # Entry bar = bar at entry_ts
    entry_bar = bars_day[bars_day["ts_utc"] == entry_ts_pd]
    if entry_bar.empty:
        return None
    eb = entry_bar.iloc[0]

    # Post-entry bars (for C6)
    max_window = max(c6_windows)
    post_entry = bars_day[bars_day["ts_utc"] > entry_ts_pd].sort_values("ts_utc").head(max_window)

    result = {
        "trading_day": row["trading_day"],
        "year": int(row["year"]),
        "instrument": row.get("instrument", ""),
        "session": row.get("session", ""),
        "dst_regime": row.get("_dst_regime", "clean"),
        "day_of_week": row.get("day_of_week"),
        "pnl_r": row["pnl_r"],
        "outcome": row["outcome"],
        "break_dir": break_dir,
        "orb_size": orb_size,
        "entry_price": row["entry_price"],
        "stop_price": row["stop_price"],
        "risk_dollars": row.get("risk_dollars"),
        "eb_open": float(eb["open"]),
        "eb_close": float(eb["close"]),
    }

    # --- C1: Break distance (confirm bar close vs ORB boundary, in R) ---
    if break_dir == "long":
        break_distance = (cb["close"] - orb_high) / orb_size
    else:
        break_distance = (orb_low - cb["close"]) / orb_size
    result["c1_break_distance_r"] = float(break_distance)

    # --- C2: Confirm bar wick ratio (wick back toward ORB / bar range) ---
    bar_range = cb["high"] - cb["low"]
    if bar_range > 0:
        if break_dir == "long":
            body_low = min(cb["open"], cb["close"])
            wick_back = body_low - cb["low"]
        else:
            body_high = max(cb["open"], cb["close"])
            wick_back = cb["high"] - body_high
        result["c2_wick_ratio"] = float(max(0.0, wick_back) / bar_range)
    else:
        result["c2_wick_ratio"] = 0.0

    # --- C3: Break speed (minutes from break_ts to confirm bar) ---
    break_ts = row.get("break_ts")
    if break_ts is not None and not pd.isna(break_ts):
        break_ts_pd = pd.Timestamp(break_ts)
        minutes = (confirm_ts_pd - break_ts_pd).total_seconds() / 60
        result["c3_break_speed_min"] = float(minutes)
    else:
        result["c3_break_speed_min"] = np.nan

    # --- C4: Entry bar engulfs back inside ORB ---
    if break_dir == "long":
        result["c4_entry_bar_engulf"] = bool(eb["close"] <= orb_high)
    else:
        result["c4_entry_bar_engulf"] = bool(eb["close"] >= orb_low)

    # --- C5: Entry bar continues in break direction ---
    if break_dir == "long":
        result["c5_entry_bar_continues"] = bool(eb["close"] > eb["open"])
    else:
        result["c5_entry_bar_continues"] = bool(eb["close"] < eb["open"])

    # --- C6: Any close back inside ORB within N bars (single pass) ---
    for n in c6_windows:
        check = post_entry.head(n)
        col = f"c6_reversal_{n}bar"
        exit_col = f"c6_exit_close_{n}bar"
        if len(check) == 0:
            result[col] = None
            result[exit_col] = np.nan
            continue
        if break_dir == "long":
            inside_mask = check["close"] <= orb_high
        else:
            inside_mask = check["close"] >= orb_low
        any_inside = bool(inside_mask.any())
        result[col] = any_inside
        if any_inside:
            # Close of first bar that comes back inside ORB
            first_idx = inside_mask.idxmax()
            result[exit_col] = float(check.loc[first_idx, "close"])
        else:
            result[exit_col] = np.nan

    return result


# =========================================================================
# Shared helpers
# =========================================================================

def split_stats(df: pd.DataFrame, cond_col: str,
                min_n: int = MIN_N) -> dict | None:
    """Compute stats for condition True vs False groups."""
    if cond_col not in df.columns:
        return None
    mask = df[cond_col] == True  # noqa: E712
    gt = df[mask]["pnl_r"]
    gf = df[~mask]["pnl_r"]
    if len(gt) < min_n or len(gf) < min_n:
        return None
    st = compute_stats(gt)
    sf = compute_stats(gf)
    delta = st["avg_r"] - sf["avg_r"]
    return {"true": st, "false": sf, "delta": delta,
            "perm_p": np.nan}  # perm_p filled later


def print_split(label: str, result: dict, note: str = "") -> None:
    """Print a split comparison row."""
    st = result["true"]
    sf = result["false"]
    delta = result["delta"]
    sig = ""
    if abs(delta) >= 0.15:
        sig = "<< ACTIONABLE" if delta < 0 else ">> ACTIONABLE"
    elif abs(delta) >= 0.08:
        sig = "<- notable"
    print(f"  {label:<30s}  TRUE  N={st['n']:4d} avgR={st['avg_r']:+.3f} "
          f"WR={st['win_rate']:.1%}  |  FALSE  N={sf['n']:4d} avgR={sf['avg_r']:+.3f} "
          f"WR={sf['win_rate']:.1%}  |  delta={delta:+.3f}  {sig}{note}")


# =========================================================================
# AUDIT MODE -- verify JOIN inflation fix
# =========================================================================

def run_audit(con, instrument: str = "MGC", session: str = "1000") -> None:
    """Audit Part 9: verify the orb_minutes JOIN fix prevents inflation."""
    print(f"\n{'=' * 80}")
    print(f"  AUDIT MODE -- {instrument} {session}")
    print(f"{'=' * 80}")

    # Query WITHOUT the orb_minutes join (original buggy version)
    broken_query = f"""
        SELECT COUNT(*) AS n
        FROM orb_outcomes oo
        JOIN daily_features df
            ON oo.symbol = df.symbol
            AND oo.trading_day = df.trading_day
        WHERE oo.symbol = '{instrument}'
          AND oo.orb_label = '{session}'
          AND oo.entry_model = '{ENTRY_MODEL}'
          AND oo.rr_target = {RR_TARGET}
          AND oo.confirm_bars = {CONFIRM_BARS}
          AND oo.pnl_r IS NOT NULL
          AND df.orb_{session}_break_dir IS NOT NULL
    """

    # Query WITH the orb_minutes join (corrected version)
    fixed_query = f"""
        SELECT COUNT(*) AS n
        FROM orb_outcomes oo
        JOIN daily_features df
            ON oo.symbol = df.symbol
            AND oo.trading_day = df.trading_day
            AND oo.orb_minutes = df.orb_minutes
        WHERE oo.symbol = '{instrument}'
          AND oo.orb_label = '{session}'
          AND oo.entry_model = '{ENTRY_MODEL}'
          AND oo.rr_target = {RR_TARGET}
          AND oo.confirm_bars = {CONFIRM_BARS}
          AND oo.pnl_r IS NOT NULL
          AND oo.orb_minutes = {ORB_MINUTES}
          AND df.orb_minutes = {ORB_MINUTES}
          AND df.orb_{session}_break_dir IS NOT NULL
    """

    n_broken = con.execute(broken_query).fetchone()[0]
    n_fixed = con.execute(fixed_query).fetchone()[0]
    ratio = n_broken / n_fixed if n_fixed > 0 else float("nan")

    print(f"\n  Row count WITHOUT orb_minutes join: {n_broken:,}")
    print(f"  Row count WITH    orb_minutes join: {n_fixed:,}")
    print(f"  Inflation ratio: {ratio:.2f}x")
    if ratio >= 2.0:
        print(f"  [OK] Significant inflation confirmed ({ratio:.2f}x) -- fix is critical")
        print("    (ratio < 3.0 expected when some orb_minutes rows are missing)")
    elif ratio > 1.0:
        print(f"  ! Mild inflation ({ratio:.2f}x) -- may indicate filtered data")
    else:
        print("  [OK] No inflation -- fix working correctly")

    # Verify G4+ reduces N (never increases it)
    df_full = load_break_context(con, instrument, session)
    n_all = len(df_full)
    df_g4 = df_full[df_full["orb_size"] >= 4.0]
    n_g4 = len(df_g4)
    print(f"\n  All breaks: N={n_all}")
    print(f"  G4+  breaks: N={n_g4}")
    if n_g4 <= n_all:
        print("  [OK] G4+ filter correctly reduces N")
    else:
        print("  [!] ERROR: G4+ increased N -- investigate!")

    # Verify DST routing
    df_full["_dst"] = build_dst_regime_column(df_full, session)
    dst_counts = df_full["_dst"].value_counts()
    print(f"\n  DST regime distribution for session {session}:")
    for regime, cnt in dst_counts.items():
        print(f"    {regime}: {cnt}")
    dst_type = DST_AFFECTED_SESSIONS.get(session, "CLEAN")
    print(f"  Expected DST type: {dst_type}")
    if dst_type in ("US", "UK"):
        if "winter" in dst_counts.index and "summer" in dst_counts.index:
            print("  [OK] Winter/summer split confirmed")
        else:
            print("  ! Missing one DST regime")
    else:
        if "clean" in dst_counts.index:
            print("  [OK] Clean session confirmed (no DST split)")
        else:
            print("  ! Expected 'clean' regime")

    print(f"\n{'=' * 80}")
    print("  Audit complete.")
    print(f"{'=' * 80}")


# =========================================================================
# Part 1: Cross-Instrument Validation
# =========================================================================

def part1_cross_instrument(all_df: pd.DataFrame) -> list[dict]:
    """4 instruments x sessions x {C5, C6_3bar} at G4+ baseline."""
    print(f"\n{'=' * 80}")
    print("  PART 1: Cross-Instrument Validation (G4+, C5 and C6_3bar)")
    print(f"{'=' * 80}")

    rows = []
    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()

    for inst in sorted(df_g4["instrument"].unique()):
        for sess in sorted(df_g4[df_g4["instrument"] == inst]["session"].unique()):
            sub = df_g4[(df_g4["instrument"] == inst) & (df_g4["session"] == sess)]
            if len(sub) < MIN_N * 2:
                continue

            note = "  [NO-GO: see TRADING_RULES.md]" if inst == "MCL" else ""
            header = f"\n  {inst} {sess} (N={len(sub)}){note}"
            print(header)

            for cond_col, label_true, label_false in [
                ("c5_entry_bar_continues", "C5=True (continues)", "C5=False (reverses)"),
                ("c6_reversal_3bar", "C6_3bar=True (reversal)", "C6_3bar=False (stays out)"),
            ]:
                res = split_stats(sub, cond_col)
                if res is None:
                    print(f"    {cond_col}: insufficient N in one group, skip")
                    continue
                print_split(cond_col, res, note)
                rows.append({
                    "part": "P1",
                    "instrument": inst,
                    "session": sess,
                    "dst_regime": "all",
                    "g_filter": "G4",
                    "condition": cond_col,
                    "n_true": res["true"]["n"],
                    "avg_r_true": res["true"]["avg_r"],
                    "n_false": res["false"]["n"],
                    "avg_r_false": res["false"]["avg_r"],
                    "delta": res["delta"],
                    "p_value": np.nan,
                    "p_bh": np.nan,
                    "sample_class_true": res["true"]["sample_class"],
                    "sample_class_false": res["false"]["sample_class"],
                    "mcl_nogo": inst == "MCL",
                })

    return rows


# =========================================================================
# Part 2: DST Splits (0900 and 1800 only)
# =========================================================================

def part2_dst_splits(all_df: pd.DataFrame) -> list[dict]:
    """DST-split analysis for all DST-affected sessions (0900, 0030, 2300, 1800).

    Per CLAUDE.md: ANY analysis of 0900/1800/0030/2300 MUST split by DST regime.
    Sessions not present in all_df are silently skipped.
    """
    print(f"\n{'=' * 80}")
    print("  PART 2: DST Splits (0900, 0030, 2300, 1800)")
    print(f"{'=' * 80}")

    rows = []
    # All sessions affected by DST per pipeline/dst.py DST_AFFECTED_SESSIONS
    dst_sessions = {"0900": "US", "0030": "US", "2300": "US", "1800": "UK"}
    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()

    for sess, dst_type in dst_sessions.items():
        sub_sess = df_g4[df_g4["session"] == sess]
        if sub_sess.empty:
            continue

        # Print blended warning first
        print(f"\n  [WARNING: blended -- see Part 2] {sess} blended (DST={dst_type}):")
        for inst in sorted(sub_sess["instrument"].unique()):
            sub = sub_sess[sub_sess["instrument"] == inst]
            bl = compute_stats(sub["pnl_r"])
            print(f"    {inst} blended: N={bl['n']} avgR={bl['avg_r']:+.3f} -- "
                  f"MISLEADING, use DST splits below")

        for inst in sorted(sub_sess["instrument"].unique()):
            sub_inst = sub_sess[sub_sess["instrument"] == inst]
            print(f"\n  {inst} {sess} by {dst_type} DST regime:")
            for regime in ["winter", "summer"]:
                sub = sub_inst[sub_inst["dst_regime"] == regime]
                if len(sub) < 10:
                    print(f"    {regime}: N={len(sub)} -- too small")
                    continue
                bl = compute_stats(sub["pnl_r"])
                print(f"    {regime}: N={bl['n']} avgR={bl['avg_r']:+.3f} "
                      f"WR={bl['win_rate']:.1%} [{bl['sample_class']}]")

                for cond_col in ["c5_entry_bar_continues", "c6_reversal_3bar"]:
                    res = split_stats(sub, cond_col, min_n=15)
                    if res is None:
                        continue
                    rows.append({
                        "part": "P2",
                        "instrument": inst,
                        "session": sess,
                        "dst_regime": regime,
                        "g_filter": "G4",
                        "condition": cond_col,
                        "n_true": res["true"]["n"],
                        "avg_r_true": res["true"]["avg_r"],
                        "n_false": res["false"]["n"],
                        "avg_r_false": res["false"]["avg_r"],
                        "delta": res["delta"],
                        "p_value": np.nan,
                        "p_bh": np.nan,
                        "sample_class_true": res["true"]["sample_class"],
                        "sample_class_false": res["false"]["sample_class"],
                        "mcl_nogo": inst == "MCL",
                    })
                    print_split(f"  {regime} {cond_col}", res)

    return rows


# =========================================================================
# Part 3: Filter Interaction (G4, G6, G8)
# =========================================================================

def part3_filter_interaction(all_df: pd.DataFrame) -> list[dict]:
    """C5 and C6_3bar delta at G4/G6/G8 per (instrument, session)."""
    print(f"\n{'=' * 80}")
    print("  PART 3: Filter Interaction (G4/G6/G8)")
    print(f"{'=' * 80}")

    rows = []
    print(f"\n  {'Instrument':>10} {'Session':>7} {'Condition':>25}  "
          f"{'G4 delta':>8} {'G6 delta':>8} {'G8 delta':>8}  Trend")

    for inst in sorted(all_df["instrument"].unique()):
        for sess in sorted(all_df[all_df["instrument"] == inst]["session"].unique()):
            sub_all = all_df[(all_df["instrument"] == inst) & (all_df["session"] == sess)]

            for cond_col in ["c5_entry_bar_continues", "c6_reversal_3bar"]:
                deltas = {}
                for g_label, g_thresh in G_FILTERS.items():
                    sub = sub_all[sub_all["orb_size"] >= g_thresh]
                    res = split_stats(sub, cond_col, min_n=20)
                    deltas[g_label] = res["delta"] if res else np.nan
                    if res:
                        rows.append({
                            "part": "P3",
                            "instrument": inst,
                            "session": sess,
                            "dst_regime": "all",
                            "g_filter": g_label,
                            "condition": cond_col,
                            "n_true": res["true"]["n"],
                            "avg_r_true": res["true"]["avg_r"],
                            "n_false": res["false"]["n"],
                            "avg_r_false": res["false"]["avg_r"],
                            "delta": res["delta"],
                            "p_value": np.nan,
                            "p_bh": np.nan,
                            "sample_class_true": res["true"]["sample_class"],
                            "sample_class_false": res["false"]["sample_class"],
                            "mcl_nogo": inst == "MCL",
                        })

                d4 = deltas.get("G4", np.nan)
                d6 = deltas.get("G6", np.nan)
                d8 = deltas.get("G8", np.nan)
                # Trend: do larger ORBs amplify or dampen the signal?
                valid_ds = [d for d in [d4, d6, d8] if not np.isnan(d)]
                trend = ""
                if len(valid_ds) >= 2:
                    if valid_ds[-1] > valid_ds[0] + 0.05:
                        trend = "amplifies ^"
                    elif valid_ds[-1] < valid_ds[0] - 0.05:
                        trend = "dampens v"
                    else:
                        trend = "stable"
                print(f"  {inst:>10} {sess:>7} {cond_col:>25}  "
                      f"{d4:>+8.3f} {d6:>+8.3f} {d8:>+8.3f}  {trend}")

    return rows


# =========================================================================
# Part 4: C5 + C6 Combination
# =========================================================================

def part4_combo(all_df: pd.DataFrame) -> list[dict]:
    """Five groups: C5+C6_3bar, C5+C6_5bar, C5 alone, C6_3bar alone, baseline."""
    print(f"\n{'=' * 80}")
    print("  PART 4: C5 + C6 Combination Groups")
    print(f"{'=' * 80}")

    rows = []
    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()

    for inst in sorted(df_g4["instrument"].unique()):
        for sess in sorted(df_g4[df_g4["instrument"] == inst]["session"].unique()):
            sub = df_g4[(df_g4["instrument"] == inst) & (df_g4["session"] == sess)]
            if len(sub) < MIN_N:
                continue

            print(f"\n  {inst} {sess} (N={len(sub)}):")
            groups = [
                ("C5=T + C6_3bar=F (continues + stays out 3bar)",
                 (sub["c5_entry_bar_continues"] == True) & (sub["c6_reversal_3bar"] == False)),
                ("C5=T + C6_5bar=F (continues + stays out 5bar)",
                 (sub["c5_entry_bar_continues"] == True) & (sub["c6_reversal_5bar"] == False)),
                ("C5=T alone (continues)",
                 sub["c5_entry_bar_continues"] == True),
                ("C6_3bar=F alone (stays outside 3bar)",
                 sub["c6_reversal_3bar"] == False),
                ("Baseline (all breaks)", pd.Series(True, index=sub.index)),
            ]

            baseline_avgr = compute_stats(sub["pnl_r"])["avg_r"]
            for label, mask in groups:
                g = sub[mask]["pnl_r"]
                st = compute_stats(g)
                delta_vs_base = st["avg_r"] - baseline_avgr if not np.isnan(st["avg_r"]) else np.nan
                flag = " <" if (not np.isnan(delta_vs_base) and delta_vs_base > 0.10) else ""
                print(f"    {label:<50s}  N={st['n']:4d} avgR={st['avg_r']:+.3f} "
                      f"WR={st['win_rate']:.1%} deltavs_base={delta_vs_base:+.3f}{flag}")
                rows.append({
                    "part": "P4",
                    "instrument": inst,
                    "session": sess,
                    "group": label,
                    "n": st["n"],
                    "avg_r": st["avg_r"],
                    "win_rate": st["win_rate"],
                    "tot_r": st["tot_r"],
                    "delta_vs_baseline": delta_vs_base,
                    "sample_class": st["sample_class"],
                    "mcl_nogo": inst == "MCL",
                })

    return rows


# =========================================================================
# Part 5: Optimal C6 Window
# =========================================================================

def part5_optimal_c6_window(all_df: pd.DataFrame) -> list[dict]:
    """Sweep C6 windows 1-10 per (instrument, session)."""
    print(f"\n{'=' * 80}")
    print("  PART 5: Optimal C6 Window (1-10 bars)")
    print(f"{'=' * 80}")

    rows = []
    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()

    for inst in sorted(df_g4["instrument"].unique()):
        for sess in sorted(df_g4[df_g4["instrument"] == inst]["session"].unique()):
            sub = df_g4[(df_g4["instrument"] == inst) & (df_g4["session"] == sess)]
            if len(sub) < MIN_N * 2:
                continue

            print(f"\n  {inst} {sess}: window -> N_stayed_out / avgR_good / avgR_reversal / delta")
            prev_delta = None
            for n in C6_WINDOWS:
                col = f"c6_reversal_{n}bar"
                if col not in sub.columns:
                    continue
                res = split_stats(sub, col, min_n=15)
                if res is None:
                    print(f"    {n:>2}bar: insufficient N")
                    continue
                # c6_reversal=False = "stays outside" = "good" group
                delta = res["delta"]  # True - False
                gain = ""
                if prev_delta is not None and abs(delta - prev_delta) < 0.01:
                    gain = "  -> elbow"
                print(f"    {n:>2}bar: N_out={res['false']['n']:4d} "
                      f"avgR_out={res['false']['avg_r']:+.3f}  "
                      f"N_rev={res['true']['n']:4d} "
                      f"avgR_rev={res['true']['avg_r']:+.3f}  "
                      f"delta={delta:+.3f}{gain}")
                rows.append({
                    "part": "P5",
                    "instrument": inst,
                    "session": sess,
                    "window_bars": n,
                    "n_stayed_outside": res["false"]["n"],
                    "avg_r_outside": res["false"]["avg_r"],
                    "n_reversal": res["true"]["n"],
                    "avg_r_reversal": res["true"]["avg_r"],
                    "delta": delta,
                    "mcl_nogo": inst == "MCL",
                })
                prev_delta = delta

    return rows


# =========================================================================
# Part 6: Year-over-Year Stability
# =========================================================================

def part6_yoy_stability(all_df: pd.DataFrame, min_abs_delta: float = 0.10) -> list[dict]:
    """Year-by-year breakdown for signals with |delta| > min_abs_delta."""
    print(f"\n{'=' * 80}")
    print(f"  PART 6: Year-over-Year Stability (|delta| > {min_abs_delta})")
    print(f"{'=' * 80}")

    rows = []
    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()

    for inst in sorted(df_g4["instrument"].unique()):
        for sess in sorted(df_g4[df_g4["instrument"] == inst]["session"].unique()):
            sub = df_g4[(df_g4["instrument"] == inst) & (df_g4["session"] == sess)]
            if len(sub) < MIN_N * 2:
                continue

            for cond_col in ["c5_entry_bar_continues", "c6_reversal_3bar"]:
                # Check if this combo has a signal worth decomposing
                overall = split_stats(sub, cond_col, min_n=MIN_N)
                if overall is None or abs(overall["delta"]) < min_abs_delta:
                    continue

                print(f"\n  {inst} {sess} {cond_col} (overall delta={overall['delta']:+.3f}):")
                print(f"    {'Year':>4} {'N_tot':>6} {'N_true':>7} {'avgR_T':>8} "
                      f"{'avgR_F':>8} {'delta':>7} flag")

                for yr in sorted(sub["year"].dropna().unique()):
                    sub_yr = sub[sub["year"] == yr]
                    res = split_stats(sub_yr, cond_col, min_n=10)
                    if res is None:
                        print(f"    {int(yr):4d}  insufficient N, skip")
                        continue
                    flip = " FLIP" if (res["delta"] * overall["delta"]) < 0 else ""
                    print(f"    {int(yr):4d}  {len(sub_yr):>6d}  {res['true']['n']:>7d}  "
                          f"{res['true']['avg_r']:>+8.3f}  {res['false']['avg_r']:>+8.3f}  "
                          f"{res['delta']:>+7.3f}{flip}")
                    rows.append({
                        "part": "P6",
                        "instrument": inst,
                        "session": sess,
                        "condition": cond_col,
                        "year": int(yr),
                        "n_total": len(sub_yr),
                        "n_true": res["true"]["n"],
                        "avg_r_true": res["true"]["avg_r"],
                        "avg_r_false": res["false"]["avg_r"],
                        "delta": res["delta"],
                        "overall_delta": overall["delta"],
                        "sign_flip": (res["delta"] * overall["delta"]) < 0,
                        "mcl_nogo": inst == "MCL",
                    })

    return rows


# =========================================================================
# Part 7: Early Exit Simulation
# =========================================================================

def part7_early_exit(all_df: pd.DataFrame) -> list[dict]:
    """Simulate early exit on C5 reversal or C6 re-entry.

    C5 rule: exit at entry bar close when entry bar reverses (C5=False).
    C6 rules: exit at close of first bar back inside ORB (at N=3 and N=5 bars).
    """
    print(f"\n{'=' * 80}")
    print("  PART 7: Early Exit Simulation")
    print(f"{'=' * 80}")

    rows = []
    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()

    for inst in sorted(df_g4["instrument"].unique()):
        try:
            spec = get_cost_spec(inst)
        except ValueError:
            print(f"  {inst}: no cost spec, skip")
            continue

        for sess in sorted(df_g4[df_g4["instrument"] == inst]["session"].unique()):
            sub = df_g4[(df_g4["instrument"] == inst) & (df_g4["session"] == sess)].copy()
            if len(sub) < MIN_N:
                continue

            print(f"\n  {inst} {sess} (N={len(sub)}):")

            # --- C5 exit simulation ---
            # Applies to trades where C5=False (entry bar reverses)
            c5_false = sub[sub["c5_entry_bar_continues"] == False].copy()
            if len(c5_false) >= 10:
                def _sim_c5_pnl(row):
                    risk_d = row["risk_dollars"]
                    if pd.isna(risk_d) or risk_d <= 0:
                        risk_d = risk_in_dollars(spec, row["entry_price"], row["stop_price"])
                    if pd.isna(row["eb_close"]) or pd.isna(row["entry_price"]):
                        return np.nan
                    direction = 1 if row["break_dir"] == "long" else -1
                    pnl_pts = (row["eb_close"] - row["entry_price"]) * direction
                    pnl_dollars = pnl_pts * spec.point_value - spec.total_friction
                    r = pnl_dollars / risk_d if risk_d > 0 else np.nan
                    return max(r, -1.0)

                c5_false["sim_pnl_r"] = c5_false.apply(_sim_c5_pnl, axis=1)
                sim_valid = c5_false["sim_pnl_r"].dropna()
                if len(sim_valid) >= 5:
                    orig_avg = c5_false["pnl_r"].mean()
                    sim_avg = sim_valid.mean()
                    orig_tot = c5_false["pnl_r"].sum()
                    sim_tot = sim_valid.sum()
                    improvement = sim_avg - orig_avg
                    print(f"    C5 exit (reversal->exit at eb_close): "
                          f"N={len(sim_valid)} orig_avgR={orig_avg:+.3f} sim_avgR={sim_avg:+.3f} "
                          f"delta={improvement:+.3f} [tot: {orig_tot:+.1f}->{sim_tot:+.1f}]")
                    rows.append({
                        "part": "P7",
                        "instrument": inst,
                        "session": sess,
                        "rule": "C5_exit",
                        "n_modified": len(sim_valid),
                        "avg_r_original": orig_avg,
                        "avg_r_simulated": sim_avg,
                        "improvement": improvement,
                        "tot_r_original": orig_tot,
                        "tot_r_simulated": sim_tot,
                        "mcl_nogo": inst == "MCL",
                    })

            # --- C6 exit simulation at N=3 and N=5 ---
            for n in [3, 5]:
                rev_col = f"c6_reversal_{n}bar"
                exit_col = f"c6_exit_close_{n}bar"
                if rev_col not in sub.columns or exit_col not in sub.columns:
                    continue
                c6_rev = sub[(sub[rev_col] == True) & sub[exit_col].notna()].copy()
                if len(c6_rev) < 10:
                    continue

                def _sim_c6_pnl(row):
                    risk_d = row["risk_dollars"]
                    if pd.isna(risk_d) or risk_d <= 0:
                        risk_d = risk_in_dollars(spec, row["entry_price"], row["stop_price"])
                    exit_price = row[exit_col]
                    if pd.isna(exit_price) or pd.isna(row["entry_price"]):
                        return np.nan
                    direction = 1 if row["break_dir"] == "long" else -1
                    pnl_pts = (exit_price - row["entry_price"]) * direction
                    pnl_dollars = pnl_pts * spec.point_value - spec.total_friction
                    r = pnl_dollars / risk_d if risk_d > 0 else np.nan
                    return max(r, -1.0)

                c6_rev["sim_pnl_r"] = c6_rev.apply(
                    lambda row: _sim_c6_pnl(row), axis=1
                )
                sim_valid = c6_rev["sim_pnl_r"].dropna()
                if len(sim_valid) >= 5:
                    orig_avg = c6_rev["pnl_r"].mean()
                    sim_avg = sim_valid.mean()
                    orig_tot = c6_rev["pnl_r"].sum()
                    sim_tot = sim_valid.sum()
                    improvement = sim_avg - orig_avg
                    print(f"    C6_{n}bar exit (reversal->exit at first inside close): "
                          f"N={len(sim_valid)} orig_avgR={orig_avg:+.3f} sim_avgR={sim_avg:+.3f} "
                          f"delta={improvement:+.3f}")
                    rows.append({
                        "part": "P7",
                        "instrument": inst,
                        "session": sess,
                        "rule": f"C6_{n}bar_exit",
                        "n_modified": len(sim_valid),
                        "avg_r_original": orig_avg,
                        "avg_r_simulated": sim_avg,
                        "improvement": improvement,
                        "tot_r_original": orig_tot,
                        "tot_r_simulated": sim_tot,
                        "mcl_nogo": inst == "MCL",
                    })

    return rows


# =========================================================================
# Part 10: Statistical Robustness
# =========================================================================

def part10_statistical_robustness(all_df: pd.DataFrame,
                                   actionable_rows: list[dict]) -> list[dict]:
    """Bootstrap CI and permutation test for signals with |delta| > 0.15."""
    print(f"\n{'=' * 80}")
    print("  PART 10: Statistical Robustness (|delta| > 0.15)")
    print(f"{'=' * 80}")

    if not actionable_rows:
        print("  No actionable signals to test.")
        return []

    df_g4 = all_df[all_df["orb_size"] >= G_FILTERS["G4"]].copy()
    results = []
    p_values_for_bh = []

    unique_combos = set()
    for r in actionable_rows:
        combo = (r.get("instrument"), r.get("session"), r.get("condition", r.get("group", "")))
        unique_combos.add(combo)

    print(f"\n  Testing {len(unique_combos)} unique (instrument x session x condition) combos:")

    for (inst, sess, cond_col) in sorted(unique_combos):
        if not cond_col.startswith("c5") and not cond_col.startswith("c6"):
            continue

        sub = df_g4[(df_g4["instrument"] == inst) & (df_g4["session"] == sess)]
        if len(sub) < MIN_N * 2:
            continue
        if cond_col not in sub.columns:
            continue

        mask_true = sub[cond_col] == True  # noqa: E712
        pnl_true = sub[mask_true]["pnl_r"]
        pnl_false = sub[~mask_true]["pnl_r"]

        if len(pnl_true) < 10 or len(pnl_false) < 10:
            continue

        # Bootstrap CIs
        ci_true = bootstrap_ci(pnl_true)
        ci_false = bootstrap_ci(pnl_false)
        delta = pnl_true.mean() - pnl_false.mean()

        # Bootstrap CI on delta
        all_vals_t = pnl_true.dropna().values
        all_vals_f = pnl_false.dropna().values
        rng = np.random.default_rng(42)
        boot_deltas = []
        for _ in range(1000):
            bt = rng.choice(all_vals_t, size=len(all_vals_t), replace=True).mean()
            bf = rng.choice(all_vals_f, size=len(all_vals_f), replace=True).mean()
            boot_deltas.append(bt - bf)
        ci_delta_lo = float(np.percentile(boot_deltas, 2.5))
        ci_delta_hi = float(np.percentile(boot_deltas, 97.5))

        # Permutation test
        perm_p = permutation_test(pnl_true, pnl_false)
        p_values_for_bh.append(perm_p)

        reliable = "RELIABLE" if (ci_delta_lo > 0) or (ci_delta_hi < 0) else "UNRELIABLE"

        results.append({
            "instrument": inst,
            "session": sess,
            "condition": cond_col,
            "n_true": len(pnl_true),
            "n_false": len(pnl_false),
            "avg_r_true": float(pnl_true.mean()),
            "avg_r_false": float(pnl_false.mean()),
            "delta": delta,
            "ci_delta_lo": ci_delta_lo,
            "ci_delta_hi": ci_delta_hi,
            "perm_p": perm_p,
            "p_bh": np.nan,
            "reliability": reliable,
        })

    # Apply BH correction
    if p_values_for_bh:
        bh_corrected = bh_fdr_correction(p_values_for_bh)
        for i, r in enumerate(results):
            r["p_bh"] = bh_corrected[i]
            bh_flag = " BH-SIG" if (not np.isnan(bh_corrected[i]) and bh_corrected[i] < 0.05) else ""
            print(f"\n  {r['instrument']} {r['session']} {r['condition']}:")
            print(f"    delta={r['delta']:+.3f} [{r['ci_delta_lo']:+.3f}, {r['ci_delta_hi']:+.3f}] "
                  f"perm_p={r['perm_p']:.4f} p_bh={r['p_bh']:.4f} "
                  f"{r['reliability']}{bh_flag}")

    return results


# =========================================================================
# Summary markdown builder
# =========================================================================

def build_summary_md(
    p1_rows: list[dict],
    p2_rows: list[dict],
    p3_rows: list[dict],
    p4_rows: list[dict],
    p5_rows: list[dict],
    p6_rows: list[dict],
    p7_rows: list[dict],
    p10_rows: list[dict],
    instruments: list[str],
    sessions: list[str],
) -> str:
    lines = []
    lines.append("# Break Quality Deep Dive -- Summary")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Instruments:** {', '.join(instruments)}")
    lines.append(f"**Sessions:** {', '.join(sessions)}")
    lines.append(f"**Entry params:** {ENTRY_MODEL} / CB={CONFIRM_BARS} / RR={RR_TARGET}")
    lines.append(f"**ORB filter:** G4+ baseline ({', '.join(G_FILTERS.keys())} tested in P3)")
    lines.append(f"**MCL note:** {MCL_NOTE}")
    lines.append("")

    # SURVIVED / DID NOT SURVIVE
    lines.append("## SURVIVED SCRUTINY")
    lines.append("")
    survived_any = False
    for r in p10_rows:
        if r.get("p_bh", 1.0) < 0.05 and r.get("reliability") == "RELIABLE":
            lines.append(
                f"- {r['instrument']} {r['session']} {r['condition']}: "
                f"delta={r['delta']:+.3f} [{r['ci_delta_lo']:+.3f},{r['ci_delta_hi']:+.3f}] "
                f"p_bh={r['p_bh']:.4f} [BH-SIG + RELIABLE]"
            )
            survived_any = True
    if not survived_any:
        lines.append("None -- no signal survived BH FDR + bootstrap reliability check.")
    lines.append("")

    lines.append("## DID NOT SURVIVE")
    lines.append("")
    n_tested = len(p10_rows)
    n_reliable = sum(1 for r in p10_rows if r.get("reliability") == "RELIABLE")
    n_bh_sig = sum(1 for r in p10_rows if r.get("p_bh", 1.0) < 0.05)
    lines.append(f"- {n_tested} combos tested in Part 10")
    lines.append(f"- {n_reliable}/{n_tested} RELIABLE (bootstrap CI doesn't cross 0)")
    lines.append(f"- {n_bh_sig}/{n_tested} BH-significant (p_bh < 0.05)")
    lines.append("")

    lines.append("## CAVEATS")
    lines.append("")
    lines.append("1. **In-sample only.** No OOS validation.")
    lines.append("2. **Fixed params** (E1/CB2/RR2.0). Signals may differ at other params.")
    lines.append("3. **C6 windows are post-entry** (observable, not look-ahead). C5 is also post-entry.")
    lines.append("4. **Part 7 simulation** assumes instant execution at bar close -- slippage not modeled beyond standard friction.")
    lines.append("5. **MCL permanently excluded** from trading per TRADING_RULES.md.")
    lines.append("6. **DST blended numbers** for 0900/1800 are misleading -- use Part 2 splits.")
    lines.append("")

    lines.append("## Part 8: Literature (manual step -- see instructions)")
    lines.append("")
    lines.append("After running this script, perform web searches:")
    lines.append("- 'opening range breakout entry bar confirmation'")
    lines.append("- 'breakout failure retest ORB'")
    lines.append("- 'breakout follow-through entry candle futures'")
    lines.append("- 'ORB breakout quality indicators academic'")
    lines.append("")
    lines.append("Update this section with findings after the manual search step.")
    lines.append("")

    lines.append("## Part 10 Statistical Robustness Table")
    lines.append("")
    if p10_rows:
        lines.append("| instrument | session | condition | N_true | N_false | delta | CI_lo | CI_hi | perm_p | p_bh | reliability |")
        lines.append("|-----------|---------|-----------|--------|---------|---|-------|-------|--------|------|-------------|")
        for r in p10_rows:
            lines.append(
                f"| {r['instrument']} | {r['session']} | {r['condition']} | "
                f"{r['n_true']} | {r['n_false']} | {r['delta']:+.3f} | "
                f"{r['ci_delta_lo']:+.3f} | {r['ci_delta_hi']:+.3f} | "
                f"{r['perm_p']:.4f} | {r['p_bh']:.4f} | {r['reliability']} |"
            )
    else:
        lines.append("No combos reached actionable threshold for robustness testing.")
    lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("- If C5 or C6 signals are RELIABLE + BH-SIG: implement as early-exit overlay (not a new strategy)")
    lines.append("- P5 elbow point -> recommended window for C6 exit rule")
    lines.append("- P7 improvement > 0.10R AND N >= 30 -> worth implementing in paper trading")
    lines.append("- Literature (P8) -> check if C5/C6 patterns match documented breakout failure modes")
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Break Quality Deep Dive Research")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--instruments", nargs="+", default=None,
                        help="Instruments to analyze (default: MGC MES MNQ MCL)")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Sessions to analyze (default: 0900 1000 1800)")
    parser.add_argument("--min-orb-size", type=float, default=4.0,
                        help="Min ORB size filter (default: 4.0 = G4+)")
    parser.add_argument("--audit", action="store_true",
                        help="Run JOIN inflation audit only, then exit")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    instruments = args.instruments or DEFAULT_INSTRUMENTS
    sessions = args.sessions or DEFAULT_SESSIONS
    min_orb = args.min_orb_size

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        if args.audit:
            run_audit(con, instrument="MGC", session="1000")
            return

        print(f"\n{'=' * 80}")
        print("  BREAK QUALITY DEEP DIVE")
        print(f"  DB: {db_path}")
        print(f"  Instruments: {instruments}  |  Sessions: {sessions}")
        print(f"  Entry: {ENTRY_MODEL} / CB={CONFIRM_BARS} / RR={RR_TARGET} / ORB={ORB_MINUTES}m")
        print(f"  Min ORB (G-filter): {min_orb}")
        print(f"  scipy available: {HAS_SCIPY}")
        print(f"{'=' * 80}")

        t_start = time.time()
        all_conditions = []

        # --- Main data collection loop ---
        for instrument in instruments:
            print(f"\n[{instrument}] Loading bar data (bulk preload)...")
            t_bars = time.time()

            # Determine date range across ALL sessions for this instrument
            # (single pass -- bars are loaded once per instrument, not per session)
            all_dates = []
            for session in sessions:
                oc = load_break_context(con, instrument, session)
                if not oc.empty:
                    all_dates.append(oc["trading_day"].min())
                    all_dates.append(oc["trading_day"].max())
            if not all_dates:
                print(f"  [{instrument}] No outcomes found for any session, skip.")
                continue

            bars_dict = load_bars_for_instrument(con, instrument,
                                                  min(all_dates), max(all_dates))
            print(f"  [{instrument}] Bars loaded for {len(bars_dict)} trading days "
                  f"in {time.time() - t_bars:.1f}s")

            # --- Condition extraction per session ---
            for session in sessions:
                print(f"  Loading outcomes: {instrument} {session}...")
                outcomes = load_break_context(con, instrument, session)
                if outcomes.empty:
                    print(f"  No outcomes found for {instrument} {session}, skip.")
                    continue
                outcomes = outcomes[outcomes["orb_size"] >= min_orb].copy()
                if len(outcomes) < MIN_N:
                    print(f"  Only {len(outcomes)} outcomes at G{int(min_orb)}+, skip.")
                    continue

                outcomes["_dst_regime"] = build_dst_regime_column(outcomes, session)
                outcomes["instrument"] = instrument
                outcomes["session"] = session

                n_skipped = 0
                for _, row in outcomes.iterrows():
                    td = row["trading_day"]
                    if isinstance(td, pd.Timestamp):
                        td = td.date()
                    bars_day = bars_dict.get(td)
                    if bars_day is None or bars_day.empty:
                        n_skipped += 1
                        continue
                    cond = extract_conditions_v2(bars_day, row.to_dict(), C6_WINDOWS)
                    if cond is not None:
                        all_conditions.append(cond)
                    else:
                        n_skipped += 1

                extracted = len(outcomes) - n_skipped
                print(f"  {instrument} {session}: {extracted}/{len(outcomes)} conditions extracted "
                      f"({n_skipped} skipped)")

        if not all_conditions:
            print("\nNo conditions extracted. Check DB and instrument/session filters.")
            return

        all_df = pd.DataFrame(all_conditions)
        print(f"\n[Data] Total: {len(all_df)} trade conditions across "
              f"{all_df['instrument'].nunique()} instruments, "
              f"{all_df['session'].nunique()} sessions")
        print(f"       Period: {all_df['trading_day'].min()} - {all_df['trading_day'].max()}")

        # --- Run all parts ---
        p1_rows = part1_cross_instrument(all_df)
        p2_rows = part2_dst_splits(all_df)
        p3_rows = part3_filter_interaction(all_df)
        p4_rows = part4_combo(all_df)
        p5_rows = part5_optimal_c6_window(all_df)
        p6_rows = part6_yoy_stability(all_df)
        p7_rows = part7_early_exit(all_df)

        # Collect actionable signals (|delta| > 0.15, N >= 30 in both groups)
        actionable = [
            r for r in (p1_rows + p2_rows + p3_rows)
            if (abs(r.get("delta", 0)) > 0.15
                and r.get("n_true", 0) >= MIN_N
                and r.get("n_false", 0) >= MIN_N
                and not r.get("mcl_nogo", False))
        ]
        print(f"\n  Actionable signals for Part 10: {len(actionable)}")

        p10_rows = part10_statistical_robustness(all_df, actionable)

        # --- Save outputs ---
        out_dir = Path("research/output")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save trade-level CSV
        csv_cols = [
            "trading_day", "year", "instrument", "session", "dst_regime", "day_of_week",
            "pnl_r", "outcome", "break_dir", "orb_size",
            "entry_price", "stop_price", "risk_dollars", "eb_open", "eb_close",
            "c1_break_distance_r", "c2_wick_ratio", "c3_break_speed_min",
            "c4_entry_bar_engulf", "c5_entry_bar_continues",
        ]
        for n in C6_WINDOWS:
            csv_cols.append(f"c6_reversal_{n}bar")
        for n in C6_WINDOWS:
            csv_cols.append(f"c6_exit_close_{n}bar")

        csv_df = all_df[[c for c in csv_cols if c in all_df.columns]]
        csv_path = out_dir / "break_quality_deep.csv"
        csv_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n  CSV saved: {csv_path} ({len(csv_df)} rows)")

        # Save summary markdown
        md = build_summary_md(
            p1_rows, p2_rows, p3_rows, p4_rows, p5_rows, p6_rows, p7_rows, p10_rows,
            instruments, sessions,
        )
        md_path = out_dir / "break_quality_deep_summary.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"  Markdown saved: {md_path}")

        print(f"\n  Total elapsed: {time.time() - t_start:.1f}s")
        print(f"\n{'=' * 80}")
        print("  INTERPRETATION GUIDE")
        print(f"{'=' * 80}")
        print("""
  C5 (entry bar direction): c5_entry_bar_continues=True means the entry bar closes
    further in the break direction. If False (reversal) has much worse avgR -> consider
    C5 as an early-exit signal.

  C6 (quick reversal): c6_reversal_Nbar=True means at least one of the first N bars
    closed back inside the ORB. The group where c6=False (stayed outside) should have
    better avgR. P5 identifies the optimal window.

  Part 7: Simulation cap of -1.0R on exits prevents unrealistic results.
    Improvement > 0.10R AND N >= 30 -> worth paper trading.

  Part 10 reliability labels:
    RELIABLE: 95% CI on delta does not cross 0 (bootstrap-confirmed directionality)
    UNRELIABLE: CI crosses 0 (direction uncertain despite large delta)
    BH-SIG: perm_p survives Benjamini-Hochberg FDR correction (p_bh < 0.05)
""")

    finally:
        con.close()


if __name__ == "__main__":
    main()
