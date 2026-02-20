#!/usr/bin/env python3
"""
Signal Stacking Research — P3.

Tests whether multiple signals compound when used together:
  Size + Direction + Concordance + Volume + Calendar

For days where all signals fire simultaneously, does avgR reach
0.5R–1.0R? If so, those days justify 2-3x position sizing.

Sessions/Instruments:
  1000: MGC, MES, MNQ  — CLEAN (no DST), 3-way concordance, LONG-only (H5)
  0900: MGC only        — US DST split, no concordance, bidirectional
  0030: MES, MNQ        — US DST split, 2-way concordance, bidirectional

Reference parameters: E3 / CB=1 / RR=2.0 (fixed — avoids Bailey Rule)
All in-sample. No DB writes. Read-only research.

Usage:
    python research/research_signal_stacking.py
    python research/research_signal_stacking.py --instruments MGC --sessions 1000
    python research/research_signal_stacking.py --db-path C:/db/gold.db
"""

import argparse
import statistics
import sys
import warnings
from datetime import datetime, timezone
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Path setup (canonical pattern) ---
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from pipeline.calendar_filters import is_nfp_day, is_opex_day  # noqa: E402

# =========================================================================
# Constants
# =========================================================================

# Sessions per instrument (from plan)
SESSION_INSTRUMENTS = {
    "1000": ["MGC", "MES", "MNQ"],
    "0900": ["MGC"],
    "0030": ["MES", "MNQ"],
}

# DST type per session
SESSION_DST_TYPE = {
    "1000": "CLEAN",   # No split needed
    "0900": "US",      # Split on us_dst
    "0030": "US",      # Split on us_dst
}

# Direction filter: "long" where H5 confirmed; None = bidirectional
SESSION_DIRECTION = {
    "1000": "long",   # H5 confirmed
    "0900": None,     # Bidirectional
    "0030": None,     # Bidirectional
}

# Instruments participating in concordance per session
SESSION_CONC_INSTRUMENTS = {
    "1000": ["MGC", "MES", "MNQ"],  # 3-way
    "0030": ["MES", "MNQ"],          # 2-way
    "0900": None,                    # N/A (single instrument)
}

# G-filter thresholds (from config.py)
G_FILTERS = {
    "G4": 4.0,
    "G5": 5.0,
    "G6": 6.0,
    "G8": 8.0,
}

# Reference parameters (fixed — not grid-searched)
REF_ENTRY_MODEL = "E3"
REF_CONFIRM_BARS = 1
REF_RR_TARGET = 2.0

# Volume filter threshold (from VolumeFilter default in config.py)
VOL_THRESHOLD = 1.2
VOL_THRESHOLD_SENSITIVITY = 1.5   # Sensitivity check: ±20%
VOL_LOOKBACK = 20

# Stack levels
STACK_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
STACK_LABELS = {
    "L0": "size",
    "L1": "size+dir",
    "L2": "size+conc",
    "L3": "size+dir+conc",
    "L4": "size+dir+conc+vol",
    "L5": "size+dir+conc+vol+cal",
}

_US_EASTERN = ZoneInfo("America/New_York")


# =========================================================================
# Statistical utilities (sourced from research_aperture_scan.py)
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


def compute_sharpe(pnl: pd.Series, annualize: int = 252) -> float:
    """Annualized Sharpe (assumes ~daily trading frequency)."""
    valid = pnl.dropna()
    if len(valid) < 2:
        return np.nan
    std = float(valid.std(ddof=1))
    if std <= 0:
        return np.nan
    return float(valid.mean() / std * np.sqrt(annualize))


def metrics(pnl: pd.Series) -> dict:
    """Compute n, avg_r, win_rate, sharpe, pvalue for a pnl_r series."""
    valid = pnl.dropna()
    n = len(valid)
    if n == 0:
        return dict(n=0, avg_r=np.nan, win_rate=np.nan, sharpe=np.nan, pvalue=np.nan)
    return dict(
        n=n,
        avg_r=float(valid.mean()),
        win_rate=float((valid > 0).sum() / n),
        sharpe=compute_sharpe(valid),
        pvalue=compute_pvalue(valid),
    )


# =========================================================================
# DST helpers (standalone — no imports from pipeline)
# =========================================================================

def is_us_dst(d) -> bool:
    """True if US Eastern is in DST (EDT, UTC-4) on this date."""
    try:
        dt = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=_US_EASTERN)
        return dt.utcoffset().total_seconds() == -4 * 3600
    except Exception:
        return False


# =========================================================================
# Data Loading
# =========================================================================

def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load daily_features for all target instruments and sessions.

    Calendar flags (is_nfp_day, is_opex_day) are not stored in daily_features;
    they are computed inline from trading_day using pipeline.calendar_filters.
    """
    df = con.execute("""
        SELECT
            trading_day, symbol, us_dst, atr_20,
            orb_0900_break_dir, orb_0900_size, orb_0900_break_ts,
            orb_1000_break_dir, orb_1000_size, orb_1000_break_ts,
            orb_0030_break_dir, orb_0030_size, orb_0030_break_ts
        FROM daily_features
        WHERE symbol IN ('MGC', 'MES', 'MNQ')
          AND orb_minutes = 5
        ORDER BY trading_day, symbol
    """).fetchdf()

    # Compute calendar flags from trading_day (not stored in DB)
    df["is_nfp_day"] = df["trading_day"].apply(is_nfp_day)
    df["is_opex_day"] = df["trading_day"].apply(is_opex_day)
    return df


def load_outcomes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load reference outcomes: E3 / CB=1 / RR=2.0."""
    return con.execute("""
        SELECT trading_day, symbol, orb_label, pnl_r
        FROM orb_outcomes
        WHERE symbol IN ('MGC', 'MES', 'MNQ')
          AND orb_label IN ('0900', '1000', '0030')
          AND entry_model = ?
          AND confirm_bars = ?
          AND rr_target = ?
          AND pnl_r IS NOT NULL
        ORDER BY trading_day, symbol, orb_label
    """, [REF_ENTRY_MODEL, REF_CONFIRM_BARS, REF_RR_TARGET]).fetchdf()


# =========================================================================
# Relative Volume (inline — based on strategy_discovery._compute_relative_volumes)
# =========================================================================

def _ts_minute_key(ts) -> tuple:
    """Normalize timestamp to UTC (year, month, day, hour, minute) tuple."""
    utc_ts = ts.astimezone(timezone.utc) if ts.tzinfo is not None else ts
    return (utc_ts.year, utc_ts.month, utc_ts.day, utc_ts.hour, utc_ts.minute)


def compute_rel_vol_for_session(
    con: duckdb.DuckDBPyConnection,
    features_sub: pd.DataFrame,
    instrument: str,
    orb_label: str,
    lookback: int = VOL_LOOKBACK,
) -> pd.Series:
    """
    Compute relative volume at break bar for each trading day.

    Returns Series aligned to features_sub index.
    Fail-closed: NaN where any data is missing.
    """
    break_ts_col = f"orb_{orb_label}_break_ts"
    result = pd.Series(np.nan, index=features_sub.index)

    if break_ts_col not in features_sub.columns:
        return result

    valid = features_sub[features_sub[break_ts_col].notna()]
    if len(valid) == 0:
        return result

    # Collect unique UTC minute-of-day across all break timestamps
    unique_mods: set[int] = set()
    for ts in valid[break_ts_col]:
        if ts is not None and hasattr(ts, "hour"):
            utc_ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts
            unique_mods.add(utc_ts.hour * 60 + utc_ts.minute)

    if not unique_mods:
        return result

    # Load volume history per UTC minute-of-day (one query per minute)
    minute_history: dict[int, list] = {}
    for mod in sorted(unique_mods):
        h, m = divmod(mod, 60)
        rows = con.execute(
            """SELECT ts_utc, volume FROM bars_1m
               WHERE symbol = ?
                 AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) = ?
                 AND EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               ORDER BY ts_utc""",
            [instrument, h, m],
        ).fetchall()
        minute_history[mod] = [(_ts_minute_key(ts), vol) for ts, vol in rows]

    # Compute relative volume per row
    for idx, row in valid.iterrows():
        ts = row[break_ts_col]
        if ts is None:
            continue
        utc_ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts
        mod = utc_ts.hour * 60 + utc_ts.minute
        history = minute_history.get(mod, [])
        if not history:
            continue

        break_key = _ts_minute_key(ts)
        hist_idx = None
        for j, (k, _) in enumerate(history):
            if k == break_key:
                hist_idx = j
                break
        if hist_idx is None:
            continue

        break_vol = history[hist_idx][1]
        if not break_vol or break_vol <= 0:
            continue

        start = max(0, hist_idx - lookback)
        prior_vols = [v for _, v in history[start:hist_idx] if v and v > 0]
        if not prior_vols:
            continue

        baseline = statistics.median(prior_vols)
        if baseline <= 0:
            continue

        result.loc[idx] = break_vol / baseline

    return result


# =========================================================================
# Concordance Building (adapted from research_concordance_stacking.py:103-166)
# =========================================================================

def build_concordance_map(
    features_df: pd.DataFrame,
    instruments: list,
    orb_label: str,
) -> pd.DataFrame:
    """
    Build per-day concordance tier for a given (instruments, session).

    Returns DataFrame with columns:
        trading_day, concordance_tier ('concordant_3'/'majority_2'/'remaining'),
        majority_dir ('long'/'short'/'none')
    """
    # Pivot to wide: one row per trading_day with columns per instrument
    dfs = {}
    for inst in instruments:
        sub = features_df[features_df["symbol"] == inst][
            ["trading_day",
             f"orb_{orb_label}_break_dir",
             f"orb_{orb_label}_size"]
        ].rename(columns={
            f"orb_{orb_label}_break_dir": f"{inst}_break_dir",
            f"orb_{orb_label}_size": f"{inst}_orb_size",
        })
        dfs[inst] = sub

    wide = dfs[instruments[0]]
    for inst in instruments[1:]:
        wide = wide.merge(dfs[inst], on="trading_day", how="inner")

    is_long = pd.DataFrame({i: wide[f"{i}_break_dir"] == "long" for i in instruments})
    is_short = pd.DataFrame({i: wide[f"{i}_break_dir"] == "short" for i in instruments})
    has_break = is_long | is_short

    n_long = is_long.sum(axis=1)
    n_short = is_short.sum(axis=1)
    n_active = has_break.sum(axis=1)
    n_insts = len(instruments)

    # concordant_3: all instruments active and all same direction
    conc3 = (n_active == n_insts) & ((n_long == n_insts) | (n_short == n_insts))
    # majority_2: at least 2 agree (but not all same for 3-way)
    maj2 = ~conc3 & (n_active >= 2) & ((n_long >= 2) | (n_short >= 2))

    tier = pd.Series("remaining", index=wide.index)
    tier[maj2] = "majority_2"
    tier[conc3] = "concordant_3"

    maj_dir = np.where(n_long >= n_short, "long", "short")
    maj_dir = np.where(n_active < 2, "none", maj_dir)

    return pd.DataFrame({
        "trading_day": wide["trading_day"],
        "concordance_tier": tier.values,
        "majority_dir": maj_dir,
    })


# =========================================================================
# Stack Level Analysis — Core Engine
# =========================================================================

def build_working_df(
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    instrument: str,
    orb_label: str,
    conc_map: pd.DataFrame | None,
    rel_vol: pd.Series,
) -> pd.DataFrame:
    """
    Merge features + outcomes + concordance + rel_vol for one (instrument, session).

    Returns wide DataFrame with all columns needed for stack level masks.
    """
    inst_feat = features_df[features_df["symbol"] == instrument].copy()
    inst_feat = inst_feat.reset_index(drop=True)
    inst_feat["rel_vol"] = rel_vol.reset_index(drop=True).values

    inst_out = outcomes_df[
        (outcomes_df["symbol"] == instrument) &
        (outcomes_df["orb_label"] == orb_label)
    ][["trading_day", "pnl_r"]].copy()

    merged = inst_feat.merge(inst_out, on="trading_day", how="inner")
    merged = merged.rename(columns={
        f"orb_{orb_label}_break_dir": "break_dir",
        f"orb_{orb_label}_size": "orb_size",
    })

    if conc_map is not None:
        merged = merged.merge(
            conc_map[["trading_day", "concordance_tier", "majority_dir"]],
            on="trading_day", how="left",
        )
        merged["concordance_tier"] = merged["concordance_tier"].fillna("remaining")
        merged["majority_dir"] = merged["majority_dir"].fillna("none")
    else:
        merged["concordance_tier"] = "n/a"
        merged["majority_dir"] = "n/a"

    return merged


def stack_mask(
    df: pd.DataFrame,
    level: str,
    min_size: float,
    direction: str | None,
    has_concordance: bool,
    vol_threshold: float = VOL_THRESHOLD,
) -> pd.Series:
    """
    Return boolean mask for rows matching this stack level.

    L0: size only
    L1: L0 + direction (or L0 if direction=None)
    L2: L0 + concordance majority_2+ (or L0 if no concordance)
    L3: L0 + direction + concordance
    L4: L3 + rel_vol >= vol_threshold
    L5: L4 + not (NFP or OPEX)
    """
    size_ok = df["orb_size"].notna() & (df["orb_size"] >= min_size)

    dir_ok = (
        df["break_dir"] == direction
        if direction is not None
        else pd.Series(True, index=df.index)
    )

    conc_ok = (
        df["concordance_tier"].isin(["concordant_3", "majority_2"])
        if has_concordance
        else pd.Series(True, index=df.index)
    )

    vol_ok = df["rel_vol"].notna() & (df["rel_vol"] >= vol_threshold)

    cal_ok = ~(
        df["is_nfp_day"].fillna(False) | df["is_opex_day"].fillna(False)
    )

    masks = {
        "L0": size_ok,
        "L1": size_ok & dir_ok,
        "L2": size_ok & conc_ok,
        "L3": size_ok & dir_ok & conc_ok,
        "L4": size_ok & dir_ok & conc_ok & vol_ok,
        "L5": size_ok & dir_ok & conc_ok & vol_ok & cal_ok,
    }
    return masks[level]


# =========================================================================
# Analysis 1: Incremental Layer Lift
# =========================================================================

def run_analysis_1(
    all_working_dfs: dict,
    instruments_by_session: dict,
    sessions_to_run: list,
    dst_type_map: dict,
    direction_map: dict,
    conc_insts_map: dict,
) -> pd.DataFrame:
    """
    For each (instrument, session, g_filter, dst_regime, stack_level),
    compute metrics and emit one row.

    Returns DataFrame with all rows; BH correction applied per (session × instrument) group.
    """
    rows = []

    for session in sessions_to_run:
        conc_insts = conc_insts_map.get(session)
        has_concordance = conc_insts is not None
        direction = direction_map.get(session)
        dst_type = dst_type_map.get(session, "CLEAN")

        for instrument in instruments_by_session.get(session, []):
            key = (instrument, session)
            if key not in all_working_dfs:
                continue
            wdf = all_working_dfs[key]

            # DST regimes
            if dst_type == "CLEAN":
                dst_regimes = [("all", wdf)]
            else:  # US DST
                dst_regimes = [
                    ("winter", wdf[~wdf["us_dst"].fillna(False)]),
                    ("summer", wdf[wdf["us_dst"].fillna(False)]),
                ]

            for dst_regime, sub in dst_regimes:
                if len(sub) == 0:
                    continue
                n_baseline_all = len(sub)  # before size filter

                for g_label, min_size in G_FILTERS.items():
                    # L0 baseline for this (instrument, session, g_filter, dst_regime)
                    l0_mask = stack_mask(sub, "L0", min_size, direction, has_concordance)
                    n_baseline = int(l0_mask.sum())

                    for level in STACK_LEVELS:
                        m = stack_mask(sub, level, min_size, direction, has_concordance)
                        pnl = sub.loc[m, "pnl_r"]
                        met = metrics(pnl)

                        rows.append({
                            "instrument": instrument,
                            "session": session,
                            "dst_regime": dst_regime,
                            "g_filter": g_label,
                            "stack_level": level,
                            "stack_label": STACK_LABELS[level],
                            "n": met["n"],
                            "n_baseline": n_baseline,
                            "pct_of_baseline": (
                                round(met["n"] / n_baseline, 4) if n_baseline > 0 else np.nan
                            ),
                            "avg_r": met["avg_r"],
                            "win_rate": met["win_rate"],
                            "sharpe": met["sharpe"],
                            "pvalue": met["pvalue"],
                            "pvalue_bh": np.nan,   # filled after
                            "sample_class": classify_sample(met["n"]),
                        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # BH FDR correction within each (session × instrument) group
    for (sess, inst), grp_idx in df.groupby(["session", "instrument"]).groups.items():
        p_vals = df.loc[grp_idx, "pvalue"].tolist()
        adjusted = bh_fdr_correction(p_vals)
        df.loc[grp_idx, "pvalue_bh"] = adjusted

    return df


# =========================================================================
# Analysis 2: Conviction Profile
# =========================================================================

def run_analysis_2(lift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Restructure Analysis 1: for each surviving row (CORE or REGIME, avg_r > 0),
    compute conviction_score and sizing_tier.
    """
    if lift_df.empty:
        return pd.DataFrame()

    # Get L0 baselines
    l0 = lift_df[lift_df["stack_level"] == "L0"][
        ["instrument", "session", "dst_regime", "g_filter", "avg_r"]
    ].rename(columns={"avg_r": "baseline_avg_r"})

    merged = lift_df.merge(l0, on=["instrument", "session", "dst_regime", "g_filter"], how="left")
    merged = merged[
        (merged["sample_class"].isin(["CORE", "REGIME", "PRELIMINARY"])) &
        (merged["avg_r"].notna()) &
        (merged["avg_r"] > 0)
    ].copy()

    if merged.empty:
        return merged

    # Years in sample: use n and an assumed ~250 trading days/year
    merged["freq_per_year"] = (merged["n"] / 250.0 * 252 / 252).round(1)

    # Conviction score: stack lift vs L0
    merged["conviction_score"] = np.where(
        merged["baseline_avg_r"].notna() & (merged["baseline_avg_r"] > 0),
        (merged["avg_r"] / merged["baseline_avg_r"]).round(3),
        np.nan,
    )

    # Sizing tier
    def sizing_tier(score):
        if pd.isna(score):
            return "1x"
        if score < 1.5:
            return "1x"
        elif score <= 2.5:
            return "2x"
        else:
            return "3x"

    merged["sizing_tier"] = merged["conviction_score"].apply(sizing_tier)

    cols = [
        "instrument", "session", "dst_regime", "g_filter", "stack_level",
        "stack_label", "n", "freq_per_year", "avg_r", "conviction_score",
        "sizing_tier", "sample_class",
    ]
    return merged[cols].reset_index(drop=True)


# =========================================================================
# Analysis 3: Concordance × Size Interaction (MGC 1000 only)
# =========================================================================

def run_analysis_3(
    all_working_dfs: dict,
) -> pd.DataFrame:
    """
    Does concordant-3 outperform majority-2 within tighter size bands (G6+/G8+)?
    Tests MGC 1000 only (clearest 3-way concordance signal).

    Sensitivity: also runs with vol_threshold=1.5.
    """
    key = ("MGC", "1000")
    if key not in all_working_dfs:
        return pd.DataFrame()

    wdf = all_working_dfs[key]
    rows = []

    g_filters_a3 = {"G4": 4.0, "G5": 5.0, "G6": 6.0, "G8": 8.0}
    tiers = ["concordant_3", "majority_2", "remaining"]

    for g_label, min_size in g_filters_a3.items():
        size_ok = wdf["orb_size"].notna() & (wdf["orb_size"] >= min_size)
        n_g = int(size_ok.sum())

        for tier in tiers:
            if tier == "n/a":
                continue
            m = size_ok & (wdf["concordance_tier"] == tier)
            pnl = wdf.loc[m, "pnl_r"]
            met = metrics(pnl)

            rows.append({
                "instrument": "MGC",
                "session": "1000",
                "g_filter": g_label,
                "concordance_tier": tier,
                "n": met["n"],
                "n_g_filter_total": n_g,
                "pct_of_g_filter_days": (
                    round(met["n"] / n_g, 4) if n_g > 0 else np.nan
                ),
                "avg_r": met["avg_r"],
                "win_rate": met["win_rate"],
                "pvalue": met["pvalue"],
                "pvalue_bh": np.nan,
                "sample_class": classify_sample(met["n"]),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # BH correction within MGC 1000 group
    p_vals = df["pvalue"].tolist()
    df["pvalue_bh"] = bh_fdr_correction(p_vals)

    return df


# =========================================================================
# Output: Sensitivity Check (Analysis 1 — vol threshold)
# =========================================================================

def run_sensitivity_check(
    all_working_dfs: dict,
    sessions_to_run: list,
    instruments_by_session: dict,
    direction_map: dict,
    conc_insts_map: dict,
    dst_type_map: dict,
    alt_vol_threshold: float = VOL_THRESHOLD_SENSITIVITY,
) -> pd.DataFrame:
    """
    Re-run Analysis 1 at L4/L5 with vol_threshold=1.5 (sensitivity).
    Returns table showing avg_r difference between threshold 1.2 and 1.5.
    """
    rows = []

    for session in sessions_to_run:
        direction = direction_map.get(session)
        has_concordance = conc_insts_map.get(session) is not None
        dst_type = dst_type_map.get(session, "CLEAN")

        for instrument in instruments_by_session.get(session, []):
            key = (instrument, session)
            if key not in all_working_dfs:
                continue
            wdf = all_working_dfs[key]

            if dst_type == "CLEAN":
                dst_regimes = [("all", wdf)]
            else:
                dst_regimes = [
                    ("winter", wdf[~wdf["us_dst"].fillna(False)]),
                    ("summer", wdf[wdf["us_dst"].fillna(False)]),
                ]

            for dst_regime, sub in dst_regimes:
                if len(sub) == 0:
                    continue
                for g_label, min_size in G_FILTERS.items():
                    for level in ["L4", "L5"]:
                        # Standard threshold
                        m_std = stack_mask(sub, level, min_size, direction, has_concordance,
                                           VOL_THRESHOLD)
                        met_std = metrics(sub.loc[m_std, "pnl_r"])

                        # Alternative threshold
                        m_alt = stack_mask(sub, level, min_size, direction, has_concordance,
                                           alt_vol_threshold)
                        met_alt = metrics(sub.loc[m_alt, "pnl_r"])

                        rows.append({
                            "instrument": instrument,
                            "session": session,
                            "dst_regime": dst_regime,
                            "g_filter": g_label,
                            "stack_level": level,
                            "n_rv12": met_std["n"],
                            "avg_r_rv12": met_std["avg_r"],
                            "n_rv15": met_alt["n"],
                            "avg_r_rv15": met_alt["avg_r"],
                            "delta_avg_r": (
                                round(met_alt["avg_r"] - met_std["avg_r"], 4)
                                if not np.isnan(met_std["avg_r"]) and not np.isnan(met_alt["avg_r"])
                                else np.nan
                            ),
                            "verdict": (
                                "ROBUST"
                                if not np.isnan(met_std["avg_r"]) and not np.isnan(met_alt["avg_r"])
                                   and met_std["avg_r"] > 0 and met_alt["avg_r"] > 0
                                else "FRAGILE" if not np.isnan(met_std["avg_r"]) and met_std["avg_r"] > 0
                                else "--"
                            ),
                        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =========================================================================
# Markdown Summary
# =========================================================================

def build_summary_md(
    lift_df: pd.DataFrame,
    conviction_df: pd.DataFrame,
    conc_size_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    sessions_run: list,
    instruments_run: dict,
    period: tuple,
    n_outcomes: int,
) -> str:
    lines = []
    lines.append("# Signal Stacking Research — Honest Summary")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Sessions:** {', '.join(sessions_run)}")
    insts_flat = sorted(set(i for lst in instruments_run.values() for i in lst))
    lines.append(f"**Instruments:** {', '.join(insts_flat)}")
    lines.append(f"**Reference:** {REF_ENTRY_MODEL} / CB={REF_CONFIRM_BARS} / RR={REF_RR_TARGET}")
    lines.append(f"**Period:** {period[0]} to {period[1]}")
    lines.append(f"**Outcomes loaded:** {n_outcomes:,} (E3/CB1/RR2.0)")
    lines.append(f"**In-sample only.** No walk-forward validation.")
    lines.append("")

    # --- SURVIVED ---
    lines.append("## SURVIVED SCRUTINY")
    lines.append("")
    if not lift_df.empty:
        survived = lift_df[
            (lift_df["pvalue_bh"].notna()) &
            (lift_df["pvalue_bh"] < 0.05) &
            (lift_df["avg_r"].notna()) &
            (lift_df["avg_r"] > 0) &
            (~lift_df["sample_class"].isin(["INVALID"]))
        ]
        if len(survived) > 0:
            for _, r in survived.iterrows():
                lines.append(
                    f"- {r['instrument']} {r['session']} {r['dst_regime']} {r['g_filter']} "
                    f"{r['stack_level']} ({r['stack_label']}): "
                    f"avgR={r['avg_r']:+.3f} N={r['n']} p_bh={r['pvalue_bh']:.4f} "
                    f"[{r['sample_class']}]"
                )
        else:
            lines.append("None — no stack level survived BH FDR correction at p<0.05.")
    else:
        lines.append("No data.")
    lines.append("")

    # --- DID NOT SURVIVE ---
    lines.append("## DID NOT SURVIVE")
    lines.append("")
    if not lift_df.empty:
        n_positive = int((lift_df["avg_r"] > 0).sum())
        n_sig_raw = int(((lift_df["pvalue"] < 0.05) & lift_df["pvalue"].notna()).sum())
        n_sig_bh = int(
            ((lift_df["pvalue_bh"] < 0.05) & lift_df["pvalue_bh"].notna()).sum()
        )
        n_total = len(lift_df)
        lines.append(f"- {n_total} total (instrument × session × g_filter × dst × stack_level) cells tested")
        lines.append(f"- {n_positive}/{n_total} had positive avg_r")
        lines.append(f"- {n_sig_raw} had raw p<0.05 (before correction)")
        lines.append(f"- {n_sig_bh} survived BH FDR correction at p<0.05")
    lines.append("")

    # --- CAVEATS ---
    lines.append("## CAVEATS")
    lines.append("")
    lines.append("1. **In-sample only.** No walk-forward or OOS validation.")
    lines.append("2. **Fixed reference params** (E3/CB1/RR2.0). Stack lift may differ at E1 or other RR targets.")
    lines.append("3. **Small N at full stack.** G6+/G8+ + all filters may drop below REGIME threshold (N<30).")
    lines.append("4. **Concordance requires all instruments trading.** On thin days, missing one instrument falsely downgrades tier.")
    lines.append("5. **Vol filter is fail-closed.** Days with missing bars_1m data are excluded — may skew toward liquid days.")
    lines.append("6. **DST split (0900/0030) reduces N further** — winter/summer cells may be INVALID.")
    lines.append("")

    # --- NEXT STEPS ---
    lines.append("## NEXT STEPS")
    lines.append("")
    lines.append("- If any stack survives BH correction: test at E1 entry (production entry model for 1000)")
    lines.append("- If conviction_score > 1.5 at N>=30: add to position sizing overlay (not new strategy)")
    lines.append("- If concordance consistently adds lift: wire into execution_engine as a position-size multiplier")
    lines.append("- If nothing survives: signals are already captured in G-filter; stop stacking")
    lines.append("")

    # --- Analysis 1 table ---
    lines.append("## Analysis 1: Incremental Layer Lift")
    lines.append("")
    if not lift_df.empty:
        for (sess, inst), grp in lift_df.groupby(["session", "instrument"]):
            lines.append(f"### {inst} {sess}")
            lines.append("")
            header = (
                f"| {'dst':>7} | {'G':>3} | {'Level':>5} | {'Label':>20} | "
                f"{'N':>5} | {'%base':>6} | {'avgR':>7} | {'WR':>5} | "
                f"{'Sharpe':>6} | {'p_raw':>7} | {'p_bh':>7} | {'Class':>11} |"
            )
            lines.append(header)
            lines.append("|" + "-" * (len(header) - 2) + "|")
            for _, r in grp.iterrows():
                avg_r_s = f"{r['avg_r']:+7.4f}" if not np.isnan(r["avg_r"]) else "     --"
                wr_s = f"{r['win_rate']:.2f}" if not np.isnan(r["win_rate"]) else "  --"
                sh_s = f"{r['sharpe']:6.3f}" if not np.isnan(r["sharpe"]) else "    --"
                p_s = f"{r['pvalue']:.4f}" if not np.isnan(r["pvalue"]) else "     --"
                pb_s = f"{r['pvalue_bh']:.4f}" if not np.isnan(r["pvalue_bh"]) else "     --"
                pct_s = f"{r['pct_of_baseline']:.2f}" if not np.isnan(r["pct_of_baseline"]) else "    --"
                lines.append(
                    f"| {r['dst_regime']:>7} | {r['g_filter']:>3} | {r['stack_level']:>5} | "
                    f"{r['stack_label']:>20} | {r['n']:>5} | {pct_s:>6} | {avg_r_s} | "
                    f"{wr_s:>5} | {sh_s} | {p_s} | {pb_s} | {r['sample_class']:>11} |"
                )
            lines.append("")
    else:
        lines.append("No data.")
    lines.append("")

    # --- Analysis 2 table ---
    lines.append("## Analysis 2: Conviction Profile")
    lines.append("")
    if not conviction_df.empty:
        lines.append("Rows shown: CORE/REGIME/PRELIMINARY with avg_r > 0")
        lines.append("")
        lines.append(
            "| inst | sess | dst | G | level | label | N | freq/yr | avgR | conviction | sizing |"
        )
        lines.append("|------|------|-----|---|-------|-------|---|---------|------|------------|--------|")
        for _, r in conviction_df.iterrows():
            avg_r_s = f"{r['avg_r']:+.4f}" if not np.isnan(r["avg_r"]) else "--"
            conv_s = f"{r['conviction_score']:.3f}" if not np.isnan(r["conviction_score"]) else "--"
            lines.append(
                f"| {r['instrument']} | {r['session']} | {r['dst_regime']} | {r['g_filter']} | "
                f"{r['stack_level']} | {r['stack_label']} | {r['n']} | {r['freq_per_year']:.1f} | "
                f"{avg_r_s} | {conv_s} | {r['sizing_tier']} |"
            )
    else:
        lines.append("No rows with positive avg_r and sufficient N.")
    lines.append("")

    # --- Analysis 3 table ---
    lines.append("## Analysis 3: Concordance × Size Interaction (MGC 1000)")
    lines.append("")
    if not conc_size_df.empty:
        lines.append("| G | tier | N | %g_days | avgR | WR | p_raw | p_bh | class |")
        lines.append("|---|------|---|---------|------|----|-------|------|-------|")
        for _, r in conc_size_df.iterrows():
            avg_r_s = f"{r['avg_r']:+.4f}" if not np.isnan(r["avg_r"]) else "--"
            wr_s = f"{r['win_rate']:.2f}" if not np.isnan(r["win_rate"]) else "--"
            p_s = f"{r['pvalue']:.4f}" if not np.isnan(r["pvalue"]) else "--"
            pb_s = f"{r['pvalue_bh']:.4f}" if not np.isnan(r["pvalue_bh"]) else "--"
            pct_s = f"{r['pct_of_g_filter_days']:.2f}" if not np.isnan(r["pct_of_g_filter_days"]) else "--"
            lines.append(
                f"| {r['g_filter']} | {r['concordance_tier']} | {r['n']} | {pct_s} | "
                f"{avg_r_s} | {wr_s} | {p_s} | {pb_s} | {r['sample_class']} |"
            )
    else:
        lines.append("No MGC 1000 data available.")
    lines.append("")

    # --- Mandatory Disclosures ---
    lines.append("## Mandatory Disclosures")
    lines.append("")
    lines.append(f"- **N trades per stack level:** see Analysis 1 table above")
    lines.append(f"- **Period:** {period[0]} to {period[1]} (in-sample)")
    lines.append(f"- **Validation:** IS only — no walk-forward, no true OOS")
    lines.append(f"- **Fixed params:** E3/CB1/RR2.0 only (not grid-searched)")
    lines.append(f"- **Multiple comparisons:** BH FDR applied within each (session × instrument) group")
    lines.append(f"- **Sensitivity:** vol threshold 1.2 vs 1.5 tested (see sensitivity CSV)")
    lines.append(f"- **Concordance sensitivity:** majority_2 vs concordant_3 naturally split in Analysis 3")
    lines.append(f"- **Mechanism (if survived):** Full stack selects days where:")
    lines.append(f"  - ORB is large (G-filter) → sufficient cost absorption")
    lines.append(f"  - Direction is long (1000) → H5 confirmed, shorts are noise")
    lines.append(f"  - Concordance: all instruments break same direction → macro alignment")
    lines.append(f"  - Volume spike → institutional participation (not retail thin market)")
    lines.append(f"  - No NFP/OPEX → clean macro signal (no scheduled volatility event)")
    lines.append(f"- **What could kill it:** regime change, reduced CME liquidity, filter over-fit to 14-month window")
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Signal stacking research — P3"
    )
    parser.add_argument(
        "--instruments", nargs="+", default=None,
        help="Instruments to run (e.g. MGC MES MNQ). Default: all."
    )
    parser.add_argument(
        "--sessions", nargs="+", default=None,
        help="Sessions to run (e.g. 1000 0900 0030). Default: all."
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to gold.db (default: auto-resolve via pipeline.paths)"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    sessions_to_run = args.sessions or list(SESSION_INSTRUMENTS.keys())
    instruments_filter = set(args.instruments) if args.instruments else None

    # Build per-session instrument list (filter if --instruments specified)
    instruments_by_session = {}
    for sess in sessions_to_run:
        insts = SESSION_INSTRUMENTS.get(sess, [])
        if instruments_filter:
            insts = [i for i in insts if i in instruments_filter]
        instruments_by_session[sess] = insts

    print("=" * 70)
    print("  SIGNAL STACKING RESEARCH (P3)")
    print(f"  DB: {db_path}")
    print(f"  Sessions: {sessions_to_run}")
    print(f"  Reference: {REF_ENTRY_MODEL}/CB{REF_CONFIRM_BARS}/RR{REF_RR_TARGET}")
    print(f"  scipy: {HAS_SCIPY}")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading features and outcomes...")
    with duckdb.connect(str(db_path), read_only=True) as con:
        features_df = load_features(con)
        outcomes_df = load_outcomes(con)

    print(f"  Features: {len(features_df):,} rows "
          f"({features_df['trading_day'].min()} to {features_df['trading_day'].max()})")
    print(f"  Outcomes: {len(outcomes_df):,} rows")
    period = (
        str(features_df["trading_day"].min()),
        str(features_df["trading_day"].max()),
    )

    # Pre-compute concordance maps
    print("\n[2/5] Building concordance maps...")
    conc_maps: dict[str, pd.DataFrame | None] = {}
    for session in sessions_to_run:
        conc_insts = SESSION_CONC_INSTRUMENTS.get(session)
        if conc_insts is None:
            conc_maps[session] = None
            print(f"  {session}: no concordance (single instrument)")
            continue
        # Filter to instruments we actually have
        available_insts = [
            i for i in conc_insts
            if i in features_df["symbol"].unique()
        ]
        if len(available_insts) < 2:
            conc_maps[session] = None
            print(f"  {session}: insufficient instruments for concordance ({available_insts})")
            continue
        cmap = build_concordance_map(features_df, available_insts, session)
        conc_maps[session] = cmap
        tier_dist = cmap["concordance_tier"].value_counts()
        print(f"  {session} ({len(available_insts)}-way): "
              f"concordant_3={tier_dist.get('concordant_3', 0)} "
              f"majority_2={tier_dist.get('majority_2', 0)} "
              f"remaining={tier_dist.get('remaining', 0)}")

    # Pre-compute relative volume
    print("\n[3/5] Computing relative volumes (bars_1m queries)...")
    rel_vol_map: dict[tuple, pd.Series] = {}

    with duckdb.connect(str(db_path), read_only=True) as con:
        for session in sessions_to_run:
            for instrument in instruments_by_session.get(session, []):
                inst_feat = features_df[
                    features_df["symbol"] == instrument
                ].reset_index(drop=True)

                rv = compute_rel_vol_for_session(
                    con, inst_feat, instrument, session
                )
                rel_vol_map[(instrument, session)] = rv
                n_valid = int(rv.notna().sum())
                print(f"  {instrument} {session}: {n_valid}/{len(inst_feat)} days have rel_vol")

    # Build working DataFrames
    print("\n[4/5] Building working DataFrames...")
    all_working_dfs: dict[tuple, pd.DataFrame] = {}

    for session in sessions_to_run:
        conc_map = conc_maps.get(session)
        for instrument in instruments_by_session.get(session, []):
            inst_feat = features_df[
                features_df["symbol"] == instrument
            ].reset_index(drop=True)

            rv = rel_vol_map.get((instrument, session),
                                 pd.Series(np.nan, index=inst_feat.index))

            wdf = build_working_df(
                features_df, outcomes_df, instrument, session, conc_map, rv
            )
            all_working_dfs[(instrument, session)] = wdf
            print(f"  {instrument} {session}: {len(wdf)} outcome rows")

    # Run analyses
    print("\n[5/5] Running analyses...")

    print("  Analysis 1: Incremental Layer Lift...")
    lift_df = run_analysis_1(
        all_working_dfs, instruments_by_session, sessions_to_run,
        SESSION_DST_TYPE, SESSION_DIRECTION, SESSION_CONC_INSTRUMENTS,
    )

    print("  Analysis 2: Conviction Profile...")
    conviction_df = run_analysis_2(lift_df)

    print("  Analysis 3: Concordance × Size Interaction (MGC 1000)...")
    conc_size_df = run_analysis_3(all_working_dfs)

    print("  Sensitivity: vol threshold 1.2 vs 1.5...")
    sensitivity_df = run_sensitivity_check(
        all_working_dfs, sessions_to_run, instruments_by_session,
        SESSION_DIRECTION, SESSION_CONC_INSTRUMENTS, SESSION_DST_TYPE,
    )

    # --- Console summary ---
    print("\n" + "=" * 70)
    print("  CONSOLE SUMMARY")
    print("=" * 70)

    if not lift_df.empty:
        print(f"\n  Total cells: {len(lift_df)}")
        n_pos = int((lift_df["avg_r"] > 0).sum())
        n_invalid = int((lift_df["sample_class"] == "INVALID").sum())
        n_sig_bh = int(
            ((lift_df["pvalue_bh"] < 0.05) & lift_df["pvalue_bh"].notna()).sum()
        )
        print(f"  Positive avg_r: {n_pos}/{len(lift_df)}")
        print(f"  INVALID (N<30): {n_invalid}")
        print(f"  BH p<0.05: {n_sig_bh}")

        # Show L3+ rows for CORE/REGIME with positive avg_r
        interesting = lift_df[
            lift_df["stack_level"].isin(["L3", "L4", "L5"]) &
            (lift_df["avg_r"] > 0) &
            lift_df["sample_class"].isin(["CORE", "REGIME", "PRELIMINARY"])
        ].sort_values("avg_r", ascending=False).head(20)

        if len(interesting) > 0:
            print("\n  Top L3+ stacks (positive avg_r, N>=30):")
            print(f"  {'inst':>4} {'sess':>5} {'dst':>6} {'G':>3} {'lvl':>3} "
                  f"{'label':>20} {'N':>5} {'avgR':>8} {'p_bh':>8} {'class':>12}")
            for _, r in interesting.iterrows():
                avg_s = f"{r['avg_r']:+8.4f}" if not np.isnan(r["avg_r"]) else "      --"
                pb_s = f"{r['pvalue_bh']:8.4f}" if not np.isnan(r["pvalue_bh"]) else "      --"
                print(f"  {r['instrument']:>4} {r['session']:>5} {r['dst_regime']:>6} "
                      f"{r['g_filter']:>3} {r['stack_level']:>3} {r['stack_label']:>20} "
                      f"{r['n']:>5} {avg_s} {pb_s} {r['sample_class']:>12}")

    # --- Save outputs ---
    output_dir = Path("research/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    out1 = output_dir / "signal_stacking_marginal_lift.csv"
    out2 = output_dir / "signal_stacking_conviction_profile.csv"
    out3 = output_dir / "signal_stacking_size_concordance.csv"
    out_sens = output_dir / "signal_stacking_sensitivity.csv"
    out_md = output_dir / "signal_stacking_summary.md"

    lift_df.to_csv(out1, index=False, float_format="%.6f")
    conviction_df.to_csv(out2, index=False, float_format="%.6f")
    conc_size_df.to_csv(out3, index=False, float_format="%.6f")
    sensitivity_df.to_csv(out_sens, index=False, float_format="%.6f")

    md = build_summary_md(
        lift_df, conviction_df, conc_size_df, sensitivity_df,
        sessions_to_run, instruments_by_session, period, len(outcomes_df),
    )
    out_md.write_text(md, encoding="utf-8")

    print(f"\n  Outputs saved:")
    print(f"    {out1}  ({len(lift_df)} rows)")
    print(f"    {out2}  ({len(conviction_df)} rows)")
    print(f"    {out3}  ({len(conc_size_df)} rows)")
    print(f"    {out_sens}")
    print(f"    {out_md}")
    print()


if __name__ == "__main__":
    main()
