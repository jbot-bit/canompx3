"""Comprehensive feature-overlay scan — institutional grade.

Per `docs/audit/hypotheses/2026-04-15-comprehensive-deployed-lane-gap-audit.md`:

Scopes:
- Alpha: feature overlay on 6 deployed MNQ lanes (TWO-PASS: unfiltered + filtered
  within-deployed-filter subset)
- Beta: MES/MGC twins of deployed (session, apt, RR) — tests if twin works on
  same filter
- Delta: feature scan on top non-deployed sessions × all 3 instruments

Methodological controls (audit-identified):
- Two-pass testing: signal must hold BOTH on unfiltered baseline AND on
  deployed-filter subset for Alpha. Filtered subset is what the lane actually
  trades.
- T0 tautology gate: correlation of new feature with deployed filter. |corr|>0.7
  flags duplicate filter.
- Per-family BH-FDR (feature families: volatility, calendar, timing, volume,
  level-distance, overnight, gap) plus global BH-FDR.
- Fire-rate flag: extreme fire rates (<5% or >95%) marked untrustworthy.
- Extended feature set: day_type categorical, DOW buckets, London/Asia session
  levels, pit_range_atr, bb_volume_ratio low tail.
- Directional asymmetry: gap_UP×long, ovn_took_pdh×long type features.
- WR monotonicity reported per cell to catch ARITHMETIC_ONLY signals.

Output:
  docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

SEED = 20260415

# ============================================================================
# Lane universe — (session, aperture, rr, instrument, scope_tag, deployed_filter_key)
# deployed_filter_key: 'ORB_G5', 'ATR_P50', 'OVNRNG_100', 'VWAP_MID_ALIGNED', None
# ============================================================================

# All active sessions per pipeline.dst.SESSION_CATALOG (mega script source of truth)
ALL_SESSIONS = [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE", "BRISBANE_1025",
]
ALL_INSTRUMENTS = ["MNQ", "MES", "MGC"]
ALL_APERTURES = [5, 15, 30]
ALL_RRS = [1.0, 1.5, 2.0]

# Deployed lanes — used to tag 'deployed' scope + attach deployed filter
DEPLOYED_LANE_SPECS = {
    ("EUROPE_FLOW", 5, 1.5, "MNQ"): "ORB_G5",
    ("SINGAPORE_OPEN", 30, 1.5, "MNQ"): "ATR_P50",
    ("COMEX_SETTLE", 5, 1.5, "MNQ"): "OVNRNG_100",
    ("NYSE_OPEN", 5, 1.0, "MNQ"): "ORB_G5",
    ("TOKYO_OPEN", 5, 1.5, "MNQ"): "ORB_G5",
    ("US_DATA_1000", 5, 1.5, "MNQ"): "VWAP_MID_ALIGNED",
}


def build_all_lanes() -> list[tuple]:
    """Generate EVERY (session, apt, rr, instr, scope_tag, filter_key) combo."""
    lanes = []
    for session in ALL_SESSIONS:
        for instr in ALL_INSTRUMENTS:
            for apt in ALL_APERTURES:
                for rr in ALL_RRS:
                    key = (session, apt, rr, instr)
                    if key in DEPLOYED_LANE_SPECS:
                        scope = "deployed"
                        filt = DEPLOYED_LANE_SPECS[key]
                    elif (session, apt, rr) in {(k[0], k[1], k[2]) for k in DEPLOYED_LANE_SPECS}:
                        # Twin — same session/apt/rr as a deployed lane, different instrument
                        scope = "twin"
                        # Use matching deployed lane's filter
                        filt = next(v for k, v in DEPLOYED_LANE_SPECS.items() if (k[0], k[1], k[2]) == (session, apt, rr))
                    else:
                        scope = "non_deployed"
                        filt = None
                    lanes.append((session, apt, rr, instr, scope, filt))
    return lanes


ALL_LANES = build_all_lanes()

OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-07").date()

OUTPUT_MD = Path("docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

# Feature-family map for per-family BH-FDR
FEATURE_FAMILY = {
    "break_delay_LT2": "timing",
    "break_delay_GT10": "timing",
    "break_bar_continues_TRUE": "timing",
    "bb_volume_ratio_HIGH": "volume",
    "bb_volume_ratio_LOW": "volume",
    "rel_vol_LOW_Q1": "volume",
    "rel_vol_HIGH_Q3": "volume",
    "pre_velocity_HIGH": "timing",
    "pre_velocity_LOW": "timing",
    "atr_20_pct_GT80": "volatility",
    "atr_20_pct_LT20": "volatility",
    "atr_vel_HIGH": "volatility",
    "atr_vel_LOW": "volatility",
    "garch_vol_pct_GT70": "volatility",
    "garch_vol_pct_LT30": "volatility",
    "pit_range_atr_HIGH": "volatility",
    "pit_range_atr_LOW": "volatility",
    "ovn_range_pct_GT80": "overnight",
    "ovn_range_pct_LT20": "overnight",
    "ovn_took_pdh_TRUE": "overnight",
    "ovn_took_pdl_TRUE": "overnight",
    "ovn_took_pdh_LONG_INTERACT": "overnight",
    "ovn_took_pdl_SHORT_INTERACT": "overnight",
    "gap_UP": "gap",
    "gap_DOWN": "gap",
    "gap_UP_LONG_INTERACT": "gap",
    "gap_DOWN_SHORT_INTERACT": "gap",
    "gap_large_GT1_ATR": "gap",
    "is_nfp_TRUE": "calendar",
    "is_opex_TRUE": "calendar",
    "is_friday_TRUE": "calendar",
    "is_monday_TRUE": "calendar",
    "dow_tue": "calendar",
    "dow_wed": "calendar",
    "dow_thu": "calendar",
    "near_session_ny_high": "level",
    "near_session_ny_low": "level",
    "near_session_london_high": "level",
    "near_session_london_low": "level",
    "near_session_asia_high": "level",
    "near_session_asia_low": "level",
    "day_type_trend_up": "daytype",
    "day_type_trend_down": "daytype",
    "day_type_range": "daytype",
}


# ============================================================================
# Data loading
# ============================================================================


def load_lane(session: str, apt: int, rr: float, instrument: str) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
        o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
        o.entry_model, o.rr_target, o.outcome, o.pnl_r,
        d.atr_20, d.atr_20_pct, d.atr_vel_ratio, d.pit_range_atr,
        d.prev_day_high, d.prev_day_low, d.prev_day_close, d.prev_day_range,
        d.gap_type, d.gap_open_points,
        d.overnight_range, d.overnight_range_pct,
        d.overnight_took_pdh, d.overnight_took_pdl,
        d.is_nfp_day, d.is_opex_day, d.is_friday, d.is_monday,
        d.day_of_week, d.day_type,
        d.garch_forecast_vol_pct, d.garch_atr_ratio,
        d.session_asia_high, d.session_asia_low,
        d.session_london_high, d.session_london_low,
        d.session_ny_high, d.session_ny_low,
        d.orb_{session}_size AS orb_size,
        d.orb_{session}_high AS orb_high,
        d.orb_{session}_low AS orb_low,
        (d.orb_{session}_high + d.orb_{session}_low) / 2.0 AS orb_mid,
        d.orb_{session}_break_dir AS break_dir,
        d.orb_{session}_break_delay_min AS break_delay_min,
        d.orb_{session}_break_bar_continues AS break_bar_continues,
        d.orb_{session}_volume AS orb_volume,
        d.orb_{session}_break_bar_volume AS break_bar_volume,
        d.rel_vol_{session} AS rel_vol,
        d.orb_{session}_vwap AS session_vwap,
        d.orb_{session}_pre_velocity AS pre_velocity
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol
      AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{session}'
      AND o.symbol = '{instrument}'
      AND o.orb_minutes = {apt}
      AND o.entry_model = 'E2'
      AND o.rr_target = {rr}
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
      AND d.orb_{session}_break_dir IN ('long','short')
    """
    try:
        df = con.execute(q).df()
    except Exception as e:
        print(f"  [{instrument} {session} O{apt} RR{rr}] load failed: {e}")
        con.close()
        return pd.DataFrame()
    con.close()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    return df


# ============================================================================
# Deployed filter computation (for two-pass testing)
# ============================================================================


def compute_deployed_filter(df: pd.DataFrame, filter_key: str | None) -> np.ndarray:
    """Return 0/1 array — 1 where the deployed filter fires (day tradable)."""
    if filter_key is None:
        return np.ones(len(df), dtype=int)
    n = len(df)

    if filter_key == "ORB_G5":
        # orb_size >= Q80 (top quintile, per-lane)
        sizes = df["orb_size"].astype(float)
        if sizes.notna().sum() < 30:
            return np.zeros(n, dtype=int)
        q80 = np.nanpercentile(sizes, 80)
        return (sizes >= q80).astype(int).values

    if filter_key == "ATR_P50":
        atr = df["atr_20"].astype(float)
        if atr.notna().sum() < 30:
            return np.zeros(n, dtype=int)
        p50 = np.nanpercentile(atr, 50)
        return (atr >= p50).astype(int).values

    if filter_key == "OVNRNG_100":
        ovn = df["overnight_range"].astype(float)
        atr = df["atr_20"].astype(float)
        ratio = ovn / atr
        return (ratio >= 1.0).fillna(False).astype(int).values

    if filter_key == "VWAP_MID_ALIGNED":
        # Approximation: orb_mid on same side as break_dir relative to vwap
        # Long: orb_mid > vwap, Short: orb_mid < vwap
        mid = df["orb_mid"].astype(float)
        vwap = df["session_vwap"].astype(float)
        bd = df["break_dir"]
        aligned = np.where(
            bd == "long",
            (mid > vwap).astype(int),
            (mid < vwap).astype(int),
        )
        aligned = np.where(pd.isna(vwap), 0, aligned)
        return aligned.astype(int)

    return np.ones(n, dtype=int)


# ============================================================================
# Feature building
# ============================================================================


def bucket_low(vals: pd.Series, pct: float) -> np.ndarray:
    vv = vals.astype(float)
    if vv.notna().sum() < 20:
        return np.zeros(len(vv), dtype=int)
    thresh = np.nanpercentile(vv, pct)
    return (vv < thresh).fillna(False).astype(int).values


def bucket_high(vals: pd.Series, pct: float) -> np.ndarray:
    vv = vals.astype(float)
    if vv.notna().sum() < 20:
        return np.zeros(len(vv), dtype=int)
    thresh = np.nanpercentile(vv, pct)
    return (vv > thresh).fillna(False).astype(int).values


def build_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if len(df) == 0:
        return {}
    feats: dict[str, np.ndarray] = {}

    # Timing
    if df["break_delay_min"].notna().any():
        bd = df["break_delay_min"].astype(float)
        feats["break_delay_LT2"] = (bd < 2).fillna(False).astype(int).values
        feats["break_delay_GT10"] = (bd > 10).fillna(False).astype(int).values
    if df["break_bar_continues"].notna().any():
        feats["break_bar_continues_TRUE"] = df["break_bar_continues"].fillna(False).astype(int).values

    # Volume
    if df["rel_vol"].notna().any():
        feats["rel_vol_LOW_Q1"] = bucket_low(df["rel_vol"], 33)
        feats["rel_vol_HIGH_Q3"] = bucket_high(df["rel_vol"], 67)
    if df["break_bar_volume"].notna().any() and df["orb_volume"].notna().any():
        orb_vol = df["orb_volume"].astype(float).replace(0, np.nan)
        bb_ratio = df["break_bar_volume"].astype(float) / orb_vol
        if bb_ratio.notna().sum() >= 20:
            feats["bb_volume_ratio_HIGH"] = bucket_high(bb_ratio, 67)
            feats["bb_volume_ratio_LOW"] = bucket_low(bb_ratio, 33)

    # Pre-ORB state
    if df["pre_velocity"].notna().any():
        feats["pre_velocity_HIGH"] = bucket_high(df["pre_velocity"], 67)
        feats["pre_velocity_LOW"] = bucket_low(df["pre_velocity"], 33)

    # Vol regime
    if df["atr_20_pct"].notna().any():
        feats["atr_20_pct_GT80"] = (df["atr_20_pct"].astype(float) > 80).fillna(False).astype(int).values
        feats["atr_20_pct_LT20"] = (df["atr_20_pct"].astype(float) < 20).fillna(False).astype(int).values
    if df["atr_vel_ratio"].notna().any():
        feats["atr_vel_HIGH"] = bucket_high(df["atr_vel_ratio"], 67)
        feats["atr_vel_LOW"] = bucket_low(df["atr_vel_ratio"], 33)
    if df["garch_forecast_vol_pct"].notna().any():
        feats["garch_vol_pct_GT70"] = (df["garch_forecast_vol_pct"].astype(float) > 70).fillna(False).astype(int).values
        feats["garch_vol_pct_LT30"] = (df["garch_forecast_vol_pct"].astype(float) < 30).fillna(False).astype(int).values
    if df["pit_range_atr"].notna().any():
        feats["pit_range_atr_HIGH"] = bucket_high(df["pit_range_atr"], 67)
        feats["pit_range_atr_LOW"] = bucket_low(df["pit_range_atr"], 33)

    # Overnight features — per `pipeline/build_daily_features.py:445-454`, these use
    # the 09:00-17:00 Brisbane window, so they LEAK FUTURE data for CME_REOPEN,
    # TOKYO_OPEN, BRISBANE_0925/1025, SINGAPORE_OPEN. Valid only for sessions
    # starting AFTER 17:00 Brisbane.
    orb_session = df["orb_label"].iloc[0] if len(df) else None
    overnight_ok = _overnight_lookhead_clean(orb_session)
    if overnight_ok:
        if df["overnight_range_pct"].notna().any():
            feats["ovn_range_pct_GT80"] = (df["overnight_range_pct"].astype(float) > 80).fillna(False).astype(int).values
            feats["ovn_range_pct_LT20"] = (df["overnight_range_pct"].astype(float) < 20).fillna(False).astype(int).values
        if df["overnight_took_pdh"].notna().any():
            feats["ovn_took_pdh_TRUE"] = df["overnight_took_pdh"].fillna(False).astype(int).values
            long_mask = (df["break_dir"] == "long").astype(int).values
            feats["ovn_took_pdh_LONG_INTERACT"] = (feats["ovn_took_pdh_TRUE"] & long_mask).astype(int)
        if df["overnight_took_pdl"].notna().any():
            feats["ovn_took_pdl_TRUE"] = df["overnight_took_pdl"].fillna(False).astype(int).values
            short_mask = (df["break_dir"] == "short").astype(int).values
            feats["ovn_took_pdl_SHORT_INTERACT"] = (feats["ovn_took_pdl_TRUE"] & short_mask).astype(int)

    # Gap (directional interactions)
    if "gap_type" in df.columns:
        gup = (df["gap_type"] == "gap_up").astype(int).values
        gdn = (df["gap_type"] == "gap_down").astype(int).values
        feats["gap_UP"] = gup
        feats["gap_DOWN"] = gdn
        long_mask = (df["break_dir"] == "long").astype(int).values
        short_mask = (df["break_dir"] == "short").astype(int).values
        feats["gap_UP_LONG_INTERACT"] = (gup & long_mask).astype(int)
        feats["gap_DOWN_SHORT_INTERACT"] = (gdn & short_mask).astype(int)
    if df["gap_open_points"].notna().any():
        atr = df["atr_20"].astype(float).replace(0, np.nan)
        gap_atr = np.abs(df["gap_open_points"].astype(float)) / atr
        feats["gap_large_GT1_ATR"] = (gap_atr > 1.0).fillna(False).astype(int).values

    # Calendar
    if "is_nfp_day" in df.columns:
        feats["is_nfp_TRUE"] = df["is_nfp_day"].fillna(False).astype(int).values
    if "is_opex_day" in df.columns:
        feats["is_opex_TRUE"] = df["is_opex_day"].fillna(False).astype(int).values
    if "is_friday" in df.columns:
        feats["is_friday_TRUE"] = df["is_friday"].fillna(False).astype(int).values
    if "is_monday" in df.columns:
        feats["is_monday_TRUE"] = df["is_monday"].fillna(False).astype(int).values
    # DOW buckets (0=Mon 1=Tue 2=Wed 3=Thu 4=Fri)
    if "day_of_week" in df.columns and df["day_of_week"].notna().any():
        dow = df["day_of_week"].astype(int)
        feats["dow_tue"] = (dow == 1).astype(int).values
        feats["dow_wed"] = (dow == 2).astype(int).values
        feats["dow_thu"] = (dow == 3).astype(int).values

    # day_type categorical
    if "day_type" in df.columns and df["day_type"].notna().any():
        feats["day_type_trend_up"] = (df["day_type"] == "trend_up").fillna(False).astype(int).values
        feats["day_type_trend_down"] = (df["day_type"] == "trend_down").fillna(False).astype(int).values
        feats["day_type_range"] = (df["day_type"] == "range").fillna(False).astype(int).values

    # Session-level proximity (scaled by ATR).
    # CRITICAL: session_{asia/london/ny}_* features use CURRENT trading day's
    # session windows. They contain LOOK-AHEAD for ORB sessions that start
    # DURING or BEFORE the referenced session's Brisbane window.
    # SESSION_WINDOWS in Brisbane: asia 09:00-17:00, london 18:00-23:00, ny 23:00-02:00.
    # An ORB session can cleanly use session_{x}_high/low only if the ORB
    # starts AFTER {x}_end Brisbane time.
    # Resolved via VALID_SESSION_PAIRS dict passed from caller.
    mid = df["orb_mid"].astype(float)
    atr = df["atr_20"].astype(float).replace(0, np.nan)
    orb_session = df["orb_label"].iloc[0] if len(df) else None
    valid_pairs = _valid_session_features(orb_session)
    for level_col, name in [
        ("session_ny_high", "near_session_ny_high"),
        ("session_ny_low", "near_session_ny_low"),
        ("session_london_high", "near_session_london_high"),
        ("session_london_low", "near_session_london_low"),
        ("session_asia_high", "near_session_asia_high"),
        ("session_asia_low", "near_session_asia_low"),
    ]:
        ref_session = level_col.split("_")[1]  # 'ny' / 'london' / 'asia'
        if ref_session not in valid_pairs:
            continue  # skip to avoid look-ahead
        if level_col in df.columns and df[level_col].notna().any():
            dist = np.abs(mid - df[level_col].astype(float)) / atr
            feats[name] = (dist < 0.15).fillna(False).astype(int).values

    return feats


def _overnight_lookhead_clean(orb_session: str | None) -> bool:
    """Overnight_* features use 09:00-17:00 Brisbane window. CLEAN only for
    ORB sessions starting at or after 17:00 Brisbane."""
    orb_start_brisbane = {
        "CME_REOPEN": 8.0, "TOKYO_OPEN": 10.0, "BRISBANE_0925": 9.42,
        "BRISBANE_1025": 10.42, "SINGAPORE_OPEN": 11.0, "LONDON_METALS": 17.0,
        "BRISBANE_1955": 19.92, "EUROPE_FLOW": 18.0, "US_DATA_830": 23.5,
        "NYSE_OPEN": 24.5, "US_DATA_1000": 25.0, "COMEX_SETTLE": 28.5,
        "CME_PRECLOSE": 30.0, "NYSE_CLOSE": 31.0,
    }
    if orb_session is None or orb_session not in orb_start_brisbane:
        return False
    return orb_start_brisbane[orb_session] >= 17.0


def _valid_session_features(orb_session: str | None) -> set[str]:
    """Return which session-level features are look-ahead-CLEAN for this ORB session.
    Derived from SESSION_WINDOWS (asia 09:00-17:00, london 18:00-23:00, ny 23:00-02:00)
    and ORB start times per pipeline.dst.SESSION_CATALOG.
    """
    # Rule: session_{x} is CLEAN iff ORB_start >= session_{x}_end (Brisbane).
    # Approximate ORB start in Brisbane:
    orb_start_brisbane = {
        "CME_REOPEN": 8.0,        # ~08:00 Brisbane (varies with DST)
        "TOKYO_OPEN": 10.0,       # 10:00
        "BRISBANE_0925": 9.42,    # 09:25
        "BRISBANE_1025": 10.42,   # 10:25
        "SINGAPORE_OPEN": 11.0,   # 11:00
        "LONDON_METALS": 17.0,    # 17:00
        "BRISBANE_1955": 19.92,   # 19:55
        "EUROPE_FLOW": 18.0,      # 18:00
        "US_DATA_830": 23.5,      # 23:30
        "NYSE_OPEN": 24.5,        # 00:30 next day = 24.5
        "US_DATA_1000": 25.0,     # 01:00 next day = 25.0
        "COMEX_SETTLE": 28.5,     # 04:30 = 28.5
        "CME_PRECLOSE": 30.0,     # 06:00 = 30.0
        "NYSE_CLOSE": 31.0,       # 07:00 = 31.0
    }
    session_ends_brisbane = {
        "asia": 17.0,
        "london": 23.0,
        "ny": 26.0,  # 02:00 next day = 26.0
    }
    if orb_session is None or orb_session not in orb_start_brisbane:
        return set()
    orb_hr = orb_start_brisbane[orb_session]
    valid = set()
    for sess, end_hr in session_ends_brisbane.items():
        if orb_hr >= end_hr:
            valid.add(sess)
    return valid


# ============================================================================
# Per-cell test — with two-pass (unfiltered + filtered), WR, T0 tautology
# ============================================================================


def t0_correlation(feat_sig: np.ndarray, filter_sig: np.ndarray) -> float:
    """Return absolute correlation between new feature and deployed filter."""
    if filter_sig.sum() == 0 or (1 - filter_sig).sum() == 0:
        return 0.0  # No variance in filter (e.g., filter_key=None → all 1s)
    try:
        c = np.corrcoef(feat_sig.astype(float), filter_sig.astype(float))[0, 1]
        return 0.0 if np.isnan(c) else float(abs(c))
    except Exception:
        return 0.0


def test_cell(
    df: pd.DataFrame,
    feature: str,
    signal: np.ndarray,
    direction: str,
    filter_sig: np.ndarray,
    pass_type: str,
) -> dict | None:
    """pass_type: 'unfiltered' or 'filtered' (applies deployed filter first)."""
    mask_dir = (df["break_dir"] == direction).values
    sig = signal[mask_dir]
    fsig = filter_sig[mask_dir]
    sub = df[mask_dir].copy()

    if pass_type == "filtered":
        keep = fsig == 1
        sub = sub[keep]
        sig = sig[keep]

    if len(sub) == 0:
        return None

    sub = sub.copy()
    sub["_sig"] = sig
    is_df = sub[sub["is_is"]]
    oos_df = sub[sub["is_oos"]]

    on_is = is_df[is_df["_sig"] == 1]["pnl_r"]
    off_is = is_df[is_df["_sig"] == 0]["pnl_r"]

    if len(on_is) < 30 or len(off_is) < 30:
        return None

    t_is, p_is = stats.ttest_ind(on_is, off_is, equal_var=False)
    expr_on = float(on_is.mean())
    expr_off = float(off_is.mean())
    delta_is = expr_on - expr_off
    wr_on = float((on_is > 0).mean())
    wr_off = float((off_is > 0).mean())
    wr_spread = wr_on - wr_off

    on_oos = oos_df[oos_df["_sig"] == 1]["pnl_r"]
    off_oos = oos_df[oos_df["_sig"] == 0]["pnl_r"]
    expr_on_oos = float(on_oos.mean()) if len(on_oos) >= 5 else float("nan")
    expr_off_oos = float(off_oos.mean()) if len(off_oos) >= 5 else float("nan")
    delta_oos = (expr_on_oos - expr_off_oos) if not np.isnan(expr_on_oos) and not np.isnan(expr_off_oos) else float("nan")
    dir_match = (not np.isnan(delta_oos)) and (np.sign(delta_is) == np.sign(delta_oos))

    fire_rate = int(sum(sig)) / max(1, len(sig))
    extreme_fire = (fire_rate < 0.05) or (fire_rate > 0.95)

    # ARITHMETIC_ONLY flag — if WR flat but ExpR moves, it's a cost screen not signal
    arithmetic_only = (abs(wr_spread) < 0.03) and (abs(delta_is) > 0.10)

    # T0 tautology vs deployed filter (only for unfiltered pass)
    t0_corr = 0.0
    if pass_type == "unfiltered":
        t0_corr = t0_correlation(signal[mask_dir], filter_sig[mask_dir])

    return {
        "feature": feature,
        "family": FEATURE_FAMILY.get(feature, "other"),
        "direction": direction,
        "pass_type": pass_type,
        "n_on_is": len(on_is),
        "n_off_is": len(off_is),
        "n_on_oos": len(on_oos),
        "expr_on_is": expr_on,
        "expr_off_is": expr_off,
        "delta_is": delta_is,
        "wr_on_is": wr_on,
        "wr_off_is": wr_off,
        "wr_spread": wr_spread,
        "expr_on_oos": expr_on_oos,
        "delta_oos": delta_oos,
        "dir_match": dir_match,
        "t_is": float(t_is),
        "p_is": float(p_is),
        "fire_rate": fire_rate,
        "extreme_fire": extreme_fire,
        "arithmetic_only": arithmetic_only,
        "t0_corr": t0_corr,
        "t0_tautology": t0_corr > 0.7,
    }


def scan_lane(
    session: str,
    apt: int,
    rr: float,
    instr: str,
    scope_tag: str,
    filter_key: str | None,
) -> list[dict]:
    df = load_lane(session, apt, rr, instr)
    if len(df) < 50:
        print(f"  [skip] {instr} {session} O{apt} RR{rr}: N={len(df)}")
        return []

    filter_sig = compute_deployed_filter(df, filter_key)
    filter_fire_rate = filter_sig.sum() / len(filter_sig)
    print(f"    filter {filter_key or 'NONE'} fire rate: {filter_fire_rate:.1%}")

    feats = build_features(df)
    if not feats:
        return []

    passes = ["unfiltered"] if filter_key is None else ["unfiltered", "filtered"]

    rows = []
    for feature_name, sig in feats.items():
        for direction in ("long", "short"):
            for pt in passes:
                res = test_cell(df, feature_name, sig, direction, filter_sig, pt)
                if res is None:
                    continue
                res["session"] = session
                res["aperture"] = apt
                res["rr"] = rr
                res["instrument"] = instr
                res["scope"] = scope_tag
                res["deployed_filter"] = filter_key or "NONE"
                rows.append(res)
    return rows


def bh_fdr_multi_framing(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply BH-FDR at MULTIPLE K framings so each cell can be evaluated at
    its natural hypothesis-family boundary:

    - K_global       = total cells scanned
    - K_family       = cells within feature family (volume / volatility / timing / ...)
    - K_lane         = cells within (session, apt, rr, instrument) — features × directions × passes
    - K_session      = cells within session (across instruments / apt / rr)
    - K_instrument   = cells within instrument
    - K_feature      = cells within a specific feature name across lanes

    Each framing emits a K_<framing> column AND bh_pass_<framing> boolean.
    Cells can pass at looser framings (per-lane) while failing stricter
    (global). This is institutional practice per Bailey-LdP 2014 and Harvey-Liu
    2015: report multiple K framings, no cherry-picking.
    """
    res = results.copy()

    def _apply_bh(df: pd.DataFrame, group_cols: list[str] | None, suffix: str) -> pd.DataFrame:
        df = df.copy()
        if group_cols is None:
            df = df.sort_values("p_is").reset_index(drop=True)
            K = len(df)
            df[f"K_{suffix}"] = K
            df[f"bh_rank_{suffix}"] = df.index + 1
            df[f"bh_crit_{suffix}"] = alpha * df[f"bh_rank_{suffix}"] / K
            df[f"bh_pass_{suffix}"] = df["p_is"] <= df[f"bh_crit_{suffix}"]
            return df
        pieces = []
        for _, grp in df.groupby(group_cols, dropna=False):
            g = grp.sort_values("p_is").reset_index(drop=True)
            K = len(g)
            g[f"K_{suffix}"] = K
            g[f"bh_rank_{suffix}"] = g.index + 1
            g[f"bh_crit_{suffix}"] = alpha * g[f"bh_rank_{suffix}"] / K
            g[f"bh_pass_{suffix}"] = g["p_is"] <= g[f"bh_crit_{suffix}"]
            pieces.append(g)
        return pd.concat(pieces, ignore_index=True)

    # Preserve original row identity so we can merge framings
    res["_row_id"] = range(len(res))

    out = res[["_row_id"] + [c for c in res.columns if c != "_row_id"]].copy()
    out = _apply_bh(out, None, "global")

    # Each additional framing merges back on _row_id
    fams = [
        (["family"], "family"),
        (["session", "aperture", "rr", "instrument"], "lane"),
        (["session"], "session"),
        (["instrument"], "instrument"),
        (["feature"], "feature"),
    ]
    for group_cols, suffix in fams:
        bh_cols = [f"K_{suffix}", f"bh_rank_{suffix}", f"bh_crit_{suffix}", f"bh_pass_{suffix}"]
        sub = _apply_bh(res, group_cols, suffix)[["_row_id"] + bh_cols]
        out = out.merge(sub, on="_row_id", how="left")

    return out.drop(columns=["_row_id"])


# Backward-compat shim
def bh_fdr(results: pd.DataFrame, alpha: float = 0.05, group_col: str | None = None) -> pd.DataFrame:
    if group_col is None:
        res = results.sort_values("p_is").reset_index(drop=True)
        K = len(res)
        res["bh_rank"] = res.index + 1
        res["bh_crit"] = alpha * res["bh_rank"] / K
        res["bh_pass"] = res["p_is"] <= res["bh_crit"]
        return res
    pieces = []
    for _, grp in results.groupby(group_col):
        g = grp.sort_values("p_is").reset_index(drop=True)
        K = len(g)
        g["bh_rank_family"] = g.index + 1
        g["bh_crit_family"] = alpha * g["bh_rank_family"] / K
        g["bh_pass_family"] = g["p_is"] <= g["bh_crit_family"]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


# ============================================================================
# Reporting
# ============================================================================


def emit(res: pd.DataFrame) -> None:
    # Apply BH-FDR at multiple K framings — each cell tagged with its pass at
    # every framing (global, family, lane, session, instrument, feature)
    res = bh_fdr_multi_framing(res, alpha=0.05)
    # Aliases for backward compat
    res["bh_pass"] = res["bh_pass_global"]
    res["bh_crit"] = res["bh_crit_global"]
    res["bh_rank"] = res["bh_rank_global"]

    # Filter: trustworthy cells (not extreme fire, not tautology, not arithmetic-only)
    trustworthy = res[
        (~res["extreme_fire"])
        & (~res["t0_tautology"])
        & (~res["arithmetic_only"])
    ].copy()

    strict = trustworthy[
        (trustworthy["t_is"].abs() >= 3.0)
        & (trustworthy["dir_match"])
        & (trustworthy["n_on_is"] >= 50)
    ].copy()

    bh_global = trustworthy[trustworthy["bh_pass"]].copy()
    bh_family = trustworthy[trustworthy["bh_pass_family"]].copy()

    promising = trustworthy[
        (trustworthy["t_is"].abs() >= 2.5)
        & (trustworthy["dir_match"])
        & (trustworthy["n_on_is"] >= 50)
    ].copy()

    # BH pass counts at each K framing
    bh_lane = trustworthy[trustworthy["bh_pass_lane"] == True]
    bh_session = trustworthy[trustworthy["bh_pass_session"] == True]
    bh_instr = trustworthy[trustworthy["bh_pass_instrument"] == True]
    bh_feat = trustworthy[trustworthy["bh_pass_feature"] == True]

    lines = [
        "# Comprehensive Scan — ALL Sessions × ALL Instruments × ALL Apertures × ALL RRs",
        "",
        "**Date:** 2026-04-15",
        f"**Total cells scanned:** {len(res)}",
        f"**Trustworthy cells** (not extreme-fire, not tautology, not arithmetic-only): {len(trustworthy)}",
        f"**Strict survivors** (|t|>=3 + dir_match + N>=50 + trustworthy): {len(strict)}",
        "",
        "## BH-FDR pass counts at each K framing",
        "",
        f"- **K_global** (K={int(trustworthy['K_global'].iloc[0]) if len(trustworthy) else 0}) strictest: {len(bh_global)} pass",
        f"- **K_family** (within feature-family, avg K~{int(trustworthy['K_family'].mean()) if len(trustworthy) else 0}): {len(bh_family)} pass",
        f"- **K_lane** (within session+apt+rr+instr, avg K~{int(trustworthy['K_lane'].mean()) if len(trustworthy) else 0}): {len(bh_lane)} pass",
        f"- **K_session** (within session across instruments, avg K~{int(trustworthy['K_session'].mean()) if len(trustworthy) else 0}): {len(bh_session)} pass",
        f"- **K_instrument** (within instrument, avg K~{int(trustworthy['K_instrument'].mean()) if len(trustworthy) else 0}): {len(bh_instr)} pass",
        f"- **K_feature** (within feature across lanes, avg K~{int(trustworthy['K_feature'].mean()) if len(trustworthy) else 0}): {len(bh_feat)} pass",
        "",
        f"**Promising** (|t|>=2.5 + dir_match + N>=50 + trustworthy): {len(promising)}",
        "",
        "## Scope definitions",
        "- **deployed**: Alpha — 6 live MNQ lanes, test overlays within their filter subset",
        "- **twin**: Beta — MES/MGC version of deployed (session,apt,RR) with same filter",
        "- **non_deployed**: Delta — top non-deployed sessions × 3 instruments (no filter)",
        "",
        "## Methodological controls",
        "- TWO-PASS: every deployed lane tested both `unfiltered` (full lane universe) and `filtered` (within deployed-filter subset). Overlay valid only if signal holds in `filtered` pass.",
        "- T0 tautology: |corr| > 0.7 with deployed filter → flagged tautology, excluded from survivors.",
        "- Extreme fire rate: <5% or >95% → flagged untrustworthy.",
        "- ARITHMETIC_ONLY: WR_spread < 3% AND |delta_is| > 0.10 → flagged as cost-screen not signal.",
        "- Per-family BH-FDR alongside global BH-FDR.",
        "",
        "## Strict Survivors (deploy candidates)",
        "",
        "| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p | BH_g | BH_f |",
        "|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|------|-------|---|---|------|------|",
    ]
    for _, r in strict.sort_values("t_is", key=abs, ascending=False).iterrows():
        bhg = "Y" if bool(r["bh_pass"]) else "."
        bhf = "Y" if bool(r["bh_pass_family"]) else "."
        lines.append(
            f"| {r['scope']} | {r['instrument']} | {r['session']} | O{r['aperture']} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['feature']} | {r['family']} | {r['pass_type']} | {r['n_on_is']} | "
            f"{r['fire_rate']:.1%} | {r['expr_on_is']:+.3f} | {r['wr_spread']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | {r['t_is']:+.2f} | {r['p_is']:.4f} | {bhg} | {bhf} |"
        )

    lines += [
        "",
        "## BH-FDR Survivors — Global (q=0.05)",
        "",
        "| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |",
        "|-------|-------|---------|-----|----|----|---------|------|------|-------|---------|------|-------|---|---|---------|",
    ]
    for _, r in bh_global.sort_values("t_is", key=abs, ascending=False).iterrows():
        lines.append(
            f"| {r['scope']} | {r['instrument']} | {r['session']} | O{r['aperture']} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['feature']} | {r['pass_type']} | {r['n_on_is']} | {r['fire_rate']:.1%} | "
            f"{r['expr_on_is']:+.3f} | {r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.5f} | {r['bh_crit']:.5f} |"
        )

    lines += [
        "",
        "## BH-FDR Survivors — Per-Family (q=0.05 within family)",
        "",
        "| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |",
        "|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|-------|---|---|",
    ]
    for _, r in bh_family.sort_values("t_is", key=abs, ascending=False).head(40).iterrows():
        lines.append(
            f"| {r['scope']} | {r['instrument']} | {r['session']} | O{r['aperture']} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['feature']} | {r['family']} | {r['pass_type']} | {r['n_on_is']} | "
            f"{r['fire_rate']:.1%} | {r['expr_on_is']:+.3f} | {r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.5f} |"
        )

    lines += [
        "",
        "## Promising cells (candidates for next-round T0-T8)",
        "",
        "| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p |",
        "|-------|-------|---------|-----|----|----|---------|------|------|---------|------|------|-------|---|---|",
    ]
    for _, r in promising.sort_values("t_is", key=abs, ascending=False).head(40).iterrows():
        lines.append(
            f"| {r['scope']} | {r['instrument']} | {r['session']} | O{r['aperture']} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['feature']} | {r['pass_type']} | {r['n_on_is']} | "
            f"{r['expr_on_is']:+.3f} | {r['wr_spread']:+.3f} | {r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.4f} |"
        )

    # Flagged cells (tautology, extreme-fire, arithmetic-only) — transparency
    flagged = res[res["t0_tautology"] | res["extreme_fire"] | res["arithmetic_only"]]
    flagged_strong = flagged[flagged["t_is"].abs() >= 3.0]
    lines += [
        "",
        "## Flagged cells (excluded despite |t|>=3) — for transparency",
        "",
        f"Excluded because: tautology |corr|>0.7 ({int(flagged['t0_tautology'].sum())}), "
        f"extreme fire <5% or >95% ({int(flagged['extreme_fire'].sum())}), "
        f"arithmetic-only WR flat ({int(flagged['arithmetic_only'].sum())})",
        "",
        "| Scope | Instr | Session | Feature | Pass | t | Fire% | T0 corr | Reason |",
        "|-------|-------|---------|---------|------|---|-------|---------|--------|",
    ]
    for _, r in flagged_strong.sort_values("t_is", key=abs, ascending=False).head(20).iterrows():
        reasons = []
        if r["t0_tautology"]:
            reasons.append(f"TAUTOLOGY(corr={r['t0_corr']:.2f})")
        if r["extreme_fire"]:
            reasons.append(f"FIRE({r['fire_rate']:.1%})")
        if r["arithmetic_only"]:
            reasons.append(f"ARITH(WR_Δ={r['wr_spread']:+.3f})")
        lines.append(
            f"| {r['scope']} | {r['instrument']} | {r['session']} | {r['feature']} | "
            f"{r['pass_type']} | {r['t_is']:+.2f} | {r['fire_rate']:.1%} | {r['t0_corr']:.2f} | {', '.join(reasons)} |"
        )

    # Baseline per-lane summary
    lines += [
        "",
        "## Baseline Per-Lane (no feature overlay)",
        "",
        "| Scope | Instr | Session | Apt | RR | Filter | N_is | N_oos | ExpR_is | ExpR_oos | Filter_fire% |",
        "|-------|-------|---------|-----|----|--------|------|-------|---------|----------|--------------|",
    ]
    seen = set()
    for row in ALL_LANES:
        session, apt, rr, instr, scope, filt = row
        if (scope, instr, session, apt, rr) in seen:
            continue
        seen.add((scope, instr, session, apt, rr))
        df_lane = load_lane(session, apt, rr, instr)
        if len(df_lane) == 0:
            continue
        fsig = compute_deployed_filter(df_lane, filt)
        fire_pct = fsig.sum() / max(1, len(fsig))
        is_df = df_lane[df_lane["is_is"]]
        oos_df = df_lane[df_lane["is_oos"]]
        is_f = is_df[fsig[df_lane["is_is"].values] == 1] if filt else is_df
        oos_f = oos_df[fsig[df_lane["is_oos"].values] == 1] if filt else oos_df
        expr_is = float(is_f["pnl_r"].mean()) if len(is_f) else float("nan")
        expr_oos = float(oos_f["pnl_r"].mean()) if len(oos_f) else float("nan")
        lines.append(
            f"| {scope} | {instr} | {session} | O{apt} | {rr:.1f} | {filt or 'NONE'} | "
            f"{len(is_f)} | {len(oos_f)} | {expr_is:+.3f} | {expr_oos:+.3f} | {fire_pct:.1%} |"
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")
    print(f"  Strict: {len(strict)}, BH_global: {len(bh_global)}, BH_family: {len(bh_family)}, Promising: {len(promising)}")
    print(f"  Flagged (tautology/extreme/arithmetic): {len(flagged)} of which |t|>=3: {len(flagged_strong)}")


def main():
    print(f"Scanning {len(ALL_LANES)} lanes")
    all_rows = []
    for i, (session, apt, rr, instr, tag, filt) in enumerate(ALL_LANES, 1):
        print(f"[{i}/{len(ALL_LANES)}] {tag:<14} {instr} {session} O{apt} RR{rr} filter={filt}")
        rows = scan_lane(session, apt, rr, instr, tag, filt)
        print(f"  → {len(rows)} cells tested")
        all_rows.extend(rows)

    if not all_rows:
        print("No cells tested.")
        return

    res = pd.DataFrame(all_rows)
    print(f"\nTotal cells: {len(res)}")
    emit(res)


if __name__ == "__main__":
    main()
