"""
1s Break Speed Research — Does 1-second resolution improve break delay measurement?

Research question: Does measuring break delay at 1s (instead of 1m) change
the optimal threshold or WR spread for break speed filtering?

Methodology:
1. Canonical ORB boundaries from daily_features (not recomputed)
2. 1s bars from raw DBN files, outright front-month only
3. Same break detection logic as pipeline (first close beyond ORB boundary)
4. Compare break_delay_sec vs break_delay_min (correlation, structure)
5. WR analysis with BH FDR at honest K across all sessions x instruments

No look-ahead: break_delay_sec = time from ORB end to first close-based break.
Same logic as 1m pipeline build_daily_features.detect_break(), finer resolution.

Contract alignment:
- Filter to outright contracts only (no spreads)
- Select front-month by volume per trading day (same as 1m pipeline)
- Pre-2019: NQ/ES/GC parent symbols. Post-2019: MNQ/MES/GC native.
- Spot-check 5 random dates per instrument for price alignment

@research-source docs/plans/2026-04-06-1s-break-speed-plan.md
@research-source memory/break_speed_signal_retest.md
"""

import re
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import databento as db
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from pipeline.build_daily_features import (
    _break_detection_window,
    _orb_utc_window,
)
from pipeline.ingest_dbn_mgc import choose_front_contract
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_1S_ROOT = PROJECT_ROOT / "data" / "raw" / "databento" / "ohlcv-1s"
ORB_MINUTES = 5  # Validated aperture for break speed
INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)  # MGC, MES, MNQ

# Sessions from asset_configs (all enabled sessions per instrument)
INSTRUMENT_SESSIONS = {}
for inst in INSTRUMENTS:
    INSTRUMENT_SESSIONS[inst] = ASSET_CONFIGS[inst].get("enabled_sessions", [])

# Brisbane timezone for trading day computation
BRISBANE_TZ = pd.Timestamp.now(tz="Australia/Brisbane").tzinfo

# Outright pattern mapping: (file prefix) -> (outright regex, prefix_len for expiry parse)
# Pre-2019 files use parent symbols, post-2019 use native micro symbols.
# MGC always uses GC (parent) data per asset_configs.
OUTRIGHT_CONFIG = {
    "MNQ": [
        ("nq_", re.compile(r"^NQ[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        ("mnq_", re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}$"), 3),
        ("daily_MNQ", re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
    "MES": [
        ("es_", re.compile(r"^ES[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        ("mes_", re.compile(r"^MES[FGHJKMNQUVXZ]\d{1,2}$"), 3),
        ("daily_MES", re.compile(r"^MES[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
    "MGC": [
        # MGC 1s files may contain GC or MGC symbols — try both
        ("mgc_", re.compile(r"^GC[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        ("daily_MGC", re.compile(r"^GC[FGHJKMNQUVXZ]\d{1,2}$"), 2),
    ],
}


def get_outright_config(instrument: str, filename: str):
    """Return (outright_pattern, prefix_len) based on file naming convention."""
    for prefix, pattern, plen in OUTRIGHT_CONFIG[instrument]:
        if filename.startswith(prefix):
            return pattern, plen
    # Fallback: try the asset_configs pattern
    cfg = ASSET_CONFIGS[instrument]
    return cfg["outright_pattern"], cfg.get("prefix_len", 3)


# ---------------------------------------------------------------------------
# PHASE 1: LOAD BREAK EVENTS FROM DAILY_FEATURES
# ---------------------------------------------------------------------------

def load_break_events() -> pd.DataFrame:
    """Load all break events with outcomes from canonical layers."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    rows = []
    for inst in INSTRUMENTS:
        for sess in INSTRUMENT_SESSIONS[inst]:
            delay_col = f"orb_{sess}_break_delay_min"
            high_col = f"orb_{sess}_high"
            low_col = f"orb_{sess}_low"
            dir_col = f"orb_{sess}_break_dir"
            ts_col = f"orb_{sess}_break_ts"

            try:
                df = con.execute(f"""
                    SELECT d.trading_day, d.symbol,
                           d.{high_col} as orb_high,
                           d.{low_col} as orb_low,
                           d.{delay_col} as break_delay_min,
                           d.{dir_col} as break_dir,
                           d.{ts_col} as break_ts_1m,
                           d.atr_20,
                           '{sess}' as session,
                           o.outcome, o.pnl_r
                    FROM daily_features d
                    JOIN orb_outcomes o
                        ON o.trading_day = d.trading_day
                        AND o.symbol = d.symbol
                        AND o.orb_minutes = d.orb_minutes
                        AND o.orb_label = '{sess}'
                        AND o.entry_model = 'E1'
                        AND o.confirm_bars = 1
                        AND o.rr_target = 1.0
                    WHERE d.symbol = '{inst}'
                      AND d.orb_minutes = {ORB_MINUTES}
                      AND d.{delay_col} IS NOT NULL
                      AND d.{high_col} IS NOT NULL
                      AND d.{low_col} IS NOT NULL
                """).fetchdf()
                if not df.empty:
                    df["instrument"] = inst
                    rows.append(df)
            except Exception as e:
                # Column may not exist for this session
                continue

    con.close()

    if not rows:
        print("ERROR: No break events found!")
        sys.exit(1)

    result = pd.concat(rows, ignore_index=True)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).dt.date
    result["break_ts_1m"] = pd.to_datetime(result["break_ts_1m"], utc=True)
    print(f"Loaded {len(result)} break events across {len(INSTRUMENTS)} instruments")
    for inst in INSTRUMENTS:
        n = len(result[result["instrument"] == inst])
        sessions = result[result["instrument"] == inst]["session"].nunique()
        print(f"  {inst}: {n} events, {sessions} sessions")
    return result


# ---------------------------------------------------------------------------
# PHASE 2: PROCESS 1S FILES — COMPUTE break_delay_sec
# ---------------------------------------------------------------------------

def vectorized_trading_day(ts_series: pd.Series) -> pd.Series:
    """Vectorized trading day computation. Same logic as pipeline."""
    ts_bris = ts_series.dt.tz_convert("Australia/Brisbane")
    shifted = ts_bris - pd.Timedelta(hours=9)
    return shifted.dt.date


def process_1s_file(filepath: Path, instrument: str, break_events: pd.DataFrame) -> list[dict]:
    """Process one 1s DBN file. Return list of {trading_day, session, break_delay_sec, ...}."""
    filename = filepath.name
    outright_pattern, prefix_len = get_outright_config(instrument, filename)

    store = db.DBNStore.from_file(str(filepath))
    try:
        df = store.to_df()
    except Exception as e:
        print(f"  WARNING: Failed to load {filename}: {e}")
        return []

    if df.empty or len(df) == 0:
        return []

    # ts_event is the index (UTC)
    df = df.reset_index()
    df.rename(columns={"ts_event": "ts_utc"}, inplace=True)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["symbol"] = df["symbol"].astype(str)

    # Filter to outright contracts only
    outright_mask = df["symbol"].str.match(outright_pattern.pattern)
    n_before = len(df)
    df = df[outright_mask].copy()

    if df.empty:
        # Try MGC fallback: maybe symbols are MGC* not GC*
        if instrument == "MGC":
            mgc_pattern = re.compile(r"^MGC[FGHJKMNQUVXZ]\d{1,2}$")
            df_full = store.to_df().reset_index()
            df_full.rename(columns={"ts_event": "ts_utc"}, inplace=True)
            df_full["ts_utc"] = pd.to_datetime(df_full["ts_utc"], utc=True)
            df_full["symbol"] = df_full["symbol"].astype(str)
            outright_mask = df_full["symbol"].str.match(mgc_pattern.pattern)
            df = df_full[outright_mask].copy()
            if not df.empty:
                outright_pattern = mgc_pattern
                prefix_len = 3
            else:
                return []
        else:
            return []

    # Assign trading days (vectorized)
    df["trading_day"] = vectorized_trading_day(df["ts_utc"])

    # Select front-month per trading day
    trading_days = df["trading_day"].unique()
    front_contracts = {}
    for td in trading_days:
        day_df = df[df["trading_day"] == td]
        volumes = day_df.groupby("symbol")["volume"].sum().to_dict()
        front = choose_front_contract(
            volumes, outright_pattern=outright_pattern, prefix_len=prefix_len
        )
        if front:
            front_contracts[td] = front

    # Filter to front-month only (vectorized via map)
    df["front_symbol"] = df["trading_day"].map(front_contracts)
    df = df[df["symbol"] == df["front_symbol"]].copy()

    if df.empty:
        return []

    # Get break events for this file's date range
    file_dates = set(df["trading_day"].unique())
    inst_events = break_events[
        (break_events["instrument"] == instrument)
        & (break_events["trading_day"].isin(file_dates))
    ]

    if inst_events.empty:
        return []

    # Group bars by trading day for fast lookup
    bars_by_day = {td: day_df for td, day_df in df.groupby("trading_day")}

    # Pre-compute break detection windows per session (cache per trading_day+session)
    window_cache = {}
    for _, event in inst_events.iterrows():
        td = event["trading_day"]
        sess = event["session"]
        key = (td, sess)
        if key not in window_cache:
            try:
                window_cache[key] = _break_detection_window(td, sess, ORB_MINUTES)
            except (ValueError, KeyError):
                window_cache[key] = None

    results = []
    for _, event in inst_events.iterrows():
        td = event["trading_day"]
        sess = event["session"]
        orb_high = event["orb_high"]
        orb_low = event["orb_low"]

        if td not in bars_by_day:
            continue

        window = window_cache.get((td, sess))
        if window is None:
            continue

        orb_end, window_end = window
        day_bars = bars_by_day[td]

        # Vectorized break detection (much faster than itertuples)
        mask = (day_bars["ts_utc"] >= orb_end) & (day_bars["ts_utc"] < window_end)
        window_bars = day_bars.loc[mask]

        if window_bars.empty:
            continue

        # Sort once, then use vectorized comparison
        window_bars = window_bars.sort_values("ts_utc")
        long_mask = window_bars["close"] > orb_high
        short_mask = window_bars["close"] < orb_low
        break_mask = long_mask | short_mask

        if not break_mask.any():
            continue

        # First bar that breaks
        first_idx = break_mask.idxmax()  # index of first True
        first_bar = window_bars.loc[first_idx]
        bar_ts = first_bar["ts_utc"]
        if isinstance(bar_ts, pd.Timestamp):
            bar_ts = bar_ts.to_pydatetime()

        delay_sec = (bar_ts - orb_end).total_seconds()
        break_dir_1s = "long" if first_bar["close"] > orb_high else "short"

        results.append({
            "trading_day": td,
            "instrument": instrument,
            "session": sess,
            "break_delay_sec": delay_sec,
            "break_delay_min_1s": delay_sec / 60.0,
            "break_delay_min_1m": event["break_delay_min"],
            "break_dir_1s": break_dir_1s,
            "break_dir_1m": event["break_dir"],
            "break_ts_1s": bar_ts,
            "break_ts_1m": event["break_ts_1m"],
            "orb_high": orb_high,
            "orb_low": orb_low,
            "atr_20": event["atr_20"],
            "outcome": event["outcome"],
            "pnl_r": event["pnl_r"],
        })

    return results


def process_instrument(instrument: str, break_events: pd.DataFrame) -> pd.DataFrame:
    """Process all 1s files for one instrument."""
    data_dir = DATA_1S_ROOT / instrument
    if not data_dir.exists():
        print(f"  WARNING: No 1s data directory for {instrument}")
        return pd.DataFrame()

    files = sorted(data_dir.glob("*.dbn.zst"))
    # Deduplicate: prefer the broader daily_ file if it overlaps a narrower daily_ file
    # In practice, just process all files; duplicates will be resolved by trading_day key
    seen_files = set()
    unique_files = []
    for f in files:
        # Skip duplicate daily files (keep the one with wider range)
        if f.name.startswith("daily_") and any(
            f2.name.startswith("daily_") and f2 != f and f2.name < f.name
            for f2 in files
        ):
            # Keep the latest daily file only
            pass
        unique_files.append(f)

    print(f"\n  Processing {instrument}: {len(unique_files)} files")

    all_results = []
    for i, filepath in enumerate(unique_files):
        print(f"    [{i+1}/{len(unique_files)}] {filepath.name}...", end=" ", flush=True)
        results = process_1s_file(filepath, instrument, break_events)
        print(f"{len(results)} breaks matched")
        all_results.extend(results)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Deduplicate: keep first occurrence per (trading_day, session)
    df = df.drop_duplicates(subset=["trading_day", "session"], keep="first")

    return df


# ---------------------------------------------------------------------------
# PHASE 3: VALIDATION — CORRELATION & SPOT-CHECKS
# ---------------------------------------------------------------------------

def validate_results(results_df: pd.DataFrame):
    """Validate 1s measurements against 1m pipeline values."""
    print("\n" + "=" * 70)
    print("PHASE 3: VALIDATION")
    print("=" * 70)

    # 3a. Direction consistency
    dir_match = (results_df["break_dir_1s"] == results_df["break_dir_1m"]).mean()
    print(f"\nDirection consistency (1s vs 1m): {dir_match:.1%}")
    if dir_match < 0.95:
        print("  WARNING: Direction mismatch > 5% — investigate!")
        mismatches = results_df[results_df["break_dir_1s"] != results_df["break_dir_1m"]]
        print(f"  {len(mismatches)} mismatches (could be genuine — 1s may detect different first break)")

    # 3b. Correlation between delays
    valid = results_df.dropna(subset=["break_delay_min_1s", "break_delay_min_1m"])
    r, p = stats.pearsonr(valid["break_delay_min_1s"], valid["break_delay_min_1m"])
    print(f"\nCorrelation (break_delay_min 1s vs 1m):")
    print(f"  Pearson r = {r:.4f}, p = {p:.2e}, N = {len(valid)}")

    # 3c. Difference distribution
    diff = valid["break_delay_min_1m"] - valid["break_delay_min_1s"]
    print(f"\nDelay difference (1m - 1s) in minutes:")
    print(f"  Mean: {diff.mean():.3f}, Median: {diff.median():.3f}")
    print(f"  Std:  {diff.std():.3f}")
    print(f"  Min:  {diff.min():.3f}, Max: {diff.max():.3f}")
    print(f"  1s is earlier in {(diff > 0).sum()} / {len(diff)} cases ({(diff > 0).mean():.1%})")
    print(f"  Same minute: {(diff.abs() < 1.0).sum()} / {len(diff)} ({(diff.abs() < 1.0).mean():.1%})")

    # 3d. Per-instrument correlation
    print(f"\nPer-instrument correlation:")
    for inst in INSTRUMENTS:
        inst_df = valid[valid["instrument"] == inst]
        if len(inst_df) > 10:
            r_inst, _ = stats.pearsonr(inst_df["break_delay_min_1s"], inst_df["break_delay_min_1m"])
            print(f"  {inst}: r={r_inst:.4f}, N={len(inst_df)}")

    # 3e. Spot-check: 5 random dates per instrument — print prices for manual verification
    print(f"\nSpot-check (5 random dates per instrument):")
    for inst in INSTRUMENTS:
        inst_df = valid[valid["instrument"] == inst]
        if len(inst_df) >= 5:
            sample = inst_df.sample(5, random_state=42)
            for _, row in sample.iterrows():
                diff_min = row["break_delay_min_1m"] - row["break_delay_min_1s"]
                print(f"  {inst} {row['session']} {row['trading_day']}: "
                      f"1m={row['break_delay_min_1m']:.0f}min, "
                      f"1s={row['break_delay_sec']:.0f}s ({row['break_delay_min_1s']:.2f}min), "
                      f"diff={diff_min:.2f}min, dir_1s={row['break_dir_1s']}")

    return r  # Return correlation for kill criterion


# ---------------------------------------------------------------------------
# PHASE 4: WR ANALYSIS — DOES 1s IMPROVE THRESHOLD SELECTION?
# ---------------------------------------------------------------------------

def wr_analysis(results_df: pd.DataFrame):
    """Compare WR spreads at 1m vs 1s thresholds."""
    print("\n" + "=" * 70)
    print("PHASE 4: WIN RATE ANALYSIS — 1s vs 1m THRESHOLDS")
    print("=" * 70)

    # Only use events with valid outcomes
    df = results_df[results_df["outcome"].isin(["win", "loss"])].copy()
    df["is_win"] = (df["outcome"] == "win").astype(float)

    # Define thresholds to test
    thresholds_sec = [60, 120, 180, 240, 300, 600]  # 1-10 min in seconds
    thresholds_min = [1, 2, 3, 4, 5, 10]  # same in minutes (for 1m comparison)

    all_tests = []  # For BH FDR

    print("\n--- Per-session WR spreads (1s thresholds) ---")
    for inst in INSTRUMENTS:
        for sess in sorted(df[df["instrument"] == inst]["session"].unique()):
            sess_df = df[(df["instrument"] == inst) & (df["session"] == sess)]
            if len(sess_df) < 50:
                continue

            # Test each 1s threshold
            for threshold_sec in thresholds_sec:
                fast = sess_df[sess_df["break_delay_sec"] <= threshold_sec]
                slow = sess_df[sess_df["break_delay_sec"] > threshold_sec]

                if len(fast) < 20 or len(slow) < 20:
                    continue

                wr_fast = fast["is_win"].mean()
                wr_slow = slow["is_win"].mean()
                wr_spread = wr_fast - wr_slow

                # Two-proportion z-test
                n1, n2 = len(fast), len(slow)
                p_pool = (fast["is_win"].sum() + slow["is_win"].sum()) / (n1 + n2)
                if p_pool == 0 or p_pool == 1:
                    continue
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                z = wr_spread / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))

                all_tests.append({
                    "instrument": inst,
                    "session": sess,
                    "threshold_sec": threshold_sec,
                    "threshold_min": threshold_sec / 60,
                    "n_fast": n1,
                    "n_slow": n2,
                    "wr_fast": wr_fast,
                    "wr_slow": wr_slow,
                    "wr_spread": wr_spread,
                    "z": z,
                    "p_value": p_val,
                    "expr_fast": fast["pnl_r"].mean(),
                    "expr_slow": slow["pnl_r"].mean(),
                })

    if not all_tests:
        print("  No tests met minimum N requirements!")
        return pd.DataFrame()

    tests_df = pd.DataFrame(all_tests)

    # BH FDR correction at honest K
    K = len(tests_df)
    tests_df = tests_df.sort_values("p_value")
    tests_df["rank"] = range(1, K + 1)
    tests_df["bh_threshold"] = tests_df["rank"] / K * 0.05
    tests_df["bh_significant"] = tests_df["p_value"] <= tests_df["bh_threshold"]

    # Find the BH cutoff
    sig = tests_df[tests_df["bh_significant"]]
    if not sig.empty:
        bh_cutoff = sig["p_value"].max()
        n_sig = len(sig)
    else:
        bh_cutoff = 0
        n_sig = 0

    print(f"\nBH FDR results: K={K}, {n_sig} significant at q=0.05")

    # Print significant results
    if n_sig > 0:
        print(f"\n{'Inst':<5} {'Session':<16} {'Thr_s':<6} {'N_fast':<7} {'N_slow':<7} "
              f"{'WR_fast':>8} {'WR_slow':>8} {'Spread':>8} {'p-value':>10} {'BH'}")
        print("-" * 95)
        for _, row in sig.iterrows():
            print(f"{row['instrument']:<5} {row['session']:<16} {row['threshold_sec']:<6.0f} "
                  f"{row['n_fast']:<7.0f} {row['n_slow']:<7.0f} "
                  f"{row['wr_fast']:>7.1%} {row['wr_slow']:>7.1%} "
                  f"{row['wr_spread']:>+7.1%} {row['p_value']:>10.4f} {'*'}")

    # Now compare: for the 3 validated sessions, is 1s better than 1m?
    validated_sessions = [
        ("MNQ", "NYSE_CLOSE"),
        ("MGC", "CME_REOPEN"),
        ("MNQ", "NYSE_OPEN"),
    ]

    print("\n--- COMPARISON: 1s vs 1m for 3 validated sessions ---")
    print(f"{'Inst':<5} {'Session':<16} {'Thr':<6} "
          f"{'WR_spread_1s':>13} {'WR_spread_1m':>13} {'Delta':>8} {'Verdict'}")
    print("-" * 80)

    for inst, sess in validated_sessions:
        sess_df = df[(df["instrument"] == inst) & (df["session"] == sess)]
        if len(sess_df) < 50:
            print(f"{inst:<5} {sess:<16} INSUFFICIENT DATA (N={len(sess_df)})")
            continue

        # 1m threshold: FAST5 (<=5 min) — the deployed threshold
        fast_1m = sess_df[sess_df["break_delay_min_1m"] <= 5]
        slow_1m = sess_df[sess_df["break_delay_min_1m"] > 5]

        # Best 1s threshold for this session (lowest p-value)
        sess_tests = tests_df[
            (tests_df["instrument"] == inst) & (tests_df["session"] == sess)
        ].sort_values("p_value")

        if sess_tests.empty:
            print(f"{inst:<5} {sess:<16} NO TESTS")
            continue

        best_1s = sess_tests.iloc[0]
        thr_sec = int(best_1s["threshold_sec"])

        # 1m WR spread at FAST5
        if len(fast_1m) >= 10 and len(slow_1m) >= 10:
            wr_spread_1m = fast_1m["is_win"].mean() - slow_1m["is_win"].mean()
        else:
            wr_spread_1m = float("nan")

        wr_spread_1s = best_1s["wr_spread"]
        delta = wr_spread_1s - wr_spread_1m if not np.isnan(wr_spread_1m) else float("nan")

        verdict = ""
        if not np.isnan(delta):
            if delta > 0.02:
                verdict = "1s BETTER (+>2%)"
            elif delta < -0.02:
                verdict = "1m BETTER"
            else:
                verdict = "EQUIVALENT"

        print(f"{inst:<5} {sess:<16} {thr_sec:<6}s "
              f"{wr_spread_1s:>+12.1%} {wr_spread_1m:>+12.1%} {delta:>+7.1%} {verdict}")

    # Check for novel sessions (BH significant, not in validated set)
    validated_set = {(i, s) for i, s in validated_sessions}
    novel = sig[~sig.apply(lambda r: (r["instrument"], r["session"]) in validated_set, axis=1)]

    print(f"\n--- NOVEL SESSIONS (BH significant, not previously validated) ---")
    if novel.empty:
        print("  None found.")
    else:
        for _, row in novel.iterrows():
            print(f"  {row['instrument']} {row['session']} @ {row['threshold_sec']:.0f}s: "
                  f"WR spread {row['wr_spread']:+.1%}, p={row['p_value']:.4f}")

    return tests_df


# ---------------------------------------------------------------------------
# PHASE 5: ATR CONFOUND CONTROL
# ---------------------------------------------------------------------------

def atr_confound_check(results_df: pd.DataFrame, tests_df: pd.DataFrame):
    """Check if 1s break speed is confounded with ATR."""
    print("\n" + "=" * 70)
    print("PHASE 5: ATR CONFOUND CONTROL")
    print("=" * 70)

    df = results_df[results_df["outcome"].isin(["win", "loss"])].copy()
    df["is_win"] = (df["outcome"] == "win").astype(float)

    # Correlation between break_delay_sec and ATR
    valid = df.dropna(subset=["break_delay_sec", "atr_20"])
    r, p = stats.pearsonr(valid["break_delay_sec"], valid["atr_20"])
    print(f"\nCorrelation (break_delay_sec vs ATR20): r={r:.4f}, p={p:.2e}")

    # For BH-significant results: re-test controlling for ATR
    sig_tests = tests_df[tests_df["bh_significant"]] if not tests_df.empty else pd.DataFrame()
    if sig_tests.empty:
        print("No BH-significant results to control.")
        return

    print(f"\nATR-controlled re-test for {len(sig_tests)} significant results:")
    for _, test in sig_tests.iterrows():
        inst = test["instrument"]
        sess = test["session"]
        thr = test["threshold_sec"]

        sess_df = df[(df["instrument"] == inst) & (df["session"] == sess)].copy()
        if len(sess_df) < 50 or sess_df["atr_20"].isna().all():
            continue

        # ATR tercile split
        sess_df["atr_tercile"] = pd.qcut(
            sess_df["atr_20"], q=3, labels=["low", "mid", "high"], duplicates="drop"
        )

        # Test break speed within each ATR tercile
        controlled_spreads = []
        for tercile in ["low", "mid", "high"]:
            t_df = sess_df[sess_df["atr_tercile"] == tercile]
            fast = t_df[t_df["break_delay_sec"] <= thr]
            slow = t_df[t_df["break_delay_sec"] > thr]
            if len(fast) >= 10 and len(slow) >= 10:
                spread = fast["is_win"].mean() - slow["is_win"].mean()
                controlled_spreads.append(spread)

        if controlled_spreads:
            avg_controlled = np.mean(controlled_spreads)
            raw_spread = test["wr_spread"]
            print(f"  {inst} {sess} @ {thr:.0f}s: raw={raw_spread:+.1%}, "
                  f"ATR-controlled={avg_controlled:+.1%} "
                  f"({'SURVIVES' if avg_controlled > 0.02 else 'WEAKENS'})")


# ---------------------------------------------------------------------------
# PHASE 6: ERA CHECK — PRE-2019 vs POST-2019
# ---------------------------------------------------------------------------

def era_check(results_df: pd.DataFrame):
    """Check if signal differs between parent (pre-2019) and native (post-2019) eras."""
    print("\n" + "=" * 70)
    print("PHASE 6: ERA CHECK — PRE-2019 vs POST-2019")
    print("=" * 70)

    df = results_df[results_df["outcome"].isin(["win", "loss"])].copy()
    df["is_win"] = (df["outcome"] == "win").astype(float)
    df["era"] = df["trading_day"].apply(lambda d: "pre-2019" if d < date(2019, 1, 1) else "post-2019")

    for inst in INSTRUMENTS:
        inst_df = df[df["instrument"] == inst]
        n_pre = len(inst_df[inst_df["era"] == "pre-2019"])
        n_post = len(inst_df[inst_df["era"] == "post-2019"])
        print(f"\n{inst}: pre-2019 N={n_pre}, post-2019 N={n_post}")

        if n_pre < 50 or n_post < 50:
            print(f"  Insufficient data for era comparison")
            continue

        # Compare break speed WR at FAST300s (5min) across eras
        for era in ["pre-2019", "post-2019"]:
            era_df = inst_df[inst_df["era"] == era]
            fast = era_df[era_df["break_delay_sec"] <= 300]
            slow = era_df[era_df["break_delay_sec"] > 300]
            if len(fast) >= 20 and len(slow) >= 20:
                spread = fast["is_win"].mean() - slow["is_win"].mean()
                print(f"  {era}: WR spread (FAST300s) = {spread:+.1%} "
                      f"(N_fast={len(fast)}, N_slow={len(slow)})")


# ---------------------------------------------------------------------------
# PHASE 7: KILL / PASS VERDICT
# ---------------------------------------------------------------------------

def verdict(correlation: float, tests_df: pd.DataFrame, results_df: pd.DataFrame):
    """Apply kill/pass criteria from plan."""
    print("\n" + "=" * 70)
    print("PHASE 7: KILL / PASS VERDICT")
    print("=" * 70)

    # Criteria from plan:
    # KILL if: 1s WR spreads within 1% of 1m for all sessions
    # KILL if: 1s reveals NO new sessions beyond 3 already validated
    # PASS if: 1s WR spread >2% better than 1m for any validated session
    # PASS if: 1s reveals new sessions with BH p < 0.05

    df = results_df[results_df["outcome"].isin(["win", "loss"])].copy()
    df["is_win"] = (df["outcome"] == "win").astype(float)

    validated_sessions = [
        ("MNQ", "NYSE_CLOSE"),
        ("MGC", "CME_REOPEN"),
        ("MNQ", "NYSE_OPEN"),
    ]

    # Check 1: Is 1s WR spread >2% better for any validated session?
    any_improvement = False
    for inst, sess in validated_sessions:
        sess_df = df[(df["instrument"] == inst) & (df["session"] == sess)]
        if len(sess_df) < 50:
            continue

        fast_1m = sess_df[sess_df["break_delay_min_1m"] <= 5]
        slow_1m = sess_df[sess_df["break_delay_min_1m"] > 5]

        if len(fast_1m) >= 10 and len(slow_1m) >= 10:
            wr_1m = fast_1m["is_win"].mean() - slow_1m["is_win"].mean()
        else:
            continue

        # Best 1s threshold
        best_spread_1s = -999
        for thr in [60, 120, 180, 240, 300]:
            fast_1s = sess_df[sess_df["break_delay_sec"] <= thr]
            slow_1s = sess_df[sess_df["break_delay_sec"] > thr]
            if len(fast_1s) >= 20 and len(slow_1s) >= 20:
                spread = fast_1s["is_win"].mean() - slow_1s["is_win"].mean()
                best_spread_1s = max(best_spread_1s, spread)

        if best_spread_1s > wr_1m + 0.02:
            any_improvement = True
            print(f"  PASS: {inst} {sess} — 1s WR spread {best_spread_1s:+.1%} > "
                  f"1m WR spread {wr_1m:+.1%} + 2%")

    # Check 2: Novel sessions with BH significance?
    validated_set = {(i, s) for i, s in validated_sessions}
    novel_sig = False
    if not tests_df.empty:
        sig = tests_df[tests_df["bh_significant"]]
        novel = sig[~sig.apply(lambda r: (r["instrument"], r["session"]) in validated_set, axis=1)]
        if not novel.empty:
            novel_sig = True
            print(f"  PASS: {len(novel)} novel BH-significant session(s) from 1s")

    print()
    if any_improvement or novel_sig:
        print("  VERDICT: PASS -- 1s resolution adds value")
        print("  Action: Build 1s break delay into pipeline")
    else:
        print("  VERDICT: KILL -- 1s adds no discovery value over 1m")
        print("  Action: Deploy 1m version (already proven).")
        print("          Archive 1s data. Save $0/mo (data is FREE).")

    # Additional info
    print(f"\n  Correlation (1s vs 1m delay): r={correlation:.4f}")
    print(f"  Total break events matched: {len(results_df)}")
    K = len(tests_df) if not tests_df.empty else 0
    n_sig = len(tests_df[tests_df["bh_significant"]]) if not tests_df.empty else 0
    print(f"  BH FDR: K={K}, significant={n_sig}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("1s BREAK SPEED RESEARCH")
    print("Does 1-second resolution improve break delay measurement?")
    print("=" * 70)
    print(f"DB: {GOLD_DB_PATH}")
    print(f"1s data: {DATA_1S_ROOT}")
    print(f"Instruments: {INSTRUMENTS}")
    print(f"ORB aperture: {ORB_MINUTES}m")

    # PHASE 1: Load break events
    print("\n" + "=" * 70)
    print("PHASE 1: LOADING BREAK EVENTS FROM DAILY_FEATURES")
    print("=" * 70)
    break_events = load_break_events()

    # PHASE 2: Process 1s files
    print("\n" + "=" * 70)
    print("PHASE 2: PROCESSING 1s FILES")
    print("=" * 70)

    all_results = []
    for inst in INSTRUMENTS:
        result_df = process_instrument(inst, break_events)
        if not result_df.empty:
            all_results.append(result_df)
            print(f"  {inst}: {len(result_df)} break delays computed from 1s data")

    if not all_results:
        print("\nERROR: No 1s break delays computed!")
        sys.exit(1)

    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal: {len(results_df)} break delays from 1s data")

    # PHASE 3: Validation
    correlation = validate_results(results_df)

    # PHASE 4: WR analysis
    tests_df = wr_analysis(results_df)

    # PHASE 5: ATR confound
    atr_confound_check(results_df, tests_df)

    # PHASE 6: Era check
    era_check(results_df)

    # PHASE 7: Verdict
    verdict(correlation, tests_df, results_df)


if __name__ == "__main__":
    main()
