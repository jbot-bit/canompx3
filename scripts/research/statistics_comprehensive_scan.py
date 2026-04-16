"""
Comprehensive Exchange Statistics Feature Scan — ALL angles.

Extracts EVERY usable feature from 16yr of CME statistics data.
Tests each as ORB WR predictor. No tunnel vision — all constructions.

Features (all prior-day, no look-ahead):
F1. settle_momentum:     (settle - prev_settle) / atr     — institutional price momentum
F2. oi_change:           (oi - prev_oi) / prev_oi         — position opening/closing
F3. oi_5d_change:        (oi - oi_5d) / oi_5d             — weekly position trend
F4. vol_change:          cleared_vol / prev_cleared_vol    — volume acceleration
F5. exchange_range_atr:  (session_high - session_low)/atr  — exchange range expansion
F6. ind_open_gap:        (ind_open - prev_close) / atr     — pre-market sentiment
F7. open_surprise:       (actual_open - ind_open) / atr    — opening deviation from indicative
F8. settle_close_spread: (settle - close) / atr            — settlement-close institutional spread

Literature:
- Karpoff (1987): volume-return relation
- Bessembinder & Seguin (1993): OI + volume as liquidity/information
- Chan (2009): momentum factors from daily data
- Carver: settlement as institutional anchor for futures

Gate order: T0 TAUTOLOGY (|corr| > 0.70 with existing features) → T1 WR QUINTILE → BH FDR
"""

import re
import sys
import warnings
from pathlib import Path

import databento as db
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from pipeline.ingest_dbn_mgc import choose_front_contract
from pipeline.paths import GOLD_DB_PATH

STATS_ROOT = PROJECT_ROOT / "data" / "raw" / "databento" / "statistics"
INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)

OUTRIGHT_PATTERNS = {
    "MES": [
        (re.compile(r"^ES[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        (re.compile(r"^MES[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
    "MGC": [
        (re.compile(r"^GC[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        (re.compile(r"^MGC[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
    "MNQ": [
        (re.compile(r"^NQ[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        (re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
}

# All stat types we extract
STAT_TYPES = {
    1: "opening_price",
    2: "indicative_open",
    3: "settlement",
    4: "session_low",
    5: "session_high",
    6: "cleared_volume",
    9: "open_interest",
}


def extract_all_stats(instrument: str) -> pd.DataFrame:
    """Extract ALL stat types per trading day, front-month by cleared volume."""
    stats_dir = STATS_ROOT / instrument
    if not stats_dir.exists():
        return pd.DataFrame()

    files = sorted(stats_dir.glob("*.dbn.zst"))
    all_rows = []
    patterns = OUTRIGHT_PATTERNS[instrument]

    for filepath in files:
        try:
            store = db.DBNStore.from_file(str(filepath))
            df = store.to_df()
        except Exception:
            continue

        if df.empty:
            continue

        df = df.reset_index()
        df.rename(columns={"ts_event": "ts_utc"}, inplace=True)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df["symbol"] = df["symbol"].astype(str)

        # Filter to relevant stat types
        df = df[df["stat_type"].isin(STAT_TYPES.keys())]
        if df.empty:
            continue

        # Filter to outright contracts
        outright_mask = pd.Series(False, index=df.index)
        for pattern, _ in patterns:
            outright_mask |= df["symbol"].str.match(pattern.pattern)
        df = df[outright_mask]
        if df.empty:
            continue

        df["cal_date"] = df["ts_utc"].dt.date

        for cal_date, day_df in df.groupby("cal_date"):
            row = {"cal_date": cal_date, "instrument": instrument}

            # Select front-month by cleared volume
            vol_rows = day_df[day_df["stat_type"] == 6]
            front_symbol = None
            for pattern, plen in patterns:
                matched_vol = vol_rows[vol_rows["symbol"].str.match(pattern.pattern)]
                if not matched_vol.empty:
                    vol_by_sym = matched_vol.groupby("symbol")["quantity"].sum().to_dict()
                    vol_by_sym = {s: v for s, v in vol_by_sym.items() if 0 < v < 2147483647}
                    if vol_by_sym:
                        front_symbol = choose_front_contract(
                            vol_by_sym, outright_pattern=pattern, prefix_len=plen
                        )
                        if front_symbol:
                            break

            if not front_symbol:
                continue

            # Extract each stat type for front-month
            front_df = day_df[day_df["symbol"] == front_symbol]

            for st, name in STAT_TYPES.items():
                st_rows = front_df[front_df["stat_type"] == st]
                if st_rows.empty:
                    row[name] = None
                    continue

                if st in (6, 9):  # volume and OI use quantity
                    val = st_rows["quantity"].iloc[-1]
                    row[name] = int(val) if 0 < val < 2147483647 else None
                elif st == 2:  # indicative open: use LAST update before market open
                    row[name] = float(st_rows.iloc[-1]["price"]) if st_rows.iloc[-1]["price"] > 0 else None
                else:  # price-based stats: use last update
                    val = float(st_rows.iloc[-1]["price"])
                    row[name] = val if val > 0 else None

            # Total cleared volume across ALL outrights (not just front)
            all_vol = vol_rows[vol_rows["quantity"] < 2147483647]["quantity"]
            row["total_cleared_volume"] = int(all_vol.sum()) if not all_vol.empty and all_vol.sum() > 0 else None

            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result = result.drop_duplicates(subset=["cal_date"], keep="last")
    result["cal_date"] = pd.to_datetime(result["cal_date"]).dt.date
    return result


def compute_features(stats_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all feature constructions from raw statistics + daily_features."""
    # Merge stats with daily_features
    merged = feat_df.merge(
        stats_df, left_on=["trading_day", "symbol"],
        right_on=["cal_date", "instrument"], how="inner",
    )
    merged = merged.sort_values(["symbol", "trading_day"])

    # Shift to prior-day (all features use PRIOR day's data)
    shift_cols = ["settlement", "cleared_volume", "total_cleared_volume",
                  "open_interest", "session_high", "session_low",
                  "indicative_open", "opening_price"]
    for col in shift_cols:
        if col in merged.columns:
            merged[f"prev_{col}"] = merged.groupby("symbol")[col].shift(1)

    # Also need 2-day and 5-day lag for momentum features
    for col in ["settlement", "open_interest"]:
        if col in merged.columns:
            merged[f"prev2_{col}"] = merged.groupby("symbol")[col].shift(2)
            merged[f"prev5_{col}"] = merged.groupby("symbol")[col].shift(5)

    # =====================================================================
    # FEATURE CONSTRUCTIONS
    # =====================================================================
    atr = merged["atr_20"]

    # F1: Settlement momentum (1-day change / ATR)
    merged["F1_settle_momentum"] = (merged["prev_settlement"] - merged["prev2_settlement"]) / atr

    # F2: OI change (1-day)
    prev_oi = merged["prev_open_interest"]
    prev2_oi = merged["prev2_open_interest"]
    merged["F2_oi_change"] = (prev_oi - prev2_oi) / prev2_oi.replace(0, np.nan)

    # F3: OI 5-day change
    prev5_oi = merged["prev5_open_interest"]
    merged["F3_oi_5d_change"] = (prev_oi - prev5_oi) / prev5_oi.replace(0, np.nan)

    # F4: Volume acceleration (day-over-day change)
    prev_vol = merged["prev_cleared_volume"]
    prev2_vol = merged.groupby("symbol")["prev_cleared_volume"].shift(1)
    merged["F4_vol_change"] = prev_vol / prev2_vol.replace(0, np.nan)

    # F5: Exchange range / ATR
    merged["F5_exchange_range_atr"] = (merged["prev_session_high"] - merged["prev_session_low"]) / atr

    # F6: Indicative open gap (pre-market sentiment)
    # ind_open is from same day (pre-market), prev_day_close is prior day
    merged["F6_ind_open_gap"] = (merged["indicative_open"] - merged["prev_day_close"]) / atr

    # F7: Opening surprise (actual open vs indicative)
    merged["F7_open_surprise"] = (merged["daily_open"] - merged["indicative_open"]) / atr

    # F8: Settlement-close spread (institutional vs electronic)
    merged["F8_settle_close_spread"] = (merged["prev_settlement"] - merged["prev_day_close"]) / atr

    # F9: Volume ratio vs 20MA (already tested, include for completeness)
    vol_20ma = merged.groupby("symbol")["prev_cleared_volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    merged["F9_vol_ratio_20ma"] = prev_vol / vol_20ma.replace(0, np.nan)

    # F10: OI level / 20MA (different from OI change)
    oi_20ma = merged.groupby("symbol")["prev_open_interest"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    merged["F10_oi_ratio_20ma"] = prev_oi / oi_20ma.replace(0, np.nan)

    return merged


def main():
    print("=" * 70)
    print("COMPREHENSIVE STATISTICS FEATURE SCAN")
    print("10 features x all sessions x 3 instruments")
    print("=" * 70)

    # Phase 1: Extract
    print("\n--- PHASE 1: EXTRACT ALL STATISTICS ---")
    all_stats = []
    for inst in INSTRUMENTS:
        print(f"  {inst}...", end=" ", flush=True)
        s = extract_all_stats(inst)
        print(f"{len(s)} days, cols: {[c for c in s.columns if c not in ('cal_date','instrument')]}")
        if not s.empty:
            all_stats.append(s)

    stats_df = pd.concat(all_stats, ignore_index=True)
    print(f"Total: {len(stats_df)} instrument-days")

    # Phase 2: Load daily_features
    print("\n--- PHASE 2: DAILY FEATURES ---")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    feat_df = con.execute("""
        SELECT trading_day, symbol, atr_20, daily_open, daily_close,
               daily_high, daily_low, gap_open_points,
               prev_day_close, prev_day_range, prev_day_direction,
               overnight_range
        FROM daily_features WHERE orb_minutes = 5 AND symbol IN ('MES','MGC','MNQ')
        ORDER BY symbol, trading_day
    """).fetchdf()
    con.close()
    feat_df["trading_day"] = pd.to_datetime(feat_df["trading_day"]).dt.date
    print(f"Daily features: {len(feat_df)} rows")

    # Phase 3: Compute features
    print("\n--- PHASE 3: COMPUTE FEATURES ---")
    merged = compute_features(stats_df, feat_df)

    feature_cols = [c for c in merged.columns if c.startswith("F")]
    print(f"Features computed: {len(feature_cols)}")
    for f in feature_cols:
        n_valid = merged[f].notna().sum()
        print(f"  {f}: {n_valid} valid rows ({n_valid/len(merged):.0%})")

    # =====================================================================
    # T0: TAUTOLOGY CHECK — each feature vs existing daily_features columns
    # =====================================================================
    print("\n" + "=" * 70)
    print("T0: TAUTOLOGY CHECK")
    print("=" * 70)

    existing_cols = ["atr_20", "prev_day_range", "gap_open_points",
                     "overnight_range", "prev_day_close"]
    existing_valid = merged.dropna(subset=existing_cols)

    surviving_features = []
    for f in feature_cols:
        f_valid = existing_valid.dropna(subset=[f])
        if len(f_valid) < 500:
            print(f"  {f}: SKIP (only {len(f_valid)} valid rows)")
            continue

        max_corr = 0
        max_col = ""
        for ec in existing_cols:
            try:
                r, _ = stats.pearsonr(f_valid[f], f_valid[ec])
                if abs(r) > abs(max_corr):
                    max_corr = r
                    max_col = ec
            except Exception:
                pass

        verdict = "TAUTOLOGY" if abs(max_corr) > 0.70 else "PASS"
        print(f"  {f}: max |corr| = {abs(max_corr):.4f} (vs {max_col}) -> {verdict}")

        if abs(max_corr) <= 0.70:
            surviving_features.append(f)

    print(f"\nSurviving features: {len(surviving_features)} / {len(feature_cols)}")
    if not surviving_features:
        print("\nVERDICT: ALL FEATURES KILLED AT T0")
        return

    # =====================================================================
    # T1: WR QUINTILE MONOTONICITY + BH FDR
    # =====================================================================
    print("\n" + "=" * 70)
    print("T1: WIN RATE ANALYSIS")
    print("=" * 70)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    all_tests = []

    sessions = set()
    for inst in INSTRUMENTS:
        sessions.update(ASSET_CONFIGS[inst].get("enabled_sessions", []))

    for feat in surviving_features:
        for sess in sorted(sessions):
            for inst in INSTRUMENTS:
                if sess not in ASSET_CONFIGS[inst].get("enabled_sessions", []):
                    continue

                try:
                    outcomes = con.execute(f"""
                        SELECT o.trading_day, o.outcome, o.pnl_r
                        FROM orb_outcomes o
                        WHERE o.symbol = '{inst}' AND o.orb_label = '{sess}'
                          AND o.orb_minutes = 5 AND o.entry_model = 'E1'
                          AND o.confirm_bars = 1 AND o.rr_target = 1.0
                          AND o.outcome IN ('win', 'loss')
                    """).fetchdf()
                except Exception:
                    continue

                if len(outcomes) < 100:
                    continue

                outcomes["trading_day"] = pd.to_datetime(outcomes["trading_day"]).dt.date
                outcomes["is_win"] = (outcomes["outcome"] == "win").astype(float)

                feat_data = merged[merged["symbol"] == inst][["trading_day", feat]].dropna()
                joined = outcomes.merge(feat_data, on="trading_day", how="inner")

                if len(joined) < 100:
                    continue

                try:
                    joined["quintile"] = pd.qcut(joined[feat], q=5, labels=False, duplicates="drop")
                except ValueError:
                    continue

                wr_by_q = joined.groupby("quintile")["is_win"].agg(["mean", "count"])
                if len(wr_by_q) < 3:
                    continue

                wr_spread = wr_by_q["mean"].iloc[-1] - wr_by_q["mean"].iloc[0]

                top = joined[joined["quintile"] == joined["quintile"].max()]
                bot = joined[joined["quintile"] == joined["quintile"].min()]
                n_top, n_bot = len(top), len(bot)
                if n_top < 20 or n_bot < 20:
                    continue

                p_pool = (top["is_win"].sum() + bot["is_win"].sum()) / (n_top + n_bot)
                if p_pool in (0, 1):
                    continue
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n_top + 1/n_bot))
                z = wr_spread / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))

                all_tests.append({
                    "feature": feat, "instrument": inst, "session": sess,
                    "wr_spread": wr_spread, "wr_q5": wr_by_q["mean"].iloc[-1],
                    "wr_q1": wr_by_q["mean"].iloc[0],
                    "n_top": n_top, "n_bot": n_bot, "N": len(joined),
                    "z": z, "p_value": p_val,
                })

    con.close()

    if not all_tests:
        print("  No tests met N requirements!")
        return

    tests_df = pd.DataFrame(all_tests)
    K = len(tests_df)
    tests_df = tests_df.sort_values("p_value")
    tests_df["rank"] = range(1, K + 1)
    tests_df["bh_threshold"] = tests_df["rank"] / K * 0.05
    tests_df["bh_significant"] = tests_df["p_value"] <= tests_df["bh_threshold"]

    sig = tests_df[tests_df["bh_significant"]]

    print(f"\n  Tests: K={K} ({len(surviving_features)} features x {len(sessions)} sessions x {len(INSTRUMENTS)} instruments)")
    print(f"  BH FDR significant: {len(sig)}")

    print("\n  Top 15 by p-value:")
    print(f"  {'Feature':<25} {'Inst':<5} {'Session':<16} {'WR_Q5':>7} {'WR_Q1':>7} {'Spread':>8} {'N':>6} {'p':>10} {'BH'}")
    print("  " + "-" * 100)
    for _, r in tests_df.head(15).iterrows():
        bh = "*" if r["bh_significant"] else ""
        print(f"  {r['feature']:<25} {r['instrument']:<5} {r['session']:<16} "
              f"{r['wr_q5']:>6.1%} {r['wr_q1']:>6.1%} {r['wr_spread']:>+7.1%} "
              f"{r['N']:>6.0f} {r['p_value']:>10.4f} {bh}")

    # Per-feature summary
    print("\n  Per-feature best results:")
    for feat in surviving_features:
        ft = tests_df[tests_df["feature"] == feat]
        best = ft.iloc[0] if not ft.empty else None
        n_sig = len(ft[ft["bh_significant"]])
        if best is not None:
            print(f"  {feat:<25}: best p={best['p_value']:.4f} ({best['instrument']} {best['session']} {best['wr_spread']:+.1%}), "
                  f"BH sig: {n_sig}/{len(ft)}")

    # Cross-instrument check for significant results
    if len(sig) > 0:
        print("\n  Cross-instrument concordance for BH-significant features:")
        for feat in sig["feature"].unique():
            ft = tests_df[tests_df["feature"] == feat]
            for inst in INSTRUMENTS:
                inst_ft = ft[ft["instrument"] == inst].sort_values("p_value")
                if not inst_ft.empty:
                    best = inst_ft.iloc[0]
                    print(f"    {feat} {inst}: best={best['wr_spread']:+.1%} p={best['p_value']:.4f} ({best['session']})")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if len(sig) == 0:
        print(f"  KILL: 0/{K} BH-significant. No novel WR prediction from statistics.")
        print("  All 10 feature constructions tested. None survive honest testing.")
    else:
        strong = sig[sig["wr_spread"].abs() >= 0.03]
        if len(strong) == 0:
            print(f"  KILL: {len(sig)} BH-sig but all weak (<3% WR spread). ARITHMETIC_ONLY.")
        else:
            print(f"  CONDITIONAL PASS: {len(strong)} strong BH-significant.")
            print("  Requires T3-T8 validation before deployment.")


if __name__ == "__main__":
    main()
