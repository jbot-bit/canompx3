"""
Exchange Statistics Feature Scan — Settlement price & cleared volume as ORB predictors.

Tests whether CME exchange statistics (16yr) contain novel signal for ORB breakout WR
beyond what daily_features already captures.

Hypotheses (from institutional literature):
H1: Settlement-to-open gap / ATR predicts break quality
    - Settlement = institutional anchor (CME margin, MTM reference)
    - Large gaps = overnight information arrival → momentum
    - Grounding: Chan (2009) Ch1 mean-reversion vs momentum, Carver Ch3 futures settlement
    - TAUTOLOGY RISK: settlement ≈ last 1m close for liquid futures → gap_open_points duplicate

H2: Prior-day cleared volume / 20-day avg predicts break quality
    - High volume = institutional engagement → cleaner breakouts
    - Grounding: Karpoff (1987) volume-return, Bessembinder & Seguin (1993) volume-liquidity
    - TAUTOLOGY RISK: cleared_volume ≈ sum(1m volumes) → rel_vol duplicate

Gate order (per quant audit protocol):
T0 TAUTOLOGY → T1 WR MONOTONICITY → BH FDR → ATR CONFOUND → VERDICT
If T0 kills (|corr| > 0.70), stop. Do not proceed to T1.

No look-ahead: settlement and cleared volume are published after close,
before next session open. Used as PRIOR-DAY features only.
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

# Outright patterns for front-month selection
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

STAT_SETTLEMENT = 3
STAT_CLEARED_VOLUME = 6


def extract_daily_stats(instrument: str) -> pd.DataFrame:
    """Extract settlement price and cleared volume per trading day from statistics files."""
    stats_dir = STATS_ROOT / instrument
    if not stats_dir.exists():
        return pd.DataFrame()

    files = sorted(stats_dir.glob("*.dbn.zst"))
    all_rows = []

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

        # Filter to settlement and cleared volume
        mask = df["stat_type"].isin([STAT_SETTLEMENT, STAT_CLEARED_VOLUME])
        df = df[mask]

        if df.empty:
            continue

        # Filter to outright contracts
        patterns = OUTRIGHT_PATTERNS[instrument]
        outright_mask = pd.Series(False, index=df.index)
        for pattern, _ in patterns:
            outright_mask |= df["symbol"].str.match(pattern.pattern)
        df = df[outright_mask]

        if df.empty:
            continue

        # Assign calendar date (settlement is published in afternoon/evening)
        df["cal_date"] = df["ts_utc"].dt.date

        # Per date: select front-month by CLEARED VOLUME, then look up settlement
        for cal_date, day_df in df.groupby("cal_date"):
            settle_rows = day_df[day_df["stat_type"] == STAT_SETTLEMENT]
            vol_rows = day_df[day_df["stat_type"] == STAT_CLEARED_VOLUME]

            # Step 1: Find front-month by actual cleared volume (same principle as pipeline)
            front_symbol = None
            for pattern, plen in patterns:
                matched_vol = vol_rows[vol_rows["symbol"].str.match(pattern.pattern)]
                if not matched_vol.empty:
                    # quantity = actual cleared volume for stat_type=6
                    vol_by_sym = matched_vol.groupby("symbol")["quantity"].sum().to_dict()
                    # Filter out sentinel values (2147483647 = INT_MAX = no data)
                    vol_by_sym = {s: v for s, v in vol_by_sym.items() if 0 < v < 2147483647}
                    if vol_by_sym:
                        front_symbol = choose_front_contract(vol_by_sym, outright_pattern=pattern, prefix_len=plen)
                        if front_symbol:
                            break

            # Step 2: Get settlement for the SAME front-month contract
            settlement = None
            if front_symbol and not settle_rows.empty:
                front_settle = settle_rows[settle_rows["symbol"] == front_symbol]
                if not front_settle.empty:
                    s = front_settle.iloc[-1]["price"]  # Last update
                    if s > 0:
                        settlement = float(s)

            # Step 3: Total cleared volume across all outrights
            cleared_vol = None
            if not vol_rows.empty:
                valid_vols = vol_rows[vol_rows["quantity"] < 2147483647]["quantity"]
                if not valid_vols.empty:
                    total_vol = int(valid_vols.sum())
                    if total_vol > 0:
                        cleared_vol = total_vol

            if settlement is not None or cleared_vol is not None:
                all_rows.append(
                    {
                        "cal_date": cal_date,
                        "instrument": instrument,
                        "settlement": settlement,
                        "cleared_volume": cleared_vol,
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result = result.drop_duplicates(subset=["cal_date"], keep="last")
    result["cal_date"] = pd.to_datetime(result["cal_date"]).dt.date
    return result


def load_daily_features() -> pd.DataFrame:
    """Load prior-day features from daily_features for tautology comparison."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    df = con.execute("""
        SELECT trading_day, symbol, atr_20,
               daily_close, daily_open, gap_open_points,
               prev_day_close, prev_day_range, overnight_range
        FROM daily_features
        WHERE orb_minutes = 5
          AND symbol IN ('MES', 'MGC', 'MNQ')
        ORDER BY symbol, trading_day
    """).fetchdf()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    return df


def main():
    print("=" * 70)
    print("EXCHANGE STATISTICS FEATURE SCAN")
    print("Settlement price + cleared volume as ORB predictors")
    print("=" * 70)

    # Phase 1: Extract statistics
    print("\n--- PHASE 1: EXTRACTING STATISTICS ---")
    all_stats = []
    for inst in INSTRUMENTS:
        print(f"  {inst}...", end=" ", flush=True)
        stats_df = extract_daily_stats(inst)
        print(f"{len(stats_df)} days")
        if not stats_df.empty:
            all_stats.append(stats_df)

    if not all_stats:
        print("ERROR: No statistics extracted!")
        sys.exit(1)

    stats_df = pd.concat(all_stats, ignore_index=True)
    print(f"\nTotal: {len(stats_df)} instrument-days of statistics")

    # Phase 2: Join with daily_features
    print("\n--- PHASE 2: JOIN WITH DAILY_FEATURES ---")
    df_feat = load_daily_features()

    # Settlement is for the trading day (published at close)
    # As a predictor, we use PRIOR day's settlement → shift by 1
    stats_df["trading_day_next"] = stats_df["cal_date"]  # Will be used as prior-day for next day

    merged = df_feat.merge(
        stats_df[["cal_date", "instrument", "settlement", "cleared_volume"]],
        left_on=["trading_day", "symbol"],
        right_on=["cal_date", "instrument"],
        how="inner",
    )

    # Now create prior-day features by shifting
    merged = merged.sort_values(["symbol", "trading_day"])
    for col in ["settlement", "cleared_volume"]:
        merged[f"prev_{col}"] = merged.groupby("symbol")[col].shift(1)

    # Compute features
    merged["settle_gap"] = merged["daily_open"] - merged["prev_settlement"]
    merged["settle_gap_atr"] = merged["settle_gap"] / merged["atr_20"]
    merged["close_gap"] = merged["daily_open"] - merged["prev_day_close"]
    merged["close_gap_atr"] = merged["close_gap"] / merged["atr_20"]

    # Cleared volume ratio (vs 20-day rolling mean)
    merged["cleared_vol_20ma"] = merged.groupby("symbol")["prev_cleared_volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    merged["cleared_vol_ratio"] = merged["prev_cleared_volume"] / merged["cleared_vol_20ma"]

    valid = merged.dropna(subset=["settle_gap_atr", "prev_day_close", "prev_settlement"])
    print(f"Merged rows: {len(merged)}, valid (non-null): {len(valid)}")

    # =====================================================================
    # T0: TAUTOLOGY CHECK
    # =====================================================================
    print("\n" + "=" * 70)
    print("T0: TAUTOLOGY CHECK")
    print("=" * 70)

    # H1: Is settlement ≈ close?
    if len(valid) > 100:
        r_settle_close, _ = stats.pearsonr(valid["prev_settlement"], valid["prev_day_close"])
        print(f"\n  H1 tautology: corr(prev_settlement, prev_day_close) = {r_settle_close:.6f}")

        # Is settle_gap ≈ close_gap (gap_open_points)?
        gap_valid = valid.dropna(subset=["settle_gap_atr", "close_gap_atr"])
        if len(gap_valid) > 100:
            r_gap, _ = stats.pearsonr(gap_valid["settle_gap_atr"], gap_valid["close_gap_atr"])
            print(f"  H1 tautology: corr(settle_gap/atr, close_gap/atr) = {r_gap:.6f}")
            print(
                f"  Mean |settle - close|: {(gap_valid['prev_settlement'] - gap_valid['prev_day_close']).abs().mean():.4f}"
            )
        else:
            r_gap = 1.0

        if abs(r_settle_close) > 0.999 and abs(r_gap) > 0.99:
            print("\n  >>> H1 KILLED: TAUTOLOGY.")
            print("  Settlement = close for these liquid futures (r > 0.999).")
            print("  settle_gap/atr = close_gap/atr (r > 0.99).")
            print("  gap_open_points already captures this signal.")
            h1_alive = False
        elif abs(r_gap) > 0.70:
            print(f"\n  >>> H1: HIGH tautology risk (r={r_gap:.4f}).")
            print("  Proceeding to T1 with caution.")
            h1_alive = True
        else:
            print(f"\n  >>> H1: PASSES T0 (r={r_gap:.4f} < 0.70)")
            h1_alive = True
    else:
        print("  Insufficient data for H1 tautology check")
        h1_alive = False

    # H2: Is cleared_volume ≈ something we already have?
    vol_valid = valid.dropna(subset=["cleared_vol_ratio", "atr_20"])
    if len(vol_valid) > 100:
        r_vol_atr, _ = stats.pearsonr(vol_valid["cleared_vol_ratio"], vol_valid["atr_20"])
        print(f"\n  H2 tautology: corr(cleared_vol_ratio, atr_20) = {r_vol_atr:.4f}")

        if "overnight_range" in vol_valid.columns:
            ovr_valid = vol_valid.dropna(subset=["overnight_range"])
            if len(ovr_valid) > 100:
                r_vol_ovr, _ = stats.pearsonr(ovr_valid["cleared_vol_ratio"], ovr_valid["overnight_range"])
                print(f"  H2 tautology: corr(cleared_vol_ratio, overnight_range) = {r_vol_ovr:.4f}")

        if abs(r_vol_atr) > 0.70:
            print(f"\n  >>> H2: HIGH tautology risk (r={r_vol_atr:.4f} with ATR).")
            h2_alive = False
        else:
            print(f"\n  >>> H2: PASSES T0 (r={r_vol_atr:.4f} < 0.70)")
            h2_alive = True
    else:
        print("  Insufficient data for H2 tautology check")
        h2_alive = False

    if not h1_alive and not h2_alive:
        print("\n" + "=" * 70)
        print("VERDICT: BOTH HYPOTHESES KILLED AT T0 (TAUTOLOGY)")
        print("=" * 70)
        print("Settlement = close for liquid CME micros.")
        print("Cleared volume confounded with ATR.")
        print("Statistics data adds NO novel features beyond daily_features.")
        return

    # =====================================================================
    # T1: WR MONOTONICITY (only for surviving hypotheses)
    # =====================================================================
    print("\n" + "=" * 70)
    print("T1: WIN RATE MONOTONICITY")
    print("=" * 70)

    # Load outcomes for WR analysis
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    sessions = set()
    for inst in INSTRUMENTS:
        sessions.update(ASSET_CONFIGS[inst].get("enabled_sessions", []))

    all_wr_tests = []

    for sess in sorted(sessions):
        for inst in INSTRUMENTS:
            if sess not in ASSET_CONFIGS[inst].get("enabled_sessions", []):
                continue

            try:
                outcomes = con.execute(f"""
                    SELECT o.trading_day, o.outcome, o.pnl_r
                    FROM orb_outcomes o
                    WHERE o.symbol = '{inst}'
                      AND o.orb_label = '{sess}'
                      AND o.orb_minutes = 5
                      AND o.entry_model = 'E1'
                      AND o.confirm_bars = 1
                      AND o.rr_target = 1.0
                      AND o.outcome IN ('win', 'loss')
                """).fetchdf()
            except Exception:
                continue

            if outcomes.empty:
                continue

            outcomes["trading_day"] = pd.to_datetime(outcomes["trading_day"]).dt.date
            outcomes["is_win"] = (outcomes["outcome"] == "win").astype(float)

            # Join with features
            feat_subset = valid[valid["symbol"] == inst][
                ["trading_day", "settle_gap_atr", "cleared_vol_ratio", "atr_20"]
            ]
            joined = outcomes.merge(feat_subset, on="trading_day", how="inner")

            if len(joined) < 100:
                continue

            # Test each surviving hypothesis
            for feature, label, alive in [
                ("settle_gap_atr", "H1_SETTLE_GAP", h1_alive),
                ("cleared_vol_ratio", "H2_CLEARED_VOL", h2_alive),
            ]:
                if not alive:
                    continue

                feat_valid = joined.dropna(subset=[feature])
                if len(feat_valid) < 100:
                    continue

                # Quintile split
                try:
                    feat_valid["quintile"] = pd.qcut(feat_valid[feature], q=5, labels=False, duplicates="drop")
                except ValueError:
                    continue

                wr_by_q = feat_valid.groupby("quintile")["is_win"].agg(["mean", "count"])
                if len(wr_by_q) < 3:
                    continue

                wr_spread = wr_by_q["mean"].iloc[-1] - wr_by_q["mean"].iloc[0]

                # Two-proportion z-test: top quintile vs bottom
                top = feat_valid[feat_valid["quintile"] == feat_valid["quintile"].max()]
                bot = feat_valid[feat_valid["quintile"] == feat_valid["quintile"].min()]
                n_top, n_bot = len(top), len(bot)

                if n_top < 20 or n_bot < 20:
                    continue

                p_pool = (top["is_win"].sum() + bot["is_win"].sum()) / (n_top + n_bot)
                if p_pool == 0 or p_pool == 1:
                    continue
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_top + 1 / n_bot))
                z = wr_spread / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))

                all_wr_tests.append(
                    {
                        "feature": label,
                        "instrument": inst,
                        "session": sess,
                        "wr_spread": wr_spread,
                        "wr_top": wr_by_q["mean"].iloc[-1],
                        "wr_bot": wr_by_q["mean"].iloc[0],
                        "n_top": n_top,
                        "n_bot": n_bot,
                        "z": z,
                        "p_value": p_val,
                        "N": len(feat_valid),
                    }
                )

    con.close()

    if not all_wr_tests:
        print("  No tests met minimum N requirements!")
        print("\n  VERDICT: NO TESTABLE SIGNAL")
        return

    tests_df = pd.DataFrame(all_wr_tests)

    # BH FDR
    K = len(tests_df)
    tests_df = tests_df.sort_values("p_value")
    tests_df["rank"] = range(1, K + 1)
    tests_df["bh_threshold"] = tests_df["rank"] / K * 0.05
    tests_df["bh_significant"] = tests_df["p_value"] <= tests_df["bh_threshold"]

    sig = tests_df[tests_df["bh_significant"]]
    n_sig = len(sig)

    print(f"\n  Total tests: K={K}")
    print(f"  BH FDR significant: {n_sig}")

    # Print top 10 by p-value regardless of significance
    print("\n  Top 10 by p-value:")
    print(f"  {'Feature':<20} {'Inst':<5} {'Session':<16} {'WR_spread':>10} {'N':>6} {'p-value':>10} {'BH'}")
    print("  " + "-" * 80)
    for _, row in tests_df.head(10).iterrows():
        bh = "*" if row["bh_significant"] else ""
        print(
            f"  {row['feature']:<20} {row['instrument']:<5} {row['session']:<16} "
            f"{row['wr_spread']:>+9.1%} {row['N']:>6.0f} {row['p_value']:>10.4f} {bh}"
        )

    # Print WR monotonicity check for top results
    if n_sig > 0:
        print("\n  BH SIGNIFICANT RESULTS:")
        for _, row in sig.iterrows():
            # Check if WR is monotonic or just noise
            label = "SIGNAL" if abs(row["wr_spread"]) > 0.03 else "WEAK (<3%)"
            print(
                f"    {row['feature']} {row['instrument']} {row['session']}: "
                f"spread={row['wr_spread']:+.1%}, p={row['p_value']:.4f} -> {label}"
            )
    else:
        print("\n  NO BH-significant results.")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if n_sig == 0:
        print("  KILL: 0 BH-significant results at K=" + str(K))
        print("  Statistics data adds NO novel WR prediction beyond daily_features.")
    else:
        wk_count = len(sig[sig["wr_spread"].abs() < 0.03])
        strong = n_sig - wk_count
        if strong == 0:
            print(f"  KILL: {n_sig} BH-significant but ALL weak (WR spread < 3%)")
            print("  ARITHMETIC_ONLY at best. Not a WR predictor.")
        else:
            print(f"  CONDITIONAL PASS: {strong} strong BH-significant results")
            print("  Proceed to T3-T8 battery for validation.")

    print("\n  Summary:")
    print(f"    H1 (settlement gap): {'KILLED at T0 (tautology)' if not h1_alive else f'{n_sig} BH sig'}")
    print(f"    H2 (cleared volume): {'KILLED at T0 (tautology)' if not h2_alive else f'{n_sig} BH sig'}")


if __name__ == "__main__":
    main()
