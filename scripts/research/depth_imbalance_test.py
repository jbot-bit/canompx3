"""
Depth imbalance at pre-break — golden nugget hunt.

Tests: Does book imbalance in the break direction predict WR?
Mechanism: Harris (2003) — limit orders cluster at S/R. If book is
lopsided in break direction (support for longs, thin resistance for shorts),
break should carry through better.

Also tests: raw bid/ask imbalance (not directional).
"""

import warnings
from pathlib import Path

import databento as db
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

GOLD_DB_PATH = Path("gold.db")

print("=== DEPTH IMBALANCE + GOLDEN NUGGET HUNT ===")
print()

for INST in ["MNQ", "MES", "MGC"]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    sessions = ["CME_PRECLOSE", "EUROPE_FLOW", "COMEX_SETTLE", "NYSE_CLOSE", "NYSE_OPEN"]

    all_breaks = []
    for sess in sessions:
        col = f"orb_{sess}_break_ts"
        dir_col = f"orb_{sess}_break_dir"
        try:
            df = con.execute(
                f"SELECT trading_day, {col} as break_ts, {dir_col} as break_dir, "
                f"'{sess}' as session "
                f"FROM daily_features WHERE symbol = '{INST}' AND orb_minutes = 5 "
                f"AND trading_day >= '2025-04-01' AND trading_day < '2026-04-01' "
                f"AND {col} IS NOT NULL AND {dir_col} IS NOT NULL"
            ).fetchdf()
            all_breaks.append(df)
        except Exception:
            pass
    con.close()

    if not all_breaks:
        continue
    breaks_df = pd.concat(all_breaks)
    breaks_df["break_ts"] = pd.to_datetime(breaks_df["break_ts"], utc=True)

    tbbo_files = sorted(Path(f"data/raw/databento/tbbo/{INST}").glob("*tbbo_12mo*.dbn.zst"))
    if not tbbo_files:
        continue

    features = {}
    for tbbo_file in tbbo_files:
        store = db.DBNStore.from_file(str(tbbo_file))
        for chunk in store.to_df(count=500000):
            chunk_ts = pd.to_datetime(chunk.index, utc=True)
            for _, brk in breaks_df.iterrows():
                td = str(brk["trading_day"]) + "_" + brk["session"]
                if td in features:
                    continue
                bt = brk["break_ts"]
                mask = (chunk_ts >= bt - pd.Timedelta(seconds=60)) & (
                    chunk_ts < bt - pd.Timedelta(seconds=5)
                )
                pre = chunk[mask]
                if len(pre) > 0:
                    bid_d = float(pre["bid_sz_00"].median())
                    ask_d = float(pre["ask_sz_00"].median())
                    # Directional imbalance: positive = favorable for break direction
                    if brk["break_dir"] == "long":
                        dir_imb = bid_d - ask_d  # bid support = good for long
                    else:
                        dir_imb = ask_d - bid_d  # ask thin = good for short
                    # Trade intensity (quotes per second in window)
                    n_quotes = len(pre)
                    intensity = n_quotes / 55.0  # 55 second window
                    features[td] = {
                        "trading_day": brk["trading_day"],
                        "session": brk["session"],
                        "dir_imb": dir_imb,
                        "raw_imb": bid_d - ask_d,
                        "intensity": intensity,
                        "total_depth": bid_d + ask_d,
                    }

    feat_df = pd.DataFrame(features.values())
    print(f"=== {INST}: {len(feat_df)} measurements ===")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    outcomes = con.execute(
        f"SELECT o.trading_day, o.orb_label, o.outcome, o.pnl_r "
        f"FROM orb_outcomes o JOIN validated_setups v "
        f"ON o.symbol = v.instrument AND o.orb_label = v.orb_label "
        f"AND o.orb_minutes = 5 AND o.entry_model = v.entry_model "
        f"AND o.confirm_bars = v.confirm_bars AND o.rr_target = v.rr_target "
        f"WHERE o.symbol = '{INST}' AND v.status = 'active' "
        f"AND o.outcome IN ('win','loss') "
        f"AND o.trading_day >= '2025-04-01' AND o.trading_day < '2026-04-01'"
    ).fetchdf()
    con.close()

    feat_df["mk"] = feat_df["trading_day"].astype(str) + "_" + feat_df["session"]
    outcomes["mk"] = outcomes["trading_day"].astype(str) + "_" + outcomes["orb_label"]
    merged = outcomes.merge(feat_df, on="mk", how="inner")
    merged["is_win"] = (merged["outcome"] == "win").astype(int)

    if len(merged) < 50:
        print(f"  Too few ({len(merged)})")
        continue

    print(f"  Matched: {len(merged)}")

    # Test each feature
    for feat_name, label in [
        ("dir_imb", "DIRECTIONAL IMBALANCE"),
        ("intensity", "QUOTE INTENSITY"),
        ("raw_imb", "RAW BID-ASK IMBALANCE"),
    ]:
        v = merged.dropna(subset=[feat_name]).copy()
        nu = v[feat_name].nunique()
        if nu < 3:
            print(f"  {label}: {nu} unique — skip")
            continue
        med = v[feat_name].median()
        lo = v[v[feat_name] <= med]
        hi = v[v[feat_name] > med]
        if len(lo) < 15 or len(hi) < 15:
            continue
        sp = hi["is_win"].mean() - lo["is_win"].mean()
        pooled = v["is_win"].mean()
        se = np.sqrt(pooled * (1 - pooled) * (1 / len(lo) + 1 / len(hi)))
        z = sp / se if se > 0 else 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = (
            "***" if p < 0.005 else "**" if p < 0.01 else "*" if p < 0.05
            else "." if p < 0.10 else " ns"
        )
        print(f"  {label:30s} lo={lo['is_win'].mean():.3f}(N={len(lo):5d}) "
              f"hi={hi['is_win'].mean():.3f}(N={len(hi):5d}) "
              f"diff={sp:+.3f} p={p:.4f}{sig}")

    # Per-session for directional imbalance (if aggregate shows anything)
    print("  PER-SESSION (directional imbalance):")
    for sess in sorted(merged["session"].unique()):
        sub = merged[merged["session"] == sess].copy()
        if len(sub) < 30:
            continue
        nu = sub["dir_imb"].nunique()
        if nu < 3:
            continue
        med_s = sub["dir_imb"].median()
        lo_s = sub[sub["dir_imb"] <= med_s]
        hi_s = sub[sub["dir_imb"] > med_s]
        if len(lo_s) < 10 or len(hi_s) < 10:
            continue
        sp_s = hi_s["is_win"].mean() - lo_s["is_win"].mean()
        pooled_s = sub["is_win"].mean()
        se_s = np.sqrt(pooled_s * (1 - pooled_s) * (1 / len(lo_s) + 1 / len(hi_s)))
        z_s = sp_s / se_s if se_s > 0 else 0
        p_s = 2 * (1 - stats.norm.cdf(abs(z_s)))
        sig_s = (
            "***" if p_s < 0.005 else "**" if p_s < 0.01 else "*" if p_s < 0.05
            else "." if p_s < 0.10 else " ns"
        )
        print(f"    {sess:22s} N={len(sub):4d} diff={sp_s:+.3f} p={p_s:.4f}{sig_s}")

    print()

print("=== VERDICT ===")
print("If any feature shows p<0.05 with consistent direction across instruments,")
print("it warrants a full T1-T8 battery. Otherwise, ALL microstructure is dead.")
