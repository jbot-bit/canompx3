"""
Pre-break depth research — proper methodology.

Tests: Does pre-break quote depth predict ORB breakout WR?
Uses depth measured 30-60 seconds BEFORE the break (exogenous).
Tested per-instrument, per-session, with ATR confound control.

Literature:
- Kyle (1985): depth = liquidity. Thin depth = higher price impact.
- Harris (2003): limit orders cluster at S/R. Sweep dynamics at breakout.
- Bessembinder & Seguin (1993): depth proxies liquidity, dampens volatility.
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

print("=== PRE-BREAK DEPTH RESEARCH ===")
print("Measuring depth 30-60s BEFORE break (exogenous)")
print()

results_all = []

for INST in ["MES", "MGC", "MNQ"]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    sessions = ["CME_PRECLOSE", "EUROPE_FLOW", "COMEX_SETTLE", "NYSE_CLOSE", "NYSE_OPEN"]

    all_breaks = []
    for sess in sessions:
        col = f"orb_{sess}_break_ts"
        try:
            df = con.execute(
                f"SELECT trading_day, {col} as break_ts, '{sess}' as session "
                f"FROM daily_features WHERE symbol = '{INST}' AND orb_minutes = 5 "
                f"AND trading_day >= '2025-04-01' AND trading_day < '2026-04-01' "
                f"AND {col} IS NOT NULL"
            ).fetchdf()
            all_breaks.append(df)
        except Exception:
            pass

    atr_df = con.execute(
        f"SELECT trading_day, atr_20 FROM daily_features "
        f"WHERE symbol = '{INST}' AND orb_minutes = 5 "
        f"AND trading_day >= '2025-04-01' AND trading_day < '2026-04-01'"
    ).fetchdf()

    con.close()
    if not all_breaks:
        continue

    breaks_df = pd.concat(all_breaks)
    breaks_df["break_ts"] = pd.to_datetime(breaks_df["break_ts"], utc=True)

    tbbo_dir = Path(f"data/raw/databento/tbbo/{INST}")
    tbbo_files = sorted(tbbo_dir.glob("*tbbo_12mo*.dbn.zst"))
    if not tbbo_files:
        continue

    # Extract PRE-BREAK depth (30-60s before break)
    pre_depth = {}
    for tbbo_file in tbbo_files:
        store = db.DBNStore.from_file(str(tbbo_file))
        for chunk in store.to_df(count=500000):
            chunk_ts = pd.to_datetime(chunk.index, utc=True)
            for _, brk in breaks_df.iterrows():
                td = str(brk["trading_day"]) + "_" + brk["session"]
                if td in pre_depth:
                    continue
                bt = brk["break_ts"]
                mask = (chunk_ts >= bt - pd.Timedelta(seconds=60)) & (
                    chunk_ts < bt - pd.Timedelta(seconds=5)
                )
                pre = chunk[mask]
                if len(pre) > 0:
                    pre_depth[td] = {
                        "trading_day": brk["trading_day"],
                        "session": brk["session"],
                        "pre_depth": float((pre["bid_sz_00"] + pre["ask_sz_00"]).median()),
                        "n_quotes": len(pre),
                    }

    feat_df = pd.DataFrame(pre_depth.values())
    print(f"=== {INST}: {len(feat_df)} pre-break depth measurements ===")
    print(
        f"  Depth: mean={feat_df['pre_depth'].mean():.0f} "
        f"std={feat_df['pre_depth'].std():.0f} "
        f"min={feat_df['pre_depth'].min():.0f} "
        f"max={feat_df['pre_depth'].max():.0f}"
    )

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    outcomes = con.execute(
        f"SELECT o.trading_day, o.orb_label, o.outcome, o.pnl_r "
        f"FROM orb_outcomes o "
        f"JOIN validated_setups v "
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
    td_col = "trading_day_x" if "trading_day_x" in merged.columns else "trading_day"
    merged = merged.merge(atr_df, left_on=td_col, right_on="trading_day", how="left", suffixes=("", "_atr"))

    print(f"  Matched outcomes: {len(merged)}")

    if len(merged) < 50:
        print("  Too few\n")
        continue

    # UNCONDITIONAL
    med = merged["pre_depth"].median()
    lo = merged[merged["pre_depth"] <= med]
    hi = merged[merged["pre_depth"] > med]
    sp = hi["is_win"].mean() - lo["is_win"].mean()
    pooled = merged["is_win"].mean()
    se = np.sqrt(pooled * (1 - pooled) * (1 / len(lo) + 1 / len(hi)))
    z = sp / se if se > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    sig = "***" if p < 0.005 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else " ns"
    direction = "THIN=BETTER" if sp < 0 else "THICK=BETTER"
    print(f"  UNCONDITIONAL: lo={lo['is_win'].mean():.3f}(N={len(lo)}) hi={hi['is_win'].mean():.3f}(N={len(hi)}) diff={sp:+.3f} p={p:.4f}{sig} [{direction}]")
    results_all.append({"inst": INST, "test": "unconditional", "diff": sp, "p": p, "direction": direction, "N": len(merged)})

    # PER-SESSION
    print("  PER-SESSION:")
    for sess in sorted(merged["session"].unique()):
        sub = merged[merged["session"] == sess]
        if len(sub) < 30:
            continue
        sub = sub.copy()
        med_s = sub["pre_depth"].median()
        lo_s = sub[sub["pre_depth"] <= med_s]
        hi_s = sub[sub["pre_depth"] > med_s]
        if len(lo_s) < 10 or len(hi_s) < 10:
            continue
        sp_s = hi_s["is_win"].mean() - lo_s["is_win"].mean()
        pooled_s = sub["is_win"].mean()
        se_s = np.sqrt(pooled_s * (1 - pooled_s) * (1 / len(lo_s) + 1 / len(hi_s)))
        z_s = sp_s / se_s if se_s > 0 else 0
        p_s = 2 * (1 - stats.norm.cdf(abs(z_s)))
        sig_s = "***" if p_s < 0.005 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "." if p_s < 0.10 else " ns"
        dir_s = "THIN=BETTER" if sp_s < 0 else "THICK=BETTER"
        print(f"    {sess:22s} N={len(sub):4d} diff={sp_s:+.3f} p={p_s:.4f}{sig_s} [{dir_s}]")
        results_all.append({"inst": INST, "test": sess, "diff": sp_s, "p": p_s, "direction": dir_s, "N": len(sub)})

    # ATR CONFOUND CONTROL
    if "atr_20" in merged.columns and merged["atr_20"].notna().sum() > 50:
        print("  ATR-CONTROLLED:")
        mc = merged.dropna(subset=["atr_20"]).copy()
        try:
            mc["atr_half"] = pd.qcut(mc["atr_20"], 2, labels=["low_atr", "high_atr"], duplicates="drop")
            for ag in ["low_atr", "high_atr"]:
                sub = mc[mc["atr_half"] == ag]
                if len(sub) < 30:
                    continue
                med_a = sub["pre_depth"].median()
                lo_a = sub[sub["pre_depth"] <= med_a]
                hi_a = sub[sub["pre_depth"] > med_a]
                if len(lo_a) < 10 or len(hi_a) < 10:
                    continue
                sp_a = hi_a["is_win"].mean() - lo_a["is_win"].mean()
                pooled_a = sub["is_win"].mean()
                se_a = np.sqrt(pooled_a * (1 - pooled_a) * (1 / len(lo_a) + 1 / len(hi_a)))
                z_a = sp_a / se_a if se_a > 0 else 0
                p_a = 2 * (1 - stats.norm.cdf(abs(z_a)))
                sig_a = "***" if p_a < 0.005 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "." if p_a < 0.10 else " ns"
                dir_a = "THIN=BETTER" if sp_a < 0 else "THICK=BETTER"
                print(f"    {ag:10s} N={len(sub):4d} diff={sp_a:+.3f} p={p_a:.4f}{sig_a} [{dir_a}]")
        except ValueError:
            print("    ATR stratification failed")

    print()

# CROSS-INSTRUMENT SUMMARY
print("=== CROSS-INSTRUMENT SUMMARY ===")
rdf = pd.DataFrame(results_all)
uncond = rdf[rdf["test"] == "unconditional"]
for _, r in uncond.iterrows():
    sig = "***" if r["p"] < 0.005 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "." if r["p"] < 0.10 else " ns"
    print(f"  {r['inst']:4s} N={r['N']:5d} diff={r['diff']:+.3f} p={r['p']:.4f}{sig} [{r['direction']}]")

dirs = list(uncond["direction"].values)
concordant = len(set(dirs)) == 1
print(f"\n  Concordance: {'YES' if concordant else 'NO — MIXED: ' + str(dirs)}")

# BH FDR
per_sess = rdf[rdf["test"] != "unconditional"].sort_values("p").reset_index(drop=True)
K = len(per_sess)
if K > 0:
    print(f"\n=== BH FDR (K={K}) ===")
    n_surv = 0
    for rank, (_, r) in enumerate(per_sess.iterrows(), 1):
        thr = 0.05 * rank / K
        surv = r["p"] < thr
        if surv:
            n_surv = rank
        m = "SURVIVES" if surv else "fails"
        print(f"  {rank:2d}. {r['inst']:4s} {r['test']:22s} diff={r['diff']:+.3f} p={r['p']:.4f} thr={thr:.4f} {m} [{r['direction']}]")
        if rank >= 20:
            break
    print(f"\n  Survivors: {n_surv}/{K}")
