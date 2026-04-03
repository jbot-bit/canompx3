"""What kills the edge? Find conditions where validated strategies lose money.

Instead of finding NEW signals, find AVOID conditions for existing strategies.
If we can identify and skip the worst 10-20% of trades, portfolio improves.

Scoped to VALIDATED strategies with filters applied.
Tests every available daily_features dimension for predictive power on losses.

@research-source: what_kills_edge.py
@data-source: orb_outcomes JOIN daily_features, filtered via validated_setups
"""

import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

strats = con.execute(
    "SELECT strategy_id, instrument, orb_label, entry_model, "
    "confirm_bars, rr_target, filter_type "
    "FROM validated_setups WHERE status = 'active' AND orb_minutes = 5"
).fetchdf()

daily = con.execute("SELECT * FROM daily_features WHERE symbol IN ('MGC', 'MNQ', 'MES') AND orb_minutes = 5").fetchdf()

outcomes = con.execute(
    "SELECT trading_day, symbol, orb_label, orb_minutes, entry_model, "
    "confirm_bars, rr_target, entry_price, stop_price, pnl_r, "
    "risk_dollars, mae_r, mfe_r "
    "FROM orb_outcomes "
    "WHERE orb_minutes = 5 AND symbol IN ('MGC', 'MNQ', 'MES') AND pnl_r IS NOT NULL"
).fetchdf()

# Build filtered trade set with ALL daily_features columns
all_trades = []
for _, strat in strats.iterrows():
    inst = strat["instrument"]
    orb_label = strat["orb_label"]
    filt = ALL_FILTERS.get(strat["filter_type"])
    if filt is None:
        continue
    inst_daily = daily[daily["symbol"] == inst].copy()
    if inst_daily.empty:
        continue
    eligible_mask = filt.matches_df(inst_daily, orb_label)
    eligible_days = inst_daily.loc[eligible_mask].copy()
    if eligible_days.empty:
        continue
    strat_outcomes = outcomes[
        (outcomes["symbol"] == inst)
        & (outcomes["orb_label"] == orb_label)
        & (outcomes["entry_model"] == strat["entry_model"])
        & (outcomes["confirm_bars"] == strat["confirm_bars"])
        & (outcomes["rr_target"] == strat["rr_target"])
    ].copy()
    if strat_outcomes.empty:
        continue
    # Join outcomes with daily_features (full columns)
    merged = strat_outcomes.merge(
        eligible_days, left_on=["trading_day", "symbol"], right_on=["trading_day", "symbol"], how="inner"
    )
    if merged.empty:
        continue
    merged["direction"] = np.where(merged["entry_price"] > merged["stop_price"], "long", "short")
    merged["instrument"] = inst
    merged["session"] = orb_label
    merged["strategy_id"] = strat["strategy_id"]
    all_trades.append(merged)

df = pd.concat(all_trades, ignore_index=True)
# Deduplicate by unique trade
df_unique = df.drop_duplicates(subset=["trading_day", "instrument", "session", "direction"])

print(f"Validated filtered trades: {len(df_unique)}")
print(f"Win rate: {(df_unique['pnl_r'] > 0).mean():.1%}")
print(f"Mean pnl_r: {df_unique['pnl_r'].mean():+.4f}")
print(f"Median pnl_r: {df_unique['pnl_r'].median():+.4f}")
print()

# =====================================================================
# PART 1: What features correlate with LOSING trades?
# Test continuous features via quintile analysis
# =====================================================================
print("=" * 90)
print("PART 1: CONTINUOUS FEATURES -- Quintile analysis")
print("Which features have the worst Q1 or Q5? (monotonic = real signal)")
print("=" * 90)
print()

# Features to test (trade-time-knowable, no look-ahead)
continuous_features = []
for col in df_unique.columns:
    if (col.startswith("prev_day_") and col not in ["prev_day_direction"]) or col in [
        "atr_20",
        "atr_20_pct",
        "gap_open_points",
        "rsi_14_at_" + (df_unique["session"].iloc[0] if len(df_unique) > 0 else ""),
    ]:
        continuous_features.append(col)

# Also check: atr_vel_ratio, day-of-week effects
for col in ["atr_vel_ratio", "daily_volume"]:
    if col in df_unique.columns:
        continuous_features.append(col)

# Filter to columns that actually exist and have enough non-null
continuous_features = [c for c in continuous_features if c in df_unique.columns]
continuous_features = [c for c in continuous_features if df_unique[c].notna().sum() > len(df_unique) * 0.5]

print(f"Testing {len(continuous_features)} continuous features:")
print(f"  {continuous_features}")
print()

results = []

for feat in continuous_features:
    valid = df_unique[df_unique[feat].notna()].copy()
    if len(valid) < 100:
        continue
    try:
        valid["q"] = pd.qcut(valid[feat], 5, labels=False, duplicates="drop") + 1
    except ValueError:
        continue

    quintile_means = []
    for q in sorted(valid["q"].unique()):
        qdata = valid[valid["q"] == q]["pnl_r"]
        quintile_means.append((q, len(qdata), qdata.mean(), (qdata > 0).sum() / len(qdata)))

    if len(quintile_means) < 4:
        continue

    # Monotonicity: correlation between quintile rank and mean pnl_r
    ranks = [x[0] for x in quintile_means]
    means = [x[2] for x in quintile_means]
    corr, corr_p = stats.spearmanr(ranks, means)

    # Spread: Q5 - Q1 mean
    spread = quintile_means[-1][2] - quintile_means[0][2]

    # WR spread
    wr_spread = quintile_means[-1][3] - quintile_means[0][3]

    # Worst quintile
    worst_q = min(quintile_means, key=lambda x: x[2])

    results.append(
        {
            "feature": feat,
            "corr": corr,
            "corr_p": corr_p,
            "spread": spread,
            "wr_spread": wr_spread,
            "worst_q": worst_q[0],
            "worst_mean": worst_q[2],
            "worst_wr": worst_q[3],
            "worst_n": worst_q[1],
            "quintiles": quintile_means,
        }
    )

# Sort by absolute spread (largest effect first)
results.sort(key=lambda x: abs(x["spread"]), reverse=True)

for r in results:
    monotonic = "MONOTONIC" if abs(r["corr"]) > 0.8 else "PARTIAL" if abs(r["corr"]) > 0.5 else "FLAT"
    wr_signal = "WR_SIGNAL" if abs(r["wr_spread"]) > 0.03 else "WR_FLAT"
    print(
        f"{r['feature']:30s}: spread={r['spread']:+.4f} WR_spread={r['wr_spread']:+.1%} "
        f"corr={r['corr']:+.2f}(p={r['corr_p']:.3f}) {monotonic} {wr_signal}"
    )
    print(f"  Worst: Q{r['worst_q']} mean={r['worst_mean']:+.4f} WR={r['worst_wr']:.1%} N={r['worst_n']}")
    # Print all quintiles
    for q, n, m, wr in r["quintiles"]:
        marker = " <<<" if m == r["worst_mean"] else ""
        print(f"    Q{q}: N={n:5d} mean={m:+.4f} WR={wr:.1%}{marker}")
    print()

# =====================================================================
# PART 2: Day-of-week effect
# =====================================================================
print("=" * 90)
print("PART 2: DAY OF WEEK")
print("=" * 90)
print()

df_unique["dow"] = pd.to_datetime(df_unique["trading_day"]).dt.dayofweek
dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

for dow in sorted(df_unique["dow"].unique()):
    dow_data = df_unique[df_unique["dow"] == dow]["pnl_r"]
    rest = df_unique[df_unique["dow"] != dow]["pnl_r"]
    if len(dow_data) > 20:
        t, p = stats.ttest_ind(dow_data, rest)
        print(
            f"  {dow_names.get(dow, dow):3s}: N={len(dow_data):5d} mean={dow_data.mean():+.4f} "
            f"WR={(dow_data > 0).sum() / len(dow_data):.1%} | vs rest p={p:.4f}"
        )

# =====================================================================
# PART 3: Gap type
# =====================================================================
print()
print("=" * 90)
print("PART 3: GAP TYPE")
print("=" * 90)
print()

for gt in sorted(df_unique["gap_type"].dropna().unique()):
    gt_data = df_unique[df_unique["gap_type"] == gt]["pnl_r"]
    rest = df_unique[df_unique["gap_type"] != gt]["pnl_r"]
    if len(gt_data) > 20:
        t, p = stats.ttest_ind(gt_data, rest)
        print(
            f"  {gt:12s}: N={len(gt_data):5d} mean={gt_data.mean():+.4f} "
            f"WR={(gt_data > 0).sum() / len(gt_data):.1%} | vs rest p={p:.4f}"
        )

# =====================================================================
# PART 4: Direction (long vs short) conditional on prev_day_direction
# =====================================================================
print()
print("=" * 90)
print("PART 4: DIRECTION x PREV_DAY_DIRECTION")
print("=" * 90)
print()

for prev_dir in ["bear", "bull"]:
    for trade_dir in ["long", "short"]:
        subset = df_unique[(df_unique["prev_day_direction"] == prev_dir) & (df_unique["direction"] == trade_dir)][
            "pnl_r"
        ]
        rest = df_unique[~((df_unique["prev_day_direction"] == prev_dir) & (df_unique["direction"] == trade_dir))][
            "pnl_r"
        ]
        if len(subset) > 20:
            t, p = stats.ttest_ind(subset, rest)
            print(
                f"  prev={prev_dir:4s} trade={trade_dir:5s}: N={len(subset):5d} "
                f"mean={subset.mean():+.4f} WR={(subset > 0).sum() / len(subset):.1%} "
                f"| vs rest p={p:.4f}"
            )

# =====================================================================
# PART 5: Volatility regime (atr_vel_ratio if available)
# =====================================================================
if "atr_vel_ratio" in df_unique.columns:
    print()
    print("=" * 90)
    print("PART 5: VOLATILITY REGIME (atr_vel_ratio)")
    print("=" * 90)
    print()
    valid = df_unique[df_unique["atr_vel_ratio"].notna()].copy()
    if len(valid) > 100:
        valid["vol_regime"] = pd.qcut(
            valid["atr_vel_ratio"], 3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop"
        )
        for regime in ["low_vol", "mid_vol", "high_vol"]:
            rdata = valid[valid["vol_regime"] == regime]["pnl_r"]
            rest = valid[valid["vol_regime"] != regime]["pnl_r"]
            if len(rdata) > 20:
                t, p = stats.ttest_ind(rdata, rest)
                print(
                    f"  {regime:8s}: N={len(rdata):5d} mean={rdata.mean():+.4f} "
                    f"WR={(rdata > 0).sum() / len(rdata):.1%} | vs rest p={p:.4f}"
                )

con.close()
print()
print("=" * 90)
print("DONE -- what_kills_edge complete")
print("=" * 90)
