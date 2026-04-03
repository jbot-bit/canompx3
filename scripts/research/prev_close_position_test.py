"""Research: Prior-day close position effect on ORB breakout outcomes.

SCOPED TO VALIDATED STRATEGIES ONLY with filters properly applied.

Hypothesis: ORB breakout outcomes differ on days where the prior day closed
near its extreme (exhaustion signal), AFTER conditioning on direction.

Tests A-D are promotion-eligible (K=6 for BH FDR).
Quintile scan and permutation test are sanity checks only.

@research-source: prev_close_position_test.py
@data-source: orb_outcomes JOIN daily_features, filtered via validated_setups
@look-ahead-safe: prev_day_* uses rows[i-1] in build_daily_features.py:1271-1281
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

# Step 1: Load validated strategies
strats = con.execute("""
    SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
           confirm_bars, rr_target, filter_type, stop_multiplier
    FROM validated_setups
    WHERE status = 'active' AND orb_minutes = 5
""").fetchdf()
print(f"Validated strategies: {len(strats)}")

# Step 2: Load daily_features (all instruments, O5 only)
daily = con.execute("""
    SELECT *
    FROM daily_features
    WHERE symbol IN ('MGC', 'MNQ', 'MES')
      AND orb_minutes = 5
""").fetchdf()
print(f"Daily features rows: {len(daily)}")

# Step 3: Load orb_outcomes (O5 only, active instruments)
outcomes = con.execute("""
    SELECT trading_day, symbol, orb_label, orb_minutes, entry_model,
           confirm_bars, rr_target, entry_price, stop_price, pnl_r
    FROM orb_outcomes
    WHERE orb_minutes = 5
      AND symbol IN ('MGC', 'MNQ', 'MES')
      AND pnl_r IS NOT NULL
""").fetchdf()
print(f"Orb outcomes rows (O5, all): {len(outcomes)}")

# Step 4: For each validated strategy, get filtered trade outcomes
all_trades = []
skipped_filters = set()

for _, strat in strats.iterrows():
    inst = strat["instrument"]
    orb_label = strat["orb_label"]
    entry_model = strat["entry_model"]
    confirm_bars = strat["confirm_bars"]
    rr_target = strat["rr_target"]
    filter_type = strat["filter_type"]

    # Get the filter object
    filt = ALL_FILTERS.get(filter_type)
    if filt is None:
        skipped_filters.add(filter_type)
        continue

    # Filter daily_features for this instrument
    inst_daily = daily[daily["symbol"] == inst].copy()
    if inst_daily.empty:
        continue

    # Apply filter
    eligible_mask = filt.matches_df(inst_daily, orb_label)
    eligible_days = inst_daily.loc[eligible_mask, ["trading_day", "symbol"]].copy()

    if eligible_days.empty:
        continue

    # Get outcomes matching this strategy's dimensions
    strat_outcomes = outcomes[
        (outcomes["symbol"] == inst)
        & (outcomes["orb_label"] == orb_label)
        & (outcomes["entry_model"] == entry_model)
        & (outcomes["confirm_bars"] == confirm_bars)
        & (outcomes["rr_target"] == rr_target)
    ].copy()

    if strat_outcomes.empty:
        continue

    # Inner join: only trades on eligible days
    filtered = strat_outcomes.merge(eligible_days[["trading_day"]], on="trading_day", how="inner")

    if filtered.empty:
        continue

    # Attach prev_day info from daily_features
    prev_cols = ["trading_day", "symbol", "prev_day_close", "prev_day_low", "prev_day_range", "prev_day_direction"]
    merged = filtered.merge(inst_daily[prev_cols], on=["trading_day", "symbol"], how="inner")

    # Compute prev_close_position
    valid = merged[merged["prev_day_range"] > 0].copy()
    if valid.empty:
        continue

    valid["prev_close_pos"] = (valid["prev_day_close"] - valid["prev_day_low"]) / valid["prev_day_range"]
    valid["direction"] = np.where(valid["entry_price"] > valid["stop_price"], "long", "short")
    valid["strategy_id"] = strat["strategy_id"]
    valid["instrument"] = inst

    all_trades.append(
        valid[
            [
                "trading_day",
                "instrument",
                "orb_label",
                "strategy_id",
                "direction",
                "pnl_r",
                "prev_day_direction",
                "prev_close_pos",
            ]
        ]
    )

if skipped_filters:
    print(f"WARNING: Skipped filters not in ALL_FILTERS: {skipped_filters}")

df = pd.concat(all_trades, ignore_index=True)
# Deduplicate: same trade_day + instrument + orb_label + direction can appear
# in multiple strategies. Keep unique trades only for the research test.
df_unique = df.drop_duplicates(subset=["trading_day", "instrument", "orb_label", "direction"])

print(f"\nTotal filtered trades (all strategies): {len(df)}")
print(f"Unique trades (deduplicated): {len(df_unique)}")
print(f"Strategies contributing: {df['strategy_id'].nunique()}")
print(f"Instruments: {sorted(df_unique['instrument'].unique())}")
print(
    f"Bear days: {(df_unique['prev_day_direction'] == 'bear').sum()}, "
    f"Bull days: {(df_unique['prev_day_direction'] == 'bull').sum()}"
)
print()

# Use df_unique for tests (avoid counting same trade multiple times)
bear = df_unique[df_unique["prev_day_direction"] == "bear"]
bull = df_unique[df_unique["prev_day_direction"] == "bull"]

all_p = []


def report_comparison(label, group1, group2, label1="Extreme", label2="Rest"):
    """Run t-test and print results."""
    if len(group1) < 10 or len(group2) < 10:
        print(f"  SKIPPED — insufficient N ({len(group1)}, {len(group2)})")
        return None
    t, p = stats.ttest_ind(group1, group2)
    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    wr1 = (group1 > 0).sum() / n1
    wr2 = (group2 > 0).sum() / n2
    print(f"  {label1:12s}: N={n1:5d}  mean_R={m1:+.5f}  WR={wr1:.1%}  std={group1.std():.3f}")
    print(f"  {label2:12s}: N={n2:5d}  mean_R={m2:+.5f}  WR={wr2:.1%}  std={group2.std():.3f}")
    print(f"  Delta mean_R: {m1 - m2:+.5f}")
    print(f"  t={t:.3f}, p={p:.6f}")
    return (label, p)


# =====================================================================
# TEST A: Bear days — extreme low close vs rest
# =====================================================================
print("=" * 80)
print("TEST A: BEAR DAYS — extreme low close (<0.2) vs rest (>=0.2)")
print("=" * 80)
bear_extreme = bear[bear["prev_close_pos"] < 0.2]["pnl_r"]
bear_rest = bear[bear["prev_close_pos"] >= 0.2]["pnl_r"]
result = report_comparison("A: Bear ext vs rest", bear_extreme, bear_rest)
if result:
    all_p.append(result)

print("  Per instrument:")
for inst in sorted(df_unique["instrument"].unique()):
    sub_ext = bear[(bear["prev_close_pos"] < 0.2) & (bear["instrument"] == inst)]["pnl_r"]
    sub_rest = bear[(bear["prev_close_pos"] >= 0.2) & (bear["instrument"] == inst)]["pnl_r"]
    if len(sub_ext) > 10 and len(sub_rest) > 10:
        t_i, p_i = stats.ttest_ind(sub_ext, sub_rest)
        print(
            f"    {inst}: ext N={len(sub_ext)}, mean={sub_ext.mean():+.5f} | "
            f"rest N={len(sub_rest)}, mean={sub_rest.mean():+.5f} | p={p_i:.4f}"
        )
print()

# =====================================================================
# TEST B: Bear days — directional split (long vs short separately)
# =====================================================================
print("=" * 80)
print("TEST B: BEAR DAYS — extreme low, LONG vs SHORT outcomes separately")
print("=" * 80)
for direction in ["long", "short"]:
    bear_dir = bear[bear["direction"] == direction]
    ext = bear_dir[bear_dir["prev_close_pos"] < 0.2]["pnl_r"]
    rest = bear_dir[bear_dir["prev_close_pos"] >= 0.2]["pnl_r"]
    result = report_comparison(f"B: Bear ext {direction}", ext, rest, f"Ext {direction}", f"Rest {direction}")
    if result:
        all_p.append(result)
    print()

# =====================================================================
# TEST C: Bull days — extreme high close vs rest
# =====================================================================
print("=" * 80)
print("TEST C: BULL DAYS — extreme high close (>0.8) vs rest (<=0.8)")
print("=" * 80)
bull_extreme = bull[bull["prev_close_pos"] > 0.8]["pnl_r"]
bull_rest = bull[bull["prev_close_pos"] <= 0.8]["pnl_r"]
result = report_comparison("C: Bull ext vs rest", bull_extreme, bull_rest)
if result:
    all_p.append(result)

print("  Per instrument:")
for inst in sorted(df_unique["instrument"].unique()):
    sub_ext = bull[(bull["prev_close_pos"] > 0.8) & (bull["instrument"] == inst)]["pnl_r"]
    sub_rest = bull[(bull["prev_close_pos"] <= 0.8) & (bull["instrument"] == inst)]["pnl_r"]
    if len(sub_ext) > 10 and len(sub_rest) > 10:
        t_i, p_i = stats.ttest_ind(sub_ext, sub_rest)
        print(
            f"    {inst}: ext N={len(sub_ext)}, mean={sub_ext.mean():+.5f} | "
            f"rest N={len(sub_rest)}, mean={sub_rest.mean():+.5f} | p={p_i:.4f}"
        )
print()

# =====================================================================
# TEST D: Bull days — directional split
# =====================================================================
print("=" * 80)
print("TEST D: BULL DAYS — extreme high, LONG vs SHORT outcomes separately")
print("=" * 80)
for direction in ["long", "short"]:
    bull_dir = bull[bull["direction"] == direction]
    ext = bull_dir[bull_dir["prev_close_pos"] > 0.8]["pnl_r"]
    rest = bull_dir[bull_dir["prev_close_pos"] <= 0.8]["pnl_r"]
    result = report_comparison(f"D: Bull ext {direction}", ext, rest, f"Ext {direction}", f"Rest {direction}")
    if result:
        all_p.append(result)
    print()

# =====================================================================
# BH FDR CORRECTION
# =====================================================================
print("=" * 80)
print("BH FDR CORRECTION")
print("=" * 80)

K = len(all_p)
print(f"K = {K} tests")
if K == 0:
    print(">>> NO TESTS RUN (insufficient data)")
else:
    sorted_p = sorted(all_p, key=lambda x: x[1])
    any_survive = False
    for rank, (name, p) in enumerate(sorted_p, 1):
        bh_threshold = 0.05 * rank / K
        survives = "SURVIVES" if p <= bh_threshold else "KILLED"
        if p <= bh_threshold:
            any_survive = True
        print(f"  Rank {rank}: {name:30s} p={p:.6f}  BH_thresh={bh_threshold:.4f}  {survives}")

    print()
    if not any_survive:
        print(">>> NO TESTS SURVIVE BH FDR. Feature has no significant effect.")
    else:
        print(">>> At least one test survives BH FDR. Check per-instrument and year stability.")

# =====================================================================
# SANITY CHECK: Within-direction quintile means
# =====================================================================
print()
print("=" * 80)
print("SANITY CHECK: Within-direction quintile means (not promotion-eligible)")
print("=" * 80)

for dir_label, subset in [("BEAR", bear), ("BULL", bull)]:
    subset = subset.copy()
    if len(subset) < 50:
        print(f"\n  {dir_label} days: SKIPPED (N={len(subset)})")
        continue
    subset["quintile"] = pd.qcut(subset["prev_close_pos"], 5, labels=False, duplicates="drop") + 1
    print(f"\n  {dir_label} days:")
    for q in sorted(subset["quintile"].unique()):
        qdata = subset[subset["quintile"] == q]["pnl_r"]
        print(f"    Q{q}: N={len(qdata):5d}  mean_R={qdata.mean():+.5f}  WR={((qdata > 0).sum() / len(qdata)):.1%}")

# =====================================================================
# PERMUTATION TEST (1000 shuffles)
# =====================================================================
print()
print("=" * 80)
print("SANITY CHECK: Permutation test — is extreme vs rest delta real?")
print("=" * 80)

rng = np.random.default_rng(42)
N_PERMS = 1000

for label, subset, threshold, direction in [
    ("Bear extreme low", bear, 0.2, "below"),
    ("Bull extreme high", bull, 0.8, "above"),
]:
    if direction == "below":
        extreme = subset[subset["prev_close_pos"] < threshold]["pnl_r"]
        rest = subset[subset["prev_close_pos"] >= threshold]["pnl_r"]
    else:
        extreme = subset[subset["prev_close_pos"] > threshold]["pnl_r"]
        rest = subset[subset["prev_close_pos"] <= threshold]["pnl_r"]

    if len(extreme) < 10 or len(rest) < 10:
        print(f"  {label}: SKIPPED (N too small)")
        continue

    observed_delta = extreme.mean() - rest.mean()
    n_ext = len(extreme)
    combined = subset["pnl_r"].values
    perm_deltas = np.zeros(N_PERMS)

    for i in range(N_PERMS):
        shuffled = rng.permutation(combined)
        perm_deltas[i] = shuffled[:n_ext].mean() - shuffled[n_ext:].mean()

    perm_p = (np.abs(perm_deltas) >= np.abs(observed_delta)).mean()
    print(f"  {label}: observed delta = {observed_delta:+.5f}")
    print(f"  Permutation p-value ({N_PERMS} shuffles): {perm_p:.4f}")
    print(f"  Rank: {(np.abs(perm_deltas) >= np.abs(observed_delta)).sum()}/{N_PERMS}")
    print()

# =====================================================================
# YEAR-BY-YEAR STABILITY
# =====================================================================
print("=" * 80)
print("YEAR-BY-YEAR STABILITY")
print("=" * 80)

for label, subset, threshold, direction in [
    ("Bear extreme low vs rest", bear, 0.2, "below"),
    ("Bull extreme high vs rest", bull, 0.8, "above"),
]:
    print(f"\n  {label}:")
    subset_copy = subset.copy()
    subset_copy["year"] = pd.to_datetime(subset_copy["trading_day"]).dt.year

    for year in sorted(subset_copy["year"].unique()):
        yr = subset_copy[subset_copy["year"] == year]
        if direction == "below":
            ext = yr[yr["prev_close_pos"] < threshold]["pnl_r"]
            rest = yr[yr["prev_close_pos"] >= threshold]["pnl_r"]
        else:
            ext = yr[yr["prev_close_pos"] > threshold]["pnl_r"]
            rest = yr[yr["prev_close_pos"] <= threshold]["pnl_r"]
        if len(ext) > 5 and len(rest) > 5:
            delta = ext.mean() - rest.mean()
            print(
                f"    {year}: ext N={len(ext):4d} mean={ext.mean():+.5f} | "
                f"rest N={len(rest):4d} mean={rest.mean():+.5f} | delta={delta:+.5f}"
            )

con.close()
print()
print("=" * 80)
print("DONE")
print("=" * 80)
