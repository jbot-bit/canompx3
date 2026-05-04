"""Full audit: Bull-day short avoidance + prev_day_range signal.

Two findings to validate:
1. Shorts after bull days underperform shorts after bear days (p=0.002)
2. Larger prev_day_range = better outcomes (WR 53.4% -> 57.8%)

Runs T1 (WR monotonicity), per-session, per-instrument, year-by-year,
permutation test for both signals.

@research-source: bull_short_avoidance_audit.py
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
    "confirm_bars, rr_target, entry_price, stop_price, pnl_r "
    "FROM orb_outcomes "
    "WHERE orb_minutes = 5 AND symbol IN ('MGC', 'MNQ', 'MES') AND pnl_r IS NOT NULL"
).fetchdf()

# Build filtered trade set
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
    filtered = strat_outcomes.merge(
        eligible_days, left_on=["trading_day", "symbol"], right_on=["trading_day", "symbol"], how="inner"
    )
    if filtered.empty:
        continue
    filtered["direction"] = np.where(filtered["entry_price"] > filtered["stop_price"], "long", "short")
    filtered["instrument"] = inst
    filtered["session"] = orb_label
    all_trades.append(filtered)

df = pd.concat(all_trades, ignore_index=True)
df_unique = df.drop_duplicates(subset=["trading_day", "instrument", "session", "direction"])

shorts = df_unique[df_unique["direction"] == "short"].copy()
print(f"Total validated filtered shorts: {len(shorts)}")
print()

# =====================================================================
# SIGNAL 1: BULL-DAY SHORT AVOIDANCE
# =====================================================================
print("#" * 90)
print("# SIGNAL 1: BULL-DAY SHORT AVOIDANCE")
print("#" * 90)
print()

bull_shorts = shorts[shorts["prev_day_direction"] == "bull"]["pnl_r"]
bear_shorts = shorts[shorts["prev_day_direction"] == "bear"]["pnl_r"]

t, p = stats.ttest_ind(bear_shorts, bull_shorts)
print(
    f"Bear-day shorts: N={len(bear_shorts):5d} mean={bear_shorts.mean():+.4f} WR={(bear_shorts > 0).sum() / len(bear_shorts):.1%}"
)
print(
    f"Bull-day shorts: N={len(bull_shorts):5d} mean={bull_shorts.mean():+.4f} WR={(bull_shorts > 0).sum() / len(bull_shorts):.1%}"
)
print(f"Delta: {bear_shorts.mean() - bull_shorts.mean():+.4f}, t={t:.3f}, p={p:.6f}")
print()

# --- T1: WR MONOTONICITY (is this WR or payoff?) ---
print("--- T1: WR MONOTONICITY ---")
print(f"Bear short WR: {(bear_shorts > 0).sum() / len(bear_shorts):.3f}")
print(f"Bull short WR: {(bull_shorts > 0).sum() / len(bull_shorts):.3f}")
print(f"WR delta: {(bear_shorts > 0).sum() / len(bear_shorts) - (bull_shorts > 0).sum() / len(bull_shorts):.3f}")
# Also check avg win and avg loss
bear_wins = bear_shorts[bear_shorts > 0]
bear_losses = bear_shorts[bear_shorts <= 0]
bull_wins = bull_shorts[bull_shorts > 0]
bull_losses = bull_shorts[bull_shorts <= 0]
print(f"Bear short: avg_win={bear_wins.mean():+.3f} avg_loss={bear_losses.mean():+.3f}")
print(f"Bull short: avg_win={bull_wins.mean():+.3f} avg_loss={bull_losses.mean():+.3f}")
wr_diff = (bear_shorts > 0).sum() / len(bear_shorts) - (bull_shorts > 0).sum() / len(bull_shorts)
if abs(wr_diff) > 0.03:
    print("VERDICT: WR_SIGNAL (3%+ WR spread)")
else:
    print("VERDICT: ARITHMETIC_ONLY (WR flat, payoff drives)")
print()

# --- Per session ---
print("--- PER SESSION ---")
session_p_values = []
for sess in sorted(shorts["session"].unique()):
    sess_shorts = shorts[shorts["session"] == sess]
    bear_s = sess_shorts[sess_shorts["prev_day_direction"] == "bear"]["pnl_r"]
    bull_s = sess_shorts[sess_shorts["prev_day_direction"] == "bull"]["pnl_r"]
    if len(bear_s) > 15 and len(bull_s) > 15:
        t_s, p_s = stats.ttest_ind(bear_s, bull_s)
        delta_s = bear_s.mean() - bull_s.mean()
        direction = "BEAR>BULL" if delta_s > 0 else "BULL>BEAR"
        session_p_values.append((sess, p_s, delta_s, len(bear_s), len(bull_s)))
        print(
            f"  {sess:20s}: bear N={len(bear_s):4d} mean={bear_s.mean():+.4f} | "
            f"bull N={len(bull_s):4d} mean={bull_s.mean():+.4f} | "
            f"delta={delta_s:+.4f} p={p_s:.4f} {direction}"
        )

print(
    f"\n  Sessions where bear > bull: {sum(1 for _, _, d, _, _ in session_p_values if d > 0)}/{len(session_p_values)}"
)
print()

# --- Per instrument ---
print("--- PER INSTRUMENT ---")
for inst in sorted(shorts["instrument"].unique()):
    inst_shorts = shorts[shorts["instrument"] == inst]
    bear_i = inst_shorts[inst_shorts["prev_day_direction"] == "bear"]["pnl_r"]
    bull_i = inst_shorts[inst_shorts["prev_day_direction"] == "bull"]["pnl_r"]
    if len(bear_i) > 15 and len(bull_i) > 15:
        t_i, p_i = stats.ttest_ind(bear_i, bull_i)
        print(
            f"  {inst}: bear N={len(bear_i)} mean={bear_i.mean():+.4f} | "
            f"bull N={len(bull_i)} mean={bull_i.mean():+.4f} | "
            f"delta={bear_i.mean() - bull_i.mean():+.4f} p={p_i:.4f}"
        )
print()

# --- Year-by-year ---
print("--- YEAR-BY-YEAR ---")
shorts_copy = shorts.copy()
shorts_copy["year"] = pd.to_datetime(shorts_copy["trading_day"]).dt.year
pos_years = 0
neg_years = 0
for year in sorted(shorts_copy["year"].unique()):
    yr = shorts_copy[shorts_copy["year"] == year]
    bear_y = yr[yr["prev_day_direction"] == "bear"]["pnl_r"]
    bull_y = yr[yr["prev_day_direction"] == "bull"]["pnl_r"]
    if len(bear_y) >= 10 and len(bull_y) >= 10:
        delta = bear_y.mean() - bull_y.mean()
        if delta > 0:
            pos_years += 1
        else:
            neg_years += 1
        print(
            f"  {year}: bear N={len(bear_y):3d} mean={bear_y.mean():+.4f} | "
            f"bull N={len(bull_y):3d} mean={bull_y.mean():+.4f} | delta={delta:+.4f}"
        )
print(f"\n  Bear > Bull years: {pos_years}/{pos_years + neg_years}")
print()

# --- Permutation test ---
print("--- PERMUTATION TEST (1000 shuffles) ---")
rng = np.random.default_rng(42)
observed = bear_shorts.mean() - bull_shorts.mean()
n_bear = len(bear_shorts)
combined = shorts["pnl_r"].values
perm_deltas = np.zeros(1000)
for i in range(1000):
    shuffled = rng.permutation(combined)
    perm_deltas[i] = shuffled[:n_bear].mean() - shuffled[n_bear:].mean()
perm_p = (np.abs(perm_deltas) >= np.abs(observed)).mean()
print(f"  Observed delta: {observed:+.4f}")
print(f"  Permutation p: {perm_p:.4f}")
print(f"  Rank: {(np.abs(perm_deltas) >= np.abs(observed)).sum()}/1000")
print()

# --- Dollar impact estimate ---
print("--- DOLLAR IMPACT (if you skip bull-day shorts) ---")
# What you give up vs what you save
bull_short_count = len(bull_shorts)
bear_short_count = len(bear_shorts)
total_short_count = bull_short_count + bear_short_count
bull_total_r = bull_shorts.sum()
bear_total_r = bear_shorts.sum()
print(f"  Bull-day shorts: {bull_short_count} trades, total R = {bull_total_r:+.1f}")
print(f"  Bear-day shorts: {bear_short_count} trades, total R = {bear_total_r:+.1f}")
print("  If you SKIP all bull-day shorts:")
print(f"    You lose {bull_total_r:+.1f}R from {bull_short_count} trades ({bull_shorts.mean():+.4f}/trade)")
print(f"    You keep {bear_total_r:+.1f}R from {bear_short_count} trades ({bear_shorts.mean():+.4f}/trade)")
print(f"    Net change: {-bull_total_r:+.1f}R (LOSING profitable trades)")
print("  SKIP is WRONG -- both groups are profitable!")
print("  Better approach: REDUCE SIZE on bull-day shorts (e.g., half contracts)")
half_size_improvement = bull_total_r * 0.5  # give up half the bull-day short R
print(f"    Half-size on bull shorts saves: {-half_size_improvement:+.1f}R of exposure")
print(
    f"    While keeping: {bull_total_r * 0.5:+.1f}R from reduced bull shorts + {bear_total_r:+.1f}R from full bear shorts"
)
print()

# =====================================================================
# SIGNAL 2: PREV_DAY_RANGE (bigger range = better)
# =====================================================================
print()
print("#" * 90)
print("# SIGNAL 2: PREV_DAY_RANGE (bigger prior range = better outcomes)")
print("#" * 90)
print()

valid = df_unique[df_unique["prev_day_range"].notna()].copy()
valid["pdr_quintile"] = pd.qcut(valid["prev_day_range"], 5, labels=False, duplicates="drop") + 1

# Full quintile table
print("--- QUINTILE TABLE ---")
for q in sorted(valid["pdr_quintile"].unique()):
    qdata = valid[valid["pdr_quintile"] == q]["pnl_r"]
    print(f"  Q{q}: N={len(qdata):5d} mean={qdata.mean():+.4f} WR={(qdata > 0).sum() / len(qdata):.1%}")

# Spearman correlation
ranks = []
means = []
for q in sorted(valid["pdr_quintile"].unique()):
    ranks.append(q)
    means.append(valid[valid["pdr_quintile"] == q]["pnl_r"].mean())
corr, corr_p = stats.spearmanr(ranks, means)
print(f"\n  Spearman corr = {corr:+.2f}, p = {corr_p:.4f}")
print()

# Q5 vs Q1 t-test
q1 = valid[valid["pdr_quintile"] == 1]["pnl_r"]
q5 = valid[valid["pdr_quintile"] == max(valid["pdr_quintile"])]["pnl_r"]
t_q, p_q = stats.ttest_ind(q5, q1)
print("--- Q5 vs Q1 ---")
print(f"  Q1: N={len(q1):5d} mean={q1.mean():+.4f} WR={(q1 > 0).sum() / len(q1):.1%}")
print(f"  Q5: N={len(q5):5d} mean={q5.mean():+.4f} WR={(q5 > 0).sum() / len(q5):.1%}")
print(f"  t={t_q:.3f}, p={p_q:.6f}")
print()

# Per instrument
print("--- PER INSTRUMENT ---")
for inst in sorted(valid["instrument"].unique()):
    inst_data = valid[valid["instrument"] == inst]
    q1_i = inst_data[inst_data["pdr_quintile"] == 1]["pnl_r"]
    q5_i = inst_data[inst_data["pdr_quintile"] == max(inst_data["pdr_quintile"])]["pnl_r"]
    if len(q1_i) > 10 and len(q5_i) > 10:
        t_i, p_i = stats.ttest_ind(q5_i, q1_i)
        print(
            f"  {inst}: Q1 mean={q1_i.mean():+.4f} WR={(q1_i > 0).sum() / len(q1_i):.1%} | "
            f"Q5 mean={q5_i.mean():+.4f} WR={(q5_i > 0).sum() / len(q5_i):.1%} | p={p_i:.4f}"
        )
print()

# Year-by-year (Q5-Q1 delta)
print("--- YEAR-BY-YEAR (Q5-Q1 delta) ---")
valid_yr = valid.copy()
valid_yr["year"] = pd.to_datetime(valid_yr["trading_day"]).dt.year
pos_yr = 0
neg_yr = 0
for year in sorted(valid_yr["year"].unique()):
    yr = valid_yr[valid_yr["year"] == year]
    q1_y = yr[yr["pdr_quintile"] == 1]["pnl_r"]
    q5_y = yr[yr["pdr_quintile"] == max(yr["pdr_quintile"])]["pnl_r"]
    if len(q1_y) >= 10 and len(q5_y) >= 10:
        delta = q5_y.mean() - q1_y.mean()
        if delta > 0:
            pos_yr += 1
        else:
            neg_yr += 1
        print(f"  {year}: Q1 mean={q1_y.mean():+.4f} | Q5 mean={q5_y.mean():+.4f} | delta={delta:+.4f}")
print(f"\n  Q5 > Q1 years: {pos_yr}/{pos_yr + neg_yr}")

# Redundancy with existing ATR/vol filters
print()
print("--- REDUNDANCY CHECK: prev_day_range vs atr_20 ---")
if "atr_20" in valid.columns:
    corr_atr = valid[["prev_day_range", "atr_20"]].dropna().corr().iloc[0, 1]
    print(f"  Correlation(prev_day_range, atr_20) = {corr_atr:.3f}")
    if abs(corr_atr) > 0.7:
        print("  HIGH REDUNDANCY -- likely already captured by ATR-based filters")
    else:
        print("  LOW-MODERATE redundancy -- may add independent information")

con.close()
print()
print("=" * 90)
print("DONE")
print("=" * 90)
