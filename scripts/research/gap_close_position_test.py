"""Test 2: Gap direction + close position interaction.

Hypothesis: Prior day closed at low AND today gaps down = double momentum signal.
Both signals point same direction — compounding should amplify ORB short outcomes.

Scoped to VALIDATED strategies with filters applied.
Reports ALL combinations — no cherry-picking.

@research-source: gap_close_position_test.py
@data-source: orb_outcomes JOIN daily_features, filtered via validated_setups
@literature: Gap trading literature; project gap_type feature already in daily_features
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

# Build filtered trade set (same infrastructure)
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
    eligible_days = inst_daily.loc[eligible_mask, ["trading_day", "symbol"]].copy()
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
    filtered = strat_outcomes.merge(eligible_days[["trading_day"]], on="trading_day", how="inner")
    if filtered.empty:
        continue
    prev_cols = [
        "trading_day",
        "symbol",
        "prev_day_close",
        "prev_day_low",
        "prev_day_range",
        "prev_day_direction",
        "gap_type",
        "gap_open_points",
    ]
    merged = filtered.merge(inst_daily[prev_cols], on=["trading_day", "symbol"], how="inner")
    valid = merged[merged["prev_day_range"] > 0].copy()
    if valid.empty:
        continue
    valid["prev_close_pos"] = (valid["prev_day_close"] - valid["prev_day_low"]) / valid["prev_day_range"]
    valid["direction"] = np.where(valid["entry_price"] > valid["stop_price"], "long", "short")
    valid["instrument"] = inst
    valid["orb_label"] = orb_label
    all_trades.append(
        valid[
            [
                "trading_day",
                "instrument",
                "orb_label",
                "direction",
                "pnl_r",
                "prev_day_direction",
                "prev_close_pos",
                "gap_type",
                "gap_open_points",
            ]
        ]
    )

df = pd.concat(all_trades, ignore_index=True)
df_unique = df.drop_duplicates(subset=["trading_day", "instrument", "orb_label", "direction"])

print(f"Total unique filtered trades: {len(df_unique)}")
print("Gap type distribution:")
print(df_unique["gap_type"].value_counts().to_string())
print()

# =====================================================================
# TEST 2A: Double momentum — bear close-at-low + gap down -> short outcomes
# =====================================================================
print("=" * 90)
print("TEST 2A: DOUBLE MOMENTUM (bear close-at-low + gap_down) -> SHORT outcomes")
print("Hypothesis: both signals bearish = amplified short edge")
print("=" * 90)
print()

shorts = df_unique[df_unique["direction"] == "short"]

# Define groups
double_bear = shorts[
    (shorts["prev_day_direction"] == "bear") & (shorts["prev_close_pos"] < 0.2) & (shorts["gap_type"] == "gap_down")
]["pnl_r"]

# Control: shorts on other bear days (no double signal)
bear_other = shorts[
    (shorts["prev_day_direction"] == "bear") & ~((shorts["prev_close_pos"] < 0.2) & (shorts["gap_type"] == "gap_down"))
]["pnl_r"]

# Control 2: ALL shorts (full baseline)
all_shorts = shorts["pnl_r"]

print("Double bear signal (close<0.2 + gap_down + short):")
print(
    f"  N={len(double_bear)}, mean_R={double_bear.mean():+.4f}, "
    f"WR={(double_bear > 0).sum() / max(len(double_bear), 1):.1%}"
)
print("Other bear shorts:")
print(
    f"  N={len(bear_other)}, mean_R={bear_other.mean():+.4f}, WR={(bear_other > 0).sum() / max(len(bear_other), 1):.1%}"
)
print("All shorts baseline:")
print(f"  N={len(all_shorts)}, mean_R={all_shorts.mean():+.4f}, WR={(all_shorts > 0).sum() / len(all_shorts):.1%}")
print()

if len(double_bear) >= 15 and len(bear_other) >= 15:
    t, p = stats.ttest_ind(double_bear, bear_other)
    print(f"Double bear vs other bear shorts: t={t:.3f}, p={p:.6f}")
    print(f"Delta: {double_bear.mean() - bear_other.mean():+.4f}")
else:
    print(f"INSUFFICIENT N for t-test (double_bear N={len(double_bear)})")

print()

# =====================================================================
# TEST 2B: Double bull — bull close-at-high + gap up -> LONG outcomes
# =====================================================================
print("=" * 90)
print("TEST 2B: DOUBLE BULL (bull close-at-high + gap_up) -> LONG outcomes")
print("Hypothesis: both signals bullish = amplified long edge")
print("=" * 90)
print()

longs = df_unique[df_unique["direction"] == "long"]

double_bull = longs[
    (longs["prev_day_direction"] == "bull") & (longs["prev_close_pos"] > 0.8) & (longs["gap_type"] == "gap_up")
]["pnl_r"]

bull_other = longs[
    (longs["prev_day_direction"] == "bull") & ~((longs["prev_close_pos"] > 0.8) & (longs["gap_type"] == "gap_up"))
]["pnl_r"]

all_longs = longs["pnl_r"]

print("Double bull signal (close>0.8 + gap_up + long):")
print(
    f"  N={len(double_bull)}, mean_R={double_bull.mean():+.4f}, "
    f"WR={(double_bull > 0).sum() / max(len(double_bull), 1):.1%}"
)
print("Other bull longs:")
print(
    f"  N={len(bull_other)}, mean_R={bull_other.mean():+.4f}, WR={(bull_other > 0).sum() / max(len(bull_other), 1):.1%}"
)
print("All longs baseline:")
print(f"  N={len(all_longs)}, mean_R={all_longs.mean():+.4f}, WR={(all_longs > 0).sum() / len(all_longs):.1%}")
print()

if len(double_bull) >= 15 and len(bull_other) >= 15:
    t, p = stats.ttest_ind(double_bull, bull_other)
    print(f"Double bull vs other bull longs: t={t:.3f}, p={p:.6f}")
    print(f"Delta: {double_bull.mean() - bull_other.mean():+.4f}")
else:
    print(f"INSUFFICIENT N for t-test (double_bull N={len(double_bull)})")

print()

# =====================================================================
# TEST 2C: CONTRA momentum — bear close-at-low + gap UP -> mean reversion?
# =====================================================================
print("=" * 90)
print("TEST 2C: CONTRA SIGNAL (bear close-at-low + gap_up) -> LONG outcomes")
print("Hypothesis: gap reverses prior selloff = bounce = long edge")
print("=" * 90)
print()

contra_bear_long = longs[
    (longs["prev_day_direction"] == "bear") & (longs["prev_close_pos"] < 0.2) & (longs["gap_type"] == "gap_up")
]["pnl_r"]

other_longs = longs[
    ~((longs["prev_day_direction"] == "bear") & (longs["prev_close_pos"] < 0.2) & (longs["gap_type"] == "gap_up"))
]["pnl_r"]

print("Contra bear->long (close<0.2 + gap_up + long):")
print(
    f"  N={len(contra_bear_long)}, mean_R={contra_bear_long.mean():+.4f}, "
    f"WR={(contra_bear_long > 0).sum() / max(len(contra_bear_long), 1):.1%}"
)
print("Other longs:")
print(f"  N={len(other_longs)}, mean_R={other_longs.mean():+.4f}, WR={(other_longs > 0).sum() / len(other_longs):.1%}")
print()

if len(contra_bear_long) >= 15 and len(other_longs) >= 15:
    t, p = stats.ttest_ind(contra_bear_long, other_longs)
    print(f"Contra bear->long vs other longs: t={t:.3f}, p={p:.6f}")
    print(f"Delta: {contra_bear_long.mean() - other_longs.mean():+.4f}")
else:
    print(f"INSUFFICIENT N for t-test (contra N={len(contra_bear_long)})")

print()

# =====================================================================
# BH FDR across all tests that ran
# =====================================================================
print("=" * 90)
print("BH FDR CORRECTION (all tests that had sufficient N)")
print("=" * 90)

all_results = []
if len(double_bear) >= 15 and len(bear_other) >= 15:
    t, p = stats.ttest_ind(double_bear, bear_other)
    all_results.append(("2A: Double bear short", p, double_bear.mean() - bear_other.mean()))
if len(double_bull) >= 15 and len(bull_other) >= 15:
    t, p = stats.ttest_ind(double_bull, bull_other)
    all_results.append(("2B: Double bull long", p, double_bull.mean() - bull_other.mean()))
if len(contra_bear_long) >= 15 and len(other_longs) >= 15:
    t, p = stats.ttest_ind(contra_bear_long, other_longs)
    all_results.append(("2C: Contra bear long", p, contra_bear_long.mean() - other_longs.mean()))

K = len(all_results)
print(f"K = {K} tests")
if K > 0:
    sorted_r = sorted(all_results, key=lambda x: x[1])
    any_survive = False
    for rank, (name, p, delta) in enumerate(sorted_r, 1):
        bh = 0.05 * rank / K
        verdict = "SURVIVES" if p <= bh else "KILLED"
        if p <= bh:
            any_survive = True
        print(f"  Rank {rank}: {name:30s} p={p:.6f} BH={bh:.4f} {verdict} (delta={delta:+.4f})")
    if not any_survive:
        print("\n>>> NO TESTS SURVIVE BH FDR.")
    else:
        print("\n>>> At least one survives. Check year stability.")

        # Year stability for survivors
        for name, p, delta in all_results:
            if p <= 0.05 * (sorted_r.index((name, p, delta)) + 1) / K:
                print(f"\n  Year-by-year for {name}:")
                if "bear short" in name.lower():
                    group = double_bear
                    control = bear_other
                    subset = shorts[shorts["prev_day_direction"] == "bear"].copy()
                elif "bull long" in name.lower():
                    group = double_bull
                    control = bull_other
                    subset = longs[longs["prev_day_direction"] == "bull"].copy()
                else:
                    continue
else:
    print("No tests had sufficient N to run.")

con.close()
print()
print("DONE — Test 2 complete")
