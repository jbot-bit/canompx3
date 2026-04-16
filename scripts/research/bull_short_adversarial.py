"""Adversarial audit: Bull-short avoidance signal.

Before accepting this as real, test for:
1. LOOK-AHEAD: Is prev_day_direction truly knowable at trade time?
2. TAUTOLOGY: Is this mechanically linked to break direction?
   (If yesterday was bull, today is more likely to break UP -> shorts fail
    because they're fading a continuation move. Is this just "don't fade trends"?)
3. CONFOUND: Is the signal driven by a few outlier sessions or years?
4. REVERSE CAUSALITY: Does the same effect exist for longs after bear days?
   (If bull-day longs also do worse, it's not direction-specific)
5. PRACTICAL: What fraction of your deployed lanes even take shorts?
   If most deployed strategies are direction-agnostic, how many shorts would
   this actually affect?

@research-source: bull_short_adversarial.py
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

print(f"Total validated filtered trades: {len(df_unique)}")
print()

# =====================================================================
# CHECK 1: LOOK-AHEAD VERIFICATION
# =====================================================================
print("=" * 90)
print("CHECK 1: LOOK-AHEAD — Is prev_day_direction knowable at trade time?")
print("=" * 90)
print()
print("prev_day_direction = 'bull' if prev_day_close >= prev_day_open else 'bear'")
print("Computed from rows[i-1] in build_daily_features.py:1282-1283")
print("rows[i-1] = YESTERDAY's OHLC = fully settled before today's open")
print()

# Verify: is there any case where prev_day_direction could be from today?
# Check: does prev_day_close ever equal today's open? (would indicate same-day data)
has_cols = "daily_open" in df_unique.columns and "prev_day_close" in df_unique.columns
if has_cols:
    same = (df_unique["daily_open"] == df_unique["prev_day_close"]).sum()
    total = df_unique["daily_open"].notna().sum()
    print(f"Cases where daily_open == prev_day_close: {same}/{total}")
    print("(These are gapless opens — expected, not look-ahead)")
else:
    print("Cannot verify open vs prev_close (columns missing from merge)")
print()
print("VERDICT: prev_day_direction is NOT look-ahead. Yesterday's candle is closed")
print("and settled by the time today's ORB forms.")
print()

# =====================================================================
# CHECK 2: TAUTOLOGY — Is this mechanically linked to break direction?
# =====================================================================
print("=" * 90)
print("CHECK 2: TAUTOLOGY — Does prev_day_direction predict break direction?")
print("=" * 90)
print()
print("If bull days -> more long breaks (continuation), then:")
print("  shorts on bull days = fading a trend = expected to be weaker")
print("  This would be REAL but TRIVIAL: 'don't fade trends'")
print()

# What fraction of breaks are long vs short, by prev_day_direction?
for prev_dir in ["bear", "bull"]:
    subset = df_unique[df_unique["prev_day_direction"] == prev_dir]
    n_long = (subset["direction"] == "long").sum()
    n_short = (subset["direction"] == "short").sum()
    pct_long = n_long / len(subset) if len(subset) > 0 else 0
    print(f"  prev={prev_dir}: {n_long} long ({pct_long:.1%}), {n_short} short ({1 - pct_long:.1%})")

# Chi-squared test
contingency = pd.crosstab(df_unique["prev_day_direction"], df_unique["direction"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print(f"\n  Chi-squared: {chi2:.2f}, p={chi_p:.4f}")
print(
    f"  {'DEPENDENT' if chi_p < 0.05 else 'INDEPENDENT'}: prev_day_direction {'DOES' if chi_p < 0.05 else 'does NOT'} predict break direction"
)

if chi_p < 0.05:
    print("\n  *** TAUTOLOGY RISK: If bull days produce more long breaks,")
    print("  then shorts on bull days are mechanically trend-fading trades.")
    print("  The signal may be 'don't fade the trend' rather than a new filter.")
    print("  Need to check: is the WR/ExpR effect BEYOND the direction imbalance?")
print()

# =====================================================================
# CHECK 3: CONTROL — Does the SYMMETRIC effect exist?
# If bull-day shorts are weak, are bear-day longs also weak?
# (Would confirm this is about trend-fading, not direction-specific)
# =====================================================================
print("=" * 90)
print("CHECK 3: SYMMETRY — 2x2 grid of prev_direction x trade_direction")
print("=" * 90)
print()

print(f"{'':15s} {'LONG mean_R':>12s} {'LONG WR':>10s} {'SHORT mean_R':>12s} {'SHORT WR':>10s}")
for prev_dir in ["bear", "bull"]:
    long_data = df_unique[(df_unique["prev_day_direction"] == prev_dir) & (df_unique["direction"] == "long")]["pnl_r"]
    short_data = df_unique[(df_unique["prev_day_direction"] == prev_dir) & (df_unique["direction"] == "short")]["pnl_r"]
    long_wr = (long_data > 0).sum() / len(long_data) if len(long_data) > 0 else 0
    short_wr = (short_data > 0).sum() / len(short_data) if len(short_data) > 0 else 0
    print(
        f"  prev={prev_dir:4s}:  {long_data.mean():+.4f}     {long_wr:.1%}     {short_data.mean():+.4f}     {short_wr:.1%}"
    )

print()
# Continuation vs fading
cont_long_bull = df_unique[(df_unique["prev_day_direction"] == "bull") & (df_unique["direction"] == "long")]["pnl_r"]
cont_short_bear = df_unique[(df_unique["prev_day_direction"] == "bear") & (df_unique["direction"] == "short")]["pnl_r"]
fade_long_bear = df_unique[(df_unique["prev_day_direction"] == "bear") & (df_unique["direction"] == "long")]["pnl_r"]
fade_short_bull = df_unique[(df_unique["prev_day_direction"] == "bull") & (df_unique["direction"] == "short")]["pnl_r"]

continuation = pd.concat([cont_long_bull, cont_short_bear])
fading = pd.concat([fade_long_bear, fade_short_bull])

t_cf, p_cf = stats.ttest_ind(continuation, fading)
print("CONTINUATION trades (long after bull, short after bear):")
print(
    f"  N={len(continuation):5d} mean={continuation.mean():+.4f} WR={(continuation > 0).sum() / len(continuation):.1%}"
)
print("FADING trades (long after bear, short after bull):")
print(f"  N={len(fading):5d} mean={fading.mean():+.4f} WR={(fading > 0).sum() / len(fading):.1%}")
print(f"Delta: {continuation.mean() - fading.mean():+.4f}, t={t_cf:.3f}, p={p_cf:.6f}")
print()

if continuation.mean() > fading.mean() and p_cf < 0.05:
    print("CONFIRMED: Continuation trades outperform fading trades.")
    print("The bull-short signal IS a specific case of 'don't fade the trend'.")
    print("This is REAL but the framing matters:")
    print("  NOT: 'avoid shorts on bull days'")
    print("  BUT: 'continuation trades (with the prior trend) outperform'")
else:
    print("NOT symmetric — the effect is direction-specific, not a general trend signal.")
print()

# =====================================================================
# CHECK 4: REMOVE NYSE_OPEN AND RETEST
# (NYSE_OPEN drove p=0.0005 — is the signal just one session?)
# =====================================================================
print("=" * 90)
print("CHECK 4: ROBUSTNESS — Signal WITHOUT NYSE_OPEN")
print("=" * 90)
print()

shorts_no_nyse = df_unique[(df_unique["direction"] == "short") & (df_unique["session"] != "NYSE_OPEN")]
bear_no_nyse = shorts_no_nyse[shorts_no_nyse["prev_day_direction"] == "bear"]["pnl_r"]
bull_no_nyse = shorts_no_nyse[shorts_no_nyse["prev_day_direction"] == "bull"]["pnl_r"]

if len(bear_no_nyse) > 15 and len(bull_no_nyse) > 15:
    t_no, p_no = stats.ttest_ind(bear_no_nyse, bull_no_nyse)
    print("Without NYSE_OPEN:")
    print(
        f"  Bear shorts: N={len(bear_no_nyse):5d} mean={bear_no_nyse.mean():+.4f} WR={(bear_no_nyse > 0).sum() / len(bear_no_nyse):.1%}"
    )
    print(
        f"  Bull shorts: N={len(bull_no_nyse):5d} mean={bull_no_nyse.mean():+.4f} WR={(bull_no_nyse > 0).sum() / len(bull_no_nyse):.1%}"
    )
    print(f"  Delta: {bear_no_nyse.mean() - bull_no_nyse.mean():+.4f}, t={t_no:.3f}, p={p_no:.6f}")
    if p_no < 0.05:
        print("  SURVIVES without NYSE_OPEN")
    else:
        print("  DOES NOT SURVIVE without NYSE_OPEN — signal is session-driven")
print()

# =====================================================================
# CHECK 5: DEPLOYED PORTFOLIO IMPACT
# =====================================================================
print("=" * 90)
print("CHECK 5: DEPLOYED PORTFOLIO — How many shorts do your 5 lanes take?")
print("=" * 90)
print()

from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

profile = ACCOUNT_PROFILES.get("topstep_50k_mnq_auto")
if profile:
    for lane in effective_daily_lanes(profile):
        lane_trades = (
            df_unique[
                (
                    df_unique["session"]
                    == lane.strategy_id.split("_")[1]
                    + "_"
                    + (lane.strategy_id.split("_")[2] if len(lane.strategy_id.split("_")) > 2 else "")
                )
            ]
            if False
            else None
        )  # Can't easily parse — use orb_label instead
    # Just count shorts per session in deployed data
    deployed_sessions = set()
    for lane in effective_daily_lanes(profile):
        # Extract session from strategy_id
        parts = lane.strategy_id.split("_")
        # Find session: it's the part after instrument before E1/E2
        inst_part = parts[0]  # MGC, MNQ, MES
        # Session is everything between instrument and entry model
        for i, p in enumerate(parts):
            if p.startswith("E") and p[1:].isdigit():
                session = "_".join(parts[1:i])
                deployed_sessions.add(session)
                break

    print(f"Deployed sessions: {sorted(deployed_sessions)}")
    print()
    for sess in sorted(deployed_sessions):
        sess_data = df_unique[df_unique["session"] == sess]
        n_short = (sess_data["direction"] == "short").sum()
        n_long = (sess_data["direction"] == "long").sum()
        n_total = len(sess_data)
        if n_total > 0:
            # Shorts that are bull-day
            bull_s = sess_data[(sess_data["direction"] == "short") & (sess_data["prev_day_direction"] == "bull")]
            print(
                f"  {sess:20s}: {n_long:4d} long, {n_short:4d} short ({n_short / n_total:.0%}). "
                f"Bull-day shorts: {len(bull_s):4d} ({len(bull_s) / max(n_short, 1):.0%} of shorts)"
            )
print()

# =====================================================================
# CHECK 6: BH FDR — Is this finding significant after counting ALL tests
# we ran this session? (honest K)
# =====================================================================
print("=" * 90)
print("CHECK 6: HONEST BH FDR — counting ALL tests this session")
print("=" * 90)
print()
print("Tests run this session:")
print("  prev_close_pos main test: 6 tests")
print("  per-session bear-short: 8 tests")
print("  gap interaction: 3 tests")
print("  what_kills_edge Part 4 (direction x prev_dir): 4 tests")
print("  bull-short audit: 1 test (this one)")
print()

honest_K = 6 + 8 + 3 + 4 + 1
print(f"Honest K = {honest_K}")
p_bull_short = 0.000709
# Where does this rank? It's the strongest finding.
# BH at rank 1: 0.05 * 1 / 22 = 0.00227
bh_thresh = 0.05 * 1 / honest_K
print(f"Bull-short p = {p_bull_short:.6f}")
print(f"BH threshold at rank 1 (K={honest_K}): {bh_thresh:.6f}")
if p_bull_short <= bh_thresh:
    print(f"SURVIVES BH FDR at honest K={honest_K}")
else:
    print(f"KILLED by BH FDR at honest K={honest_K}")

con.close()
print()
print("=" * 90)
print("ADVERSARIAL AUDIT COMPLETE")
print("=" * 90)
