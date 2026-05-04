"""Test 1: Per-session bear-short momentum continuation.

Hypothesis: After a bear day that closed at the absolute low (prev_close_pos < 0.2),
short ORB breaks perform better the next day (momentum continuation).

Scoped to VALIDATED strategies with filters applied.
Reports ALL sessions — no cherry-picking.

@research-source: per_session_bear_short_test.py
@data-source: orb_outcomes JOIN daily_features, filtered via validated_setups
@literature: Jegadeesh & Titman (1993) short-term momentum; project break_speed finding
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
        valid[["trading_day", "instrument", "orb_label", "direction", "pnl_r", "prev_day_direction", "prev_close_pos"]]
    )

df = pd.concat(all_trades, ignore_index=True)
df_unique = df.drop_duplicates(subset=["trading_day", "instrument", "orb_label", "direction"])

bear_shorts = df_unique[(df_unique["prev_day_direction"] == "bear") & (df_unique["direction"] == "short")]
print(f"Total bear-day short trades (validated, filtered): {len(bear_shorts)}")
print()

# =====================================================================
print("=" * 90)
print("TEST 1: PER-SESSION Bear extreme-low SHORT vs rest SHORT")
print("Hypothesis: momentum continuation after selloff-to-low")
print("NO BIAS: reporting ALL sessions, not cherry-picking")
print("=" * 90)
print()

sessions = sorted(bear_shorts["orb_label"].unique())
session_results = []

for sess in sessions:
    sess_data = bear_shorts[bear_shorts["orb_label"] == sess]
    ext = sess_data[sess_data["prev_close_pos"] < 0.2]["pnl_r"]
    rest = sess_data[sess_data["prev_close_pos"] >= 0.2]["pnl_r"]

    if len(ext) < 15 or len(rest) < 15:
        print(f"{sess:20s}: SKIPPED (ext N={len(ext)}, rest N={len(rest)})")
        continue

    t, p = stats.ttest_ind(ext, rest)
    delta = ext.mean() - rest.mean()
    wr_ext = (ext > 0).sum() / len(ext)
    wr_rest = (rest > 0).sum() / len(rest)

    session_results.append((sess, len(ext), ext.mean(), wr_ext, len(rest), rest.mean(), wr_rest, delta, p))
    print(
        f"{sess:20s}: ext N={len(ext):4d} mean={ext.mean():+.4f} WR={wr_ext:.1%} | "
        f"rest N={len(rest):4d} mean={rest.mean():+.4f} WR={wr_rest:.1%} | "
        f"delta={delta:+.4f} p={p:.4f}"
    )

# BH FDR
print()
print("=" * 90)
print("BH FDR CORRECTION")
print("=" * 90)
K = len(session_results)
print(f"K = {K} sessions tested")

any_survive = False
if K > 0:
    sorted_results = sorted(session_results, key=lambda x: x[8])
    for rank, (sess, n_ext, _m_ext, _wr_ext, _n_rest, _m_rest, _wr_rest, delta, p) in enumerate(sorted_results, 1):
        bh = 0.05 * rank / K
        verdict = "SURVIVES" if p <= bh else "KILLED"
        if p <= bh:
            any_survive = True
        print(f"  Rank {rank}: {sess:20s} p={p:.4f} BH={bh:.4f} {verdict} (delta={delta:+.4f}, ext_N={n_ext})")

if not any_survive:
    print("\n>>> NO SESSIONS SURVIVE BH FDR.")
else:
    print("\n>>> At least one session survives. Check instrument + year stability.")

# Instrument breakdown for p < 0.15
print()
print("=" * 90)
print("INSTRUMENT BREAKDOWN (sessions with p < 0.15)")
print("=" * 90)
found_any = False
for sess, _n_ext, _m_ext, _wr_ext, _n_rest, _m_rest, _wr_rest, _delta, p in session_results:
    if p < 0.15:
        found_any = True
        print(f"\n  {sess} (p={p:.4f}):")
        sess_data = bear_shorts[bear_shorts["orb_label"] == sess]
        for inst in sorted(sess_data["instrument"].unique()):
            inst_data = sess_data[sess_data["instrument"] == inst]
            ext_i = inst_data[inst_data["prev_close_pos"] < 0.2]["pnl_r"]
            rest_i = inst_data[inst_data["prev_close_pos"] >= 0.2]["pnl_r"]
            if len(ext_i) > 5 and len(rest_i) > 5:
                t_i, p_i = stats.ttest_ind(ext_i, rest_i)
                print(
                    f"    {inst}: ext N={len(ext_i)} mean={ext_i.mean():+.4f} | "
                    f"rest N={len(rest_i)} mean={rest_i.mean():+.4f} | p={p_i:.4f}"
                )
if not found_any:
    print("  None with p < 0.15")

# Year stability for best session
if session_results:
    best = min(session_results, key=lambda x: x[8])
    best_sess = best[0]
    print()
    print("=" * 90)
    print(f"YEAR-BY-YEAR: {best_sess} (lowest p={best[8]:.4f})")
    print("=" * 90)
    sess_data = bear_shorts[bear_shorts["orb_label"] == best_sess].copy()
    sess_data["year"] = pd.to_datetime(sess_data["trading_day"]).dt.year
    pos_years = 0
    neg_years = 0
    for year in sorted(sess_data["year"].unique()):
        yr = sess_data[sess_data["year"] == year]
        ext = yr[yr["prev_close_pos"] < 0.2]["pnl_r"]
        rest = yr[yr["prev_close_pos"] >= 0.2]["pnl_r"]
        if len(ext) >= 3 and len(rest) >= 3:
            delta = ext.mean() - rest.mean()
            if delta > 0:
                pos_years += 1
            else:
                neg_years += 1
            print(
                f"  {year}: ext N={len(ext):3d} mean={ext.mean():+.4f} | "
                f"rest N={len(rest):3d} mean={rest.mean():+.4f} | delta={delta:+.4f}"
            )
    print(f"\n  Positive delta years: {pos_years}, Negative: {neg_years}")

con.close()
print()
print("DONE — Test 1 complete")
