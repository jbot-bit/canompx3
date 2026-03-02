#!/usr/bin/env python3
"""Quick ranking of STABLE session slots for concentrated trading plan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import duckdb
import pandas as pd
import numpy as np
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

slots = [
    ("MGC", "CME_REOPEN",   "MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G5"),
    ("MGC", "TOKYO_OPEN",   "MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5_CONT"),
    ("MNQ", "CME_PRECLOSE", "MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5"),
    ("MNQ", "CME_REOPEN",   "MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV12_N20"),
    ("MNQ", "NYSE_OPEN",    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4"),
    ("MES", "COMEX_SETTLE", "MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6"),
    ("MES", "NYSE_OPEN",    "MES_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20"),
    ("M2K", "NYSE_OPEN",    "M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O30"),
]

print("=== STABLE Session Slot Ranking ===\n")

all_trades = {}

for inst, sess, strat_id in slots:
    spec = get_cost_spec(inst)

    trades = con.execute("""
        SELECT oo.trading_day, oo.pnl_r
        FROM validated_setups vs
        JOIN strategy_trade_days std ON vs.strategy_id = std.strategy_id
        JOIN orb_outcomes oo
          ON oo.symbol = vs.instrument
          AND oo.orb_label = vs.orb_label
          AND oo.orb_minutes = vs.orb_minutes
          AND oo.entry_model = vs.entry_model
          AND oo.rr_target = vs.rr_target
          AND oo.confirm_bars = vs.confirm_bars
          AND oo.trading_day = std.trading_day
        WHERE vs.strategy_id = ?
          AND oo.outcome IN ('win', 'loss')
          AND oo.pnl_r IS NOT NULL
        ORDER BY oo.trading_day
    """, [strat_id]).fetchdf()

    if trades.empty:
        print(f"  {inst} {sess}: NO TRADES")
        continue

    key = f"{inst} {sess}"
    all_trades[key] = trades.copy()
    all_trades[key]["slot"] = key
    all_trades[key]["instrument"] = inst

    n = len(trades)
    avg_r = trades["pnl_r"].mean()
    wr = (trades["pnl_r"] > 0).mean()
    std_r = trades["pnl_r"].std()
    sharpe = avg_r / std_r * np.sqrt(252) if std_r > 0 else 0

    td_min = trades["trading_day"].min()
    td_max = trades["trading_day"].max()
    total_biz_days = np.busday_count(
        pd.Timestamp(td_min).date(), pd.Timestamp(td_max).date()
    )
    freq_day = n / total_biz_days if total_biz_days > 0 else 0

    recent = trades[trades["trading_day"] >= pd.Timestamp("2025-03-01")]
    recent_r = recent["pnl_r"].mean() if len(recent) >= 10 else float("nan")
    recent_n = len(recent)

    yearly = trades.copy()
    yearly["yr"] = pd.to_datetime(yearly["trading_day"]).dt.year
    yrstats = yearly.groupby("yr")["pnl_r"].agg(["mean", "count"])
    yrs_pos = (yrstats["mean"] > 0).sum()
    yrs_total = len(yrstats)

    print(f"  {key:25s}  ExpR={avg_r:+.4f}  Sharpe={sharpe:.2f}  WR={wr:.1%}  N={n}  Yrs+={yrs_pos}/{yrs_total}  Freq={freq_day:.2f}/day  Cost=${spec.total_friction:.2f}")
    if not np.isnan(recent_r):
        print(f"    -> Recent 12m: N={recent_n}, avgR={recent_r:+.4f}")
    # Year breakdown
    for _, row in yrstats.iterrows():
        pass  # skip for brevity

print()

# === Now simulate the TOP 3 portfolio ===
print("=" * 80)
print("TOP 3 CONCENTRATED PORTFOLIO SIMULATION")
print("=" * 80)

# Pick top 3 by composite score: STABLE status + ExpR + recent trend
# From the data: MGC CME_REOPEN, MNQ CME_PRECLOSE, MES COMEX_SETTLE
top3_keys = ["MGC CME_REOPEN", "MNQ CME_PRECLOSE", "MES COMEX_SETTLE"]

# Also test alternative: MGC CME_REOPEN, MGC TOKYO_OPEN, MNQ NYSE_OPEN
alt3_keys = ["MGC CME_REOPEN", "MGC TOKYO_OPEN", "MNQ NYSE_OPEN"]

for label, keys in [("Option A: Cross-instrument", top3_keys),
                     ("Option B: Evening cluster", alt3_keys)]:
    print(f"\n--- {label}: {', '.join(keys)} ---")

    # Merge trades from selected slots, aggregate by trading_day
    frames = []
    for k in keys:
        if k in all_trades:
            df = all_trades[k][["trading_day", "pnl_r", "slot"]].copy()
            frames.append(df)

    if not frames:
        print("  No trade data!")
        continue

    combined = pd.concat(frames)
    combined["trading_day"] = pd.to_datetime(combined["trading_day"])

    # Daily P&L: sum of all trades that day (could be 0-3 trades)
    daily = combined.groupby("trading_day").agg(
        total_r=("pnl_r", "sum"),
        n_trades=("pnl_r", "count"),
    ).reset_index().sort_values("trading_day")

    # Fill in missing trading days (days with 0 trades = 0 P&L)
    all_days = pd.bdate_range(daily["trading_day"].min(), daily["trading_day"].max())
    daily = daily.set_index("trading_day").reindex(all_days, fill_value=0).reset_index()
    daily.rename(columns={"index": "trading_day"}, inplace=True)

    total_r = daily["total_r"].sum()
    avg_daily_r = daily["total_r"].mean()
    daily_std = daily["total_r"].std()
    sharpe_daily = avg_daily_r / daily_std * np.sqrt(252) if daily_std > 0 else 0

    n_trades = combined.shape[0]
    n_days_active = (daily["n_trades"] > 0).sum()
    n_total_days = len(daily)
    active_pct = n_days_active / n_total_days

    # Avg trades per active day
    avg_trades_per_day = combined.groupby("trading_day").size().mean()

    # Win rate of daily P&L
    daily_wr = (daily["total_r"] > 0).sum() / (daily["total_r"] != 0).sum() if (daily["total_r"] != 0).sum() > 0 else 0

    # Max drawdown in R
    cumr = daily["total_r"].cumsum()
    running_max = cumr.cummax()
    dd = cumr - running_max
    max_dd = dd.min()

    # Yearly breakdown
    daily["yr"] = daily["trading_day"].dt.year
    yearly = daily.groupby("yr")["total_r"].agg(["sum", "count"])
    yearly.columns = ["total_r", "days"]

    print(f"  Total R: {total_r:+.1f}R over {n_total_days} trading days ({n_total_days/252:.1f} yrs)")
    print(f"  Trades: {n_trades} total, {avg_trades_per_day:.1f} avg/active day")
    print(f"  Active days: {n_days_active}/{n_total_days} ({active_pct:.0%})")
    print(f"  Avg daily R: {avg_daily_r:+.4f}")
    print(f"  Daily Sharpe (ann): {sharpe_daily:.2f}")
    print(f"  Daily win rate: {daily_wr:.1%}")
    print(f"  Max drawdown: {max_dd:+.1f}R")
    print(f"\n  Year-by-year:")
    for yr, row in yearly.iterrows():
        yr_avg = row["total_r"] / row["days"] if row["days"] > 0 else 0
        print(f"    {yr}: {row['total_r']:+6.1f}R ({row['days']} days, avg {yr_avg:+.4f}R/day)")

# === Per-slot trade frequency ===
print("\n" + "=" * 80)
print("TRADE FREQUENCY BY SLOT")
print("=" * 80)
for k in top3_keys + [x for x in alt3_keys if x not in top3_keys]:
    if k not in all_trades:
        continue
    df = all_trades[k]
    n = len(df)
    td_range = (pd.to_datetime(df["trading_day"].max()) - pd.to_datetime(df["trading_day"].min())).days
    trades_per_month = n / (td_range / 30.44) if td_range > 0 else 0
    print(f"  {k:25s}: {n} trades over {td_range/365.25:.1f} yrs = {trades_per_month:.1f} trades/month")

con.close()
