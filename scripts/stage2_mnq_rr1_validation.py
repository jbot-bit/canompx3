"""STAGE 2 PRE-REGISTERED VALIDATION: MNQ RR1.0 at 4 sessions.

Pre-registration (written BEFORE running):
  Hypothesis: MNQ ORB breakout at RR1.0 has positive ExpR due to
  short-term momentum after range break (Crabel 1990, Gao 2018).
  Mechanism: Liquidity imbalance at session open creates momentum.
  RR1.0 captures the most reliable part before mean-reversion.

  Candidates: 4 sessions (CME_PRECLOSE, NYSE_OPEN, COMEX_SETTLE, US_DATA_1000)
  Selected from Stage 1 grid search on pre-2024 data.

  Test: 2025 holdout (never used for selection)
  Statistical bar: BH FDR at q=0.05 across 4 tests
  BH threshold for rank 4/4: 0.05 * 4/4 = 0.05
  BH threshold for rank 1/4: 0.05 * 1/4 = 0.0125

  Kill criteria per setup:
  - p > 0.05 on holdout -> DEAD
  - ExpR < 0 on holdout -> DEAD
  - Max DD > 30R -> needs risk cap
  - Survives 3-tick slippage stress test

  Kill criteria portfolio:
  - Fewer than 2/4 survive -> insufficient diversification
  - Total annual R < 50R -> not worth the operational complexity
"""
import sys; sys.path.insert(0, r"C:\Users\joshd\canompx3")
import duckdb, numpy as np, pandas as pd, statistics
from scipy.stats import ttest_1samp
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

SETUPS = [
    ("CME_PRECLOSE", 1.0),
    ("NYSE_OPEN", 1.0),
    ("COMEX_SETTLE", 1.0),
    ("US_DATA_1000", 1.0),
]

print("="*70)
print("STAGE 2: PRE-REGISTERED VALIDATION")
print("4 candidates, BH FDR q=0.05, 2025 holdout")
print("="*70)

results = []
for session, rr in SETUPS:
    df = con.execute(f"""
        SELECT pnl_r, trading_day, entry_price, stop_price
        FROM orb_outcomes
        WHERE symbol='MNQ' AND entry_model='E2' AND confirm_bars=1
              AND rr_target={rr} AND pnl_r IS NOT NULL AND orb_label='{session}'
        ORDER BY trading_day
    """).fetchdf()

    td = pd.to_datetime(df["trading_day"])
    train = df[td < pd.Timestamp("2024-01-01")]  # Stage 1 discovery data
    holdout = df[td >= pd.Timestamp("2025-01-01")]  # Never touched until now

    pnl_h = holdout["pnl_r"].values
    t, p = ttest_1samp(pnl_h, 0)
    wr = (pnl_h > 0).mean()

    # Risk
    risk_pts = np.abs(pd.to_numeric(holdout["entry_price"], errors="coerce") -
                      pd.to_numeric(holdout["stop_price"], errors="coerce")).values
    avg_risk_d = np.nanmean(risk_pts) * 2  # $2/pt MNQ

    # DD
    cumr = np.cumsum(pnl_h)
    peak = np.maximum.accumulate(cumr)
    dd = (cumr - peak).min()

    # Consecutive losses
    streak = 0; max_streak = 0
    for pv in pnl_h:
        if pv < 0: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    # Slippage stress (3 ticks extra)
    extra_cost = 3 * 0.25 * 2 / avg_risk_d if avg_risk_d > 0 else 0
    stress_expr = pnl_h.mean() - extra_cost

    # Year-by-year on holdout
    yr = td[td >= pd.Timestamp("2025-01-01")].dt.year
    yr_data = {}
    for y in sorted(yr.unique()):
        yp = pnl_h[yr.values == y]
        yr_data[y] = yp.mean()

    results.append({
        "session": session, "rr": rr, "expr": pnl_h.mean(), "p": p,
        "wr": wr, "n": len(pnl_h), "dd": dd, "max_streak": max_streak,
        "avg_risk_d": avg_risk_d, "stress_expr": stress_expr,
        "train_expr": train["pnl_r"].mean(), "train_n": len(train),
        "yr_data": yr_data,
    })

# Sort by p-value for BH FDR
results.sort(key=lambda x: x["p"])

# BH FDR at q=0.05
print(f"\n{'Session':<20} {'ExpR':>7} {'WR':>5} {'N':>5} {'p':>10} {'BH_thresh':>10} {'BH':>5} {'DD':>6} {'3tick':>7}")
print("-"*80)

n_tests = len(results)
survivors = 0
for rank, r in enumerate(results, 1):
    bh_thresh = 0.05 * rank / n_tests
    passes_bh = r["p"] <= bh_thresh
    if passes_bh:
        survivors += 1

    status = "PASS" if passes_bh and r["expr"] > 0 else "FAIL"
    print(f"  {r['session']:<18} {r['expr']:>+7.4f} {r['wr']:>5.0%} {r['n']:>5} {r['p']:>10.6f} {bh_thresh:>10.4f} {status:>5} {r['dd']:>+6.1f} {r['stress_expr']:>+7.4f}")

# Portfolio summary
print(f"\n{'='*70}")
print(f"PORTFOLIO SUMMARY (BH FDR survivors)")
print(f"{'='*70}")

portfolio_pnl = []
portfolio_trades = 0
portfolio_annual_d = 0

for r in results:
    if r["p"] <= 0.05 * (results.index(r) + 1) / n_tests and r["expr"] > 0:
        annual_trades = r["n"] * (252 / len(pd.date_range("2025-01-01", "2026-03-20")))
        annual_r = r["expr"] * annual_trades
        annual_d = annual_r * r["avg_risk_d"]
        portfolio_annual_d += annual_d
        portfolio_trades += annual_trades

        print(f"  {r['session']:<18} {r['expr']:+.4f}R x {annual_trades:.0f}/yr = {annual_r:+.0f}R = ${annual_d:+,.0f}")
        print(f"    Train: {r['train_expr']:+.4f}R (N={r['train_n']}) | OOS degradation: {(1 - r['expr']/r['train_expr'])*100:.0f}%")
        for y, yexpr in r["yr_data"].items():
            print(f"    {y}: {yexpr:+.4f}R")

print(f"\n  TOTAL: {portfolio_trades:.0f} trades/yr | ${portfolio_annual_d:+,.0f}/yr per micro")
print(f"  5 Tradeify accounts: ${portfolio_annual_d * 5:+,.0f}/yr")
print(f"  At 50% live degradation: ${portfolio_annual_d * 0.5:+,.0f}/yr (1 micro)")
print(f"  At 50% degradation, 5 accounts: ${portfolio_annual_d * 0.5 * 5:+,.0f}/yr")

# Kill criteria check
print(f"\n{'='*70}")
print("KILL CRITERIA")
print("="*70)
print(f"  Survivors: {survivors}/4 (need >= 2): {'PASS' if survivors >= 2 else 'FAIL'}")
total_annual_r = sum(r["expr"] * r["n"] * (252/len(pd.date_range("2025-01-01","2026-03-20"))) for r in results if r["expr"] > 0)
print(f"  Total annual R: {total_annual_r:.0f} (need >= 50): {'PASS' if total_annual_r >= 50 else 'FAIL'}")
all_survive_stress = all(r["stress_expr"] > 0 for r in results if r["expr"] > 0)
print(f"  All survive 3-tick stress: {'PASS' if all_survive_stress else 'FAIL'}")

con.close()
