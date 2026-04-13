"""Analyze lane constraints, DD math, and portfolio allocation."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import duckdb
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import ACCOUNT_PROFILES

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# === WHY 6 LANES? ===
prof = ACCOUNT_PROFILES["topstep_50k_mnq_auto"]
print("=== TOPSTEP 50K EXPRESS CONSTRAINTS ===")
print(f"  Account size: ${prof.account_size:,}")
print(f"  Max slots: {prof.max_slots}")
print(f"  Stop multiplier: {prof.stop_multiplier}")
print(f"  Copies: {prof.copies}")

# TopStep scaling plan
try:
    from trading_app.topstep_scaling_plan import SCALING_LADDER
    print("\n=== TOPSTEP SCALING LADDER ===")
    for step in SCALING_LADDER:
        print(f"  {step}")
except Exception as e:
    print(f"\n  Scaling ladder: {e}")

# Risk per trade — compute from actual data
print("\n=== ACTUAL RISK PER TRADE (2024+, MNQ S0.75) ===")
# MNQ: 1 point = $2.00 per contract, stop_multiplier = 0.75
# Risk = orb_size_points * $2.00 * 0.75
sessions = ["COMEX_SETTLE", "EUROPE_FLOW", "CME_PRECLOSE", "NYSE_OPEN",
            "TOKYO_OPEN", "SINGAPORE_OPEN", "US_DATA_1000"]
for sess in sessions:
    for om in [5, 15, 30]:
        r = con.execute(f"""
            SELECT
                COUNT(*) as n,
                ROUND(AVG(orb_{sess}_size), 1) as avg_orb_pts,
                ROUND(AVG(orb_{sess}_size) * 2.0 * 0.75, 2) as avg_risk_usd,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY orb_{sess}_size) * 2.0 * 0.75, 2) as p95_risk_usd
            FROM daily_features
            WHERE symbol = 'MNQ' AND orb_minutes = {om}
            AND orb_{sess}_size IS NOT NULL
            AND trading_day >= '2024-01-01' AND trading_day < '2026-01-01'
        """).fetchone()
        if r[0] > 100:
            print(f"  {sess:20s} O{om:2d}: avg_orb={r[1]:5.1f}pts  avg_risk=${r[2]:6.2f}  p95_risk=${r[3]:6.2f}  (N={r[0]})")

# Worst-case simultaneous risk
print("\n=== WORST CASE: All lanes fire same day ===")
# Sessions are spread across the day — most won't overlap
# But in theory, all could trigger on the same trading day
r = con.execute("""
    SELECT trading_day, COUNT(DISTINCT orb_label) as sessions_with_trade
    FROM orb_outcomes
    WHERE symbol = 'MNQ' AND entry_model = 'E2' AND orb_minutes = 5
    AND orb_label IN ('COMEX_SETTLE','EUROPE_FLOW','CME_PRECLOSE','NYSE_OPEN','TOKYO_OPEN','SINGAPORE_OPEN')
    AND pnl_r IS NOT NULL AND trading_day >= '2024-01-01'
    GROUP BY trading_day
    ORDER BY sessions_with_trade DESC
    LIMIT 5
""").fetchall()
for row in r:
    print(f"  {row[0]}: {row[1]} sessions traded")

# Average number of sessions per day
r2 = con.execute("""
    SELECT ROUND(AVG(sess_count), 1), ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY sess_count), 0)
    FROM (
        SELECT trading_day, COUNT(DISTINCT orb_label) as sess_count
        FROM orb_outcomes
        WHERE symbol = 'MNQ' AND entry_model = 'E2' AND orb_minutes = 5
        AND orb_label IN ('COMEX_SETTLE','EUROPE_FLOW','CME_PRECLOSE','NYSE_OPEN','TOKYO_OPEN','SINGAPORE_OPEN')
        AND pnl_r IS NOT NULL AND trading_day >= '2024-01-01'
        GROUP BY trading_day
    )
""").fetchone()
print(f"  Average sessions/day: {r2[0]}, P95: {r2[1]}")

# Maximum daily loss (all lanes lose at S0.75)
# Avg ORB ~25pts at O5, risk = 25 * $2 * 0.75 = $37.50 per lane
# 6 lanes all losing = 6 * $37.50 = $225
# TopStep trailing DD = $2000 for Express 50K
print("\n=== DRAWDOWN BUDGET ===")
print("  TopStep 50K Express trailing DD: $2,000")
print("  TopStep 50K Express daily loss limit: $1,000")
print(f"  Average risk per trade (O5, S0.75): ~$37.50")
print(f"  Worst case 6 lanes all stop out: ~$225 (11% of trailing DD)")
print(f"  Worst case 7 lanes all stop out: ~$262 (13% of trailing DD)")
print(f"  Worst case 10 lanes all stop out: ~$375 (19% of trailing DD)")
print()
print("  CONCLUSION: 6 lanes is NOT a DD constraint.")
print("  The 6-slot limit is from the SCALING PLAN Day-1 cap (2 lots max),")
print("  not drawdown math. With 1 contract per lane, DD headroom is huge.")

# === CORRELATION MATRIX ===
print("\n" + "=" * 80)
print("PAIRWISE CORRELATION MATRIX — Best strategy per session")
print("=" * 80)

import numpy as np
from scipy import stats

best_per_session = {
    "COMEX_SETTLE": ("COMEX_SETTLE", 1.5, 5),
    "EUROPE_FLOW": ("EUROPE_FLOW", 1.5, 5),
    "CME_PRECLOSE": ("CME_PRECLOSE", 1.0, 5),
    "NYSE_OPEN": ("NYSE_OPEN", 1.0, 5),
    "TOKYO_OPEN": ("TOKYO_OPEN", 1.5, 5),
    "SINGAPORE_OPEN": ("SINGAPORE_OPEN", 1.5, 30),
    "US_DATA_1000_VWAP": ("US_DATA_1000", 1.5, 15),
}

# Get daily pnl for each
daily_pnl = {}
for label, (sess, rr, om) in best_per_session.items():
    rows = con.execute(f"""
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        WHERE o.entry_model = 'E2' AND o.symbol = 'MNQ' AND o.orb_label = '{sess}'
        AND o.orb_minutes = {om} AND o.rr_target = {rr} AND o.confirm_bars = 1
        AND o.pnl_r IS NOT NULL AND o.trading_day >= '2020-01-01' AND o.trading_day < '2026-01-01'
    """).fetchall()
    daily_pnl[label] = {str(r[0]): r[1] for r in rows}

labels = list(best_per_session.keys())
n = len(labels)

# Print header
header = f"{'':20s}" + "".join(f"{l[:8]:>10s}" for l in labels)
print(header)

for i in range(n):
    row_str = f"{labels[i]:20s}"
    for j in range(n):
        if i == j:
            row_str += f"{'1.000':>10s}"
        else:
            common = set(daily_pnl[labels[i]].keys()) & set(daily_pnl[labels[j]].keys())
            if len(common) < 30:
                row_str += f"{'N/A':>10s}"
            else:
                a = np.array([daily_pnl[labels[i]][d] for d in sorted(common)])
                b = np.array([daily_pnl[labels[j]][d] for d in sorted(common)])
                corr, _ = stats.pearsonr(a, b)
                row_str += f"{corr:+10.3f}"
    print(row_str)

# Compute portfolio stats: current 6 vs proposed 7
print("\n=== PORTFOLIO COMPARISON ===")
# Sum daily pnl across lanes
all_days = set()
for v in daily_pnl.values():
    all_days |= set(v.keys())
all_days = sorted(all_days)

for config_name, lane_set in [
    ("Current 6 lanes", ["COMEX_SETTLE", "EUROPE_FLOW", "CME_PRECLOSE", "NYSE_OPEN", "TOKYO_OPEN", "SINGAPORE_OPEN"]),
    ("Swap SING->VWAP", ["COMEX_SETTLE", "EUROPE_FLOW", "CME_PRECLOSE", "NYSE_OPEN", "TOKYO_OPEN", "US_DATA_1000_VWAP"]),
    ("All 7 lanes", ["COMEX_SETTLE", "EUROPE_FLOW", "CME_PRECLOSE", "NYSE_OPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "US_DATA_1000_VWAP"]),
]:
    daily_sum = []
    for d in all_days:
        total = sum(daily_pnl[l].get(d, 0) for l in lane_set)
        daily_sum.append(total)
    arr = np.array(daily_sum)
    nonzero = arr[arr != 0]
    if len(nonzero) > 0:
        mean_r = np.mean(nonzero)
        std_r = np.std(nonzero, ddof=1)
        sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
        max_dd = 0
        peak = 0
        cumsum = np.cumsum(nonzero)
        for v in cumsum:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        print(f"  {config_name:25s}: mean/day={mean_r:+.4f}R  Sharpe={sharpe:.2f}  maxDD={max_dd:.1f}R  totalR={np.sum(nonzero):.0f}R")

con.close()
