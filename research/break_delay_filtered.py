"""Break delay tested WITHIN deployed lane filters — the correct test.
O5 default per methodology rules. Exact deployed parameters.

# e2-lookahead-policy: tainted
# break_delay_min (orb_{sess}_break_delay_min) is used as a predictor of pnl_r on E2 entries.
# On E2, ~41% of trades have entry_ts < break_ts (range-touch fires before close-outside-ORB),
# making break_delay_min post-entry on that subset. All findings from this script should be
# treated as unreliable for E2 lanes. Clean re-derivation using pre-ORB features (atr_20_pct,
# ovn_range_pct) is required before any result can be cited.
# Registry: docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md
"""

import numpy as np
import duckdb
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# Deployed lanes with their EXACT filter SQL conditions
LANES = {
    "NYSE_CLOSE VOL_RV12_N20 O15 RR1.0": {
        "session": "NYSE_CLOSE",
        "orb_minutes": 15,
        "rr_target": 1.0,
        "filter_sql": 'AND d."rel_vol_NYSE_CLOSE" >= 1.2',
    },
    "NYSE_OPEN X_MES_ATR60 O15 RR1.0": {
        "session": "NYSE_OPEN",
        "orb_minutes": 15,
        "rr_target": 1.0,
        # X_MES_ATR60: MES atr_20_pct >= 60 on same trading day
        "filter_sql": """AND d.trading_day IN (
            SELECT trading_day FROM daily_features
            WHERE symbol='MES' AND orb_minutes=5 AND atr_20_pct >= 60
        )""",
    },
    "COMEX_SETTLE ATR70_VOL O5 RR1.0": {
        "session": "COMEX_SETTLE",
        "orb_minutes": 5,
        "rr_target": 1.0,
        "filter_sql": 'AND d.atr_20_pct >= 70 AND d."rel_vol_COMEX_SETTLE" >= 1.2',
    },
    "SINGAPORE_OPEN ORB_G8 O15 RR4.0": {
        "session": "SINGAPORE_OPEN",
        "orb_minutes": 15,
        "rr_target": 4.0,
        "filter_sql": 'AND d."orb_SINGAPORE_OPEN_size" >= 8.0',
    },
    "US_DATA_1000 X_MES_ATR60 O5 RR1.0 S075": {
        "session": "US_DATA_1000",
        "orb_minutes": 5,
        "rr_target": 1.0,
        "filter_sql": """AND d.trading_day IN (
            SELECT trading_day FROM daily_features
            WHERE symbol='MES' AND orb_minutes=5 AND atr_20_pct >= 60
        )""",
        "stop_mult": 0.75,
    },
}

print("=" * 110)
print("BREAK DELAY WITHIN DEPLOYED LANE FILTERS")
print("Testing: does break speed add signal ON TOP OF existing filters?")
print("=" * 110)

all_pvalues = []
results = []

for lane_name, cfg in LANES.items():
    sess = cfg["session"]
    om = cfg["orb_minutes"]
    rr = cfg["rr_target"]
    filt_sql = cfg["filter_sql"]
    stop_mult = cfg.get("stop_mult")
    delay_col = f"orb_{sess}_break_delay_min"

    # Build stop_mult condition
    stop_cond = ""
    if stop_mult:
        stop_cond = f"AND o.stop_multiplier = {stop_mult}"

    # For S075 lanes, pnl_r in orb_outcomes is at S1.0 — need to check
    # Actually orb_outcomes stores outcomes per (rr, cb, em, stop_mult) combo
    # Let's just match on rr_target and not filter stop_mult in outcomes
    # (stop_mult affects stop price, which is in the outcomes row)

    query = f"""
        SELECT o.pnl_r, o.outcome, o.trading_day,
               d."{delay_col}" as delay,
               o.entry_ts
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MNQ'
          AND o.orb_label = '{sess}'
          AND d.orb_minutes = {om}
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target = {rr}
          AND d."{delay_col}" IS NOT NULL
          AND o.entry_ts IS NOT NULL
          {stop_cond}
          {filt_sql}
    """

    try:
        r = con.sql(query).fetchnumpy()
    except Exception as e:
        print(f"\n--- {lane_name} ---")
        print(f"  QUERY ERROR: {e}")
        continue

    pnl = np.array(r["pnl_r"])
    delay = np.array(r["delay"])

    if len(pnl) < 30:
        print(f"\n--- {lane_name} ---")
        print(f"  N={len(pnl)} -- too small")
        continue

    # Overall baseline
    baseline_mean = pnl.mean()
    baseline_wr = (pnl > 0).mean()

    # Buckets
    fast = pnl[delay <= 5]
    medium = pnl[(delay > 5) & (delay <= 15)]
    slow = pnl[delay > 15]

    print(f"\n--- {lane_name} ---")
    print(f"  Baseline: N={len(pnl):,}  mean={baseline_mean:+.4f}  WR={baseline_wr:.1%}")
    print()
    print(f"  {'Bucket':<10s} {'N':>6s} {'mean':>10s} {'median':>10s} {'WR':>8s} {'std':>8s}")
    print(f"  {'-' * 55}")

    for label, sub in [("FAST<=5m", fast), ("MED 5-15m", medium), ("SLOW>15m", slow)]:
        if len(sub) < 5:
            print(f"  {label:<10s} {len(sub):>6d}       ---       ---      ---      ---")
            continue
        print(
            f"  {label:<10s} {len(sub):>6d} {sub.mean():>+10.4f} {np.median(sub):>+10.4f} "
            f"{(sub > 0).mean():>8.1%} {sub.std(ddof=1):>8.4f}"
        )

    # FAST vs SLOW test
    if len(fast) >= 10 and len(slow) >= 10:
        t, p = stats.ttest_ind(fast, slow, equal_var=False)
        n1, n2 = len(fast), len(slow)
        pooled = np.sqrt(((n1 - 1) * fast.std(ddof=1) ** 2 + (n2 - 1) * slow.std(ddof=1) ** 2) / (n1 + n2 - 2))
        d = (fast.mean() - slow.mean()) / pooled if pooled > 0 else 0
        all_pvalues.append(p)
        results.append(
            {
                "lane": lane_name,
                "fast_mean": fast.mean(),
                "slow_mean": slow.mean(),
                "d": d,
                "p": p,
                "fast_n": n1,
                "slow_n": n2,
            }
        )
        print(f"\n  FAST vs SLOW: d={d:+.3f}  p={p:.4f}  delta={fast.mean() - slow.mean():+.4f}")
        if abs(d) >= 0.2:
            print(f"  --> ECONOMICALLY SIGNIFICANT (|d|>=0.2)")
        elif abs(d) >= 0.1:
            print(f"  --> Notable but small (0.1 <= |d| < 0.2)")
        else:
            print(f"  --> Negligible effect (|d| < 0.1)")
    elif len(fast) < 10:
        print(f"\n  FAST N={len(fast)} too small for test")
    else:
        print(f"\n  SLOW N={len(slow)} too small for test")

    # Year-by-year if we have enough data
    if len(fast) >= 30 and len(slow) >= 30:
        years_data = {}
        trading_days = np.array(r["trading_day"])
        import pandas as _pd

        td_series = _pd.Series(trading_days)
        year_arr = td_series.dt.year.values
        for yr in sorted(set(year_arr)):
            yr_mask = year_arr == yr
            yr_fast = pnl[yr_mask & (delay <= 5)]
            yr_slow = pnl[yr_mask & (delay > 15)]
            if len(yr_fast) >= 5 and len(yr_slow) >= 5:
                years_data[yr] = (yr_fast.mean(), yr_slow.mean())

        if years_data:
            fast_better = sum(1 for f, s in years_data.values() if f > s)
            total = len(years_data)
            print(
                f"\n  Year-by-year: FAST > SLOW in {fast_better}/{total} years "
                f"({'STABLE' if fast_better / total >= 0.6 else 'FRAGILE'})"
            )

# BH FDR across all lane tests
print("\n" + "=" * 110)
print("BH FDR CORRECTION")
print("=" * 110)

if all_pvalues:
    reject, adj_p, _, _ = multipletests(all_pvalues, alpha=0.05, method="fdr_bh")
    K = len(all_pvalues)
    print(f"K = {K}")
    for i, res in enumerate(results):
        sig = "YES" if reject[i] else "no"
        print(f"  {res['lane']:<45s} d={res['d']:+.3f}  raw_p={res['p']:.4f}  BH_p={adj_p[i]:.4f}  sig={sig}")

# Also test MGC TOKYO_OPEN (shadow lane)
print("\n--- MGC TOKYO_OPEN ORB_G4_CONT O5 RR2.0 (shadow) ---")
try:
    r = con.sql("""
        SELECT o.pnl_r, d."orb_TOKYO_OPEN_break_delay_min" as delay
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MGC' AND o.orb_label = 'TOKYO_OPEN'
          AND d.orb_minutes = 5 AND o.entry_model = 'E2' AND o.confirm_bars = 1
          AND o.rr_target = 2.0
          AND d."orb_TOKYO_OPEN_size" >= 4.0
          AND d."orb_TOKYO_OPEN_break_bar_continues" = true
          AND d."orb_TOKYO_OPEN_break_delay_min" IS NOT NULL
    """).fetchnumpy()
    pnl = np.array(r["pnl_r"])
    delay = np.array(r["delay"])
    fast = pnl[delay <= 5]
    slow = pnl[delay > 15]
    print(f"  Baseline: N={len(pnl):,}  mean={pnl.mean():+.4f}")
    print(f"  FAST<=5m:  N={len(fast):>4d}  mean={fast.mean():+.4f}")
    print(f"  SLOW>15m:  N={len(slow):>4d}  mean={slow.mean():+.4f}")
    if len(fast) >= 10 and len(slow) >= 10:
        t, p = stats.ttest_ind(fast, slow, equal_var=False)
        n1, n2 = len(fast), len(slow)
        pooled = np.sqrt(((n1 - 1) * fast.std(ddof=1) ** 2 + (n2 - 1) * slow.std(ddof=1) ** 2) / (n1 + n2 - 2))
        d = (fast.mean() - slow.mean()) / pooled if pooled > 0 else 0
        print(f"  d={d:+.3f}  p={p:.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

con.close()

print("\n" + "=" * 110)
print("VERDICT: Does break delay add signal to FILTERED strategies?")
print("=" * 110)
