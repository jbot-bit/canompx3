"""Adversarial audit of VWAP_MID_ALIGNED filter at US_DATA_1000 O15."""

import sys

sys.stdout.reconfigure(encoding="utf-8")
import duckdb
import numpy as np
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

ALIGNED_SQL = """
    d.orb_US_DATA_1000_vwap IS NOT NULL
    AND (
        (d.orb_US_DATA_1000_break_dir = 'long'
         AND (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 > d.orb_US_DATA_1000_vwap)
        OR
        (d.orb_US_DATA_1000_break_dir = 'short'
         AND (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 < d.orb_US_DATA_1000_vwap)
    )
"""

BASE_SQL = """
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.entry_model = 'E2' AND o.symbol = 'MNQ' AND o.orb_label = 'US_DATA_1000'
    AND o.orb_minutes = 15 AND o.confirm_bars = 1
    AND o.pnl_r IS NOT NULL AND o.trading_day < '2026-01-01'
"""

print("=" * 90)
print("ADVERSARIAL AUDIT: VWAP_MID_ALIGNED US_DATA_1000 O15")
print("=" * 90)

# TEST 1: Inverse filter
print("\n=== TEST 1: Aligned vs Anti-aligned ===")
for rr in [1.0, 1.5, 2.0]:
    a = con.execute(f"SELECT AVG(o.pnl_r), COUNT(*) {BASE_SQL} AND o.rr_target = {rr} AND {ALIGNED_SQL}").fetchone()
    b = con.execute(
        f"SELECT AVG(o.pnl_r), COUNT(*) {BASE_SQL} AND o.rr_target = {rr} AND d.orb_US_DATA_1000_vwap IS NOT NULL AND NOT ({ALIGNED_SQL})"
    ).fetchone()
    print(f"  RR{rr}: aligned={a[0]:+.4f}(N={a[1]})  anti={b[0]:+.4f}(N={b[1]})  delta={a[0] - b[0]:+.4f}")

# TEST 2: Two-sample t-test
print("\n=== TEST 2: Welch t-test ===")
for rr in [1.0, 1.5]:
    a_rows = con.execute(f"SELECT o.pnl_r {BASE_SQL} AND o.rr_target = {rr} AND {ALIGNED_SQL}").fetchall()
    b_rows = con.execute(
        f"SELECT o.pnl_r {BASE_SQL} AND o.rr_target = {rr} AND d.orb_US_DATA_1000_vwap IS NOT NULL AND NOT ({ALIGNED_SQL})"
    ).fetchall()
    a_vals = [x[0] for x in a_rows]
    b_vals = [x[0] for x in b_rows]
    t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    print(f"  RR{rr}: t={t:+.3f}, p={p:.6f}")

# TEST 3: Aperture specificity
print("\n=== TEST 3: O5 vs O15 vs O30 ===")
for om in [5, 15, 30]:
    r = con.execute(f"""
        SELECT AVG(o.pnl_r), COUNT(*)
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol AND d.orb_minutes = {om}
        WHERE o.entry_model = 'E2' AND o.symbol = 'MNQ' AND o.orb_label = 'US_DATA_1000'
        AND o.orb_minutes = {om} AND o.rr_target = 1.5 AND o.confirm_bars = 1
        AND o.pnl_r IS NOT NULL AND o.trading_day < '2026-01-01'
        AND d.orb_US_DATA_1000_vwap IS NOT NULL
        AND (
            (d.orb_US_DATA_1000_break_dir = 'long'
             AND (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 > d.orb_US_DATA_1000_vwap)
            OR
            (d.orb_US_DATA_1000_break_dir = 'short'
             AND (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 < d.orb_US_DATA_1000_vwap)
        )
    """).fetchone()
    if r[0] is not None:
        print(f"  O{om}: ExpR={r[0]:+.4f}  N={r[1]}")

# TEST 4: Cross-session
print("\n=== TEST 4: VWAP alignment at OTHER sessions (RR1.5, O15) ===")
for sess in ["NYSE_OPEN", "COMEX_SETTLE", "EUROPE_FLOW", "CME_PRECLOSE", "TOKYO_OPEN", "NYSE_CLOSE"]:
    try:
        a = con.execute(f"""
            SELECT AVG(o.pnl_r), COUNT(*)
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol AND d.orb_minutes = 15
            WHERE o.entry_model = 'E2' AND o.symbol = 'MNQ' AND o.orb_label = '{sess}'
            AND o.orb_minutes = 15 AND o.rr_target = 1.5 AND o.confirm_bars = 1
            AND o.pnl_r IS NOT NULL AND o.trading_day < '2026-01-01'
            AND d.orb_{sess}_vwap IS NOT NULL
            AND (
                (d.orb_{sess}_break_dir = 'long'
                 AND (d.orb_{sess}_high + d.orb_{sess}_low) / 2.0 > d.orb_{sess}_vwap)
                OR
                (d.orb_{sess}_break_dir = 'short'
                 AND (d.orb_{sess}_high + d.orb_{sess}_low) / 2.0 < d.orb_{sess}_vwap)
            )
        """).fetchone()
        b = con.execute(f"""
            SELECT AVG(o.pnl_r), COUNT(*)
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol AND d.orb_minutes = 15
            WHERE o.entry_model = 'E2' AND o.symbol = 'MNQ' AND o.orb_label = '{sess}'
            AND o.orb_minutes = 15 AND o.rr_target = 1.5 AND o.confirm_bars = 1
            AND o.pnl_r IS NOT NULL AND o.trading_day < '2026-01-01'
            AND d.orb_{sess}_vwap IS NOT NULL
            AND NOT (
                (d.orb_{sess}_break_dir = 'long'
                 AND (d.orb_{sess}_high + d.orb_{sess}_low) / 2.0 > d.orb_{sess}_vwap)
                OR
                (d.orb_{sess}_break_dir = 'short'
                 AND (d.orb_{sess}_high + d.orb_{sess}_low) / 2.0 < d.orb_{sess}_vwap)
            )
        """).fetchone()
        if a[0] is not None and b[0] is not None:
            delta = a[0] - b[0]
            sig = "*" if abs(delta) > 0.05 else ""
            print(f"  {sess:20s}  aligned={a[0]:+.4f}(N={a[1]})  anti={b[0]:+.4f}(N={b[1]})  delta={delta:+.4f} {sig}")
    except Exception as e:
        print(f"  {sess:20s}  ERROR: {str(e)[:50]}")

# TEST 5: Permutation test
print("\n=== TEST 5: 5000-permutation null test ===")
all_pnl = con.execute(f"""
    SELECT o.pnl_r
    FROM orb_outcomes o
    WHERE o.entry_model = 'E2' AND o.symbol = 'MNQ' AND o.orb_label = 'US_DATA_1000'
    AND o.orb_minutes = 15 AND o.rr_target = 1.5 AND o.confirm_bars = 1
    AND o.pnl_r IS NOT NULL AND o.trading_day < '2026-01-01'
""").fetchall()
pnl_arr = np.array([x[0] for x in all_pnl])
n_total = len(pnl_arr)
n_pass = 772
observed_mean = 0.2101

np.random.seed(42)
null_means = np.array([np.mean(np.random.choice(pnl_arr, size=n_pass, replace=False)) for _ in range(5000)])
p_perm = np.mean(null_means >= observed_mean)
print(f"  Observed aligned mean: {observed_mean:+.4f}")
print(f"  Null distribution: mean={np.mean(null_means):+.4f}, std={np.std(null_means):.4f}")
print(f"  Permutation p-value: {p_perm:.4f}")
print(f"  (How often does random 52% subset beat {observed_mean}?)")

# TEST 6: Is VWAP alignment just a proxy for gap direction?
print("\n=== TEST 6: Confounding check — correlation with gap direction ===")
r = con.execute("""
    SELECT
        CASE WHEN (d.orb_US_DATA_1000_break_dir = 'long'
             AND (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 > d.orb_US_DATA_1000_vwap)
            OR (d.orb_US_DATA_1000_break_dir = 'short'
             AND (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 < d.orb_US_DATA_1000_vwap)
            THEN 1 ELSE 0 END as vwap_aligned,
        CASE WHEN d.gap_open_points > 0 THEN 1
             WHEN d.gap_open_points < 0 THEN -1 ELSE 0 END as gap_dir,
        CASE WHEN d.overnight_range_pct > 50 THEN 1 ELSE 0 END as high_ovn,
        d.atr_vel_ratio
    FROM daily_features d
    WHERE d.symbol = 'MNQ' AND d.orb_minutes = 15
    AND d.orb_US_DATA_1000_vwap IS NOT NULL
    AND d.orb_US_DATA_1000_break_dir IS NOT NULL
    AND d.trading_day < '2026-01-01'
""").fetchall()

vwap = np.array([x[0] for x in r])
gap = np.array([x[1] for x in r])
ovn = np.array([x[2] for x in r])
atr_vel = np.array([x[3] for x in r if x[3] is not None])
vwap_sub = vwap[: len(atr_vel)]

corr_gap, p_gap = stats.pointbiserialr(vwap, gap)
corr_ovn, p_ovn = stats.pointbiserialr(vwap, ovn)
corr_atr, p_atr = stats.pearsonr(vwap_sub, atr_vel)
print(f"  VWAP vs gap_dir:       r={corr_gap:+.3f}  p={p_gap:.4f}")
print(f"  VWAP vs high_overnight: r={corr_ovn:+.3f}  p={p_ovn:.4f}")
print(f"  VWAP vs atr_vel_ratio: r={corr_atr:+.3f}  p={p_atr:.4f}")

con.close()
