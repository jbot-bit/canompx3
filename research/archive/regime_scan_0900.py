"""
Regime-focused scan: MGC 0900 E1 in 2025-2026.
DOW, ORB size tiers, DST split, and negative correlations.
Correct joins, zero look-ahead.
"""
import duckdb
import numpy as np
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

SAFE_JOIN = """
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
"""

REGIME_START = '2025-01-01'

def get_pnl(session, em, rr, cb, size_min, extra="", regime=True):
    regime_clause = f"AND o.trading_day >= '{REGIME_START}'" if regime else ""
    size_col = f"orb_{session}_size"
    return con.execute(f"""
        SELECT o.pnl_r {SAFE_JOIN}
        WHERE o.symbol = 'MGC' AND o.orb_label = '{session}'
          AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
          AND d.{size_col} >= {size_min}
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          {regime_clause}
          {extra}
    """).fetchnumpy()['pnl_r']


# ================================================================
# 1. DOW BREAKDOWN — 0900 E1 in 2025+
# ================================================================
print("=" * 80)
print("MGC 0900 E1 — DOW BREAKDOWN (2025-2026 REGIME)")
print("=" * 80)

for rr in [2.0, 2.5, 3.0]:
    for cb in [2, 3]:
        rows = con.execute(f"""
            SELECT
                CASE EXTRACT(isodow FROM o.trading_day)
                    WHEN 1 THEN 'Mon'
                    WHEN 2 THEN 'Tue'
                    WHEN 3 THEN 'Wed'
                    WHEN 4 THEN 'Thu'
                    WHEN 5 THEN 'Fri'
                END as dow,
                COUNT(*) as n,
                AVG(o.pnl_r) as expr,
                SUM(o.pnl_r) as totr,
                SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
            {SAFE_JOIN}
            WHERE o.symbol = 'MGC' AND o.orb_label = '0900'
              AND o.entry_model = 'E1' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
              AND d.orb_0900_size >= 4.0
              AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
              AND o.trading_day >= '{REGIME_START}'
            GROUP BY 1 ORDER BY 1
        """).fetchall()
        if rows and sum(r[1] for r in rows) >= 20:
            total_n = sum(r[1] for r in rows)
            total_r = sum(r[3] for r in rows)
            print(f'\nRR{rr} CB{cb} G4+ (N={total_n}, TotalR={total_r:+.1f}):')
            for r in rows:
                tag = ' <<<' if r[1] >= 5 and r[2] > 0.5 else (' AVOID' if r[1] >= 5 and r[2] < -0.5 else '')
                print(f'  {r[0]:<5} N={r[1]:<4} ExpR={r[2]:>+.4f}  TotalR={r[3]:>+.1f}  WR={r[4]:.0f}%{tag}')


# ================================================================
# 2. ORB SIZE TIERS — 0900 E1 in 2025+
# ================================================================
print("\n" + "=" * 80)
print("MGC 0900 E1 — ORB SIZE TIERS (2025-2026 REGIME)")
print("=" * 80)

for rr in [2.0, 2.5, 3.0]:
    for cb in [2]:
        rows = con.execute(f"""
            SELECT
                CASE WHEN d.orb_0900_size < 4 THEN 'a_G0-4'
                     WHEN d.orb_0900_size < 6 THEN 'b_G4-6'
                     WHEN d.orb_0900_size < 8 THEN 'c_G6-8'
                     WHEN d.orb_0900_size < 12 THEN 'd_G8-12'
                     ELSE 'e_G12+' END as tier,
                COUNT(*) as n,
                AVG(o.pnl_r) as expr,
                SUM(o.pnl_r) as totr,
                SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
            {SAFE_JOIN}
            WHERE o.symbol = 'MGC' AND o.orb_label = '0900'
              AND o.entry_model = 'E1' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
              AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
              AND o.trading_day >= '{REGIME_START}'
            GROUP BY 1 ORDER BY 1
        """).fetchall()
        if rows:
            print(f'\nRR{rr} CB{cb} ALL sizes:')
            for r in rows:
                tag = ' <<<' if r[1] >= 8 and r[2] > 0.3 else (' AVOID' if r[1] >= 8 and r[2] < -0.3 else '')
                print(f'  {r[0]:<12} N={r[1]:<4} ExpR={r[2]:>+.4f}  TotalR={r[3]:>+.1f}  WR={r[4]:.0f}%{tag}')


# ================================================================
# 3. DST SPLIT — 0900 in 2025+
# ================================================================
print("\n" + "=" * 80)
print("MGC 0900 E1 — DST SPLIT (2025-2026 REGIME)")
print("=" * 80)

for rr in [2.0, 2.5, 3.0]:
    rows = con.execute(f"""
        SELECT
            CASE WHEN d.us_dst THEN 'SUMMER(DST)' ELSE 'WINTER(STD)' END as regime,
            COUNT(*) as n,
            AVG(o.pnl_r) as expr,
            SUM(o.pnl_r) as totr,
            SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
        {SAFE_JOIN}
        WHERE o.symbol = 'MGC' AND o.orb_label = '0900'
          AND o.entry_model = 'E1' AND o.rr_target = {rr} AND o.confirm_bars = 2
          AND d.orb_0900_size >= 4.0
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          AND o.trading_day >= '{REGIME_START}'
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    if rows:
        print(f'\nRR{rr} CB2 G4+:')
        for r in rows:
            print(f'  {r[0]:<15} N={r[1]:<4} ExpR={r[2]:>+.4f}  TotalR={r[3]:>+.1f}  WR={r[4]:.0f}%')


# ================================================================
# 4. NEGATIVE CORRELATIONS — What to AVOID in 2025+
# ================================================================
print("\n" + "=" * 80)
print("ALL SESSIONS — WORST PERFORMERS (2025-2026) — THINGS TO AVOID")
print("=" * 80)

results = []
for sess in ['0900', '1000', '1100', '1800', '2300', '0030']:
    size_col = f"orb_{sess}_size"
    for em in ['E1', 'E3']:
        for rr in [2.0, 2.5, 3.0]:
            for cb in [2, 3]:
                row = con.execute(f"""
                    SELECT COUNT(*) as n,
                           AVG(o.pnl_r) as expr,
                           SUM(o.pnl_r) as totr,
                           SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
                    {SAFE_JOIN}
                    WHERE o.symbol = 'MGC' AND o.orb_label = '{sess}'
                      AND o.entry_model = '{em}' AND o.rr_target = {rr} AND o.confirm_bars = {cb}
                      AND d.{size_col} >= 4.0
                      AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
                      AND o.trading_day >= '{REGIME_START}'
                """).fetchone()
                if row[0] >= 20:
                    results.append({
                        'config': f'{sess} {em} RR{rr} CB{cb} G4+',
                        'n': row[0], 'expr': row[1], 'totr': row[2], 'wr': row[3]
                    })

# Sort by ExpR ascending (worst first)
results.sort(key=lambda x: x['expr'])
print("\nTOP 15 WORST configs (N>=20):")
for i, r in enumerate(results[:15]):
    print(f"  {i+1:>2}. {r['config']:<30} N={r['n']:<4} ExpR={r['expr']:>+.4f}  TotalR={r['totr']:>+.1f}  WR={r['wr']:.0f}%")

print("\nTOP 15 BEST configs (N>=20):")
results.sort(key=lambda x: x['expr'], reverse=True)
for i, r in enumerate(results[:15]):
    print(f"  {i+1:>2}. {r['config']:<30} N={r['n']:<4} ExpR={r['expr']:>+.4f}  TotalR={r['totr']:>+.1f}  WR={r['wr']:.0f}%")


# ================================================================
# 5. MGC 2300 DEATH CERTIFICATE — monthly since 2024
# ================================================================
print("\n" + "=" * 80)
print("MGC 2300 DEATH CERTIFICATE — Monthly PnL Since 2024")
print("=" * 80)

for rr in [2.0, 2.5, 3.0]:
    rows = con.execute(f"""
        SELECT
            EXTRACT(year FROM o.trading_day) || '-' || LPAD(CAST(EXTRACT(month FROM o.trading_day) AS VARCHAR), 2, '0') as month,
            COUNT(*) as n,
            AVG(o.pnl_r) as expr,
            SUM(o.pnl_r) as totr
        {SAFE_JOIN}
        WHERE o.symbol = 'MGC' AND o.orb_label = '2300'
          AND o.entry_model = 'E1' AND o.rr_target = {rr} AND o.confirm_bars = 2
          AND d.orb_2300_size >= 4.0
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          AND o.trading_day >= '2024-01-01'
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    if rows:
        print(f'\nRR{rr} CB2 G4+:')
        neg_months = 0
        total_months = 0
        for r in rows:
            mark = ' NEG' if r[3] < 0 else ''
            print(f'  {r[0]} N={r[1]:<3} ExpR={r[2]:>+.4f} TotalR={r[3]:>+.1f}{mark}')
            total_months += 1
            if r[3] < 0:
                neg_months += 1
        print(f'  --- {neg_months}/{total_months} months negative ---')


# ================================================================
# 6. STATISTICAL TESTS — 0900 DOW effect in 2025+
# ================================================================
print("\n" + "=" * 80)
print("STATISTICAL TESTS — 0900 E1 RR2.5 CB2 G4+ (2025+ REGIME)")
print("=" * 80)

base = get_pnl('0900', 'E1', 2.5, 2, 4.0, regime=True)
print(f"Baseline: N={len(base)}, ExpR={np.mean(base):+.4f}")

test_results = []

# DOW tests
for dow_name, dow_val in [('Mon', 1), ('Tue', 2), ('Wed', 3), ('Thu', 4), ('Fri', 5)]:
    on = get_pnl('0900', 'E1', 2.5, 2, 4.0,
                  f"AND EXTRACT(isodow FROM o.trading_day) = {dow_val}", regime=True)
    off = get_pnl('0900', 'E1', 2.5, 2, 4.0,
                   f"AND EXTRACT(isodow FROM o.trading_day) != {dow_val}", regime=True)
    if len(on) >= 5 and len(off) >= 10:
        t, p = stats.ttest_ind(on, off, equal_var=False)
        test_results.append({
            'name': f'DOW_{dow_name}',
            'n_on': len(on), 'mean_on': np.mean(on),
            'n_off': len(off), 'mean_off': np.mean(off),
            'delta': np.mean(on) - np.mean(off),
            'p': p
        })

# DST test
dst_on = get_pnl('0900', 'E1', 2.5, 2, 4.0, "AND d.us_dst = true", regime=True)
dst_off = get_pnl('0900', 'E1', 2.5, 2, 4.0, "AND d.us_dst = false", regime=True)
if len(dst_on) >= 5 and len(dst_off) >= 5:
    t, p = stats.ttest_ind(dst_on, dst_off, equal_var=False)
    test_results.append({
        'name': 'DST_summer_vs_winter',
        'n_on': len(dst_on), 'mean_on': np.mean(dst_on),
        'n_off': len(dst_off), 'mean_off': np.mean(dst_off),
        'delta': np.mean(dst_on) - np.mean(dst_off),
        'p': p
    })

# ORB size: G6+ vs G4-6
g6plus = get_pnl('0900', 'E1', 2.5, 2, 6.0, regime=True)
g4to6 = get_pnl('0900', 'E1', 2.5, 2, 4.0,
                  "AND d.orb_0900_size < 6", regime=True)
if len(g6plus) >= 5 and len(g4to6) >= 5:
    t, p = stats.ttest_ind(g6plus, g4to6, equal_var=False)
    test_results.append({
        'name': 'G6+_vs_G4-6',
        'n_on': len(g6plus), 'mean_on': np.mean(g6plus),
        'n_off': len(g4to6), 'mean_off': np.mean(g4to6),
        'delta': np.mean(g6plus) - np.mean(g4to6),
        'p': p
    })

# Sort by p
test_results.sort(key=lambda x: x['p'])
n_tests = len(test_results)
for i, r in enumerate(test_results):
    bh = 0.05 * (i + 1) / n_tests
    sig = 'BH-SIG' if r['p'] <= bh else ''
    raw = '***' if r['p'] < 0.005 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else ''
    print(f"  {r['name']:<25} ON: N={r['n_on']:<3} ExpR={r['mean_on']:>+.4f}  |  OFF: N={r['n_off']:<3} ExpR={r['mean_off']:>+.4f}  |  delta={r['delta']:>+.4f}  p={r['p']:.4f} {raw} {sig}")

print("\nCAVEAT: 2025-2026 regime is ~14 months. Small N per bucket.")
print("These are REGIME observations, not CORE findings.")

con.close()
print("\nDone.")
