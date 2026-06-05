"""AUDIT 02 - COVERAGE MATRIX. Read-only truth layers (orb_outcomes, daily_features, bars_1m)."""
import os, duckdb
from pipeline.dst import SESSION_CATALOG
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS

con = duckdb.connect(os.environ["DUCKDB_PATH"], read_only=True)

print("="*110)
print("AUDIT 02 - COVERAGE MATRIX (truth layers only)")
print("="*110)
print(f"ACTIVE_ORB_INSTRUMENTS: {list(ACTIVE_ORB_INSTRUMENTS)}")
print()

# schema of orb_outcomes
cols = [r[1] for r in con.execute("PRAGMA table_info('orb_outcomes')").fetchall()]
print(f"orb_outcomes columns ({len(cols)}): {cols}")
print()

# distinct session labels actually present in orb_outcomes
print("DISTINCT orb_label values present in orb_outcomes:")
print("-"*110)
rows = con.execute("""
    SELECT orb_label, COUNT(*) AS n_rows,
           COUNT(DISTINCT symbol) AS n_sym,
           COUNT(DISTINCT trading_day) AS n_days,
           MIN(trading_day) AS first_day,
           MAX(trading_day) AS last_day
    FROM orb_outcomes
    GROUP BY orb_label
    ORDER BY orb_label
""").fetchall()
print(f"  {'orb_label':<18}{'n_rows':>10}{'n_sym':>7}{'n_days':>8}  {'first':<12}{'last':<12}")
for r in rows:
    print(f"  {str(r[0]):<18}{r[1]:>10}{r[2]:>7}{r[3]:>8}  {str(r[4]):<12}{str(r[5]):<12}")

present = {r[0] for r in rows}
catalog = set(SESSION_CATALOG.keys())
print()
print(f"In catalog but ZERO orb_outcomes rows: {sorted(catalog - present)}")
print(f"In orb_outcomes but NOT in current catalog (legacy/renamed): {sorted(present - catalog)}")
print()

# per-session x symbol x orb_minutes coverage, split pre-2026 vs 2026 holdout
print("PER-SESSION x SYMBOL coverage (pre-2026 discovery rows vs 2026 holdout rows):")
print("-"*110)
print(f"  {'orb_label':<16}{'sym':<5}{'apertures':<14}{'pre2026_rows':>13}{'h2026_rows':>12}{'first':<12}{'last':<12}")
rows2 = con.execute("""
    SELECT orb_label, symbol,
           STRING_AGG(DISTINCT CAST(orb_minutes AS VARCHAR), ',') AS apertures,
           SUM(CASE WHEN trading_day < DATE '2026-01-01' THEN 1 ELSE 0 END) AS pre2026,
           SUM(CASE WHEN trading_day >= DATE '2026-01-01' THEN 1 ELSE 0 END) AS h2026,
           MIN(trading_day) AS f, MAX(trading_day) AS l
    FROM orb_outcomes
    GROUP BY orb_label, symbol
    ORDER BY orb_label, symbol
""").fetchall()
for r in rows2:
    print(f"  {str(r[0]):<16}{str(r[1]):<5}{str(r[2]):<14}{r[3]:>13}{r[4]:>12}{str(r[5]):<12}{str(r[6]):<12}")
