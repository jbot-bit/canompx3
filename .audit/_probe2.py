import os, duckdb
con = duckdb.connect(os.environ["DUCKDB_PATH"], read_only=True)
for col in ["entry_model","confirm_bars","rr_target","orb_minutes"]:
    vals = con.execute(f"SELECT DISTINCT {col} FROM orb_outcomes WHERE symbol IN ('MNQ','MES','MGC') ORDER BY 1").fetchall()
    print(f"{col}: {[v[0] for v in vals]}")
# pnl_r distribution sanity: confirm losses == -1.0 exactly, wins are fractional RR
print()
r = con.execute("""
  SELECT outcome, COUNT(*), ROUND(MIN(pnl_r),3), ROUND(MAX(pnl_r),3), ROUND(AVG(pnl_r),4)
  FROM orb_outcomes WHERE symbol='MNQ' AND orb_label='TOKYO_OPEN' AND orb_minutes=5
    AND entry_model='E2' AND rr_target=2.0 AND confirm_bars=1 AND trading_day < DATE '2026-01-01'
  GROUP BY outcome
""").fetchall()
print("MNQ TOKYO_OPEN E2 5m rr2.0 cb1 pre-2026 outcome dist:")
for row in r: print("  ", row)
# does a 'direction' or long/short column exist?
cols = [c[1] for c in con.execute("PRAGMA table_info('orb_outcomes')").fetchall()]
print()
print("has 'direction'?", 'direction' in cols, "| break-dir-ish cols:", [c for c in cols if 'dir' in c.lower() or 'side' in c.lower() or 'long' in c.lower()])
