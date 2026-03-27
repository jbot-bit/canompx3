"""Mining escalation Parts 3-5. Run once, delete after."""
import io
import sys

import duckdb
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pipeline.db_config import configure_connection

con = duckdb.connect("gold.db", read_only=True)
configure_connection(con)

def qr(sql):
    return con.execute(sql).fetchall()

def q1(sql):
    return con.execute(sql).fetchone()

print("PART 3: INTERACTION TESTS")
print("=" * 70)

print("  Test 1: overnight_range Q5 + PDH sweep on COMEX_SETTLE")
base = """
    WITH ranked AS (
        SELECT o.trading_day, o.symbol, o.orb_minutes, o.pnl_r,
               NTILE(5) OVER (ORDER BY d.overnight_range) as bin
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
          AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
          AND d.overnight_range IS NOT NULL AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
    )
    SELECT COUNT(*), ROUND(AVG(ranked.pnl_r), 4),
           ROUND(AVG(CASE WHEN ranked.pnl_r > 0 THEN 1.0 ELSE 0.0 END)*100, 1)
    FROM ranked
    JOIN daily_features d2 ON ranked.trading_day=d2.trading_day AND ranked.symbol=d2.symbol AND ranked.orb_minutes=d2.orb_minutes
"""
for label, w in [
    ("Both (Q5+PDH)", "WHERE ranked.bin=5 AND d2.overnight_took_pdh=true"),
    ("Range_Q5 only", "WHERE ranked.bin=5"),
    ("PDH_true only", "WHERE d2.overnight_took_pdh=true"),
    ("Neither", "WHERE ranked.bin IN (1,2,3) AND d2.overnight_took_pdh=false"),
]:
    r = q1(base + w)
    print(f"    {label:<30} N={r[0]:>4}, ExpR={r[1]:+.4f}, WR={r[2]:.1f}%")

print("\n  Test 2: overnight_range Q1 vs Q5 on US_DATA_1000")
for qn in [1, 5]:
    r = q1(f"""
        WITH ranked AS (
            SELECT o.pnl_r, NTILE(5) OVER (ORDER BY d.overnight_range) as bin
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='US_DATA_1000' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND d.overnight_range IS NOT NULL AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
        ) SELECT COUNT(*), ROUND(AVG(pnl_r),4), ROUND(AVG(CASE WHEN pnl_r>0 THEN 1.0 ELSE 0.0 END)*100,1)
        FROM ranked WHERE bin={qn}
    """)
    print(f"    Q{qn}: N={r[0]}, ExpR={r[1]:+.4f}, WR={r[2]:.1f}%")

print("\n  Test 3: Both PDH AND PDL swept overnight")
for sess in ["US_DATA_1000", "COMEX_SETTLE", "NYSE_OPEN"]:
    for label, w in [
        ("Both swept", "d.overnight_took_pdh=true AND d.overnight_took_pdl=true"),
        ("PDH only", "d.overnight_took_pdh=true AND d.overnight_took_pdl=false"),
        ("Neither", "d.overnight_took_pdh=false AND d.overnight_took_pdl=false"),
    ]:
        r = q1(f"""
            SELECT COUNT(*), COALESCE(ROUND(AVG(o.pnl_r), 4), 0)
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='{sess}' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND d.overnight_took_pdh IS NOT NULL AND d.overnight_took_pdl IS NOT NULL
              AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01' AND {w}
        """)
        print(f"    {sess:<18} {label:<15} N={r[0]:>4}, ExpR={r[1]:+.4f}")

print("\nPART 4: KILL REVIEWS")
print("=" * 70)

print("  Kill review 1: Friday NYSE_CLOSE win size")
for name, filt in [("Friday", "EXTRACT(DOW FROM o.trading_day)=5"), ("Non-Fri", "EXTRACT(DOW FROM o.trading_day)!=5")]:
    wins = [float(r[0]) for r in qr(f"""
        SELECT o.pnl_r FROM orb_outcomes o
        WHERE o.symbol='MNQ' AND o.orb_label='NYSE_CLOSE' AND o.orb_minutes=5
          AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
          AND {filt} AND o.pnl_r > 0 AND o.trading_day<'2026-01-01'
    """)]
    print(f"    {name}: N_wins={len(wins)}, mean={np.mean(wins):.4f}, median={np.median(wins):.4f}, P90={np.percentile(wins,90):.4f}")

print("\n  Kill review 2: Top month x session per-year")
for mo, sess in [(4, "COMEX_SETTLE"), (8, "US_DATA_1000"), (1, "US_DATA_1000")]:
    pos = 0
    for yr in range(2016, 2026):
        r = q1(f"""SELECT AVG(o.pnl_r) FROM orb_outcomes o
            WHERE o.symbol='MNQ' AND o.orb_label='{sess}' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND EXTRACT(MONTH FROM o.trading_day)={mo} AND EXTRACT(YEAR FROM o.trading_day)={yr}""")
        if r[0] is not None and float(r[0]) > 0:
            pos += 1
    print(f"    M{mo} x {sess}: {pos}/10 positive years")

print("\n  Kill review 3: GARCH additive to overnight_range (COMEX)")
for label, w in [
    ("Range_Q5+GARCH_Q5", "r1.bin=5 AND r2.gbin=5"),
    ("Range_Q5 alone", "r1.bin=5"),
    ("GARCH_Q5 alone", "r2.gbin=5"),
]:
    r = q1(f"""
        WITH r1 AS (
            SELECT o.trading_day, o.symbol, o.orb_minutes, o.pnl_r,
                   NTILE(5) OVER (ORDER BY d.overnight_range) as bin
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND d.overnight_range IS NOT NULL AND d.garch_forecast_vol IS NOT NULL
              AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
        ), r2 AS (
            SELECT o.trading_day, o.symbol, o.orb_minutes,
                   NTILE(5) OVER (ORDER BY d.garch_forecast_vol) as gbin
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND d.overnight_range IS NOT NULL AND d.garch_forecast_vol IS NOT NULL
              AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
        )
        SELECT COUNT(*), ROUND(AVG(r1.pnl_r), 4)
        FROM r1 JOIN r2 ON r1.trading_day=r2.trading_day AND r1.symbol=r2.symbol AND r1.orb_minutes=r2.orb_minutes
        WHERE {w}
    """)
    print(f"    {label:<25} N={r[0]:>4}, ExpR={r[1]:+.4f}")

print("\nPART 5: CROSS-SESSION EXTENSIONS")
print("=" * 70)

for sess in ["NYSE_OPEN", "US_DATA_1000", "NYSE_CLOSE", "CME_PRECLOSE"]:
    q1r = q1(f"""
        WITH ranked AS (
            SELECT o.pnl_r, NTILE(5) OVER (ORDER BY d.overnight_range) as bin
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='{sess}' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND d.overnight_range IS NOT NULL AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
        ) SELECT COUNT(*), ROUND(AVG(pnl_r),4) FROM ranked WHERE bin=1
    """)
    q5r = q1(f"""
        WITH ranked AS (
            SELECT o.pnl_r, NTILE(5) OVER (ORDER BY d.overnight_range) as bin
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='{sess}' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND d.overnight_range IS NOT NULL AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
        ) SELECT COUNT(*), ROUND(AVG(pnl_r),4) FROM ranked WHERE bin=5
    """)
    spread = float(q5r[1] or 0) - float(q1r[1] or 0)
    tag = "EXTENDS" if spread > 0.05 else ("WEAK" if spread > 0 else "REVERSES")
    print(f"  overnight_range x {sess:<18} Q1:ExpR={q1r[1]:+.4f} Q5:ExpR={q5r[1]:+.4f} spread={spread:+.4f} [{tag}]")

print()
for sess in ["NYSE_OPEN", "COMEX_SETTLE"]:
    rows = qr(f"""
        SELECT d.overnight_took_pdh as flag, COUNT(*), ROUND(AVG(o.pnl_r), 4)
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='MNQ' AND o.orb_label='{sess}' AND o.orb_minutes=5
          AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
          AND d.overnight_took_pdh IS NOT NULL AND o.pnl_r IS NOT NULL AND o.trading_day<'2026-01-01'
        GROUP BY flag ORDER BY flag
    """)
    if len(rows) == 2:
        gap = float(rows[1][2]) - float(rows[0][2])
        tag = "EXTENDS" if gap > 0.03 else "WEAK"
        print(f"  overnight_took_pdh x {sess:<18} F:ExpR={rows[0][2]:+.4f} T:ExpR={rows[1][2]:+.4f} gap={gap:+.4f} [{tag}]")

con.close()
