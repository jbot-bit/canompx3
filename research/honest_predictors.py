"""
Honest predictor scan: find PRIOR data that predicts today's 1100 outcome.
No look-ahead bias — every predictor is known BEFORE 1100 entry time.
"""
import duckdb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

BASE = """
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MGC' AND o.orb_label = '1100'
      AND o.entry_model = 'E1' AND o.rr_target = 2.5 AND o.confirm_bars = 2
      AND d.orb_1100_size >= 4.0
      AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
"""

def run_test(name, select_expr, extra_where="", group_col="pred"):
    print(f"\n--- {name} ---")
    q = f"""
        SELECT {select_expr} as pred,
            COUNT(*) as n, AVG(o.pnl_r) as expr,
            SUM(o.pnl_r) as totr,
            SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
        {BASE}
        {extra_where}
        GROUP BY 1 ORDER BY 1
    """
    rows = con.execute(q).fetchall()
    for row in rows:
        tag = " ***" if row[1] >= 50 and row[2] > 0.3 else ""
        print(f"  {str(row[0]):<20} N={row[1]:<6} ExpR={row[2]:>+.4f}  TotalR={row[3]:>+.1f}  WR={row[4]:.1f}%{tag}")


print("=" * 70)
print("HONEST PREDICTORS OF 1100 TRADE QUALITY (NO LOOK-AHEAD)")
print("All predictors known BEFORE 1100 entry time.")
print("=" * 70)

# 1. 0900 double-break (happens before 1100)
run_test("0900 double-break -> 1100 outcome",
         "d.orb_0900_double_break",
         "AND d.orb_0900_double_break IS NOT NULL")

# 2. 1000 double-break (happens before 1100)
run_test("1000 double-break -> 1100 outcome",
         "d.orb_1000_double_break",
         "AND d.orb_1000_double_break IS NOT NULL")

# 3. 0900 outcome
run_test("0900 outcome -> 1100 outcome",
         "d.orb_0900_outcome",
         "AND d.orb_0900_outcome IS NOT NULL")

# 4. 1000 outcome
run_test("1000 outcome -> 1100 outcome",
         "d.orb_1000_outcome",
         "AND d.orb_1000_outcome IS NOT NULL")

# 5. Yesterday's 1100 double-break
print("\n--- YESTERDAY 1100 dbl-break -> TODAY 1100 outcome ---")
q = f"""
    WITH lagged AS (
        SELECT d.trading_day, d.symbol,
            LAG(d.orb_1100_double_break) OVER (PARTITION BY d.symbol ORDER BY d.trading_day) as prev_dbl
        FROM daily_features d WHERE d.symbol = 'MGC' AND d.orb_minutes = 5
    )
    SELECT l.prev_dbl as pred,
        COUNT(*) as n, AVG(o.pnl_r) as expr,
        SUM(o.pnl_r) as totr,
        SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
    {BASE}
    AND l.prev_dbl IS NOT NULL
    GROUP BY 1 ORDER BY 1
""".replace(BASE, f"""
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    JOIN lagged l ON o.trading_day = l.trading_day AND o.symbol = l.symbol
    WHERE o.symbol = 'MGC' AND o.orb_label = '1100'
      AND o.entry_model = 'E1' AND o.rr_target = 2.5 AND o.confirm_bars = 2
      AND d.orb_1100_size >= 4.0
      AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
""")
# Rewrite properly
rows = con.execute("""
    WITH lagged AS (
        SELECT d.trading_day, d.symbol,
            LAG(d.orb_1100_double_break) OVER (PARTITION BY d.symbol ORDER BY d.trading_day) as prev_dbl
        FROM daily_features d WHERE d.symbol = 'MGC' AND d.orb_minutes = 5
    )
    SELECT l.prev_dbl as pred,
        COUNT(*) as n, AVG(o.pnl_r) as expr,
        SUM(o.pnl_r) as totr,
        SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    JOIN lagged l ON o.trading_day = l.trading_day AND o.symbol = l.symbol
    WHERE o.symbol = 'MGC' AND o.orb_label = '1100'
      AND o.entry_model = 'E1' AND o.rr_target = 2.5 AND o.confirm_bars = 2
      AND d.orb_1100_size >= 4.0
      AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
      AND l.prev_dbl IS NOT NULL
    GROUP BY 1 ORDER BY 1
""").fetchall()
for row in rows:
    lbl = "NO prev dbl" if row[0] == False else "YES prev dbl"
    tag = " ***" if row[1] >= 50 and row[2] > 0.3 else ""
    print(f"  {lbl:<20} N={row[1]:<6} ExpR={row[2]:>+.4f}  TotalR={row[3]:>+.1f}  WR={row[4]:.1f}%{tag}")

# 6. 0900 ORB size buckets
run_test("0900 ORB size -> 1100 outcome",
         """CASE WHEN d.orb_0900_size < 2 THEN 'a_tiny(<2)'
            WHEN d.orb_0900_size < 4 THEN 'b_small(2-4)'
            WHEN d.orb_0900_size < 8 THEN 'c_med(4-8)'
            ELSE 'd_large(8+)' END""",
         "AND d.orb_0900_size IS NOT NULL")

# 7. ATR/ORB ratio
run_test("ATR20 / 1100_ORB ratio -> 1100 outcome",
         """CASE WHEN d.atr_20 / NULLIF(d.orb_1100_size, 0) < 2 THEN 'a_ratio<2'
            WHEN d.atr_20 / NULLIF(d.orb_1100_size, 0) < 4 THEN 'b_ratio2-4'
            WHEN d.atr_20 / NULLIF(d.orb_1100_size, 0) < 8 THEN 'c_ratio4-8'
            ELSE 'd_ratio8+' END""",
         "AND d.atr_20 IS NOT NULL")

# 8. RSI at 0900
run_test("RSI14 at 0900 -> 1100 outcome",
         """CASE WHEN d.rsi_14_at_CME_REOPEN < 30 THEN 'a_oversold(<30)'
            WHEN d.rsi_14_at_CME_REOPEN < 50 THEN 'b_bearish(30-50)'
            WHEN d.rsi_14_at_CME_REOPEN < 70 THEN 'c_bullish(50-70)'
            ELSE 'd_overbought(70+)' END""",
         "AND d.rsi_14_at_CME_REOPEN IS NOT NULL")

# 9. Gap open
run_test("Gap open -> 1100 outcome",
         """CASE WHEN d.gap_open_points < -2 THEN 'a_gap_dn_big'
            WHEN d.gap_open_points < 0 THEN 'b_gap_dn_sm'
            WHEN d.gap_open_points < 2 THEN 'c_gap_up_sm'
            ELSE 'd_gap_up_big' END""",
         "AND d.gap_open_points IS NOT NULL")

# 10. Combo: both 0900+1000 no-dbl-break
run_test("BOTH 0900+1000 regime -> 1100 outcome",
         """CASE WHEN d.orb_0900_double_break = false AND d.orb_1000_double_break = false THEN 'a_both_clean'
            WHEN d.orb_0900_double_break = false OR d.orb_1000_double_break = false THEN 'b_one_clean'
            ELSE 'c_both_dbl' END""",
         "AND d.orb_0900_double_break IS NOT NULL AND d.orb_1000_double_break IS NOT NULL")

# 11. 1100 ORB size relative to 0900 ORB size (both known before entry)
run_test("1100/0900 ORB size ratio -> 1100 outcome",
         """CASE WHEN d.orb_1100_size / NULLIF(d.orb_0900_size, 0) < 0.5 THEN 'a_1100_much_smaller'
            WHEN d.orb_1100_size / NULLIF(d.orb_0900_size, 0) < 1.0 THEN 'b_1100_smaller'
            WHEN d.orb_1100_size / NULLIF(d.orb_0900_size, 0) < 2.0 THEN 'c_1100_bigger'
            ELSE 'd_1100_much_bigger' END""",
         "AND d.orb_0900_size IS NOT NULL AND d.orb_0900_size > 0")

# 12. Prior day range / ATR (volatility persistence)
run_test("Prior-day range/ATR -> 1100 outcome",
         """CASE WHEN (d.daily_high - d.daily_low) / NULLIF(d.atr_20, 0) < 0.5 THEN 'a_quiet(<0.5)'
            WHEN (d.daily_high - d.daily_low) / NULLIF(d.atr_20, 0) < 1.0 THEN 'b_normal(0.5-1)'
            WHEN (d.daily_high - d.daily_low) / NULLIF(d.atr_20, 0) < 1.5 THEN 'c_active(1-1.5)'
            ELSE 'd_volatile(1.5+)' END""",
         "AND d.atr_20 IS NOT NULL AND d.atr_20 > 0")

# --- Now expand to ALL sessions, not just 1100 ---
print("\n" + "=" * 70)
print("CROSS-SESSION: 0900/1000 regime predicting later sessions")
print("=" * 70)

for target_sess in ['1100', '1800', 'LONDON_OPEN', '2300', '0030']:
    size_col = f"orb_{target_sess}_size"
    print(f"\n--- 0900+1000 regime -> {target_sess} outcome ---")
    rows = con.execute(f"""
        SELECT
            CASE WHEN d.orb_0900_double_break = false AND d.orb_1000_double_break = false THEN 'a_both_clean'
                WHEN d.orb_0900_double_break = false OR d.orb_1000_double_break = false THEN 'b_one_clean'
                ELSE 'c_both_dbl' END as regime,
            COUNT(*) as n, AVG(o.pnl_r) as expr,
            SUM(o.pnl_r) as totr,
            SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MGC' AND o.orb_label = '{target_sess}'
          AND o.entry_model = 'E1' AND o.rr_target = 2.5 AND o.confirm_bars = 2
          AND d.{size_col} >= 4.0
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          AND d.orb_0900_double_break IS NOT NULL
          AND d.orb_1000_double_break IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    for row in rows:
        tag = " ***" if row[1] >= 50 and row[2] > 0.3 else ""
        print(f"  {row[0]:<20} N={row[1]:<6} ExpR={row[2]:>+.4f}  TotalR={row[3]:>+.1f}  WR={row[4]:.1f}%{tag}")

con.close()
print("\nDone.")
