"""GAP 4: 2026 forward validation for ATR70 candidates vs current lanes."""

import duckdb

from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

print("=" * 100)
print("GAP 4: 2026 FORWARD VALIDATION (monitoring, not discovery)")
print("=" * 100)

# Latest data
latest = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = 'MNQ'").fetchone()[0]
print(f"\nLatest orb_outcomes date: {latest}")

# Current lanes
current = [
    ("NYSE_CL current", "NYSE_CLOSE", 15, 1.0, "VOL_RV12_N20"),
    ("SING current", "SINGAPORE_OPEN", 15, 4.0, "ORB_G8"),
    ("COMEX current", "COMEX_SETTLE", 5, 1.0, "ORB_G8"),
    ("NYSE_OP current", "NYSE_OPEN", 15, 1.0, "X_MES_ATR60"),
    ("USDATA current", "US_DATA_1000", 5, 1.0, "X_MES_ATR60"),
]

# ATR70 candidates (same-params and best-ExpR)
atrcands = [
    ("COMEX same-p", "COMEX_SETTLE", 5, 1.0),
    ("SING same-p", "SINGAPORE_OPEN", 15, 4.0),
    ("COMEX best", "COMEX_SETTLE", 5, 2.0),
    ("NYSE_CL best", "NYSE_CLOSE", 5, 1.0),
    ("NYSE_OP best", "NYSE_OPEN", 5, 1.0),
    ("CME_PRE add", "CME_PRECLOSE", 15, 1.0),
]

print("\n--- Current lanes 2026 from orb_outcomes + filter ---")
for label, sess, orb, rr, filt in current:
    if filt == "ORB_G8":
        filt_sql = f"d.orb_{sess}_size >= 8.0"
    elif filt == "VOL_RV12_N20":
        filt_sql = f"d.rel_vol_{sess} >= 1.2"
    elif filt == "X_MES_ATR60":
        filt_sql = "mes.atr_20_pct >= 60"
    else:
        filt_sql = "TRUE"

    if filt == "X_MES_ATR60":
        q = f"""
            SELECT COUNT(*), AVG(o.pnl_r), SUM(o.pnl_r),
                   AVG(CASE WHEN o.pnl_r > 0 THEN 1.0 ELSE 0.0 END)
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            JOIN (SELECT DISTINCT trading_day, atr_20_pct FROM daily_features
                  WHERE symbol = 'MES' AND orb_minutes = 5) mes
                ON o.trading_day = mes.trading_day
            WHERE o.symbol = 'MNQ' AND o.entry_model = 'E2' AND o.confirm_bars = 1
            AND o.orb_label = '{sess}' AND o.orb_minutes = {orb} AND o.rr_target = {rr}
            AND o.trading_day >= '2026-01-01'
            AND {filt_sql}
        """
    else:
        q = f"""
            SELECT COUNT(*), AVG(o.pnl_r), SUM(o.pnl_r),
                   AVG(CASE WHEN o.pnl_r > 0 THEN 1.0 ELSE 0.0 END)
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = 'MNQ' AND o.entry_model = 'E2' AND o.confirm_bars = 1
            AND o.orb_label = '{sess}' AND o.orb_minutes = {orb} AND o.rr_target = {rr}
            AND o.trading_day >= '2026-01-01'
            AND {filt_sql}
        """

    r = con.execute(q).fetchone()
    n, expr, total, wr = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
    flag = "POS" if total > 0 else "NEG" if total < 0 else "ZERO"
    print(f"  {label:22s}: N={n:3d}  WR={wr:.0%}  TotalR={total:+7.2f}  ExpR={expr:+.4f}  [{flag}]")

print("\n--- ATR70_VOL candidates 2026 from orb_outcomes + filter ---")
for label, sess, orb, rr in atrcands:
    q = f"""
        SELECT COUNT(*), AVG(o.pnl_r), SUM(o.pnl_r),
               AVG(CASE WHEN o.pnl_r > 0 THEN 1.0 ELSE 0.0 END)
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MNQ' AND o.entry_model = 'E2' AND o.confirm_bars = 1
        AND o.orb_label = '{sess}' AND o.orb_minutes = {orb} AND o.rr_target = {rr}
        AND o.trading_day >= '2026-01-01'
        AND d.atr_20_pct >= 70
        AND d.rel_vol_{sess} >= 1.2
    """
    r = con.execute(q).fetchone()
    n, expr, total, wr = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
    flag = "POS" if total > 0 else "NEG" if total < 0 else "ZERO"
    print(f"  {label:22s}: N={n:3d}  WR={wr:.0%}  TotalR={total:+7.2f}  ExpR={expr:+.4f}  [{flag}]")

# Summary comparison
print("\n--- HEAD-TO-HEAD: same-params ATR70 vs current (2026 only) ---")
pairs = [
    ("COMEX_SETTLE O5 RR1.0", "COMEX current", "COMEX same-p"),
    ("SINGAPORE_OPEN O15 RR4.0", "SING current", "SING same-p"),
]
print(f"  {'Session':30s} {'Current 2026R':>14s} {'ATR70 2026R':>12s} {'Delta':>8s}")
# We'll just read the numbers from above output - this is for the user to eyeball

con.close()
