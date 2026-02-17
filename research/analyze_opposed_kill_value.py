"""
Opposed Kill Insurance Value Analysis (1000 session).

For every 1000 trade where IB breaks OPPOSED to trade direction:
1. Find trades that eventually get STOPPED OUT (loss = -1R)
2. Check: was the stop price BEYOND the opposite IB level?
3. Compare: "theoretical loss" at stop (-1R) vs mark-to-market at IB break bar

Report:
- Frequency: % of opposed losing trades where stop > IB range
- Savings: total R-units saved by killing at IB break vs waiting for stop
- If savings > 0R: opposed kill is active alpha
- If savings ~ 0R: opposed kill is regime insurance
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from datetime import date, datetime, timedelta, timezone

from pipeline.paths import GOLD_DB_PATH


def analyze_opposed_kill(db_path: Path = GOLD_DB_PATH, instrument: str = "MGC"):
    """Analyze 1000 session opposed IB breaks vs stop-loss outcomes."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Get all 1000 session trading days with ORB data
        rows = con.execute("""
            SELECT trading_day,
                   orb_1000_high, orb_1000_low, orb_1000_size,
                   orb_1000_break_dir, orb_1000_outcome,
                   orb_0900_high, orb_0900_low
            FROM daily_features
            WHERE symbol = ?
              AND orb_minutes = 5
              AND orb_1000_break_dir IS NOT NULL
              AND orb_0900_high IS NOT NULL
              AND orb_0900_low IS NOT NULL
            ORDER BY trading_day
        """, [instrument]).fetchall()

        print(f"1000 session break days with 0900 IB data: {len(rows)}")
        print()

        # IB = 0900 to 1100 Brisbane = 23:00 to 01:00 UTC (120 minutes)
        # For each 1000 break, determine IB range from bars_1m
        # IB period: 23:00 UTC to 01:00 UTC (120 min from 0900 Brisbane)

        total_opposed = 0
        opposed_stopped = 0
        opposed_stop_beyond_ib = 0
        total_r_saved = 0.0
        r_savings_list = []

        for row in rows:
            td = row[0]
            orb_high = row[1]
            orb_low = row[2]
            orb_size = row[3]
            break_dir = row[4]
            outcome = row[5]

            if orb_size is None or orb_size <= 0:
                continue

            # Compute IB range from bars_1m (23:00 UTC to 01:00 UTC)
            prev_day = td - timedelta(days=1)
            ib_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                                23, 0, tzinfo=timezone.utc)
            ib_end = datetime(td.year, td.month, td.day,
                              1, 0, tzinfo=timezone.utc)

            ib_bars = con.execute("""
                SELECT high, low FROM bars_1m
                WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
            """, [instrument, ib_start, ib_end]).fetchall()

            if len(ib_bars) < 60:  # Need substantial IB data
                continue

            ib_high = max(b[0] for b in ib_bars)
            ib_low = min(b[1] for b in ib_bars)

            # Determine IB break direction
            # After IB (01:00 UTC), find first bar that breaks IB high or low
            post_ib_start = ib_end
            # Look up to 07:00 UTC (end of Asia session)
            post_ib_end = datetime(td.year, td.month, td.day,
                                   7, 0, tzinfo=timezone.utc)

            post_ib_bars = con.execute("""
                SELECT ts_utc, open, high, low, close FROM bars_1m
                WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
                ORDER BY ts_utc
            """, [instrument, post_ib_start, post_ib_end]).fetchall()

            ib_break_dir = None
            ib_break_bar = None
            for bar in post_ib_bars:
                if bar[4] > ib_high:  # close > IB high
                    ib_break_dir = "long"
                    ib_break_bar = bar
                    break
                elif bar[4] < ib_low:  # close < IB low
                    ib_break_dir = "short"
                    ib_break_bar = bar
                    break

            if ib_break_dir is None:
                continue  # IB never broke

            # Is IB opposed to ORB break?
            is_opposed = (break_dir != ib_break_dir)
            if not is_opposed:
                continue

            total_opposed += 1

            # For opposed trades: simulate E1 entry
            # Entry = next bar open after ORB break (approximate: use ORB break price)
            # Stop = opposite ORB level
            if break_dir == "long":
                stop_price = orb_low
                entry_price = orb_high  # approximate
                risk_points = entry_price - stop_price
            else:
                stop_price = orb_high
                entry_price = orb_low  # approximate
                risk_points = stop_price - entry_price

            if risk_points <= 0:
                continue

            # Did this trade get stopped out? (outcome contains loss info)
            # Use orb_outcomes table for precise result
            outcome_rows = con.execute("""
                SELECT pnl_r, entry_price, stop_price, target_price
                FROM orb_outcomes
                WHERE symbol = ?
                  AND trading_day = ?
                  AND orb_label = '1000'
                  AND entry_model = 'E1'
                  AND confirm_bars = 1
                  AND rr_target = 2.5
                LIMIT 1
            """, [instrument, td]).fetchall()

            if not outcome_rows:
                continue

            actual_pnl_r = outcome_rows[0][0]
            actual_entry = outcome_rows[0][1]
            actual_stop = outcome_rows[0][2]

            if actual_pnl_r is None:
                continue

            is_loss = actual_pnl_r < 0

            if is_loss:
                opposed_stopped += 1

            # Check if stop was beyond IB level
            if break_dir == "long":
                stop_beyond_ib = actual_stop < ib_low
            else:
                stop_beyond_ib = actual_stop > ib_high

            if is_loss and stop_beyond_ib:
                opposed_stop_beyond_ib += 1

            # Compute mark-to-market at IB break bar
            if ib_break_bar is not None and actual_entry is not None:
                actual_risk = abs(actual_entry - actual_stop) if actual_stop else risk_points
                if actual_risk > 0:
                    if break_dir == "long":
                        mtm_points = ib_break_bar[4] - actual_entry  # close - entry
                    else:
                        mtm_points = actual_entry - ib_break_bar[4]

                    mtm_r = mtm_points / actual_risk

                    # R saved = actual loss - what we'd lose at IB kill
                    # If actual_pnl_r = -1.1 and mtm_r = -0.3, we saved 0.8R
                    if is_loss:
                        r_saved = actual_pnl_r - mtm_r  # negative - negative = positive if mtm less bad
                        # Wait, actual_pnl_r is negative (loss), mtm_r could be anything
                        # Savings = mtm_r - actual_pnl_r (how much better the early exit is)
                        r_saved = mtm_r - actual_pnl_r
                        total_r_saved += r_saved
                        r_savings_list.append({
                            "date": td,
                            "break_dir": break_dir,
                            "ib_break_dir": ib_break_dir,
                            "actual_pnl_r": actual_pnl_r,
                            "mtm_at_ib_break_r": round(mtm_r, 4),
                            "r_saved": round(r_saved, 4),
                            "stop_beyond_ib": stop_beyond_ib,
                        })

        # Report
        print("=" * 60)
        print("OPPOSED KILL INSURANCE VALUE ANALYSIS")
        print("=" * 60)
        print(f"Total opposed IB breaks (1000 session): {total_opposed}")
        print(f"Opposed trades that got stopped out: {opposed_stopped}")
        print(f"  Of which stop was beyond IB range: {opposed_stop_beyond_ib}")
        print()

        if total_opposed > 0:
            print(f"Opposed frequency: {total_opposed} days")
            print(f"Loss rate when opposed: {opposed_stopped}/{total_opposed} = "
                  f"{opposed_stopped/total_opposed:.1%}")

        if r_savings_list:
            avg_savings = total_r_saved / len(r_savings_list)
            positive_saves = [s for s in r_savings_list if s["r_saved"] > 0]
            negative_saves = [s for s in r_savings_list if s["r_saved"] < 0]

            print(f"\nR-unit savings from IB kill vs waiting for stop:")
            print(f"  Total R saved: {total_r_saved:+.2f}R across {len(r_savings_list)} stopped trades")
            print(f"  Average R saved per stopped trade: {avg_savings:+.4f}R")
            print(f"  Trades where IB kill saved R: {len(positive_saves)}")
            print(f"  Trades where IB kill cost R: {len(negative_saves)}")
            print()

            if avg_savings > 0.05:
                print("VERDICT: ACTIVE ALPHA -- IB opposed kill saves meaningful R-units")
                print(f"  The stop is typically further away than the IB level,")
                print(f"  so killing at IB break gives a structurally earlier exit.")
            elif avg_savings > -0.05:
                print("VERDICT: REGIME INSURANCE -- IB opposed kill is ~breakeven on R")
                print(f"  The stops are already tight enough that IB kill is mostly redundant,")
                print(f"  but it prevents catastrophic loss in regime shifts.")
            else:
                print("VERDICT: NEUTRAL/NEGATIVE -- IB opposed kill may be counterproductive")
                print(f"  Some opposed trades recover. Consider keeping fixed target.")

            # Show worst and best saves
            if r_savings_list:
                r_savings_list.sort(key=lambda x: x["r_saved"])
                print(f"\nWorst 5 (IB kill cost R):")
                for s in r_savings_list[:5]:
                    print(f"  {s['date']}: {s['break_dir']} trade, IB broke {s['ib_break_dir']}, "
                          f"actual={s['actual_pnl_r']:+.3f}R, mtm_at_ib={s['mtm_at_ib_break_r']:+.3f}R, "
                          f"saved={s['r_saved']:+.3f}R")

                print(f"\nBest 5 (IB kill saved R):")
                for s in r_savings_list[-5:]:
                    print(f"  {s['date']}: {s['break_dir']} trade, IB broke {s['ib_break_dir']}, "
                          f"actual={s['actual_pnl_r']:+.3f}R, mtm_at_ib={s['mtm_at_ib_break_r']:+.3f}R, "
                          f"saved={s['r_saved']:+.3f}R")

        else:
            print("No opposed+stopped trades found to analyze savings.")

    finally:
        con.close()


if __name__ == "__main__":
    analyze_opposed_kill()
