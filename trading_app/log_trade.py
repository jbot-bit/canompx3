"""Manual trade entry CLI for Phase 1 forward data collection.

Records live trades to paper_trades table with execution_source='live'.
Computes pnl_r from entry/exit prices and lane parameters.
Captures actual slippage for the MNQ slippage pilot.

Usage:
    python -m trading_app.log_trade --session NYSE_OPEN --direction long \
      --entry 22150.25 --exit 22162.50 --slippage-entry 1 --slippage-exit 0 \
      --notes "clean fill"

    python -m trading_app.log_trade --session COMEX_SETTLE --direction short \
      --entry 22100.00 --stop-hit --slippage-entry 2 --notes "gap through stop"
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cost_model import COST_SPECS
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH

LANE_DEFS = {
    "NYSE_CLOSE": {
        "strategy_id": "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15",
        "lane_name": "NYSE_CLOSE_VOL",
        "instrument": "MNQ",
        "filter_type": "VOL_RV12_N20",
        "rr_target": 1.0,
        "orb_minutes": 15,
        "entry_model": "E2",
    },
    "SINGAPORE_OPEN": {
        "strategy_id": "MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ORB_G8_O15",
        "lane_name": "SING_G8",
        "instrument": "MNQ",
        "filter_type": "ORB_G8",
        "rr_target": 4.0,
        "orb_minutes": 15,
        "entry_model": "E2",
    },
    "COMEX_SETTLE": {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8",
        "lane_name": "COMEX_G8",
        "instrument": "MNQ",
        "filter_type": "ORB_G8",
        "rr_target": 1.0,
        "orb_minutes": 5,
        "entry_model": "E2",
    },
    "NYSE_OPEN": {
        "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15",
        "lane_name": "NYSE_OPEN_XMES",
        "instrument": "MNQ",
        "filter_type": "X_MES_ATR60",
        "rr_target": 1.0,
        "orb_minutes": 15,
        "entry_model": "E2",
    },
    "TOKYO_OPEN": {
        "strategy_id": "MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075",
        "lane_name": "MGC_TOKYO",
        "instrument": "MGC",
        "filter_type": "ORB_G4_CONT",
        "rr_target": 2.0,
        "orb_minutes": 5,
        "entry_model": "E2",
    },
}


def compute_pnl_r(direction: str, entry: float, exit_price: float, stop: float) -> float:
    """Compute pnl_r = (exit - entry) / risk, adjusted for direction."""
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    if direction == "long":
        return (exit_price - entry) / risk
    else:
        return (entry - exit_price) / risk


def log_trade(args):
    session = args.session
    if session not in LANE_DEFS:
        print(f"ERROR: Unknown session '{session}'")
        sys.exit(1)

    lane = LANE_DEFS[session]
    instrument = lane["instrument"]
    cost = COST_SPECS[instrument]
    today = date.today()

    direction = args.direction
    entry_price = args.entry
    slippage_entry = args.slippage_entry or 0
    slippage_exit = args.slippage_exit or 0
    notes = args.notes or ""

    # Get today's ORB for stop/target computation
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    orb_row = con.execute(
        f"""SELECT orb_{session}_high, orb_{session}_low, orb_{session}_size
            FROM daily_features
            WHERE symbol = ? AND trading_day = ? AND orb_minutes = ?""",
        [instrument, today, lane["orb_minutes"]],
    ).fetchone()

    if not orb_row:
        print(f"WARNING: No ORB data for {instrument} {session} O{lane['orb_minutes']} on {today}")
        print("Enter stop and target manually.")
        stop_price = float(input("Stop price: "))
        target_price = float(input("Target price: "))
    else:
        orb_high, orb_low, orb_size = orb_row
        risk = orb_size if orb_size else abs(orb_high - orb_low)
        if direction == "long":
            stop_price = orb_low
            target_price = entry_price + (risk * lane["rr_target"])
        else:
            stop_price = orb_high
            target_price = entry_price - (risk * lane["rr_target"])

    # Determine exit
    if args.stop_hit:
        exit_price = stop_price
        exit_reason = "stop"
    elif args.target_hit:
        exit_price = target_price
        exit_reason = "target"
    elif args.exit is not None:
        exit_price = args.exit
        exit_reason = "manual" if args.exit != target_price else "target"
    else:
        print("ERROR: Must specify --exit, --stop-hit, or --target-hit")
        con.close()
        sys.exit(1)

    # Compute P&L
    pnl_r = compute_pnl_r(direction, entry_price, exit_price, stop_price)
    risk_dollars = abs(entry_price - stop_price) * cost.point_value
    pnl_dollar = pnl_r * risk_dollars

    now = datetime.now(timezone.utc)

    # Write to DB
    con_w = duckdb.connect(str(GOLD_DB_PATH))
    configure_connection(con_w, writing=True)
    con_w.execute(
        """INSERT INTO paper_trades
           (trading_day, orb_label, entry_time, direction, entry_price, stop_price,
            target_price, exit_price, exit_time, exit_reason, pnl_r, slippage_ticks,
            strategy_id, lane_name, instrument, orb_minutes, rr_target, filter_type,
            entry_model, execution_source, pnl_dollar, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            today,
            session,
            now.isoformat(),
            direction,
            entry_price,
            stop_price,
            target_price,
            exit_price,
            now.isoformat(),
            exit_reason,
            round(pnl_r, 4),
            slippage_entry,
            lane["strategy_id"],
            lane["lane_name"],
            instrument,
            lane["orb_minutes"],
            lane["rr_target"],
            lane["filter_type"],
            lane["entry_model"],
            "live",
            round(pnl_dollar, 2),
            notes,
        ],
    )

    # Get updated stats
    stats = con_w.execute(
        """SELECT COUNT(*) as n, ROUND(SUM(pnl_r), 2) as cum_r,
                  ROUND(AVG(CASE WHEN pnl_r > 0 THEN 1.0 ELSE 0.0 END)*100, 1) as wr
           FROM paper_trades WHERE orb_label = ?""",
        [session],
    ).fetchone()

    live_n = con_w.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE execution_source = 'live' AND slippage_ticks IS NOT NULL"
    ).fetchone()[0]

    con_w.close()
    con.close()

    # Print confirmation
    outcome = "WIN" if pnl_r > 0 else ("LOSS" if pnl_r < 0 else "SCRATCH")
    print(f"\n{'=' * 60}")
    print(f"TRADE LOGGED: {session} | {direction.upper()} | {outcome}")
    print(f"  Entry: {entry_price} | Exit: {exit_price} | Stop: {stop_price}")
    print(f"  PnL: {pnl_r:+.4f}R = ${pnl_dollar:+.2f}")
    print(f"  Slippage: entry={slippage_entry} ticks, exit={slippage_exit} ticks")
    print(f"{'=' * 60}")
    print(f"Lane stats: N={stats[0]}, CumR={stats[1]:+.2f}, WR={stats[2]}%")
    print(f"Slippage pilot: {live_n}/30 live trades recorded")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Log a manual trade to paper_trades")
    parser.add_argument("--session", required=True, choices=list(LANE_DEFS.keys()))
    parser.add_argument("--direction", required=True, choices=["long", "short"])
    parser.add_argument("--entry", required=True, type=float, help="Entry fill price")
    parser.add_argument("--exit", type=float, help="Exit fill price")
    parser.add_argument("--stop-hit", action="store_true", help="Exited at stop")
    parser.add_argument("--target-hit", action="store_true", help="Exited at target")
    parser.add_argument("--slippage-entry", type=int, default=0, help="Entry slippage in ticks")
    parser.add_argument("--slippage-exit", type=int, default=0, help="Exit slippage in ticks")
    parser.add_argument("--notes", type=str, default="", help="Trade notes")
    args = parser.parse_args()
    log_trade(args)


if __name__ == "__main__":
    main()
