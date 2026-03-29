#!/usr/bin/env python3
"""
Backfill and sync paper trades for Apex MNQ lanes into gold.db.

Reads pre-computed outcomes from orb_outcomes x daily_features,
applies each lane's filter gate, and writes to paper_trades table.

Usage:
    python -m trading_app.paper_trade_logger           # Full backfill from 2026-01-01
    python -m trading_app.paper_trade_logger --sync     # Incremental (new days only)
    python -m trading_app.paper_trade_logger --dry-run  # Show what would be inserted
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path

import duckdb

from pipeline.db_config import configure_connection
from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH

logger = get_logger(__name__)

# ── OOS boundary (fail-closed) ───────────────────────────────────────
OOS_START = datetime.date(2026, 1, 1)


@dataclass(frozen=True)
class LaneDef:
    """Definition of one paper-trade lane.

    Lane definitions here mirror prop_profiles.py daily_lanes for the
    apex_150k profile. If lanes change there, update here too.
    """

    strategy_id: str
    lane_name: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str
    filter_sql: str  # WHERE clause fragment referencing daily_features alias


# ── 5 Apex MNQ lanes (source: prop_profiles.py apex_50k_manual.daily_lanes) ──
LANES: tuple[LaneDef, ...] = (
    LaneDef(
        strategy_id="MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15",
        lane_name="NYSE_CLOSE_VOL",
        instrument="MNQ",
        orb_label="NYSE_CLOSE",
        orb_minutes=15,
        rr_target=1.0,
        entry_model="E2",
        confirm_bars=1,
        filter_type="VOL_RV12_N20",
        filter_sql="d.rel_vol_NYSE_CLOSE >= 1.2",
    ),
    LaneDef(
        strategy_id="MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ORB_G8_O15",
        lane_name="SING_G8",
        instrument="MNQ",
        orb_label="SINGAPORE_OPEN",
        orb_minutes=15,
        rr_target=4.0,
        entry_model="E2",
        confirm_bars=1,
        filter_type="ORB_G8",
        filter_sql="d.orb_SINGAPORE_OPEN_size >= 8.0",
    ),
    LaneDef(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL",
        lane_name="COMEX_ATR70",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        orb_minutes=5,
        rr_target=1.0,
        entry_model="E2",
        confirm_bars=1,
        filter_type="ATR70_VOL",
        filter_sql="d.atr_20_pct >= 70 AND d.rel_vol_COMEX_SETTLE >= 1.2",
    ),
    LaneDef(
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15",
        lane_name="NYSE_OPEN_XMES",
        instrument="MNQ",
        orb_label="NYSE_OPEN",
        orb_minutes=15,
        rr_target=1.0,
        entry_model="E2",
        confirm_bars=1,
        filter_type="X_MES_ATR60",
        # Cross-asset: MES atr_20_pct from a separate daily_features row
        filter_sql="d_mes.atr_20_pct >= 60.0",
    ),
    LaneDef(
        strategy_id="MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075",
        lane_name="US_DATA_XMES",
        instrument="MNQ",
        orb_label="US_DATA_1000",
        orb_minutes=5,
        rr_target=1.0,
        entry_model="E2",
        confirm_bars=1,
        filter_type="X_MES_ATR60",
        filter_sql="d_mes.atr_20_pct >= 60.0",
    ),
)


def _validate_lanes() -> None:
    """Assert LANES match canonical apex_50k_manual profile in prop_profiles.py."""
    from trading_app.prop_profiles import ACCOUNT_PROFILES

    expected = {lane.strategy_id for lane in ACCOUNT_PROFILES["apex_50k_manual"].daily_lanes}
    actual = {lane.strategy_id for lane in LANES}
    if actual != expected:
        diff = actual.symmetric_difference(expected)
        raise RuntimeError(f"Lane mismatch vs prop_profiles.py apex_50k_manual: {diff}")


def _is_cross_asset(lane: LaneDef) -> bool:
    return lane.filter_type.startswith("X_")


def _build_query(lane: LaneDef, *, since: datetime.date | None = None) -> tuple[str, list]:
    """Build the SELECT query for one lane. Returns (sql, params)."""
    start = since if since is not None else OOS_START

    if _is_cross_asset(lane):
        # Cross-asset filter: join MES daily_features for atr_20_pct
        sql = f"""
            SELECT
                o.trading_day, o.orb_label, o.entry_ts,
                CASE WHEN o.entry_price > o.stop_price THEN 'long' ELSE 'short' END,
                o.entry_price, o.stop_price, o.target_price,
                o.exit_price, o.exit_ts, o.outcome, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d_mes
                ON d_mes.symbol = 'MES'
                AND d_mes.trading_day = o.trading_day
                AND d_mes.orb_minutes = 5
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.orb_minutes = ?
              AND o.rr_target = ?
              AND o.entry_model = ?
              AND o.confirm_bars = ?
              AND o.outcome IS NOT NULL
              AND {lane.filter_sql}
              AND o.trading_day >= ?
            ORDER BY o.trading_day
        """
        params = [
            lane.instrument,
            lane.orb_label,
            lane.orb_minutes,
            lane.rr_target,
            lane.entry_model,
            lane.confirm_bars,
            start,
        ]
    else:
        # Same-instrument filter: join own daily_features at matching aperture
        sql = f"""
            SELECT
                o.trading_day, o.orb_label, o.entry_ts,
                CASE WHEN o.entry_price > o.stop_price THEN 'long' ELSE 'short' END,
                o.entry_price, o.stop_price, o.target_price,
                o.exit_price, o.exit_ts, o.outcome, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d
                ON d.symbol = o.symbol
                AND d.trading_day = o.trading_day
                AND d.orb_minutes = ?
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.orb_minutes = ?
              AND o.rr_target = ?
              AND o.entry_model = ?
              AND o.confirm_bars = ?
              AND o.outcome IS NOT NULL
              AND {lane.filter_sql}
              AND o.trading_day >= ?
            ORDER BY o.trading_day
        """
        params = [
            lane.orb_minutes,  # daily_features JOIN
            lane.instrument,
            lane.orb_label,
            lane.orb_minutes,
            lane.rr_target,
            lane.entry_model,
            lane.confirm_bars,
            start,
        ]

    return sql, params


def backfill(
    db_path: Path | str | None = None,
    sync: bool = False,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Backfill paper_trades from orb_outcomes x daily_features.

    Returns dict of lane_name -> {count, cumulative_r, min_day, max_day}.
    """
    _validate_lanes()
    path = str(db_path) if db_path else str(GOLD_DB_PATH)
    results: dict[str, dict] = {}

    # Ensure table exists BEFORE opening backfill connection (no dual writes)
    from trading_app.db_manager import init_trading_app_schema

    init_trading_app_schema(db_path=Path(path))

    with duckdb.connect(path) as con:
        configure_connection(con, writing=True)

        for lane in LANES:
            since = None
            if sync:
                row = con.execute(
                    "SELECT MAX(trading_day) FROM paper_trades WHERE strategy_id = ?",
                    [lane.strategy_id],
                ).fetchone()
                if row and row[0] is not None:
                    since = row[0] + datetime.timedelta(days=1)

            sql, params = _build_query(lane, since=since)
            rows = con.execute(sql, params).fetchall()

            # ── Fail-closed OOS boundary assertion ────────────────────
            for r in rows:
                trading_day = r[0]
                if isinstance(trading_day, datetime.datetime):
                    trading_day = trading_day.date()
                if trading_day < OOS_START:
                    raise ValueError(
                        f"OOS BOUNDARY VIOLATION: {lane.strategy_id} row "
                        f"trading_day={trading_day} < {OOS_START}. Aborting."
                    )

            if dry_run:
                total_r = sum(r[10] for r in rows if r[10] is not None)
                logger.info(f"[DRY RUN] {lane.lane_name}: {len(rows)} trades, cumulative R={total_r:+.2f}")
                results[lane.lane_name] = {
                    "count": len(rows),
                    "cumulative_r": round(total_r, 4),
                    "min_day": str(rows[0][0]) if rows else None,
                    "max_day": str(rows[-1][0]) if rows else None,
                }
                continue

            # Idempotent DELETE+INSERT
            if sync and since is not None:
                con.execute(
                    "DELETE FROM paper_trades WHERE strategy_id = ? AND trading_day >= ?",
                    [lane.strategy_id, since],
                )
            elif not sync:
                con.execute(
                    "DELETE FROM paper_trades WHERE strategy_id = ?",
                    [lane.strategy_id],
                )

            if rows:
                insert_data = [
                    (
                        r[0],
                        r[1],
                        r[2],
                        r[3],
                        r[4],
                        r[5],
                        r[6],
                        r[7],
                        r[8],
                        r[9],
                        r[10],
                        lane.strategy_id,
                        lane.lane_name,
                        lane.instrument,
                        lane.orb_minutes,
                        lane.rr_target,
                        lane.filter_type,
                        lane.entry_model,
                    )
                    for r in rows
                ]
                con.executemany(
                    """
                    INSERT INTO paper_trades (
                        trading_day, orb_label, entry_time, direction,
                        entry_price, stop_price, target_price, exit_price,
                        exit_time, exit_reason, pnl_r,
                        strategy_id, lane_name, instrument,
                        orb_minutes, rr_target, filter_type, entry_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_data,
                )

            con.commit()

            summary = con.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(pnl_r), 0),
                       MIN(trading_day), MAX(trading_day)
                FROM paper_trades WHERE strategy_id = ?
                """,
                [lane.strategy_id],
            ).fetchone()

            results[lane.lane_name] = {
                "count": summary[0],
                "cumulative_r": round(summary[1], 4),
                "min_day": str(summary[2]) if summary[2] else None,
                "max_day": str(summary[3]) if summary[3] else None,
            }

            action = "synced" if sync else "backfilled"
            logger.info(f"{lane.lane_name}: {action} {len(rows)} new rows (total {summary[0]}, cumR={summary[1]:+.2f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Paper trade logger")
    parser.add_argument("--sync", action="store_true", help="Incremental sync (new days only)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted")
    args = parser.parse_args()

    results = backfill(sync=args.sync, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("PAPER TRADE BACKFILL SUMMARY")
    print("=" * 60)
    total_trades = 0
    total_r = 0.0
    for lane_name, info in results.items():
        n = info["count"]
        cr = info["cumulative_r"]
        total_trades += n
        total_r += cr
        print(f"  {lane_name:<20s}  trades={n:3d}  cumR={cr:+.2f}  ({info['min_day']} to {info['max_day']})")
    print(f"  {'TOTAL':<20s}  trades={total_trades:3d}  cumR={total_r:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
