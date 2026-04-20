#!/usr/bin/env python3
"""
Backfill and sync paper trades for prop lanes into gold.db.

Reads pre-computed outcomes from orb_outcomes, applies each lane's filter
via matches_row (canonical filter interface from config.py), and writes
to paper_trades table.

Lanes derived from prop_profiles.py at runtime — zero hardcoded strategy_ids.
Filter application uses matches_row, not SQL WHERE fragments.

Usage:
    python -m trading_app.paper_trade_logger           # Full backfill from 2026-01-01
    python -m trading_app.paper_trade_logger --sync     # Incremental (new days only)
    python -m trading_app.paper_trade_logger --dry-run  # Show what would be inserted
    python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto
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
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter
from trading_app.paper_trade_store import PaperTradeRecord, delete_backfill_rows, upsert_backfill_trade

logger = get_logger(__name__)

# ── OOS boundary (fail-closed) ───────────────────────────────────────
OOS_START = datetime.date(2026, 1, 1)


@dataclass(frozen=True)
class LaneDef:
    """Definition of one paper-trade lane (derived from prop_profiles)."""

    strategy_id: str
    lane_name: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str


def build_lanes(profile_id: str | None = None) -> tuple[LaneDef, ...]:
    """Build lane definitions from a prop profile's daily_lanes.

    Single source of truth: prop_profiles.py → parse_strategy_id.
    No hardcoded strategy_ids or filter_sql in this module.
    If profile_id is None, uses first active profile.
    """
    from trading_app.prop_profiles import ACCOUNT_PROFILES, parse_strategy_id

    if profile_id is None:
        for pid, p in ACCOUNT_PROFILES.items():
            if p.active:
                profile_id = pid
                break
        else:
            raise RuntimeError("No active profiles found")

    from trading_app.prop_profiles import effective_daily_lanes

    profile = ACCOUNT_PROFILES[profile_id]
    lanes = []
    for spec in effective_daily_lanes(profile):
        parsed = parse_strategy_id(spec.strategy_id)
        lane_name = f"{spec.orb_label}_{parsed['filter_type'][:12]}"
        lanes.append(
            LaneDef(
                strategy_id=spec.strategy_id,
                lane_name=lane_name,
                instrument=spec.instrument,
                orb_label=spec.orb_label,
                orb_minutes=parsed["orb_minutes"],
                rr_target=parsed["rr_target"],
                entry_model=parsed["entry_model"],
                confirm_bars=parsed["confirm_bars"],
                filter_type=parsed["filter_type"],
            )
        )
    return tuple(lanes)


def _inject_cross_asset_atrs(
    con: duckdb.DuckDBPyConnection,
    feat_row: dict,
    instrument: str,
    trading_day: datetime.date,
) -> None:
    """Inject cross-asset ATR percentiles into daily features row.

    Required for CrossAssetATRFilter.matches_row (fail-closed without it).
    Mirrors paper_trader._inject_cross_asset_atrs_for_replay.
    """
    cross_sources = {f.source_instrument for f in ALL_FILTERS.values() if isinstance(f, CrossAssetATRFilter)}
    for source in cross_sources:
        if source == instrument:
            continue
        src_result = con.execute(
            """SELECT atr_20_pct FROM daily_features
               WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
                 AND trading_day = ?
               LIMIT 1""",
            [source, trading_day],
        ).fetchone()
        if src_result and src_result[0] is not None:
            feat_row[f"cross_atr_{source}_pct"] = float(src_result[0])


def _query_outcomes(
    con: duckdb.DuckDBPyConnection,
    lane: LaneDef,
    since: datetime.date | None = None,
) -> list[tuple]:
    """Query raw outcomes for a lane (no filter applied — filter in Python)."""
    start = since if since is not None else OOS_START
    return con.execute(
        """
        SELECT
            o.trading_day, o.orb_label, o.entry_ts,
            CASE WHEN o.entry_price > o.stop_price THEN 'long' ELSE 'short' END,
            o.entry_price, o.stop_price, o.target_price,
            o.exit_price, o.exit_ts, o.outcome, o.pnl_r
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.rr_target = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.outcome IS NOT NULL
          AND o.trading_day >= ?
        ORDER BY o.trading_day
        """,
        [
            lane.instrument,
            lane.orb_label,
            lane.orb_minutes,
            lane.rr_target,
            lane.entry_model,
            lane.confirm_bars,
            start,
        ],
    ).fetchall()


def _load_features(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    since: datetime.date | None = None,
) -> dict[datetime.date, dict]:
    """Load daily_features indexed by trading_day."""
    start = since if since is not None else OOS_START
    rows = con.execute(
        """
        SELECT * FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5 AND trading_day >= ?
        ORDER BY trading_day
        """,
        [instrument, start],
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    result = {}
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        result[d["trading_day"]] = d
    return result


def _is_cross_asset_filter(filter_type: str) -> bool:
    """Check if this filter needs cross-asset ATR injection."""
    f = ALL_FILTERS.get(filter_type)
    return isinstance(f, CrossAssetATRFilter)


def backfill(
    db_path: Path | str | None = None,
    sync: bool = False,
    dry_run: bool = False,
    profile_id: str | None = None,
) -> dict[str, dict]:
    """Backfill paper_trades from orb_outcomes, filtered by matches_row.

    Returns dict of strategy_id -> {count, cumulative_r, min_day, max_day}.
    """
    # Resolve profile
    if profile_id is None:
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for pid, p in ACCOUNT_PROFILES.items():
            if p.active:
                profile_id = pid
                break
        else:
            raise RuntimeError("No active profiles found")

    lanes = build_lanes(profile_id)
    logger.info(f"Profile: {profile_id}, {len(lanes)} lanes")

    path = str(db_path) if db_path else str(GOLD_DB_PATH)
    results: dict[str, dict] = {}

    # Ensure table exists BEFORE opening backfill connection (no dual writes)
    from trading_app.db_manager import init_trading_app_schema

    init_trading_app_schema(db_path=Path(path))

    with duckdb.connect(path) as con:
        configure_connection(con, writing=True)

        for lane in lanes:
            since = None
            if sync:
                row = con.execute(
                    "SELECT MAX(trading_day) FROM paper_trades WHERE strategy_id = ?",
                    [lane.strategy_id],
                ).fetchone()
                if row and row[0] is not None:
                    since = row[0] + datetime.timedelta(days=1)

            # 1. Load raw outcomes (no filter)
            raw_outcomes = _query_outcomes(con, lane, since=since)

            # 2. Load daily_features for filter application
            features = _load_features(con, lane.instrument, since=since)

            # Idempotent DELETE before filter check (ensures stale rows are cleared
            # even if the filter_type becomes unknown — audit finding #11)
            if sync and since is not None:
                delete_backfill_rows(con, strategy_id=lane.strategy_id, since=since)
            elif not sync:
                delete_backfill_rows(con, strategy_id=lane.strategy_id)

            # 3. Apply filter via matches_row (canonical filter interface)
            strat_filter = ALL_FILTERS.get(lane.filter_type)
            if strat_filter is None:
                logger.warning(f"Unknown filter {lane.filter_type} — skipping {lane.strategy_id}")
                con.commit()
                results[lane.strategy_id] = {"count": 0, "cumulative_r": 0, "min_day": None, "max_day": None}
                continue

            needs_cross = _is_cross_asset_filter(lane.filter_type)
            filtered_rows = []
            for r in raw_outcomes:
                trading_day = r[0]
                if isinstance(trading_day, datetime.datetime):
                    trading_day = trading_day.date()

                # OOS boundary assertion (fail-closed)
                if trading_day < OOS_START:
                    raise ValueError(
                        f"OOS BOUNDARY VIOLATION: {lane.strategy_id} row "
                        f"trading_day={trading_day} < {OOS_START}. Aborting."
                    )

                feat_row = features.get(trading_day)
                if feat_row is None:
                    continue

                # Break must exist for this session
                if feat_row.get(f"orb_{lane.orb_label}_break_dir") is None:
                    continue

                # Inject cross-asset ATR if needed
                if needs_cross:
                    _inject_cross_asset_atrs(con, feat_row, lane.instrument, trading_day)

                # Apply filter
                if not strat_filter.matches_row(feat_row, lane.orb_label):
                    continue

                filtered_rows.append(r)

            if dry_run:
                total_r = sum(r[10] for r in filtered_rows if r[10] is not None)
                logger.info(
                    f"[DRY RUN] {lane.strategy_id}: "
                    f"{len(filtered_rows)}/{len(raw_outcomes)} trades pass filter, "
                    f"cumulative R={total_r:+.2f}"
                )
                results[lane.strategy_id] = {
                    "count": len(filtered_rows),
                    "cumulative_r": round(total_r, 4),
                    "min_day": str(filtered_rows[0][0]) if filtered_rows else None,
                    "max_day": str(filtered_rows[-1][0]) if filtered_rows else None,
                }
                continue

            if filtered_rows:
                for r in filtered_rows:
                    upsert_backfill_trade(
                        con,
                        PaperTradeRecord(
                            trading_day=r[0],
                            orb_label=r[1],
                            entry_time=r[2],
                            direction=r[3],
                            entry_price=r[4],
                            stop_price=r[5],
                            target_price=r[6],
                            exit_price=r[7],
                            exit_time=r[8],
                            exit_reason=r[9],
                            pnl_r=r[10],
                            strategy_id=lane.strategy_id,
                            lane_name=lane.lane_name,
                            instrument=lane.instrument,
                            orb_minutes=lane.orb_minutes,
                            rr_target=lane.rr_target,
                            filter_type=lane.filter_type,
                            entry_model=lane.entry_model,
                            execution_source="backfill",
                        ),
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

            results[lane.strategy_id] = {
                "count": summary[0],
                "cumulative_r": round(summary[1], 4),
                "min_day": str(summary[2]) if summary[2] else None,
                "max_day": str(summary[3]) if summary[3] else None,
            }

            action = "synced" if sync else "backfilled"
            logger.info(
                f"{lane.strategy_id}: {action} {len(filtered_rows)} trades (total {summary[0]}, cumR={summary[1]:+.2f})"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Paper trade logger")
    parser.add_argument("--sync", action="store_true", help="Incremental sync (new days only)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted")
    parser.add_argument("--profile", type=str, default=None, help="Profile ID (default: first active)")
    args = parser.parse_args()

    results = backfill(sync=args.sync, dry_run=args.dry_run, profile_id=args.profile)

    print()
    print("=" * 60)
    print("PAPER TRADE BACKFILL SUMMARY")
    print("=" * 60)
    total_trades = 0
    total_r = 0.0
    for sid, info in results.items():
        n = info["count"]
        cr = info["cumulative_r"]
        total_trades += n
        total_r += cr
        print(f"  {sid:<55s}  trades={n:3d}  cumR={cr:+.2f}")
    print(f"  {'TOTAL':<55s}  trades={total_trades:3d}  cumR={total_r:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
