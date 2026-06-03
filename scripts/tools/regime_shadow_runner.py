#!/usr/bin/env python3
"""
Forward-only REGIME shadow accumulator.

Drives EVERY active REGIME-tier lane (from regime_shadow_universe, record-ALL)
into the `paper_trades` ledger as forward-monitoring evidence, using the
canonical pre-computed `orb_outcomes` as the trade source. Each recorded row
carries `execution_source = 'shadow'` so it is structurally invisible to every
live / monitoring consumer (which filter execution_source = 'live' or 'backfill').

FORWARD-ONLY GUARANTEE (capital-safety invariant):
  - A persisted forward_start boundary (docs/runtime/regime_shadow_universe.yaml,
    default = first-run date) is the EARLIEST trading_day that may be written as
    shadow. Any candidate row with trading_day < forward_start is refused with a
    ValueError — history is NEVER relabelled as shadow.
  - Idempotent: re-running deletes only this lane's shadow rows on/after the
    boundary and re-inserts, so a sync never double-counts and never touches
    'live'/'backfill' rows.

SINGLE-WRITER GUARANTEE (G1 — capital-safety invariant):
  - Before opening a write connection, the runner checks the canonical live-bot
    instance lock (trading_app.live.instance_lock). If ANY live instrument lock
    is held by a live process, the sync FAILS CLOSED rather than contending the
    gold.db write lock. Intended trigger is CanonMPX_DailyRefresh (single-writer,
    off-peak) — never concurrent with a live session.

This module makes NO changes to live allocation, profiles, account gates, or
capital logic. Its only write is appending source='shadow' rows to paper_trades.
Outcomes are pre-computed canonical — no bar replay, no re-encoded logic.

Usage:
    python -m scripts.tools.regime_shadow_runner --dry-run     # report, no writes
    python -m scripts.tools.regime_shadow_runner               # sync shadow rows
    python -m scripts.tools.regime_shadow_runner --prune-orphans  # remove stale shadow lanes
    python -m scripts.tools.regime_shadow_runner --oos-context-report  # RO only
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from pipeline.db_config import configure_connection
from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH
from scripts.tools.regime_shadow_universe import (
    UNIVERSE_YAML,
    RegimeLane,
    build_universe,
)

logger = get_logger(__name__)

SHADOW_SOURCE = "shadow"
# OOS boundary for the optional read-only context report only (NEVER a write
# boundary for shadow rows — those are gated by the persisted forward_start).
OOS_CONTEXT_START = datetime.date(2026, 1, 1)


@dataclass
class LaneResult:
    """Per-lane shadow-sync outcome (auditable; no silent drops)."""

    strategy_id: str
    instrument: str
    orb_label: str
    scanned: int = 0
    appended: int = 0
    skipped_no_feature: int = 0
    skipped_no_break: int = 0
    skipped_filter: int = 0
    skipped_no_pnl: int = 0
    error: str | None = None


@dataclass
class RunSummary:
    """Whole-run summary returned by sync_shadow (printed by main)."""

    forward_start: datetime.date
    dry_run: bool
    lanes_total: int = 0
    lanes_scanned: int = 0
    lanes_errored: int = 0
    trades_appended: int = 0
    orphans_found: list[str] = field(default_factory=list)
    orphans_pruned: int = 0
    lane_results: list[LaneResult] = field(default_factory=list)


# ── G1: single-writer live-session guard ────────────────────────────────────


def assert_no_live_session() -> None:
    """Fail closed if a live trading bot holds an instance lock.

    Reuses the canonical live-bot lock (trading_app.live.instance_lock): a live
    session writes %TEMP%/canompx3/bot_<INSTRUMENT>.lock with its PID. We READ
    those lock files (never acquire them — that would contend the live bot's own
    lock) and refuse to open a gold.db write connection while any live PID is
    alive. This enforces the repo invariant "NEVER run two write processes
    against the same DuckDB file simultaneously" for the shadow writer.

    No re-encoded logic: PID liveness is delegated to instance_lock.is_pid_alive.
    """
    from trading_app.live.instance_lock import _LOCK_DIR, is_pid_alive

    if not _LOCK_DIR.exists():
        return
    for lock_path in _LOCK_DIR.glob("bot_*.lock"):
        try:
            content = lock_path.read_text().strip()
        except OSError:
            continue
        if not content:
            continue
        try:
            pid = int(content)
        except ValueError:
            continue
        if is_pid_alive(pid):
            raise RuntimeError(
                f"LIVE SESSION ACTIVE: {lock_path.name} held by live PID {pid}. "
                "Refusing to write shadow rows while a live bot holds the gold.db "
                "write path (single-writer invariant). Run after the live session "
                "ends, or via CanonMPX_DailyRefresh (off-peak, exclusive)."
            )


# ── forward_start persistence ────────────────────────────────────────────


def _min_shadow_trading_day(db_path: Path | str) -> datetime.date | None:
    """MIN(trading_day) of existing source='shadow' rows, or None (read-only)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        # paper_trades may not exist on a brand-new DB.
        exists = con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'paper_trades'").fetchone()
        if not exists:
            return None
        row = con.execute(
            "SELECT MIN(trading_day) FROM paper_trades WHERE execution_source = ?",
            [SHADOW_SOURCE],
        ).fetchone()
    if row and row[0] is not None:
        md = row[0]
        return md.date() if isinstance(md, datetime.datetime) else md
    return None


def resolve_forward_start(
    universe_yaml: Path | str | None = None,
    today: datetime.date | None = None,
    db_path: Path | str | None = None,
) -> tuple[datetime.date, bool]:
    """Return (forward_start, was_persisted).

    Boundary authority, in order:
      1. If the universe YAML carries a `forward_start`, use it (fixed boundary).
      2. G2 fail-closed fallback: if the YAML is MISSING but shadow rows already
         exist in the DB, re-derive the boundary as MIN(trading_day) of those
         rows — NEVER reset to `today` (that would re-open the relabel window for
         the gap between the real first-shadow day and today). Reported as
         persisted=True so the caller does not re-stamp a wrong boundary.
      3. Otherwise (no YAML, no shadow rows) the boundary is `today` and the
         caller persists it so all later runs share the same earliest-shadow date.
    """
    if today is None:
        today = datetime.date.today()
    path = Path(universe_yaml) if universe_yaml else UNIVERSE_YAML
    if path.exists():
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fs = data.get("forward_start")
        if fs:
            return datetime.date.fromisoformat(str(fs)), True

    # YAML missing — guard against accidental boundary reset.
    if db_path is not None:
        recovered = _min_shadow_trading_day(db_path)
        if recovered is not None:
            logger.warning(
                "universe YAML missing but shadow rows exist; re-deriving "
                "forward_start=%s from MIN(trading_day) (not today=%s)",
                recovered.isoformat(),
                today.isoformat(),
            )
            return recovered, True

    return today, False


def _persist_forward_start(
    lanes: list[RegimeLane],
    forward_start: datetime.date,
    universe_yaml: Path | str | None,
    as_of_date: datetime.date,
) -> None:
    """Write the universe snapshot so the boundary is fixed for later runs."""
    from scripts.tools.regime_shadow_universe import write_universe_yaml

    write_universe_yaml(
        lanes,
        forward_start=forward_start,
        path=universe_yaml,
        as_of_date=as_of_date,
    )


# ── outcome sourcing (reuses paper_trade_logger's canonical pattern) ───────


def _shadow_rows_for_lane(
    con: duckdb.DuckDBPyConnection,
    lane: RegimeLane,
    forward_start: datetime.date,
) -> tuple[list[tuple], LaneResult]:
    """Compute the shadow rows for one lane from canonical orb_outcomes.

    Reuses the proven filter-application path from trading_app.paper_trade_logger
    (matches_row + per-aperture daily_features + cross-asset ATR injection) so we
    never re-encode filter logic. Returns (insert_tuples, lane_result).

    FAIL-CLOSED: raises ValueError if any sourced row precedes forward_start —
    the canonical query already filters >= forward_start, so this is a tripwire
    against a future query regression silently back-relabelling history.
    """
    from trading_app.config import ALL_FILTERS
    from trading_app.paper_trade_logger import (
        LaneDef,
        _inject_cross_asset_atrs,
        _is_cross_asset_filter,
        _load_features,
        _query_outcomes,
    )

    result = LaneResult(strategy_id=lane.strategy_id, instrument=lane.instrument, orb_label=lane.orb_label)

    lane_def = LaneDef(
        strategy_id=lane.strategy_id,
        lane_name=f"{lane.orb_label}_{lane.filter_type[:12]}",
        instrument=lane.instrument,
        orb_label=lane.orb_label,
        orb_minutes=lane.orb_minutes,
        rr_target=lane.rr_target,
        entry_model=lane.entry_model,
        confirm_bars=lane.confirm_bars,
        filter_type=lane.filter_type,
    )

    strat_filter = ALL_FILTERS.get(lane.filter_type)
    if strat_filter is None:
        result.error = f"unknown filter_type {lane.filter_type}"
        logger.warning("%s: %s — skipping", lane.strategy_id, result.error)
        return [], result

    # Forward-only at the source: only days >= boundary are candidates.
    raw = _query_outcomes(con, lane_def, since=forward_start)
    features = _load_features(con, lane.instrument, lane.orb_minutes, since=forward_start)
    needs_cross = _is_cross_asset_filter(lane.filter_type)

    insert_rows: list[tuple] = []
    for r in raw:
        result.scanned += 1
        trading_day = r[0]
        if isinstance(trading_day, datetime.datetime):
            trading_day = trading_day.date()

        # FAIL-CLOSED forward-only tripwire — never relabel history as shadow.
        if trading_day < forward_start:
            raise ValueError(
                f"FORWARD BOUNDARY VIOLATION: {lane.strategy_id} row trading_day="
                f"{trading_day} < forward_start={forward_start}. Aborting."
            )

        feat_row = features.get(trading_day)
        if feat_row is None:
            result.skipped_no_feature += 1
            continue
        if feat_row.get(f"orb_{lane.orb_label}_break_dir") is None:
            result.skipped_no_break += 1
            continue
        if needs_cross:
            _inject_cross_asset_atrs(con, feat_row, lane.instrument, trading_day, lane.orb_minutes)
        if not strat_filter.matches_row(feat_row, lane.orb_label):
            result.skipped_filter += 1
            continue
        if r[10] is None:  # pnl_r
            result.skipped_no_pnl += 1
            continue

        # Column order mirrors paper_trade_logger insert + execution_source.
        insert_rows.append(
            (
                trading_day,  # trading_day
                r[1],  # orb_label
                r[2],  # entry_time
                r[3],  # direction
                r[4],  # entry_price
                r[5],  # stop_price
                r[6],  # target_price
                r[7],  # exit_price
                r[8],  # exit_time
                r[9],  # exit_reason
                r[10],  # pnl_r
                lane.strategy_id,
                lane_def.lane_name,
                lane.instrument,
                lane.orb_minutes,
                lane.rr_target,
                lane.filter_type,
                lane.entry_model,
                SHADOW_SOURCE,  # execution_source
            )
        )

    result.appended = len(insert_rows)
    return insert_rows, result


# ── main sync ──────────────────────────────────────────────────────────────


def _find_orphan_shadow_strategies(
    con: duckdb.DuckDBPyConnection,
    universe_ids: set[str],
    forward_start: datetime.date,
) -> list[str]:
    """Return shadow strategy_ids on/after the boundary NOT in the current
    universe (G5 stale-orphan reconcile). Read-only."""
    rows = con.execute(
        "SELECT DISTINCT strategy_id FROM paper_trades WHERE execution_source = ? AND trading_day >= ?",
        [SHADOW_SOURCE, forward_start],
    ).fetchall()
    return sorted(r[0] for r in rows if r[0] not in universe_ids)


def sync_shadow(
    db_path: Path | str | None = None,
    *,
    dry_run: bool = False,
    prune_orphans: bool = False,
    as_of_date: datetime.date | None = None,
    universe_yaml: Path | str | None = None,
) -> RunSummary:
    """Forward-only shadow sync (record-ALL). Append-only on source='shadow' rows.

    Idempotent: per lane, deletes existing shadow rows on/after forward_start
    then re-inserts. NEVER deletes or writes 'live'/'backfill' rows.

    G5: orphan shadow lanes (strategy_ids no longer in the universe) are always
    reported; pruned ONLY when prune_orphans=True (never silent).
    """
    path = Path(db_path) if db_path else GOLD_DB_PATH
    if as_of_date is None:
        as_of_date = datetime.date.today()

    forward_start, was_persisted = resolve_forward_start(universe_yaml, today=as_of_date, db_path=path)
    # RECORD-ALL: every active REGIME lane is included by build_universe.
    lanes = build_universe(db_path=path, as_of_date=as_of_date)

    summary = RunSummary(forward_start=forward_start, dry_run=dry_run, lanes_total=len(lanes))
    universe_ids = {x.strategy_id for x in lanes}

    if dry_run:
        with duckdb.connect(str(path), read_only=True) as con:
            for lane in lanes:
                rows, res = _shadow_rows_for_lane(con, lane, forward_start)
                summary.lane_results.append(res)
                summary.lanes_scanned += 1
                if res.error:
                    summary.lanes_errored += 1
                summary.trades_appended += len(rows)
            summary.orphans_found = _find_orphan_shadow_strategies(con, universe_ids, forward_start)
        return summary

    # G1: refuse to write while a live bot holds the gold.db write path. Checked
    # BEFORE any side effect (incl. persisting the boundary YAML) so an aborted
    # sync leaves no partial state.
    assert_no_live_session()

    # Persist the boundary on first ever run (so it's fixed thereafter).
    if not was_persisted:
        _persist_forward_start(lanes, forward_start, universe_yaml, as_of_date)

    # Ensure schema, then a single writing connection (no dual writers).
    from trading_app.db_manager import init_trading_app_schema

    init_trading_app_schema(db_path=path)

    with duckdb.connect(str(path)) as con:
        configure_connection(con, writing=True)
        for lane in lanes:
            rows, res = _shadow_rows_for_lane(con, lane, forward_start)
            summary.lane_results.append(res)
            summary.lanes_scanned += 1
            if res.error:
                summary.lanes_errored += 1
                con.commit()
                continue

            # Idempotent: clear only THIS lane's shadow rows >= boundary.
            con.execute(
                "DELETE FROM paper_trades WHERE strategy_id = ? AND execution_source = ? AND trading_day >= ?",
                [lane.strategy_id, SHADOW_SOURCE, forward_start],
            )
            if rows:
                con.executemany(
                    """
                    INSERT INTO paper_trades (
                        trading_day, orb_label, entry_time, direction,
                        entry_price, stop_price, target_price, exit_price,
                        exit_time, exit_reason, pnl_r,
                        strategy_id, lane_name, instrument,
                        orb_minutes, rr_target, filter_type, entry_model,
                        execution_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            con.commit()
            summary.trades_appended += len(rows)

        # G5: reconcile orphan shadow lanes (report always; prune only if asked).
        summary.orphans_found = _find_orphan_shadow_strategies(con, universe_ids, forward_start)
        if prune_orphans and summary.orphans_found:
            for sid in summary.orphans_found:
                con.execute(
                    "DELETE FROM paper_trades WHERE strategy_id = ? AND execution_source = ? AND trading_day >= ?",
                    [sid, SHADOW_SOURCE, forward_start],
                )
                summary.orphans_pruned += 1
            con.commit()

    return summary


def oos_context_report(
    db_path: Path | str | None = None,
    as_of_date: datetime.date | None = None,
) -> list[LaneResult]:
    """READ-ONLY OOS context report (2026-01-01 onward). Computes what the
    REGIME universe WOULD have recorded since the OOS boundary — for operator
    context only. Inserts NOTHING. Never writes shadow rows for pre-forward_start
    history."""
    path = Path(db_path) if db_path else GOLD_DB_PATH
    if as_of_date is None:
        as_of_date = datetime.date.today()
    lanes = build_universe(db_path=path, as_of_date=as_of_date)
    results: list[LaneResult] = []
    with duckdb.connect(str(path), read_only=True) as con:
        for lane in lanes:
            # Reuse the same path but with the OOS boundary; this never writes.
            rows, res = _shadow_rows_for_lane(con, lane, OOS_CONTEXT_START)
            res.appended = len(rows)  # "would-record" count, NOT written
            results.append(res)
    return results


def _print_summary(summary: RunSummary) -> None:
    mode = "DRY-RUN (no writes)" if summary.dry_run else "SYNC"
    print("=" * 78)
    print(f"REGIME SHADOW RUNNER — {mode}")
    print("=" * 78)
    print(f"forward_start boundary : {summary.forward_start.isoformat()}")
    print(f"lanes (REGIME, all)    : {summary.lanes_total}")
    print(f"lanes scanned          : {summary.lanes_scanned}")
    print(f"lanes errored          : {summary.lanes_errored}")
    print(f"trades {'would-append' if summary.dry_run else 'appended':<13}: {summary.trades_appended}")
    if summary.orphans_found:
        verb = "pruned" if summary.orphans_pruned else "found (NOT pruned; pass --prune-orphans)"
        print(f"orphan shadow lanes    : {len(summary.orphans_found)} {verb}")
        for sid in summary.orphans_found:
            print(f"    orphan: {sid}")
    print("-" * 78)
    for res in summary.lane_results:
        tag = f"ERR({res.error})" if res.error else f"+{res.appended}"
        print(
            f"  {res.strategy_id:<48s} scan={res.scanned:<4d} {tag:<14s} "
            f"skip[feat={res.skipped_no_feature} brk={res.skipped_no_break} "
            f"filt={res.skipped_filter} npnl={res.skipped_no_pnl}]"
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only REGIME shadow accumulator (record-ALL)")
    parser.add_argument("--dry-run", action="store_true", help="Report only; no DB writes")
    parser.add_argument(
        "--prune-orphans",
        action="store_true",
        help="Remove shadow rows for strategies no longer in the universe (default OFF)",
    )
    parser.add_argument(
        "--oos-context-report", action="store_true", help="READ-ONLY OOS (2026-01-01+) context report; inserts nothing"
    )
    args = parser.parse_args()

    if args.oos_context_report:
        results = oos_context_report()
        total = sum(r.appended for r in results)
        print("=" * 78)
        print(f"REGIME OOS CONTEXT REPORT (read-only, since {OOS_CONTEXT_START.isoformat()}) — NOT written")
        print("=" * 78)
        for r in results:
            print(f"  {r.strategy_id:<48s} would-record={r.appended:<4d} (scanned {r.scanned})")
        print(f"  TOTAL would-record (NOT inserted): {total}")
        print("=" * 78)
        return

    summary = sync_shadow(dry_run=args.dry_run, prune_orphans=args.prune_orphans)
    _print_summary(summary)


if __name__ == "__main__":
    main()
