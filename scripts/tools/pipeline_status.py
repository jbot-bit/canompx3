#!/usr/bin/env python3
"""
Pipeline staleness engine — detect which pipeline steps need rebuilding.

Queries MAX dates across all pipeline tables for a given instrument and
compares each table to its upstream dependency. Returns a structured dict
with dates, staleness flags, and a list of stale steps.

Usage:
    python scripts/tools/pipeline_status.py --instrument MGC
    python scripts/tools/pipeline_status.py  # all active instruments
"""

import subprocess
import sys
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Schema for rebuild_manifest — used by _ensure_manifest_table()
_REBUILD_MANIFEST_DDL = """
CREATE TABLE IF NOT EXISTS rebuild_manifest (
    rebuild_id TEXT PRIMARY KEY,
    instrument TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    failed_step TEXT,
    steps_completed TEXT[],
    trigger TEXT NOT NULL
);
"""


def _ensure_manifest_table(con: duckdb.DuckDBPyConnection) -> bool:
    """Create rebuild_manifest if it doesn't exist (idempotent).

    Returns True if table is available, False if DB is read-only and table missing.
    """
    try:
        con.execute(_REBUILD_MANIFEST_DDL)
        return True
    except duckdb.InvalidInputException:
        # Read-only connection — check if table already exists
        try:
            con.execute("SELECT 1 FROM rebuild_manifest LIMIT 0")
            return True
        except duckdb.CatalogException:
            return False


def _trading_days_between(d1: date | None, d2: date | None) -> int:
    """Count weekdays between d1 and d2 (exclusive of d1, inclusive of d2).

    Returns 0 if d1 >= d2 or either is None.
    Used to avoid weekend/holiday false positives in staleness detection.
    """
    if d1 is None or d2 is None or d1 >= d2:
        return 0
    count = 0
    current = d1 + timedelta(days=1)
    while current <= d2:
        if current.weekday() < 5:  # Mon-Fri
            count += 1
        current += timedelta(days=1)
    return count


def is_stale(
    table_date: date | None,
    reference_date: date | None,
    max_gap_trading_days: int = 1,
) -> bool:
    """Return True if table_date is None or gap to reference exceeds threshold.

    Compares table_date against reference_date (its upstream). If reference_date
    is also None, the table is not considered stale (nothing to rebuild from).
    """
    if reference_date is None:
        return False  # upstream has no data — nothing to be stale against
    if table_date is None:
        return True  # upstream has data but this table is empty
    gap = _trading_days_between(table_date, reference_date)
    return gap > max_gap_trading_days


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

PREFLIGHT_RULES: dict[str, dict] = {
    "outcome_builder": {
        "table": "daily_features",
        "query": "SELECT COUNT(*) FROM daily_features WHERE symbol = '{instrument}' AND orb_minutes = {orb_minutes}",
        "fix": "python pipeline/build_daily_features.py --instrument {instrument} --start 2019-01-01 --end 2026-12-31",
        "desc": "daily_features rows for {instrument} O{orb_minutes}",
    },
    "strategy_discovery": {
        "table": "orb_outcomes",
        "query": "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = '{instrument}' AND orb_minutes = {orb_minutes}",
        "fix": "python trading_app/outcome_builder.py --instrument {instrument} --orb-minutes {orb_minutes}",
        "desc": "orb_outcomes rows for {instrument} O{orb_minutes}",
    },
    "strategy_validator": {
        "table": "experimental_strategies",
        "query": "SELECT COUNT(*) FROM experimental_strategies WHERE instrument = '{instrument}'",
        "fix": "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes {orb_minutes}",
        "desc": "experimental_strategies rows for {instrument}",
    },
    "build_edge_families": {
        "table": "validated_setups",
        "query": "SELECT COUNT(*) FROM validated_setups WHERE instrument = '{instrument}' AND status = 'active'",
        "fix": "python trading_app/strategy_validator.py --instrument {instrument} --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75",
        "desc": "active validated_setups for {instrument}",
    },
    "select_family_rr": {
        "table": "edge_families",
        "query": "SELECT COUNT(*) FROM edge_families WHERE instrument = '{instrument}'",
        "fix": "python scripts/tools/build_edge_families.py --instrument {instrument}",
        "desc": "edge_families rows for {instrument}",
    },
}


def preflight_check(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    step: str,
    orb_minutes: int = 5,
) -> tuple[bool, str]:
    """Check if prerequisites for *step* are met for *instrument*.

    Returns (ok, message). If ok is False, message contains a fix command.
    """
    if step not in PREFLIGHT_RULES:
        return (True, f"No pre-flight rule for step '{step}'")

    rule = PREFLIGHT_RULES[step]
    query = rule["query"].format(instrument=instrument, orb_minutes=orb_minutes)
    desc = rule["desc"].format(instrument=instrument, orb_minutes=orb_minutes)
    fix = rule["fix"].format(instrument=instrument, orb_minutes=orb_minutes)

    row = con.execute(query).fetchone()
    count = row[0] if row else 0

    if count == 0:
        return (False, f"PRE-FLIGHT FAIL: no {desc}. Fix: {fix}")
    return (True, f"Pre-flight OK: {count} {desc}")


# ---------------------------------------------------------------------------
# Rebuild manifest
# ---------------------------------------------------------------------------


def write_manifest(
    con: duckdb.DuckDBPyConnection,
    rebuild_id: str,
    instrument: str,
    status: str,
    failed_step: str | None = None,
    steps_completed: list[str] | None = None,
    trigger: str = "MANUAL",
) -> None:
    """Write or update a rebuild manifest row.

    Uses parameterized SQL for safety. Sets started_at to now (UTC).
    Sets completed_at to now if status is COMPLETED or FAILED, else NULL.
    """
    _ensure_manifest_table(con)
    now = datetime.now(UTC)
    completed_at = now if status in ("COMPLETED", "FAILED") else None
    steps_arr = steps_completed if steps_completed else []

    con.execute(
        """
        INSERT OR REPLACE INTO rebuild_manifest
            (rebuild_id, instrument, started_at, completed_at, status, failed_step, steps_completed, trigger)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [rebuild_id, instrument, now, completed_at, status, failed_step, steps_arr, trigger],
    )
    con.commit()


def read_last_manifest(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> dict | None:
    """Return the most recent manifest row for *instrument*, or None."""
    row = con.execute(
        """
        SELECT rebuild_id, instrument, started_at, completed_at, status,
               failed_step, steps_completed, trigger
        FROM rebuild_manifest
        WHERE instrument = $1
        ORDER BY started_at DESC
        LIMIT 1
        """,
        [instrument],
    ).fetchone()
    if row is None:
        return None
    return {
        "rebuild_id": row[0],
        "instrument": row[1],
        "started_at": row[2],
        "completed_at": row[3],
        "status": row[4],
        "failed_step": row[5],
        "steps_completed": row[6],
        "trigger": row[7],
    }


def get_resume_point(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> dict | None:
    """Return the most recent FAILED manifest for *instrument*, or None."""
    row = con.execute(
        """
        SELECT rebuild_id, failed_step, steps_completed
        FROM rebuild_manifest
        WHERE instrument = $1 AND status = 'FAILED'
        ORDER BY started_at DESC
        LIMIT 1
        """,
        [instrument],
    ).fetchone()
    if row is None:
        return None
    return {
        "rebuild_id": row[0],
        "failed_step": row[1],
        "steps_completed": row[2],
    }


# ---------------------------------------------------------------------------
# Rebuild orchestration
# ---------------------------------------------------------------------------

REBUILD_STEPS = [
    ("outcome_builder_O5", "python trading_app/outcome_builder.py --instrument {instrument} --force --orb-minutes 5"),
    ("outcome_builder_O15", "python trading_app/outcome_builder.py --instrument {instrument} --force --orb-minutes 15"),
    ("outcome_builder_O30", "python trading_app/outcome_builder.py --instrument {instrument} --force --orb-minutes 30"),
    ("discovery_O5", "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes 5"),
    ("discovery_O15", "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes 15"),
    ("discovery_O30", "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes 30"),
    (
        "validator",
        "python trading_app/strategy_validator.py --instrument {instrument} --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75",
    ),
    ("retire_e3", "python scripts/migrations/retire_e3_strategies.py"),
    ("edge_families", "python scripts/tools/build_edge_families.py --instrument {instrument}"),
    ("family_rr_locks", "python scripts/tools/select_family_rr.py"),
    ("repo_map", "python scripts/tools/gen_repo_map.py"),
    ("health_check", "python pipeline/health_check.py"),
    ("pinecone_sync", "python scripts/tools/sync_pinecone.py"),
]


def build_step_list(instrument: str, resume_from: list[str] | None = None) -> list[dict]:
    """Build ordered list of rebuild steps, optionally skipping completed ones.

    Args:
        instrument: Instrument symbol to format into commands.
        resume_from: List of step names already completed (skipped).

    Returns:
        List of dicts with 'name' and 'cmd' keys.
    """
    skip = set(resume_from) if resume_from else set()
    steps = []
    for name, cmd_template in REBUILD_STEPS:
        if name in skip:
            continue
        steps.append(
            {
                "name": name,
                "cmd": cmd_template.format(instrument=instrument),
            }
        )
    return steps


def _parse_step_preflight(step_name: str) -> tuple[str, int]:
    """Extract preflight base name and orb_minutes from a step name.

    E.g. 'outcome_builder_O15' -> ('outcome_builder', 15)
         'validator' -> ('validator', 5)
    """
    if "_O" in step_name:
        parts = step_name.rsplit("_O", 1)
        base = parts[0]
        try:
            orb_min = int(parts[1])
        except (ValueError, IndexError):
            return step_name, 5
        return base, orb_min
    return step_name, 5


def run_rebuild(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    dry_run: bool = False,
    resume: bool = False,
    trigger: str = "CLI",
) -> bool:
    """Execute the full rebuild chain for *instrument*.

    Args:
        con: DuckDB connection (for manifest writes and preflight checks).
        instrument: Instrument symbol.
        dry_run: If True, print steps without executing.
        resume: If True, skip steps completed in the last FAILED manifest.
        trigger: Trigger label for the manifest record.

    Returns:
        True if all steps passed (or dry_run), False on any failure.
    """
    resume_completed: list[str] = []
    if resume:
        rp = get_resume_point(con, instrument)
        if rp and rp["steps_completed"]:
            resume_completed = list(rp["steps_completed"])
            print(f"Resuming from after: {', '.join(resume_completed)}")
        else:
            print("No failed rebuild found to resume — running full chain.")

    steps = build_step_list(instrument, resume_from=resume_completed if resume_completed else None)
    total = len(steps)

    if dry_run:
        print(f"DRY RUN — {total} steps for {instrument}:")
        for i, step in enumerate(steps, 1):
            print(f"  [{i}/{total}] {step['name']}: {step['cmd']}")
        return True

    rebuild_id = str(uuid.uuid4())
    completed: list[str] = list(resume_completed)  # carry forward previously completed

    # Write RUNNING manifest
    write_manifest(con, rebuild_id, instrument, "RUNNING", trigger=trigger, steps_completed=completed)
    print(f"Rebuild started: {rebuild_id} for {instrument} ({total} steps)")

    for i, step in enumerate(steps, 1):
        step_name = step["name"]
        step_cmd = step["cmd"]

        # Pre-flight check
        step_base, orb_min = _parse_step_preflight(step_name)
        ok, msg = preflight_check(con, instrument, step_base, orb_minutes=orb_min)
        if not ok:
            print(f"  [{i}/{total}] {step_name} — {msg}")
            write_manifest(
                con, rebuild_id, instrument, "FAILED", failed_step=step_name, steps_completed=completed, trigger=trigger
            )
            return False

        # Execute
        print(f"  [{i}/{total}] {step_name}")
        print(f"    CMD: {step_cmd}")
        result = subprocess.run(step_cmd, shell=True, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"    FAILED (exit code {result.returncode})")
            write_manifest(
                con, rebuild_id, instrument, "FAILED", failed_step=step_name, steps_completed=completed, trigger=trigger
            )
            return False

        print("    PASSED")
        completed.append(step_name)

    # All steps passed
    write_manifest(con, rebuild_id, instrument, "COMPLETED", steps_completed=completed, trigger=trigger)
    print(f"Rebuild COMPLETED: {rebuild_id}")
    return True


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

APERTURES = [5, 15, 30]


def staleness_engine(con: duckdb.DuckDBPyConnection, instrument: str) -> dict:
    """Query MAX dates across all pipeline tables for *instrument*.

    Returns a dict with keys:
        bars_1m           - max trading date for bars_1m (date or None)
        bars_5m           - max trading date for bars_5m (date or None)
        daily_features    - dict of {aperture: date} for each orb_minutes
        daily_features_min - min across all aperture max-dates (bottleneck)
        orb_outcomes      - max trading_day in orb_outcomes
        experimental      - max created_at date in experimental_strategies
        validated         - max promoted_at date for active validated_setups
        edge_families     - max created_at date in edge_families
        family_rr_locks   - max updated_at date in family_rr_locks
        last_rebuild      - max completed_at date from rebuild_manifest
        stale_steps       - list of step names that are stale
    """
    result: dict = {}

    # --- bars_1m: max date (cast TIMESTAMPTZ to DATE) ---
    row = con.execute(
        "SELECT MAX(ts_utc::DATE) FROM bars_1m WHERE symbol = ?",
        [instrument],
    ).fetchone()
    result["bars_1m"] = row[0] if row and row[0] is not None else None

    # --- bars_5m ---
    row = con.execute(
        "SELECT MAX(ts_utc::DATE) FROM bars_5m WHERE symbol = ?",
        [instrument],
    ).fetchone()
    result["bars_5m"] = row[0] if row and row[0] is not None else None

    # --- daily_features per aperture ---
    df_dates: dict[int, date | None] = {}
    for ap in APERTURES:
        row = con.execute(
            "SELECT MAX(trading_day) FROM daily_features WHERE symbol = ? AND orb_minutes = ?",
            [instrument, ap],
        ).fetchone()
        df_dates[ap] = row[0] if row and row[0] is not None else None
    result["daily_features"] = df_dates

    # Bottleneck: the minimum across all aperture max-dates
    non_null = [d for d in df_dates.values() if d is not None]
    result["daily_features_min"] = min(non_null) if non_null else None

    # --- orb_outcomes ---
    row = con.execute(
        "SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = ?",
        [instrument],
    ).fetchone()
    result["orb_outcomes"] = row[0] if row and row[0] is not None else None

    # --- experimental_strategies (uses 'instrument' column, not 'symbol') ---
    row = con.execute(
        "SELECT MAX(created_at::DATE) FROM experimental_strategies WHERE instrument = ?",
        [instrument],
    ).fetchone()
    result["experimental"] = row[0] if row and row[0] is not None else None

    # --- validated_setups (active only) ---
    row = con.execute(
        "SELECT MAX(promoted_at::DATE) FROM validated_setups WHERE instrument = ? AND status = 'active'",
        [instrument],
    ).fetchone()
    result["validated"] = row[0] if row and row[0] is not None else None

    # --- edge_families ---
    row = con.execute(
        "SELECT MAX(created_at::DATE) FROM edge_families WHERE instrument = ?",
        [instrument],
    ).fetchone()
    result["edge_families"] = row[0] if row and row[0] is not None else None

    # --- family_rr_locks (may not have instrument column) ---
    try:
        row = con.execute(
            "SELECT MAX(updated_at::DATE) FROM family_rr_locks WHERE instrument = ?",
            [instrument],
        ).fetchone()
        result["family_rr_locks"] = row[0] if row and row[0] is not None else None
    except duckdb.BinderException:
        # Table exists but no instrument column — query without filter
        try:
            row = con.execute("SELECT MAX(updated_at::DATE) FROM family_rr_locks").fetchone()
            result["family_rr_locks"] = row[0] if row and row[0] is not None else None
        except Exception:
            result["family_rr_locks"] = None

    # --- rebuild_manifest (last completed rebuild) ---
    if _ensure_manifest_table(con):
        row = con.execute(
            "SELECT MAX(completed_at::DATE) FROM rebuild_manifest WHERE instrument = ? AND status = 'completed'",
            [instrument],
        ).fetchone()
        result["last_rebuild"] = row[0] if row and row[0] is not None else None
    else:
        result["last_rebuild"] = None

    # --- Staleness detection ---
    # Each step compared to its UPSTREAM, not to today.
    stale_steps: list[str] = []

    # bars_5m should track bars_1m
    if is_stale(result["bars_5m"], result["bars_1m"]):
        stale_steps.append("bars_5m")

    # daily_features should track bars_5m (per aperture)
    for ap in APERTURES:
        if is_stale(df_dates.get(ap), result["bars_5m"]):
            stale_steps.append(f"daily_features_O{ap}")

    # orb_outcomes should track daily_features_min
    if is_stale(result["orb_outcomes"], result["daily_features_min"]):
        stale_steps.append("orb_outcomes")

    # experimental should track orb_outcomes
    if is_stale(result["experimental"], result["orb_outcomes"]):
        stale_steps.append("experimental_strategies")

    # validated should track experimental
    if is_stale(result["validated"], result["experimental"]):
        stale_steps.append("validated_setups")

    # edge_families should track validated
    if is_stale(result["edge_families"], result["validated"]):
        stale_steps.append("edge_families")

    result["stale_steps"] = stale_steps
    return result


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_status(instrument: str, status: dict) -> str:
    """Human-readable text output showing dates and staleness for each table."""
    lines = [f"=== Pipeline Status: {instrument} ===", ""]

    def _fmt(d: date | None) -> str:
        return str(d) if d is not None else "(none)"

    def _stale_tag(step_name: str) -> str:
        return " ** STALE **" if step_name in status["stale_steps"] else ""

    lines.append(f"  bars_1m           : {_fmt(status['bars_1m'])}")
    lines.append(f"  bars_5m           : {_fmt(status['bars_5m'])}{_stale_tag('bars_5m')}")

    for ap in APERTURES:
        d = status["daily_features"].get(ap)
        lines.append(f"  daily_features O{ap:<2} : {_fmt(d)}{_stale_tag(f'daily_features_O{ap}')}")

    lines.append(f"  daily_features min: {_fmt(status['daily_features_min'])}")
    lines.append(f"  orb_outcomes      : {_fmt(status['orb_outcomes'])}{_stale_tag('orb_outcomes')}")
    lines.append(f"  experimental      : {_fmt(status['experimental'])}{_stale_tag('experimental_strategies')}")
    lines.append(f"  validated (active): {_fmt(status['validated'])}{_stale_tag('validated_setups')}")
    lines.append(f"  edge_families     : {_fmt(status['edge_families'])}{_stale_tag('edge_families')}")
    lines.append(f"  family_rr_locks   : {_fmt(status['family_rr_locks'])}")
    lines.append(f"  last_rebuild      : {_fmt(status['last_rebuild'])}")

    lines.append("")
    if status["stale_steps"]:
        lines.append(f"  Stale steps: {', '.join(status['stale_steps'])}")
    else:
        lines.append("  All steps up to date.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline staleness status and rebuild orchestration",
    )

    # Actions (mutually exclusive group)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true", help="Show staleness status (default)")
    actions.add_argument("--rebuild", action="store_true", help="Run rebuild chain for one instrument")
    actions.add_argument("--rebuild-all", action="store_true", help="Run rebuild chain for all stale instruments")
    actions.add_argument("--resume", action="store_true", help="Resume last failed rebuild")
    actions.add_argument("--write-manifest", action="store_true", help="Write a manifest record (for shell scripts)")

    # Options
    parser.add_argument("--instrument", type=str, default=None, help="Instrument symbol")
    parser.add_argument(
        "--status-value", type=str, choices=["COMPLETED", "FAILED", "RUNNING"], help="Status for --write-manifest"
    )
    parser.add_argument(
        "--trigger", type=str, default="CLI", choices=["CLI", "SHELL", "MANUAL"], help="Trigger type (default: CLI)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would execute without running")
    parser.add_argument("--db-path", type=str, default=None, help="Database path (default: GOLD_DB_PATH)")

    args = parser.parse_args()

    db_path = args.db_path if args.db_path else str(GOLD_DB_PATH)

    # --- --write-manifest ---
    if args.write_manifest:
        if not args.instrument:
            parser.error("--write-manifest requires --instrument")
        if not args.status_value:
            parser.error("--write-manifest requires --status-value")
        con = duckdb.connect(db_path)
        try:
            rid = str(uuid.uuid4())
            write_manifest(con, rid, args.instrument, args.status_value, trigger=args.trigger)
            print(f"Manifest written: {rid} {args.instrument} {args.status_value}")
        finally:
            con.close()
        return

    # --- --rebuild ---
    if args.rebuild:
        if not args.instrument:
            parser.error("--rebuild requires --instrument")
        con = duckdb.connect(db_path)
        try:
            ok = run_rebuild(con, args.instrument, dry_run=args.dry_run, trigger=args.trigger)
        finally:
            con.close()
        sys.exit(0 if ok else 1)

    # --- --rebuild-all ---
    if args.rebuild_all:
        con = duckdb.connect(db_path)
        try:
            for inst in ACTIVE_ORB_INSTRUMENTS:
                status = staleness_engine(con, inst)
                if not status["stale_steps"]:
                    print(f"{inst}: up to date, skipping.")
                    continue
                print(f"{inst}: stale ({', '.join(status['stale_steps'])}), rebuilding...")
                ok = run_rebuild(con, inst, dry_run=args.dry_run, trigger=args.trigger)
                if not ok:
                    print(f"{inst}: rebuild FAILED — stopping.")
                    sys.exit(1)
        finally:
            con.close()
        print("All stale instruments rebuilt successfully.")
        return

    # --- --resume ---
    if args.resume:
        if not args.instrument:
            parser.error("--resume requires --instrument")
        con = duckdb.connect(db_path)
        try:
            ok = run_rebuild(con, args.instrument, dry_run=args.dry_run, resume=True, trigger=args.trigger)
        finally:
            con.close()
        sys.exit(0 if ok else 1)

    # --- Default: --status ---
    instruments = [args.instrument] if args.instrument else list(ACTIVE_ORB_INSTRUMENTS)
    con = duckdb.connect(db_path, read_only=True)
    try:
        for inst in instruments:
            status = staleness_engine(con, inst)
            print(format_status(inst, status))
            print()
    finally:
        con.close()


if __name__ == "__main__":
    main()
