#!/usr/bin/env python3
"""Migrate ORB session names from fixed/old-dynamic to event-based names.

Renames orb_label values across orb_outcomes, experimental_strategies,
validated_setups, and edge_families. Renames daily_features columns from
orb_{old}_* to orb_{new}_*. Handles the 1130 -> SINGAPORE_OPEN merge
(drops 1130 data since 1100 also maps to SINGAPORE_OPEN).

Usage:
    python scripts/tools/migrate_session_names.py --dry-run   # preview changes
    python scripts/tools/migrate_session_names.py --backup     # backup then migrate
    python scripts/tools/migrate_session_names.py              # execute migration

The script is idempotent: running twice is safe (second run finds 0 rows).
"""

import argparse
import duckdb
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session rename map: old_label -> new_label
# ---------------------------------------------------------------------------
SESSION_RENAME_MAP = {
    # Fixed -> Dynamic event name
    "0900": "CME_REOPEN",
    "1000": "TOKYO_OPEN",
    "1100": "SINGAPORE_OPEN",
    "1130": "SINGAPORE_OPEN",       # Merge -- 1130 data will be DROPPED
    "1800": "LONDON_METALS",
    "2300": "US_DATA_830",
    "0030": "NYSE_OPEN",
    # Dynamic -> Renamed dynamic
    "CME_OPEN": "CME_REOPEN",
    "US_EQUITY_OPEN": "NYSE_OPEN",
    "US_DATA_OPEN": "US_DATA_830",
    "LONDON_OPEN": "LONDON_METALS",
    "US_POST_EQUITY": "US_DATA_1000",
    "CME_CLOSE": "CME_PRECLOSE",
}

# For strategy_id replacement, order matters to avoid partial matches.
# Process longer names first so e.g. _US_EQUITY_OPEN_ is replaced before
# a shorter suffix like _CME_OPEN_ could match a substring.
STRATEGY_ID_REPLACEMENTS = [
    # Dynamic sessions -- longer names first
    ("_US_EQUITY_OPEN_", "_NYSE_OPEN_"),
    ("_US_POST_EQUITY_", "_US_DATA_1000_"),
    ("_US_DATA_OPEN_", "_US_DATA_830_"),
    ("_LONDON_OPEN_", "_LONDON_METALS_"),
    ("_CME_CLOSE_", "_CME_PRECLOSE_"),
    ("_CME_OPEN_", "_CME_REOPEN_"),
    # Fixed sessions
    ("_0900_", "_CME_REOPEN_"),
    ("_1000_", "_TOKYO_OPEN_"),
    ("_1100_", "_SINGAPORE_OPEN_"),
    ("_1130_", "_SINGAPORE_OPEN_"),
    ("_1800_", "_LONDON_METALS_"),
    ("_2300_", "_US_DATA_830_"),
    ("_0030_", "_NYSE_OPEN_"),
]

# All ORB column suffixes that follow the orb_{label}_{suffix} pattern
ORB_COLUMN_SUFFIXES = [
    "high",
    "low",
    "size",
    "volume",
    "break_dir",
    "break_ts",
    "break_delay_min",
    "break_bar_continues",
    "break_bar_volume",
    "outcome",
    "mae_r",
    "mfe_r",
    "double_break",
]

# Compression columns: orb_{label}_compression_z, orb_{label}_compression_tier
# Only exist for 0900, 1000, 1800 in current schema
COMPRESSION_SUFFIXES = [
    "compression_z",
    "compression_tier",
]

# rel_vol columns use prefix pattern: rel_vol_{label}
# Handled separately from orb_{label}_* columns


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Check whether a table exists in the database."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table],
    ).fetchall()
    return len(rows) > 0


def _column_exists(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """Check whether a column exists on a table."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = ?",
        [table, column],
    ).fetchall()
    return len(rows) > 0


def _get_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return set of column names for a table."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    return {r[0] for r in rows}


def _count_rows(con: duckdb.DuckDBPyConnection, table: str, col: str, val: str) -> int:
    """Count rows matching col = val."""
    return con.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" = ?', [val]
    ).fetchone()[0]


# -----------------------------------------------------------------------
# Step 1: orb_outcomes -- DELETE 1130, then UPDATE orb_label
# -----------------------------------------------------------------------
def migrate_orb_outcomes(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename orb_label in orb_outcomes. Delete 1130 data first."""
    table = "orb_outcomes"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)

    # 1a. Delete 1130 rows to avoid PK collision with 1100 -> SINGAPORE_OPEN
    n_1130 = _count_rows(con, table, "orb_label", "1130")
    if n_1130 > 0:
        logger.info("  DELETE 1130 rows: %d", n_1130)
        if not dry_run:
            con.execute(f"DELETE FROM {table} WHERE orb_label = '1130'")
    else:
        logger.info("  DELETE 1130 rows: 0 (nothing to delete)")

    # 1b. Rename remaining labels
    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue  # already deleted
        n = _count_rows(con, table, "orb_label", old)
        if n > 0:
            logger.info("  UPDATE orb_label '%s' -> '%s': %d rows", old, new, n)
            if not dry_run:
                con.execute(
                    f"UPDATE {table} SET orb_label = ? WHERE orb_label = ?",
                    [new, old],
                )
        else:
            logger.info("  UPDATE orb_label '%s' -> '%s': 0 rows (skip)", old, new)


# -----------------------------------------------------------------------
# Step 2: experimental_strategies -- DELETE 1130, UPDATE orb_label + strategy_id
# -----------------------------------------------------------------------
def migrate_experimental_strategies(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename orb_label and strategy_id in experimental_strategies."""
    table = "experimental_strategies"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)

    # 2a. Delete 1130 rows
    n_1130 = _count_rows(con, table, "orb_label", "1130")
    if n_1130 > 0:
        logger.info("  DELETE 1130 rows: %d", n_1130)
        if not dry_run:
            con.execute(f"DELETE FROM {table} WHERE orb_label = '1130'")
    else:
        logger.info("  DELETE 1130 rows: 0 (nothing to delete)")

    # 2b. Rename orb_label
    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue
        n = _count_rows(con, table, "orb_label", old)
        if n > 0:
            logger.info("  UPDATE orb_label '%s' -> '%s': %d rows", old, new, n)
            if not dry_run:
                con.execute(
                    f"UPDATE {table} SET orb_label = ? WHERE orb_label = ?",
                    [new, old],
                )
        else:
            logger.info("  UPDATE orb_label '%s' -> '%s': 0 rows (skip)", old, new)

    # 2c. Rename strategy_id (chained REPLACE, longest first)
    _rename_strategy_ids(con, table, "strategy_id", dry_run)

    # 2d. Also fix canonical_strategy_id if it references old names
    if _column_exists(con, table, "canonical_strategy_id"):
        _rename_strategy_ids(con, table, "canonical_strategy_id", dry_run, label="canonical_strategy_id")


# -----------------------------------------------------------------------
# Step 3: validated_setups -- DELETE 1130, UPDATE orb_label + strategy_id + promoted_from
# -----------------------------------------------------------------------
def migrate_validated_setups(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename orb_label, strategy_id, promoted_from in validated_setups."""
    table = "validated_setups"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)

    # 3a. Delete 1130 rows
    n_1130 = _count_rows(con, table, "orb_label", "1130")
    if n_1130 > 0:
        logger.info("  DELETE 1130 rows: %d", n_1130)
        if not dry_run:
            con.execute(f"DELETE FROM {table} WHERE orb_label = '1130'")
    else:
        logger.info("  DELETE 1130 rows: 0 (nothing to delete)")

    # 3b. Rename orb_label
    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue
        n = _count_rows(con, table, "orb_label", old)
        if n > 0:
            logger.info("  UPDATE orb_label '%s' -> '%s': %d rows", old, new, n)
            if not dry_run:
                con.execute(
                    f"UPDATE {table} SET orb_label = ? WHERE orb_label = ?",
                    [new, old],
                )
        else:
            logger.info("  UPDATE orb_label '%s' -> '%s': 0 rows (skip)", old, new)

    # 3c. Rename strategy_id
    _rename_strategy_ids(con, table, "strategy_id", dry_run)

    # 3d. Rename promoted_from (FK to experimental_strategies.strategy_id)
    if _column_exists(con, table, "promoted_from"):
        _rename_strategy_ids(con, table, "promoted_from", dry_run, label="promoted_from")


# -----------------------------------------------------------------------
# Step 4: validated_setups_archive -- UPDATE original_strategy_id
# -----------------------------------------------------------------------
def migrate_validated_setups_archive(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename original_strategy_id in validated_setups_archive."""
    table = "validated_setups_archive"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)
    _rename_strategy_ids(con, table, "original_strategy_id", dry_run, label="original_strategy_id")


# -----------------------------------------------------------------------
# Step 5: strategy_trade_days -- UPDATE strategy_id
# -----------------------------------------------------------------------
def migrate_strategy_trade_days(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename strategy_id in strategy_trade_days."""
    table = "strategy_trade_days"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)
    _rename_strategy_ids(con, table, "strategy_id", dry_run)


# -----------------------------------------------------------------------
# Step 6: edge_families -- UPDATE head_strategy_id (family_hash is MD5, no session names)
# -----------------------------------------------------------------------
def migrate_edge_families(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename head_strategy_id in edge_families."""
    table = "edge_families"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)
    _rename_strategy_ids(con, table, "head_strategy_id", dry_run, label="head_strategy_id")


# -----------------------------------------------------------------------
# Step 7: nested_outcomes -- DELETE 1130, UPDATE orb_label
# -----------------------------------------------------------------------
def migrate_nested_outcomes(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename orb_label in nested_outcomes (if table exists)."""
    table = "nested_outcomes"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)

    # Delete 1130 rows
    n_1130 = _count_rows(con, table, "orb_label", "1130")
    if n_1130 > 0:
        logger.info("  DELETE 1130 rows: %d", n_1130)
        if not dry_run:
            con.execute(f"DELETE FROM {table} WHERE orb_label = '1130'")
    else:
        logger.info("  DELETE 1130 rows: 0 (nothing to delete)")

    # Rename orb_label
    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue
        n = _count_rows(con, table, "orb_label", old)
        if n > 0:
            logger.info("  UPDATE orb_label '%s' -> '%s': %d rows", old, new, n)
            if not dry_run:
                con.execute(
                    f"UPDATE {table} SET orb_label = ? WHERE orb_label = ?",
                    [new, old],
                )
        else:
            logger.info("  UPDATE orb_label '%s' -> '%s': 0 rows (skip)", old, new)


# -----------------------------------------------------------------------
# Step 8: nested_strategies + nested_validated -- orb_label + strategy_id
# -----------------------------------------------------------------------
def migrate_nested_strategies(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename orb_label and strategy_id in nested_strategies."""
    table = "nested_strategies"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)
    # Delete 1130
    n_1130 = _count_rows(con, table, "orb_label", "1130")
    if n_1130 > 0:
        logger.info("  DELETE 1130 rows: %d", n_1130)
        if not dry_run:
            con.execute(f"DELETE FROM {table} WHERE orb_label = '1130'")

    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue
        n = _count_rows(con, table, "orb_label", old)
        if n > 0:
            logger.info("  UPDATE orb_label '%s' -> '%s': %d rows", old, new, n)
            if not dry_run:
                con.execute(
                    f"UPDATE {table} SET orb_label = ? WHERE orb_label = ?",
                    [new, old],
                )

    _rename_strategy_ids(con, table, "strategy_id", dry_run)


def migrate_nested_validated(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename orb_label, strategy_id, promoted_from in nested_validated."""
    table = "nested_validated"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s ---", table)
    # Delete 1130
    n_1130 = _count_rows(con, table, "orb_label", "1130")
    if n_1130 > 0:
        logger.info("  DELETE 1130 rows: %d", n_1130)
        if not dry_run:
            con.execute(f"DELETE FROM {table} WHERE orb_label = '1130'")

    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue
        n = _count_rows(con, table, "orb_label", old)
        if n > 0:
            logger.info("  UPDATE orb_label '%s' -> '%s': %d rows", old, new, n)
            if not dry_run:
                con.execute(
                    f"UPDATE {table} SET orb_label = ? WHERE orb_label = ?",
                    [new, old],
                )

    _rename_strategy_ids(con, table, "strategy_id", dry_run)
    if _column_exists(con, table, "promoted_from"):
        _rename_strategy_ids(con, table, "promoted_from", dry_run, label="promoted_from")


# -----------------------------------------------------------------------
# Step 9: daily_features -- RENAME COLUMNS (orb_{old}_* -> orb_{new}_*)
# -----------------------------------------------------------------------
def migrate_daily_features_columns(con: duckdb.DuckDBPyConnection, dry_run: bool) -> None:
    """Rename ORB columns on daily_features from old session names to new names.

    Strategy:
      1. For each (old_label -> new_label) rename, rename all orb_{old}_* columns
         to orb_{new}_* (using ALTER TABLE ... RENAME COLUMN).
      2. For 1130 (merged into SINGAPORE_OPEN), DROP the orb_1130_* columns
         AFTER renaming orb_1100_* to orb_SINGAPORE_OPEN_*.
      3. Also rename rel_vol_{old} -> rel_vol_{new} columns.
      4. Also rename compression columns: orb_{old}_compression_z/tier.
      5. Also rename rsi_14_at_0900 -> rsi_14_at_CME_REOPEN (if present).
    """
    table = "daily_features"
    if not _table_exists(con, table):
        logger.info("  [SKIP] %s does not exist", table)
        return

    logger.info("--- %s (column renames) ---", table)
    existing_cols = _get_columns(con, table)

    # Build rename plan: list of (old_col, new_col) -- skip if old doesn't exist
    # or new already exists (idempotent)
    renames: list[tuple[str, str]] = []
    drops: list[str] = []

    # Determine which fixed labels map to which new labels (excluding 1130 merge)
    # Process 1130 columns separately as DROPs
    label_renames = {}
    for old, new in SESSION_RENAME_MAP.items():
        if old == "1130":
            continue  # 1130 columns will be DROPPED
        label_renames[old] = new

    # Plan renames for orb_{label}_* columns
    for old_label, new_label in label_renames.items():
        for suffix in ORB_COLUMN_SUFFIXES:
            old_col = f"orb_{old_label}_{suffix}"
            new_col = f"orb_{new_label}_{suffix}"
            if old_col in existing_cols and new_col not in existing_cols:
                renames.append((old_col, new_col))

        # Compression columns (only exist for some sessions)
        for csuf in COMPRESSION_SUFFIXES:
            old_col = f"orb_{old_label}_{csuf}"
            new_col = f"orb_{new_label}_{csuf}"
            if old_col in existing_cols and new_col not in existing_cols:
                renames.append((old_col, new_col))

        # rel_vol_{label}
        old_rv = f"rel_vol_{old_label}"
        new_rv = f"rel_vol_{new_label}"
        if old_rv in existing_cols and new_rv not in existing_cols:
            renames.append((old_rv, new_rv))

    # Plan renames for rsi_14_at_0900 -> rsi_14_at_CME_REOPEN
    if "rsi_14_at_0900" in existing_cols and "rsi_14_at_CME_REOPEN" not in existing_cols:
        renames.append(("rsi_14_at_0900", "rsi_14_at_CME_REOPEN"))

    # Plan DROPs for 1130 columns (after 1100 rename is done)
    for suffix in ORB_COLUMN_SUFFIXES:
        col_1130 = f"orb_1130_{suffix}"
        if col_1130 in existing_cols:
            drops.append(col_1130)
    for csuf in COMPRESSION_SUFFIXES:
        col_1130 = f"orb_1130_{csuf}"
        if col_1130 in existing_cols:
            drops.append(col_1130)
    rv_1130 = "rel_vol_1130"
    if rv_1130 in existing_cols:
        drops.append(rv_1130)

    # Execute renames
    if renames:
        logger.info("  RENAME %d columns:", len(renames))
        for old_col, new_col in renames:
            logger.info("    %s -> %s", old_col, new_col)
            if not dry_run:
                con.execute(
                    f'ALTER TABLE {table} RENAME COLUMN "{old_col}" TO "{new_col}"'
                )
    else:
        logger.info("  RENAME: 0 columns (already migrated or not present)")

    # Execute drops for 1130
    if drops:
        logger.info("  DROP %d columns (1130 merge):", len(drops))
        for col in drops:
            logger.info("    %s", col)
            if not dry_run:
                con.execute(f'ALTER TABLE {table} DROP COLUMN "{col}"')
    else:
        logger.info("  DROP 1130 columns: 0 (already dropped or not present)")


# -----------------------------------------------------------------------
# Shared helper: strategy_id REPLACE chain
# -----------------------------------------------------------------------
def _rename_strategy_ids(
    con: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    dry_run: bool,
    label: str | None = None,
) -> None:
    """Apply chained REPLACE on a strategy_id column.

    Processes longest old patterns first to avoid partial-match corruption.
    Reports the total count of rows that were modified.
    """
    col_label = label or column

    # Count total affected rows (any pattern match)
    like_clauses = " OR ".join(
        f'"{column}" LIKE \'%{old}%\'' for old, _ in STRATEGY_ID_REPLACEMENTS
    )
    if not like_clauses:
        return

    n_total = con.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE {like_clauses}'
    ).fetchone()[0]

    if n_total == 0:
        logger.info("  REPLACE %s: 0 rows (skip)", col_label)
        return

    logger.info("  REPLACE %s: %d rows to update", col_label, n_total)

    if dry_run:
        # Show per-pattern counts for dry-run visibility
        for old, new in STRATEGY_ID_REPLACEMENTS:
            n = con.execute(
                f"SELECT COUNT(*) FROM \"{table}\" WHERE \"{column}\" LIKE '%{old}%'",
            ).fetchone()[0]
            if n > 0:
                logger.info("    pattern '%s' -> '%s': %d rows", old, new, n)
        return

    # Build a chained REPLACE expression: REPLACE(REPLACE(REPLACE(col, ...), ...), ...)
    # Process in STRATEGY_ID_REPLACEMENTS order (longest first)
    expr = f'"{column}"'
    for old, new in STRATEGY_ID_REPLACEMENTS:
        expr = f"REPLACE({expr}, '{old}', '{new}')"

    con.execute(f'UPDATE "{table}" SET "{column}" = {expr} WHERE {like_clauses}')
    logger.info("    Done.")


# -----------------------------------------------------------------------
# Main migrate function
# -----------------------------------------------------------------------
def migrate(db_path: str, dry_run: bool = False) -> None:
    """Run the full session name migration.

    All changes are within a single transaction -- if any step fails,
    the entire migration is rolled back (all-or-nothing).
    """
    mode_label = "DRY RUN" if dry_run else "LIVE"
    logger.info("=" * 70)
    logger.info("SESSION NAME MIGRATION (%s)", mode_label)
    logger.info("=" * 70)
    logger.info("Database: %s", db_path)
    logger.info("")

    # Open in read-only for dry run, read-write for live
    con = duckdb.connect(str(db_path), read_only=dry_run)
    try:
        if not dry_run:
            # Begin an implicit transaction (DuckDB auto-begins on first write)
            # We rely on con.commit() at the end for all-or-nothing semantics
            pass

        # --- Row-level renames ---
        # Order matters: delete/rename in child tables before parent tables
        # to respect foreign key constraints.

        # 1. orb_outcomes (child of daily_features via FK)
        migrate_orb_outcomes(con, dry_run)

        # 2. nested_outcomes (child of daily_features via FK, if exists)
        migrate_nested_outcomes(con, dry_run)

        # 3. experimental_strategies
        migrate_experimental_strategies(con, dry_run)

        # 4. validated_setups_archive (FK to validated_setups)
        migrate_validated_setups_archive(con, dry_run)

        # 5. validated_setups (FK to experimental_strategies)
        migrate_validated_setups(con, dry_run)

        # 6. strategy_trade_days
        migrate_strategy_trade_days(con, dry_run)

        # 7. edge_families (FK to validated_setups)
        migrate_edge_families(con, dry_run)

        # 8. nested_strategies + nested_validated (if they exist)
        migrate_nested_strategies(con, dry_run)
        migrate_nested_validated(con, dry_run)

        # --- Column renames on daily_features ---
        # 9. daily_features columns (RENAME + DROP for 1130)
        migrate_daily_features_columns(con, dry_run)

        if not dry_run:
            con.commit()
            logger.info("")
            logger.info("MIGRATION COMMITTED SUCCESSFULLY")
        else:
            logger.info("")
            logger.info("DRY RUN COMPLETE -- no changes written")

    except Exception:
        if not dry_run:
            logger.error("MIGRATION FAILED -- rolling back")
            try:
                con.rollback()
            except Exception:
                pass  # connection may already be closed
        raise
    finally:
        con.close()

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ORB session names from fixed/old-dynamic to event-based names"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without executing",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy gold.db to gold.db.bak.premigration before running",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Resolve database path
    db_path = os.environ.get(
        "DUCKDB_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "gold.db"),
    )

    if not Path(db_path).exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    if args.backup:
        backup_path = db_path + ".bak.premigration"
        logger.info("Creating backup: %s -> %s", db_path, backup_path)
        shutil.copy2(db_path, backup_path)
        logger.info("Backup created (%d bytes)", Path(backup_path).stat().st_size)

    migrate(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
