#!/usr/bin/env python3
"""
Database backup — copies gold.db to backups/ with date stamp.

Features:
  - Rolling backups with auto-prune (default: keep 5)
  - Labeled rebuild backups (tagged with rebuild_id, keep 3)
  - Integrity verification (DuckDB opens read-only, bars_1m exists)
  - Restore from any backup (with pre-restore safety snapshot)
  - List all available backups

Usage:
    python scripts/infra/backup_db.py                    # rolling backup
    python scripts/infra/backup_db.py --keep 10           # keep more copies
    python scripts/infra/backup_db.py --list              # show available backups
    python scripts/infra/backup_db.py --restore           # restore latest
    python scripts/infra/backup_db.py --restore --file gold_20260312_140000.db
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "gold.db"
BACKUP_DIR = PROJECT_ROOT / "backups"
DEFAULT_KEEP = 5
REBUILD_KEEP = 3


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def verify_backup(backup_path: Path) -> bool:
    """Verify a backup file is a valid DuckDB database.

    Checks:
      1. File exists and size > 0
      2. DuckDB can open it read-only
      3. bars_1m table exists (core pipeline table)

    Returns True if valid, False if corrupt.
    """
    if not backup_path.exists():
        print(f"  Verify FAIL: file does not exist: {backup_path}")
        return False

    if backup_path.stat().st_size == 0:
        print(f"  Verify FAIL: file is empty: {backup_path}")
        return False

    try:
        import duckdb

        con = duckdb.connect(str(backup_path), read_only=True)
        try:
            con.execute("SELECT 1 FROM bars_1m LIMIT 0")
        finally:
            con.close()
    except Exception as e:
        print(f"  Verify FAIL: cannot open or query: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# Backup operations
# ---------------------------------------------------------------------------


def backup_db(keep: int = DEFAULT_KEEP) -> Path | None:
    """Create a timestamped rolling backup.

    Returns the backup path on success, None on failure.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return None

    BACKUP_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = BACKUP_DIR / f"gold_{stamp}.db"

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"Backing up {DB_PATH.name} ({size_mb:.1f} MB) -> {dest.name}")
    shutil.copy2(DB_PATH, dest)

    # Verify the backup
    if not verify_backup(dest):
        print(f"  CORRUPT backup detected — deleting {dest.name}")
        dest.unlink(missing_ok=True)
        return None

    print(f"  Done: {dest} (verified)")

    # Prune old rolling backups (exclude rebuild-labeled backups)
    rolling = sorted(
        [p for p in BACKUP_DIR.glob("gold_*.db") if "REBUILD" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in rolling[keep:]:
        old.unlink()
        print(f"  Pruned: {old.name}")

    remaining = len(list(BACKUP_DIR.glob("gold_*.db")))
    print(f"  {remaining} backup(s) retained")
    return dest


def labeled_backup(rebuild_id: str, keep: int = REBUILD_KEEP) -> Path | None:
    """Create a rebuild-labeled backup after successful rebuild.

    Format: gold_REBUILD_{rebuild_id[:8]}_{timestamp}.db
    These are kept separately from rolling backups with their own retention.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return None

    BACKUP_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag = rebuild_id[:8]
    dest = BACKUP_DIR / f"gold_REBUILD_{tag}_{stamp}.db"

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"Labeled backup ({size_mb:.1f} MB) -> {dest.name}")
    shutil.copy2(DB_PATH, dest)

    if not verify_backup(dest):
        print(f"  CORRUPT labeled backup — deleting {dest.name}")
        dest.unlink(missing_ok=True)
        return None

    print(f"  Done: {dest} (verified)")

    # Prune old rebuild backups
    rebuild_backups = sorted(
        BACKUP_DIR.glob("gold_REBUILD_*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in rebuild_backups[keep:]:
        old.unlink()
        print(f"  Pruned rebuild backup: {old.name}")

    return dest


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def list_backups() -> list[dict]:
    """List all available backups with metadata.

    Returns list of dicts with keys: name, path, size_mb, modified, type.
    Sorted newest-first.
    """
    if not BACKUP_DIR.exists():
        return []

    backups = []
    for p in sorted(BACKUP_DIR.glob("gold_*.db"), key=lambda x: x.stat().st_mtime, reverse=True):
        backups.append(
            {
                "name": p.name,
                "path": p,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "type": "rebuild" if "REBUILD" in p.name else "rolling",
            }
        )
    return backups


def restore_db(target_file: str | None = None) -> bool:
    """Restore gold.db from a backup.

    Args:
        target_file: Specific backup filename (e.g. "gold_20260312_140000.db").
                     If None, restores the most recent backup.

    Process:
      1. Find the backup to restore from
      2. Verify the backup is valid
      3. Snapshot current gold.db as gold_pre_restore_{stamp}.db (safety net)
      4. Copy backup -> gold.db
      5. Verify the restored DB

    Returns True on success, False on failure.
    """
    if not BACKUP_DIR.exists():
        print("No backups directory found")
        return False

    # Find backup to restore
    if target_file:
        source = BACKUP_DIR / target_file
        if not source.exists():
            print(f"Backup not found: {source}")
            print("Available backups:")
            for b in list_backups():
                print(f"  {b['name']} ({b['size_mb']} MB, {b['modified']}, {b['type']})")
            return False
    else:
        backups = sorted(BACKUP_DIR.glob("gold_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not backups:
            print("No backups available")
            return False
        source = backups[0]
        print(f"Restoring from most recent: {source.name}")

    # Verify backup before restoring
    print(f"Verifying backup: {source.name}")
    if not verify_backup(source):
        print("  Backup is corrupt — aborting restore")
        return False
    print("  Backup verified OK")

    # Safety snapshot of current DB
    if DB_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety = BACKUP_DIR / f"gold_pre_restore_{stamp}.db"
        print(f"Safety snapshot: {DB_PATH.name} -> {safety.name}")
        shutil.copy2(DB_PATH, safety)

    # Restore
    size_mb = source.stat().st_size / (1024 * 1024)
    print(f"Restoring: {source.name} ({size_mb:.1f} MB) -> {DB_PATH.name}")
    shutil.copy2(source, DB_PATH)

    # Verify restored DB
    if not verify_backup(DB_PATH):
        print("  CRITICAL: Restored DB failed verification!")
        return False

    print("  Restore complete and verified")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backup and restore gold.db")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--backup", action="store_true", default=True, help="Create rolling backup (default)")
    actions.add_argument("--restore", action="store_true", help="Restore from backup")
    actions.add_argument("--list", action="store_true", dest="list_backups", help="List available backups")

    parser.add_argument(
        "--keep", type=int, default=DEFAULT_KEEP, help=f"Number of rolling backups to retain (default: {DEFAULT_KEEP})"
    )
    parser.add_argument("--file", type=str, default=None, help="Specific backup file to restore (with --restore)")

    args = parser.parse_args()

    if args.list_backups:
        backups = list_backups()
        if not backups:
            print("No backups found")
            sys.exit(0)
        print(f"{'Name':<50} {'Size':>8} {'Modified':<20} {'Type':<8}")
        print("-" * 90)
        for b in backups:
            print(f"{b['name']:<50} {b['size_mb']:>6.1f}MB {b['modified']:<20} {b['type']:<8}")
        sys.exit(0)

    if args.restore:
        ok = restore_db(target_file=args.file)
        sys.exit(0 if ok else 1)

    # Default: rolling backup
    result = backup_db(keep=args.keep)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
