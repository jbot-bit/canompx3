#!/usr/bin/env python3
"""
Database backup — copies gold.db to backups/ with date stamp.

Keeps last N copies (default 5), deletes older.

Usage:
    python scripts/backup_db.py
    python scripts/backup_db.py --keep 10
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "gold.db"
BACKUP_DIR = PROJECT_ROOT / "backups"
DEFAULT_KEEP = 5


def backup_db(keep: int = DEFAULT_KEEP) -> Path | None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return None

    BACKUP_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"gold_{stamp}.db"

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"Backing up {DB_PATH.name} ({size_mb:.1f} MB) -> {dest.name}")
    shutil.copy2(DB_PATH, dest)
    print(f"  Done: {dest}")

    # Prune old backups
    backups = sorted(BACKUP_DIR.glob("gold_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[keep:]:
        old.unlink()
        print(f"  Pruned: {old.name}")

    remaining = len(list(BACKUP_DIR.glob("gold_*.db")))
    print(f"  {remaining} backup(s) retained")
    return dest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backup gold.db")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help=f"Number of backups to retain (default: {DEFAULT_KEEP})")
    args = parser.parse_args()

    result = backup_db(keep=args.keep)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
