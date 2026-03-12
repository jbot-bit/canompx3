"""Tests for scripts/infra/backup_db.py — backup, verify, restore, labeled."""

import shutil

import duckdb
import pytest

from scripts.infra.backup_db import (
    BACKUP_DIR,
    backup_db,
    labeled_backup,
    list_backups,
    restore_db,
    verify_backup,
)


@pytest.fixture
def fake_db(tmp_path, monkeypatch):
    """Create a minimal but valid DuckDB at the expected path and patch module paths."""
    import scripts.infra.backup_db as mod

    db_path = tmp_path / "gold.db"
    backup_dir = tmp_path / "backups"

    # Create a valid DuckDB with bars_1m
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT, source_symbol TEXT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("INSERT INTO bars_1m VALUES ('2024-01-01', 'MGC', 100,101,99,100.5, 50, 'GC')")
    con.close()

    monkeypatch.setattr(mod, "DB_PATH", db_path)
    monkeypatch.setattr(mod, "BACKUP_DIR", backup_dir)

    return db_path, backup_dir


def test_verify_backup_valid(fake_db):
    """verify_backup returns True for a valid DuckDB file."""
    db_path, _ = fake_db
    assert verify_backup(db_path)


def test_verify_backup_missing(tmp_path):
    """verify_backup returns False for nonexistent file."""
    assert not verify_backup(tmp_path / "nonexistent.db")


def test_verify_backup_empty(tmp_path):
    """verify_backup returns False for empty file."""
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    assert not verify_backup(empty)


def test_verify_backup_corrupt(tmp_path):
    """verify_backup returns False for corrupt file."""
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a database")
    assert not verify_backup(corrupt)


def test_backup_db_creates_verified_copy(fake_db):
    """backup_db creates a timestamped copy and verifies it."""
    db_path, backup_dir = fake_db
    result = backup_db(keep=3)
    assert result is not None
    assert result.exists()
    assert result.parent == backup_dir
    assert result.name.startswith("gold_")
    # Verify the backup is valid
    assert verify_backup(result)


def test_backup_db_prunes_old(fake_db):
    """backup_db keeps only the specified number of rolling backups."""
    for _ in range(5):
        backup_db(keep=2)
    _, backup_dir = fake_db
    rolling = [p for p in backup_dir.glob("gold_*.db") if "REBUILD" not in p.name]
    assert len(rolling) == 2


def test_labeled_backup(fake_db):
    """labeled_backup creates a REBUILD-tagged backup."""
    result = labeled_backup("test-rebuild-id-12345678")
    assert result is not None
    assert "REBUILD" in result.name
    assert "test-reb" in result.name  # first 8 chars of rebuild_id
    assert verify_backup(result)


def test_labeled_backup_prunes_old(fake_db):
    """labeled_backup keeps only the specified number of rebuild backups."""
    for i in range(5):
        labeled_backup(f"rebuild-{i:08d}", keep=2)
    _, backup_dir = fake_db
    rebuild_backups = list(backup_dir.glob("gold_REBUILD_*.db"))
    assert len(rebuild_backups) == 2


def test_list_backups(fake_db):
    """list_backups returns metadata for all backups."""
    backup_db()
    labeled_backup("test-list-id")
    backups = list_backups()
    assert len(backups) == 2
    types = {b["type"] for b in backups}
    assert "rolling" in types
    assert "rebuild" in types
    for b in backups:
        assert "name" in b
        assert "size_mb" in b
        assert "modified" in b


def test_restore_db_from_latest(fake_db):
    """restore_db restores from the most recent backup."""
    db_path, backup_dir = fake_db
    # Create a backup
    backup_db()

    # Corrupt the DB
    db_path.write_bytes(b"corrupted")
    assert not verify_backup(db_path)

    # Restore
    ok = restore_db()
    assert ok
    assert verify_backup(db_path)

    # Safety snapshot should exist
    safety = list(backup_dir.glob("gold_pre_restore_*.db"))
    assert len(safety) == 1


def test_restore_db_specific_file(fake_db):
    """restore_db can restore a specific named backup."""
    db_path, backup_dir = fake_db
    result = backup_db()
    backup_name = result.name

    # Corrupt
    db_path.write_bytes(b"corrupted")

    ok = restore_db(target_file=backup_name)
    assert ok
    assert verify_backup(db_path)


def test_restore_db_nonexistent_file(fake_db):
    """restore_db returns False for a nonexistent backup file."""
    ok = restore_db(target_file="nonexistent.db")
    assert not ok


def test_restore_db_no_backups(fake_db):
    """restore_db returns False when no backups exist."""
    ok = restore_db()
    assert not ok
