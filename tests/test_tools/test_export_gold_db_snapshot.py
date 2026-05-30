"""Tests for the repo-native gold.db snapshot exporter."""

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from scripts.tools.export_gold_db_snapshot import export_snapshot


@pytest.fixture
def snapshot_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "gold.db"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE daily_features (trading_day DATE, symbol VARCHAR)")
    con.execute("INSERT INTO daily_features VALUES ('2026-05-29', 'MNQ')")
    con.execute("CREATE TABLE validated_setups (strategy_id VARCHAR, instrument VARCHAR)")
    con.execute("INSERT INTO validated_setups VALUES ('S1', 'MNQ')")
    con.close()
    return db_path


def test_export_snapshot_writes_parquet_and_manifest(snapshot_db: Path, tmp_path: Path) -> None:
    root = tmp_path / "approved_snapshots"
    now = datetime.fromtimestamp(snapshot_db.stat().st_mtime, tz=UTC)

    manifest = export_snapshot(
        db_path=snapshot_db,
        output_dir=root / "run-1",
        snapshot_root=root,
        tables=["daily_features", "validated_setups"],
        now=now,
    )

    manifest_path = root / "run-1" / "manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["snapshot_id"] == "run-1"
    assert manifest["source_db"]["path"] == str(snapshot_db)
    assert manifest["tables"]["daily_features"]["row_count"] == 1
    assert manifest["horizon"]["daily_features"]["max_trading_day"] == "2026-05-29"
    assert (root / "run-1" / "daily_features" / "daily_features.parquet").exists()


def test_export_snapshot_refuses_output_outside_approved_root(snapshot_db: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="approved snapshot root"):
        export_snapshot(
            db_path=snapshot_db,
            output_dir=tmp_path / "outside",
            snapshot_root=tmp_path / "approved",
            tables=["daily_features"],
        )


def test_export_snapshot_refuses_unapproved_tables(snapshot_db: Path, tmp_path: Path) -> None:
    root = tmp_path / "approved"

    with pytest.raises(ValueError, match="not approved"):
        export_snapshot(
            db_path=snapshot_db,
            output_dir=root / "run-1",
            snapshot_root=root,
            tables=["paper_trades"],
        )


def test_export_snapshot_refuses_stale_source_db(snapshot_db: Path, tmp_path: Path) -> None:
    root = tmp_path / "approved"
    stale_now = datetime.fromtimestamp(snapshot_db.stat().st_mtime, tz=UTC).replace(year=2027)

    with pytest.raises(ValueError, match="stale"):
        export_snapshot(
            db_path=snapshot_db,
            output_dir=root / "run-1",
            snapshot_root=root,
            tables=["daily_features"],
            now=stale_now,
            max_age_hours=1,
        )


def test_export_snapshot_refuses_deprecated_scratch_db(tmp_path: Path) -> None:
    root = tmp_path / "approved"

    with pytest.raises(ValueError, match="deprecated scratch DB"):
        export_snapshot(
            db_path=Path("C:/db/gold.db"),
            output_dir=root / "run-1",
            snapshot_root=root,
            tables=["daily_features"],
        )
