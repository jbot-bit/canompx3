from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

import pytest


def _load_module():
    return importlib.import_module("scripts.tools.sweep_orphan_rings")


def _write_ring(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sweep_empty_dir_returns_no_deletions(tmp_path):
    mod = _load_module()

    deleted = mod.sweep(tmp_path, pid_status=lambda pid: "dead")

    assert deleted == []


def test_sweep_deletes_ring_when_writer_pid_is_provably_dead(tmp_path):
    mod = _load_module()
    ring = tmp_path / "MNQ.json"
    _write_ring(ring, {"symbol": "MNQ", "writer_pid": 4242, "bars": []})

    deleted = mod.sweep(tmp_path, pid_status=lambda pid: "dead")

    assert deleted == [ring]
    assert not ring.exists()


def test_sweep_preserves_ring_when_writer_pid_is_alive(tmp_path):
    mod = _load_module()
    ring = tmp_path / "MES.json"
    _write_ring(ring, {"symbol": "MES", "writer_pid": 5151, "bars": []})

    deleted = mod.sweep(tmp_path, pid_status=lambda pid: "alive")

    assert deleted == []
    assert ring.exists()


def test_sweep_preserves_corrupt_json(tmp_path, caplog):
    mod = _load_module()
    ring = tmp_path / "MGC.json"
    _write_ring(ring, "{not-json")

    with caplog.at_level(logging.INFO):
        deleted = mod.sweep(tmp_path, pid_status=lambda pid: "dead")

    assert deleted == []
    assert ring.exists()
    assert "corrupt JSON" in caplog.text


def test_sweep_preserves_ring_when_writer_pid_missing(tmp_path, caplog):
    mod = _load_module()
    ring = tmp_path / "MNQ.json"
    _write_ring(ring, {"symbol": "MNQ", "bars": []})

    with caplog.at_level(logging.INFO):
        deleted = mod.sweep(tmp_path, pid_status=lambda pid: "dead")

    assert deleted == []
    assert ring.exists()
    assert "missing writer_pid" in caplog.text


def test_sweep_preserves_ring_when_writer_pid_is_non_int(tmp_path, caplog):
    mod = _load_module()
    ring = tmp_path / "MNQ.json"
    _write_ring(ring, {"symbol": "MNQ", "writer_pid": "abc", "bars": []})

    with caplog.at_level(logging.INFO):
        deleted = mod.sweep(tmp_path, pid_status=lambda pid: "dead")

    assert deleted == []
    assert ring.exists()
    assert "ambiguous writer_pid" in caplog.text


@pytest.mark.parametrize("writer_pid", [True, 0, -7])
def test_sweep_preserves_ring_when_writer_pid_is_not_positive_int(tmp_path, caplog, writer_pid):
    mod = _load_module()
    ring = tmp_path / "MGC.json"
    _write_ring(ring, {"symbol": "MGC", "writer_pid": writer_pid, "bars": []})

    with caplog.at_level(logging.INFO):
        deleted = mod.sweep(tmp_path, pid_status=lambda pid: "dead")

    assert deleted == []
    assert ring.exists()
    assert "writer_pid" in caplog.text


def test_main_dry_run_reports_without_deleting(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    ring = tmp_path / "MES.json"
    _write_ring(ring, {"symbol": "MES", "writer_pid": 9090, "bars": []})
    monkeypatch.setattr(mod, "_probe_pid_status", lambda pid: "dead")

    rc = mod.main(["--ring-dir", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "DRY-RUN" in out
    assert str(ring) in out
    assert ring.exists()
