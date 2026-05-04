"""Tests for _crg_context_lines() in .claude/hooks/session-start.py.

Covers:
- Returns 'missing' line when graph DB absent
- Returns 'fresh' when last_updated within 7 days
- Returns 'stale (Nd old)' when last_updated > 7 days
- Returns [] (fail-silent) on malformed DB
"""

from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "session-start.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("session_start_hook", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_graph_db(db_path: Path, last_updated: str | None, n_nodes: int = 100, n_files: int = 20) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, type TEXT)")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        for i in range(n_nodes):
            ntype = "file" if i < n_files else "function"
            cur.execute("INSERT INTO nodes (id, type) VALUES (?, ?)", (i, ntype))
        if last_updated:
            cur.execute(
                "INSERT INTO metadata (key, value) VALUES ('last_updated', ?)",
                (last_updated,),
            )
        conn.commit()
    finally:
        conn.close()


class TestCrgContextLines:
    def test_missing_db_returns_missing_line(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        hook = _load_hook()
        monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)
        lines = hook._crg_context_lines()
        assert len(lines) == 1
        assert "missing" in lines[0]
        assert "code-review-graph build" in lines[0]

    def test_fresh_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        hook = _load_hook()
        monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)
        recent = (datetime.now() - timedelta(days=2)).isoformat()
        _make_graph_db(tmp_path / ".code-review-graph" / "graph.db", recent)
        lines = hook._crg_context_lines()
        assert len(lines) == 1
        assert "fresh" in lines[0]
        assert "100 nodes" in lines[0]
        assert "20 files" in lines[0]

    def test_stale_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        hook = _load_hook()
        monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)
        old = (datetime.now() - timedelta(days=15)).isoformat()
        _make_graph_db(tmp_path / ".code-review-graph" / "graph.db", old)
        lines = hook._crg_context_lines()
        assert len(lines) == 1
        assert "stale" in lines[0]
        assert "d old" in lines[0]

    def test_malformed_db_fail_silent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        hook = _load_hook()
        monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)
        db = tmp_path / ".code-review-graph" / "graph.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_bytes(b"not a valid sqlite file")
        lines = hook._crg_context_lines()
        # Either [] (open fails) or one line with 0 nodes (open succeeds, queries fail).
        # Both are acceptable fail-silent behaviors.
        assert lines == [] or (len(lines) == 1 and "0 nodes" in lines[0])

    def test_no_last_updated_metadata(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        hook = _load_hook()
        monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)
        _make_graph_db(tmp_path / ".code-review-graph" / "graph.db", None)
        lines = hook._crg_context_lines()
        assert len(lines) == 1
        assert "unknown" in lines[0]
