from __future__ import annotations

from pathlib import Path

import pytest


def test_open_read_only_with_retry_succeeds_first_attempt(monkeypatch):
    import duckdb as _duckdb

    from pipeline import db_connect

    sentinel = object()
    calls: list[tuple[str, bool | None]] = []

    def fake_connect(path: str, *, read_only: bool | None = None):
        calls.append((path, read_only))
        return sentinel

    monkeypatch.setattr(_duckdb, "connect", fake_connect)

    conn = db_connect.open_read_only_with_retry(Path("/tmp/gold.db"), attempts=3)

    assert conn is sentinel
    assert calls == [(str(Path("/tmp/gold.db")), True)]


def test_open_read_only_with_retry_retries_lock_class_ioexception(monkeypatch):
    import duckdb as _duckdb

    from pipeline import db_connect

    sentinel = object()
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_connect(path: str, *, read_only: bool | None = None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _duckdb.IOException("IO Error: gold.db being used by another process")
        assert read_only is True
        return sentinel

    monkeypatch.setattr(_duckdb, "connect", fake_connect)
    monkeypatch.setattr(db_connect.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(db_connect.random, "random", lambda: 0.5)

    conn = db_connect.open_read_only_with_retry("ignored.db", attempts=4, base_delay=0.1, max_delay=1.0)

    assert conn is sentinel
    assert calls["n"] == 3
    assert sleeps == [0.1, 0.2]


def test_open_read_only_with_retry_raises_last_lock_exception(monkeypatch):
    import duckdb as _duckdb

    from pipeline import db_connect

    calls = {"n": 0}

    def fake_connect(path: str, *, read_only: bool | None = None):
        calls["n"] += 1
        raise _duckdb.IOException(f"could not set lock on file gold.db (call {calls['n']})")

    monkeypatch.setattr(_duckdb, "connect", fake_connect)
    monkeypatch.setattr(db_connect.time, "sleep", lambda _delay: None)

    with pytest.raises(_duckdb.IOException, match="call 3"):
        db_connect.open_read_only_with_retry("ignored.db", attempts=3, base_delay=0.1, max_delay=1.0)

    assert calls["n"] == 3


def test_open_read_only_with_retry_reraises_non_lock_ioexception(monkeypatch):
    import duckdb as _duckdb

    from pipeline import db_connect

    calls = {"n": 0}

    def fake_connect(path: str, *, read_only: bool | None = None):
        calls["n"] += 1
        raise _duckdb.IOException("IO Error: Permission denied on gold.db")

    monkeypatch.setattr(_duckdb, "connect", fake_connect)

    with pytest.raises(_duckdb.IOException, match="Permission denied"):
        db_connect.open_read_only_with_retry("ignored.db", attempts=3)

    assert calls["n"] == 1


def test_open_writer_with_retry_uses_default_write_mode(monkeypatch):
    import duckdb as _duckdb

    from pipeline import db_connect

    received: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_connect(*args, **kwargs):
        received.append((args, kwargs))
        return object()

    monkeypatch.setattr(_duckdb, "connect", fake_connect)

    db_connect.open_writer_with_retry(Path("/tmp/gold.db"), attempts=1)

    assert received == [((str(Path("/tmp/gold.db")),), {})]
