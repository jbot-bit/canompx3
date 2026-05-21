"""Stage 1 tests for cockpit-v3: /api/signals-recent endpoint + _read_recent_signals helper.

Covers:
- empty case (no file today) returns []
- since_ts filter drops older records
- limit cap honored
- malformed JSON line skipped, not raised
- bad since_ts gracefully ignored
- legacy snapshot HTML file present (rollback guarantee)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_today_jsonl(tmp_dir: Path, lines: list[str]) -> Path:
    today = datetime.now(UTC).date()
    path = tmp_dir / f"live_signals_{today.isoformat()}.jsonl"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def test_read_recent_signals_returns_empty_when_no_file(monkeypatch, tmp_path):
    from trading_app.live import bot_dashboard as bd

    monkeypatch.setattr(bd, "SIGNALS_DIR", tmp_path)
    assert bd._read_recent_signals() == []


def test_read_recent_signals_returns_newest_first(monkeypatch, tmp_path):
    from trading_app.live import bot_dashboard as bd

    rec_old = {"ts": "2026-01-01T00:00:00+00:00", "type": "A"}
    rec_new = {"ts": "2026-01-02T00:00:00+00:00", "type": "B"}
    _write_today_jsonl(tmp_path, [json.dumps(rec_old), json.dumps(rec_new)])

    monkeypatch.setattr(bd, "SIGNALS_DIR", tmp_path)
    out = bd._read_recent_signals()
    assert len(out) == 2
    assert out[0]["type"] == "B"  # newest first
    assert out[1]["type"] == "A"


def test_read_recent_signals_since_filter_drops_older(monkeypatch, tmp_path):
    from trading_app.live import bot_dashboard as bd

    rec_old = {"ts": "2026-01-01T00:00:00+00:00", "type": "A"}
    rec_new = {"ts": "2026-01-02T00:00:00+00:00", "type": "B"}
    _write_today_jsonl(tmp_path, [json.dumps(rec_old), json.dumps(rec_new)])

    monkeypatch.setattr(bd, "SIGNALS_DIR", tmp_path)
    out = bd._read_recent_signals(since_ts="2026-01-01T12:00:00+00:00")
    assert len(out) == 1
    assert out[0]["type"] == "B"


def test_read_recent_signals_limit_cap(monkeypatch, tmp_path):
    from trading_app.live import bot_dashboard as bd

    recs = [{"ts": f"2026-01-0{i}T00:00:00+00:00", "type": f"R{i}"} for i in range(1, 6)]
    _write_today_jsonl(tmp_path, [json.dumps(r) for r in recs])

    monkeypatch.setattr(bd, "SIGNALS_DIR", tmp_path)
    out = bd._read_recent_signals(limit=2)
    assert len(out) == 2
    assert out[0]["type"] == "R5"  # newest first
    assert out[1]["type"] == "R4"


def test_read_recent_signals_skips_malformed_line(monkeypatch, tmp_path, caplog):
    from trading_app.live import bot_dashboard as bd

    good = {"ts": "2026-01-01T00:00:00+00:00", "type": "OK"}
    _write_today_jsonl(tmp_path, ["{not json", json.dumps(good)])

    monkeypatch.setattr(bd, "SIGNALS_DIR", tmp_path)
    out = bd._read_recent_signals()
    assert len(out) == 1
    assert out[0]["type"] == "OK"


def test_read_recent_signals_bad_since_ts_ignored(monkeypatch, tmp_path):
    from trading_app.live import bot_dashboard as bd

    rec = {"ts": "2026-01-01T00:00:00+00:00", "type": "A"}
    _write_today_jsonl(tmp_path, [json.dumps(rec)])

    monkeypatch.setattr(bd, "SIGNALS_DIR", tmp_path)
    out = bd._read_recent_signals(since_ts="not-a-timestamp")
    assert len(out) == 1  # filter ignored, all records returned


def test_legacy_html_snapshot_exists():
    """Stage 1 rollback guarantee: bot_dashboard_legacy.html must exist as a verbatim snapshot."""
    legacy = REPO_ROOT / "trading_app" / "live" / "bot_dashboard_legacy.html"
    assert legacy.exists(), "Stage 1 rollback file missing — cannot revert UI changes"
    assert legacy.stat().st_size > 100_000, "Legacy snapshot suspiciously small"


def test_api_signals_recent_route_registered():
    from trading_app.live import bot_dashboard as bd

    paths = {getattr(r, "path", None) for r in bd.app.routes}
    assert "/api/signals-recent" in paths
