"""Tests for scripts.tools.profile_check_drift."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import profile_check_drift


class DummyConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummyDuckDB:
    def __init__(self, connection: DummyConnection) -> None:
        self.connection = connection

    def connect(self, path: str, *, read_only: bool) -> DummyConnection:
        assert path
        assert read_only is True
        return self.connection


def test_run_profile_returns_sorted_machine_readable_timings(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "gold.db"
    db_path.write_text("", encoding="utf-8")
    connection = DummyConnection()
    seen: list[str] = []

    def no_db_check() -> None:
        seen.append("no-db")

    def db_check(*, con: DummyConnection | None) -> None:
        assert con is connection
        seen.append("db")

    perf_values = iter([0.0, 0.0, 0.3, 0.3, 1.8, 1.8])

    monkeypatch.setattr(profile_check_drift.check_drift, "_import_duckdb_or_exit", lambda: DummyDuckDB(connection))
    monkeypatch.setattr(profile_check_drift.check_drift, "_get_db_path", lambda: db_path)
    monkeypatch.setattr(
        profile_check_drift.check_drift,
        "CHECKS",
        (
            ("short check", no_db_check, False, False),
            ("long db check", db_check, False, True),
        ),
    )
    monkeypatch.setattr(profile_check_drift.time, "perf_counter", lambda: next(perf_values))

    profile = profile_check_drift._run_profile()

    assert seen == ["no-db", "db"]
    assert connection.closed is True
    assert profile["check_count"] == 2
    assert profile["total_seconds"] == 1.8
    assert profile["slow_count"] == 2
    assert profile["over_1s_count"] == 1
    assert [item["label"] for item in profile["checks"]] == ["long db check", "short check"]
    assert profile["checks"][0]["requires_db"] is True


def test_main_json_emits_parseable_contract(monkeypatch, capsys) -> None:
    def noisy_profile() -> dict[str, object]:
        print("incidental profiler noise")
        return {
            "total_seconds": 0.5,
            "check_count": 1,
            "slow_threshold_seconds": 0.2,
            "slow_count": 1,
            "slow_sum_seconds": 0.5,
            "over_1s_count": 0,
            "over_1s_sum_seconds": 0,
            "checks": [
                {
                    "duration_seconds": 0.5,
                    "label": "sample",
                    "requires_db": False,
                    "status": "ok",
                }
            ],
        }

    monkeypatch.setattr(profile_check_drift, "_run_profile", noisy_profile)

    assert profile_check_drift.main(["--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["check_count"] == 1
    assert payload["checks"][0]["label"] == "sample"
    assert "incidental profiler noise" in captured.err


def test_main_text_keeps_human_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        profile_check_drift,
        "_run_profile",
        lambda: {
            "total_seconds": 0.5,
            "check_count": 1,
            "slow_threshold_seconds": 0.2,
            "slow_count": 1,
            "slow_sum_seconds": 0.5,
            "over_1s_count": 0,
            "over_1s_sum_seconds": 0,
            "checks": [
                {
                    "duration_seconds": 0.5,
                    "label": "sample",
                    "requires_db": True,
                    "status": "ok",
                }
            ],
        },
    )

    assert profile_check_drift.main([]) == 0
    out = capsys.readouterr().out

    assert "Total wall time: 0.50s across 1 checks" in out
    assert "sample" in out
    assert "checks exceed 200ms" in out
