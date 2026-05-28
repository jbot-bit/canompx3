"""Tests for the canonical planned-launch surface.

Covers:
- write/read roundtrip with profile lookup
- mode precedence (SIGNAL/DEMO/LIVE) — guards against the bot_dashboard.html:4060
  ternary class of bug by asserting that the WRITER respects the caller's mode
- staleness handling
- fail-visible behavior for missing / malformed / schema-mismatched files
- invalid input refused with ValueError
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.live import planned_launch


@pytest.fixture
def tmp_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect PLANNED_LAUNCH_PATH to a tmp file for the duration of the test."""
    target = tmp_path / "bot_planned_launch.json"
    monkeypatch.setattr(planned_launch, "PLANNED_LAUNCH_PATH", target)
    return target


def test_write_then_read_roundtrip(tmp_artifact: Path) -> None:
    payload = planned_launch.write_planned_launch(
        profile_id="topstep_50k_mnq_auto",
        mode="DEMO",
        source="CLI",
    )
    assert payload["profile_id"] == "topstep_50k_mnq_auto"
    assert payload["mode"] == "DEMO"
    assert payload["source"] == "CLI"
    assert payload["schema_version"] == 1
    assert payload["copies"] >= 1
    assert isinstance(payload["instruments"], list)
    assert payload["broker_accounts_count"] == payload["copies"]

    read_back = planned_launch.read_planned_launch()
    assert read_back["status"] == "ok"
    assert read_back["mode"] == "DEMO"
    assert read_back["profile_id"] == "topstep_50k_mnq_auto"
    assert read_back["age_seconds"] < 5


@pytest.mark.parametrize("mode", ["SIGNAL", "DEMO", "LIVE"])
def test_all_three_modes_writable_and_distinguishable(tmp_artifact: Path, mode: str) -> None:
    """Mode-precedence guard: every mode must be writable AND read back as itself.

    Counterexample to the bot_dashboard.html:4060 inverted ternary — the
    canonical surface must NOT collapse SIGNAL→DEMO or hide LIVE.
    """
    planned_launch.write_planned_launch(
        profile_id="topstep_50k_mnq_auto",
        mode=mode,
        source="CLI",
    )
    read_back = planned_launch.read_planned_launch()
    assert read_back["status"] == "ok"
    assert read_back["mode"] == mode, f"mode {mode!r} round-tripped as {read_back['mode']!r}"


def test_mode_normalized_to_uppercase(tmp_artifact: Path) -> None:
    payload = planned_launch.write_planned_launch(
        profile_id="topstep_50k_mnq_auto",
        mode="live",
        source="CLI",
    )
    assert payload["mode"] == "LIVE"


def test_invalid_mode_rejected(tmp_artifact: Path) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        planned_launch.write_planned_launch(
            profile_id="topstep_50k_mnq_auto",
            mode="paper",
            source="CLI",
        )


def test_invalid_source_rejected(tmp_artifact: Path) -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        planned_launch.write_planned_launch(
            profile_id="topstep_50k_mnq_auto",
            mode="DEMO",
            source="some_other_launcher",
        )


def test_unknown_profile_rejected(tmp_artifact: Path) -> None:
    with pytest.raises(ValueError, match="not in ACCOUNT_PROFILES"):
        planned_launch.write_planned_launch(
            profile_id="nonexistent_profile_xyz",
            mode="DEMO",
            source="CLI",
        )


def test_unknown_profile_allowed_when_copies_and_instruments_supplied(tmp_artifact: Path) -> None:
    """Bypass for tests / non-registered launchers — caller takes responsibility."""
    payload = planned_launch.write_planned_launch(
        profile_id="adhoc_test",
        mode="SIGNAL",
        source="CLI",
        copies=1,
        instruments=["MNQ"],
    )
    assert payload["profile_id"] == "adhoc_test"
    assert payload["copies"] == 1
    assert payload["instruments"] == ["MNQ"]


def test_read_missing_returns_unknown(tmp_artifact: Path) -> None:
    assert not tmp_artifact.exists()
    result = planned_launch.read_planned_launch()
    assert result["status"] == "unknown"
    assert "no planned launch file" in result["reason"]


def test_read_malformed_json_returns_unknown(tmp_artifact: Path) -> None:
    tmp_artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact.write_text("{not valid json", encoding="utf-8")
    result = planned_launch.read_planned_launch()
    assert result["status"] == "unknown"
    assert "read_error" in result["reason"]


def test_read_schema_mismatch_returns_unknown(tmp_artifact: Path) -> None:
    tmp_artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact.write_text(
        json.dumps({"schema_version": 999, "mode": "DEMO"}),
        encoding="utf-8",
    )
    result = planned_launch.read_planned_launch()
    assert result["status"] == "unknown"
    assert "schema_version mismatch" in result["reason"]


def test_read_missing_required_fields_returns_unknown(tmp_artifact: Path) -> None:
    tmp_artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact.write_text(
        json.dumps({"schema_version": 1, "mode": "DEMO"}),
        encoding="utf-8",
    )
    result = planned_launch.read_planned_launch()
    assert result["status"] == "unknown"
    assert "missing fields" in result["reason"]


def test_read_invalid_mode_returns_unknown(tmp_artifact: Path) -> None:
    tmp_artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "x",
                "mode": "PAPER",
                "copies": 1,
                "instruments": ["MNQ"],
                "broker_accounts_count": 1,
                "source": "CLI",
                "ts": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    result = planned_launch.read_planned_launch()
    assert result["status"] == "unknown"
    assert "invalid mode" in result["reason"]


def test_stale_file_flagged(tmp_artifact: Path) -> None:
    """A file older than STALE_AFTER_SECONDS must read as stale, never as ok."""
    stale_ts = (datetime.now(UTC) - timedelta(seconds=planned_launch.STALE_AFTER_SECONDS + 60)).isoformat()
    tmp_artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "topstep_50k_mnq_auto",
                "mode": "LIVE",
                "copies": 3,
                "instruments": ["MNQ"],
                "broker_accounts_count": 3,
                "source": "START_BOT.bat",
                "ts": stale_ts,
            }
        ),
        encoding="utf-8",
    )
    result = planned_launch.read_planned_launch()
    assert result["status"] == "stale"
    assert result["mode"] == "LIVE"
    assert result["age_seconds"] > planned_launch.STALE_AFTER_SECONDS


def test_clear_removes_file(tmp_artifact: Path) -> None:
    planned_launch.write_planned_launch(
        profile_id="topstep_50k_mnq_auto",
        mode="DEMO",
        source="CLI",
    )
    assert tmp_artifact.exists()
    planned_launch.clear_planned_launch()
    assert not tmp_artifact.exists()
    planned_launch.clear_planned_launch()  # idempotent — must not raise


def test_clear_when_missing_is_noop(tmp_artifact: Path) -> None:
    assert not tmp_artifact.exists()
    planned_launch.clear_planned_launch()  # no exception


def test_atomic_write_no_partial_file(tmp_artifact: Path) -> None:
    """The .json.tmp staging file must be replaced atomically — never left behind."""
    planned_launch.write_planned_launch(
        profile_id="topstep_50k_mnq_auto",
        mode="SIGNAL",
        source="CLI",
    )
    assert tmp_artifact.exists()
    assert not tmp_artifact.with_suffix(".json.tmp").exists()


def test_broker_accounts_count_defaults_to_copies(tmp_artifact: Path) -> None:
    payload = planned_launch.write_planned_launch(
        profile_id="adhoc",
        mode="LIVE",
        source="CLI",
        copies=3,
        instruments=["MNQ"],
    )
    assert payload["copies"] == 3
    assert payload["broker_accounts_count"] == 3


def test_broker_accounts_count_explicit_overrides_copies(tmp_artifact: Path) -> None:
    """Lets preflight refine the count after broker discovery."""
    payload = planned_launch.write_planned_launch(
        profile_id="adhoc",
        mode="LIVE",
        source="CLI",
        copies=3,
        instruments=["MNQ"],
        broker_accounts_count=2,
    )
    assert payload["copies"] == 3
    assert payload["broker_accounts_count"] == 2
