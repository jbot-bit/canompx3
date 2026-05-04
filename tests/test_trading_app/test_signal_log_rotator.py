"""Tests for SignalLogRotator — R4 (Ralph iter 181).

Coverage:
  1. Daily rollover mid-write: mock clock crosses 09:00 Brisbane, assert two files created.
  2. OSError once → _notify fires; second error within rate window → no double-notify.
  3. Old file cleanup: create 35 daily files, run rotator, assert 30 newest remain.
  4. trade_matcher._load_signals compat: still finds signals after rotation produces daily files.
"""

import json
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from trading_app.live.signal_log_rotator import (
    DISK_FULL_NOTIFY_WINDOW_SECS,
    SignalLogRotator,
    signals_file_for_day,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def log_dir(tmp_path):
    """Temporary directory for signal log files."""
    return tmp_path / "signals"


def _make_rotator(log_dir: Path, trading_day_fn=None, notify_fn=None, retention_days=30):
    if trading_day_fn is None:
        trading_day_fn = lambda: date(2026, 4, 25)
    if notify_fn is None:
        notify_fn = MagicMock()
    return SignalLogRotator(
        log_dir=log_dir,
        trading_day_fn=trading_day_fn,
        notify_fn=notify_fn,
        retention_days=retention_days,
    )


# ---------------------------------------------------------------------------
# Test 1: Daily rollover mid-write
# ---------------------------------------------------------------------------


class TestDailyRollover:
    """Clock crosses 09:00 Brisbane → new file is created, old file preserved."""

    def test_rollover_creates_two_files(self, log_dir):
        """Writing across a trading-day boundary produces two separate files."""
        # Day 1: 2026-04-24 (before 09:00 Brisbane on Apr 25 = still Apr 24 trading day)
        day1 = date(2026, 4, 24)
        day2 = date(2026, 4, 25)
        current_day = [day1]  # mutable container so lambda can update it

        rotator = _make_rotator(log_dir, trading_day_fn=lambda: current_day[0])

        # Write in day1
        rotator.write('{"ts": "2026-04-24T22:59:00Z", "type": "ENTRY"}\n')
        assert signals_file_for_day(log_dir, day1).exists(), "Day1 file must exist after first write"
        assert not signals_file_for_day(log_dir, day2).exists(), "Day2 file must not exist yet"

        # Roll to day2
        current_day[0] = day2
        rotator.write('{"ts": "2026-04-24T23:01:00Z", "type": "EXIT"}\n')

        assert signals_file_for_day(log_dir, day1).exists(), "Day1 file must persist after rollover"
        assert signals_file_for_day(log_dir, day2).exists(), "Day2 file must be created on rollover"

        # Day1 must still contain original record
        day1_lines = signals_file_for_day(log_dir, day1).read_text(encoding="utf-8").strip().split("\n")
        assert len(day1_lines) == 1
        assert json.loads(day1_lines[0])["type"] == "ENTRY"

        # Day2 must contain new record
        day2_lines = signals_file_for_day(log_dir, day2).read_text(encoding="utf-8").strip().split("\n")
        assert len(day2_lines) == 1
        assert json.loads(day2_lines[0])["type"] == "EXIT"

    def test_multiple_writes_same_day_go_to_same_file(self, log_dir):
        """Multiple writes on the same trading day all land in the same file."""
        day = date(2026, 4, 25)
        rotator = _make_rotator(log_dir, trading_day_fn=lambda: day)

        for i in range(5):
            rotator.write(f'{{"n": {i}}}\n')

        target = signals_file_for_day(log_dir, day)
        lines = target.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5, f"Expected 5 records, got {len(lines)}"
        for i, line in enumerate(lines):
            assert json.loads(line)["n"] == i


# ---------------------------------------------------------------------------
# Test 2: OSError notify rate-limiting
# ---------------------------------------------------------------------------


class TestOSErrorNotifyRateLimit:
    """First OSError → _notify called. Second within window → no double-notify."""

    def test_first_oserror_triggers_notify(self, log_dir):
        """First write failure calls notify_fn once."""
        notify = MagicMock()
        rotator = _make_rotator(log_dir, notify_fn=notify)

        with patch("builtins.open", side_effect=OSError("No space left on device")):
            rotator.write('{"type": "ENTRY"}\n')

        notify.assert_called_once()
        call_msg = notify.call_args[0][0]
        assert "DISK FULL" in call_msg or "SIGNAL LOG ERROR" in call_msg, f"Expected disk-full message, got: {call_msg}"

    def test_second_oserror_within_window_no_double_notify(self, log_dir):
        """Second OSError within DISK_FULL_NOTIFY_WINDOW_SECS does not trigger another notify."""
        notify = MagicMock()
        rotator = _make_rotator(log_dir, notify_fn=notify)

        # First error — sets _last_oserror_notify_at
        with patch("builtins.open", side_effect=OSError("disk full")):
            rotator.write('{"type": "A"}\n')

        assert notify.call_count == 1, "First error must notify"

        # Second error in same window — must not notify again
        with patch("builtins.open", side_effect=OSError("disk full")):
            rotator.write('{"type": "B"}\n')

        assert notify.call_count == 1, "Second error within window must NOT double-notify"

    def test_oserror_after_window_triggers_notify_again(self, log_dir):
        """OSError after the rate-limit window expires fires another notify."""
        notify = MagicMock()
        rotator = _make_rotator(log_dir, notify_fn=notify)

        with patch("builtins.open", side_effect=OSError("disk full")):
            rotator.write('{"type": "A"}\n')

        assert notify.call_count == 1

        # Backdate the last notify timestamp so we appear outside the window.
        expired_ts = datetime.now(UTC) - timedelta(seconds=DISK_FULL_NOTIFY_WINDOW_SECS + 1)
        rotator._last_oserror_notify_at = expired_ts

        with patch("builtins.open", side_effect=OSError("disk full again")):
            rotator.write('{"type": "B"}\n')

        assert notify.call_count == 2, "Error after window expiry must notify again"

    def test_successful_write_after_oserror_does_not_notify(self, log_dir):
        """After a successful write following an error, no spurious notify fires."""
        notify = MagicMock()
        rotator = _make_rotator(log_dir, notify_fn=notify)

        with patch("builtins.open", side_effect=OSError("disk full")):
            rotator.write('{"type": "A"}\n')

        assert notify.call_count == 1

        # Successful write — should not notify
        rotator.write('{"type": "B"}\n')
        assert notify.call_count == 1, "Successful write must not trigger additional notify"


# ---------------------------------------------------------------------------
# Test 3: Old file cleanup (retention)
# ---------------------------------------------------------------------------


class TestOldFileCleanup:
    """Create N+5 files, trigger rollover, assert only N newest remain."""

    def _create_dated_files(self, log_dir: Path, n: int, base_date: date) -> list[Path]:
        """Create n daily signal files dated base_date - n + 1 through base_date."""
        created = []
        for i in range(n):
            d = base_date - timedelta(days=n - 1 - i)
            p = signals_file_for_day(log_dir, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"type": "dummy"}\n', encoding="utf-8")
            created.append(p)
        return created

    def test_prune_keeps_retention_days_newest(self, log_dir):
        """With 35 files and retention=30, the 5 oldest are pruned."""
        retention = 30
        total = 35
        base = date(2026, 4, 25)

        self._create_dated_files(log_dir, total, base)

        # Create rotator on the day AFTER base — triggers prune on first write.
        next_day = base + timedelta(days=1)
        current_day = [next_day]
        rotator = _make_rotator(log_dir, trading_day_fn=lambda: current_day[0], retention_days=retention)
        rotator.write('{"type": "start"}\n')

        remaining = sorted(log_dir.glob("live_signals_*.jsonl"))
        # retention files + the new file we just wrote
        assert len(remaining) == retention + 1, (
            f"Expected {retention + 1} files after prune, found {len(remaining)}: "
            + ", ".join(p.name for p in remaining)
        )

        # The 5 oldest must be gone
        oldest_5_cutoff = base - timedelta(days=total - 1) + timedelta(days=4)
        for p in remaining:
            try:
                date_str = p.stem.replace("live_signals_", "")
                file_date = date.fromisoformat(date_str)
                assert file_date > oldest_5_cutoff or file_date == next_day, (
                    f"File {p.name} should have been pruned (date {file_date} <= cutoff {oldest_5_cutoff})"
                )
            except ValueError:
                pass  # Non-dated file — skip

    def test_prune_failure_does_not_block_write(self, log_dir):
        """If glob/unlink fails during pruning, write still succeeds."""
        retention = 2
        base = date(2026, 4, 25)
        self._create_dated_files(log_dir, 5, base)

        next_day = base + timedelta(days=1)
        current_day = [next_day]
        rotator = _make_rotator(log_dir, trading_day_fn=lambda: current_day[0], retention_days=retention)

        # Simulate pruning failure via exception in the partition-listing helper.
        # Patching Path.glob directly fails on Windows (read-only descriptor),
        # so the rotator exposes _list_partition_files() as the patch surface.
        with patch.object(rotator, "_list_partition_files", side_effect=PermissionError("access denied")):
            # Write must succeed despite prune failure
            rotator.write('{"type": "ok"}\n')

        target = signals_file_for_day(log_dir, next_day)
        assert target.exists(), "Write must succeed even when prune fails"

    def test_fewer_files_than_retention_does_not_prune(self, log_dir):
        """When file count is below retention, nothing is deleted."""
        retention = 30
        base = date(2026, 4, 25)
        self._create_dated_files(log_dir, 5, base)

        next_day = base + timedelta(days=1)
        current_day = [next_day]
        rotator = _make_rotator(log_dir, trading_day_fn=lambda: current_day[0], retention_days=retention)
        rotator.write('{"type": "ok"}\n')

        # 5 original + 1 new
        remaining = sorted(log_dir.glob("live_signals_*.jsonl"))
        assert len(remaining) == 6, f"Expected 6 files (no prune), got {len(remaining)}"


# ---------------------------------------------------------------------------
# Test 4: trade_matcher._load_signals compat
# ---------------------------------------------------------------------------


class TestTradeMatcherCompat:
    """_load_signals reads daily-suffix files and returns merged records."""

    def test_load_signals_reads_daily_files(self, tmp_path):
        """_load_signals finds live_signals_YYYY-MM-DD.jsonl files and returns all records."""
        from scripts.tools.trade_matcher import _load_signals

        root = tmp_path

        # Create two daily signal files
        day1 = root / "live_signals_2026-04-24.jsonl"
        day2 = root / "live_signals_2026-04-25.jsonl"
        day1.write_text('{"day": 1, "type": "ENTRY"}\n', encoding="utf-8")
        day2.write_text('{"day": 2, "type": "EXIT"}\n', encoding="utf-8")

        with patch("scripts.tools.trade_matcher.PROJECT_ROOT", root):
            signals = _load_signals()

        assert len(signals) == 2, f"Expected 2 signals, got {len(signals)}: {signals}"
        days = {s["day"] for s in signals}
        assert days == {1, 2}

    def test_load_signals_returns_empty_when_no_files(self, tmp_path):
        """_load_signals returns [] when no signal files exist."""
        from scripts.tools.trade_matcher import _load_signals

        with patch("scripts.tools.trade_matcher.PROJECT_ROOT", tmp_path):
            signals = _load_signals()

        assert signals == []

    def test_load_signals_reads_legacy_file(self, tmp_path):
        """Legacy live_signals.jsonl is still loaded for migration compat."""
        from scripts.tools.trade_matcher import _load_signals

        legacy = tmp_path / "live_signals.jsonl"
        legacy.write_text('{"legacy": true}\n', encoding="utf-8")

        with patch("scripts.tools.trade_matcher.PROJECT_ROOT", tmp_path):
            signals = _load_signals()

        assert len(signals) == 1
        assert signals[0]["legacy"] is True

    def test_load_signals_skips_corrupt_lines(self, tmp_path):
        """Corrupt JSON lines in a daily file are skipped without raising."""
        from scripts.tools.trade_matcher import _load_signals

        f = tmp_path / "live_signals_2026-04-25.jsonl"
        f.write_text('{"ok": 1}\nNOT_JSON\n{"ok": 2}\n', encoding="utf-8")

        with patch("scripts.tools.trade_matcher.PROJECT_ROOT", tmp_path):
            signals = _load_signals()

        assert len(signals) == 2
        assert [s["ok"] for s in signals] == [1, 2]
