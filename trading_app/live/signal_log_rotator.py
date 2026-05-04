"""Daily rotation helper for live_signals JSONL log files.

R4 fix (Ralph iter 181): live_signals.jsonl was a single unbounded append-only
file. Multi-day signal-only sessions grew it to GB scale. A bare `except OSError`
in `_write_signal_record` swallowed disk-full silently — institutional-rigor.md § 6
("no silent failures") violation.

Design:
- Files are named `live_signals_YYYY-MM-DD.jsonl` where the date is the Brisbane
  trading day (09:00 Brisbane = day boundary, canonical from pipeline.dst).
- All writes go through `SignalLogRotator.write()`. The rotator tracks the active
  trading day and opens a new file when the day rolls.
- Thread safety: safe for single-threaded asyncio (one event loop, no concurrent
  writes). If moved to multi-process, callers must acquire an external lock before
  write(); this is documented here and on the write() method.
- OSError on write: first error per session triggers `_notify_fn` (rate-limited to
  one notify per `DISK_FULL_NOTIFY_WINDOW_SECS` seconds). Subsequent errors within
  the window are logged at WARNING only. Rationale: disk-full is an operator emergency;
  one notify is enough — Telegram spam for every write call would be worse.
- Retention: old files are pruned to keep at most `retention_days` files. Pruning
  is best-effort: failure to prune logs a WARNING and does NOT block the write path.
  Retention days default to env var `LIVE_SIGNALS_RETENTION_DAYS` (default 30).
  Rationale: 30 days at ~100 records/day × 500 bytes/record ≈ 1.5 MB — trivial.
  Paranoid bound: even at 10k records/day × 1 KB/record = 10 MB/day → 300 MB/month,
  still well within any disk budget. Operator may increase via env var.
"""

import logging
import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# One notify per this many seconds for disk-full errors (institutional-rigor.md § 6 —
# no silent failures; rate-limit so Telegram is not flooded on every write call).
# Rationale: 300s = 5 minutes — enough for operator to notice and act; not so short
# that a sustained disk-full event fires dozens of Telegram messages.
DISK_FULL_NOTIFY_WINDOW_SECS: float = 300.0

# File stem and suffix for signal log partitions.
_FILE_STEM = "live_signals"
_FILE_SUFFIX = ".jsonl"


def _default_retention_days() -> int:
    """Read retention days from env, default 30.

    Configurable via LIVE_SIGNALS_RETENTION_DAYS. The 30-day default is generous:
    at 10k records/day × 1 KB each = 10 MB/day → 300 MB/month — trivial on any
    modern disk. Operator may lower via env for tighter storage budgets.
    """
    raw = os.environ.get("LIVE_SIGNALS_RETENTION_DAYS", "30")
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("LIVE_SIGNALS_RETENTION_DAYS=%r is not an integer; using default 30", raw)
        return 30


def signals_file_for_day(log_dir: Path, trading_day: date) -> Path:
    """Return the canonical path for a given trading day's signal log.

    Exposed at module level so trade_matcher and Live Monitor UI can compute
    the expected filename without instantiating a rotator.
    """
    return log_dir / f"{_FILE_STEM}_{trading_day.isoformat()}{_FILE_SUFFIX}"


class SignalLogRotator:
    """Writes signal records to daily-partitioned JSONL files.

    Usage::

        rotator = SignalLogRotator(
            log_dir=Path("..."),
            trading_day_fn=lambda: self.trading_day,
            notify_fn=self._notify,
        )
        rotator.write('{"ts": "...", ...}\\n')

    Thread safety: designed for single-threaded asyncio. If used from multiple
    processes or threads, the caller is responsible for external locking.
    """

    def __init__(
        self,
        log_dir: Path,
        trading_day_fn: Callable[[], date],
        notify_fn: Callable[[str], None],
        retention_days: int | None = None,
    ) -> None:
        self._log_dir = log_dir
        self._trading_day_fn = trading_day_fn
        self._notify_fn = notify_fn
        self._retention_days = retention_days if retention_days is not None else _default_retention_days()
        # Last successfully-written trading day — used to detect rotation.
        self._active_day: date | None = None
        # UTC timestamp of last OSError notification (rate-limit guard).
        self._last_oserror_notify_at: datetime | None = None
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def current_file(self) -> Path:
        """Return the path for the current trading day's signal log."""
        return signals_file_for_day(self._log_dir, self._trading_day_fn())

    def write(self, line: str) -> None:
        """Append `line` to today's signal log file.

        Handles rotation automatically when `trading_day_fn()` returns a new date.
        On OSError: notifies operator via `notify_fn` (rate-limited to one call per
        DISK_FULL_NOTIFY_WINDOW_SECS) then logs at WARNING. Does NOT raise — the
        write path must never kill the trading loop (institutional-rigor.md § 6).
        """
        current_day = self._trading_day_fn()
        if current_day != self._active_day:
            # Trading day rolled — switch to new file and prune old files.
            self._active_day = current_day
            self._prune_old_files()

        target = signals_file_for_day(self._log_dir, current_day)
        try:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            self._handle_oserror(exc, target)

    def _handle_oserror(self, exc: OSError, target: Path) -> None:
        """Rate-limited notify + log for OSError on signal write.

        First error in the rate-limit window: log.error + notify operator.
        Subsequent errors within the window: log.warning only (no Telegram spam).

        institutional-rigor.md § 6: silent failures are banned. This is the
        non-silent replacement for the previous bare `except OSError: log.warning`.
        """
        now = datetime.now(UTC)
        within_window = (
            self._last_oserror_notify_at is not None
            and (now - self._last_oserror_notify_at).total_seconds() < DISK_FULL_NOTIFY_WINDOW_SECS
        )
        log.error("Could not write signal record to %s: %s", target, exc)
        if not within_window:
            self._last_oserror_notify_at = now
            try:
                self._notify_fn(
                    f"DISK FULL / SIGNAL LOG ERROR: cannot write {target.name}: {exc}. "
                    "Signal log suspended until disk space restored."
                )
            except Exception as notify_exc:
                # _notify itself must not kill the loop — defensive wrap.
                log.error("Failed to send disk-full notification: %s", notify_exc)
        else:
            log.warning("Signal log write error suppressed (within notify window): %s", exc)

    def _list_partition_files(self) -> list[Path]:
        """Return all signal log partition files in `_log_dir`, sorted ascending.

        Extracted from _prune_old_files so tests can inject failure modes by
        patching this method (Path.glob itself is a read-only descriptor on
        Windows and cannot be patched on a Path instance).
        """
        pattern = f"{_FILE_STEM}_*{_FILE_SUFFIX}"
        return sorted(self._log_dir.glob(pattern))

    def _prune_old_files(self) -> None:
        """Delete signal log files older than `retention_days`.

        Best-effort: failures log a WARNING and do NOT block the write path.
        Glob pattern matches only our own file stem to avoid touching unrelated files.
        """
        try:
            files = self._list_partition_files()
            # Keep the `retention_days` newest files; delete the rest.
            if len(files) > self._retention_days:
                for old_file in files[: -self._retention_days]:
                    try:
                        old_file.unlink()
                        log.info("Signal log rotation: pruned %s", old_file.name)
                    except OSError as exc:
                        log.warning(
                            "Signal log rotation: could not prune %s: %s",
                            old_file.name,
                            exc,
                        )
        except Exception as exc:
            # Do not let a glob failure block the write path.
            log.warning("Signal log rotation: pruning scan failed: %s", exc)
