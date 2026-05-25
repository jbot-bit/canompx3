"""Live-telemetry maturity gate — 30-trading-day floor (advisory).

The gate counts distinct UTC dates with at least one qualifying uptime
record (SESSION_START, SIGNAL_ENTRY, ORDER_ENTRY, exits, blocks, kill,
circuit-breaker) in ``live_signals_*.jsonl`` for the requested instrument.
At or above ``MIN_TELEMETRY_TRADING_DAYS``, ``verdict = VERDICT_MATURE``;
otherwise ``VERDICT_UNVERIFIED_INSUFFICIENT_TELEMETRY``. The module is a
pure read-and-count primitive — it does not decide blocking semantics.

DOCTRINE NOTE (2026-05-18 — preflight FAIL→WARN demotion):

The original module docstring framed the 30-day floor as the
"operational analog of the C8 power floor" and cited
``pre_registered_criteria.md`` Criterion 8 + ``backtesting-methodology.md``
RULE 3.2/3.3. That framing was an author-asserted analogy, not canonical
doctrine. The cited criteria govern OOS statistical validation on
historical ``orb_outcomes`` data with ``--holdout-date 2026-01-01``; they
do not mandate live-bot uptime accumulation before deployment.

No institutional document at the time of this note requires N>=30 distinct
live trading_days as a deployment precondition. The preflight call-site
(``scripts/run_live_session.py::_check_telemetry_maturity``) therefore
treats this gate as advisory:

- ``--signal-only``: OK / non-blocking (this is the path that accumulates).
- ``--demo``: WARN (no real capital at risk).
- ``--live`` + Express-Funded prop profile (``is_express_funded=True``):
  WARN (Topstep/XFA — funded-account wrapper insulates real capital).
- ``--live`` + real-capital broker (profile with ``is_express_funded=False``,
  unknown profile, or no profile): FAIL — telemetry-maturity stays a hard
  precondition for real-capital live deployment, as a precaution, until
  this gate is either promoted through proper doctrine (authority, scope,
  cadence, failure contract) or formally retired.

This primitive is intentionally preserved so that a future promotion can
re-cite ``evaluate_telemetry_maturity`` from a doctrine-grounded call-site
without re-implementing the scan. Do NOT lower
``MIN_TELEMETRY_TRADING_DAYS`` from this file; if the threshold needs to
change, change it in the doctrine that establishes it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Bright-line floor. Do not loosen without amending pre_registered_criteria.md
# Criterion 8 + RULE 3.3 + this module's grounding section together.
MIN_TELEMETRY_TRADING_DAYS: int = 30

# Signal-log record types that count as evidence the bot was actually up
# during that trading_day. SESSION_START is the canonical uptime marker
# (written once per (instrument, session) when the orchestrator starts a
# session). Other record types (SIGNAL_ENTRY, ORDER_ENTRY, ENTRY_BLOCKED_*,
# SIGNAL_EXIT, KILL_SWITCH) imply uptime but also imply gate activity, so
# they count toward maturity too -- the gate only refuses on the absence of
# any record on that trading_day.
_UPTIME_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "SESSION_START",
        "SIGNAL_ENTRY",
        "ORDER_ENTRY",
        "SIGNAL_EXIT",
        "ORDER_EXIT",
        "ENTRY_BLOCKED_PAUSED",
        "ENTRY_BLOCKED_DD_HALT",
        "ENTRY_BLOCKED_ORPHAN",
        "KILL_SWITCH",
        "CIRCUIT_BREAKER",
    }
)

# Verdict strings. Use these as exact equality compares from callers --
# do not paraphrase or translate.
VERDICT_UNVERIFIED: str = "UNVERIFIED_INSUFFICIENT_TELEMETRY"
VERDICT_MATURE: str = "TELEMETRY_MATURE"


@dataclass(frozen=True)
class TelemetryMaturityReport:
    """Result of an instrument-scoped maturity check.

    Attributes:
        verdict: VERDICT_UNVERIFIED or VERDICT_MATURE -- the bright-line gate.
        instrument: instrument symbol the report covers (e.g. "MNQ").
        profile_id: optional profile id filter used by the scan.
        profile_scoped: True when profile_id filtering was applied.
        n_unique_trading_days: count of distinct trading_days with at least
            one qualifying uptime record.
        min_required: floor used for the gate (MIN_TELEMETRY_TRADING_DAYS).
        trading_days: sorted list of distinct trading_days observed.
        signal_files_scanned: count of jsonl files read.
        records_scanned: count of jsonl lines parsed (malformed lines skipped
            with a debug log; never raise).
        records_qualifying: count of records whose type was in the uptime set
            AND whose instrument matched.
    """

    verdict: str
    instrument: str
    profile_id: str | None
    profile_scoped: bool
    n_unique_trading_days: int
    min_required: int
    trading_days: tuple[date, ...]
    signal_files_scanned: int
    records_scanned: int
    records_qualifying: int

    @property
    def is_mature(self) -> bool:
        return self.verdict == VERDICT_MATURE

    def justification(self) -> str:
        """One-line reason string for inclusion in diagnostic reports."""
        if self.is_mature:
            return (
                f"{self.instrument}: telemetry mature -- "
                f"{self.n_unique_trading_days} distinct trading_days >= floor "
                f"{self.min_required}"
            )
        return (
            f"{self.instrument}: UNVERIFIED_INSUFFICIENT_TELEMETRY -- "
            f"{self.n_unique_trading_days} distinct trading_days < floor "
            f"{self.min_required} (need {self.min_required - self.n_unique_trading_days} more)"
        )


def _trading_day_from_iso(ts_iso: str) -> date | None:
    """Parse an ISO-8601 timestamp and return its UTC date.

    Returns None on parse failure (caller will skip the record). The bot's
    trading-day convention (Brisbane 09:00 rollover, per CLAUDE.md) is NOT
    applied here -- this gate counts distinct UTC dates because the
    cross-session rollover would discard early-session SESSION_START records
    near the rollover boundary. Counting UTC dates is the conservative,
    bias-free choice for a maturity gate -- it slightly under-counts on
    rollover-boundary sessions but never over-counts.
    """
    if not isinstance(ts_iso, str):
        return None
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:  # pyright: ignore[reportUnnecessaryComparison]
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).date()


def _scan_signal_file(path: Path, instrument: str, profile_id: str | None) -> tuple[set[date], int, int]:
    """Read one signal log file. Return (distinct_dates, records_scanned, qualifying)."""
    dates: set[date] = set()
    n_scanned = 0
    n_qualifying = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.debug("telemetry_maturity: cannot read %s: %s", path, exc)
        return dates, n_scanned, n_qualifying
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        n_scanned += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.debug("telemetry_maturity: skipping malformed line in %s", path)
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("instrument") != instrument:
            continue
        if profile_id is not None and rec.get("profile_id") != profile_id:
            continue
        if rec.get("type") not in _UPTIME_RECORD_TYPES:
            continue
        td = _trading_day_from_iso(rec.get("ts", ""))
        if td is None:
            continue
        dates.add(td)
        n_qualifying += 1
    return dates, n_scanned, n_qualifying


def evaluate_telemetry_maturity(
    signals_dir: Path,
    instrument: str,
    min_trading_days: int = MIN_TELEMETRY_TRADING_DAYS,
    profile_id: str | None = None,
) -> TelemetryMaturityReport:
    """Return whether enough distinct trading_days of uptime exist for ``instrument``.

    Args:
        signals_dir: directory containing ``live_signals_YYYY-MM-DD.jsonl``
            files. Per ``session_orchestrator.SIGNALS_DIR`` this is repo root.
        instrument: instrument symbol (e.g. "MNQ", "MES", "MGC"). Only records
            whose ``instrument`` field matches are counted.
        min_trading_days: floor for the gate. Default MIN_TELEMETRY_TRADING_DAYS.
            Override only for tests; production callers must use the constant.
        profile_id: optional profile id. When provided, only records with an
            exact matching ``profile_id`` qualify. Records written before this
            field existed intentionally do not count toward profile maturity.

    Returns:
        TelemetryMaturityReport with verdict = VERDICT_MATURE iff the
        distinct-trading-day count is >= min_trading_days. VERDICT_UNVERIFIED
        otherwise -- including when signals_dir is empty or missing.

    Fail-closed contract: if signals_dir does not exist, or contains no files,
    or all files fail to parse, the verdict is VERDICT_UNVERIFIED. The gate
    never raises on missing or malformed telemetry -- that would let a caller
    silently bypass the gate by deleting log files.
    """
    if not isinstance(min_trading_days, int) or min_trading_days < 1:
        raise ValueError(f"min_trading_days must be a positive int, got {min_trading_days!r}")

    files: list[Path] = []
    if signals_dir.exists() and signals_dir.is_dir():
        files = sorted(signals_dir.glob("live_signals_*.jsonl"))

    all_dates: set[date] = set()
    total_scanned = 0
    total_qualifying = 0
    for f in files:
        dates, n_scanned, n_qualifying = _scan_signal_file(f, instrument, profile_id)
        all_dates.update(dates)
        total_scanned += n_scanned
        total_qualifying += n_qualifying

    n_unique = len(all_dates)
    verdict = VERDICT_MATURE if n_unique >= min_trading_days else VERDICT_UNVERIFIED
    return TelemetryMaturityReport(
        verdict=verdict,
        instrument=instrument,
        profile_id=profile_id,
        profile_scoped=profile_id is not None,
        n_unique_trading_days=n_unique,
        min_required=min_trading_days,
        trading_days=tuple(sorted(all_dates)),
        signal_files_scanned=len(files),
        records_scanned=total_scanned,
        records_qualifying=total_qualifying,
    )


def iter_known_uptime_record_types() -> Iterable[str]:
    """Read-only accessor for the canonical uptime-record-type set.

    Tests use this to assert that a representative set of record types is
    included; downstream callers should not need to enumerate.
    """
    return iter(_UPTIME_RECORD_TYPES)
