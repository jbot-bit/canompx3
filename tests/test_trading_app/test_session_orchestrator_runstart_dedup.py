"""Stage 3 / F1b — RED regression: run()-start dedup re-seed after day correction.

THE GAP (verified by direct read, 2026-06-08, session_orchestrator.py @ 9ba23eb2):

  - L2 init dedup seeds `engine.mark_strategy_traded` from the journal using
    `self.trading_day` computed at __init__ (:315-319).
  - run() re-checks the day at :3827 and CORRECTS `self.trading_day` to a new
    day at :3833 ("init was stale"), but does NOT re-seed the dedup for the
    corrected day. The span :3833 -> :3987 (feed loop) contains no
    get_strategy_ids_for_day / mark_strategy_traded call.
  - L3 (:1818) only re-seeds on a *running* rollover, not on the run()-start
    correction.

REACHABLE: process starts at 08:59 Brisbane (init computes day D), init latency
pushes run() past 09:00 -> corrected to D+1; a journal row from D's session is
no longer deduped. L1 (query_open fail-close, :679) catches any OPEN position,
so this is a defense-in-depth gap, not a standalone capital hole — but it is a
silent un-seeded path the F1b stage exists to close.

This test binds to the FIX's seam: a single reusable seeder
`SessionOrchestrator._seed_journal_dedup(day)` that BOTH the init path and the
run()-start correction call. That method does not exist yet -> RED is structural
(AttributeError), proving production has no shared re-seed seam.

Run (RED expected against current code):
  cd C:/Users/joshd/canompx3-live-safety
  python -m pytest .live-safety-prep/test_runstart_dedup_reseed.py -v
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_app.live.session_orchestrator import SessionOrchestrator


class _FakeEngine:
    """Records mark_strategy_traded calls — the dedup sink we assert on."""

    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark_strategy_traded(self, strategy_id: str) -> None:
        self.marked.append(strategy_id)


class _FakeJournal:
    """Real-shaped stand-in for TradeJournal: maps trading_day -> traded sids.

    Mirrors get_strategy_ids_for_day's contract (DISTINCT strategy_id WHERE
    trading_day = ?) and the is_healthy gate the seed path checks.
    """

    def __init__(self, by_day: dict[date, set[str]], healthy: bool = True) -> None:
        self._by_day = by_day
        self.is_healthy = healthy

    def get_strategy_ids_for_day(self, trading_day) -> set[str]:
        return set(self._by_day.get(trading_day, set()))


def _make_orchestrator_shell(journal, engine, *, signal_only: bool = False):
    """Build a bare orchestrator without the heavy broker/auth/contract ctor.

    object.__new__ skips __init__; we attach only the attributes the dedup-seed
    seam reads. This isolates the seam under test from broker/feed dependencies.
    """
    orch = object.__new__(SessionOrchestrator)
    orch.journal = journal
    orch.engine = engine
    orch.signal_only = signal_only
    orch.instrument = "MNQ"
    return orch


def test_seed_journal_dedup_seam_exists():
    """FIX must expose ONE reusable seeder used by init AND run()-start.

    RED on current code: AttributeError — no shared seam exists; the two seed
    blocks (:1014 init, :1818 rollover) are inlined and run() has none.
    """
    assert hasattr(SessionOrchestrator, "_seed_journal_dedup"), (
        "F1b fix must extract a reusable SessionOrchestrator._seed_journal_dedup(day) "
        "and call it from the run()-start day-correction path (:3833)."
    )


def test_runstart_correction_reseeds_corrected_day():
    """The CORE proof: after the day is corrected D -> D+1, the corrected day's
    journal strategies must be marked-traded so they cannot re-enter.

    Day D has strategy 'S_old' traded; day D+1 (corrected) is empty. The fix
    must re-seed for the CORRECTED day. Here we assert the seam marks exactly
    the corrected day's set (empty), proving it re-queries with the right day
    rather than leaving the init-seeded D set stale.
    """
    day_d = date(2026, 6, 8)
    day_d1 = date(2026, 6, 9)
    journal = _FakeJournal({day_d: {"S_old"}, day_d1: {"S_new"}})
    engine = _FakeEngine()
    orch = _make_orchestrator_shell(journal, engine)

    # The fix's seam: re-seed for the corrected day.
    orch._seed_journal_dedup(day_d1)

    assert engine.marked == ["S_new"], (
        f"run()-start correction must seed dedup for the CORRECTED day (D+1='S_new'), got {engine.marked}"
    )


def test_seed_is_failclosed_when_journal_unhealthy_live():
    """Parity with :1002-1007: unhealthy journal in live mode must RAISE,
    never silently skip dedup (silent skip = re-entry risk)."""
    journal = _FakeJournal({}, healthy=False)
    orch = _make_orchestrator_shell(journal, _FakeEngine(), signal_only=False)
    with pytest.raises(RuntimeError):
        orch._seed_journal_dedup(date(2026, 6, 9))


def test_seed_is_noop_when_journal_unhealthy_signal_only():
    """Parity with :1008-1012: signal-only tolerates an unhealthy journal
    (no capital), seeds nothing, does not raise."""
    journal = _FakeJournal({}, healthy=False)
    engine = _FakeEngine()
    orch = _make_orchestrator_shell(journal, engine, signal_only=True)
    orch._seed_journal_dedup(date(2026, 6, 9))  # must not raise
    assert engine.marked == []


def test_seed_no_correction_is_idempotent_noop():
    """When run() day == init day (no correction), re-seeding the same day is a
    harmless no-op at the engine level (mark_strategy_traded is set-semantics in
    production). Here: same sids re-marked is acceptable; assert it re-queries."""
    day_d = date(2026, 6, 8)
    journal = _FakeJournal({day_d: {"S_old"}})
    engine = _FakeEngine()
    orch = _make_orchestrator_shell(journal, engine)
    orch._seed_journal_dedup(day_d)
    assert engine.marked == ["S_old"]


# ---------------------------------------------------------------------------
# WIRING PROOF (point 2): the helper passing is NOT enough — run() must ACTUALLY
# invoke it at the corrected-day point. A passing helper that run() never calls
# is the exact silent gap this stage exists to close. These two tests bind the
# wiring, not just the helper.
# ---------------------------------------------------------------------------


class _SeedReached(Exception):
    """Sentinel raised by the spy so run() aborts AT the seed call site,
    proving the call is reached with the corrected day independent of any
    downstream broker/feed/calendar dependency."""

    def __init__(self, day):
        self.day = day


def test_run_invokes_seed_for_corrected_day_WIRING():
    """Drive the REAL run() preamble: set trading_day to a stale value so the
    :3827 correction fires, spy on _seed_journal_dedup. The spy raises the moment
    run() calls it, capturing the day it was called with. PASS iff run() reached
    the call AND passed the corrected (today's) day — NOT the stale init day."""
    orch = object.__new__(SessionOrchestrator)
    orch.journal = _FakeJournal({})
    orch.engine = _FakeEngine()
    orch.signal_only = True
    orch.instrument = "MNQ"

    # Compute today's Brisbane trading day exactly as run() does (:3822-3826).
    bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
    today = (bris_now - timedelta(days=1)).date() if bris_now.hour < 9 else bris_now.date()
    # Force a stale init day so the correction branch fires.
    orch.trading_day = today - timedelta(days=3)

    captured = {}

    def _spy(day):
        captured["day"] = day
        raise _SeedReached(day)

    orch._seed_journal_dedup = _spy  # type: ignore[method-assign]

    with pytest.raises(_SeedReached):
        asyncio.run(orch.run())

    assert captured.get("day") == today, (
        "run() must call _seed_journal_dedup with the CORRECTED day "
        f"(today={today}), got {captured.get('day')}. If this fails, the "
        "re-seed is wired to the wrong day or not wired at all."
    )


def test_run_source_calls_seed_after_day_correction_TRACE():
    """Static trace backstop: confirm the _seed_journal_dedup call physically
    follows the `self.trading_day = actual_day` correction inside run(). Guards
    against a future refactor silently severing the wiring even if the dynamic
    test above is skipped on a platform/timezone edge."""
    src = inspect.getsource(SessionOrchestrator.run)
    assert "self.trading_day = actual_day" in src, "correction line missing from run()"
    correction_idx = src.index("self.trading_day = actual_day")
    seed_idx = src.find("_seed_journal_dedup", correction_idx)
    assert seed_idx != -1, (
        "run() does NOT call _seed_journal_dedup after the day correction — the wiring is missing (silent gap)."
    )
