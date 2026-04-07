"""Tests for pipeline.dst.orb_utc_window and compute_trading_day_utc_range.

These functions are the canonical source of truth for ORB window timing
across the backtest (outcome_builder), live engine (execution_engine), and
feature builder (build_daily_features). See pipeline/dst.py module docstring
and the E2 canonical-window refactor (2026-04-07) for architectural context.

Test corpus composition:
  - 180 deterministic cases: 12 dynamic sessions x 3 apertures (5/15/30) x
    5 sample dates (winter, spring, summer, fall, DST-transition)
  - 4 DST transition tests (US spring forward / fall back; UK BST on/off)
  - 5 boundary / fail-closed tests (unknown session, orb_minutes <= 0,
    orb_minutes > 60, orb_minutes = 1 and 60 as valid edges)
  - 1 idempotency test
  - 2,268-case snapshot equivalence test (all (day, session, aperture) triples
    captured at stage baseline, asserting new canonical function matches
    predecessor pipeline/build_daily_features._orb_utc_window byte-for-byte)
  - 1 trading-day-range test for compute_trading_day_utc_range

Total: ~195 explicit cases + 2,268 snapshot cases = ~2,463 assertions.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from pipeline.build_daily_features import _orb_utc_window as predecessor_orb_utc_window
from pipeline.dst import (
    BRISBANE_TZ,
    DYNAMIC_ORB_RESOLVERS,
    TRADING_DAY_START_HOUR_LOCAL,
    UTC_TZ,
    compute_trading_day_utc_range,
    orb_utc_window,
)

_ORB_APERTURES = (5, 15, 30)
_SAMPLE_DATES = (
    date(2024, 1, 15),  # winter (no DST anywhere)
    date(2024, 4, 15),  # US DST active, UK BST active (spring)
    date(2024, 7, 15),  # both DST active (high summer)
    date(2024, 10, 15),  # US DST active, UK BST still active (fall before UK roll)
    date(2024, 11, 15),  # US DST off, UK BST off (fall after both rolls)
)

_BASELINE_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "runtime"
    / "baselines"
    / "2026-04-07-orb-window-snapshot.json"
)


# =========================================================================
# Section 1 — Deterministic corpus (12 sessions x 3 apertures x 5 dates)
# =========================================================================


class TestOrbUtcWindowDeterministic:
    """Cartesian corpus: every session x every aperture x every sample date.

    Each case asserts: (a) function returns without raising, (b) start < end,
    (c) end - start == orb_minutes, (d) both endpoints are in UTC,
    (e) start is within the trading day's UTC range.
    """

    @pytest.mark.parametrize("orb_label", sorted(DYNAMIC_ORB_RESOLVERS))
    @pytest.mark.parametrize("orb_minutes", _ORB_APERTURES)
    @pytest.mark.parametrize("trading_day", _SAMPLE_DATES)
    def test_cartesian_corpus(
        self, orb_label: str, orb_minutes: int, trading_day: date
    ) -> None:
        start, end = orb_utc_window(trading_day, orb_label, orb_minutes)

        # Structural invariants
        assert start < end, f"start must be strictly before end: {start} / {end}"
        assert end - start == timedelta(minutes=orb_minutes), (
            f"duration must equal orb_minutes: got {end - start}, "
            f"expected {timedelta(minutes=orb_minutes)}"
        )
        assert start.tzinfo is not None, "start must be timezone-aware"
        assert end.tzinfo is not None, "end must be timezone-aware"
        assert start.utcoffset() == timedelta(0), "start must be in UTC"
        assert end.utcoffset() == timedelta(0), "end must be in UTC"

        # Window must fall within trading day's UTC range
        td_start, td_end = compute_trading_day_utc_range(trading_day)
        assert td_start <= start < td_end, (
            f"ORB start {start} must be within trading day "
            f"[{td_start}, {td_end}) for {trading_day} {orb_label}"
        )

    def test_corpus_covers_all_sessions(self) -> None:
        """Sanity: ensure the cartesian corpus iterates every dynamic session."""
        assert len(DYNAMIC_ORB_RESOLVERS) >= 10, (
            f"Corpus should cover >=10 sessions; found {len(DYNAMIC_ORB_RESOLVERS)}"
        )

    def test_corpus_size(self) -> None:
        """Sanity: the parametrize corpus size matches expected (>=180)."""
        expected = len(DYNAMIC_ORB_RESOLVERS) * len(_ORB_APERTURES) * len(_SAMPLE_DATES)
        assert expected >= 180, (
            f"Cartesian corpus should be >=180 cases; expected {expected}"
        )


# =========================================================================
# Section 2 — DST transition correctness
# =========================================================================


class TestOrbUtcWindowDstTransitions:
    """Verify DST-sensitive sessions resolve correctly across transition days.

    Sessions with fixed US local times (NYSE_OPEN at 09:30 ET, NYSE_CLOSE at
    16:00 ET, etc.) shift by one hour across the US spring-forward / fall-back
    transitions. The canonical function must honour this via zoneinfo math.
    """

    def test_nyse_open_us_dst_spring_forward(self) -> None:
        """NYSE open 09:30 ET: EDT (DST) = 13:30 UTC; EST (std) = 14:30 UTC.

        US DST 2024 spring forward = Mar 10. Before the switch, US is in EST
        (UTC-5), 09:30 ET = 14:30 UTC. After, EDT (UTC-4), 09:30 ET = 13:30 UTC.
        """
        before_switch = date(2024, 3, 9)  # EST, NYSE open 14:30 UTC
        after_switch = date(2024, 3, 11)  # EDT, NYSE open 13:30 UTC

        start_before, _ = orb_utc_window(before_switch, "NYSE_OPEN", 5)
        start_after, _ = orb_utc_window(after_switch, "NYSE_OPEN", 5)

        assert start_before.hour == 14 and start_before.minute == 30, (
            f"EST NYSE open must be 14:30 UTC; got {start_before}"
        )
        assert start_after.hour == 13 and start_after.minute == 30, (
            f"EDT NYSE open must be 13:30 UTC; got {start_after}"
        )

    def test_nyse_close_us_dst_fall_back(self) -> None:
        """NYSE close 16:00 ET: EDT = 20:00 UTC; EST = 21:00 UTC.

        US DST 2024 fall back = Nov 3. Before: EDT, 16:00 ET = 20:00 UTC.
        After: EST, 16:00 ET = 21:00 UTC.
        """
        before_switch = date(2024, 11, 1)  # EDT
        after_switch = date(2024, 11, 4)  # EST

        start_before, _ = orb_utc_window(before_switch, "NYSE_CLOSE", 5)
        start_after, _ = orb_utc_window(after_switch, "NYSE_CLOSE", 5)

        assert start_before.hour == 20 and start_before.minute == 0, (
            f"EDT NYSE close must be 20:00 UTC; got {start_before}"
        )
        assert start_after.hour == 21 and start_after.minute == 0, (
            f"EST NYSE close must be 21:00 UTC; got {start_after}"
        )

    def test_london_metals_uk_bst_active(self) -> None:
        """LONDON_METALS 08:00 London: BST (DST) = 07:00 UTC; GMT = 08:00 UTC.

        UK BST 2024 starts Mar 31. Test a mid-summer date to ensure BST applies.
        """
        bst_date = date(2024, 7, 15)  # BST active
        gmt_date = date(2024, 1, 15)  # GMT (winter)

        start_bst, _ = orb_utc_window(bst_date, "LONDON_METALS", 5)
        start_gmt, _ = orb_utc_window(gmt_date, "LONDON_METALS", 5)

        assert start_bst.hour == 7 and start_bst.minute == 0, (
            f"BST London 08:00 must be 07:00 UTC; got {start_bst}"
        )
        assert start_gmt.hour == 8 and start_gmt.minute == 0, (
            f"GMT London 08:00 must be 08:00 UTC; got {start_gmt}"
        )

    def test_cme_reopen_us_dst_hour_shift(self) -> None:
        """CME_REOPEN is anchored to CME Globex reopen at 17:00 CT.

        Per pipeline/dst.py module docstring: "CME_REOPEN - CME Globex electronic
        reopen at 5:00 PM CT". The resolver `cme_open_brisbane` is US-DST-sensitive:
        the hour-of-day in UTC shifts by exactly 1 hour across the winter/summer
        boundary.
          - Winter (CST, UTC-6): 17:00 CT = 23:00 UTC
          - Summer (CDT, UTC-5): 17:00 CT = 22:00 UTC

        This test pins the hour-of-day (23 ↔ 22) — which is the load-bearing
        DST behaviour. Calendar-date semantics are verified authoritatively
        by test_snapshot_equivalence_full (2,268 real-world cases) and
        test_predecessor_equivalence_sample (180 synthetic cases). Adding
        fragile date-arithmetic assertions here would duplicate coverage
        with higher maintenance cost.
        """
        winter = date(2024, 1, 15)  # CST (US DST off)
        summer = date(2024, 7, 15)  # CDT (US DST on)

        start_winter, _ = orb_utc_window(winter, "CME_REOPEN", 5)
        start_summer, _ = orb_utc_window(summer, "CME_REOPEN", 5)

        # Winter CT = UTC-6 → 17:00 CT = 23:00 UTC hour
        assert start_winter.hour == 23, (
            f"winter CME_REOPEN must be 23:00 UTC hour; got {start_winter}"
        )
        # Summer CT = UTC-5 → 17:00 CT = 22:00 UTC hour
        assert start_summer.hour == 22, (
            f"summer CME_REOPEN must be 22:00 UTC hour; got {start_summer}"
        )
        # Both start exactly on the hour (:00)
        assert start_winter.minute == 0
        assert start_summer.minute == 0


# =========================================================================
# Section 3 — Boundary / fail-closed validation
# =========================================================================


class TestOrbUtcWindowBoundaries:
    """Fail-closed behaviour on invalid inputs, plus edge-of-valid cases.

    Canonical contract per pipeline/dst.orb_utc_window docstring:
      - ValueError on unknown orb_label
      - ValueError on orb_minutes outside [1, 60]
      - Accept orb_minutes = 1 and 60 as edge-of-valid
    """

    def test_unknown_orb_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ORB label"):
            orb_utc_window(date(2024, 1, 15), "BOGUS_SESSION", 5)

    def test_orb_minutes_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="orb_minutes=0"):
            orb_utc_window(date(2024, 1, 15), "CME_REOPEN", 0)

    def test_orb_minutes_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="orb_minutes=-5"):
            orb_utc_window(date(2024, 1, 15), "CME_REOPEN", -5)

    def test_orb_minutes_over_max_raises(self) -> None:
        with pytest.raises(ValueError, match="orb_minutes=61"):
            orb_utc_window(date(2024, 1, 15), "CME_REOPEN", 61)

    def test_orb_minutes_one_accepted(self) -> None:
        start, end = orb_utc_window(date(2024, 1, 15), "CME_REOPEN", 1)
        assert end - start == timedelta(minutes=1)

    def test_orb_minutes_sixty_accepted(self) -> None:
        start, end = orb_utc_window(date(2024, 1, 15), "CME_REOPEN", 60)
        assert end - start == timedelta(minutes=60)


# =========================================================================
# Section 4 — Idempotency
# =========================================================================


class TestOrbUtcWindowIdempotent:
    """Calling orb_utc_window with identical inputs must return identical outputs.

    Property: no hidden state, no monkey-patched globals, pure function.
    """

    def test_same_inputs_same_outputs(self) -> None:
        td = date(2024, 7, 15)
        for label in sorted(DYNAMIC_ORB_RESOLVERS):
            for aperture in _ORB_APERTURES:
                r1 = orb_utc_window(td, label, aperture)
                r2 = orb_utc_window(td, label, aperture)
                assert r1 == r2, (
                    f"orb_utc_window not idempotent for {td} {label} {aperture}: "
                    f"{r1} vs {r2}"
                )


# =========================================================================
# Section 5 — compute_trading_day_utc_range
# =========================================================================


class TestComputeTradingDayUtcRange:
    """The 24-hour UTC window for a trading day, anchored at 09:00 Brisbane."""

    def test_winter_day(self) -> None:
        start, end = compute_trading_day_utc_range(date(2024, 1, 15))
        # 09:00 Brisbane on 2024-01-15 = 23:00 UTC on 2024-01-14
        assert start == datetime(2024, 1, 14, 23, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 15, 23, 0, 0, tzinfo=UTC_TZ)

    def test_summer_day(self) -> None:
        # Brisbane has no DST — formula is stable year-round
        start, end = compute_trading_day_utc_range(date(2024, 7, 15))
        assert start == datetime(2024, 7, 14, 23, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 7, 15, 23, 0, 0, tzinfo=UTC_TZ)

    def test_duration_is_24_hours(self) -> None:
        for td in _SAMPLE_DATES:
            start, end = compute_trading_day_utc_range(td)
            assert end - start == timedelta(hours=24), (
                f"Trading day {td} duration must be 24h; got {end - start}"
            )

    def test_start_is_09_brisbane_local(self) -> None:
        for td in _SAMPLE_DATES:
            start, _ = compute_trading_day_utc_range(td)
            local = start.astimezone(BRISBANE_TZ)
            assert local.hour == TRADING_DAY_START_HOUR_LOCAL, (
                f"Trading day {td} must start at 09:00 Brisbane; got {local.isoformat()}"
            )
            assert local.minute == 0
            assert local.date() == td


# =========================================================================
# Section 6 — Snapshot equivalence vs predecessor (2,268 cases)
# =========================================================================


class TestOrbUtcWindowSnapshotEquivalence:
    """Byte-identical equivalence against the baseline snapshot + predecessor.

    Two parallel checks:
      1. Against docs/runtime/baselines/2026-04-07-orb-window-snapshot.json
         (frozen expected output for 2,268 real-world triples)
      2. Against pipeline/build_daily_features._orb_utc_window (predecessor
         function — direct python equivalence)

    If either check fails, the canonical function has diverged from its
    contract and Stages 2+ migration will break. Do NOT weaken these tests
    to paper over a divergence — fix the divergence.
    """

    def test_snapshot_file_exists(self) -> None:
        assert _BASELINE_SNAPSHOT_PATH.exists(), (
            f"Baseline snapshot missing: {_BASELINE_SNAPSHOT_PATH}. "
            f"Stage 0 should have captured it."
        )

    def test_snapshot_equivalence_full(self) -> None:
        """All 2,268 baseline triples must match exactly."""
        snapshot = json.loads(_BASELINE_SNAPSHOT_PATH.read_text(encoding="utf-8"))

        mismatches: list[str] = []
        checked = 0
        for key, expected in snapshot.items():
            trading_day_str, orb_label, orb_minutes_str = key.split("|")
            trading_day = date.fromisoformat(trading_day_str)
            orb_minutes = int(orb_minutes_str)

            start, end = orb_utc_window(trading_day, orb_label, orb_minutes)
            expected_start = datetime.fromisoformat(expected["start_utc"])
            expected_end = datetime.fromisoformat(expected["end_utc"])

            checked += 1
            if start != expected_start or end != expected_end:
                mismatches.append(
                    f"{key}: got ({start.isoformat()}, {end.isoformat()}) "
                    f"expected ({expected_start.isoformat()}, {expected_end.isoformat()})"
                )

        assert checked > 0, "Snapshot was empty — should contain 2,268 cases"
        assert not mismatches, (
            f"{len(mismatches)} / {checked} snapshot mismatches:\n"
            + "\n".join(mismatches[:10])
            + ("\n..." if len(mismatches) > 10 else "")
        )

    def test_predecessor_equivalence_sample(self) -> None:
        """Cross-verify against pipeline/build_daily_features._orb_utc_window.

        Sample the cartesian corpus (180 cases) and assert byte-equality.
        This guards against future divergence if the canonical and
        predecessor functions drift during migration stages.
        """
        mismatches: list[str] = []
        for orb_label in sorted(DYNAMIC_ORB_RESOLVERS):
            for orb_minutes in _ORB_APERTURES:
                for trading_day in _SAMPLE_DATES:
                    new_result = orb_utc_window(trading_day, orb_label, orb_minutes)
                    old_result = predecessor_orb_utc_window(
                        trading_day, orb_label, orb_minutes
                    )
                    if new_result != old_result:
                        mismatches.append(
                            f"{trading_day} {orb_label} {orb_minutes}: "
                            f"new={new_result} old={old_result}"
                        )
        assert not mismatches, (
            f"{len(mismatches)} divergences between canonical and predecessor:\n"
            + "\n".join(mismatches[:10])
        )
