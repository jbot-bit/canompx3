"""Finding 4 (audit 2026-06-06): bars present but daily_features absent (0-of-3 gap).

Tests `check_active_instrument_bars_without_daily_features` — the fail-closed
drift check that catches active-instrument trading days with >= 200 bars_1m rows
but ZERO daily_features rows, excluding the in-progress day via the canonical
Option W wall-clock guard.

Known-violation injection per .claude/rules/integrity-guardian.md § 7: a check
that cannot demonstrably catch a planted violation is dead-on-arrival.
"""

from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from pipeline import check_drift
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import compute_trading_day_utc_range

ACTIVE_SYM = ACTIVE_ORB_INSTRUMENTS[0]  # canonical, not a hardcoded literal


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE bars_1m (symbol VARCHAR, ts_utc TIMESTAMPTZ)")
    con.execute("CREATE TABLE daily_features (symbol VARCHAR, trading_day DATE, orb_minutes INTEGER)")
    return con


def _insert_bars(con, symbol: str, trading_day: date, n: int) -> None:
    """Insert n bars inside trading_day's canonical UTC window."""
    start, _ = compute_trading_day_utc_range(trading_day)
    rows = [(symbol, start + timedelta(minutes=i)) for i in range(n)]
    con.executemany("INSERT INTO bars_1m VALUES (?, ?)", rows)


def _insert_features(con, symbol: str, trading_day: date) -> None:
    """Insert the full 3-aperture daily_features set for a day (built state)."""
    con.executemany(
        "INSERT INTO daily_features VALUES (?, ?, ?)",
        [(symbol, trading_day, m) for m in (5, 15, 30)],
    )


# A past day whose trading window has fully elapsed relative to FAR_FUTURE.
PAST_DAY = date(2025, 1, 6)
FAR_FUTURE = datetime(2025, 6, 1, tzinfo=UTC)


class TestBarsWithoutDailyFeatures:
    def test_catches_bars_present_features_absent(self):
        """200+ bars, 0 daily_features, elapsed day → blocking violation."""
        con = _con()
        _insert_bars(con, ACTIVE_SYM, PAST_DAY, 1320)
        # deliberately NO daily_features rows for PAST_DAY
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=FAR_FUTURE)
        assert len(v) == 1, v
        assert ACTIVE_SYM in v[0]
        assert str(PAST_DAY) in v[0]
        assert "0-of-3" in v[0]

    def test_passes_when_features_present(self):
        """Bars AND full daily_features → no violation (the built, healthy case)."""
        con = _con()
        _insert_bars(con, ACTIVE_SYM, PAST_DAY, 1320)
        _insert_features(con, ACTIVE_SYM, PAST_DAY)
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=FAR_FUTURE)
        assert v == [], v

    def test_excludes_in_progress_day(self):
        """A gap day whose window has NOT elapsed (live partial) is excluded."""
        con = _con()
        _insert_bars(con, ACTIVE_SYM, PAST_DAY, 1320)
        # now is BEFORE the day's window has closed → in-progress → not a violation
        _, window_end = compute_trading_day_utc_range(PAST_DAY)
        in_progress_now = window_end - timedelta(hours=1)
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=in_progress_now)
        assert v == [], v

    def test_ignores_thin_bar_days_below_floor(self):
        """< 200 bars (e.g. a thin live partial) is below the floor → not flagged."""
        con = _con()
        _insert_bars(con, ACTIVE_SYM, PAST_DAY, 60)  # thin partial, no features
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=FAR_FUTURE)
        assert v == [], v

    def test_clean_db_passes(self):
        """Empty DB → no violations (no false positive on a fresh DB)."""
        con = _con()
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=FAR_FUTURE)
        assert v == [], v

    def test_known_blind_spot_sub_200_bars_not_flagged(self):
        """DOCUMENTED limitation (audit gate 2026-06-06): 1..199 bars + 0 features
        is intentionally NOT flagged — bar count cannot distinguish a thin live
        partial from a legit short session (built days observed as low as ~59 bars).
        This test pins the blind spot so a future floor change is a deliberate,
        test-breaking decision, not a silent regression."""
        con = _con()
        _insert_bars(con, ACTIVE_SYM, PAST_DAY, 199)  # one below the floor, 0 features
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=FAR_FUTURE)
        assert v == [], "sub-200 days are a documented blind spot — should not flag"

    def test_floor_boundary_exactly_200_flags(self):
        """Boundary: exactly 200 bars + 0 features IS flagged (>= floor)."""
        con = _con()
        _insert_bars(con, ACTIVE_SYM, PAST_DAY, 200)
        v = check_drift.check_active_instrument_bars_without_daily_features(con=con, now=FAR_FUTURE)
        assert len(v) == 1, v


def test_check_is_registered_blocking_requires_db():
    """The check must be wired into CHECKS as blocking + requires_db, else dead."""
    entry = next(
        (e for e in check_drift.CHECKS if e[1] is check_drift.check_active_instrument_bars_without_daily_features),
        None,
    )
    assert entry is not None, "check not registered in CHECKS"
    is_advisory, requires_db = entry[2], entry[3]
    assert is_advisory is False, "Finding 4 must be blocking (fail-closed), not advisory"
    assert requires_db is True, "check reads bars_1m + daily_features → requires_db"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
