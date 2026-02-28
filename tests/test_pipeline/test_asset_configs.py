"""Tests for pipeline.asset_configs — canonical instrument source."""

from pipeline.asset_configs import (
    ACTIVE_ORB_INSTRUMENTS,
    get_active_instruments,
)


class TestActiveInstruments:
    """Test the canonical active instruments list and accessor."""

    def test_exact_contents(self):
        """Active instruments are exactly the 4 traded ORB instruments."""
        assert get_active_instruments() == ["M2K", "MES", "MGC", "MNQ"]

    def test_sorted(self):
        """List is alphabetically sorted."""
        instruments = get_active_instruments()
        assert instruments == sorted(instruments)

    def test_returns_copy(self):
        """Mutating the returned list must NOT affect the module constant."""
        instruments = get_active_instruments()
        instruments.append("FAKE")
        assert "FAKE" not in ACTIVE_ORB_INSTRUMENTS
        assert len(get_active_instruments()) == 4

    def test_no_dead_instruments(self):
        """Dead-for-ORB instruments must not appear."""
        instruments = get_active_instruments()
        for dead in ("MCL", "SIL", "M6E"):
            assert dead not in instruments

    def test_no_source_aliases(self):
        """Source contract aliases (ES, NQ) must not appear."""
        instruments = get_active_instruments()
        for alias in ("ES", "NQ"):
            assert alias not in instruments

    def test_constant_matches_function(self):
        """Module constant and function return the same data."""
        assert list(ACTIVE_ORB_INSTRUMENTS) == get_active_instruments()
