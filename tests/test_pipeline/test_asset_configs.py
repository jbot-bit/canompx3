"""Tests for pipeline.asset_configs — canonical instrument source."""

from pipeline.asset_configs import (
    ASSET_CONFIGS,
    ACTIVE_ORB_INSTRUMENTS,
    get_active_instruments,
    list_instruments,
)


class TestActiveInstruments:
    """Test the canonical active instruments list and accessor."""

    def test_exact_contents(self):
        """Active instruments are exactly the 3 traded ORB instruments."""
        assert get_active_instruments() == ["MES", "MGC", "MNQ"]

    def test_sorted(self):
        """List is alphabetically sorted."""
        instruments = get_active_instruments()
        assert instruments == sorted(instruments)

    def test_returns_copy(self):
        """Mutating the returned list must NOT affect the module constant."""
        instruments = get_active_instruments()
        instruments.append("FAKE")
        assert "FAKE" not in ACTIVE_ORB_INSTRUMENTS
        assert len(get_active_instruments()) == 3

    def test_no_dead_instruments(self):
        """Dead-for-ORB instruments must not appear."""
        instruments = get_active_instruments()
        for dead in ("MCL", "SIL", "M6E", "MBT", "M2K"):
            assert dead not in instruments

    def test_no_source_aliases(self):
        """Source contract aliases (ES, NQ) must not appear."""
        instruments = get_active_instruments()
        for alias in ("ES", "NQ"):
            assert alias not in instruments

    def test_constant_matches_function(self):
        """Module constant and function return the same data."""
        assert list(ACTIVE_ORB_INSTRUMENTS) == get_active_instruments()

    def test_research_only_2yy_not_active(self):
        """2YY is onboarded for research without entering the active ORB universe."""
        assert "2YY" in list_instruments()
        assert "2YY" not in get_active_instruments()
        assert ASSET_CONFIGS["2YY"]["orb_active"] is False

    def test_research_only_zt_not_active(self):
        """ZT is onboarded for research without entering the active ORB universe."""
        assert "ZT" in list_instruments()
        assert "ZT" not in get_active_instruments()
        assert ASSET_CONFIGS["ZT"]["orb_active"] is False
