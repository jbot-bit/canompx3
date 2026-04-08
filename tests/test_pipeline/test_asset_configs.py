"""Tests for pipeline.asset_configs — canonical instrument source."""

import pytest

from pipeline.asset_configs import (
    ACTIVE_ORB_INSTRUMENTS,
    ASSET_CONFIGS,
    get_active_instruments,
    get_outright_root,
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


class TestGetOutrightRoot:
    """Test the canonical outright-root derivation helper.

    `get_outright_root(instrument)` extracts the contract root prefix from
    `outright_pattern` regex (single source of truth). Used by every script
    that needs to map instrument → vendor parent symbol (e.g., Databento
    `MGC.FUT`). Replaces parallel hardcoded dicts in
    `scripts/tools/refresh_data.py` and `scripts/databento_daily.py` that
    drifted from canonical post-Phase-2 (Apr 8 2026 — `bars_1m` real-micro
    redownload).
    """

    # Canonical truth table — every config in ASSET_CONFIGS should resolve.
    # Updated post-Phase-2: MGC now derives from native MGC pattern, not GC.
    EXPECTED_ROOTS = {
        # Active micros (real micro contracts post-Phase-2)
        "MGC": "MGC",
        "MNQ": "MNQ",
        "MES": "MES",
        # Parent contracts (preserved post-Phase-2 for backfill history)
        "GC": "GC",
        "NQ": "NQ",
        "ES": "ES",
        # Dead micros (use full-size parent data with cost adjustment)
        "M2K": "RTY",
        "MBT": "BTC",
        "MCL": "CL",
        "M6E": "6E",
        "SIL": "SI",
        # Research-only (native parent symbols)
        "2YY": "2YY",
        "ZT": "ZT",
    }

    def test_resolves_all_configured_instruments(self):
        """Every instrument in ASSET_CONFIGS must resolve to its canonical root."""
        for instrument, expected_root in self.EXPECTED_ROOTS.items():
            assert get_outright_root(instrument) == expected_root, (
                f"{instrument} should derive root {expected_root!r}, "
                f"got {get_outright_root(instrument)!r}"
            )

    def test_coverage_matches_asset_configs(self):
        """EXPECTED_ROOTS must cover every entry in ASSET_CONFIGS — no drift."""
        configured = set(ASSET_CONFIGS.keys())
        expected = set(self.EXPECTED_ROOTS.keys())
        missing_from_test = configured - expected
        extra_in_test = expected - configured
        assert not missing_from_test, (
            f"ASSET_CONFIGS has instruments not in EXPECTED_ROOTS: {missing_from_test}. "
            f"Add them to TestGetOutrightRoot.EXPECTED_ROOTS."
        )
        assert not extra_in_test, (
            f"EXPECTED_ROOTS has instruments not in ASSET_CONFIGS: {extra_in_test}."
        )

    def test_mgc_post_phase_2_returns_mgc_not_gc(self):
        """Regression guard: post-Phase-2, MGC must derive 'MGC', NOT 'GC'.

        Pre-Phase-2 (commit 82e8b60), MGC's outright_pattern was
        `^GC[FGHJKMNQUVXZ]\\d{1,2}$` because the data source was full-size
        Gold (parent). Phase 2 redownloaded real micro Gold from MGC.FUT and
        flipped the pattern to `^MGC[FGHJKMNQUVXZ]\\d{1,2}$`. Any consumer
        still expecting 'GC' would silently re-corrupt the canonical data.
        This test pins the post-Phase-2 reality.
        """
        assert get_outright_root("MGC") == "MGC"

    def test_case_insensitive(self):
        """Lowercase input must work — matches get_asset_config() behavior."""
        assert get_outright_root("mgc") == "MGC"
        assert get_outright_root("mnq") == "MNQ"
        assert get_outright_root("m2k") == "RTY"

    def test_unknown_instrument_raises(self):
        """Unknown instrument fails-closed with ValueError."""
        with pytest.raises(ValueError, match="Unknown instrument"):
            get_outright_root("FAKE")

    def test_empty_instrument_raises(self):
        """Empty string fails-closed."""
        with pytest.raises(ValueError, match="Unknown instrument"):
            get_outright_root("")

    def test_returns_str(self):
        """Helper must return str, not re.Match or None."""
        result = get_outright_root("MNQ")
        assert isinstance(result, str)
        assert len(result) > 0
