"""Tests for pipeline.asset_configs — canonical instrument source."""

import pytest

from pipeline.asset_configs import (
    ACTIVE_ORB_INSTRUMENTS,
    ASSET_CONFIGS,
    DEPLOYABLE_ORB_INSTRUMENTS,
    get_active_instruments,
    get_deployable_instruments,
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


class TestDeployableInstruments:
    """Test the DEPLOYABLE_ORB_INSTRUMENTS canonical subset.

    Active-but-research-only instruments (e.g. MGC, whose real-micro data
    horizon is insufficient for T7 era-discipline survival) belong in
    ACTIVE_ORB_INSTRUMENTS (pipeline runs on them) but NOT in
    DEPLOYABLE_ORB_INSTRUMENTS (pulse/alerting should not flag the
    by-design empty deployable state).
    """

    def test_subset_of_active(self):
        """Every deployable instrument must also be active — strict subset."""
        assert set(DEPLOYABLE_ORB_INSTRUMENTS) <= set(ACTIVE_ORB_INSTRUMENTS)

    def test_sorted(self):
        assert list(DEPLOYABLE_ORB_INSTRUMENTS) == sorted(DEPLOYABLE_ORB_INSTRUMENTS)

    def test_helper_returns_copy(self):
        """Mutating the returned list must NOT affect the module constant."""
        got = get_deployable_instruments()
        got.append("FAKE")
        assert "FAKE" not in DEPLOYABLE_ORB_INSTRUMENTS

    def test_helper_matches_constant(self):
        assert get_deployable_instruments() == list(DEPLOYABLE_ORB_INSTRUMENTS)

    def test_mgc_active_not_deployable(self):
        """MGC is research-only: active for pipeline, not deployable for alerts."""
        assert "MGC" in ACTIVE_ORB_INSTRUMENTS
        assert "MGC" not in DEPLOYABLE_ORB_INSTRUMENTS
        assert ASSET_CONFIGS["MGC"].get("deployable_expected") is False

    def test_mes_mnq_both(self):
        """MES and MNQ are active AND deployable."""
        for inst in ("MES", "MNQ"):
            assert inst in ACTIVE_ORB_INSTRUMENTS
            assert inst in DEPLOYABLE_ORB_INSTRUMENTS
            # Default True when key is absent; may also be explicitly True.
            assert ASSET_CONFIGS[inst].get("deployable_expected", True) is True

    def test_dead_instruments_in_neither(self):
        for dead in ("MCL", "SIL", "M6E", "MBT", "M2K"):
            assert dead not in ACTIVE_ORB_INSTRUMENTS
            assert dead not in DEPLOYABLE_ORB_INSTRUMENTS


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

    def test_non_canonical_pattern_raises(self, monkeypatch):
        """Helper raises ValueError if outright_pattern is structurally non-canonical.

        Closes a coverage gap caught by Bloomey self-review of commit 838ab85
        (Move C). The unknown-instrument failure path was tested, but the
        non-canonical-pattern path was not. Inject a fake config with a
        regex that does not match the canonical `^<ROOT>[<MONTH_CODES>]\\d+$`
        shape and verify the helper fails-closed instead of returning garbage.
        """
        import re

        fake_cfg = {"outright_pattern": re.compile(r"^WEIRDPATTERN$")}
        monkeypatch.setitem(ASSET_CONFIGS, "_FAKE_BAD_PATTERN", fake_cfg)
        with pytest.raises(ValueError, match="Non-canonical outright_pattern"):
            get_outright_root("_FAKE_BAD_PATTERN")


class TestParentSymbol:
    """Phase 3a: every ASSET_CONFIGS entry must declare parent_symbol.

    `parent_symbol: str | None` is the canonical parent-contract mapping
    consumed by `pipeline.data_era`. Added 2026-04-08 (Phase 3a foundation).
    """

    EXPECTED = {
        # Active micros → parent preserved post-Phase-2
        "MGC": "GC",
        "MNQ": "NQ",
        "MES": "ES",
        # Dead micros → use parent data as proxy (cost model adjusts)
        "M2K": "RTY",
        "MBT": "BTC",
        "M6E": "6E",
        "MCL": "CL",
        "SIL": "SI",
        # Native parents and research-only → None
        "NQ": None,
        "ES": None,
        "GC": None,
        "2YY": None,
        "ZT": None,
    }

    def test_all_configs_declare_parent_symbol(self):
        """Every ASSET_CONFIGS entry must have the parent_symbol key (even if None)."""
        for inst, cfg in ASSET_CONFIGS.items():
            assert "parent_symbol" in cfg, (
                f"{inst} config missing 'parent_symbol' field — "
                f"Phase 3a requires all configs to declare it"
            )

    def test_expected_coverage_matches_configs(self):
        """EXPECTED must cover every entry in ASSET_CONFIGS — drift guard."""
        configured = set(ASSET_CONFIGS.keys())
        expected = set(self.EXPECTED.keys())
        missing = configured - expected
        extra = expected - configured
        assert not missing, f"ASSET_CONFIGS has instruments not in test EXPECTED: {missing}"
        assert not extra, f"Test EXPECTED has instruments not in ASSET_CONFIGS: {extra}"

    def test_active_micros_map_to_canonical_parents(self):
        assert ASSET_CONFIGS["MGC"]["parent_symbol"] == "GC"
        assert ASSET_CONFIGS["MNQ"]["parent_symbol"] == "NQ"
        assert ASSET_CONFIGS["MES"]["parent_symbol"] == "ES"

    def test_dead_micros_declare_parent(self):
        """Dead micros use parent data — parent_symbol must reflect the actual source."""
        assert ASSET_CONFIGS["M2K"]["parent_symbol"] == "RTY"
        assert ASSET_CONFIGS["MBT"]["parent_symbol"] == "BTC"
        assert ASSET_CONFIGS["M6E"]["parent_symbol"] == "6E"
        assert ASSET_CONFIGS["MCL"]["parent_symbol"] == "CL"
        assert ASSET_CONFIGS["SIL"]["parent_symbol"] == "SI"

    def test_parents_and_research_have_none(self):
        """NQ/ES/GC are parents themselves; 2YY/ZT are research-only natives."""
        for inst in ("NQ", "ES", "GC", "2YY", "ZT"):
            assert ASSET_CONFIGS[inst]["parent_symbol"] is None, (
                f"{inst} should have parent_symbol=None (parent/native)"
            )

    def test_all_expected_values_match(self):
        """Full coverage matrix — catches any drift between config and expectations."""
        for inst, expected_parent in self.EXPECTED.items():
            actual = ASSET_CONFIGS[inst]["parent_symbol"]
            assert actual == expected_parent, (
                f"{inst}.parent_symbol should be {expected_parent!r}, got {actual!r}"
            )
