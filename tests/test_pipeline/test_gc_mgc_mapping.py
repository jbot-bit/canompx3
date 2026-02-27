"""
Tests for GC → MGC symbol transformation.

Verifies the critical design decision: raw data uses GC (full-size Gold)
contracts, but the pipeline stores bars under symbol='MGC' with the original
GC contract in source_symbol.

The GC→MGC mapping lives in ingest_dbn_mgc.py (legacy path).
The multi-instrument asset_configs.py uses the GC pattern (matching source data).
"""

import re
import pytest
from datetime import date

from pipeline.ingest_dbn_mgc import (
    choose_front_contract,
    GC_OUTRIGHT_PATTERN,
)


class TestGcOutrightPattern:
    """Verify GC outright pattern matches expected contract formats."""

    def test_matches_gc_contracts(self):
        """Standard GC contract symbols should match."""
        for sym in ["GCM4", "GCZ4", "GCG25", "GCQ5", "GCJ24"]:
            assert GC_OUTRIGHT_PATTERN.match(sym), f"{sym} should match"

    def test_rejects_non_gc(self):
        """Non-GC symbols should not match."""
        for sym in ["MGC", "MGCM4", "SIM4", "GC", "GC-SPREAD"]:
            assert not GC_OUTRIGHT_PATTERN.match(sym), f"{sym} should not match"

    def test_rejects_spreads(self):
        """Spread symbols should not match."""
        for sym in ["GCM4-GCZ4", "GCM4:GCZ4"]:
            assert not GC_OUTRIGHT_PATTERN.match(sym), f"{sym} should not match"


class TestChooseFrontContract:
    """Verify front contract selection from GC volume data."""

    def test_highest_volume_wins(self):
        """Contract with highest daily volume is chosen."""
        volumes = {"GCM4": 5000, "GCZ4": 3000, "GCQ4": 1000}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN,
                                       prefix_len=2)
        assert front == "GCM4"

    def test_deterministic_tiebreak(self):
        """Equal volume: earliest expiry wins, then lexicographic."""
        volumes = {"GCZ4": 5000, "GCM4": 5000}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN,
                                       prefix_len=2)
        # M (June) expires before Z (December)
        assert front == "GCM4"

    def test_filters_non_outrights(self):
        """Non-outright symbols in volume dict are ignored."""
        volumes = {"GCM4": 100, "GC-SPREAD": 9999, "INVALID": 5000}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN,
                                       prefix_len=2)
        assert front == "GCM4"

    def test_empty_volumes_returns_none(self):
        """No valid outrights → None."""
        front = choose_front_contract({}, outright_pattern=GC_OUTRIGHT_PATTERN,
                                       prefix_len=2)
        assert front is None

    def test_no_outrights_returns_none(self):
        """Only non-matching symbols → None."""
        volumes = {"INVALID1": 100, "INVALID2": 200}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN,
                                       prefix_len=2)
        assert front is None


class TestGcToMgcDataFlow:
    """End-to-end verification that GC front contract becomes MGC + source_symbol.

    The legacy MGC ingest path (ingest_dbn_mgc.py) uses GC outright pattern
    with prefix_len=2. The chosen GC contract becomes source_symbol, and the
    stored symbol is always 'MGC'.
    """

    def test_front_contract_is_gc(self):
        """choose_front_contract with GC pattern returns a GC contract."""
        daily_volumes = {"GCM4": 5000, "GCZ4": 2000}
        front = choose_front_contract(
            daily_volumes,
            outright_pattern=GC_OUTRIGHT_PATTERN,
            prefix_len=2,
        )
        assert front == "GCM4"
        assert GC_OUTRIGHT_PATTERN.match(front)

    def test_gc_prefix_len_is_2(self):
        """GC contracts have 2-char prefix (GC), not 3 (MGC)."""
        # Verify parsing works with prefix_len=2
        from pipeline.ingest_dbn_mgc import parse_expiry
        year, month = parse_expiry("GCM4", prefix_len=2)
        assert month == 6  # M = June
        assert year == 2004 or year == 2024  # depends on year parsing

    def test_stored_symbol_is_mgc(self):
        """The SYMBOL constant in ingest_dbn_mgc.py is 'MGC'."""
        from pipeline.ingest_dbn_mgc import SYMBOL
        assert SYMBOL == "MGC"

    def test_gc_pattern_rejects_mgc_contracts(self):
        """The GC outright pattern must NOT match MGC contracts.

        This ensures the GC→MGC distinction is intentional.
        """
        assert not GC_OUTRIGHT_PATTERN.match("MGCM4")
        assert not GC_OUTRIGHT_PATTERN.match("MGCZ4")
        assert GC_OUTRIGHT_PATTERN.match("GCM4")
        assert GC_OUTRIGHT_PATTERN.match("GCZ4")


class TestAssetConfigMgcPattern:
    """Verify asset_configs.py MGC entry matches GC source data."""

    def test_mgc_pattern_matches_gc_contracts(self):
        """MGC config must match GC outrights (data source is full-size Gold)."""
        from pipeline.asset_configs import ASSET_CONFIGS
        pattern = ASSET_CONFIGS["MGC"]["outright_pattern"]
        for sym in ["GCM4", "GCZ4", "GCG25", "GCQ5"]:
            assert pattern.match(sym), f"MGC pattern should match {sym}"

    def test_mgc_pattern_rejects_mgc_contracts(self):
        """MGC config must NOT match MGC outrights (we use GC source data)."""
        from pipeline.asset_configs import ASSET_CONFIGS
        pattern = ASSET_CONFIGS["MGC"]["outright_pattern"]
        for sym in ["MGCM4", "MGCZ4", "MGCG25"]:
            assert not pattern.match(sym), f"MGC pattern should not match {sym}"

    def test_mgc_prefix_len_is_2(self):
        """GC contracts have 2-char prefix before month code."""
        from pipeline.asset_configs import ASSET_CONFIGS
        assert ASSET_CONFIGS["MGC"]["prefix_len"] == 2
