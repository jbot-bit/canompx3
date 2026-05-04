"""
Tests for GC → MGC symbol transformation (LEGACY ingest path) AND
post-Phase-2 canonical asset_configs (REAL micro path).

This file has TWO conceptual halves:

1. LEGACY ingest path (`pipeline/ingest_dbn_mgc.py`) — historical full-size
   Gold (GC) data is preserved under symbol='MGC' in `bars_1m`. The legacy
   ingest module still uses `GC_OUTRIGHT_PATTERN` and is consumed by other
   pipeline modules for historical contract selection logic. These tests
   exercise that legacy module — DO NOT delete them; they pin invariants
   that downstream code depends on.

2. POST-PHASE-2 canonical path (`pipeline/asset_configs.py`) — Phase 2 of
   canonical-data-redownload (commit 82e8b60, Apr 8 2026) replaced the
   parent-futures data masquerading as MNQ/MES/MGC with REAL micro futures
   downloaded from the actual contracts. After that landed:
   - `ASSET_CONFIGS["MGC"]["outright_pattern"]` matches `^MGC...` (was `^GC...`)
   - `ASSET_CONFIGS["MGC"]["prefix_len"]` = 3 (was 2)
   - `ASSET_CONFIGS["ES"]["symbol"]` = "ES" (was "MES")
   The TestAssetConfigMgcPattern + TestAssetConfigMesPattern classes were
   updated 2026-04-08 (Move C of canonical-data-redownload sweep) to assert
   the post-Phase-2 reality. Earlier assertions encoding the pre-Phase-2
   pattern were caught by `pytest tests/test_pipeline/test_gc_mgc_mapping.py`
   showing 4 failures after Phase 2 landed but before this sweep — see
   `docs/runtime/stages/move-c-phase-2-regressions.md`.
"""

import re
from datetime import date

import pytest

from pipeline.ingest_dbn_mgc import (
    GC_OUTRIGHT_PATTERN,
    choose_front_contract,
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
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert front == "GCM4"

    def test_deterministic_tiebreak(self):
        """Equal volume: earliest expiry wins, then lexicographic."""
        volumes = {"GCZ4": 5000, "GCM4": 5000}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        # M (June) expires before Z (December)
        assert front == "GCM4"

    def test_filters_non_outrights(self):
        """Non-outright symbols in volume dict are ignored."""
        volumes = {"GCM4": 100, "GC-SPREAD": 9999, "INVALID": 5000}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert front == "GCM4"

    def test_empty_volumes_returns_none(self):
        """No valid outrights → None."""
        front = choose_front_contract({}, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert front is None

    def test_no_outrights_returns_none(self):
        """Only non-matching symbols → None."""
        volumes = {"INVALID1": 100, "INVALID2": 200}
        front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
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
    """Verify asset_configs.py MGC entry matches REAL MGC micro contracts.

    POST-PHASE-2 (commit 82e8b60, Apr 8 2026): The MGC config was flipped from
    `^GC[FGHJKMNQUVXZ]\\d+$` (parent proxy) to `^MGC[FGHJKMNQUVXZ]\\d+$` (real
    micro) when Phase 2 of canonical-data-redownload replaced the parent
    futures data with real micro data from MGC.FUT (launched 2023-09-11).
    These tests pin the post-Phase-2 reality.
    """

    def test_mgc_pattern_matches_mgc_contracts(self):
        """MGC config must match real MGC micro outrights (post-Phase-2)."""
        from pipeline.asset_configs import ASSET_CONFIGS

        pattern = ASSET_CONFIGS["MGC"]["outright_pattern"]
        for sym in ["MGCM4", "MGCZ4", "MGCG25", "MGCQ5"]:
            assert pattern.match(sym), f"MGC pattern should match {sym}"

    def test_mgc_pattern_rejects_gc_parent_contracts(self):
        """MGC config must NOT match GC parent outrights (post-Phase-2 separation).

        Pre-Phase-2 the GC parent data was stored under symbol='MGC'. Phase 2
        relabeled all parent rows to symbol='GC' and downloaded real MGC.FUT
        data. The MGC outright_pattern must reject any GC* symbol that arrives
        — those belong under the GC config (parent), not MGC (real micro).
        """
        from pipeline.asset_configs import ASSET_CONFIGS

        pattern = ASSET_CONFIGS["MGC"]["outright_pattern"]
        for sym in ["GCM4", "GCZ4", "GCG25", "GCQ5"]:
            assert not pattern.match(sym), f"MGC pattern should not match parent {sym}"

    def test_mgc_prefix_len_is_3(self):
        """Real MGC contracts have 3-char prefix (MGC) before month code (post-Phase-2)."""
        from pipeline.asset_configs import ASSET_CONFIGS

        assert ASSET_CONFIGS["MGC"]["prefix_len"] == 3


class TestAssetConfigMesPattern:
    """Verify MES and ES configs handle the ES→MES source data split."""

    def test_mes_pattern_matches_mes_contracts(self):
        """MES config must match native MES outrights (2024+)."""
        from pipeline.asset_configs import ASSET_CONFIGS

        pattern = ASSET_CONFIGS["MES"]["outright_pattern"]
        for sym in ["MESM4", "MESZ4", "MESH25"]:
            assert pattern.match(sym), f"MES pattern should match {sym}"

    def test_mes_dbn_path_exists(self):
        """MES config dbn_path must point to a non-empty directory.

        Local-data sentinel: this test's purpose is to alert the developer
        when the MES DBN store is missing. CI runners legitimately have no
        local DBN data per CLAUDE.md ("ONE database ... local disk, no
        cloud sync"), so we skip the check there. Same env-aware pattern
        as pipeline.check_drift._skip_db_check_for_ci.
        """
        from pipeline.asset_configs import ASSET_CONFIGS

        path = ASSET_CONFIGS["MES"]["dbn_path"]
        if not path.exists():
            pytest.skip(
                f"MES DBN data not present (expected at {path}). Local-data sentinel — by design absent on CI runners."
            )
        assert path.is_dir(), f"MES dbn_path is not a directory: {path}"

    def test_es_config_exists(self):
        """An ES config entry must exist for pre-2024 ES→MES mapping."""
        from pipeline.asset_configs import ASSET_CONFIGS

        assert "ES" in ASSET_CONFIGS, "ES config needed for pre-2024 data"

    def test_es_pattern_matches_es_contracts(self):
        """ES config must match ES outrights."""
        from pipeline.asset_configs import ASSET_CONFIGS

        pattern = ASSET_CONFIGS["ES"]["outright_pattern"]
        for sym in ["ESH5", "ESM9", "ESZ24"]:
            assert pattern.match(sym), f"ES pattern should match {sym}"

    def test_es_pattern_rejects_mes_contracts(self):
        """ES config must NOT match MES outrights."""
        from pipeline.asset_configs import ASSET_CONFIGS

        pattern = ASSET_CONFIGS["ES"]["outright_pattern"]
        for sym in ["MESM4", "MESZ4"]:
            assert not pattern.match(sym), f"ES pattern should not match {sym}"

    def test_es_stores_as_es_symbol(self):
        """ES config must store data under symbol='ES' (post-Phase-2 relabeling).

        Pre-Phase-2 (commit 82e8b60, Apr 8 2026), ES parent data was stored
        under symbol='MES' (mislabeled). Phase 2 relabeled all parent rows to
        symbol='ES' and downloaded real micro MES.FUT data into symbol='MES'.
        ES is now its own canonical entry — historical parent data, $50/pt
        cost specs, no orb_active.
        """
        from pipeline.asset_configs import ASSET_CONFIGS

        assert ASSET_CONFIGS["ES"]["symbol"] == "ES"

    def test_es_prefix_len_is_2(self):
        """ES contracts have 2-char prefix (ES) before month code."""
        from pipeline.asset_configs import ASSET_CONFIGS

        assert ASSET_CONFIGS["ES"]["prefix_len"] == 2
