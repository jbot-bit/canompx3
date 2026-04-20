"""Tests for MNQ pilot caller (the part that wraps canonical reprice_e2_entry).

Canonical reprice logic itself is covered by `test_databento_microstructure.py`
(synthetic edge cases) and `test_reprice_e2_entry_regression.py` (real cached
MGC days). This file covers ONLY the MNQ-pilot-specific glue:
filename parsing, manifest-from-cache building, and the --reprice-cache
end-to-end path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.paths import PROJECT_ROOT
from research.research_mnq_e2_slippage_pilot import (
    build_manifest_from_cache,
    parse_cache_filename,
)

MNQ_CACHE_DIR = PROJECT_ROOT / "research" / "data" / "tbbo_mnq_pilot"


class TestParseCacheFilename:
    """Filename regex → (trading_day, orb_label)."""

    def test_parses_standard_filename(self):
        day, session = parse_cache_filename("2024-09-05_TOKYO_OPEN_MNQ.dbn.zst")
        assert day == "2024-09-05"
        assert session == "TOKYO_OPEN"

    def test_parses_multiword_session(self):
        day, session = parse_cache_filename("2023-03-20_US_DATA_830_MNQ.dbn.zst")
        assert day == "2023-03-20"
        assert session == "US_DATA_830"

    def test_rejects_non_mnq_file(self):
        result = parse_cache_filename("2024-09-05_TOKYO_OPEN_MGC.dbn.zst")
        assert result is None

    def test_rejects_malformed_filename(self):
        assert parse_cache_filename("random_file.zst") is None
        assert parse_cache_filename("2024-09-05.dbn.zst") is None


class TestBuildManifestFromCache:
    """Reverse-engineer manifest from cached filenames via daily_features join."""

    def _require_cache(self):
        if not MNQ_CACHE_DIR.exists() or not any(MNQ_CACHE_DIR.glob("*_MNQ.dbn.zst")):
            pytest.skip(f"MNQ cache absent: {MNQ_CACHE_DIR}")

    def test_manifest_has_required_fields(self):
        """Each manifest row must have orb_high, orb_low, break_dir, atr_20
        sourced from daily_features (NOT dummy values)."""
        self._require_cache()
        manifest = build_manifest_from_cache(MNQ_CACHE_DIR)
        assert len(manifest) > 0, "No manifest rows built from cache"

        required_fields = {
            "trading_day",
            "orb_label",
            "cache_path",
            "orb_high",
            "orb_low",
            "break_dir",
            "atr_20",
        }
        sample = manifest[0]
        assert required_fields.issubset(sample.keys()), (
            f"Missing required fields: {required_fields - sample.keys()}"
        )

    def test_manifest_orb_high_never_dummy(self):
        """Prior caller passed dummy `orb_level ± 1.0` — guard against
        regression where orb_high/low come from a constant-delta dummy."""
        self._require_cache()
        manifest = build_manifest_from_cache(MNQ_CACHE_DIR)
        valid_rows = [r for r in manifest if r.get("orb_high") is not None]
        assert len(valid_rows) > 0
        for row in valid_rows[:10]:
            delta = abs(float(row["orb_high"]) - float(row["orb_low"]))
            assert delta != 1.0, (
                f"Suspect dummy orb bounds for {row['trading_day']} "
                f"{row['orb_label']}: delta=1.0"
            )

    def test_manifest_excludes_days_missing_daily_features(self):
        """Cached files with no daily_features row (e.g., data gap) must be
        excluded with error='daily_features missing', not crashed or silently
        dropped with valid=True."""
        self._require_cache()
        manifest = build_manifest_from_cache(MNQ_CACHE_DIR)
        # Rows should either have valid bounds OR be explicitly marked invalid
        for row in manifest:
            if row.get("orb_high") is None or row.get("orb_low") is None:
                assert row.get("error") is not None, (
                    f"Row {row['trading_day']} {row['orb_label']} has null "
                    f"bounds but no error marker"
                )


class TestRepriceCacheIntegration:
    """End-to-end --reprice-cache output shape."""

    def _require_cache(self):
        if not MNQ_CACHE_DIR.exists() or not any(MNQ_CACHE_DIR.glob("*_MNQ.dbn.zst")):
            pytest.skip(f"MNQ cache absent: {MNQ_CACHE_DIR}")

    def test_reprice_cache_produces_non_empty_results(self):
        """Running reprice against the cache manifest must produce valid
        rows (not all errors)."""
        self._require_cache()
        from research.research_mnq_e2_slippage_pilot import reprice_cache_manifest

        manifest = build_manifest_from_cache(MNQ_CACHE_DIR)
        results = reprice_cache_manifest(manifest[:5])  # first 5 only for speed
        assert len(results) == 5
        # At least SOME of the 5 must succeed (no cache absence; first 5
        # taken in a rough sort should include a valid one)
        valid = [r for r in results if r.get("error") is None]
        assert len(valid) > 0, f"All 5 cache results errored: {results}"

    def test_reprice_cache_no_dummy_pollution(self):
        """A valid result row must NOT have orb_level that equals
        orb_high AND orb_low simultaneously (would be dummy single-side
        pollution from the pre-rewrite caller)."""
        self._require_cache()
        from research.research_mnq_e2_slippage_pilot import reprice_cache_manifest

        manifest = build_manifest_from_cache(MNQ_CACHE_DIR)
        results = reprice_cache_manifest(manifest[:5])
        for r in results:
            if r.get("error") is not None:
                continue
            assert r["orb_high"] != r["orb_low"], (
                f"Dummy bounds pollution for {r['trading_day']} "
                f"{r['orb_label']}: high == low"
            )
