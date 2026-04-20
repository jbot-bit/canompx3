"""Cached-file regression tests for canonical reprice_e2_entry.

Purpose: protect against silent canonical drift. If these tests fail, the
2026-04-20 MGC adversarial re-examination's baseline (mean=6.75 ticks,
median=0, max=263) is invalidated and the parent audit must be re-evaluated.

Synthetic edge-case coverage (no-cross, empty, pre-ORB filter) already lives
in `test_databento_microstructure.py`. This file ONLY covers cached-file
regression — published MGC pilot values reproducible from the 80-file cache.

Fixture source: `research/output/mgc_e2_repriced_entries.csv` + cached
`research/data/tbbo_pilot/*_MGC_FUT.dbn.zst` files.

If CI runner lacks the tbbo cache (local-data-only per CLAUDE.md "no cloud
sync"), tests skip with the env-aware pattern per HANDOFF 2026-04-19
"env-aware skipif pattern (NOT a blanket skip — DO NOT delete)".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pipeline.build_daily_features import _orb_utc_window
from pipeline.paths import PROJECT_ROOT
from research.databento_microstructure import RepricedEntry, load_tbbo_df, reprice_e2_entry

MGC_CACHE_DIR = PROJECT_ROOT / "research" / "data" / "tbbo_pilot"
MGC_MANIFEST = PROJECT_ROOT / "research" / "output" / "mgc_e2_microstructure_pilot_days.csv"
MGC_PUBLISHED_RESULTS = PROJECT_ROOT / "research" / "output" / "mgc_e2_repriced_entries.csv"

MGC_TICK_SIZE = 0.10
MGC_ORB_LABEL = "CME_REOPEN"
MGC_ORB_MINUTES = 5


def _require_mgc_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MGC manifest + published results; skip if either absent on CI."""
    if not MGC_MANIFEST.exists():
        pytest.skip(f"MGC manifest absent on CI: {MGC_MANIFEST}")
    if not MGC_PUBLISHED_RESULTS.exists():
        pytest.skip(f"MGC published results absent on CI: {MGC_PUBLISHED_RESULTS}")
    manifest = pd.read_csv(MGC_MANIFEST)
    published = pd.read_csv(MGC_PUBLISHED_RESULTS)
    return manifest, published


def _require_cache_file(day: str) -> Path:
    cache_path = MGC_CACHE_DIR / f"{day}_MGC_FUT.dbn.zst"
    if not cache_path.exists():
        pytest.skip(f"TBBO cache absent on CI: {cache_path}")
    return cache_path


def _run_reprice_from_published_row(row: pd.Series) -> RepricedEntry:
    """Given one manifest row, reproduce reprice_e2_entry call and return result.

    orb_end_utc = canonical orb_start + orb_minutes via `_orb_utc_window`.
    The MGC pilot's `model_entry_ts_utc` is `orb_end + break_delay_min` (not
    just orb_end), so using it as the filter gate would EXCLUDE trades
    between orb_end and first break. The canonical orb_end is what gates
    reprice_e2_entry's post-ORB window.
    """
    day = str(row["trading_day"])
    cache_path = _require_cache_file(day)
    tbbo_df = load_tbbo_df(cache_path)
    assert not tbbo_df.empty, f"Cached TBBO empty after front-month filter: {day}"

    orb_start_utc, orb_end_utc_dt = _orb_utc_window(date.fromisoformat(day), MGC_ORB_LABEL, MGC_ORB_MINUTES)
    assert orb_start_utc is not None
    orb_end_utc = orb_end_utc_dt.isoformat()

    return reprice_e2_entry(
        tbbo_df=tbbo_df,
        orb_high=float(row["orb_CME_REOPEN_high"]),
        orb_low=float(row["orb_CME_REOPEN_low"]),
        break_dir=str(row["break_dir"]),
        model_entry_price=float(row["model_entry_price"]),
        model_entry_ts_utc=str(row["model_entry_ts_utc"]),
        trading_day=day,
        symbol_pulled="MGC.FUT",
        tick_size=MGC_TICK_SIZE,
        modeled_slippage_ticks=int(row["modeled_entry_slippage_ticks"]),
        orb_end_utc=orb_end_utc,
    )


class TestMgcCachedFileRegression:
    """Regression guard for the 2026-04-20 MGC adversarial re-examination
    baseline. If these drift silently, the parent audit is invalidated."""

    def test_mgc_2017_04_26_long_clean_zero_slippage(self):
        """Non-event day: 2017-04-26 long MGC. Published result: slippage=0.0
        ticks, trigger=1270.8, fill=1270.8, 72 tbbo records."""
        manifest, published = _require_mgc_fixtures()
        manifest_row = manifest[manifest["trading_day"] == "2017-04-26"].iloc[0]
        published_row = published[
            (published["trading_day"] == "2017-04-26") & (published["symbol_pulled"] == "MGC.FUT")
        ].iloc[0]

        result = _run_reprice_from_published_row(manifest_row)

        assert result.error is None, f"Expected no error, got {result.error}"
        assert result.actual_slippage_ticks == pytest.approx(float(published_row["actual_slippage_ticks"]), abs=0.1), (
            f"MGC 2017-04-26 slippage regression: "
            f"got {result.actual_slippage_ticks}, "
            f"published {published_row['actual_slippage_ticks']}"
        )
        assert result.trigger_trade_price == pytest.approx(float(published_row["trigger_trade_price"]), abs=0.05)
        assert result.estimated_fill_price == pytest.approx(float(published_row["estimated_fill_price"]), abs=0.05)

    def test_mgc_2018_01_18_event_day_extreme_slippage_preserved(self):
        """Known event day (gap-open): 2018-01-18 long MGC. Published
        result: slippage=263.0 ticks, trigger=1354.4, fill=1354.4.

        This is the SINGLE outlier driving MGC's mean=6.75 (vs median=0). If
        the regression silently goes to 0 or a small value, the parent
        audit's 3.4× modeled friction claim is invalidated and needs review."""
        manifest, published = _require_mgc_fixtures()
        manifest_row = manifest[manifest["trading_day"] == "2018-01-18"].iloc[0]
        published_row = published[
            (published["trading_day"] == "2018-01-18") & (published["symbol_pulled"] == "MGC.FUT")
        ].iloc[0]

        result = _run_reprice_from_published_row(manifest_row)

        assert result.error is None, f"Expected no error, got {result.error}"
        # Tolerance ±1 tick on a 263-tick outlier to allow for tiny floating drift.
        assert result.actual_slippage_ticks == pytest.approx(float(published_row["actual_slippage_ticks"]), abs=1.0), (
            f"MGC 2018-01-18 event-day regression: "
            f"got {result.actual_slippage_ticks}, published "
            f"{published_row['actual_slippage_ticks']}. Silent drift on this "
            f"outlier invalidates parent audit §4 (MGC 3.4× modeled claim)."
        )


class TestLoadTbboDfFrontMonth:
    """Front-month filter keeps reprice scan on a single contract."""

    def test_load_tbbo_df_retains_single_symbol(self):
        """After load_tbbo_df's front-month filter, only ONE unique symbol
        should remain. Multi-symbol would mix prices across contracts and
        corrupt the first-cross detection."""
        cache_path = _require_cache_file("2017-04-26")
        df = load_tbbo_df(cache_path)
        assert not df.empty
        unique_symbols = df["symbol"].unique()
        assert len(unique_symbols) == 1, (
            f"load_tbbo_df returned {len(unique_symbols)} symbols ({list(unique_symbols)}); front-month filter broken"
        )
