"""Equivalence test for the lane-correlation bulk-cache refactor.

Loads ≥5 candidates spanning the four filter-class branches:
  - NO_FILTER (early-return path; daily_features never loaded)
  - VolumeFilter (rel_vol enrichment)
  - CrossAssetATRFilter (cross-asset ATR injection)
  - Ordinary daily-features filter (no enrichment, just matches_row)

And ≥2 distinct (instrument, orb_minutes) pairs so cache key collisions
are exercised. Asserts byte-identical `pnl_series` dicts produced by the
single-strategy path vs. the cached pairwise path.

Skipped if gold.db is unavailable in CI.
"""

from __future__ import annotations

from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter, VolumeFilter
from trading_app.lane_allocator import compute_lane_scores, compute_pairwise_correlation
from trading_app.lane_correlation import (
    _load_lane_daily_pnl,
    _load_lane_daily_pnl_cached,
)


def _build_diverse_candidates(rebalance_date: date) -> list:
    """Pick ≥5 candidates spanning all four filter-class branches and ≥2
    distinct (instrument, orb_minutes) pairs.
    """
    scores = compute_lane_scores(rebalance_date=rebalance_date)
    deploy = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]

    picked: list = []
    seen_classes: set[str] = set()
    seen_im_om: set[tuple[str, int]] = set()

    def _class_of(filter_type: str) -> str:
        if filter_type == "NO_FILTER":
            return "NO_FILTER"
        f = ALL_FILTERS.get(filter_type)
        if isinstance(f, VolumeFilter):
            return "VolumeFilter"
        if isinstance(f, CrossAssetATRFilter):
            return "CrossAssetATRFilter"
        if f is not None:
            return "OrdinaryFilter"
        return "Unknown"

    target_classes = {"NO_FILTER", "VolumeFilter", "CrossAssetATRFilter", "OrdinaryFilter"}

    # Pass 1: one per class.
    for s in deploy:
        c = _class_of(s.filter_type)
        if c in target_classes and c not in seen_classes:
            picked.append(s)
            seen_classes.add(c)
            seen_im_om.add((s.instrument, s.orb_minutes))
        if seen_classes == target_classes:
            break

    # Pass 2: ensure ≥2 distinct (instrument, orb_minutes) pairs.
    if len(seen_im_om) < 2:
        for s in deploy:
            if (s.instrument, s.orb_minutes) not in seen_im_om:
                picked.append(s)
                seen_im_om.add((s.instrument, s.orb_minutes))
                break

    # Pass 3: pad to ≥5 with any deploy candidates not yet picked.
    picked_ids = {s.strategy_id for s in picked}
    for s in deploy:
        if len(picked) >= 5:
            break
        if s.strategy_id not in picked_ids:
            picked.append(s)
            picked_ids.add(s.strategy_id)

    return picked


@pytest.mark.skipif(
    not GOLD_DB_PATH.exists(),
    reason="gold.db unavailable in this environment",
)
def test_cached_path_byte_identical_to_uncached_path():
    candidates = _build_diverse_candidates(date(2026, 5, 21))
    assert len(candidates) >= 5, f"need ≥5 diverse candidates, got {len(candidates)}"

    # ============== Uncached reference (single-strategy path) ==============
    con_ref = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con_ref)
    try:
        uncached_pnl: dict[str, dict] = {}
        for s in candidates:
            lane = {
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "orb_minutes": s.orb_minutes,
                "entry_model": "E2",
                "rr_target": s.rr_target,
                "confirm_bars": s.confirm_bars,
                "filter_type": s.filter_type,
            }
            uncached_pnl[s.strategy_id] = _load_lane_daily_pnl(con_ref, lane)
    finally:
        con_ref.close()

    # ============== Cached path (what compute_pairwise_correlation uses) ===
    con_test = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con_test)
    try:
        outcomes_cache: dict = {}
        features_cache: dict = {}
        applied_enrichments: set = set()
        cached_pnl: dict[str, dict] = {}
        for s in candidates:
            lane = {
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "orb_minutes": s.orb_minutes,
                "entry_model": "E2",
                "rr_target": s.rr_target,
                "confirm_bars": s.confirm_bars,
                "filter_type": s.filter_type,
            }
            cached_pnl[s.strategy_id] = _load_lane_daily_pnl_cached(
                con_test,
                lane,
                outcomes_cache,
                features_cache,
                applied_enrichments,
            )
    finally:
        con_test.close()

    # ============== Byte-identical assertion ==============
    # Compare key sets first — catches dropped rows immediately.
    for sid in uncached_pnl:
        ref_keys = set(uncached_pnl[sid].keys())
        test_keys = set(cached_pnl[sid].keys())
        assert ref_keys == test_keys, (
            f"{sid}: trading_day sets differ. "
            f"uncached_only={sorted(ref_keys - test_keys)[:5]} "
            f"cached_only={sorted(test_keys - ref_keys)[:5]}"
        )

    # Compare values per-day with exact equality (floats are summed deterministically).
    for sid in uncached_pnl:
        for day in uncached_pnl[sid]:
            ref_v = uncached_pnl[sid][day]
            test_v = cached_pnl[sid][day]
            assert ref_v == test_v, (
                f"{sid}@{day}: uncached={ref_v} != cached={test_v}"
            )


@pytest.mark.skipif(
    not GOLD_DB_PATH.exists(),
    reason="gold.db unavailable in this environment",
)
def test_compute_pairwise_correlation_matches_per_strategy_path():
    """End-to-end equivalence: compute_pairwise_correlation() against an
    independent _load_lane_daily_pnl-built pnl_series.
    """
    candidates = _build_diverse_candidates(date(2026, 5, 21))[:5]

    # Reference pairs via the original per-strategy path.
    con_ref = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con_ref)
    try:
        pnl: dict[str, dict] = {}
        for s in candidates:
            lane = {
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "orb_minutes": s.orb_minutes,
                "entry_model": "E2",
                "rr_target": s.rr_target,
                "confirm_bars": s.confirm_bars,
                "filter_type": s.filter_type,
            }
            pnl[s.strategy_id] = _load_lane_daily_pnl(con_ref, lane)
    finally:
        con_ref.close()

    # Build reference pairs identically to the production loop.
    from trading_app.lane_correlation import _pearson

    sids = [s.strategy_id for s in candidates]
    ref_pairs: dict[tuple[str, str], float] = {}
    for i, a in enumerate(sids):
        for j, b in enumerate(sids):
            if j <= i:
                continue
            key = (a, b) if a < b else (b, a)
            shared = sorted(set(pnl[a]) & set(pnl[b]))
            if len(shared) >= 5:
                xs = [pnl[a][d] for d in shared]
                ys = [pnl[b][d] for d in shared]
                ref_pairs[key] = _pearson(xs, ys)
            else:
                ref_pairs[key] = 0.0

    # Production path.
    actual_pairs = compute_pairwise_correlation(candidates)

    assert set(actual_pairs.keys()) == set(ref_pairs.keys())
    for key in ref_pairs:
        assert actual_pairs[key] == ref_pairs[key], (
            f"pair {key}: ref={ref_pairs[key]} != actual={actual_pairs[key]}"
        )
