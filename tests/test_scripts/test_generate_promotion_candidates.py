"""Tests for promotion candidate report generation."""

import duckdb
import pytest

from scripts.tools.generate_promotion_candidates import (
    build_day_sets,
    enrich_candidate,
    find_uncovered_candidates,
    format_terminal,
    generate_html,
    generate_spec_code,
)
from trading_app.config import ALL_FILTERS, OrbSizeFilter
from trading_app.live_config import LIVE_PORTFOLIO


def _live_portfolio_triplets() -> set[tuple[str, str, str]]:
    return {(s.orb_label, s.entry_model, s.filter_type) for s in LIVE_PORTFOLIO}


def _unused_triplet() -> tuple[str, str, str]:
    """Return an (orb_label, entry_model, filter_type) that is NOT in LIVE_PORTFOLIO.

    Picks from a whitelist of canonical options and returns the first triplet
    that isn't already covered. Fail-closed if the whitelist is exhausted —
    that means LIVE_PORTFOLIO needs broader candidates in this fixture.
    """
    covered = _live_portfolio_triplets()
    candidates = [
        ("CME_PRECLOSE", "E2", "ORB_G8"),
        ("CME_PRECLOSE", "E2", "ORB_G5"),
        ("LONDON_METALS", "E2", "ORB_G8"),
        ("NYSE_OPEN", "E2", "ORB_G8"),
        ("SINGAPORE_OPEN", "E2", "ORB_G8"),
    ]
    for triplet in candidates:
        if triplet not in covered:
            return triplet
    raise RuntimeError("All candidate triplets collide with LIVE_PORTFOLIO — extend the whitelist.")


@pytest.fixture
def seeded_promotion_db(tmp_path):
    """Temp gold.db with canonical schema + one ROBUST, FDR+WF+Active candidate.

    Uses `trading_app.db_manager.init_trading_app_schema` for
    validated_setups/edge_families/experimental_strategies/orb_outcomes
    schema — keeps the fixture in lockstep with production schema.
    """
    from pipeline.init_db import DAILY_FEATURES_SCHEMA
    from trading_app.db_manager import init_trading_app_schema

    db_path = tmp_path / "gold.db"
    # daily_features must exist first because orb_outcomes has an FK into it.
    con = duckdb.connect(str(db_path))
    try:
        con.execute(DAILY_FEATURES_SCHEMA)
    finally:
        con.close()
    init_trading_app_schema(db_path=db_path)

    orb_label, entry_model, filter_type = _unused_triplet()

    con = duckdb.connect(str(db_path))
    try:
        strategy_id = f"MNQ_{orb_label}_{entry_model}_CB1_{filter_type}_RR1.5"
        con.execute(
            """
            INSERT INTO validated_setups (
                strategy_id, instrument, orb_label, entry_model, orb_minutes,
                rr_target, confirm_bars, filter_type, status, deployment_scope,
                sample_size, win_rate, expectancy_r, sharpe_ann, max_drawdown_r,
                years_tested, all_years_positive, stress_test_passed, yearly_results,
                fdr_significant, fdr_adjusted_p,
                wf_passed, wf_windows, wfe,
                skewness, kurtosis_excess, stop_multiplier
            ) VALUES (?, 'MNQ', ?, ?, 5, 1.5, 1, ?, 'active', 'deployable',
                      200, 0.55, 0.22, 1.4, 3.0,
                      6, TRUE, TRUE, '{}',
                      TRUE, 0.01,
                      TRUE, 3, 0.75,
                      0.1, 0.2, 1.0)
            """,
            [strategy_id, orb_label, entry_model, filter_type],
        )
        con.execute(
            """
            INSERT INTO edge_families (
                family_hash, instrument, member_count, trade_day_count,
                head_strategy_id, head_expectancy_r, head_sharpe_ann,
                robustness_status, cv_expectancy, trade_tier, pbo
            ) VALUES (?, 'MNQ', 1, 200, ?, 0.22, 1.4, 'ROBUST', 0.18, 'CORE', 0.15)
            """,
            [
                f"MNQ_{orb_label}_{entry_model}_{filter_type}",
                strategy_id,
            ],
        )
        con.execute(
            """
            INSERT INTO experimental_strategies (
                strategy_id, instrument, orb_label, entry_model, orb_minutes,
                rr_target, confirm_bars, filter_type, sample_size, median_risk_points,
                is_canonical
            ) VALUES (?, 'MNQ', ?, ?, 5, 1.5, 1, ?, 200, 20.0, TRUE)
            """,
            [strategy_id, orb_label, entry_model, filter_type],
        )
    finally:
        con.close()
    return db_path


class TestFindUncoveredCandidates:
    def test_excludes_live_portfolio(self, seeded_promotion_db):
        """Candidates must NOT include (orb_label, entry_model, filter_type) already in LIVE_PORTFOLIO."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        covered = _live_portfolio_triplets()
        for c in candidates:
            key = (c["orb_label"], c["entry_model"], c["filter_type"])
            assert key not in covered, f"{c['strategy_id']} is already in LIVE_PORTFOLIO"

    def test_candidates_are_fdr_wf_robust(self, seeded_promotion_db):
        """Every candidate must be FDR-significant, WF-passed, and ROBUST."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        for c in candidates:
            assert c["fdr_significant"] is True, f"{c['strategy_id']} not FDR-sig"
            assert c["wf_passed"] is True, f"{c['strategy_id']} not WF-passed"
            assert c["robustness_status"] == "ROBUST", f"{c['strategy_id']} not ROBUST"

    def test_sorted_by_expr_desc(self, seeded_promotion_db):
        """Candidates must be sorted by ExpR descending."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if len(candidates) > 1:
            exprs = [c["expectancy_r"] for c in candidates]
            assert exprs == sorted(exprs, reverse=True)

    def test_returns_list_of_dicts(self, seeded_promotion_db):
        candidates = find_uncovered_candidates(seeded_promotion_db)
        assert isinstance(candidates, list)
        if candidates:
            assert isinstance(candidates[0], dict)
            assert "strategy_id" in candidates[0]
            assert "instrument" in candidates[0]


class TestEnrichCandidate:
    def test_has_required_fields(self, seeded_promotion_db):
        """Enriched candidate must have year_by_year, decay_slope, dollar_gate_results."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = enrich_candidate(candidates[0])
        assert "year_by_year" in enriched
        assert "decay_slope" in enriched
        assert "dollar_gate_results" in enriched
        assert isinstance(enriched["year_by_year"], list)
        assert isinstance(enriched["dollar_gate_results"], dict)

    def test_year_by_year_has_fields(self, seeded_promotion_db):
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = enrich_candidate(candidates[0])
        if enriched["year_by_year"]:
            yr = enriched["year_by_year"][0]
            assert "year" in yr
            assert "n" in yr
            assert "avg_r" in yr


class TestGenerateSpecCode:
    def test_produces_valid_python(self):
        code = generate_spec_code("CME_PRECLOSE", "E2", "ORB_G8")
        assert "LiveStrategySpec(" in code
        assert "CME_PRECLOSE" in code
        assert "E2" in code
        assert "ORB_G8" in code
        compile(code, "<test>", "eval")

    def test_family_id_format(self):
        code = generate_spec_code("TOKYO_OPEN", "E2", "VOL_RV12_N20")
        assert "TOKYO_OPEN_E2_VOL_RV12_N20" in code

    def test_core_tier_for_large_sample(self):
        code = generate_spec_code("CME_PRECLOSE", "E2", "ORB_G8", sample_size=500)
        assert '"core"' in code
        assert "high_vol" not in code

    def test_regime_tier_for_small_sample(self):
        code = generate_spec_code("TOKYO_OPEN", "E2", "ORB_G4", sample_size=91)
        assert '"regime"' in code
        assert '"high_vol"' in code

    def test_boundary_100_is_core(self):
        code = generate_spec_code("CME_PRECLOSE", "E2", "ORB_G8", sample_size=100)
        assert '"core"' in code

    def test_boundary_99_is_regime(self):
        code = generate_spec_code("CME_PRECLOSE", "E2", "ORB_G8", sample_size=99)
        assert '"regime"' in code


class TestFormatTerminal:
    def test_includes_summary(self, seeded_promotion_db):
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        output = format_terminal(enriched)
        assert "PROMOTION CANDIDATES" in output
        assert "LiveStrategySpec" in output

    def test_includes_strategy_ids(self, seeded_promotion_db):
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        output = format_terminal(enriched)
        assert enriched[0]["strategy_id"] in output


class TestGenerateHtml:
    def test_contains_required_sections(self, seeded_promotion_db):
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        html = generate_html(enriched)
        assert "PROMOTION CANDIDATES" in html
        assert "LiveStrategySpec" in html
        assert "Year-by-Year" in html
        assert "</html>" in html

    def test_contains_all_candidate_ids(self, seeded_promotion_db):
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        html = generate_html(enriched)
        for c in enriched:
            assert c["strategy_id"] in html


class TestBuildDaySets:
    def test_returns_frozensets(self, seeded_promotion_db):
        """build_day_sets must return frozensets of strings."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        orb_candidates = [
            c
            for c in candidates
            if c["filter_type"] in ALL_FILTERS and isinstance(ALL_FILTERS[c["filter_type"]], OrbSizeFilter)
        ]
        if not orb_candidates:
            pytest.skip("No OrbSizeFilter candidates to build day sets for")
        con = duckdb.connect(str(seeded_promotion_db), read_only=True)
        try:
            result = build_day_sets(con, orb_candidates[:2])
        finally:
            con.close()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, frozenset)
            if v:
                assert isinstance(next(iter(v)), str)

    def test_vol_filter_candidates_skipped(self, seeded_promotion_db):
        """VolumeFilter candidates should not appear in day_sets keys."""
        # Synthesise a fake VOL candidate
        fake_vol = {
            "instrument": "MNQ",
            "orb_label": "TOKYO_OPEN",
            "entry_model": "E2",
            "filter_type": "VOL_RV12_N20",
            "orb_minutes": 5,
        }
        con = duckdb.connect(str(seeded_promotion_db), read_only=True)
        try:
            result = build_day_sets(con, [fake_vol])
        finally:
            con.close()
        # VOL candidate should produce no keys (VolumeFilter skipped)
        assert not any(k[3] == "VOL_RV12_N20" for k in result)


class TestOverlapFields:
    def test_enrich_with_no_day_sets_gives_none_overlap(self, seeded_promotion_db):
        """enrich_candidate with day_sets=None must set overlap fields to None."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = enrich_candidate(candidates[0], day_sets=None)
        assert enriched["overlap_pct"] is None
        assert enriched["overlap_with"] is None
        assert enriched["marginal_days"] is None

    def test_overlap_pct_in_range(self, seeded_promotion_db):
        """overlap_pct must be in [0, 1] when computed."""
        candidates = find_uncovered_candidates(seeded_promotion_db)
        orb_candidates = [
            c
            for c in candidates
            if c["filter_type"] in ALL_FILTERS and isinstance(ALL_FILTERS[c["filter_type"]], OrbSizeFilter)
        ]
        if not orb_candidates:
            pytest.skip("No OrbSizeFilter candidates")
        con = duckdb.connect(str(seeded_promotion_db), read_only=True)
        try:
            day_sets = build_day_sets(con, orb_candidates[:3])
        finally:
            con.close()
        for c in orb_candidates[:3]:
            enriched = enrich_candidate(c, day_sets=day_sets)
            if enriched["overlap_pct"] is not None:
                assert 0.0 <= enriched["overlap_pct"] <= 1.0
            if enriched["marginal_days"] is not None:
                assert enriched["marginal_days"] >= 0
