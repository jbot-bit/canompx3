"""Tests for promotion candidate report generation."""

import pytest

from scripts.tools.generate_promotion_candidates import (
    enrich_candidate,
    find_uncovered_candidates,
    format_terminal,
    generate_html,
    generate_spec_code,
)
from pipeline.paths import GOLD_DB_PATH
from trading_app.live_config import LIVE_PORTFOLIO


class TestFindUncoveredCandidates:
    def test_excludes_live_portfolio(self):
        """Candidates must NOT include (orb_label, entry_model, filter_type) already in LIVE_PORTFOLIO."""
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        covered = {(s.orb_label, s.entry_model, s.filter_type) for s in LIVE_PORTFOLIO}
        for c in candidates:
            key = (c["orb_label"], c["entry_model"], c["filter_type"])
            assert key not in covered, f"{c['strategy_id']} is already in LIVE_PORTFOLIO"

    def test_candidates_are_fdr_wf_robust(self):
        """Every candidate must be FDR-significant, WF-passed, and ROBUST."""
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        for c in candidates:
            assert c["fdr_significant"] is True, f"{c['strategy_id']} not FDR-sig"
            assert c["wf_passed"] is True, f"{c['strategy_id']} not WF-passed"
            assert c["robustness_status"] == "ROBUST", f"{c['strategy_id']} not ROBUST"

    def test_sorted_by_expr_desc(self):
        """Candidates must be sorted by ExpR descending."""
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        if len(candidates) > 1:
            exprs = [c["expectancy_r"] for c in candidates]
            assert exprs == sorted(exprs, reverse=True)

    def test_returns_list_of_dicts(self):
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        assert isinstance(candidates, list)
        if candidates:
            assert isinstance(candidates[0], dict)
            assert "strategy_id" in candidates[0]
            assert "instrument" in candidates[0]


class TestEnrichCandidate:
    def test_has_required_fields(self):
        """Enriched candidate must have year_by_year, decay_slope, dollar_gate_results."""
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = enrich_candidate(candidates[0])
        assert "year_by_year" in enriched
        assert "decay_slope" in enriched
        assert "dollar_gate_results" in enriched
        assert isinstance(enriched["year_by_year"], list)
        assert isinstance(enriched["dollar_gate_results"], dict)

    def test_year_by_year_has_fields(self):
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
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
    def test_includes_summary(self):
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        output = format_terminal(enriched)
        assert "PROMOTION CANDIDATES" in output
        assert "LiveStrategySpec" in output

    def test_includes_strategy_ids(self):
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        output = format_terminal(enriched)
        assert enriched[0]["strategy_id"] in output


class TestGenerateHtml:
    def test_contains_required_sections(self):
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        html = generate_html(enriched)
        assert "PROMOTION CANDIDATES" in html
        assert "LiveStrategySpec" in html
        assert "Year-by-Year" in html
        assert "</html>" in html

    def test_contains_all_candidate_ids(self):
        candidates = find_uncovered_candidates(GOLD_DB_PATH)
        if not candidates:
            pytest.skip("No uncovered candidates")
        enriched = [enrich_candidate(c) for c in candidates[:3]]
        html = generate_html(enriched)
        for c in enriched:
            assert c["strategy_id"] in html
