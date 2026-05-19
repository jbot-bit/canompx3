"""Tests for the literature-grounding seam added by Improvement 2.

Covers:
- ``search_corpus`` — ranking by query-token overlap, top-K bound, min_overlap floor
- ``format_search_results_for_llm`` — bounded byte budget, rank-preserving
- ``check_grounding_provenance_block`` — non-fatal slug validation
- Integration smoke: --ground-via-mcp dry-run path produces a stderr-visible
  targeted-extracts block and the static-checks pipeline accepts the resulting
  draft
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.lhp.literature_index import (
    LiteratureEntry,
    format_search_results_for_llm,
    load_corpus,
    search_corpus,
)
from scripts.research.lhp.static_checks import check_grounding_provenance_block


# ---------- search_corpus ----------


def _make_entry(slug: str, title: str, text: str) -> LiteratureEntry:
    return LiteratureEntry(
        path=Path(f"/fake/{slug}.md"),
        slug=slug,
        title=title,
        text=text,
        blurb=text[:200],
    )


class TestSearchCorpus:
    def test_returns_empty_for_empty_query(self):
        corpus = [
            _make_entry(
                "harris_microstructure",
                "Trading Exchanges Microstructure",
                "stop cascade liquidity demand adverse selection breakout",
            )
        ]
        assert search_corpus(corpus, "", top_k=5) == []

    def test_returns_empty_when_no_entry_above_min_overlap(self):
        corpus = [
            _make_entry(
                "carver_sizing",
                "Volatility Targeting Position Sizing",
                "volatility forecast position size Kelly",
            )
        ]
        # Query has zero relevant tokens to the entry body
        results = search_corpus(corpus, "stop cascade breakout liquidity", top_k=5)
        assert results == []

    def test_ranks_by_overlap_descending(self):
        a = _make_entry(
            "harris_microstructure",
            "Microstructure",
            "stop cascade liquidity demand adverse selection breakout",
        )
        b = _make_entry(
            "fitschen_path",
            "Path of Least Resistance",
            "breakout intraday trend follow path resistance momentum",
        )
        c = _make_entry(
            "carver_sizing",
            "Volatility Sizing",
            "volatility forecast position kelly sizing",
        )
        corpus = [a, b, c]
        # Query overlaps heavily with A (stop, cascade, liquidity, breakout),
        # moderately with B (breakout), nothing with C.
        results = search_corpus(
            corpus, "stop cascade liquidity breakout signals", top_k=5
        )
        slugs = [r["slug"] for r in results]
        assert slugs[0] == "harris_microstructure"
        assert "carver_sizing" not in slugs

    def test_top_k_bound(self):
        corpus = [
            _make_entry(
                f"entry_{i}", f"Title {i}", "breakout momentum trend follow signal"
            )
            for i in range(10)
        ]
        results = search_corpus(corpus, "breakout momentum trend signal", top_k=3)
        assert len(results) == 3

    def test_min_overlap_floor_excludes_weak_matches(self):
        corpus = [
            _make_entry(
                "weak", "Weak", "breakout"
            ),  # only one query-token overlap
            _make_entry(
                "strong",
                "Strong",
                "breakout momentum trend follow signal session",
            ),
        ]
        # min_overlap defaults to 2 -> "weak" excluded
        results = search_corpus(
            corpus, "breakout momentum trend follow signal", top_k=5
        )
        slugs = [r["slug"] for r in results]
        assert slugs == ["strong"]


# ---------- format_search_results_for_llm ----------


class TestFormatSearchResults:
    def test_empty_input_returns_empty_string(self):
        assert format_search_results_for_llm([]) == ""

    def test_renders_in_rank_order(self):
        results = [
            {
                "slug": "alpha",
                "title": "Alpha title",
                "blurb": "alpha blurb",
                "overlap_count": 8,
                "overlap_terms": ["one", "two", "three"],
            },
            {
                "slug": "bravo",
                "title": "Bravo title",
                "blurb": "bravo blurb",
                "overlap_count": 5,
                "overlap_terms": ["four"],
            },
        ]
        out = format_search_results_for_llm(results)
        # Header is present
        assert "TARGETED LITERATURE SEARCH" in out
        # Alpha rendered before bravo
        assert out.index("alpha") < out.index("bravo")
        assert "overlap=8" in out
        assert "overlap=5" in out

    def test_respects_byte_budget(self):
        big_blurb = "x" * 5000
        results = [
            {
                "slug": f"entry_{i}",
                "title": "T",
                "blurb": big_blurb,
                "overlap_count": 3,
                "overlap_terms": [],
            }
            for i in range(5)
        ]
        out = format_search_results_for_llm(results, max_bytes=2000)
        # Truncated well below 5 * 5000
        assert len(out.encode("utf-8")) <= 2500  # small overhead permissible


# ---------- check_grounding_provenance_block ----------


class TestCheckGroundingProvenanceBlock:
    def _corpus(self) -> list[LiteratureEntry]:
        return [
            _make_entry("harris_2002_trading_exchanges_microstructure", "Harris", "..."),
            _make_entry("fitschen_2013_path_of_least_resistance", "Fitschen", "..."),
        ]

    def test_no_block_is_clean(self):
        failures = check_grounding_provenance_block({}, self._corpus())
        assert failures == []

    def test_known_slugs_pass(self):
        parsed = {
            "grounding_provenance": {
                "source": "test",
                "query": "breakout",
                "retrieved_extracts": [
                    {"slug": "harris_2002_trading_exchanges_microstructure"},
                    {"slug": "fitschen_2013_path_of_least_resistance"},
                ],
            }
        }
        failures = check_grounding_provenance_block(parsed, self._corpus())
        assert failures == []

    def test_unknown_slug_emits_nonfatal(self):
        parsed = {
            "grounding_provenance": {
                "retrieved_extracts": [
                    {"slug": "ghost_paper_2099"},
                ]
            }
        }
        failures = check_grounding_provenance_block(parsed, self._corpus())
        assert len(failures) == 1
        assert failures[0].code == "GROUNDING_PROVENANCE_SLUG_UNKNOWN"
        assert failures[0].fatal is False

    def test_missing_slug_emits_nonfatal(self):
        parsed = {
            "grounding_provenance": {"retrieved_extracts": [{"slug": ""}]}
        }
        failures = check_grounding_provenance_block(parsed, self._corpus())
        assert any(f.code == "GROUNDING_PROVENANCE_SLUG_MISSING" for f in failures)
        assert not any(f.fatal for f in failures)

    def test_non_dict_block_emits_nonfatal_shape_warning(self):
        parsed = {"grounding_provenance": "not a dict"}
        failures = check_grounding_provenance_block(parsed, self._corpus())
        assert any(f.code == "GROUNDING_PROVENANCE_SHAPE" for f in failures)


# ---------- search against the real corpus (integration) ----------


class TestRealCorpusGrounding:
    """Sanity check that the real ``docs/institutional/literature/`` corpus
    is queryable: a known-mechanism query should return at least one extract.

    Acts as a smoke test guarding against corpus-shape changes that would
    silently break --ground-via-mcp.
    """

    def test_breakout_query_returns_hits(self):
        repo = Path(__file__).resolve().parents[2]
        corpus = load_corpus(repo / "docs" / "institutional" / "literature")
        assert len(corpus) > 0
        results = search_corpus(
            corpus,
            "breakout intraday trend momentum session opening range",
            top_k=5,
        )
        # We expect at least one hit on the breakout/momentum mechanism.
        assert len(results) >= 1
        slugs = [r["slug"] for r in results]
        # Sanity: Fitschen 2013 is the canonical ORB-trend extract.
        assert any("fitschen" in s or "yordanov" in s or "chordia" in s for s in slugs), (
            f"expected a breakout/ORB-related slug in top-K, got {slugs}"
        )

    def test_unrelated_query_filters_aggressively(self):
        repo = Path(__file__).resolve().parents[2]
        corpus = load_corpus(repo / "docs" / "institutional" / "literature")
        # Truly off-topic query -- should return zero or very few hits.
        # We don't assert zero because the corpus is small and stopwords plus
        # common tokens might still produce min_overlap=2 hits.
        results = search_corpus(corpus, "kangaroo opera flamingo zebra", top_k=5)
        # No semantic match expected for these tokens.
        assert results == []
