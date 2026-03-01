"""Tests for trading_app.ai.grounding."""

import pytest

from trading_app.ai.grounding import (
    build_grounding_prompt,
    build_interpretation_prompt,
)


@pytest.fixture
def mock_corpus():
    return {
        "CANONICAL_LOGIC": "R-multiple = (pnl * 10 - 5.74) / risk",
        "TRADE_MANAGEMENT_RULES": "30-min kill rule for 1800 E3",
        "CONFIG": 'ENTRY_MODELS = ["E1", "E3"]\nCORE_MIN_SAMPLES = 100',
        "COST_MODEL": "total_friction = 5.74\npoint_value = 10.0",
    }


@pytest.fixture
def mock_schema():
    return "bars_1m:\n  ts_utc (TIMESTAMPTZ)\n  symbol (VARCHAR)\n"


class TestBuildGroundingPrompt:
    def test_contains_critical_rules(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "5.74" in prompt
        assert "R-multiple" in prompt.lower() or "r-multiple" in prompt.lower()

    def test_contains_glossary(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "E1" in prompt
        assert "E3" in prompt
        assert "ORB" in prompt
        assert "CORE" in prompt

    def test_contains_template_list(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "strategy_lookup" in prompt
        assert "performance_stats" in prompt
        assert "table_counts" in prompt

    def test_contains_schema(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "bars_1m" in prompt

    def test_contains_no_filter_warning(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "NO_FILTER" in prompt
        assert "negative" in prompt.lower()

    def test_json_format_instruction(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert '"template"' in prompt
        assert '"parameters"' in prompt


class TestBuildInterpretationPrompt:
    def test_contains_question(self, mock_corpus):
        prompt = build_interpretation_prompt(
            mock_corpus, "What is the best strategy?", "some data"
        )
        assert "What is the best strategy?" in prompt

    def test_contains_data(self, mock_corpus):
        prompt = build_interpretation_prompt(
            mock_corpus, "question", "orb_label=0900 win_rate=0.42"
        )
        assert "orb_label=0900" in prompt

    def test_contains_honesty_rules(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "q", "d")
        assert "honest" in prompt.lower() or "limitations" in prompt.lower()
        assert "INVALID" in prompt
        assert "REGIME" in prompt
