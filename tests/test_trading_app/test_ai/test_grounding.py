"""Tests for trading_app.ai.grounding — multi-instrument, canonical-source-grounded.

Post-Stage-2 of claude-api-modernization: grounding prompts are built from
canonical sources (ACTIVE_ORB_INSTRUMENTS, SESSION_CATALOG, COST_SPECS,
CORE_MIN_SAMPLES/REGIME_MIN_SAMPLES), not hardcoded MGC-only rules.
"""

import pytest

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS
from pipeline.dst import SESSION_CATALOG
from trading_app.ai.grounding import (
    build_grounding_prompt,
    build_interpretation_prompt,
)
from trading_app.config import CORE_MIN_SAMPLES, REGIME_MIN_SAMPLES


@pytest.fixture
def mock_corpus():
    """Mock of the expanded 8-entry corpus returned by load_corpus()."""
    return {
        "TRADING_RULES": "R = (pnl * point_value - friction) / risk",
        "TRADE_MANAGEMENT_RULES": "30-min kill rule for retrace entries",
        "CONFIG": 'ENTRY_MODELS = ["E1", "E2"]\nCORE_MIN_SAMPLES = 100',
        "COST_MODEL": "friction varies per instrument",
        "RESEARCH_RULES": "FDR correction required for any discovery claim.",
        "CLAUDE_MD": "Canonical sources only. No hardcoded lists.",
        "PRE_REGISTERED_CRITERIA": "12 locked criteria every validated strategy must meet.",
        "MECHANISM_PRIORS": "ORB edge priors and signal-to-role mapping.",
    }


@pytest.fixture
def mock_schema():
    return "bars_1m:\n  ts_utc (TIMESTAMPTZ)\n  symbol (VARCHAR)\n"


class TestBuildGroundingPrompt:
    """The grounding prompt is the AI's worldview. It MUST reflect live project state."""

    def test_mentions_all_active_instruments(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        for inst in ACTIVE_ORB_INSTRUMENTS:
            assert inst in prompt, f"{inst} missing from grounding prompt"

    def test_no_mgc_only_framing(self, mock_corpus, mock_schema):
        """The AI must not be led to believe this is an MGC-only research system."""
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        lower = prompt.lower()
        assert "mgc-only" not in lower
        assert "mgc research system" not in lower
        assert "(micro gold futures) research system" not in lower

    def test_lists_all_sessions(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        for label in SESSION_CATALOG:
            assert label in prompt, f"Session {label} missing from grounding prompt"

    def test_includes_session_event_context(self, mock_corpus, mock_schema):
        """SESSION_CATALOG carries event descriptions (e.g., 'COMEX gold settlement').

        Including them gives the AI DST-aware context instead of bare labels.
        """
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        # Sample a few distinctive event strings
        assert "COMEX gold settlement" in prompt
        assert "NYSE cash open" in prompt
        assert "Tokyo Stock Exchange open" in prompt

    def test_uses_canonical_classification_thresholds(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert str(CORE_MIN_SAMPLES) in prompt
        assert str(REGIME_MIN_SAMPLES) in prompt

    def test_costs_cover_all_active_instruments(self, mock_corpus, mock_schema):
        """Every active instrument's friction must appear in the prompt."""
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        for inst in ACTIVE_ORB_INSTRUMENTS:
            assert inst in prompt, f"{inst} not mentioned in cost section"
            spec = COST_SPECS[inst]
            friction = spec.commission_rt + spec.spread_doubled + spec.slippage
            # Friction values differ per instrument; any common 2dp formatting counts
            assert f"{friction:.2f}" in prompt, f"{inst} friction {friction:.2f} missing from prompt"

    def test_contains_template_list(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "strategy_lookup" in prompt
        assert "performance_stats" in prompt
        assert "table_counts" in prompt

    def test_contains_schema(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "bars_1m" in prompt

    def test_contains_entry_model_glossary(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "ORB" in prompt
        assert "E1" in prompt
        assert "E2" in prompt

    def test_no_filter_warning(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert "NO_FILTER" in prompt
        assert "negative" in prompt.lower()

    def test_json_format_instruction(self, mock_corpus, mock_schema):
        prompt = build_grounding_prompt(mock_corpus, mock_schema)
        assert '"template"' in prompt
        assert '"parameters"' in prompt


class TestBuildInterpretationPrompt:
    def test_contains_question(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "What is the best strategy?", "some data")
        assert "What is the best strategy?" in prompt

    def test_contains_data(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "question", "orb_label=0900 win_rate=0.42")
        assert "orb_label=0900" in prompt

    def test_contains_honesty_rules(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "q", "d")
        assert "honest" in prompt.lower() or "limitations" in prompt.lower()
        assert "INVALID" in prompt
        assert "REGIME" in prompt

    def test_mentions_all_active_instruments(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "q", "d")
        for inst in ACTIVE_ORB_INSTRUMENTS:
            assert inst in prompt, f"{inst} missing from interpretation prompt"

    def test_no_mgc_only_framing(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "q", "d")
        lower = prompt.lower()
        assert "mgc-only" not in lower
        assert "mgc research system" not in lower
        assert "(micro gold futures) research system" not in lower

    def test_uses_canonical_thresholds(self, mock_corpus):
        prompt = build_interpretation_prompt(mock_corpus, "q", "d")
        assert str(CORE_MIN_SAMPLES) in prompt
        assert str(REGIME_MIN_SAMPLES) in prompt
