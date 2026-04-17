"""
Main AI query interface. Two-pass design:

  User question -> Pass 1: Sonnet 4.6 extracts QueryIntent via messages.parse()
                -> SQL adapter executes safely
                -> Pass 2: Opus 4.7 interprets results with adaptive thinking
                -> Returns explanation + data + warnings

Stage 3 of claude-api-modernization:
  - Model pins come from canonical `claude_client` (Sonnet 4.6 / Opus 4.7)
  - Pass 1 uses `messages.parse(output_format=QueryIntentSchema)` — no more
    manual JSON / markdown-fence recovery hacks
  - Pass 2 uses adaptive thinking + content-block filtering
  - `cache_control={"type": "ephemeral"}` on grounding prompt (~10× cheaper
    for repeat calls within the 5-minute cache window)
  - API-key check delegated to canonical `get_client()`
"""

from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field

from trading_app.ai.claude_client import (
    CLAUDE_REASONING_MODEL,
    CLAUDE_STRUCTURED_MODEL,
    get_client,
)
from trading_app.ai.corpus import get_db_stats, get_schema_definitions, load_corpus
from trading_app.ai.grounding import build_grounding_prompt, build_interpretation_prompt
from trading_app.ai.sql_adapter import (
    QueryIntent,
    QueryTemplate,
    SQLAdapter,
)
from trading_app.config import generate_strategy_warnings


class QueryParameters(BaseModel):
    """Typed, closed schema for query parameters — enables strict structured outputs.

    Every optional field corresponds to a recognized parameter name. Extra
    parameters are rejected at validation time. Claude populates only the
    subset relevant to each question.
    """

    model_config = {"extra": "forbid"}

    orb_label: str | None = None
    entry_model: str | None = None
    filter_type: str | None = None
    direction: str | None = None
    min_sample_size: int | None = None
    limit: int | None = None
    table_name: str | None = None
    instrument: str | None = None


class QueryIntentSchema(BaseModel):
    """Pydantic schema for Pass 1 structured output.

    `template=None` is a valid response — signals "no template fits this question".
    Converted to the internal `QueryIntent` dataclass for sql_adapter compatibility.
    """

    model_config = {"extra": "forbid"}

    template: str | None = None
    parameters: QueryParameters = Field(default_factory=QueryParameters)
    explanation: str = ""


@dataclass
class QueryResult:
    """Result of an AI query."""

    query: str
    intent: QueryIntent | None = None
    data: pd.DataFrame | None = None
    explanation: str = ""
    warnings: list[str] = field(default_factory=list)
    grounding_refs: list[str] = field(default_factory=list)


def _generate_warnings(df: pd.DataFrame) -> list[str]:
    """Generate auto-warnings — delegates to shared implementation in config.py."""
    return generate_strategy_warnings(df)


def _extract_text_block(response) -> str:
    """Pick the text content from a response.

    With adaptive thinking enabled, `content` becomes
    `[ThinkingBlock, ..., TextBlock]` — relying on `content[0].text` silently
    returns thinking output instead of the model's answer.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    return ""


class QueryAgent:
    """AI-powered query interface to the trading database."""

    def __init__(self, db_path: str, api_key: str | None = None):
        self.db_path = db_path

        # API-key resolution and client construction delegated to the canonical
        # module. Raises `ValueError("ANTHROPIC_API_KEY required...")` when no
        # key is available.
        self.client = get_client(api_key=api_key)

        self.corpus = load_corpus()
        self.adapter = SQLAdapter(db_path)
        self.schema_summary = get_schema_definitions(db_path)
        self.db_stats = get_db_stats(db_path)

    def query(self, question: str) -> QueryResult:
        """Main entry point. Ask a question in plain English."""
        result = QueryResult(query=question)

        intent = self._extract_intent(question)
        result.intent = intent

        if intent is None or intent.template is None:
            result.explanation = (
                "Could not determine a suitable query for this question. "
                "Try rephrasing or ask about strategies, performance, "
                "schema, or ORB size distributions."
            )
            return result

        try:
            df = self.adapter.execute(intent)
            result.data = df
        except Exception as e:
            result.explanation = f"Query execution error: {e}"
            result.warnings.append("Query failed -- check parameters.")
            return result

        result.warnings = _generate_warnings(df)
        result.grounding_refs = self._get_grounding_refs(intent)
        result.explanation = self._interpret_results(question, df)

        return result

    def _extract_intent(self, question: str) -> QueryIntent | None:
        """Pass 1: structured-output intent extraction via Sonnet 4.6.

        Uses `messages.parse()` with a Pydantic schema — Claude's response
        is validated against `QueryIntentSchema` before we see it. No manual
        JSON parsing, no markdown-fence recovery, no sampling parameters.
        """
        system_prompt = build_grounding_prompt(self.corpus, self.schema_summary)

        response = self.client.messages.parse(
            model=CLAUDE_STRUCTURED_MODEL,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
            output_format=QueryIntentSchema,
            cache_control={"type": "ephemeral"},
        )

        schema: QueryIntentSchema = response.parsed_output
        if schema.template is None:
            return None

        try:
            template_enum = QueryTemplate(schema.template)
        except ValueError:
            return None

        return QueryIntent(
            template=template_enum,
            parameters=schema.parameters.model_dump(exclude_none=True),
            explanation=schema.explanation,
        )

    def _interpret_results(self, question: str, df: pd.DataFrame) -> str:
        """Pass 2: reasoning interpretation via Opus 4.7 + adaptive thinking.

        Adaptive thinking replaces the removed `budget_tokens` parameter. The
        response `content` list may include a `ThinkingBlock` before the text
        block, so we filter explicitly rather than index `content[0]`.
        """
        if df.empty:
            return "No results found for this query."

        data_summary = df.to_string(index=False, max_rows=50)
        if len(df) > 50:
            data_summary += f"\n... ({len(df)} total rows, showing first 50)"

        prompt = build_interpretation_prompt(self.corpus, question, data_summary)

        response = self.client.messages.create(
            model=CLAUDE_REASONING_MODEL,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )

        return _extract_text_block(response)

    def _get_grounding_refs(self, intent: QueryIntent) -> list[str]:
        """Determine which canonical docs are relevant to this query."""
        refs = []
        template = intent.template

        if template in (
            QueryTemplate.STRATEGY_LOOKUP,
            QueryTemplate.PERFORMANCE_STATS,
            QueryTemplate.YEARLY_BREAKDOWN,
            QueryTemplate.TRADE_HISTORY,
        ):
            refs.append("pipeline/cost_model.py")
            refs.append("trading_app/config.py")

        if template == QueryTemplate.REGIME_COMPARE:
            refs.append("TRADING_RULES.md")

        if template in (QueryTemplate.VALIDATED_SUMMARY, QueryTemplate.CORRELATION):
            refs.append("trading_app/config.py")

        if template == QueryTemplate.ORB_SIZE_DIST:
            refs.append("TRADING_RULES.md")

        return refs
