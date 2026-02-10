"""
Main AI query interface. Two-pass design:

  User question -> Pass 1: Claude extracts QueryIntent
                -> SQL adapter executes safely
                -> Pass 2: Claude interprets results
                -> Returns explanation + data + warnings
"""

import json
import os
from dataclasses import dataclass, field

import pandas as pd

from trading_app.ai.corpus import load_corpus, get_schema_definitions, get_db_stats
from trading_app.ai.grounding import build_grounding_prompt, build_interpretation_prompt
from trading_app.ai.sql_adapter import (
    QueryIntent,
    QueryTemplate,
    SQLAdapter,
)


@dataclass
class QueryResult:
    """Result of an AI query."""

    query: str
    intent: QueryIntent | None = None
    data: pd.DataFrame | None = None
    explanation: str = ""
    warnings: list[str] = field(default_factory=list)
    grounding_refs: list[str] = field(default_factory=list)


# Auto-warning rules
_WARNING_RULES = {
    "NO_FILTER": "NO_FILTER strategies have negative expectancy -- house wins.",
    "ORB_L": "L-filter (less-than) strategies have negative expectancy -- house wins.",
}

# Classification thresholds (mirror config.py)
_CORE_MIN = 100
_REGIME_MIN = 30


def _generate_warnings(df: pd.DataFrame) -> list[str]:
    """Generate auto-warnings based on data content."""
    warnings = []
    if df is None or df.empty:
        return warnings

    # Check for dangerous filter types
    if "filter_type" in df.columns:
        filter_types = df["filter_type"].unique()
        for ft in filter_types:
            if ft == "NO_FILTER":
                warnings.append(_WARNING_RULES["NO_FILTER"])
            elif str(ft).startswith("ORB_L"):
                warnings.append(_WARNING_RULES["ORB_L"])

    # Check sample sizes
    if "sample_size" in df.columns:
        small = df[df["sample_size"] < _REGIME_MIN]
        if len(small) > 0:
            warnings.append(
                f"{len(small)} result(s) have sample_size < {_REGIME_MIN} (INVALID -- not tradeable)."
            )
        regime = df[(df["sample_size"] >= _REGIME_MIN) & (df["sample_size"] < _CORE_MIN)]
        if len(regime) > 0:
            warnings.append(
                f"{len(regime)} result(s) have sample_size {_REGIME_MIN}-{_CORE_MIN - 1} "
                f"(REGIME -- conditional overlay only, not standalone)."
            )

    return list(set(warnings))


class QueryAgent:
    """AI-powered query interface to the trading database."""

    def __init__(self, db_path: str, api_key: str | None = None):
        self.db_path = db_path
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY required. Pass api_key or set env var."
            )

        self.corpus = load_corpus()
        self.adapter = SQLAdapter(db_path)
        self.schema_summary = get_schema_definitions(db_path)
        self.db_stats = get_db_stats(db_path)

        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def query(self, question: str) -> QueryResult:
        """Main entry point. Ask a question in plain English."""
        result = QueryResult(query=question)

        # Pass 1: Extract intent
        intent = self._extract_intent(question)
        result.intent = intent

        if intent is None or intent.template is None:
            result.explanation = (
                "Could not determine a suitable query for this question. "
                "Try rephrasing or ask about strategies, performance, "
                "schema, or ORB size distributions."
            )
            return result

        # Execute query
        try:
            df = self.adapter.execute(intent)
            result.data = df
        except Exception as e:
            result.explanation = f"Query execution error: {e}"
            result.warnings.append("Query failed -- check parameters.")
            return result

        # Generate auto-warnings
        result.warnings = _generate_warnings(df)

        # Determine grounding references
        result.grounding_refs = self._get_grounding_refs(intent)

        # Pass 2: Interpret results
        result.explanation = self._interpret_results(question, df)

        return result

    def _extract_intent(self, question: str) -> QueryIntent | None:
        """Pass 1: Use Claude to extract query intent from natural language."""
        system_prompt = build_grounding_prompt(self.corpus, self.schema_summary)

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )

        text = response.content[0].text.strip()

        # Parse JSON response
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```" in text:
                json_str = text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                parsed = json.loads(json_str.strip())
            else:
                return None

        template_name = parsed.get("template")
        if template_name is None:
            return None

        try:
            template = QueryTemplate(template_name)
        except ValueError:
            return None

        return QueryIntent(
            template=template,
            parameters=parsed.get("parameters", {}),
            explanation=parsed.get("explanation", ""),
        )

    def _interpret_results(self, question: str, df: pd.DataFrame) -> str:
        """Pass 2: Use Claude to interpret query results."""
        if df.empty:
            return "No results found for this query."

        # Summarize data for the prompt
        data_summary = df.to_string(index=False, max_rows=50)
        if len(df) > 50:
            data_summary += f"\n... ({len(df)} total rows, showing first 50)"

        prompt = build_interpretation_prompt(self.corpus, question, data_summary)

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()

    def _get_grounding_refs(self, intent: QueryIntent) -> list[str]:
        """Determine which canonical docs are relevant to this query."""
        refs = []
        template = intent.template

        # Cost model is always relevant for R-multiple queries
        if template in (
            QueryTemplate.STRATEGY_LOOKUP,
            QueryTemplate.PERFORMANCE_STATS,
            QueryTemplate.YEARLY_BREAKDOWN,
            QueryTemplate.TRADE_HISTORY,
        ):
            refs.append("pipeline/cost_model.py")
            refs.append("trading_app/config.py")

        if template == QueryTemplate.REGIME_COMPARE:
            refs.append("CANONICAL_LOGIC.txt")

        if template in (QueryTemplate.VALIDATED_SUMMARY, QueryTemplate.CORRELATION):
            refs.append("trading_app/config.py")

        if template == QueryTemplate.ORB_SIZE_DIST:
            refs.append("CANONICAL_LOGIC.txt")

        return refs
