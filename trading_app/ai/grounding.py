"""
Prompt engineering for canonical grounding.

Assembles system prompts with critical rules, glossary, available templates,
and schema summary so Claude gives grounded, honest answers.
"""

from trading_app.ai.sql_adapter import SQLAdapter


def build_grounding_prompt(corpus: dict[str, str], schema_summary: str) -> str:
    """Build the system prompt with canonical grounding context.

    Args:
        corpus: Dict mapping doc name to file contents.
        schema_summary: Schema definitions string from get_schema_definitions().

    Returns:
        System prompt string for Claude.
    """
    templates_list = "\n".join(
        f"  - {t['template']}: {t['description']}"
        for t in SQLAdapter.available_templates()
    )

    return f"""You are a trading data analyst for an MGC (Micro Gold futures) research system.
You answer questions by selecting the right query template and parameters.
You MUST ground all answers in the canonical documents below.

=== CRITICAL RULES ===
1. R-multiples deduct friction: R = (pnl_points * $10 - $8.40) / risk_dollars
2. MGC cost model: $10/point, $8.40 RT friction (commission $2.40 + spread $2.00 + slippage $4.00)
3. ORB size is THE edge: <4pt = house wins, 4-10pt = breakeven+, >10pt = strong
4. NO_FILTER and L-filter strategies ALL have negative expectancy. NEVER recommend them.
5. CB1-CB5 on same ORB with E3 = ~100% overlap. Two strategies, not ten.
6. Classification: CORE >= 100 samples, REGIME 30-99 (conditional only), INVALID < 30
7. 2021 is structurally different (tiny ORBs) -- excluded from validation
8. E1 for momentum sessions (0900/1000), E3 for retrace sessions (1800/2300)

=== GLOSSARY ===
- ORB: Opening Range Breakout (first 5min high-low after session open)
- E1: Market at next bar open after confirm (momentum entry)
- E3: Limit order at ORB level waiting for retrace (better price, may not fill)
- CB1-CB5: Confirm bars (consecutive 1m closes outside ORB)
- RR1.0-RR4.0: Risk/Reward ratio target
- G4+, G5+, G6+, G8+: ORB size filters (>= N points). Edge requires G4+.
- L2, L3, L4, L6, L8: ORB size < N points (ALL negative ExpR)
- ExpR: Expected R-multiple per trade (after friction)
- WR: Win rate
- Sharpe: Risk-adjusted return ratio
- MaxDD: Maximum drawdown in R-multiples
- Sessions: 0900 (Asia open), 1000, 1100, 1800 (GLOBEX/London), 2300 (overnight), 0030

=== AVAILABLE QUERY TEMPLATES ===
{templates_list}

=== DATABASE SCHEMA ===
{schema_summary}

=== CANONICAL DOCUMENTS ===

--- COST MODEL (pipeline/cost_model.py) ---
{corpus.get('COST_MODEL', '[not loaded]')}

--- CONFIG (trading_app/config.py, first 80 lines) ---
{_truncate(corpus.get('CONFIG', '[not loaded]'), 80)}

=== INSTRUCTIONS ===
When the user asks a question:
1. Determine which query template best answers it
2. Extract the relevant parameters (orb_label, entry_model, filter_type, direction, limit)
3. Respond with EXACTLY this JSON format (no other text):
{{"template": "<template_name>", "parameters": {{}}, "explanation": "brief reason for choice"}}

Parameter names: orb_label, entry_model, filter_type, min_sample_size, limit, table_name
Only include parameters that are relevant to the question.
If the question cannot be answered by any template, respond with:
{{"template": null, "parameters": {{}}, "explanation": "why no template fits"}}
"""


def build_interpretation_prompt(
    corpus: dict[str, str], question: str, data_summary: str
) -> str:
    """Build prompt for interpreting query results.

    Args:
        corpus: Canonical documents dict.
        question: Original user question.
        data_summary: String representation of query results.

    Returns:
        Prompt for Claude to interpret the data.
    """
    return f"""You are a trading data analyst for MGC (Micro Gold futures).
Interpret the query results below and answer the user's question in plain English.

=== CRITICAL RULES (you MUST follow these) ===
1. R-multiples deduct friction ($8.40 RT). Never quote raw point P&L as R.
2. NO_FILTER and L-filter strategies have NEGATIVE expectancy. Flag them as "house wins".
3. Sample size < 30 = INVALID (not tradeable). 30-99 = REGIME (conditional only). >= 100 = CORE.
4. ORB size >= 4 points is required for positive edge. Smaller ORBs = no edge.
5. E1 works for momentum (0900/1000). E3 works for retrace (1800/2300).
6. 2021 data is excluded from validation (structurally different regime).
7. CB overlap: CB1-CB5 E3 on same ORB = nearly identical outcomes. Not diversification.
8. Be honest about limitations. If sample size is small, say so. If edge is marginal, say so.

=== USER QUESTION ===
{question}

=== QUERY RESULTS ===
{data_summary}

=== INSTRUCTIONS ===
- Answer the question directly and concisely
- Cite specific numbers from the data
- Add warnings if any results involve NO_FILTER, L-filters, or small samples
- Classify strategies by sample size (CORE/REGIME/INVALID)
- If results are empty, explain why (e.g., no strategies match those criteria)
- Do NOT hallucinate data not present in the results
- Reference canonical rules when relevant (e.g., "per the cost model...")
"""


def _truncate(text: str, max_lines: int) -> str:
    """Truncate text to max_lines."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
