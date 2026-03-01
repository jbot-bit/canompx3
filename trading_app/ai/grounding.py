"""
Prompt engineering for canonical grounding.

Assembles system prompts with critical rules, glossary, available templates,
and schema summary so Claude gives grounded, honest answers.
"""

from pipeline.cost_model import COST_SPECS
from trading_app.ai.sql_adapter import SQLAdapter

# Derive MGC cost values from canonical source
_mgc = COST_SPECS["MGC"]
_mgc_friction = _mgc.commission_rt + _mgc.spread_doubled + _mgc.slippage
_mgc_pv = _mgc.point_value


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
1. R-multiples deduct friction: R = (pnl_points * ${_mgc_pv:.0f} - ${_mgc_friction:.2f}) / risk_dollars
2. MGC cost model: ${_mgc_pv:.0f}/point, ${_mgc_friction:.2f} RT friction (commission ${_mgc.commission_rt:.2f} + spread ${_mgc.spread_doubled:.2f} + slippage ${_mgc.slippage:.2f})
3. ORB size is THE edge: <4pt = house wins, 4-10pt = breakeven+, >10pt = strong
4. NO_FILTER and L-filter strategies ALL have negative expectancy. NEVER recommend them.
5. CB1-CB5 on same ORB with E3 = ~100% overlap. Two strategies, not ten.
6. Classification: CORE >= 100 samples, REGIME 30-99 (conditional only), INVALID < 30
7. 2021 is structurally different (tiny ORBs) -- excluded from validation
8. E1 for momentum sessions (CME_REOPEN/TOKYO_OPEN), E3 for retrace sessions (LONDON_METALS/US_DATA_830)

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
- Sessions: CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE, BRISBANE_0925

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
1. R-multiples deduct friction (${_mgc_friction:.2f} RT). Never quote raw point P&L as R.
2. NO_FILTER and L-filter strategies have NEGATIVE expectancy. Flag them as "house wins".
3. Sample size < 30 = INVALID (not tradeable). 30-99 = REGIME (conditional only). >= 100 = CORE.
4. ORB size >= 4 points is required for positive edge. Smaller ORBs = no edge.
5. E1 works for momentum (CME_REOPEN/TOKYO_OPEN). E3 works for retrace (LONDON_METALS/US_DATA_830).
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
