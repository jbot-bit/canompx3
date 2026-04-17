"""Prompt engineering for canonical multi-instrument grounding.

Assembles system prompts Claude uses to answer quant questions. Every
grounding value comes from a canonical project source — no hardcoded
instrument lists, session names, friction numbers, or classification
thresholds. When the project state changes (new instrument, session
retirement, threshold tweak), the prompt updates automatically.

Canonical sources consumed:
  - pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS — which instruments
    the project actually trades
  - pipeline.dst.SESSION_CATALOG — sessions + DST-aware event descriptions
  - pipeline.cost_model.COST_SPECS — per-instrument friction + point value
  - trading_app.config.ENTRY_MODELS — active entry models (E1, E2, ...)
  - trading_app.config.CORE_MIN_SAMPLES / REGIME_MIN_SAMPLES — classification

Stage 2 of claude-api-modernization. See docs/runtime/stages/claude-api-modernization.md.
"""

from __future__ import annotations

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS
from pipeline.dst import SESSION_CATALOG
from trading_app.ai.sql_adapter import SQLAdapter
from trading_app.config import CORE_MIN_SAMPLES, ENTRY_MODELS, REGIME_MIN_SAMPLES


def _build_cost_section() -> str:
    """Per-instrument cost lines from COST_SPECS — sorted by instrument name."""
    lines = []
    for inst in ACTIVE_ORB_INSTRUMENTS:
        spec = COST_SPECS[inst]
        friction = spec.commission_rt + spec.spread_doubled + spec.slippage
        lines.append(
            f"  - {inst}: ${spec.point_value:.0f}/point, ${friction:.2f} RT friction "
            f"(commission ${spec.commission_rt:.2f} + spread ${spec.spread_doubled:.2f} "
            f"+ slippage ${spec.slippage:.2f})"
        )
    return "\n".join(lines)


def _build_sessions_section() -> str:
    """Session labels + DST-aware event context from SESSION_CATALOG."""
    lines = []
    for label, entry in SESSION_CATALOG.items():
        event = entry.get("event", "")
        lines.append(f"  - {label}: {event}" if event else f"  - {label}")
    return "\n".join(lines)


def _build_classification_rule() -> str:
    """Classification thresholds from canonical config."""
    return (
        f"CORE >= {CORE_MIN_SAMPLES} samples, "
        f"REGIME {REGIME_MIN_SAMPLES}-{CORE_MIN_SAMPLES - 1} (conditional only), "
        f"INVALID < {REGIME_MIN_SAMPLES}"
    )


def build_grounding_prompt(corpus: dict[str, str], schema_summary: str) -> str:
    """Build the system prompt with canonical grounding context.

    Args:
        corpus: Dict mapping doc name to file contents (from load_corpus).
        schema_summary: Schema definitions from get_schema_definitions().

    Returns:
        System prompt string for Claude.
    """
    templates_list = "\n".join(
        f"  - {t['template']}: {t['description']}"
        for t in SQLAdapter.available_templates()
    )
    instruments_str = ", ".join(ACTIVE_ORB_INSTRUMENTS)
    entry_models_str = ", ".join(ENTRY_MODELS)
    classification = _build_classification_rule()

    return f"""You are a trading data analyst for a multi-instrument ORB breakout research system.
Active instruments: {instruments_str}.
You answer questions by selecting the right query template and parameters.
You MUST ground all answers in the canonical documents below.

=== CRITICAL RULES ===
1. R-multiples deduct friction. R = (pnl_points * point_value - friction) / risk_dollars (per-instrument; see cost section).
2. Classification: {classification}
3. ORB size filters (G4+/G5+/G6+/G8+) gate edge — small ORB sessions have negative expectancy.
4. NO_FILTER and L-filter strategies ALL have negative expectancy. NEVER recommend them.
5. CB1-CB5 on same ORB with E3 = ~100% overlap. Count as one strategy, not many.
6. Entry-model routing: E1 for momentum sessions, E2 stop-market is industry-standard (Crabel, always CB1), E3 retrace (soft-retired Feb 2026).
7. All sessions are DST-dynamic (resolvers in pipeline.dst). Brisbane local time varies per trading day.

=== COST MODEL (per active instrument) ===
{_build_cost_section()}

=== GLOSSARY ===
- ORB: Opening Range Breakout (first N-minute high-low window after session open; N ∈ {{5, 15, 30}})
- Entry models ({entry_models_str}): E1 = market at next bar after confirm, E2 = stop-market at ORB level + 1-tick slip, E3 = limit on retrace
- CB1-CB5: Confirm bars (consecutive 1-minute closes outside ORB)
- RR1.0-RR4.0: Risk/Reward ratio target
- G4+, G5+, G6+, G8+: ORB size filters (>= N points). Edge requires G4+.
- L2, L3, L4, L6, L8: ORB size < N points (ALL negative ExpR)
- ExpR: Expected R-multiple per trade (after friction)
- WR: Win rate. Sharpe: Risk-adjusted return ratio. MaxDD: Max drawdown in R-multiples.

=== SESSIONS (DST-aware, resolver per trading day — times shift summer/winter) ===
{_build_sessions_section()}

=== AVAILABLE QUERY TEMPLATES ===
{templates_list}

=== DATABASE SCHEMA ===
{schema_summary}

=== CANONICAL DOCUMENTS ===

--- COST MODEL (pipeline/cost_model.py) ---
{corpus.get("COST_MODEL", "[not loaded]")}

--- CONFIG (trading_app/config.py, first 80 lines) ---
{_truncate(corpus.get("CONFIG", "[not loaded]"), 80)}

=== INSTRUCTIONS ===
When the user asks a question:
1. Determine which query template best answers it
2. Extract the relevant parameters (orb_label, entry_model, filter_type, direction, limit)
3. Respond with EXACTLY this JSON format (no other text):
{{"template": "<template_name>", "parameters": {{}}, "explanation": "brief reason for choice"}}

Parameter names: orb_label, entry_model, filter_type, min_sample_size, limit, table_name, instrument
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
    instruments_str = ", ".join(ACTIVE_ORB_INSTRUMENTS)
    classification = _build_classification_rule()

    return f"""You are a trading data analyst for a multi-instrument ORB breakout research system.
Active instruments: {instruments_str}.
Interpret the query results below and answer the user's question in plain English.

=== CRITICAL RULES (you MUST follow these) ===
1. R-multiples deduct friction. Per-instrument friction differs — never quote raw point P&L as R.
2. NO_FILTER and L-filter strategies have NEGATIVE expectancy. Flag them as "house wins".
3. Sample size classification: {classification} ("INVALID" = not tradeable, "REGIME" = conditional only).
4. ORB size >= 4 points is required for positive edge. Smaller ORBs = no edge.
5. Entry models: E1 for momentum, E2 stop-market (Crabel industry standard), E3 retrace (soft-retired).
6. CB overlap: CB1-CB5 E3 on same ORB = nearly identical outcomes. Not diversification.
7. Be honest about limitations. If sample size is small or edge is marginal, say so.

=== USER QUESTION ===
{question}

=== QUERY RESULTS ===
{data_summary}

=== INSTRUCTIONS ===
- Answer the question directly and concisely
- Cite specific numbers from the data
- Add warnings if any results involve NO_FILTER, L-filters, or small samples
- Classify strategies by sample size: INVALID / REGIME / CORE
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
