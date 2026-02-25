"""
MCP server for the Gold Trading Database.

Exposes 4 read-only tools via stdio (fastmcp):
  - list_available_queries: discover what query templates exist
  - query_trading_db: run a pre-approved SQL template
  - get_strategy_fitness: FIT/WATCH/DECAY/STALE status
  - get_canonical_context: load grounding docs for AI context

Usage:
    claude mcp add gold-db --scope project -- python trading_app/mcp_server.py
"""

import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

# Ensure project root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from trading_app.ai.sql_adapter import (
    QueryIntent,
    QueryTemplate,
    SQLAdapter,
)
from trading_app.strategy_fitness import compute_fitness, compute_portfolio_fitness
from trading_app.config import CORE_MIN_SAMPLES, REGIME_MIN_SAMPLES, generate_strategy_warnings

DB_PATH = str(GOLD_DB_PATH)

# Hard server-side row cap (defense-in-depth over sql_adapter's 1000)
MAX_MCP_ROWS = 5000

# Only these parameter keys are forwarded to SQLAdapter. Anything else is rejected.
_ALLOWED_PARAMS = {"orb_label", "entry_model", "filter_type", "min_sample_size", "limit", "instrument", "rr_target", "confirm_bars"}

# ---------------------------------------------------------------------------
# Warnings — thresholds imported from config.py (single source of truth)
# ---------------------------------------------------------------------------

_CORE_MIN = CORE_MIN_SAMPLES
_REGIME_MIN = REGIME_MIN_SAMPLES

def _generate_warnings(df) -> list[str]:
    """Generate auto-warnings — delegates to shared implementation in config.py."""
    return generate_strategy_warnings(df)

# ---------------------------------------------------------------------------
# Core logic (plain functions, testable without MCP)
# ---------------------------------------------------------------------------

def _list_available_queries() -> list[dict[str, str]]:
    """List all pre-approved query templates and their descriptions."""
    return SQLAdapter.available_templates()

def _query_trading_db(
    template: str,
    orb_label: str | None = None,
    entry_model: str | None = None,
    filter_type: str | None = None,
    min_sample_size: int | None = None,
    instrument: str = "MGC",
    limit: int = 50,
    rr_target: float | None = None,
    confirm_bars: int | None = None,
) -> dict:
    """Run a pre-approved SQL query against the trading database.

    Guardrails:
    - Template must be a valid QueryTemplate enum value (no raw SQL).
    - Only allowlisted parameter keys are forwarded (rejects unknown params).
    - Limit is server-side capped at MAX_MCP_ROWS.
    - SQLAdapter opens DuckDB with read_only=True (no writes possible).
    """
    # G1: Template must be a valid enum member
    try:
        qt = QueryTemplate(template)
    except ValueError:
        valid = [t.value for t in QueryTemplate]
        return {"error": f"Unknown template '{template}'. Valid: {valid}"}

    # G2: Build params from allowlisted keys only
    params = {}
    if orb_label is not None:
        params["orb_label"] = orb_label
    if entry_model is not None:
        params["entry_model"] = entry_model
    if filter_type is not None:
        params["filter_type"] = filter_type
    if min_sample_size is not None:
        params["min_sample_size"] = min_sample_size
    if instrument is not None:
        params["instrument"] = instrument
    if rr_target is not None:
        params["rr_target"] = rr_target
    if confirm_bars is not None:
        params["confirm_bars"] = confirm_bars

    # G3: Server-side row cap
    params["limit"] = min(int(limit), MAX_MCP_ROWS)

    intent = QueryIntent(template=qt, parameters=params)

    # G4: SQLAdapter enforces read_only=True on every connection
    adapter = SQLAdapter(DB_PATH)
    try:
        df = adapter.execute(intent)
    except Exception as e:
        return {"error": str(e)}

    # G5: Truncate result if it somehow exceeds cap
    if len(df) > MAX_MCP_ROWS:
        df = df.head(MAX_MCP_ROWS)

    warnings = _generate_warnings(df)

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "row_count": len(df),
        "warnings": warnings,
    }

def _get_strategy_fitness(
    strategy_id: str | None = None,
    instrument: str = "MGC",
    rolling_months: int = 18,
    summary_only: bool = False,
) -> dict:
    """Get fitness status for strategies.

    When summary_only=True, returns only status counts and non-FIT strategies
    (avoids 150K+ character payloads that blow up MCP context).
    """
    db_path = GOLD_DB_PATH
    as_of = date.today()

    if strategy_id:
        try:
            score = compute_fitness(
                strategy_id, db_path=db_path,
                as_of_date=as_of, rolling_months=rolling_months,
            )
        except ValueError as e:
            return {"error": str(e)}
        return asdict(score)

    report = compute_portfolio_fitness(
        db_path=db_path, instrument=instrument,
        as_of_date=as_of, rolling_months=rolling_months,
    )

    if summary_only:
        # Return only summary + non-FIT strategies (compact)
        non_fit = [
            asdict(s) for s in report.scores
            if s.fitness_status != "FIT"
        ]
        return {
            "as_of_date": report.as_of_date.isoformat(),
            "summary": report.summary,
            "strategy_count": len(report.scores),
            "non_fit_strategies": non_fit,
            "non_fit_count": len(non_fit),
        }

    return {
        "as_of_date": report.as_of_date.isoformat(),
        "summary": report.summary,
        "strategy_count": len(report.scores),
        "scores": [asdict(s) for s in report.scores],
    }

def _get_canonical_context() -> dict:
    """Load canonical grounding documents for AI context."""
    from trading_app.ai.corpus import load_corpus
    return load_corpus()

# ---------------------------------------------------------------------------
# MCP server (thin wrappers around core logic)
# ---------------------------------------------------------------------------

def _build_server():
    """Build and return the FastMCP server instance."""
    from fastmcp import FastMCP

    mcp = FastMCP(
        "gold-db",
        instructions=(
            "Gold futures trading database with 4 instruments: MGC (10yr), MNQ (2yr), MES (5yr), M2K (5yr). "
            "735 FDR-validated ORB breakout strategies. "
            "WORKFLOW: (1) list_available_queries to discover templates, "
            "(2) query_trading_db for strategy lookups/comparisons/raw outcomes, "
            "(3) get_strategy_fitness for rolling regime assessment (FIT/WATCH/DECAY/STALE), "
            "(4) get_canonical_context to load trading rules before complex analysis. "
            "Use instrument parameter to filter (default MGC). "
            "For recent performance, use get_strategy_fitness with rolling_months. "
            "NEVER query orb_outcomes directly without filters — use templates."
        ),
    )

    @mcp.tool()
    def list_available_queries() -> list[dict[str, str]]:
        """List all pre-approved query templates and their descriptions.

        Call this first to discover what queries are available before
        calling query_trading_db.
        """
        return _list_available_queries()

    @mcp.tool()
    def query_trading_db(
        template: str,
        orb_label: str | None = None,
        entry_model: str | None = None,
        filter_type: str | None = None,
        min_sample_size: int | None = None,
        instrument: str = "MGC",
        limit: int = 50,
        rr_target: float | None = None,
        confirm_bars: int | None = None,
    ) -> dict:
        """Run a pre-approved SQL query against the trading database.

        Args:
            template: Query template name (use list_available_queries to see options).
                      Validated strategies: strategy_lookup, performance_stats,
                      validated_summary, yearly_breakdown, trade_history.
                      Raw outcomes: outcomes_stats, entry_model_compare,
                      dow_breakdown, dst_split, filter_compare.
                      Infrastructure: schema_info, table_counts, orb_size_dist,
                      regime_compare, correlation, double_break_stats,
                      gap_analysis, rolling_stability.
            orb_label: ORB session filter. E.g.: CME_REOPEN, TOKYO_OPEN, LONDON_METALS, NYSE_OPEN.
            entry_model: Entry model filter. One of: E0, E1, E3.
            filter_type: ORB size filter. Examples: ORB_G4, ORB_G6, NO_FILTER.
            min_sample_size: Minimum number of trades.
            instrument: Instrument filter (default MGC). One of: MGC, MNQ, MES, M2K.
            limit: Max rows to return (default 50, server cap 5000).
            rr_target: Risk-reward target (raw outcomes templates). One of: 1.0, 1.5, 2.0, 2.5, 3.0.
            confirm_bars: Confirmation bars (raw outcomes templates). One of: 1, 2, 3.

        Returns:
            Dict with 'columns', 'rows', 'row_count', and 'warnings' keys.
        """
        return _query_trading_db(
            template=template, orb_label=orb_label, entry_model=entry_model,
            filter_type=filter_type, min_sample_size=min_sample_size,
            instrument=instrument, limit=limit,
            rr_target=rr_target, confirm_bars=confirm_bars,
        )

    @mcp.tool()
    def get_strategy_fitness(
        strategy_id: str | None = None,
        instrument: str = "MGC",
        rolling_months: int = 18,
        summary_only: bool = False,
    ) -> dict:
        """Get fitness status (FIT/WATCH/DECAY/STALE) for strategies.

        Assesses whether strategies are still working in the current regime
        using a 3-layer framework: structural edge, rolling regime, decay monitoring.

        Args:
            strategy_id: Single strategy ID (e.g. 'MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4').
                         If None, returns fitness for ALL validated strategies.
            instrument: Instrument symbol (default 'MGC').
            rolling_months: Rolling window in months (default 18).
            summary_only: If True, return only summary + non-FIT strategies (compact).

        Returns:
            Dict with fitness scores and summary counts.
        """
        return _get_strategy_fitness(
            strategy_id=strategy_id, instrument=instrument,
            rolling_months=rolling_months, summary_only=summary_only,
        )

    @mcp.tool()
    def get_canonical_context() -> dict:
        """Load canonical grounding documents (cost model, logic rules, config).

        Returns the full text of critical project documents for AI context.
        Use this to ground your analysis in the project's actual trading rules.
        """
        return _get_canonical_context()

    return mcp

if __name__ == "__main__":
    mcp = _build_server()
    mcp.run()
