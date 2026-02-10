"""
Safe, read-only query executor with pre-approved SQL templates.

AI picks template + parameters, never writes raw SQL.
All queries use duckdb.connect(read_only=True) and parameterized queries.
"""

from dataclasses import dataclass, field
from enum import Enum

import duckdb
import pandas as pd


class QueryTemplate(str, Enum):
    """Pre-approved query templates."""

    STRATEGY_LOOKUP = "strategy_lookup"
    PERFORMANCE_STATS = "performance_stats"
    VALIDATED_SUMMARY = "validated_summary"
    YEARLY_BREAKDOWN = "yearly_breakdown"
    TRADE_HISTORY = "trade_history"
    SCHEMA_INFO = "schema_info"
    TABLE_COUNTS = "table_counts"
    ORB_SIZE_DIST = "orb_size_dist"
    REGIME_COMPARE = "regime_compare"
    CORRELATION = "correlation"


@dataclass
class QueryIntent:
    """Parsed intent from user question."""

    template: QueryTemplate
    parameters: dict = field(default_factory=dict)
    explanation: str = ""


# Maximum rows returned by any query
MAX_RESULT_ROWS = 1000

# Valid ORB labels for parameter validation
VALID_ORB_LABELS = {"0900", "1000", "1100", "1800", "2300", "0030"}

# Valid entry models
VALID_ENTRY_MODELS = {"E1", "E2", "E3"}

# Valid filter types (subset for validation)
VALID_FILTER_PREFIXES = {"NO_FILTER", "ORB_G", "ORB_L", "VOL_"}

def _validate_orb_label(label: str) -> str:
    """Validate and return ORB label."""
    if label not in VALID_ORB_LABELS:
        raise ValueError(f"Invalid ORB label '{label}'. Valid: {sorted(VALID_ORB_LABELS)}")
    return label


def _validate_entry_model(em: str) -> str:
    """Validate and return entry model."""
    if em not in VALID_ENTRY_MODELS:
        raise ValueError(f"Invalid entry model '{em}'. Valid: {sorted(VALID_ENTRY_MODELS)}")
    return em


def _validate_filter_type(ft: str) -> str:
    """Validate filter type string."""
    if not any(ft.startswith(p) for p in VALID_FILTER_PREFIXES):
        raise ValueError(f"Invalid filter_type '{ft}'. Must start with one of: {VALID_FILTER_PREFIXES}")
    return ft


# SQL templates -- all SELECT-only, parameterized
_TEMPLATES = {
    QueryTemplate.STRATEGY_LOOKUP: """
        SELECT orb_label, entry_model, filter_type,
               rr_target, confirm_bars, sample_size, win_rate,
               expectancy_r, sharpe_ratio, max_drawdown_r
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        {where_clauses}
        ORDER BY expectancy_r DESC
        LIMIT ?
    """,
    QueryTemplate.PERFORMANCE_STATS: """
        SELECT orb_label, entry_model, filter_type,
               COUNT(*) as total_strategies,
               AVG(win_rate) as avg_win_rate,
               AVG(expectancy_r) as avg_expectancy_r,
               AVG(sharpe_ratio) as avg_sharpe,
               MIN(max_drawdown_r) as worst_drawdown
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        {where_clauses}
        GROUP BY orb_label, entry_model, filter_type
        ORDER BY avg_expectancy_r DESC
    """,
    QueryTemplate.VALIDATED_SUMMARY: """
        SELECT orb_label, COUNT(*) as count,
               AVG(win_rate) as avg_wr,
               AVG(expectancy_r) as avg_expr,
               AVG(sharpe_ratio) as avg_sharpe
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        GROUP BY orb_label
        ORDER BY orb_label
    """,
    QueryTemplate.YEARLY_BREAKDOWN: """
        SELECT orb_label, entry_model, filter_type,
               rr_target, confirm_bars, sample_size, win_rate,
               expectancy_r, sharpe_ratio, max_drawdown_r,
               yearly_results
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        {where_clauses}
        ORDER BY expectancy_r DESC
        LIMIT ?
    """,
    QueryTemplate.TRADE_HISTORY: """
        SELECT trading_day, orb_label, entry_model,
               rr_target, confirm_bars, entry_price, stop_price,
               target_price, pnl_r, outcome
        FROM orb_outcomes
        WHERE symbol = 'MGC'
        {where_clauses}
        ORDER BY trading_day DESC
        LIMIT ?
    """,
    QueryTemplate.SCHEMA_INFO: """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        {where_clauses}
        ORDER BY table_name, ordinal_position
    """,
    QueryTemplate.TABLE_COUNTS: """
        SELECT '{table_name}' as table_name, COUNT(*) as row_count
        FROM {table_name}
    """,
    QueryTemplate.ORB_SIZE_DIST: """
        SELECT
            CASE
                WHEN orb_{orb_label}_size < 2 THEN '< 2 pts'
                WHEN orb_{orb_label}_size < 4 THEN '2-4 pts'
                WHEN orb_{orb_label}_size < 6 THEN '4-6 pts'
                WHEN orb_{orb_label}_size < 8 THEN '6-8 pts'
                WHEN orb_{orb_label}_size < 10 THEN '8-10 pts'
                ELSE '10+ pts'
            END as size_bucket,
            COUNT(*) as days,
            AVG(orb_{orb_label}_size) as avg_size
        FROM daily_features
        WHERE orb_{orb_label}_size IS NOT NULL
        AND orb_minutes = 5
        GROUP BY size_bucket
        ORDER BY MIN(orb_{orb_label}_size)
    """,
    QueryTemplate.REGIME_COMPARE: """
        SELECT r.orb_label, r.entry_model, r.filter_type,
               r.rr_target, r.confirm_bars,
               r.sample_size as regime_n, r.expectancy_r as regime_expr,
               r.sharpe_ratio as regime_sharpe,
               v.sample_size as full_n, v.expectancy_r as full_expr,
               v.sharpe_ratio as full_sharpe
        FROM regime_strategies r
        LEFT JOIN validated_setups v
          ON r.orb_label = v.orb_label
          AND r.entry_model = v.entry_model
          AND r.filter_type = v.filter_type
          AND r.rr_target = v.rr_target
          AND r.confirm_bars = v.confirm_bars
        WHERE LOWER(r.validation_status) = 'validated'
        {where_clauses}
        ORDER BY r.expectancy_r DESC
        LIMIT ?
    """,
    QueryTemplate.CORRELATION: """
        SELECT orb_label, entry_model, filter_type,
               rr_target, confirm_bars, sample_size,
               expectancy_r, sharpe_ratio
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        ORDER BY sharpe_ratio DESC
        LIMIT ?
    """,
}


class SQLAdapter:
    """Safe, read-only query executor."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, intent: QueryIntent) -> pd.DataFrame:
        """Execute a query from a validated intent. Returns DataFrame."""
        template = intent.template
        params = intent.parameters

        if template == QueryTemplate.TABLE_COUNTS:
            return self._execute_table_counts()

        if template == QueryTemplate.ORB_SIZE_DIST:
            return self._execute_orb_size_dist(params)

        sql, bind_params = self._build_query(template, params)

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql, bind_params).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    def _execute_table_counts(self) -> pd.DataFrame:
        """Get row counts for all tables."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()

            rows = []
            for (table_name,) in tables:
                count = con.execute(
                    f'SELECT COUNT(*) FROM "{table_name}"'
                ).fetchone()[0]
                rows.append({"table_name": table_name, "row_count": count})
            return pd.DataFrame(rows)
        finally:
            con.close()

    def _execute_orb_size_dist(self, params: dict) -> pd.DataFrame:
        """Execute ORB size distribution query with validated label."""
        orb_label = _validate_orb_label(params.get("orb_label", "0900"))

        # Build SQL with validated orb_label embedded (safe -- validated against allowlist)
        sql = _TEMPLATES[QueryTemplate.ORB_SIZE_DIST].replace("{orb_label}", orb_label)

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    def _build_query(self, template: QueryTemplate, params: dict) -> tuple[str, list]:
        """Build parameterized SQL from template and parameters."""
        sql_template = _TEMPLATES[template]
        where_parts = []
        bind_params = []

        # Build WHERE clauses from parameters
        if "orb_label" in params:
            orb_label = _validate_orb_label(params["orb_label"])
            where_parts.append("AND orb_label = ?")
            bind_params.append(orb_label)

        if "entry_model" in params:
            em = _validate_entry_model(params["entry_model"])
            where_parts.append("AND entry_model = ?")
            bind_params.append(em)

        if "filter_type" in params:
            ft = _validate_filter_type(params["filter_type"])
            where_parts.append("AND filter_type = ?")
            bind_params.append(ft)

        if "min_sample_size" in params:
            min_n = int(params["min_sample_size"])
            where_parts.append("AND sample_size >= ?")
            bind_params.append(min_n)

        if "table_name" in params and template == QueryTemplate.SCHEMA_INFO:
            where_parts.append("AND table_name = ?")
            bind_params.append(params["table_name"])

        where_clause = "\n        ".join(where_parts)
        sql = sql_template.replace("{where_clauses}", where_clause)

        # Add LIMIT parameter if template has one
        if "LIMIT ?" in sql:
            limit = min(int(params.get("limit", 50)), MAX_RESULT_ROWS)
            bind_params.append(limit)

        return sql, bind_params

    @staticmethod
    def available_templates() -> list[dict[str, str]]:
        """Return list of available templates with descriptions."""
        descriptions = {
            QueryTemplate.STRATEGY_LOOKUP: "Look up validated strategies by session, filter, entry model",
            QueryTemplate.PERFORMANCE_STATS: "Aggregate performance stats (win rate, ExpR, Sharpe)",
            QueryTemplate.VALIDATED_SUMMARY: "Summary of validated strategies per session",
            QueryTemplate.YEARLY_BREAKDOWN: "Year-by-year performance for specific strategies",
            QueryTemplate.TRADE_HISTORY: "Individual trade outcomes from orb_outcomes",
            QueryTemplate.SCHEMA_INFO: "Database schema (tables, columns, types)",
            QueryTemplate.TABLE_COUNTS: "Row counts for all tables",
            QueryTemplate.ORB_SIZE_DIST: "Distribution of ORB sizes for a session",
            QueryTemplate.REGIME_COMPARE: "Compare regime (2025-only) vs full-period strategies",
            QueryTemplate.CORRELATION: "Top strategies by Sharpe for correlation analysis",
        }
        return [
            {"template": t.value, "description": descriptions.get(t, "")}
            for t in QueryTemplate
        ]
