"""
Safe, read-only query executor with pre-approved SQL templates.

AI picks template + parameters, never writes raw SQL.
All queries use duckdb.connect(read_only=True) and parameterized queries.
"""

from dataclasses import dataclass, field
from enum import Enum

import duckdb
import pandas as pd

from pipeline.asset_configs import get_active_instruments


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
    DOUBLE_BREAK_STATS = "double_break_stats"
    GAP_ANALYSIS = "gap_analysis"
    ROLLING_STABILITY = "rolling_stability"
    OUTCOMES_STATS = "outcomes_stats"
    ENTRY_MODEL_COMPARE = "entry_model_compare"
    DOW_BREAKDOWN = "dow_breakdown"
    DST_SPLIT = "dst_split"
    FILTER_COMPARE = "filter_compare"


@dataclass
class QueryIntent:
    """Parsed intent from user question."""

    template: QueryTemplate
    parameters: dict = field(default_factory=dict)
    explanation: str = ""


# Maximum rows returned by any query
MAX_RESULT_ROWS = 1000

# Valid ORB labels for parameter validation
VALID_ORB_LABELS = {
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "US_DATA_830", "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE",
    "CME_PRECLOSE", "NYSE_CLOSE",
}

# Valid entry models
VALID_ENTRY_MODELS = {"E1", "E2", "E3"}

# Valid filter types (subset for validation)
VALID_FILTER_PREFIXES = {"NO_FILTER", "ORB_G", "ORB_L", "VOL_", "DIR_", "DOW_", "M6E_"}

# Valid instruments (from canonical source — pipeline.asset_configs)
VALID_INSTRUMENTS = set(get_active_instruments())

VALID_RR_TARGETS = {1.0, 1.5, 2.0, 2.5, 3.0, 4.0}
VALID_CONFIRM_BARS = {1, 2, 3, 4, 5}

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


def _validate_instrument(inst: str) -> str:
    """Validate and return instrument."""
    if inst not in VALID_INSTRUMENTS:
        raise ValueError(f"Invalid instrument '{inst}'. Valid: {sorted(VALID_INSTRUMENTS)}")
    return inst


def _validate_rr_target(rr) -> float:
    """Validate and return RR target."""
    rr = float(rr)
    if rr not in VALID_RR_TARGETS:
        raise ValueError(f"Invalid rr_target {rr}. Valid: {sorted(VALID_RR_TARGETS)}")
    return rr


def _validate_confirm_bars(cb) -> int:
    """Validate and return confirm bars."""
    cb = int(cb)
    if cb not in VALID_CONFIRM_BARS:
        raise ValueError(f"Invalid confirm_bars {cb}. Valid: {sorted(VALID_CONFIRM_BARS)}")
    return cb


# DST regime column per session.
# All sessions are now dynamic (DST-aware resolvers) — no fixed sessions remain.
# DST split analysis is no longer applicable since sessions self-adjust.
_DST_SESSION_MAP = {}


def _orb_size_filter_sql(filter_type: str | None, orb_label: str) -> str | None:
    """Convert ORB size filter_type to SQL WHERE clause on daily_features.

    Handles simple filters (ORB_G4, ORB_L12) and band filters (ORB_G4_L12).
    Returns None for NO_FILTER/None (no filtering needed).
    Raises ValueError for non-ORB filters (VOL_, DIR_, DOW_, composites)
    that require Python-side evaluation and cannot be applied in SQL.
    """
    if not filter_type or filter_type == "NO_FILTER":
        return None
    _validate_filter_type(filter_type)
    # orb_label validated against allowlist -- safe for f-string
    # [1,20] is a security guard, not a business rule. Current grid uses G4-G8 (max 8pt).
    col = f"d.orb_{orb_label}_size"

    if filter_type.startswith("ORB_G"):
        rest = filter_type[5:]  # after "ORB_G"
        if "_L" in rest:
            # Band filter: ORB_G4_L12 -> >= 4 AND < 12
            parts = rest.split("_L")
            lo = int(parts[0])
            hi = int(parts[1])
            if not (1 <= lo <= 20) or not (1 <= hi <= 20):
                raise ValueError(f"ORB band filter thresholds {lo}/{hi} out of range [1, 20]")
            return f"{col} >= {lo} AND {col} < {hi}"
        try:
            threshold = int(rest)
        except ValueError:
            raise ValueError(
                f"Filter '{filter_type}' contains a non-ORB component that cannot "
                f"be applied in SQL. Only pure ORB size filters (ORB_G4, ORB_L8, "
                f"ORB_G4_L12) are supported in raw outcomes queries."
            ) from None
        if not (1 <= threshold <= 20):
            raise ValueError(f"ORB filter threshold {threshold} out of range [1, 20]")
        return f"{col} >= {threshold}"
    if filter_type.startswith("ORB_L"):
        threshold = int(filter_type[5:])
        if not (1 <= threshold <= 20):
            raise ValueError(f"ORB filter threshold {threshold} out of range [1, 20]")
        return f"{col} < {threshold}"
    # Non-ORB filter (VOL_, DIR_, DOW_) — fail-closed
    raise ValueError(
        f"Filter '{filter_type}' requires Python-side evaluation and cannot be "
        f"applied in raw SQL outcomes queries. Only ORB size filters "
        f"(ORB_G4, ORB_L8, ORB_G4_L12, NO_FILTER) are supported here."
    )


def _compute_group_stats(df: pd.DataFrame) -> dict:
    """Compute standard stats (N, win_rate, avg_pnl_r, sharpe) from a trades group."""
    if df.empty:
        return {"N": 0, "win_rate": None, "avg_pnl_r": None, "sharpe": None}
    n = len(df)
    win_rate = round(float((df["outcome"] == "win").mean()) * 100, 1)
    avg_pnl = round(float(df["pnl_r"].mean()), 4)
    std_pnl = float(df["pnl_r"].std())
    sharpe = round(avg_pnl / std_pnl, 3) if std_pnl > 0 else None
    return {"N": n, "win_rate": win_rate, "avg_pnl_r": avg_pnl, "sharpe": sharpe}


# SQL templates -- all SELECT-only, parameterized
_TEMPLATES = {
    QueryTemplate.STRATEGY_LOOKUP: """
        SELECT orb_label, entry_model, filter_type,
               rr_target, confirm_bars, sample_size, win_rate,
               expectancy_r, sharpe_ratio, max_drawdown_r
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        {instrument_clause}
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
        {instrument_clause}
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
        {instrument_clause}
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
        {instrument_clause}
        {where_clauses}
        ORDER BY expectancy_r DESC
        LIMIT ?
    """,
    QueryTemplate.TRADE_HISTORY: """
        SELECT trading_day, orb_label, entry_model,
               rr_target, confirm_bars, entry_price, stop_price,
               target_price, pnl_r, outcome
        FROM orb_outcomes
        WHERE 1=1
        {instrument_clause_symbol}
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
        {instrument_filter}
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
        {instrument_clause}
        ORDER BY sharpe_ratio DESC
        LIMIT ?
    """,
    QueryTemplate.DOUBLE_BREAK_STATS: """
        SELECT
            EXTRACT(YEAR FROM trading_day) as year,
            COUNT(*) as total_days,
            SUM(CASE WHEN orb_{orb_label}_double_break THEN 1 ELSE 0 END) as double_break_days,
            ROUND(AVG(CASE WHEN orb_{orb_label}_double_break THEN 1.0 ELSE 0.0 END) * 100, 1) as db_pct,
            AVG(orb_{orb_label}_size) as avg_orb_size
        FROM daily_features
        WHERE orb_minutes = 5
          AND orb_{orb_label}_size IS NOT NULL
          {instrument_filter}
        GROUP BY year
        ORDER BY year
    """,
    QueryTemplate.GAP_ANALYSIS: """
        SELECT
            EXTRACT(YEAR FROM trading_day) as year,
            COUNT(*) as days,
            AVG(gap_open_points) as avg_gap,
            AVG(ABS(gap_open_points)) as avg_abs_gap,
            MIN(gap_open_points) as min_gap,
            MAX(gap_open_points) as max_gap,
            COUNT(CASE WHEN ABS(gap_open_points) > 1.0 THEN 1 END) as gaps_over_1pt,
            COUNT(CASE WHEN ABS(gap_open_points) > 2.0 THEN 1 END) as gaps_over_2pt
        FROM daily_features
        WHERE orb_minutes = 5
          AND gap_open_points IS NOT NULL
          {instrument_filter}
        GROUP BY year
        ORDER BY year
    """,
    QueryTemplate.ROLLING_STABILITY: """
        SELECT
            rv.orb_label,
            rv.entry_model,
            rv.filter_type,
            COUNT(DISTINCT rv.run_label) as windows_passed,
            AVG(rv.expectancy_r) as avg_expr,
            AVG(rv.sharpe_ratio) as avg_sharpe,
            AVG(rv.sample_size) as avg_sample,
            MIN(rv.sample_size) as min_sample,
            AVG(rv.win_rate) as avg_wr,
            AVG(rv.max_drawdown_r) as avg_max_dd
        FROM regime_validated rv
        WHERE rv.run_label LIKE 'rolling_%'
        {where_clauses}
        GROUP BY rv.orb_label, rv.entry_model, rv.filter_type
        ORDER BY windows_passed DESC, avg_sharpe DESC
        LIMIT ?
    """,
    # --- Raw outcomes templates (SAFE_JOIN based) ---
    # NOTE: These 5 static SQL entries exist for structural test compliance
    # (test_all_templates_have_sql).  Actual execution uses custom _execute_*
    # methods which build queries dynamically via _build_outcomes_base().
    QueryTemplate.OUTCOMES_STATS: """
        SELECT o.pnl_r, o.outcome, o.mae_r, o.mfe_r, o.trading_day
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.outcome IN ('win', 'loss', 'early_exit')
          AND o.pnl_r IS NOT NULL
    """,
    QueryTemplate.ENTRY_MODEL_COMPARE: """
        SELECT o.entry_model, o.pnl_r, o.outcome
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.outcome IN ('win', 'loss', 'early_exit')
          AND o.pnl_r IS NOT NULL
    """,
    QueryTemplate.DOW_BREAKDOWN: """
        SELECT o.trading_day, o.pnl_r, o.outcome
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.outcome IN ('win', 'loss', 'early_exit')
          AND o.pnl_r IS NOT NULL
    """,
    # Runtime selects us_dst or uk_dst based on session (see _execute_dst_split)
    QueryTemplate.DST_SPLIT: """
        SELECT o.pnl_r, o.outcome, d.us_dst AS dst_active
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.outcome IN ('win', 'loss', 'early_exit')
          AND o.pnl_r IS NOT NULL
    """,
    QueryTemplate.FILTER_COMPARE: """
        SELECT o.pnl_r, o.outcome
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.outcome IN ('win', 'loss', 'early_exit')
          AND o.pnl_r IS NOT NULL
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

        if template == QueryTemplate.DOUBLE_BREAK_STATS:
            return self._execute_double_break_stats(params)

        if template == QueryTemplate.GAP_ANALYSIS:
            return self._execute_gap_analysis(params)

        if template == QueryTemplate.REGIME_COMPARE:
            return self._execute_regime_compare(params)

        if template == QueryTemplate.ROLLING_STABILITY:
            return self._execute_rolling_stability(params)

        if template == QueryTemplate.OUTCOMES_STATS:
            return self._execute_outcomes_stats(params)
        if template == QueryTemplate.ENTRY_MODEL_COMPARE:
            return self._execute_entry_model_compare(params)
        if template == QueryTemplate.DOW_BREAKDOWN:
            return self._execute_dow_breakdown(params)
        if template == QueryTemplate.DST_SPLIT:
            return self._execute_dst_split(params)
        if template == QueryTemplate.FILTER_COMPARE:
            return self._execute_filter_compare(params)

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
        orb_label = _validate_orb_label(params.get("orb_label", "CME_REOPEN"))

        # Build SQL with validated orb_label embedded (safe -- validated against allowlist)
        sql = _TEMPLATES[QueryTemplate.ORB_SIZE_DIST].replace("{orb_label}", orb_label)

        bind_params = []
        if "instrument" in params:
            inst = _validate_instrument(params["instrument"])
            sql = sql.replace("{instrument_filter}", "AND symbol = ?")
            bind_params.append(inst)
        else:
            sql = sql.replace("{instrument_filter}", "")

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql, bind_params).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    def _execute_double_break_stats(self, params: dict) -> pd.DataFrame:
        """Execute double-break stats query with validated ORB label."""
        orb_label = _validate_orb_label(params.get("orb_label", "CME_REOPEN"))

        # Safe: orb_label validated against allowlist
        sql = _TEMPLATES[QueryTemplate.DOUBLE_BREAK_STATS].replace("{orb_label}", orb_label)

        bind_params = []
        if "instrument" in params:
            inst = _validate_instrument(params["instrument"])
            sql = sql.replace("{instrument_filter}", "AND symbol = ?")
            bind_params.append(inst)
        else:
            sql = sql.replace("{instrument_filter}", "")

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql, bind_params).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    def _execute_gap_analysis(self, params: dict) -> pd.DataFrame:
        """Execute gap analysis query."""
        sql = _TEMPLATES[QueryTemplate.GAP_ANALYSIS]

        bind_params = []
        if "instrument" in params:
            inst = _validate_instrument(params["instrument"])
            sql = sql.replace("{instrument_filter}", "AND symbol = ?")
            bind_params.append(inst)
        else:
            sql = sql.replace("{instrument_filter}", "")

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql, bind_params).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    def _execute_regime_compare(self, params: dict) -> pd.DataFrame:
        """Execute regime compare with table-qualified column names.

        Fixes ambiguous column reference when both r. and v. tables have
        orb_label, entry_model, filter_type.
        """
        sql_template = _TEMPLATES[QueryTemplate.REGIME_COMPARE]
        where_parts = []
        bind_params = []

        if "instrument" in params:
            _validate_instrument(params["instrument"])
            where_parts.append("AND r.instrument = ?")
            bind_params.append(params["instrument"])

        # Qualify with r. prefix to avoid ambiguity
        if "orb_label" in params:
            _validate_orb_label(params["orb_label"])
            where_parts.append("AND r.orb_label = ?")
            bind_params.append(params["orb_label"])

        if "entry_model" in params:
            _validate_entry_model(params["entry_model"])
            where_parts.append("AND r.entry_model = ?")
            bind_params.append(params["entry_model"])

        if "filter_type" in params:
            _validate_filter_type(params["filter_type"])
            where_parts.append("AND r.filter_type = ?")
            bind_params.append(params["filter_type"])

        if "min_sample_size" in params:
            where_parts.append("AND r.sample_size >= ?")
            bind_params.append(int(params["min_sample_size"]))

        where_clause = "\n        ".join(where_parts)
        sql = sql_template.replace("{where_clauses}", where_clause)

        limit = min(int(params.get("limit", 50)), MAX_RESULT_ROWS)
        bind_params.append(limit)

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql, bind_params).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    def _execute_rolling_stability(self, params: dict) -> pd.DataFrame:
        """Execute rolling stability query with rv.-qualified WHERE clauses."""
        sql_template = _TEMPLATES[QueryTemplate.ROLLING_STABILITY]
        where_parts = []
        bind_params = []

        if "instrument" in params:
            _validate_instrument(params["instrument"])
            where_parts.append("AND rv.instrument = ?")
            bind_params.append(params["instrument"])

        if "orb_label" in params:
            _validate_orb_label(params["orb_label"])
            where_parts.append("AND rv.orb_label = ?")
            bind_params.append(params["orb_label"])

        if "entry_model" in params:
            _validate_entry_model(params["entry_model"])
            where_parts.append("AND rv.entry_model = ?")
            bind_params.append(params["entry_model"])

        if "filter_type" in params:
            _validate_filter_type(params["filter_type"])
            where_parts.append("AND rv.filter_type = ?")
            bind_params.append(params["filter_type"])

        if "min_sample_size" in params:
            where_parts.append("AND rv.sample_size >= ?")
            bind_params.append(int(params["min_sample_size"]))

        where_clause = "\n        ".join(where_parts)
        sql = sql_template.replace("{where_clauses}", where_clause)

        limit = min(int(params.get("limit", 50)), MAX_RESULT_ROWS)
        bind_params.append(limit)

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            result = con.execute(sql, bind_params).fetchdf()
            return result.head(MAX_RESULT_ROWS)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Raw outcomes templates (SAFE_JOIN based)
    # ------------------------------------------------------------------

    def _build_outcomes_base(self, params: dict, extra_cols: str = "") -> tuple[str, list]:
        """Build SAFE_JOIN query for raw outcomes analysis.

        Returns (sql, bind_params).  All raw outcomes templates share this
        foundation: orb_outcomes JOIN daily_features on the canonical triple key.
        """
        instrument = _validate_instrument(params.get("instrument", "MGC"))
        if "orb_label" not in params:
            raise ValueError("orb_label is required")
        orb_label = _validate_orb_label(params["orb_label"])

        extra = f", {extra_cols}" if extra_cols else ""

        wheres = [
            "o.symbol = ?",
            "o.orb_label = ?",
            "o.outcome IN ('win', 'loss', 'early_exit')",
            "o.pnl_r IS NOT NULL",
        ]
        bind: list = [instrument, orb_label]

        if "entry_model" in params:
            wheres.append("o.entry_model = ?")
            bind.append(_validate_entry_model(params["entry_model"]))
        if "rr_target" in params:
            wheres.append("o.rr_target = ?")
            bind.append(_validate_rr_target(params["rr_target"]))
        if "confirm_bars" in params:
            wheres.append("o.confirm_bars = ?")
            bind.append(_validate_confirm_bars(params["confirm_bars"]))

        # ORB size filter (raises ValueError for non-ORB filters)
        filter_sql = _orb_size_filter_sql(params.get("filter_type"), orb_label)
        if filter_sql:
            wheres.append(filter_sql)

        where_clause = "\n          AND ".join(wheres)

        sql = f"""
            SELECT o.pnl_r, o.outcome, o.mae_r, o.mfe_r, o.trading_day{extra}
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE {where_clause}
            ORDER BY o.trading_day
        """
        return sql, bind

    def _execute_outcomes_stats(self, params: dict) -> pd.DataFrame:
        """Raw outcomes stats: N, win_rate, avg_pnl_r, sharpe, max_drawdown, MAE/MFE."""
        sql, bind = self._build_outcomes_base(params)

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            df = con.execute(sql, bind).fetchdf()
        finally:
            con.close()

        if df.empty:
            return pd.DataFrame([{
                "N": 0, "win_rate": None, "avg_pnl_r": None,
                "sharpe": None, "max_drawdown_r": None,
                "avg_mae": None, "avg_mfe": None,
            }])

        stats = _compute_group_stats(df)

        # Max drawdown from equity curve
        cum = df["pnl_r"].cumsum()
        peak = cum.cummax()
        stats["max_drawdown_r"] = round(float((cum - peak).min()), 2)

        stats["avg_mae"] = (
            round(float(df["mae_r"].mean()), 4) if df["mae_r"].notna().any() else None
        )
        stats["avg_mfe"] = (
            round(float(df["mfe_r"].mean()), 4) if df["mfe_r"].notna().any() else None
        )

        return pd.DataFrame([stats])

    def _execute_entry_model_compare(self, params: dict) -> pd.DataFrame:
        """Side-by-side E1 vs E2 vs E3 comparison."""
        # Strip entry_model — we want all models
        params_all = {k: v for k, v in params.items() if k != "entry_model"}
        sql, bind = self._build_outcomes_base(params_all, extra_cols="o.entry_model")

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            df = con.execute(sql, bind).fetchdf()
        finally:
            con.close()

        if df.empty:
            return pd.DataFrame(columns=["entry_model", "N", "win_rate", "avg_pnl_r", "sharpe"])

        rows = []
        for em in sorted(df["entry_model"].unique()):
            subset = df[df["entry_model"] == em]
            stats = _compute_group_stats(subset)
            stats["entry_model"] = em
            rows.append(stats)

        result = pd.DataFrame(rows)
        return result[["entry_model", "N", "win_rate", "avg_pnl_r", "sharpe"]]

    def _execute_dow_breakdown(self, params: dict) -> pd.DataFrame:
        """Day-of-week performance splits."""
        sql, bind = self._build_outcomes_base(params)

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            df = con.execute(sql, bind).fetchdf()
        finally:
            con.close()

        if df.empty:
            return pd.DataFrame(columns=["day_of_week", "day_name", "N", "win_rate", "avg_pnl_r"])

        dow_names = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday",
            3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
        }
        df["dow"] = pd.to_datetime(df["trading_day"]).dt.dayofweek

        rows = []
        for dow in sorted(df["dow"].unique()):
            subset = df[df["dow"] == dow]
            stats = _compute_group_stats(subset)
            stats["day_of_week"] = int(dow)
            stats["day_name"] = dow_names.get(int(dow), str(dow))
            rows.append(stats)

        result = pd.DataFrame(rows)
        return result[["day_of_week", "day_name", "N", "win_rate", "avg_pnl_r"]]

    def _execute_dst_split(self, params: dict) -> pd.DataFrame:
        """DST on vs off performance split.

        DEPRECATED: All sessions are now dynamic (DST-aware resolvers).
        DST split analysis is no longer applicable since sessions self-adjust.
        """
        raise ValueError(
            "DST split is no longer applicable. All sessions are now dynamic "
            "(DST-aware resolvers) and self-adjust per day. No fixed sessions remain."
        )

    def _execute_filter_compare(self, params: dict) -> pd.DataFrame:
        """Compare ORB size filter levels side-by-side.

        Fetches all trades once, then computes stats for NO_FILTER, G4, G5, G6.
        Each level is cumulative (G6 is a subset of G4).
        """
        if "orb_label" not in params:
            raise ValueError("orb_label is required for filter_compare")
        orb_label = _validate_orb_label(params["orb_label"])
        size_col = f"d.orb_{orb_label}_size"

        # Fetch all trades with ORB size (no filter applied)
        params_no_filter = {k: v for k, v in params.items() if k != "filter_type"}
        sql, bind = self._build_outcomes_base(
            params_no_filter, extra_cols=f"{size_col} AS orb_size"
        )

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            df = con.execute(sql, bind).fetchdf()
        finally:
            con.close()

        if df.empty:
            return pd.DataFrame(columns=["filter_applied", "N", "win_rate", "avg_pnl_r", "sharpe"])

        # Cumulative filter levels (each includes all trades above threshold)
        filters = [
            ("NO_FILTER", df),
            ("ORB_G4", df[df["orb_size"] >= 4]),
            ("ORB_G5", df[df["orb_size"] >= 5]),
            ("ORB_G6", df[df["orb_size"] >= 6]),
            ("ORB_G8", df[df["orb_size"] >= 8]),
        ]

        rows = []
        for name, subset in filters:
            stats = _compute_group_stats(subset)
            stats["filter_applied"] = name
            rows.append(stats)

        result = pd.DataFrame(rows)
        return result[["filter_applied", "N", "win_rate", "avg_pnl_r", "sharpe"]]

    def _build_query(self, template: QueryTemplate, params: dict) -> tuple[str, list]:
        """Build parameterized SQL from template and parameters."""
        sql_template = _TEMPLATES[template]
        where_parts = []
        bind_params = []

        # Instrument filtering (uses 'instrument' column on most tables, 'symbol' on orb_outcomes)
        instrument_clause = ""
        instrument_clause_symbol = ""
        if "instrument" in params:
            inst = _validate_instrument(params["instrument"])
            instrument_clause = "AND instrument = ?"
            instrument_clause_symbol = "AND symbol = ?"
            # Determine which placeholder this template uses
            if "{instrument_clause_symbol}" in sql_template:
                bind_params.append(inst)
            elif "{instrument_clause}" in sql_template:
                bind_params.append(inst)

        sql_template = sql_template.replace("{instrument_clause}", instrument_clause)
        sql_template = sql_template.replace("{instrument_clause_symbol}", instrument_clause_symbol)

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
            QueryTemplate.DOUBLE_BREAK_STATS: "Double-break frequency by year for an ORB session",
            QueryTemplate.GAP_ANALYSIS: "Overnight gap statistics by year",
            QueryTemplate.ROLLING_STABILITY: "Rolling window stability: which strategy families pass across 12/18-month windows",
            QueryTemplate.OUTCOMES_STATS: "Raw outcomes stats (N, win rate, ExpR, Sharpe, drawdown, MAE/MFE) for any slice",
            QueryTemplate.ENTRY_MODEL_COMPARE: "Side-by-side E1 vs E2 vs E3 comparison for same session",
            QueryTemplate.DOW_BREAKDOWN: "Day-of-week performance splits for a session",
            QueryTemplate.DST_SPLIT: "DEPRECATED: All sessions are now DST-clean (dynamic resolvers). No longer applicable.",
            QueryTemplate.FILTER_COMPARE: "Compare NO_FILTER vs G4 vs G5 vs G6 for same session",
        }
        return [
            {"template": t.value, "description": descriptions.get(t, "")}
            for t in QueryTemplate
        ]
