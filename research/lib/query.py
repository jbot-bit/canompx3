"""Composable SQL query builder for research scripts.

Bakes in the canonical triple-join (SAFE_JOIN) to prevent the most common
research bug: missing orb_minutes in the JOIN producing 3x row inflation.
"""

# Canonical triple-join -- ALWAYS use this when joining orb_outcomes to daily_features.
# See .claude/rules/daily-features-joins.md for rationale.
SAFE_JOIN = """\
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes"""


def session_col(orb_label: str, stem: str) -> str:
    """Build daily_features column name: orb_{label}_{stem}.

    Example: session_col("1000", "size") -> "orb_1000_size"
    """
    return f"orb_{orb_label}_{stem}"


def outcomes_query(
    instrument: str,
    session: str,
    entry_model: str,
    extra_cols: list[str] | None = None,
    filters: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> str:
    """Build standard outcomes+features query with safe triple-join.

    Returns SQL string that SELECTs trading_day, pnl_r, outcome, and any
    extra_cols from the joined outcomes/features tables.

    Usage:
        sql = outcomes_query("MGC", "1000", "E0",
            extra_cols=["d.atr_5d"],
            filters=["d.orb_1000_size >= 4"],
            date_range=("2021-01-01", "2025-12-31"),
        )
        df = query_df(sql)
    """
    cols = ["o.trading_day", "o.pnl_r", "o.outcome", "o.mae_r", "o.mfe_r"]
    if extra_cols:
        cols.extend(extra_cols)
    select = ", ".join(cols)

    wheres = [
        f"o.symbol = '{instrument}'",
        f"o.orb_label = '{session}'",
        f"o.entry_model = '{entry_model}'",
        "o.outcome IN ('win', 'loss', 'early_exit')",
        "o.pnl_r IS NOT NULL",
    ]
    if filters:
        wheres.extend(filters)
    if date_range:
        wheres.append(f"o.trading_day >= '{date_range[0]}'")
        wheres.append(f"o.trading_day <= '{date_range[1]}'")
    where = "\n    AND ".join(wheres)

    return f"SELECT {select}\n{SAFE_JOIN}\n    WHERE {where}"


def with_dst_split(
    base_sql: str,
    session: str,
    regime_source: str,
) -> tuple[str, str]:
    """Wrap a base SQL query with DST regime filtering.

    Returns (dst_on_sql, dst_off_sql) -- summer and winter variants.
    regime_source: "US" for 0900/0030/2300 sessions, "UK" for 1800.

    DST columns (us_dst, uk_dst) live in daily_features.
    MANDATORY per CLAUDE.md: any analysis touching DST-sensitive sessions
    MUST split by regime and report both halves.
    """
    col = "d.us_dst" if regime_source.upper() == "US" else "d.uk_dst"
    # Wrap base_sql as CTE to add DST filter
    on_sql = f"WITH base AS ({base_sql})\nSELECT * FROM base WHERE {col} = TRUE"
    off_sql = f"WITH base AS ({base_sql})\nSELECT * FROM base WHERE {col} = FALSE"
    return on_sql, off_sql
