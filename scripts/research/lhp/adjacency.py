"""Read-only adjacency queries against ``validated_setups``.

Builds the ``adjacency_context`` the LLM sees: a tight summary of currently-
active strategies plus the axes (rr_target, orb_minutes, confirm_bars,
filter_type) where adjacent unexplored cells exist.

The context is informational only. We never tell the LLM which cell to
propose — only that the gap exists. Direction-of-effect is the LLM's job to
ground in the literature, not ours to suggest.
"""

from __future__ import annotations

import duckdb

# Canonical DB path. Imported at function-call time so importing this module
# without duckdb installed (e.g., during static analysis) is safe.


def _connect_read_only(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def list_active_strategies(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return currently-active validated_setups rows for adjacency context.

    Columns mirror the trade-book contract: instrument, session, orb_minutes,
    entry_model, confirm_bars, rr_target, filter_type, sample_size,
    expectancy_r, sharpe_ratio. ORDER BY expectancy_r DESC so the LLM sees
    strong examples first.
    """
    rows = con.execute(
        """
        SELECT
            instrument,
            orb_label AS session,
            orb_minutes,
            entry_model,
            confirm_bars,
            rr_target,
            filter_type,
            sample_size,
            expectancy_r,
            sharpe_ratio
        FROM validated_setups
        WHERE status = 'active'
        ORDER BY expectancy_r DESC NULLS LAST
        """
    ).fetchall()
    cols = [
        "instrument",
        "session",
        "orb_minutes",
        "entry_model",
        "confirm_bars",
        "rr_target",
        "filter_type",
        "sample_size",
        "expectancy_r",
        "sharpe_ratio",
    ]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def adjacency_summary_for_llm(active: list[dict], max_rows: int = 20) -> str:
    """Build a compact one-line-per-strategy summary for the LLM."""
    lines: list[str] = []
    for row in active[:max_rows]:
        lines.append(
            f"- {row['instrument']} {row['session']} O{row['orb_minutes']} "
            f"{row['entry_model']} CB{row['confirm_bars']} RR{row['rr_target']} "
            f"{row['filter_type']} :: N={row['sample_size']} ExpR={row['expectancy_r']}"
        )
    return "\n".join(lines)


__all__ = [
    "_connect_read_only",
    "adjacency_summary_for_llm",
    "list_active_strategies",
]
