"""Read-only adjacency queries against ``validated_setups``.

Builds the ``adjacency_context`` the LLM sees: a tight summary of currently-
active strategies plus the axes (rr_target, orb_minutes, confirm_bars,
filter_type) where adjacent unexplored cells exist.

The context is informational only. We never tell the LLM which cell to
propose — only that the gap exists. Direction-of-effect is the LLM's job to
ground in the literature, not ours to suggest.

Also exposes ``screen_candidate_mode_a`` — a delegating wrapper around
``trading_app.strategy_validator._evaluate_criterion_8_oos`` so the
proposer can pre-screen seed candidates under Mode A strict OOS before
spending an LLM call on a Mode B grandfathered baseline.
"""

from __future__ import annotations

from typing import Any

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


def screen_candidate_mode_a(
    candidate: dict[str, Any],
    *,
    db_path: str | None = None,
    strict_oos_n: bool = True,
) -> dict[str, Any]:
    """Score a candidate ``validated_setups`` row under Mode A strict OOS.

    Delegates to ``trading_app.strategy_validator._evaluate_criterion_8_oos``
    so we never re-encode the Criterion 8 logic. The validator function
    recomputes IS-baseline-vs-OOS against canonical ``orb_outcomes`` with
    ``trading_day >= HOLDOUT_SACRED_FROM`` — the same path the runner takes
    when it later evaluates the locked pre-reg.

    Parameters
    ----------
    candidate
        ``validated_setups``-shaped dict with at least: ``instrument`` (or
        ``symbol``), ``orb_label`` (or ``session``), ``orb_minutes``,
        ``entry_model``, ``confirm_bars``, ``rr_target``, ``filter_type``,
        ``expectancy_r``. Extra keys are ignored.
    db_path
        Optional gold.db override. Defaults to ``pipeline.paths.GOLD_DB_PATH``
        via the validator function.
    strict_oos_n
        When True (default for the proposer), insufficient OOS sample yields
        REJECTED — same as Pathway B validator gate. The proposer is
        drafting individual-mode candidates, so Pathway B strictness is
        the correct policy.

    Returns
    -------
    dict with keys:
        - ``passes_criterion_8``: True if validator would not reject.
        - ``oos_is_ratio``: OOS_ExpR / IS_ExpR (None when N_oos=0).
        - ``n_oos``: OOS trade count after filter application.
        - ``oos_expectancy_r``: Mode A strict OOS ExpR (None when N_oos=0).
        - ``c8_oos_status``: validator's status field.
        - ``reason``: validator's rejection reason (None on pass).
    """
    from trading_app.strategy_validator import _evaluate_criterion_8_oos

    row_dict: dict[str, Any] = dict(candidate)
    if "instrument" not in row_dict and "symbol" in row_dict:
        row_dict["instrument"] = row_dict["symbol"]
    if "orb_label" not in row_dict and "session" in row_dict:
        row_dict["orb_label"] = row_dict["session"]

    from pathlib import Path

    effective_path: Path | None = Path(db_path) if db_path else None
    verdict = _evaluate_criterion_8_oos(row_dict, effective_path, strict_oos_n=strict_oos_n)

    passes = verdict["status"] is None or verdict["c8_oos_status"] == "NO_OOS_DATA"
    return {
        "passes_criterion_8": passes,
        "oos_is_ratio": verdict.get("oos_is_ratio"),
        "n_oos": verdict.get("n_oos"),
        "oos_expectancy_r": verdict.get("oos_expectancy_r"),
        "c8_oos_status": verdict.get("c8_oos_status"),
        "reason": verdict.get("reason"),
    }


__all__ = [
    "_connect_read_only",
    "adjacency_summary_for_llm",
    "list_active_strategies",
    "screen_candidate_mode_a",
]
