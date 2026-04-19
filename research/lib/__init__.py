"""research.lib -- shared utilities for research scripts.

Usage:
    from research.lib import query_df, outcomes_query, ttest_1s, bh_fdr, write_markdown
"""

from research.lib.audit import assert_no_inflation
from research.lib.db import connect_db, query_df
from research.lib.io import format_stats_table, output_dir, write_csv, write_markdown
from research.lib.query import SAFE_JOIN, outcomes_query, session_col, with_dst_split
from research.lib.stats import (
    bh_fdr,
    compute_metrics,
    expanding_stat,
    mannwhitney_2s,
    ttest_1s,
    year_by_year,
)

__all__ = [
    "assert_no_inflation",
    "connect_db",
    "query_df",
    "format_stats_table",
    "output_dir",
    "write_csv",
    "write_markdown",
    "SAFE_JOIN",
    "outcomes_query",
    "session_col",
    "with_dst_split",
    "bh_fdr",
    "compute_metrics",
    "expanding_stat",
    "mannwhitney_2s",
    "ttest_1s",
    "year_by_year",
]
