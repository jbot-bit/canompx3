"""Database connection lifecycle for research scripts.

All research scripts should use connect_db() or query_df() instead of
inline duckdb.connect() + os.environ boilerplate.
"""

from contextlib import contextmanager

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH


@contextmanager
def connect_db(read_only: bool = True):
    """Open a DuckDB connection to gold.db. Closes on exit.

    Usage:
        with connect_db() as con:
            df = con.execute(sql).fetchdf()
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Opens and closes connection automatically.

    Usage:
        df = query_df("SELECT * FROM orb_outcomes WHERE symbol = ?", ["MGC"])
    """
    with connect_db() as con:
        if params:
            return con.execute(sql, params).fetchdf()
        return con.execute(sql).fetchdf()
