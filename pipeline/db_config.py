"""Standard DuckDB connection tuning. Call immediately after connect()."""

import tempfile
from pathlib import Path


def configure_connection(con, *, writing: bool = False):
    """Configure a DuckDB connection with standard PRAGMAs.

    Args:
        con: DuckDB connection object.
        writing: If True, skip maintaining insertion order (10-20% faster inserts).
    """
    con.execute("SET memory_limit = '8GB'")
    tmp_dir = Path(tempfile.gettempdir()) / "duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp_dir.as_posix()}'")
    if writing:
        con.execute("SET preserve_insertion_order = false")
