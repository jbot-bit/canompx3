"""Standard DuckDB connection tuning. Call immediately after connect()."""


def configure_connection(con, *, writing: bool = False):
    """Configure a DuckDB connection with standard PRAGMAs.

    Args:
        con: DuckDB connection object.
        writing: If True, skip maintaining insertion order (10-20% faster inserts).
    """
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET temp_directory = 'C:/db/.tmp'")
    if writing:
        con.execute("SET preserve_insertion_order = false")
