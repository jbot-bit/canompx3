"""
Load canonical grounding documents for AI query context.

Reads project documentation into memory for prompt injection.
No RAG needed -- total corpus fits easily in Claude's context window.
"""

from pathlib import Path

from pipeline.cost_model import COST_SPECS

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Derive MGC friction from canonical source
_mgc_spec = COST_SPECS["MGC"]
_mgc_friction = _mgc_spec.commission_rt + _mgc_spec.spread_doubled + _mgc_spec.slippage

# Canonical documents to load (path relative to project root, priority label)
CORPUS_FILES = {
    "TRADING_RULES": {
        "path": "TRADING_RULES.md",
        "priority": "CRITICAL",
        "description": "Trading rules, sessions, filters, cost model, R-multiples",
    },
    "TRADE_MANAGEMENT_RULES": {
        "path": "artifacts/TRADE_MANAGEMENT_RULES.md",
        "priority": "HIGH",
        "description": "Glossary and verified trade management rules",
    },
    "CONFIG": {
        "path": "trading_app/config.py",
        "priority": "CRITICAL",
        "description": "Filters, entry models, classification thresholds",
    },
    "COST_MODEL": {
        "path": "pipeline/cost_model.py",
        "priority": "CRITICAL",
        "description": f"MGC friction ${_mgc_friction:.2f}, R-multiple math",
    },
}


def load_corpus() -> dict[str, str]:
    """Read all canonical docs into memory.

    Returns dict mapping document name to file contents.
    Missing files are skipped with a warning in the value.
    """
    corpus = {}
    for name, info in CORPUS_FILES.items():
        fpath = PROJECT_ROOT / info["path"]
        if fpath.exists():
            corpus[name] = fpath.read_text(encoding="utf-8")
        else:
            corpus[name] = f"[MISSING: {fpath}]"
    return corpus


def get_corpus_file_paths() -> list[str]:
    """Return list of relative paths referenced by corpus (for drift check)."""
    return [info["path"] for info in CORPUS_FILES.values()]


def get_schema_definitions(db_path: str) -> str:
    """Extract table/column info from DuckDB information_schema.

    Returns a formatted string describing all tables and their columns.
    """
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()

        lines = []
        for (table_name,) in tables:
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
            col_strs = [f"  {cname} ({dtype})" for cname, dtype in cols]
            lines.append(f"{table_name}:")
            lines.extend(col_strs)
            lines.append("")
        return "\n".join(lines)
    finally:
        con.close()


def get_db_stats(db_path: str) -> str:
    """Get row counts per table for context."""
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()

        lines = []
        for (table_name,) in tables:
            count = con.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            lines.append(f"{table_name}: {count:,} rows")
        return "\n".join(lines)
    finally:
        con.close()
