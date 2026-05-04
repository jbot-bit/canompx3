"""
Load canonical grounding documents for AI query context.

Reads project documentation into memory for prompt injection.
No RAG needed -- total corpus fits easily in Claude's context window.
"""

import logging
from pathlib import Path

from pipeline.cost_model import COST_SPECS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Derive MGC friction from canonical source
_mgc_spec = COST_SPECS["MGC"]
_mgc_friction = _mgc_spec.commission_rt + _mgc_spec.spread_doubled + _mgc_spec.slippage

# Canonical documents to load (path relative to project root, priority label).
#
# Stage 2 of claude-api-modernization: expanded from 4 to 8 entries so the AI
# has access to the same institutional-rigor and research-methodology doctrine
# used by human analysts and the embedded MCP server's `get_canonical_context`
# tool. Adding or removing an entry here propagates automatically to every
# caller of `load_corpus()` (query_agent.py, mcp_server.py).
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
        "description": (
            "Per-instrument friction and point values. MGC baseline: "
            f"${_mgc_friction:.2f} RT — see COST_SPECS for all active instruments."
        ),
    },
    "RESEARCH_RULES": {
        "path": "RESEARCH_RULES.md",
        "priority": "CRITICAL",
        "description": "Research methodology, BH FDR, canonical layer discipline",
    },
    "CLAUDE_MD": {
        "path": "CLAUDE.md",
        "priority": "CRITICAL",
        "description": "Guardrails, authority hierarchy, 2-pass protocol",
    },
    "PRE_REGISTERED_CRITERIA": {
        "path": "docs/institutional/pre_registered_criteria.md",
        "priority": "CRITICAL",
        "description": "12 locked criteria every validated strategy must meet",
    },
    "MECHANISM_PRIORS": {
        "path": "docs/institutional/mechanism_priors.md",
        "priority": "HIGH",
        "description": "ORB edge priors, signal-to-role mapping, staged deployment",
    },
}


def load_corpus() -> dict[str, str]:
    """Read all canonical docs into memory.

    Missing HIGH-priority files are logged and replaced with a sentinel string.
    Missing CRITICAL-priority files raise RuntimeError — fail-closed per
    integrity-guardian.md § 3, because degraded grounding can produce
    confidently-wrong answers, which is strictly worse than a hard abort.
    """
    corpus = {}
    missing_critical: list[str] = []
    for name, info in CORPUS_FILES.items():
        fpath = PROJECT_ROOT / info["path"]
        if fpath.exists():
            corpus[name] = fpath.read_text(encoding="utf-8")
        else:
            logger.warning("Corpus file missing: %s (%s)", info["path"], name)
            corpus[name] = f"[MISSING: {fpath}]"
            if info.get("priority") == "CRITICAL":
                missing_critical.append(f"{name} ({info['path']})")
    if missing_critical:
        raise RuntimeError(
            "CRITICAL corpus files missing: "
            + ", ".join(missing_critical)
            + ". Fail-closed per integrity-guardian.md § 3 — grounding must not "
            "run with missing critical docs."
        )
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
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
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
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()

        lines = []
        for (table_name,) in tables:
            count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            lines.append(f"{table_name}: {count:,} rows")
        return "\n".join(lines)
    finally:
        con.close()
